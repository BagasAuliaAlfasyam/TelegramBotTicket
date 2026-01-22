#!/usr/bin/env python3
"""
Unified Bot Runner
===================
Run both collecting and reporting bots in a single process.
Includes hourly monitoring stats scheduler.

Usage:
    python scripts/run_all.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import Config
from src.services.sheets import GoogleSheetsClient
from src.services.storage import S3Uploader
from src.ml.classifier import MLClassifier
from src.ml.tracking import MLTrackingClient
from src.bots.collector import build_collecting_application
from src.bots.admin import build_reporting_application


def setup_logging(debug: bool = False) -> None:
    """Configure logging for both bots."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("gspread").setLevel(logging.WARNING)


async def run_collecting_bot(
    config: Config,
    sheets_client: GoogleSheetsClient,
    s3_uploader: S3Uploader,
    ml_classifier: MLClassifier | None,
    ml_tracking: MLTrackingClient | None,
    logger: logging.Logger,
) -> None:
    """Run collecting bot asynchronously."""
    if not config.telegram_collecting_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN_COLLECTING not set, skipping collecting bot")
        return
    
    logger.info("Starting Collecting Bot...")
    
    application = build_collecting_application(
        config=config,
        sheets_client=sheets_client,
        s3_uploader=s3_uploader,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
    )
    
    # Initialize and start polling
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=["message", "edited_message"])
    
    logger.info("✅ Collecting Bot is running")
    
    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour, loop forever
    except asyncio.CancelledError:
        logger.info("Stopping Collecting Bot...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


async def run_reporting_bot(
    config: Config,
    sheets_client: GoogleSheetsClient,
    ml_classifier: MLClassifier | None,
    ml_tracking: MLTrackingClient | None,
    logger: logging.Logger,
) -> None:
    """Run reporting bot asynchronously."""
    if not config.telegram_reporting_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN_REPORTING not set, skipping reporting bot")
        return
    
    logger.info("Starting Reporting Bot...")
    
    application = build_reporting_application(
        config=config,
        sheets_client=sheets_client,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
    )
    
    # Initialize and start polling
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=["message"])
    
    logger.info("✅ Reporting Bot is running")
    
    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Stopping Reporting Bot...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


async def run_monitoring_scheduler(
    ml_classifier: MLClassifier | None,
    ml_tracking: MLTrackingClient | None,
    logger: logging.Logger,
    interval_seconds: int = 3600,  # 1 hour default
) -> None:
    """
    Background scheduler to update Monitoring sheet stats hourly.
    
    Args:
        ml_classifier: ML classifier for model version
        ml_tracking: ML tracking client
        logger: Logger instance
        interval_seconds: Update interval (default 1 hour)
    """
    if not ml_tracking:
        logger.warning("ML Tracking not initialized, monitoring scheduler disabled")
        return
    
    logger.info("Starting Monitoring Scheduler (interval: %d seconds)", interval_seconds)
    
    try:
        while True:
            # Wait for the interval first (stats at startup might be empty)
            await asyncio.sleep(interval_seconds)
            
            try:
                model_version = ml_classifier.model_version if ml_classifier else "unknown"
                
                logger.info("Scheduler: Updating monitoring stats...")
                stats = ml_tracking.calculate_and_update_daily_stats(model_version)
                
                if stats:
                    logger.info(
                        "Scheduler: Stats updated - %d predictions, %.1f%% avg confidence",
                        stats.get("total_predictions", 0),
                        stats.get("avg_confidence", 0) * 100
                    )
                else:
                    logger.debug("Scheduler: No predictions today to update")
                    
            except Exception as e:
                logger.error("Scheduler: Failed to update stats: %s", e)
                
    except asyncio.CancelledError:
        logger.info("Stopping Monitoring Scheduler...")
        raise


async def main_async(
    config: Config,
    sheets_client: GoogleSheetsClient,
    s3_uploader: S3Uploader,
    ml_classifier: MLClassifier | None,
    ml_tracking: MLTrackingClient | None,
    logger: logging.Logger,
) -> None:
    """Main async entry point - runs both bots concurrently."""
    
    # Create tasks for both bots
    tasks = []
    
    # Collecting bot task
    collecting_task = asyncio.create_task(
        run_collecting_bot(
            config=config,
            sheets_client=sheets_client,
            s3_uploader=s3_uploader,
            ml_classifier=ml_classifier,
            ml_tracking=ml_tracking,
            logger=logger,
        ),
        name="collecting_bot"
    )
    tasks.append(collecting_task)
    
    # Reporting bot task
    reporting_task = asyncio.create_task(
        run_reporting_bot(
            config=config,
            sheets_client=sheets_client,
            ml_classifier=ml_classifier,
            ml_tracking=ml_tracking,
            logger=logger,
        ),
        name="reporting_bot"
    )
    tasks.append(reporting_task)
    
    # Monitoring scheduler task (hourly stats update)
    scheduler_task = asyncio.create_task(
        run_monitoring_scheduler(
            ml_classifier=ml_classifier,
            ml_tracking=ml_tracking,
            logger=logger,
            interval_seconds=3600,  # 1 hour
        ),
        name="monitoring_scheduler"
    )
    tasks.append(scheduler_task)
    
    logger.info("=" * 50)
    logger.info("All bots are running. Press Ctrl+C to stop.")
    logger.info("Monitoring stats update: every 1 hour")
    logger.info("=" * 50)
    
    # Wait for all tasks, handle shutdown gracefully
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down all bots...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All bots stopped.")


def main() -> None:
    """Entry point with signal handling."""
    # Load configuration BEFORE async loop
    config = Config.from_env()
    setup_logging(config.debug)
    
    logger = logging.getLogger(__name__)
    
    # Validate config
    errors = config.validate()
    if errors:
        logger.error("Configuration errors: %s", errors)
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("Starting Unified Bot Runner")
    logger.info("=" * 50)
    logger.info("Config: %s", config)
    
    # Initialize shared services SYNCHRONOUSLY (before async loop)
    logger.info("Initializing services...")
    sheets_client = GoogleSheetsClient(config)
    s3_uploader = S3Uploader(config)
    
    # Initialize ML components (optional) - SYNCHRONOUSLY
    ml_classifier = None
    ml_tracking = None
    
    try:
        ml_classifier = MLClassifier(config)
        if ml_classifier.is_loaded:
            logger.info("ML Classifier loaded: %s (%d classes)", 
                       ml_classifier.model_version, ml_classifier.num_classes)
        else:
            logger.warning("ML Classifier failed to load, running without predictions")
            ml_classifier = None
    except Exception as e:
        logger.warning("ML Classifier init failed: %s", e)
    
    try:
        if ml_classifier:
            ml_tracking = MLTrackingClient(config, spreadsheet=sheets_client.spreadsheet)
            logger.info("ML Tracking initialized")
    except Exception as e:
        logger.warning("ML Tracking init failed (continuing without): %s", e)
    
    # Now run async part
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    main_task = loop.create_task(main_async(
        config=config,
        sheets_client=sheets_client,
        s3_uploader=s3_uploader,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
        logger=logger,
    ))
    
    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        main_task.cancel()
        try:
            loop.run_until_complete(main_task)
        except asyncio.CancelledError:
            pass
    finally:
        loop.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()

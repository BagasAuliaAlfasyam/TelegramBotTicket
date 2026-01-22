"""
Run Reporting Bot
==================
Entry point for the reporting (admin) bot.
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from telegram.ext import ApplicationBuilder, CommandHandler

from src.core import Config, setup_logging
from src.ml import MLClassifier, MLTrackingClient
from src.bots import AdminCommandHandler, TrendAlertService


def main() -> None:
    """Run the reporting bot with admin commands."""
    # Load configuration
    config = Config.from_env()
    setup_logging(config.debug)
    
    logger = logging.getLogger(__name__)
    
    # Validate reporting token
    if not config.telegram_reporting_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN_REPORTING is required")
        sys.exit(1)
    
    logger.info("Starting reporting bot...")
    
    # Initialize ML components
    ml_classifier = None
    ml_tracking = None
    
    try:
        ml_classifier = MLClassifier(config)
        if ml_classifier.is_loaded:
            logger.info("ML Classifier loaded: v%s", ml_classifier.model_version)
    except Exception as e:
        logger.warning("ML Classifier init failed: %s", e)
    
    try:
        ml_tracking = MLTrackingClient(config)
        logger.info("ML Tracking initialized")
    except Exception as e:
        logger.warning("ML Tracking init failed: %s", e)
    
    # Initialize admin handler
    admin_handler = AdminCommandHandler(
        config=config,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
        admin_chat_ids=config.admin_user_ids,
    )
    
    # Build application
    application = ApplicationBuilder().token(config.telegram_reporting_bot_token).build()
    
    # Register admin commands
    application.add_handler(CommandHandler("stats", admin_handler.stats))
    application.add_handler(CommandHandler("report", admin_handler.report))
    application.add_handler(CommandHandler("modelstatus", admin_handler.model_status))
    application.add_handler(CommandHandler("pendingreview", admin_handler.pending_review))
    application.add_handler(CommandHandler("retrainstatus", admin_handler.retrain_status))
    application.add_handler(CommandHandler("helpml", admin_handler.help_admin))
    
    # Setup trend alerts scheduler (every 6 hours)
    if ml_tracking and config.telegram_admin_chat_id:
        trend_service = TrendAlertService(
            ml_tracking=ml_tracking,
            alert_chat_id=config.telegram_admin_chat_id,
        )
        
        async def send_trend_alert(context):
            alert_msg = trend_service.check_and_alert()
            if alert_msg:
                await context.bot.send_message(
                    chat_id=config.telegram_admin_chat_id,
                    text=alert_msg,
                    parse_mode="Markdown"
                )
        
        # Schedule every 6 hours
        job_queue = application.job_queue
        if job_queue:
            job_queue.run_repeating(send_trend_alert, interval=6*60*60, first=60)
            logger.info("Trend alert scheduler started (every 6 hours)")
    
    logger.info("Reporting bot is running. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()

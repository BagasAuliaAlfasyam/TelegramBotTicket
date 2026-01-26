"""
Run Collecting Bot
===================

Entry point untuk menjalankan collecting bot saja.

Bot ini menangani:
    - Reply ops di grup Telegram
    - Parsing format "solving, APP -initials"
    - Prediksi ML untuk kategorisasi Symtomps
    - Simpan data ke Google Sheets (Logs + ML_Tracking)
    - Upload media ke S3/MinIO

Usage:
    python scripts/run_collecting.py

Author: Bagas Aulia Alfasyam
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Config, setup_logging
from src.services import GoogleSheetsClient, S3Uploader
from src.ml import MLClassifier, MLTrackingClient
from src.bots import build_collecting_application


def main() -> None:
    """
    Jalankan collecting bot.
    
    Proses inisialisasi:
        1. Load konfigurasi dari .env
        2. Validasi konfigurasi wajib
        3. Inisialisasi Google Sheets client
        4. Inisialisasi S3 uploader
        5. Inisialisasi ML Classifier (optional)
        6. Inisialisasi ML Tracking (optional)
        7. Build dan jalankan Telegram bot
    """
    # Load configuration
    config = Config.from_env()
    setup_logging(config.debug)
    
    logger = logging.getLogger(__name__)
    
    # Validate config
    errors = config.validate()
    if errors:
        logger.error("Configuration errors: %s", errors)
        sys.exit(1)
    
    logger.info("Starting collecting bot...")
    logger.info("Config: %s", config)
    
    # Initialize services
    sheets_client = GoogleSheetsClient(config)
    s3_uploader = S3Uploader(config)
    
    # Initialize ML components (optional - graceful fallback if not available)
    ml_classifier = None
    ml_tracking = None
    
    try:
        ml_classifier = MLClassifier(config)
        if ml_classifier.is_loaded:
            logger.info("ML Classifier loaded: v%s (%d classes)", 
                       ml_classifier.model_version, ml_classifier.num_classes)
        else:
            logger.warning("ML Classifier failed to load, running without predictions")
            ml_classifier = None
    except Exception as e:
        logger.warning("ML Classifier init failed: %s", e)
    
    try:
        if ml_classifier:
            # Reuse spreadsheet from sheets_client to avoid duplicate connections
            ml_tracking = MLTrackingClient(config, spreadsheet=sheets_client.spreadsheet)
            logger.info("ML Tracking initialized")
    except Exception as e:
        logger.warning("ML Tracking init failed: %s", e)
    
    # Build and run application
    application = build_collecting_application(
        config=config,
        sheets_client=sheets_client,
        s3_uploader=s3_uploader,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
    )
    
    logger.info("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=["message", "edited_message"])


if __name__ == "__main__":
    main()

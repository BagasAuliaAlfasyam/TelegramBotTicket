"""Entrypoint script for running the collecting Telegram bot."""
from __future__ import annotations

import logging

from collecting_bot import build_collecting_application
from config import load_config
from google_sheets_client import GoogleSheetsClient
from s3_uploader import S3Uploader


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    config = load_config()
    sheets_client = GoogleSheetsClient(config)
    s3_uploader = S3Uploader(config)
    application = build_collecting_application(config, sheets_client, s3_uploader)
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()

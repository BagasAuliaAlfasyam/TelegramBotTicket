"""
Collector Bot — Entry Point
=============================
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.collector.src.bot import OpsCollector
from services.shared.config import CollectorBotConfig, setup_logging
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)


def build_application(config: CollectorBotConfig) -> Application:
    collector = OpsCollector(config)
    app = ApplicationBuilder().token(config.telegram_collecting_bot_token).build()

    app.add_handler(CommandHandler(["health", "ping"], collector.health))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), collector.handle_ops_reply))
    # EDITED_MESSAGE intentionally NOT handled — edits must not re-trigger full flow + duplicate notifs
    return app


def main():
    config = CollectorBotConfig.from_env()
    setup_logging(config.debug, "collector-bot")

    app = build_application(config)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

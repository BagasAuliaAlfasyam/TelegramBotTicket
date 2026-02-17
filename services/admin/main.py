"""
Admin Bot — Entry Point
=========================
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.admin.src.bot import AdminCommandHandler
from services.shared.config import AdminBotConfig, setup_logging
from telegram import BotCommand
from telegram.ext import Application, ApplicationBuilder, CommandHandler


def build_application(config: AdminBotConfig) -> Application:
    handler = AdminCommandHandler(
        data_api_url=config.data_api_url,
        prediction_api_url=config.prediction_api_url,
        training_api_url=config.training_api_url,
        admin_user_ids=config.admin_user_ids,
    )

    app = ApplicationBuilder().token(config.telegram_reporting_bot_token).build()

    app.add_handler(CommandHandler("stats", handler.stats))
    app.add_handler(CommandHandler("report", handler.report))
    app.add_handler(CommandHandler("modelstatus", handler.model_status))
    app.add_handler(CommandHandler("pendingreview", handler.pending_review))
    app.add_handler(CommandHandler("updatestats", handler.update_stats))
    app.add_handler(CommandHandler("retrainstatus", handler.retrain_status))
    app.add_handler(CommandHandler("retrain", handler.retrain))
    app.add_handler(CommandHandler("reloadmodel", handler.reload_model))
    app.add_handler(CommandHandler("tiketreport", handler.tiket_report))
    app.add_handler(CommandHandler("trendbulan", handler.trend_bulan))
    app.add_handler(CommandHandler("trendmingguan", handler.trend_mingguan))
    app.add_handler(CommandHandler("helpml", handler.help_admin))
    app.add_handler(CommandHandler("help", handler.help_admin))

    async def post_init(application: Application) -> None:
        commands = [
            BotCommand("help", "Tampilkan bantuan"),
            BotCommand("stats", "Statistik ML"),
            BotCommand("report", "Laporan performa"),
            BotCommand("modelstatus", "Status model + Gemini"),
            BotCommand("retrain", "Retrain model"),
            BotCommand("reloadmodel", "Reload model terbaru"),
            BotCommand("tiketreport", "Laporan tiket & SLA"),
            BotCommand("trendbulan", "Trend bulanan"),
        ]
        await application.bot.set_my_commands(commands)

    app.post_init = post_init
    return app


def main():
    config = AdminBotConfig.from_env()
    setup_logging(config.debug, "admin-bot")
    app = build_application(config)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

"""
Admin Bot — Entry Point
=========================
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.admin.src.bot import AdminCommandHandler, TrendAlertService
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

    # -- Statistik & Monitoring --
    app.add_handler(CommandHandler("stats", handler.stats))
    app.add_handler(CommandHandler("report", handler.report))
    app.add_handler(CommandHandler("modelstatus", handler.model_status))
    app.add_handler(CommandHandler("pendingreview", handler.pending_review))

    # -- Model Management --
    app.add_handler(CommandHandler("retrain", handler.retrain))
    app.add_handler(CommandHandler("retrainstatus", handler.retrain_status))
    app.add_handler(CommandHandler("reloadmodel", handler.reload_model))
    app.add_handler(CommandHandler("updatestats", handler.update_stats))
    app.add_handler(CommandHandler("mlflowstatus", handler.mlflow_status))
    app.add_handler(CommandHandler("mlflowpromote", handler.mlflow_promote))

    # -- Laporan --
    app.add_handler(CommandHandler("tiketreport", handler.tiket_report))
    app.add_handler(CommandHandler("trendbulan", handler.trend_bulan))
    app.add_handler(CommandHandler("trendmingguan", handler.trend_mingguan))

    # -- Help --
    app.add_handler(CommandHandler("help", handler.help_admin))
    app.add_handler(CommandHandler("start", handler.help_admin))

    async def post_init(application: Application) -> None:
        commands = [
            BotCommand("help", "📖 Tampilkan bantuan"),
            BotCommand("stats", "📊 Dashboard ML real-time"),
            BotCommand("report", "📈 Laporan mingguan/bulanan"),
            BotCommand("modelstatus", "🤖 Status model & Gemini"),
            BotCommand("pendingreview", "📋 Tiket perlu review"),
            BotCommand("retrain", "🔧 Training ulang model"),
            BotCommand("retrainstatus", "🔄 Cek progress training"),
            BotCommand("reloadmodel", "🔃 Load model terbaru"),
            BotCommand("tiketreport", "📋 Laporan tiket & SLA"),
            BotCommand("trendbulan", "📊 Top symtomps bulanan per app"),
            BotCommand("trendmingguan", "📅 Top symtomps mingguan"),
            BotCommand("updatestats", "📊 Update statistik hourly"),
            BotCommand("mlflowstatus", "📦 Status MLflow registry"),
            BotCommand("mlflowpromote", "🏆 Promote model version"),
        ]
        await application.bot.set_my_commands(commands)

        # Start TrendAlertService if admin_chat_id is set
        admin_ids = config.admin_user_ids
        if admin_ids:
            alert_service = TrendAlertService(
                data_url=config.data_api_url,
                prediction_url=config.prediction_api_url,
                bot=application.bot,
                admin_chat_id=admin_ids[0],  # Alert to first admin
                interval_seconds=3600,
            )
            await alert_service.start()
            application.bot_data["alert_service"] = alert_service

    app.post_init = post_init
    return app


def main():
    config = AdminBotConfig.from_env()
    setup_logging(config.debug, "admin-bot")
    app = build_application(config)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

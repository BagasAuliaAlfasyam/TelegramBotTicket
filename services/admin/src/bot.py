"""
Admin Bot — Microservice Version
==================================
All data operations go through Data API and Prediction API via HTTP.
No direct Google Sheets or ML model access.

Commands preserved from monolith:
  /stats, /report, /modelstatus, /pendingreview
  /retrain, /reloadmodel, /mlflowstatus
  /tiketreport, /trendbulan, /trendmingguan
  /updatestats, /retrainstatus, /helpml
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
from telegram import Update
from telegram.ext import ContextTypes

_LOGGER = logging.getLogger(__name__)
TZ = ZoneInfo("Asia/Jakarta")


class AdminCommandHandler:
    """Admin commands — calls Data API, Prediction API, Training API."""

    def __init__(
        self,
        data_api_url: str,
        prediction_api_url: str,
        training_api_url: str,
        admin_user_ids: list[int] | None = None,
    ):
        self._data_url = data_api_url.rstrip("/")
        self._prediction_url = prediction_api_url.rstrip("/")
        self._training_url = training_api_url.rstrip("/")
        self._admin_ids = admin_user_ids or []
        self._http = httpx.AsyncClient(timeout=60.0)

    def _is_admin(self, uid: int) -> bool:
        return not self._admin_ids or uid in self._admin_ids

    # =================== /stats ===================
    async def stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            resp = await self._http.get(f"{self._data_url}/stats/realtime")
            s = resp.json()
            if not s or s.get("total_predictions", 0) == 0:
                await update.message.reply_text("📊 No predictions recorded yet.")
                return

            total = s["total_predictions"]
            auto = s.get("auto_count", 0)
            auto_pct = auto / total * 100 if total else 0

            # Get model info
            model_ver = "unknown"
            try:
                mr = await self._http.get(f"{self._prediction_url}/model/info")
                model_ver = mr.json().get("version", "unknown")
            except Exception:
                pass

            msg = (
                f"📊 <b>ML Stats (All Time)</b>\n\n"
                f"📈 Total Predictions: {total}\n"
                f"🎯 Avg Confidence: {s.get('avg_confidence', 0):.1f}%\n\n"
                f"<b>Distribution:</b>\n"
                f"  ✅ AUTO (≥80%): {auto} ({auto_pct:.1f}%)\n"
                f"  🔶 HIGH (70-80%): {s.get('high_count', 0)}\n"
                f"  🟡 MEDIUM (50-70%): {s.get('medium_count', 0)}\n"
                f"  🔴 MANUAL (&lt;50%): {s.get('manual_count', 0)}\n\n"
                f"<b>Review:</b>\n"
                f"  📋 Pending: {s.get('pending_count', 0)}\n"
                f"  ✅ Reviewed: {s.get('reviewed_count', 0)}\n"
            )
            if s.get("gemini_count", 0) > 0 or s.get("hybrid_count", 0) > 0:
                msg += (
                    f"\n<b>Gemini Cascade:</b>\n"
                    f"  🤖 Gemini calls: {s.get('gemini_count', 0)}\n"
                    f"  🔀 Hybrid: {s.get('hybrid_count', 0)}\n"
                )
            msg += f"\n🤖 Model: {model_ver}"
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /report ===================
    async def report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        period = args[0].lower() if args else "weekly"
        if period not in ("weekly", "monthly"):
            await update.message.reply_text("Usage: /report [weekly|monthly]")
            return
        try:
            resp = await self._http.get(f"{self._data_url}/stats/{period}")
            s = resp.json()
            if not s or s.get("total_predictions", 0) == 0:
                await update.message.reply_text(f"📊 No data for {period}.")
                return

            total = s["total_predictions"]
            auto = s.get("auto_count", 0)
            auto_pct = auto / total * 100 if total else 0

            msg = (
                f"📊 <b>{period.title()} Report</b>\n"
                f"Generated: {datetime.now(TZ).strftime('%d %b %Y %H:%M')}\n\n"
                f"📈 Total: {total}\n"
                f"🎯 Avg Conf: {s.get('avg_confidence', 0):.1f}%\n"
                f"⚡ Automation: {auto_pct:.1f}%\n\n"
                f"✅ AUTO: {auto}  🔶 HIGH: {s.get('high_review_count', 0)}\n"
                f"🟡 MEDIUM: {s.get('medium_review_count', 0)}  🔴 MANUAL: {s.get('manual_count', 0)}\n"
                f"📝 Reviewed: {s.get('reviewed_count', 0)}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /modelstatus ===================
    async def model_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            resp = await self._http.get(f"{self._prediction_url}/model/info")
            m = resp.json()
            thresholds = m.get('thresholds', {})
            msg = (
                f"🤖 <b>Model Status</b>\n\n"
                f"Version: {m.get('version', '?')}\n"
                f"Classes: {m.get('num_classes', 0)}\n"
                f"Loaded: {'✅' if m.get('is_loaded') else '❌'}\n"
                f"Gemini: {'✅ Enabled' if m.get('gemini_enabled') else '❌ Disabled'}\n"
                f"Thresholds: AUTO={thresholds.get('AUTO', 0.80)}, HIGH={thresholds.get('HIGH_REVIEW', 0.70)}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /reloadmodel ===================
    async def reload_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        stage = args[0].capitalize() if args else "Production"
        await update.message.reply_text(f"🔄 Reloading model ({stage})...")
        try:
            resp = await self._http.post(
                f"{self._prediction_url}/model/reload",
                json={"stage": stage},
            )
            r = resp.json()
            if r.get("success"):
                await update.message.reply_text(
                    f"✅ Model reloaded!\nVersion: {r.get('version', '?')}\nClasses: {r.get('num_classes', 0)}"
                )
            else:
                await update.message.reply_text(f"❌ Reload failed: {r.get('message', 'unknown')}")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /retrain ===================
    async def retrain(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        args_lower = [a.lower() for a in args]
        force = "force" in args_lower
        tune = any(a.startswith("tune") for a in args_lower)
        tune_trials = 50
        for a in args_lower:
            if a.startswith("tune") and a[4:].isdigit():
                tune_trials = int(a[4:])

        await update.message.reply_text(
            f"🚀 Starting training...\n"
            f"Mode: {'Optuna ' + str(tune_trials) + ' trials' if tune else 'Fixed params'}"
            f"{' (force)' if force else ''}"
        )
        try:
            resp = await self._http.post(
                f"{self._training_url}/train",
                json={"force": force, "tune": tune, "tune_trials": tune_trials},
                timeout=300.0,
            )
            r = resp.json()
            if r.get("status") == "running":
                await update.message.reply_text(
                    "🚀 Training started in background.\n"
                    "Use /retrainstatus to check progress.\n"
                    "Use /reloadmodel after training completes.",
                    parse_mode="HTML",
                )
            elif r.get("success") and r.get("f1_score"):
                await update.message.reply_text(
                    f"✅ <b>Training Complete!</b>\n\n"
                    f"Model: {r.get('model_version', '?')}\n"
                    f"F1: {r.get('f1_score', 'N/A')}\n"
                    f"Samples: {r.get('n_samples', '?')}\n\n"
                    f"Use /reloadmodel to load the new model.",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text(f"❌ Training failed: {r.get('message', 'unknown')}")
        except httpx.TimeoutException:
            await update.message.reply_text("⏳ Training started (running in background). Check /retrainstatus later.")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /retrainstatus ===================
    async def retrain_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            resp = await self._http.get(f"{self._training_url}/status")
            r = resp.json()
            data_resp = await self._http.get(f"{self._data_url}/training/data")
            d = data_resp.json()
            total = d.get("total_samples", 0)
            msg = (
                f"🔄 <b>Retrain Status</b>\n\n"
                f"Training data: {total} samples\n"
                f"Status: {r.get('status', 'idle')}\n"
                f"Last train: {r.get('last_trained', 'never')}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /updatestats ===================
    async def update_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            resp = await self._http.post(f"{self._data_url}/stats/hourly")
            r = resp.json()
            await update.message.reply_text(
                f"✅ Stats updated.\n{r.get('stats', {})}"
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /tiketreport ===================
    async def tiket_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Ticket report — delegates to Data API for raw data, calculates locally."""
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        await update.message.reply_text("⏳ Generating tiket report... (data from Data API)")
        try:
            # For now, a simplified version that calls stats
            resp = await self._http.get(f"{self._data_url}/stats/realtime")
            s = resp.json()
            total = s.get("total_predictions", 0)
            await update.message.reply_text(
                f"📊 <b>Ticket Report</b>\n\nTotal tickets processed: {total}\n"
                f"(Full ticket report will be expanded in future.)",
                parse_mode="HTML",
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /trendbulan ===================
    async def trend_bulan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        if not args:
            await update.message.reply_text("Usage: /trendbulan [MIT|MIS] [bulan] [tahun]")
            return
        await update.message.reply_text(
            f"⏳ Retrieving trend data for {args[0].upper()}..."
        )
        # Simplified — full implementation would call Data API with filter params
        await update.message.reply_text(
            "📊 Trend bulan report will be expanded. Use /stats for current ML metrics."
        )

    # =================== /trendmingguan ===================
    async def trend_mingguan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        await update.message.reply_text("📊 Weekly trend — coming soon in microservice v2.1")

    # =================== /pendingreview ===================
    async def pending_review(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            resp = await self._http.get(f"{self._data_url}/stats/realtime")
            s = resp.json()
            pending = s.get("pending_count", 0)
            msg = (
                f"📋 <b>Pending Review</b>\n\n"
                f"Total pending: {pending}\n"
            )
            if pending == 0:
                msg += "✅ No items pending review!"
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    # =================== /helpml ===================
    async def help_admin(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        msg = (
            "🤖 <b>Admin Commands (Microservice)</b>\n\n"
            "📊 <b>MONITORING</b>\n"
            "/stats — Statistik ML real-time\n"
            "/report [weekly|monthly] — Laporan performa\n"
            "/modelstatus — Info model + Gemini status\n"
            "/pendingreview — Pending review\n"
            "/updatestats — Update statistik\n\n"
            "🔄 <b>RETRAINING</b>\n"
            "/retrainstatus — Cek kesiapan retrain\n"
            "/retrain [force] [tune] — Training ulang\n"
            "/reloadmodel [stage] — Hot reload model\n\n"
            "📋 <b>REPORTS</b>\n"
            "/tiketreport — Laporan tiket & SLA\n"
            "/trendbulan [MIT|MIS] — Trend bulanan\n"
            "/trendmingguan — Top tiket mingguan\n\n"
            "/help — Tampilkan bantuan ini"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

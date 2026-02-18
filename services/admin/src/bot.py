"""
Admin Bot — Microservice Version
==================================
All data operations go through Data API and Prediction API via HTTP.
No direct Google Sheets or ML model access.
"""
from __future__ import annotations

import logging
from datetime import datetime
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

    # ---- Helpers ----

    def _is_admin(self, uid: int) -> bool:
        return not self._admin_ids or uid in self._admin_ids

    async def _reply(self, update: Update, text: str, parse_mode: str = "HTML") -> None:
        """Reply to the user's command message (quotes the command)."""
        await update.message.reply_text(
            text,
            parse_mode=parse_mode,
            reply_to_message_id=update.message.message_id,
        )

    async def _api_get(self, url: str) -> dict:
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def _api_post(self, url: str, json: dict | None = None, timeout: float = 60.0) -> dict:
        resp = await self._http.post(url, json=json, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # =================== /help ===================
    async def help_admin(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        msg = (
            "🤖 <b>MyTech Admin Bot</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📊 <b>STATISTIK &amp; MONITORING</b>\n"
            "├ /stats — Dashboard ML real-time\n"
            "├ /report — Laporan mingguan/bulanan\n"
            "├ /modelstatus — Info model &amp; Gemini\n"
            "└ /pendingreview — Tiket perlu review\n\n"
            "🔧 <b>MODEL MANAGEMENT</b>\n"
            "├ /retrain — Training ulang model\n"
            "├ /retrainstatus — Cek progress training\n"
            "└ /reloadmodel — Load model terbaru\n\n"
            "📋 <b>LAPORAN TIKET</b>\n"
            "└ /tiketreport — Ringkasan tiket &amp; SLA\n\n"
            "💡 <i>Semua command bisa langsung di-tap!</i>"
        )
        await self._reply(update, msg)

    # =================== /stats ===================
    async def stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            s = await self._api_get(f"{self._data_url}/stats/realtime")

            total = s.get("total_predictions", 0)
            if total == 0:
                await self._reply(update, "📊 Belum ada prediksi tercatat.")
                return

            auto = s.get("auto_count", 0)
            high = s.get("high_count", 0)
            medium = s.get("medium_count", 0)
            manual = s.get("manual_count", 0)
            auto_pct = auto / total * 100 if total else 0

            # Model version
            model_ver = "unknown"
            try:
                mi = await self._api_get(f"{self._prediction_url}/model/info")
                model_ver = mi.get("version", "unknown")
            except Exception:
                pass

            msg = (
                f"📊 <b>ML Dashboard</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📈 Total Prediksi: <b>{total:,}</b>\n"
                f"🎯 Rata-rata Confidence: <b>{s.get('avg_confidence', 0):.1f}%</b>\n"
                f"⚡ Automation Rate: <b>{auto_pct:.1f}%</b>\n\n"
                f"<b>Distribusi Prediksi:</b>\n"
                f"  ✅ AUTO (≥80%): {auto:,}\n"
                f"  🔶 HIGH (70-80%): {high:,}\n"
                f"  🟡 MEDIUM (50-70%): {medium:,}\n"
                f"  🔴 MANUAL (&lt;50%): {manual:,}\n\n"
                f"<b>Review Status:</b>\n"
                f"  📋 Pending: {s.get('pending_count', 0):,}\n"
                f"  ✅ Reviewed: {s.get('reviewed_count', 0):,}\n"
            )

            gemini = s.get("gemini_count", 0)
            hybrid = s.get("hybrid_count", 0)
            if gemini > 0 or hybrid > 0:
                msg += (
                    f"\n<b>Gemini Cascade:</b>\n"
                    f"  🤖 Gemini: {gemini:,}  🔀 Hybrid: {hybrid:,}\n"
                )

            msg += f"\n🤖 Model: <code>{model_ver}</code>"
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("stats failed")
            await self._reply(update, f"❌ Gagal mengambil statistik.\n<code>{e}</code>")

    # =================== /report ===================
    async def report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        period = args[0].lower() if args else "weekly"
        if period not in ("weekly", "monthly"):
            await self._reply(update, "📖 <b>Cara pakai:</b>\n/report — Laporan mingguan\n/report monthly — Laporan bulanan")
            return
        try:
            s = await self._api_get(f"{self._data_url}/stats/{period}")
            total = s.get("total_predictions", 0)
            if total == 0:
                await self._reply(update, f"📊 Belum ada data untuk periode <b>{period}</b>.")
                return

            auto = s.get("auto_count", 0)
            auto_pct = auto / total * 100 if total else 0
            label = "Mingguan" if period == "weekly" else "Bulanan"

            msg = (
                f"📊 <b>Laporan {label}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 {datetime.now(TZ).strftime('%d %b %Y %H:%M WIB')}\n\n"
                f"📈 Total Prediksi: <b>{total:,}</b>\n"
                f"🎯 Avg Confidence: <b>{s.get('avg_confidence', 0):.1f}%</b>\n"
                f"⚡ Automation Rate: <b>{auto_pct:.1f}%</b>\n\n"
                f"  ✅ AUTO: {auto:,}\n"
                f"  🔶 HIGH: {s.get('high_review_count', 0):,}\n"
                f"  🟡 MEDIUM: {s.get('medium_review_count', 0):,}\n"
                f"  🔴 MANUAL: {s.get('manual_count', 0):,}\n"
                f"  📝 Reviewed: {s.get('reviewed_count', 0):,}"
            )
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("report failed")
            await self._reply(update, f"❌ Gagal mengambil laporan.\n<code>{e}</code>")

    # =================== /modelstatus ===================
    async def model_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            m = await self._api_get(f"{self._prediction_url}/model/info")
            thresholds = m.get("thresholds", {})
            classes = m.get("classes", [])
            top_classes = ", ".join(classes[:5]) if classes else "-"
            trained_at = m.get("trained_at", "-")
            if trained_at and trained_at != "-":
                try:
                    dt = datetime.fromisoformat(trained_at)
                    trained_at = dt.strftime("%d %b %Y %H:%M")
                except Exception:
                    pass

            msg = (
                f"🤖 <b>Model Status</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📦 Version: <code>{m.get('version', '?')}</code>\n"
                f"📊 Classes: <b>{m.get('num_classes', 0)}</b>\n"
                f"✅ Loaded: {'Ya ✅' if m.get('is_loaded') else 'Tidak ❌'}\n"
                f"🤖 Gemini: {'Aktif ✅' if m.get('gemini_enabled') else 'Nonaktif ❌'}\n"
                f"📅 Trained: {trained_at}\n\n"
                f"<b>Thresholds:</b>\n"
                f"  AUTO ≥ {thresholds.get('AUTO', 0.80)}\n"
                f"  HIGH ≥ {thresholds.get('HIGH_REVIEW', 0.70)}\n"
                f"  MEDIUM ≥ {thresholds.get('MEDIUM_REVIEW', 0.50)}\n\n"
                f"<b>Top Classes:</b>\n  {top_classes}"
            )
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("modelstatus failed")
            await self._reply(update, f"❌ Gagal mengambil info model.\n<code>{e}</code>")

    # =================== /pendingreview ===================
    async def pending_review(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            s = await self._api_get(f"{self._data_url}/stats/realtime")
            pending = s.get("pending_count", 0)
            reviewed = s.get("reviewed_count", 0)
            total = pending + reviewed

            msg = (
                f"📋 <b>Review Status</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"⏳ Pending: <b>{pending:,}</b>\n"
                f"✅ Reviewed: <b>{reviewed:,}</b>\n"
                f"📊 Total: {total:,}\n\n"
            )
            if pending == 0:
                msg += "🎉 Semua prediksi sudah di-review!"
            else:
                msg += f"⚠️ Ada <b>{pending}</b> prediksi yang perlu review."
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("pendingreview failed")
            await self._reply(update, f"❌ Gagal mengambil data review.\n<code>{e}</code>")

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

        mode_text = f"Optuna {tune_trials} trials" if tune else "Fixed params"
        await self._reply(
            update,
            f"🚀 <b>Memulai Training...</b>\n\n"
            f"Mode: {mode_text}{' (force)' if force else ''}\n"
            f"⏳ Proses berjalan di background...",
        )
        try:
            r = await self._api_post(
                f"{self._training_url}/train",
                json={"force": force, "tune": tune, "tune_trials": tune_trials},
                timeout=300.0,
            )
            if r.get("status") == "running":
                await self._reply(
                    update,
                    "✅ <b>Training dimulai!</b>\n\n"
                    "Gunakan:\n"
                    "├ /retrainstatus — Cek progress\n"
                    "└ /reloadmodel — Load model setelah selesai",
                )
            elif r.get("success") and r.get("f1_score"):
                await self._reply(
                    update,
                    f"✅ <b>Training Selesai!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📦 Model: <code>{r.get('model_version', '?')}</code>\n"
                    f"🎯 F1 Score: <b>{r.get('f1_score', 'N/A')}</b>\n"
                    f"📊 Samples: {r.get('n_samples', '?'):,}\n\n"
                    f"➡️ Gunakan /reloadmodel untuk load model baru.",
                )
            else:
                await self._reply(update, f"❌ Training gagal: {r.get('message', 'unknown')}")
        except httpx.TimeoutException:
            await self._reply(update, "⏳ Training berjalan di background (timeout > 5min).\nCek /retrainstatus nanti.")
        except Exception as e:
            _LOGGER.exception("retrain failed")
            await self._reply(update, f"❌ Gagal memulai training.\n<code>{e}</code>")

    # =================== /retrainstatus ===================
    async def retrain_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            r = await self._api_get(f"{self._training_url}/status")
            d = await self._api_get(f"{self._data_url}/training/data")
            total = d.get("total_samples", 0)

            status = r.get("status", "idle")
            status_emoji = {"idle": "💤", "running": "🔄", "completed": "✅", "failed": "❌"}.get(status, "❓")
            last_trained = r.get("last_trained", "belum pernah")

            msg = (
                f"🔄 <b>Training Status</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"📊 Training Data: <b>{total:,}</b> samples\n"
                f"{status_emoji} Status: <b>{status}</b>\n"
                f"📅 Terakhir Train: {last_trained}\n"
            )

            if status == "idle":
                msg += "\n💡 Gunakan /retrain untuk mulai training."
            elif status == "running":
                msg += "\n⏳ Training sedang berjalan..."
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("retrainstatus failed")
            await self._reply(update, f"❌ Gagal mengambil status training.\n<code>{e}</code>")

    # =================== /reloadmodel ===================
    async def reload_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        stage = args[0].capitalize() if args else "Production"
        await self._reply(update, f"🔄 Reloading model (<code>{stage}</code>)...")
        try:
            r = await self._api_post(f"{self._prediction_url}/model/reload", json={"stage": stage})
            if r.get("success"):
                # Get updated model info
                try:
                    mi = await self._api_get(f"{self._prediction_url}/model/info")
                    ver = mi.get("version", "?")
                    classes = mi.get("num_classes", 0)
                except Exception:
                    ver = r.get("new_version", "?")
                    classes = "?"
                await self._reply(
                    update,
                    f"✅ <b>Model Reloaded!</b>\n\n"
                    f"📦 Version: <code>{ver}</code>\n"
                    f"📊 Classes: <b>{classes}</b>\n"
                    f"🔄 {r.get('old_version', '?')} → {r.get('new_version', '?')}",
                )
            else:
                await self._reply(update, f"❌ Reload gagal: {r.get('message', 'unknown')}")
        except Exception as e:
            _LOGGER.exception("reloadmodel failed")
            await self._reply(update, f"❌ Gagal reload model.\n<code>{e}</code>")

    # =================== /tiketreport ===================
    async def tiket_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            s = await self._api_get(f"{self._data_url}/stats/realtime")
            total = s.get("total_predictions", 0)
            auto = s.get("auto_count", 0)
            pending = s.get("pending_count", 0)
            auto_pct = auto / total * 100 if total else 0

            # Also get training data count for more info
            try:
                d = await self._api_get(f"{self._data_url}/training/data")
                training_samples = d.get("total_samples", 0)
            except Exception:
                training_samples = "?"

            msg = (
                f"📋 <b>Laporan Tiket</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 {datetime.now(TZ).strftime('%d %b %Y %H:%M WIB')}\n\n"
                f"📈 Total Tiket Diproses: <b>{total:,}</b>\n"
                f"⚡ Automation Rate: <b>{auto_pct:.1f}%</b>\n"
                f"📋 Pending Review: <b>{pending:,}</b>\n"
                f"📊 Training Samples: <b>{training_samples:,}</b>\n\n"
                f"<b>Distribusi:</b>\n"
                f"  ✅ AUTO: {auto:,}\n"
                f"  🔶 HIGH: {s.get('high_count', 0):,}\n"
                f"  🟡 MEDIUM: {s.get('medium_count', 0):,}\n"
                f"  🔴 MANUAL: {s.get('manual_count', 0):,}"
            )
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("tiketreport failed")
            await self._reply(update, f"❌ Gagal membuat laporan tiket.\n<code>{e}</code>")

"""
Admin Bot — Microservice Version
==================================
All data operations go through Data API and Prediction API via HTTP.
No direct Google Sheets or ML model access.
"""
from __future__ import annotations

import asyncio
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

    async def _api_post(self, url: str, json: dict | None = None,
                       params: dict | None = None, timeout: float = 60.0) -> dict:
        resp = await self._http.post(url, json=json, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _analyze_trend(stats: dict) -> str:
        """Auto-generate trend insights from stats."""
        insights: list[str] = []
        total = stats.get("total_predictions", 0)
        if total == 0:
            return ""
        auto = stats.get("auto_count", 0)
        manual = stats.get("manual_count", 0)
        auto_rate = auto / total * 100
        manual_rate = manual / total * 100
        avg_conf = stats.get("avg_confidence", 0)

        if auto_rate >= 90:
            insights.append("\u2705 Excellent automation rate")
        elif auto_rate >= 80:
            insights.append("\u2705 Good automation rate")
        elif auto_rate < 70:
            insights.append("\u26a0\ufe0f Low automation rate — consider retraining")

        if avg_conf >= 90:
            insights.append("\u2705 High model confidence")
        elif avg_conf < 80:
            insights.append("\u26a0\ufe0f Avg confidence dropping — monitor closely")

        if manual_rate > 20:
            insights.append("\U0001f534 High manual classification rate")

        pending = stats.get("pending_count", 0)
        if pending > 50:
            insights.append(f"\U0001f4cb Large review queue ({pending:,} pending)")

        return "\n".join(insights) if insights else "No significant trends detected."

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
            "├ /pendingreview — Tiket perlu review\n"
            "└ /updatestats — Hitung ulang stats harian\n\n"
            "🔧 <b>MODEL MANAGEMENT</b>\n"
            "├ /retrain — Training ulang model\n"
            "├ /retrainstatus — Cek progress training\n"
            "├ /reloadmodel — Load model terbaru\n"
            "├ /mlflowstatus — Status MLflow registry\n"
            "└ /mlflowpromote — Promote version\n"
            "   <code>/mlflowpromote [version]</code>\n\n"
            "📋 <b>LAPORAN TIKET</b>\n"
            "├ /tiketreport — Laporan tiket &amp; SLA\n"
            "│  <code>/tiketreport monthly [bln] [thn] [MIT|MIS]</code>\n"
            "│  <code>/tiketreport quarterly [q] [thn] [MIT|MIS]</code>\n"
            "├ /trendbulan — Top 10 symtomps per app\n"
            "│  <code>/trendbulan MIT [bln] [thn]</code>\n"
            "└ /trendmingguan — Top 5 symtomps per minggu\n"
            "   <code>/trendmingguan [minggu] [bln] [thn] [MIT|MIS]</code>\n\n"
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
                f"  📝 Reviewed: {s.get('reviewed_count', 0):,}\n"
            )

            # Trend analysis
            trend = self._analyze_trend(s)
            if trend:
                msg += f"\n<b>📈 Trend Analysis:</b>\n{trend}"

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
                f"📅 Trained: {trained_at}\n"
            )

            # Training metrics
            accuracy = m.get("training_accuracy")
            samples = m.get("training_samples")
            if accuracy:
                try:
                    msg += f"🎯 Training Accuracy: <b>{float(accuracy)*100:.1f}%</b>\n"
                except (ValueError, TypeError):
                    pass
            if samples:
                msg += f"📋 Training Samples: <b>{samples:,}</b>\n"

            msg += (
                f"\n<b>Thresholds:</b>\n"
                f"  ✅ AUTO ≥ {float(thresholds.get('AUTO', 0.80))*100:.0f}%\n"
                f"  🔶 HIGH ≥ {float(thresholds.get('HIGH_REVIEW', 0.70))*100:.0f}%\n"
                f"  🟡 MEDIUM ≥ {float(thresholds.get('MEDIUM_REVIEW', 0.50))*100:.0f}%\n"
                f"  🔴 MANUAL &lt; {float(thresholds.get('MEDIUM_REVIEW', 0.50))*100:.0f}%\n"
            )

            # MLflow info
            mlflow_info = m.get("mlflow", {})
            if mlflow_info.get("enabled"):
                prod = mlflow_info.get("production_model", {})
                prod_ver = prod.get("version", "—")
                msg += (
                    f"\n<b>☁️ MLflow:</b>\n"
                    f"  URI: <code>{mlflow_info.get('tracking_uri', '—')}</code>\n"
                    f"  🟢 Production: v{prod_ver}\n"
                )

            msg += f"\n<b>Top Classes:</b>\n  {top_classes}"
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

            # Priority breakdown from confidence distribution
            manual = s.get("manual_count", 0)
            high = s.get("high_count", 0)
            medium = s.get("medium_count", 0)

            msg = (
                f"📋 <b>Review Status</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"⏳ Pending: <b>{pending:,}</b>\n"
                f"✅ Reviewed: <b>{reviewed:,}</b>\n"
                f"📊 Total: {total:,}\n\n"
                f"<b>Priority Breakdown:</b>\n"
                f"  🔴 MANUAL (Critical): {manual:,}\n"
                f"  🔶 HIGH REVIEW: {high:,}\n"
                f"  🟡 MEDIUM REVIEW: {medium:,}\n"
            )

            # Google Sheets review link
            sheet_url = s.get("sheet_url", "")
            if sheet_url:
                msg += f"\n\U0001f4ce <a href=\"{sheet_url}\">Buka Google Sheets</a>\n"

            if pending == 0:
                msg += "\n\U0001f389 Semua prediksi sudah di-review!"
            elif pending > 50:
                msg += f"\n\u26a0\ufe0f <b>{pending}</b> prediksi perlu review \u2014 prioritaskan MANUAL!"
            else:
                msg += f"\n\u26a0\ufe0f Ada <b>{pending}</b> prediksi yang perlu review."
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

        # Check reviewed count threshold (unless force)
        if not force:
            try:
                s = await self._api_get(f"{self._data_url}/stats/realtime")
                reviewed = s.get("reviewed_count", 0)
                if reviewed < 100:
                    await self._reply(
                        update,
                        f"⚠️ <b>Data belum cukup</b>\n\n"
                        f"Reviewed: {reviewed}/100 minimum\n"
                        f"Butuh {100 - reviewed} lagi.\n\n"
                        f"Gunakan <code>/retrain force</code> untuk skip pengecekan.",
                    )
                    return
            except Exception:
                pass

        mode_text = f"Optuna {tune_trials} trials" if tune else "Fixed params"
        progress_msg = await update.message.reply_text(
            f"🚀 <b>Memulai Training...</b>\n\n"
            f"Mode: {mode_text}{' (force)' if force else ''}\n"
            f"⏳ Proses berjalan...",
            parse_mode="HTML",
            reply_to_message_id=update.message.message_id,
        )

        try:
            r = await self._api_post(
                f"{self._training_url}/train",
                json={"force": force, "tune": tune, "tune_trials": tune_trials},
                timeout=10.0,
            )

            if r.get("status") == "running":
                # Async progress polling
                asyncio.create_task(self._poll_training(progress_msg))
            elif r.get("success") and r.get("f1_score"):
                await progress_msg.edit_text(
                    f"✅ <b>Training Selesai!</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"📦 Model: <code>{r.get('model_version', '?')}</code>\n"
                    f"🎯 F1 Score: <b>{r.get('f1_score', 'N/A')}</b>\n"
                    f"📊 Samples: {r.get('n_samples', '?'):,}\n\n"
                    f"➡️ /reloadmodel untuk load model baru\n"
                    f"➡️ /mlflowpromote untuk promote ke Production",
                    parse_mode="HTML",
                )
            else:
                await progress_msg.edit_text(
                    f"❌ Training gagal: {r.get('message', 'unknown')}",
                    parse_mode="HTML",
                )
        except httpx.TimeoutException:
            # Training started but took too long for initial response — poll
            asyncio.create_task(self._poll_training(progress_msg))
        except Exception as e:
            _LOGGER.exception("retrain failed")
            await progress_msg.edit_text(f"❌ Gagal memulai training.\n<code>{e}</code>", parse_mode="HTML")

    async def _poll_training(self, msg) -> None:
        """Poll training status and live-edit the progress message."""
        import time as _time
        start = _time.time()
        last_status = ""
        try:
            for _ in range(120):  # max 30 min (120 * 15s)
                await asyncio.sleep(15)
                elapsed = int(_time.time() - start)
                mm, ss = divmod(elapsed, 60)
                try:
                    r = await self._api_get(f"{self._training_url}/status")
                except Exception:
                    continue

                status = r.get("status", "unknown")
                if status == "completed":
                    result = r.get("last_result", {})
                    await msg.edit_text(
                        f"✅ <b>Training Selesai!</b> ({mm}m {ss}s)\n"
                        f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"🎯 F1 Score: <b>{result.get('f1_score', 'N/A')}</b>\n"
                        f"📊 Samples: {result.get('n_samples', '?')}\n"
                        f"📦 Version: <code>{result.get('model_version', '?')}</code>\n\n"
                        f"➡️ /reloadmodel untuk load model baru\n"
                        f"➡️ /mlflowpromote untuk promote ke Production",
                        parse_mode="HTML",
                    )
                    # Auto-reload and mark trained
                    try:
                        await self._api_post(f"{self._prediction_url}/model/reload", json={"stage": "Staging"})
                        await self._api_post(f"{self._data_url}/training/mark")
                    except Exception:
                        pass
                    return
                elif status == "failed":
                    await msg.edit_text(
                        f"❌ <b>Training Gagal</b> ({mm}m {ss}s)\n\n"
                        f"{r.get('last_result', {}).get('message', 'Unknown error')}",
                        parse_mode="HTML",
                    )
                    return
                elif status != last_status:
                    last_status = status
                    await msg.edit_text(
                        f"🔄 <b>Training...</b> ({mm}m {ss}s)\n\n"
                        f"Status: {status}",
                        parse_mode="HTML",
                    )
        except Exception as e:
            _LOGGER.exception("Training poll failed: %s", e)

    # =================== /retrainstatus ===================
    async def retrain_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            r = await self._api_get(f"{self._training_url}/status")
            d = await self._api_get(f"{self._data_url}/training/data")
            total = d.get("total_samples", 0)
            tracking_data = d.get("tracking_data", [])

            status = r.get("status", "idle")
            status_emoji = {"idle": "\U0001f4a4", "running": "\U0001f504", "completed": "\u2705", "failed": "\u274c"}.get(status, "\u2753")
            last_trained = r.get("last_trained", "belum pernah")

            msg = (
                f"\U0001f504 <b>Training Status</b>\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
                f"{status_emoji} Status: <b>{status}</b>\n"
                f"\U0001f4c5 Terakhir Train: {last_trained}\n"
                f"\U0001f4ca Total Data: <b>{total:,}</b> samples\n"
            )

            # Class distribution from tracking data
            if tracking_data:
                from collections import Counter
                # Count by review status
                reviewed = sum(1 for t in tracking_data if t.get("review_status") in ("APPROVED", "CORRECTED"))
                trained = sum(1 for t in tracking_data if t.get("review_status") == "TRAINED")
                auto_approved = sum(1 for t in tracking_data if t.get("review_status") == "auto_approved")

                msg += (
                    f"\n\U0001f4cb <b>Data Breakdown:</b>\n"
                    f"\u251c \u2705 Reviewed (siap train): <b>{reviewed}</b>\n"
                    f"\u251c \U0001f393 Sudah Trained: <b>{trained}</b>\n"
                    f"\u2514 \U0001f916 Auto-approved: <b>{auto_approved}</b>\n"
                )

                # Class distribution
                symtomps_counter = Counter(t.get("symtomps", "unknown") for t in tracking_data if t.get("symtomps"))
                if symtomps_counter:
                    msg += f"\n\U0001f3f7 <b>Distribusi Kelas (Top 10):</b>\n"
                    # Get current model classes if available
                    current_classes = set()
                    try:
                        model_info = await self._api_get(f"{self._prediction_url}/model/info")
                        current_classes = set(model_info.get("classes", []))
                    except Exception:
                        pass

                    for cls, cnt in symtomps_counter.most_common(10):
                        new_badge = " \U0001f195" if current_classes and cls not in current_classes else ""
                        msg += f"  {cls}: <b>{cnt}</b>{new_badge}\n"

                    if current_classes:
                        new_classes = set(symtomps_counter.keys()) - current_classes
                        if new_classes:
                            msg += f"\n\U0001f195 <b>{len(new_classes)} kelas baru</b> terdeteksi!\n"

                # Readiness check
                if reviewed >= 100:
                    msg += f"\n\u2705 <b>READY</b> - Data cukup untuk training ({reviewed} reviewed)"
                elif reviewed > 0:
                    msg += f"\n\u26a0\ufe0f Butuh {100 - reviewed} reviewed lagi (min 100)"
                else:
                    msg += f"\n\U0001f534 Belum ada data reviewed"

            if status == "idle":
                msg += "\n\n\U0001f4a1 Gunakan /retrain untuk mulai training."
            elif status == "running":
                msg += "\n\n\u23f3 Training sedang berjalan..."
            elif status == "completed":
                last_result = r.get("last_result", {})
                if last_result:
                    msg += (
                        f"\n\n\U0001f3c6 <b>Hasil Terakhir:</b>\n"
                        f"\u251c \U0001f3af F1: {last_result.get('f1_score', 'N/A')}\n"
                        f"\u251c \U0001f4e6 Version: {last_result.get('model_version', '?')}\n"
                        f"\u2514 \U0001f4ca Samples: {last_result.get('n_samples', '?')}"
                    )
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("retrainstatus failed")
            await self._reply(update, f"\u274c Gagal mengambil status training.\n<code>{e}</code>")

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

    # =================== /updatestats ===================
    async def update_stats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger hourly stats calculation and write to Monitoring sheet."""
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        await self._reply(update, "\U0001f504 Menghitung statistik...")
        try:
            # Get current model version
            model_version = "unknown"
            try:
                mi = await self._api_get(f"{self._prediction_url}/model/info")
                model_version = mi.get("version", "unknown")
            except Exception:
                pass

            r = await self._api_post(
                f"{self._data_url}/stats/hourly",
                params={"model_version": model_version},
                timeout=30.0,
            )
            if r.get("success"):
                st = r.get("stats", {})
                total = st.get("total_predictions", 0)
                auto = st.get("auto_count", 0)
                manual = st.get("manual_count", 0)
                reviewed = st.get("reviewed_count", 0)
                pending = st.get("pending_count", 0)
                conf = st.get("avg_confidence", 0)
                auto_rate = (auto / total * 100) if total > 0 else 0

                await self._reply(
                    update,
                    f"\u2705 <b>Stats Updated!</b>\n"
                    f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
                    f"\U0001f4e6 Model: <code>{model_version}</code>\n"
                    f"\U0001f4ca Total Prediksi: <b>{total:,}</b>\n"
                    f"\U0001f916 Auto-classified: <b>{auto:,}</b> ({auto_rate:.1f}%)\n"
                    f"\u270d\ufe0f Manual: <b>{manual:,}</b>\n"
                    f"\u2705 Reviewed: <b>{reviewed:,}</b>\n"
                    f"\u23f3 Pending: <b>{pending:,}</b>\n"
                    f"\U0001f3af Avg Confidence: <b>{conf:.1%}</b>\n\n"
                    f"\U0001f4dd Ditulis ke sheet Monitoring.",
                )
            else:
                await self._reply(update, f"\u274c Gagal update stats: {r.get('detail', 'unknown')}")
        except Exception as e:
            _LOGGER.exception("updatestats failed")
            await self._reply(update, f"\u274c Gagal update stats.\n<code>{e}</code>")

    # =================== /mlflowstatus ===================
    async def mlflow_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Show MLflow model registry status with all versions."""
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        try:
            r = await self._api_get(f"{self._prediction_url}/mlflow/status")
            versions = r.get("versions", [])
            prod_ver = r.get("production_version")

            msg = (
                f"\U0001f4ca <b>MLflow Status</b>\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
                f"\U0001f3e2 Tracking: <code>{r.get('tracking_uri', '?')}</code>\n"
                f"\U0001f4e6 Model: <code>{r.get('model_name', '?')}</code>\n"
                f"\U0001f7e2 Production: <b>v{prod_ver}</b>\n\n" if prod_ver else
                f"\U0001f4ca <b>MLflow Status</b>\n"
                f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
                f"\U0001f3e2 Tracking: <code>{r.get('tracking_uri', '?')}</code>\n"
                f"\U0001f4e6 Model: <code>{r.get('model_name', '?')}</code>\n"
                f"\u26a0\ufe0f No Production version\n\n"
            )

            if versions:
                stage_emoji = {
                    "Production": "\U0001f7e2",
                    "Staging": "\U0001f7e1",
                    "Archived": "\u26aa",
                    "None": "\u26ab",
                }
                msg += "\U0001f4cb <b>Model Versions:</b>\n"
                for v in versions[:10]:
                    emoji = stage_emoji.get(v.get("stage", "None"), "\u26ab")
                    ver = v.get("version", "?")
                    stage = v.get("stage", "None")
                    created = v.get("created_at", "?")[:10] if v.get("created_at") else "?"
                    desc = v.get("description", "")
                    desc_text = f" - {desc[:30]}" if desc else ""
                    msg += f"  {emoji} v{ver} [{stage}] ({created}){desc_text}\n"

            msg += f"\n\U0001f4a1 /mlflowpromote [version] untuk promote ke Production"
            await self._reply(update, msg)
        except Exception as e:
            _LOGGER.exception("mlflowstatus failed")
            await self._reply(update, f"\u274c Gagal mengambil MLflow status.\n<code>{e}</code>")

    # =================== /mlflowpromote ===================
    async def mlflow_promote(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Promote a model version to Production stage."""
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        args = ctx.args or []
        if not args or not args[0].isdigit():
            await self._reply(
                update,
                "\u26a0\ufe0f <b>Usage:</b> <code>/mlflowpromote [version]</code>\n\n"
                "Contoh: <code>/mlflowpromote 3</code>\n\n"
                "\U0001f4a1 Gunakan /mlflowstatus untuk melihat daftar version.",
            )
            return

        version = args[0]
        stage = args[1].capitalize() if len(args) > 1 else "Production"

        await self._reply(update, f"\U0001f504 Promoting v{version} to {stage}...")
        try:
            r = await self._api_post(
                f"{self._prediction_url}/mlflow/promote",
                params={"version": version, "stage": stage},
            )
            if r.get("success"):
                await self._reply(
                    update,
                    f"\u2705 <b>Model Promoted!</b>\n\n"
                    f"\U0001f4e6 Version: <b>v{r.get('version', version)}</b>\n"
                    f"\U0001f3c6 Stage: <b>{r.get('stage', stage)}</b>\n\n"
                    f"\u27a1\ufe0f Gunakan /reloadmodel untuk load model baru.",
                )
            else:
                await self._reply(update, f"\u274c Promote gagal: {r.get('message', 'unknown')}")
        except Exception as e:
            _LOGGER.exception("mlflowpromote failed")
            await self._reply(update, f"\u274c Gagal promote model.\n<code>{e}</code>")

    # =================== /tiketreport ===================
    async def tiket_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /tiketreport [monthly|quarterly] [bulan/quarter] [tahun] [MIT|MIS]

        Contoh:
            /tiketreport → bulan ini (semua app)
            /tiketreport monthly 12 2025 → Desember 2025
            /tiketreport monthly 2 2026 MIT → Feb 2026 MyTech only
            /tiketreport quarterly → quarter ini
            /tiketreport quarterly 4 2025 MIS → Q4 2025 MyStaff only
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return

        args = ctx.args or []
        period = args[0].lower() if args else "monthly"
        if period not in ("monthly", "quarterly"):
            await self._reply(
                update,
                "📖 <b>Cara pakai:</b>\n"
                "  /tiketreport — Bulan ini (semua app)\n"
                "  /tiketreport monthly [bulan] [tahun] [MIT|MIS]\n"
                "  /tiketreport quarterly [quarter] [tahun] [MIT|MIS]\n\n"
                "<b>Filter App (opsional):</b>\n"
                "  MIT = MyTech | MIS = MyStaff\n\n"
                "<b>Contoh:</b>\n"
                "  /tiketreport monthly 12 2025 → Desember 2025\n"
                "  /tiketreport monthly 2 2026 MIT → Feb 2026 MyTech\n"
                "  /tiketreport quarterly 4 2025 MIS → Q4 2025 MyStaff",
            )
            return

        # Parse optional app filter (last arg could be MIT or MIS)
        app_filter: str | None = None
        _APP_NAMES = {"MIT": "MyTech", "MIS": "MyStaff"}
        # Check if last arg is an app code
        if len(args) >= 2 and args[-1].upper() in _APP_NAMES:
            app_filter = args[-1].upper()
            args = args[:-1]  # remove app filter from args for period parsing

        await self._reply(update, "⏳ Mengambil data dari Logs sheet...")

        try:
            from collections import Counter
            from datetime import timedelta

            # Fetch all logs via Data API
            data = await self._api_get(f"{self._data_url}/logs/all")
            all_rows = data.get("rows", [])
            if len(all_rows) <= 1:
                await self._reply(update, "❌ Tidak ada data di Logs sheet.")
                return

            headers = all_rows[0]
            # Normalize headers (strip whitespace)
            headers = [h.strip() for h in headers]

            # Find columns — flexible matching
            def _col(*names: str) -> int:
                for n in names:
                    nl = n.lower()
                    for i, h in enumerate(headers):
                        if h.lower() == nl:
                            return i
                return -1

            date_col = _col("Ticket Date", "Column 1")
            sla_time_col = _col("SLA Response Time")
            sla_status_col = _col("SLA Status")
            symtomps_col = _col("Symtomps")
            app_col = _col("App")
            solver_col = _col("Solver Name", "Solver")

            if date_col == -1:
                await self._reply(update, "❌ Kolom 'Ticket Date' tidak ditemukan.")
                return

            # Calculate date range
            today = datetime.now(TZ).date()
            nama_bulan = [
                "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                "Juli", "Agustus", "September", "Oktober", "November", "Desember",
            ]

            if period == "monthly":
                if len(args) >= 3:
                    try:
                        month, year = int(args[1]), int(args[2])
                        if not 1 <= month <= 12:
                            await self._reply(update, "❌ Bulan harus 1-12")
                            return
                        start_date = datetime(year, month, 1).date()
                        end_date = (
                            datetime(year + 1, 1, 1).date() - timedelta(days=1)
                            if month == 12
                            else datetime(year, month + 1, 1).date() - timedelta(days=1)
                        )
                    except ValueError:
                        await self._reply(update, "❌ Format: /tiketreport monthly [bulan] [tahun]")
                        return
                else:
                    month, year = today.month, today.year
                    start_date = today.replace(day=1)
                    end_date = today
                period_label = f"{nama_bulan[month]} {year}"
            else:  # quarterly
                if len(args) >= 3:
                    try:
                        quarter, year = int(args[1]), int(args[2])
                        if not 1 <= quarter <= 4:
                            await self._reply(update, "❌ Quarter harus 1-4")
                            return
                        start_month = (quarter - 1) * 3 + 1
                        start_date = datetime(year, start_month, 1).date()
                        end_month = start_month + 2
                        end_date = (
                            datetime(year + 1, 1, 1).date() - timedelta(days=1)
                            if end_month == 12
                            else datetime(year, end_month + 1, 1).date() - timedelta(days=1)
                        )
                    except ValueError:
                        await self._reply(update, "❌ Format: /tiketreport quarterly [quarter] [tahun]")
                        return
                else:
                    quarter = (today.month - 1) // 3 + 1
                    start_month = (quarter - 1) * 3 + 1
                    year = today.year
                    start_date = today.replace(month=start_month, day=1)
                    end_date = today
                q_months = [nama_bulan[start_month + i] for i in range(3)]
                period_label = f"Q{quarter} {year} ({', '.join(q_months)})"

            # Append app filter to label
            if app_filter:
                period_label += f" — {_APP_NAMES[app_filter]} ({app_filter})"

            # Filter and analyze
            total_tickets = 0
            sla_times: list[float] = []
            sla_met = 0
            sla_breach = 0
            symtomps_counter: Counter = Counter()
            app_counter: Counter = Counter()
            solver_counter: Counter = Counter()

            for row in all_rows[1:]:
                if len(row) <= date_col or not row[date_col]:
                    continue

                # Parse date
                date_str = row[date_col].split()[0]
                ticket_date = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                    try:
                        ticket_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if not ticket_date or ticket_date < start_date or ticket_date > end_date:
                    continue

                # App filter
                if app_filter and app_col != -1:
                    row_app = row[app_col].strip().upper() if len(row) > app_col else ""
                    if row_app != app_filter:
                        continue

                total_tickets += 1

                # SLA Time (handle comma decimal separator e.g. "0,57")
                if sla_time_col != -1 and len(row) > sla_time_col and row[sla_time_col]:
                    try:
                        t = float(row[sla_time_col].replace(",", "."))
                        if t > 0:
                            sla_times.append(t)
                    except ValueError:
                        pass

                # SLA Status
                if sla_status_col != -1 and len(row) > sla_status_col:
                    st = row[sla_status_col].strip().upper()
                    if st in ("MET", "OK", "WITHIN SLA"):
                        sla_met += 1
                    elif st in ("BREACH", "BREACHED", "OVER SLA", "LATE", "TERLAMBAT"):
                        sla_breach += 1

                # Symtomps
                if symtomps_col != -1 and len(row) > symtomps_col and row[symtomps_col].strip():
                    symtomps_counter[row[symtomps_col].strip()] += 1

                # App
                if app_col != -1 and len(row) > app_col and row[app_col].strip():
                    app_counter[row[app_col].strip().upper()] += 1

                # Solver
                if solver_col != -1 and len(row) > solver_col and row[solver_col].strip():
                    solver_counter[row[solver_col].strip()] += 1

            if total_tickets == 0:
                await self._reply(
                    update,
                    f"📊 <b>Laporan Tiket — {period_label}</b>\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                )
                return

            # Calculations
            avg_sla = sum(sla_times) / len(sla_times) if sla_times else 0
            min_sla = min(sla_times) if sla_times else 0
            max_sla = max(sla_times) if sla_times else 0
            sla_total = sla_met + sla_breach
            sla_pct = (sla_met / sla_total * 100) if sla_total > 0 else 0
            top_sym = symtomps_counter.most_common(10)

            # Build message
            msg = (
                f"📊 <b>Laporan Tiket — {period_label}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 {datetime.now(TZ).strftime('%d %b %Y %H:%M WIB')}\n\n"
                f"<b>📋 Ringkasan</b>\n"
                f"  Total Tiket: <b>{total_tickets:,}</b>\n"
            )

            # App breakdown
            if app_counter:
                parts = [f"{k}: {v:,}" for k, v in app_counter.most_common()]
                msg += f"  Per Aplikasi: {' | '.join(parts)}\n"

            msg += (
                f"\n<b>⏱️ SLA Response Time</b>\n"
                f"  Rata-rata: <b>{avg_sla:.1f} menit</b>\n"
                f"  Tercepat: {min_sla:.1f} menit\n"
                f"  Terlama: {max_sla:.1f} menit\n\n"
                f"<b>📈 SLA Compliance</b>\n"
                f"  ✅ Tercapai: {sla_met:,}\n"
                f"  ❌ Terlambat: {sla_breach:,}\n"
                f"  📊 Rate: <b>{sla_pct:.1f}%</b>\n"
            )

            # Solver breakdown
            if solver_counter:
                msg += "\n<b>👤 Per Solver</b>\n"
                for name, count in solver_counter.most_common():
                    pct = count / total_tickets * 100
                    msg += f"  {name}: {count:,} ({pct:.0f}%)\n"

            # Top symtomps
            if top_sym:
                msg += "\n<b>🏷️ Top 10 Symtomps</b>\n"
                for i, (sym, count) in enumerate(top_sym, 1):
                    pct = count / total_tickets * 100
                    msg += f"  {i}. {sym}: {count:,} ({pct:.1f}%)\n"

            await self._reply(update, msg)

        except Exception as e:
            _LOGGER.exception("tiketreport failed")
            await self._reply(update, f"❌ Gagal membuat laporan tiket.\n<code>{e}</code>")

    # =================== /trendbulan ===================
    async def trend_bulan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /trendbulan [MIT|MIS] [bulan] [tahun]
        Top 10 Symtomps per aplikasi dalam bulan tertentu.
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return

        args = ctx.args or []
        _APP_NAMES = {"MIT": "MyTech", "MIS": "MyStaff"}

        if not args:
            await self._reply(
                update,
                "📖 <b>Cara pakai:</b>\n"
                "  <code>/trendbulan MIT</code> → MyTech bulan ini\n"
                "  <code>/trendbulan MIS</code> → MyStaff bulan ini\n"
                "  <code>/trendbulan MIT 12 2025</code> → MyTech Des 2025\n\n"
                "<b>Keterangan:</b> MIT = MyTech | MIS = MyStaff",
            )
            return

        app_type = args[0].upper()
        if app_type not in _APP_NAMES:
            await self._reply(update, f"❌ Aplikasi '{args[0]}' tidak valid.\nGunakan <b>MIT</b> (MyTech) atau <b>MIS</b> (MyStaff)")
            return

        app_name = _APP_NAMES[app_type]
        today = datetime.now(TZ).date()

        try:
            bulan = int(args[1]) if len(args) > 1 else today.month
            tahun = int(args[2]) if len(args) > 2 else today.year
            if not 1 <= bulan <= 12:
                await self._reply(update, "❌ Bulan harus 1-12")
                return
        except ValueError:
            await self._reply(update, "❌ Format bulan/tahun tidak valid. Gunakan angka.")
            return

        nama_bulan = [
            "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember",
        ]

        await self._reply(update, f"⏳ Mengambil data {app_name} bulan {bulan}/{tahun}...")

        try:
            from collections import Counter
            from datetime import timedelta
            import calendar

            data = await self._api_get(f"{self._data_url}/logs/all")
            all_rows = data.get("rows", [])
            if len(all_rows) <= 1:
                await self._reply(update, "❌ Tidak ada data di Logs sheet.")
                return

            headers = [h.strip() for h in all_rows[0]]

            def _col(*names: str) -> int:
                for n in names:
                    nl = n.lower()
                    for i, h in enumerate(headers):
                        if h.lower() == nl:
                            return i
                return -1

            date_col = _col("Ticket Date", "Column 1")
            app_col = _col("App")
            symtomps_col = _col("Symtomps")

            last_day = calendar.monthrange(tahun, bulan)[1]
            start_date = datetime(tahun, bulan, 1).date()
            end_date = datetime(tahun, bulan, last_day).date()

            total_tickets = 0
            symtomps_counter: Counter = Counter()

            for row in all_rows[1:]:
                if len(row) <= date_col or not row[date_col]:
                    continue

                # App filter
                if app_col != -1:
                    row_app = row[app_col].strip().upper() if len(row) > app_col else ""
                    if row_app != app_type:
                        continue

                # Parse date
                date_str = row[date_col].split()[0]
                ticket_date = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                    try:
                        ticket_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if not ticket_date or ticket_date < start_date or ticket_date > end_date:
                    continue

                total_tickets += 1
                if symtomps_col != -1 and len(row) > symtomps_col and row[symtomps_col].strip():
                    symtomps_counter[row[symtomps_col].strip()] += 1

            if total_tickets == 0:
                await self._reply(
                    update,
                    f"📊 <b>Trend Tiket {app_name} ({app_type})</b>\n"
                    f"📅 Periode: {nama_bulan[bulan]} {tahun}\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                )
                return

            top_sym = symtomps_counter.most_common(10)
            msg = (
                f"📊 <b>Trend Tiket {app_name} ({app_type})</b>\n"
                f"📅 Periode: {nama_bulan[bulan]} {tahun}\n"
                f"📋 Total: <b>{total_tickets:,}</b> tiket\n\n"
                f"<b>Top 10 Symtomps:</b>\n"
            )
            for i, (sym, count) in enumerate(top_sym, 1):
                pct = count / total_tickets * 100
                msg += f"  {i}. {sym} — {count} ({pct:.1f}%)\n"

            if len(symtomps_counter) > 10:
                others = sum(c for s, c in symtomps_counter.items() if s not in dict(top_sym))
                msg += f"\n<i>+{len(symtomps_counter) - 10} symtomps lainnya ({others} tiket)</i>"

            await self._reply(update, msg)

        except Exception as e:
            _LOGGER.exception("trendbulan failed")
            await self._reply(update, f"❌ Gagal mengambil trend bulanan.\n<code>{e}</code>")

    # =================== /trendmingguan ===================
    async def trend_mingguan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /trendmingguan [minggu] [bulan] [tahun] [MIT|MIS]
        Top 5 Symtomps per minggu dengan contoh tiket.

        Pembagian minggu: 1=1-7, 2=8-14, 3=15-21, 4=22-28, 5=29-31
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return

        args = ctx.args or []
        today = datetime.now(TZ).date()
        _APP_NAMES = {"MIT": "MyTech", "MIS": "MyStaff"}

        current_week = min((today.day - 1) // 7 + 1, 5)

        try:
            minggu = int(args[0]) if len(args) > 0 else current_week
            bulan = int(args[1]) if len(args) > 1 else today.month
            tahun = int(args[2]) if len(args) > 2 else today.year
            app_type = args[3].upper() if len(args) > 3 else None

            if not 1 <= minggu <= 5:
                await self._reply(
                    update,
                    "❌ Minggu harus 1-5\n\n"
                    "• Minggu 1: tanggal 1-7\n• Minggu 2: tanggal 8-14\n"
                    "• Minggu 3: tanggal 15-21\n• Minggu 4: tanggal 22-28\n"
                    "• Minggu 5: tanggal 29-31",
                )
                return
            if not 1 <= bulan <= 12:
                await self._reply(update, "❌ Bulan harus 1-12")
                return
            if app_type and app_type not in _APP_NAMES:
                await self._reply(
                    update,
                    f"❌ Aplikasi '{args[3]}' tidak valid.\n"
                    "Gunakan <b>MIT</b> (MyTech) atau <b>MIS</b> (MyStaff), atau kosongkan.",
                )
                return
        except ValueError:
            await self._reply(
                update,
                "📖 <b>Cara pakai:</b>\n"
                "  <code>/trendmingguan</code> → Minggu ini\n"
                "  <code>/trendmingguan 2</code> → Minggu ke-2 bulan ini\n"
                "  <code>/trendmingguan 1 12 2025 MIT</code> → Minggu 1 Des 2025 MyTech\n\n"
                "<b>Minggu:</b> 1=tgl 1-7 | 2=8-14 | 3=15-21 | 4=22-28 | 5=29-31",
            )
            return

        app_label = f" {_APP_NAMES[app_type]}" if app_type else ""
        nama_bulan = [
            "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember",
        ]

        await self._reply(update, f"⏳ Mengambil data minggu ke-{minggu}{app_label} bulan {bulan}/{tahun}...")

        try:
            from collections import Counter, defaultdict
            import calendar

            data = await self._api_get(f"{self._data_url}/logs/all")
            all_rows = data.get("rows", [])
            if len(all_rows) <= 1:
                await self._reply(update, "❌ Tidak ada data di Logs sheet.")
                return

            headers = [h.strip() for h in all_rows[0]]

            def _col(*names: str) -> int:
                for n in names:
                    nl = n.lower()
                    for i, h in enumerate(headers):
                        if h.lower() == nl:
                            return i
                return -1

            date_col = _col("Ticket Date", "Column 1")
            app_col = _col("App")
            symtomps_col = _col("Symtomps")
            raw_text_col = _col("tech raw text")

            last_day_of_month = calendar.monthrange(tahun, bulan)[1]
            week_ranges = {1: (1, 7), 2: (8, 14), 3: (15, 21), 4: (22, 28), 5: (29, last_day_of_month)}
            start_day, end_day = week_ranges[minggu]
            end_day = min(end_day, last_day_of_month)

            if start_day > last_day_of_month:
                await self._reply(
                    update,
                    f"❌ Bulan {bulan}/{tahun} tidak memiliki minggu ke-{minggu}\n"
                    f"(Bulan ini hanya sampai tanggal {last_day_of_month})",
                )
                return

            start_date = datetime(tahun, bulan, start_day).date()
            end_date = datetime(tahun, bulan, end_day).date()

            total_tickets = 0
            symtomps_counter: Counter = Counter()
            ticket_examples: dict[str, list[str]] = defaultdict(list)

            for row in all_rows[1:]:
                if len(row) <= date_col or not row[date_col]:
                    continue

                # App filter
                if app_type and app_col != -1:
                    row_app = row[app_col].strip().upper() if len(row) > app_col else ""
                    if row_app != app_type:
                        continue

                # Parse date
                date_str = row[date_col].split()[0]
                ticket_date = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                    try:
                        ticket_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if not ticket_date or ticket_date < start_date or ticket_date > end_date:
                    continue

                total_tickets += 1

                symtomps = row[symtomps_col].strip() if symtomps_col != -1 and len(row) > symtomps_col else ""
                raw_text = row[raw_text_col].strip() if raw_text_col != -1 and len(row) > raw_text_col else ""

                if symtomps:
                    symtomps_counter[symtomps] += 1
                    if len(ticket_examples[symtomps]) < 3:
                        ticket_name = (raw_text[:50] + "...") if len(raw_text) > 50 else raw_text
                        if ticket_name:
                            ticket_examples[symtomps].append(ticket_name)

            if total_tickets == 0:
                await self._reply(
                    update,
                    f"📅 <b>Trend Mingguan{app_label}</b>\n"
                    f"Minggu ke-{minggu} ({start_day}-{end_day} {nama_bulan[bulan]} {tahun})\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                )
                return

            top_sym = symtomps_counter.most_common(5)

            msg = (
                f"📅 <b>Trend Mingguan{app_label}</b>\n"
                f"Minggu ke-{minggu} ({start_day}-{end_day} {nama_bulan[bulan]} {tahun})\n"
                f"📋 Total: <b>{total_tickets:,}</b> tiket\n\n"
                f"<b>Top 5 Symtomps:</b>\n"
            )

            for i, (sym, count) in enumerate(top_sym, 1):
                pct = count / total_tickets * 100
                msg += f"\n  {i}. <b>{sym}</b> — {count} tiket ({pct:.1f}%)\n"
                for ex in ticket_examples.get(sym, [])[:3]:
                    msg += f"     └ <i>{ex}</i>\n"

            await self._reply(update, msg)

        except Exception as e:
            _LOGGER.exception("trendmingguan failed")
            await self._reply(update, f"❌ Gagal mengambil trend mingguan.\n<code>{e}</code>")


# ============================================================
# TrendAlertService — Background auto-alert for anomalies
# ============================================================

class TrendAlertService:
    """
    Background service that periodically checks stats and sends
    alerts to admin chat when thresholds are breached.
    """

    # Thresholds
    AUTO_RATE_MIN = 0.70        # Alert if automation rate drops below 70%
    MANUAL_RATE_MAX = 0.25      # Alert if manual rate exceeds 25%
    CONFIDENCE_MIN = 0.80       # Alert if avg confidence drops below 80%
    PENDING_MAX = 50            # Alert if pending queue exceeds 50

    def __init__(
        self,
        data_url: str,
        prediction_url: str,
        bot,                    # telegram.Bot instance
        admin_chat_id: int,
        interval_seconds: int = 3600,  # default: check every hour
    ):
        self._data_url = data_url
        self._prediction_url = prediction_url
        self._bot = bot
        self._chat_id = admin_chat_id
        self._interval = interval_seconds
        self._task: asyncio.Task | None = None
        self._client = httpx.AsyncClient(timeout=15.0)

    async def start(self) -> None:
        """Start the background alert loop."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            _LOGGER.info("TrendAlertService started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        """Stop the background alert loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._client.aclose()
        _LOGGER.info("TrendAlertService stopped")

    async def _loop(self) -> None:
        """Main loop that periodically checks and alerts."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                alert = await self.check_and_alert()
                if alert:
                    await self._bot.send_message(
                        chat_id=self._chat_id,
                        text=alert,
                        parse_mode="HTML",
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                _LOGGER.exception("TrendAlertService error: %s", e)
                await asyncio.sleep(60)  # Back off on error

    async def check_and_alert(self) -> str | None:
        """
        Evaluate current stats against thresholds.
        Returns alert message if any threshold is breached, else None.
        """
        try:
            r = await self._client.get(f"{self._data_url}/stats/realtime")
            r.raise_for_status()
            stats = r.json()
        except Exception:
            return None

        total = stats.get("total_predictions", 0)
        if total == 0:
            return None

        auto = stats.get("auto_count", 0)
        manual = stats.get("manual_count", 0)
        pending = stats.get("pending_count", 0)
        confidence = stats.get("avg_confidence", 0)

        auto_rate = auto / total
        manual_rate = manual / total

        alerts: list[str] = []

        if auto_rate < self.AUTO_RATE_MIN:
            alerts.append(
                f"\U0001f534 <b>Automation Rate Turun!</b>\n"
                f"  Rate: {auto_rate:.1%} (threshold: {self.AUTO_RATE_MIN:.0%})"
            )

        if manual_rate > self.MANUAL_RATE_MAX:
            alerts.append(
                f"\U0001f7e0 <b>Manual Rate Tinggi!</b>\n"
                f"  Rate: {manual_rate:.1%} (threshold: {self.MANUAL_RATE_MAX:.0%})"
            )

        if confidence < self.CONFIDENCE_MIN:
            alerts.append(
                f"\u26a0\ufe0f <b>Confidence Rendah!</b>\n"
                f"  Avg: {confidence:.1%} (threshold: {self.CONFIDENCE_MIN:.0%})"
            )

        if pending > self.PENDING_MAX:
            alerts.append(
                f"\U0001f4e5 <b>Pending Queue Penuh!</b>\n"
                f"  Count: {pending} (threshold: {self.PENDING_MAX})"
            )

        if not alerts:
            return None

        header = (
            f"\U0001f6a8 <b>TREND ALERT</b>\n"
            f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n\n"
        )
        body = "\n\n".join(alerts)
        footer = (
            f"\n\n\U0001f4ca Total: {total:,} | Auto: {auto:,} | Manual: {manual:,}\n"
            f"\U0001f4a1 Gunakan /report untuk detail lengkap."
        )

        return header + body + footer
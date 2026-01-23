"""
Admin Command Handlers
=======================
Telegram admin commands for ML monitoring and management.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

if TYPE_CHECKING:
    from src.core.config import Config
    from src.ml.classifier import MLClassifier
    from src.ml.tracking import MLTrackingClient
    from src.services.sheets import GoogleSheetsClient

_LOGGER = logging.getLogger(__name__)

# Timezone
TZ = ZoneInfo("Asia/Jakarta")


class AdminCommandHandler:
    """Handler for admin commands related to ML monitoring."""
    
    def __init__(
        self,
        config: "Config",
        ml_classifier: Optional["MLClassifier"] = None,
        ml_tracking: Optional["MLTrackingClient"] = None,
        admin_chat_ids: Optional[list[int]] = None,
    ):
        self._config = config
        self._ml_classifier = ml_classifier
        self._ml_tracking = ml_tracking
        self._admin_chat_ids = admin_chat_ids or []
        
    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin."""
        if not self._admin_chat_ids:
            return True  # If no admin list, allow all
        return user_id in self._admin_chat_ids
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /stats - Show today's ML prediction statistics
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_tracking:
            await update.message.reply_text("❌ ML Tracking not initialized")
            return
            
        try:
            # Calculate stats real-time from ML_Tracking instead of cached Monitoring sheet
            model_version = self._ml_classifier.model_version if self._ml_classifier else "unknown"
            stats = self._ml_tracking.get_realtime_stats()
            pending = self._ml_tracking.get_pending_review_count()
            
            if not stats or stats.get("total_predictions", 0) == 0:
                await update.message.reply_text(
                    "📊 **Today's ML Stats**\n\n"
                    "No predictions recorded yet.\n\n"
                    "Stats are calculated from ML_Tracking sheet.",
                    parse_mode="Markdown"
                )
                return
            
            total = stats.get("total_predictions", 0)
            auto_count = stats.get("auto_count", 0)
            high_count = stats.get("high_count", 0)
            medium_count = stats.get("medium_count", 0)
            manual_count = stats.get("manual_count", 0)
            avg_conf = stats.get("avg_confidence", 0)
            reviewed = stats.get("reviewed_count", 0)
            
            # Calculate percentages
            auto_pct = (auto_count / total * 100) if total > 0 else 0
            
            message = (
                f"📊 **ML Stats (All Time)**\n\n"
                f"📈 **Total Predictions:** {total}\n"
                f"🎯 **Avg Confidence:** {avg_conf:.1f}%\n\n"
                f"**Distribution:**\n"
                f"  ✅ AUTO (≥80%): {auto_count} ({auto_pct:.1f}%)\n"
                f"  🔶 HIGH (70-80%): {high_count}\n"
                f"  🟡 MEDIUM (50-70%): {medium_count}\n"
                f"  🔴 MANUAL (<50%): {manual_count}\n\n"
                f"**Review Status:**\n"
                f"  📋 Pending Review: {pending.get('total_pending', 0)}\n"
                f"  ✅ Reviewed: {reviewed}\n"
            )
            
            if self._ml_classifier:
                message += f"\n🤖 Model: {self._ml_classifier.model_version}"
                
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error getting stats: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /report [weekly|monthly] - Generate ML performance report
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_tracking:
            await update.message.reply_text("❌ ML Tracking not initialized")
            return
        
        # Parse period argument
        args = context.args or []
        period = args[0].lower() if args else "weekly"
        
        if period not in ["weekly", "monthly"]:
            await update.message.reply_text(
                "Usage: /report [weekly|monthly]\n"
                "Example: /report weekly"
            )
            return
            
        try:
            if period == "weekly":
                stats = self._ml_tracking.get_weekly_stats()
                period_label = "7 Hari Terakhir"
            else:
                stats = self._ml_tracking.get_monthly_stats()
                period_label = "30 Hari Terakhir"
            
            if not stats or stats.get("total_predictions", 0) == 0:
                await update.message.reply_text(
                    f"📊 **{period.title()} Report**\n\n"
                    f"No data available for {period_label}.",
                    parse_mode="Markdown"
                )
                return
            
            total = stats.get("total_predictions", 0)
            avg_conf = stats.get("avg_confidence", 0)
            auto_count = stats.get("auto_count", 0)
            high_count = stats.get("high_review_count", 0)
            medium_count = stats.get("medium_review_count", 0)
            manual_count = stats.get("manual_count", 0)
            reviewed = stats.get("reviewed_count", 0)
            accuracy = stats.get("accuracy", 0)
            
            auto_pct = (auto_count / total * 100) if total > 0 else 0
            review_rate = ((high_count + medium_count + manual_count) / total * 100) if total > 0 else 0
            
            message = (
                f"📊 **{period.title()} Report - {period_label}**\n"
                f"Generated: {datetime.now(TZ).strftime('%d %b %Y %H:%M')}\n\n"
                f"**Summary:**\n"
                f"  📈 Total Predictions: {total}\n"
                f"  🎯 Avg Confidence: {avg_conf:.1f}%\n"
                f"  ⚡ Automation Rate: {auto_pct:.1f}%\n"
                f"  📋 Review Rate: {review_rate:.1f}%\n\n"
                f"**Distribution:**\n"
                f"  ✅ AUTO: {auto_count}\n"
                f"  🔶 HIGH REVIEW: {high_count}\n"
                f"  🟡 MEDIUM REVIEW: {medium_count}\n"
                f"  🔴 MANUAL: {manual_count}\n\n"
                f"**Quality Metrics:**\n"
                f"  📝 Total Reviewed: {reviewed}\n"
                f"  ✅ Accuracy (reviewed): {accuracy:.1f}%\n"
            )
            
            if self._ml_classifier:
                message += f"\n🤖 Model: {self._ml_classifier.model_version} ({self._ml_classifier.num_classes} classes)"
            
            # Add trend analysis
            trend_msg = self._analyze_trend(stats)
            if trend_msg:
                message += f"\n\n**📈 Trend Analysis:**\n{trend_msg}"
                
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error generating report: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    def _analyze_trend(self, stats: dict) -> str:
        """Analyze trends and generate insights."""
        insights = []
        
        total = stats.get("total_predictions", 0)
        auto_count = stats.get("auto_count", 0)
        avg_conf = stats.get("avg_confidence", 0)
        manual_count = stats.get("manual_count", 0)
        
        if total == 0:
            return ""
        
        auto_rate = (auto_count / total * 100)
        manual_rate = (manual_count / total * 100)
        
        # Check automation rate
        if auto_rate >= 90:
            insights.append("✅ Excellent automation rate (≥90%)")
        elif auto_rate >= 80:
            insights.append("👍 Good automation rate (80-90%)")
        elif auto_rate < 70:
            insights.append("⚠️ Low automation rate (<70%), consider retraining")
        
        # Check confidence
        if avg_conf >= 90:
            insights.append("✅ High average confidence")
        elif avg_conf < 80:
            insights.append("⚠️ Average confidence dropping, monitor closely")
        
        # Check manual rate
        if manual_rate > 20:
            insights.append("🔴 High manual classification rate (>20%)")
        
        return "\n".join(insights) if insights else "No significant trends detected."
    
    async def model_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /modelstatus - Show current model information
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_classifier:
            await update.message.reply_text("❌ ML Classifier not initialized")
            return
            
        try:
            metadata = self._ml_classifier.get_metadata()
            
            message = (
                f"🤖 **Model Status**\n\n"
                f"**Version:** {metadata.get('version', 'unknown')}\n"
                f"**Classes:** {metadata.get('num_classes', 0)}\n"
                f"**Training Samples:** {metadata.get('training_samples', 'N/A')}\n"
                f"**Training Accuracy:** {metadata.get('training_accuracy', 0):.2f}%\n"
                f"**Trained At:** {metadata.get('trained_at', 'N/A')}\n\n"
                f"**Thresholds:**\n"
                f"  AUTO: ≥{metadata.get('threshold_auto', 0.90)*100:.0f}%\n"
                f"  HIGH REVIEW: {metadata.get('threshold_high', 0.85)*100:.0f}%-{metadata.get('threshold_auto', 0.90)*100:.0f}%\n"
                f"  MEDIUM REVIEW: {metadata.get('threshold_medium', 0.70)*100:.0f}%-{metadata.get('threshold_high', 0.85)*100:.0f}%\n"
                f"  MANUAL: <{metadata.get('threshold_medium', 0.70)*100:.0f}%\n"
            )
            
            # Add class list (top 10)
            classes = metadata.get("classes", [])
            if classes:
                message += f"\n**Sample Classes ({len(classes)} total):**\n"
                message += ", ".join(classes[:5]) + "..."
                
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error getting model status: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def pending_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /pendingreview - Show items pending manual review
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_tracking:
            await update.message.reply_text("❌ ML Tracking not initialized")
            return
            
        try:
            pending = self._ml_tracking.get_pending_review_count()
            
            total_pending = pending.get("total_pending", 0)
            manual = pending.get("manual_pending", 0)
            high = pending.get("high_review_pending", 0)
            medium = pending.get("medium_review_pending", 0)
            
            if total_pending == 0:
                await update.message.reply_text(
                    "📋 **Pending Review**\n\n"
                    "✅ No items pending review!",
                    parse_mode="Markdown"
                )
                return
            
            message = (
                f"📋 **Pending Review**\n\n"
                f"**Total Pending:** {total_pending}\n\n"
                f"By Priority:\n"
                f"  🔴 MANUAL (Critical): {manual}\n"
                f"  🔶 HIGH REVIEW: {high}\n"
                f"  🟡 MEDIUM REVIEW: {medium}\n\n"
                f"Use Google Sheets to review and correct predictions."
            )
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error getting pending review: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def help_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /helpml - Show available ML admin commands
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        message = (
            "🤖 **Daftar Command Admin ML**\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "📊 **MONITORING**\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "**/stats**\n"
            "└ Lihat statistik prediksi hari ini\n"
            "└ (total, accuracy, automation rate)\n\n"
            
            "**/report**\n"
            "└ Laporan performa model\n"
            "└ Contoh: `/report weekly` atau `/report monthly`\n\n"
            
            "**/modelstatus**\n"
            "└ Info model yang aktif\n"
            "└ (versi, jumlah class, threshold)\n\n"
            
            "**/pendingreview**\n"
            "└ Lihat tiket yang butuh review manual\n\n"
            
            "**/updatestats**\n"
            "└ Update statistik ke Monitoring sheet\n"
            "└ (otomatis jalan tiap jam)\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔄 **RETRAINING**\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "**/retrainstatus**\n"
            "└ Cek apakah data cukup untuk retrain\n"
            "└ (minimal 100 data reviewed)\n\n"
            
            "**/retrain**\n"
            "└ Retrain model jika threshold tercapai\n"
            "└ Model langsung aktif tanpa restart!\n\n"
            
            "**/retrain force**\n"
            "└ Paksa retrain tanpa cek threshold\n\n"
            
            "**/reloadmodel**\n"
            "└ Load ulang model tanpa restart\n"
            "└ Contoh: `/reloadmodel v2` untuk versi spesifik\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "**/help** atau **/helpml**\n"
            "└ Tampilkan pesan bantuan ini"
        )
        
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def retrain_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /retrainstatus - Check if there's enough data for retraining
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_tracking:
            await update.message.reply_text("❌ ML Tracking not initialized")
            return
        
        try:
            # Get reviewed count (APPROVED + CORRECTED) from ML_Tracking
            reviewed_count = self._ml_tracking.get_reviewed_count()
            trained_count = self._ml_tracking.get_trained_count()
            class_dist = self._ml_tracking.get_reviewed_class_distribution()
            
            min_samples = 100  # Minimum for retraining
            ready = reviewed_count >= min_samples
            
            if self._ml_classifier:
                current_version = self._ml_classifier.model_version
                current_samples = self._ml_classifier.get_metadata().get("training_samples", "N/A")
                current_classes = self._ml_classifier.get_metadata().get("num_classes", 0)
            else:
                current_version = "N/A"
                current_samples = "N/A"
                current_classes = 0
            
            status_emoji = "✅" if ready else "⏳"
            
            message = (
                f"🔄 **Retrain Status**\n\n"
                f"**Current Model:** {current_version}\n"
                f"**Training Samples:** {current_samples}\n"
                f"**Current Classes:** {current_classes}\n\n"
                f"**New Data Available:**\n"
                f"  📝 Ready for Training: {reviewed_count}\n"
                f"  ✅ Already Trained: {trained_count}\n\n"
            )
            
            # Add class distribution info
            if class_dist:
                # Sort by count descending
                sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
                
                # Check for new classes (not in current model)
                new_classes = []
                if self._ml_classifier and hasattr(self._ml_classifier, '_label_encoder'):
                    le = self._ml_classifier._label_encoder
                    if le is not None:
                        if isinstance(le, dict):
                            existing = set(le.values())
                        else:
                            existing = set(le.classes_)
                        new_classes = [(c, n) for c, n in sorted_classes if c not in existing]
                
                message += "**Class Distribution (New Data):**\n"
                for cls, count in sorted_classes[:10]:  # Top 10
                    is_new = " 🆕" if any(c == cls for c, _ in new_classes) else ""
                    message += f"  • {cls}: {count}{is_new}\n"
                
                if len(sorted_classes) > 10:
                    message += f"  ... +{len(sorted_classes) - 10} more classes\n"
                
                if new_classes:
                    message += f"\n🆕 **New Classes:** {len(new_classes)}\n"
                    for cls, count in new_classes:
                        message += f"  • {cls}: {count} samples\n"
                
                message += "\n"
            
            message += f"**Status:** {status_emoji} {'Ready for retrain!' if ready else f'Need {min_samples - reviewed_count} more samples'}\n\n"
            
            if ready:
                message += (
                    "💡 Use `/retrain` to start training\n"
                    "Or `/retrain force` to force retrain"
                )
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error checking retrain status: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def reload_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /reloadmodel [version] - Hot reload ML model without restarting bot
        
        Example:
            /reloadmodel       - Reload current version
            /reloadmodel v3    - Load version v3
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
            
        if not self._ml_classifier:
            await update.message.reply_text("❌ ML Classifier not initialized")
            return
        
        # Parse version argument
        args = context.args or []
        new_version = args[0] if args else None
        
        old_version = self._ml_classifier.model_version
        target_version = new_version or old_version
        
        await update.message.reply_text(
            f"🔄 Reloading model...\n"
            f"Current: {old_version}\n"
            f"Target: {target_version}"
        )
        
        try:
            success = self._ml_classifier.reload(new_version)
            
            if success:
                new_info = self._ml_classifier.get_metadata()
                await update.message.reply_text(
                    f"✅ **Model Reloaded Successfully!**\n\n"
                    f"**Version:** {new_info.get('version', 'unknown')}\n"
                    f"**Classes:** {new_info.get('num_classes', 0)}\n"
                    f"**Previous:** {old_version}\n\n"
                    f"Bot is now using the new model. No restart needed! 🎉",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ **Model Reload Failed!**\n\n"
                    f"Could not load model version: {target_version}\n"
                    f"Check if model files exist in models/ folder.\n\n"
                    f"Bot is still using: {old_version}",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            _LOGGER.error("Error reloading model: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def retrain(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /retrain [--force] - Retrain model dan auto-reload
        
        Example:
            /retrain        - Retrain jika threshold tercapai (reviewed >= 100)
            /retrain force  - Force retrain tanpa check threshold
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        
        # Parse args
        args = context.args or []
        force = "force" in [a.lower() for a in args]
        
        await update.message.reply_text(
            "🔄 **Starting Retrain Process...**\n\n"
            f"Mode: {'Force' if force else 'Check threshold (≥100 reviewed)'}\n"
            "This may take 10-30 minutes...",
            parse_mode="Markdown"
        )
        
        try:
            import asyncio
            import subprocess
            import sys
            from pathlib import Path
            
            # Get script path
            script_path = Path(__file__).parent.parent.parent / "scripts" / "retrain.py"
            
            if not script_path.exists():
                await update.message.reply_text(f"❌ Retrain script not found: {script_path}")
                return
            
            # Build command
            cmd = [sys.executable, "-u", str(script_path)]  # -u for unbuffered output
            if force:
                cmd.append("--force")
            else:
                cmd.extend(["--check-threshold", "100"])
            
            _LOGGER.info("Starting retrain subprocess: %s", " ".join(cmd))
            
            # Send initial progress message that we'll edit
            progress_msg = await update.message.reply_text(
                "🔄 **Training Progress**\n\n"
                "⏳ Starting...",
                parse_mode="Markdown"
            )
            
            # Run with Popen for real-time output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            
            output_lines = []
            start_time = asyncio.get_event_loop().time()
            last_update = start_time
            update_interval = 5  # Update Telegram every 5 seconds
            current_status = "Starting..."
            
            async def update_progress():
                """Background task to update progress with elapsed time."""
                nonlocal last_update
                while process.returncode is None:
                    await asyncio.sleep(5)
                    now = asyncio.get_event_loop().time()
                    elapsed = int(now - start_time)
                    mins, secs = divmod(elapsed, 60)
                    try:
                        await progress_msg.edit_text(
                            f"🔄 **Training Progress**\n\n"
                            f"{current_status}\n\n"
                            f"⏱ Elapsed: {mins}m {secs}s",
                            parse_mode="Markdown"
                        )
                    except Exception:
                        pass
            
            # Start progress updater
            progress_task = asyncio.create_task(update_progress())
            
            async def read_output():
                nonlocal current_status
                
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    
                    line_text = line.decode('utf-8', errors='ignore').strip()
                    if line_text:
                        output_lines.append(line_text)
                        _LOGGER.info("[retrain] %s", line_text)
                        
                        # Update current status based on keywords
                        if "Loading training data" in line_text:
                            current_status = "📥 Loading data..."
                        elif "From Logs" in line_text:
                            current_status = f"📥 {line_text}"
                        elif "From ML_Tracking" in line_text:
                            current_status = f"📥 {line_text}"
                        elif "Deduplicated" in line_text:
                            current_status = f"🔄 {line_text}"
                        elif "Total training data" in line_text:
                            current_status = f"📊 {line_text}"
                        elif "Preprocessing" in line_text:
                            current_status = "⚙️ Preprocessing..."
                        elif "TF-IDF features" in line_text:
                            current_status = f"⚙️ {line_text}"
                        elif "Training LightGBM" in line_text:
                            current_status = "🧠 Training model (5-fold CV)..."
                        elif "CV Macro F1" in line_text:
                            current_status = f"📈 {line_text}"
                        elif "Calibrating" in line_text:
                            current_status = "🎯 Calibrating probabilities..."
                        elif "Saving" in line_text or "Model saved" in line_text:
                            current_status = "💾 Saving model..."
                        elif "SUCCESS" in line_text or "Complete" in line_text:
                            current_status = "✅ Complete!"
            
            # Wait for process to complete (no timeout)
            await read_output()
            await process.wait()
            progress_task.cancel()
            
            # Get full output
            output = "\n".join(output_lines)
            
            _LOGGER.info("Retrain subprocess finished with code: %d", process.returncode)
            
            # Check result
            if process.returncode == 0:
                
                # Check if actually retrained or skipped
                if "skipping retrain" in output.lower():
                    await update.message.reply_text(
                        "⏳ **Retrain Skipped**\n\n"
                        "Threshold not reached yet.\n"
                        "Use `/retrain force` to force retrain.",
                        parse_mode="Markdown"
                    )
                    return
                
                # Extract new version from output
                import re
                version_match = re.search(r'New model: (v\d+)', output)
                new_version = version_match.group(1) if version_match else None
                
                # Extract metrics
                f1_match = re.search(r'Macro F1: ([\d.]+)', output)
                f1_score = f1_match.group(1) if f1_match else "N/A"
                
                classes_match = re.search(r'Classes: (\d+)', output)
                n_classes = classes_match.group(1) if classes_match else "N/A"
                
                await update.message.reply_text(
                    f"✅ **Retrain Complete!**\n\n"
                    f"📦 Version: {new_version or 'unknown'}\n"
                    f"📊 Classes: {n_classes}\n"
                    f"📈 Macro F1: {f1_score}\n\n"
                    f"🔄 Auto-reloading model...",
                    parse_mode="Markdown"
                )
                
                # Auto reload the new model
                if new_version and self._ml_classifier:
                    success = self._ml_classifier.reload(new_version)
                    if success:
                        # Mark reviewed data as TRAINED
                        trained_count = 0
                        if self._ml_tracking:
                            trained_count = self._ml_tracking.mark_as_trained()
                        
                        await update.message.reply_text(
                            f"🎉 **Model Active!**\n\n"
                            f"Bot is now using {new_version}.\n"
                            f"📝 Marked {trained_count} rows as TRAINED.\n"
                            f"No restart needed!",
                            parse_mode="Markdown"
                        )
                    else:
                        await update.message.reply_text(
                            f"⚠️ Retrain succeeded but reload failed.\n"
                            f"Use `/reloadmodel {new_version}` manually.",
                            parse_mode="Markdown"
                        )
                else:
                    # Mark reviewed data as TRAINED anyway
                    trained_count = 0
                    if self._ml_tracking:
                        trained_count = self._ml_tracking.mark_as_trained()
                    
                    await update.message.reply_text(
                        f"⚠️ Retrain succeeded!\n"
                        f"📝 Marked {trained_count} rows as TRAINED.\n"
                        f"Use `/reloadmodel` to load the new model.",
                        parse_mode="Markdown"
                    )
            else:
                # Error - get last few lines as error message
                error_msg = "\n".join(output_lines[-10:]) if output_lines else "Unknown error"
                await update.message.reply_text(
                    f"❌ **Retrain Failed**\n\n"
                    f"```\n{error_msg[:500]}\n```",
                    parse_mode="Markdown"
                )
                
        except Exception as e:
            _LOGGER.exception("Error during retrain: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")

    async def update_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /updatestats - Hitung ulang dan update statistik ke Monitoring sheet
        
        Otomatis jalan tiap jam, tapi bisa trigger manual via command ini.
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        
        if not self._ml_tracking:
            await update.message.reply_text("❌ ML Tracking not initialized")
            return
        
        await update.message.reply_text("🔄 Calculating and updating stats...")
        
        try:
            model_version = "unknown"
            if self._ml_classifier:
                model_version = self._ml_classifier.model_version
            
            stats = self._ml_tracking.calculate_and_update_daily_stats(model_version)
            
            if not stats:
                await update.message.reply_text(
                    "📊 **Stats Update**\n\n"
                    "No predictions today yet.",
                    parse_mode="Markdown"
                )
                return
            
            message = (
                f"✅ **Stats Updated to Monitoring Sheet!**\n\n"
                f"📅 Date: {stats.get('date', 'N/A')}\n"
                f"📈 Total Predictions: {stats.get('total_predictions', 0)}\n"
                f"🎯 Avg Confidence: {stats.get('avg_confidence', 0)*100:.1f}%\n\n"
                f"**Distribution:**\n"
                f"  ✅ AUTO: {stats.get('auto_count', 0)}\n"
                f"  🔶 HIGH: {stats.get('high_count', 0)}\n"
                f"  🟡 MEDIUM: {stats.get('medium_count', 0)}\n"
                f"  🔴 MANUAL: {stats.get('manual_count', 0)}\n\n"
                f"📝 Reviewed: {stats.get('reviewed_count', 0)}\n"
                f"🤖 Model: {stats.get('model_version', 'N/A')}"
            )
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error updating stats: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")


class TrendAlertService:
    """Background service for trend analysis and auto-alerts."""
    
    def __init__(
        self,
        ml_tracking: "MLTrackingClient",
        alert_chat_id: Optional[int] = None,
    ):
        self._ml_tracking = ml_tracking
        self._alert_chat_id = alert_chat_id
        self._last_alert_date = None
    
    def check_and_alert(self) -> Optional[str]:
        """
        Check for concerning trends and generate alert message if needed.
        Returns alert message if there's something to alert, None otherwise.
        """
        try:
            stats = self._ml_tracking.get_weekly_stats()
            if not stats or stats.get("total_predictions", 0) < 10:
                return None
            
            alerts = []
            total = stats.get("total_predictions", 0)
            auto_count = stats.get("auto_count", 0)
            manual_count = stats.get("manual_count", 0)
            avg_conf = stats.get("avg_confidence", 0)
            
            auto_rate = (auto_count / total * 100) if total > 0 else 0
            manual_rate = (manual_count / total * 100) if total > 0 else 0
            
            # Check for concerning patterns
            if auto_rate < 70:
                alerts.append(f"⚠️ Low automation rate: {auto_rate:.1f}% (target: ≥80%)")
            
            if manual_rate > 25:
                alerts.append(f"🔴 High manual classification: {manual_rate:.1f}% (target: <15%)")
            
            if avg_conf < 80:
                alerts.append(f"⚠️ Declining confidence: {avg_conf:.1f}% (target: ≥85%)")
            
            # Check pending review queue
            pending = self._ml_tracking.get_pending_review_count()
            total_pending = pending.get("total_pending", 0)
            if total_pending > 50:
                alerts.append(f"📋 Large review queue: {total_pending} items pending")
            
            if not alerts:
                return None
            
            message = (
                "🚨 **ML Model Alert**\n\n"
                + "\n".join(alerts)
                + f"\n\nLast 7 days: {total} predictions"
                + "\n\nConsider reviewing and retraining the model."
            )
            
            return message
            
        except Exception as e:
            _LOGGER.error("Error checking trends: %s", e)
            return None


def build_reporting_application(
    config: "Config",
    sheets_client: "GoogleSheetsClient",
    ml_classifier: Optional["MLClassifier"] = None,
    ml_tracking: Optional["MLTrackingClient"] = None,
):
    """
    Build the reporting/admin bot Application.
    
    Args:
        config: Application configuration
        sheets_client: Google Sheets client (shared)
        ml_classifier: Optional ML classifier for status
        ml_tracking: Optional ML tracking client
        
    Returns:
        Configured Application instance
    """
    from telegram.ext import Application
    from telegram import BotCommand
    
    # Admin command handler
    admin_handler = AdminCommandHandler(
        config=config,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking,
        admin_chat_ids=None,  # Allow all for now
    )
    
    # Build application
    application = ApplicationBuilder().token(config.telegram_reporting_bot_token).build()
    
    # Register admin commands
    application.add_handler(CommandHandler("stats", admin_handler.stats))
    application.add_handler(CommandHandler("report", admin_handler.report))
    application.add_handler(CommandHandler("modelstatus", admin_handler.model_status))
    application.add_handler(CommandHandler("pendingreview", admin_handler.pending_review))
    application.add_handler(CommandHandler("updatestats", admin_handler.update_stats))
    application.add_handler(CommandHandler("retrainstatus", admin_handler.retrain_status))
    application.add_handler(CommandHandler("retrain", admin_handler.retrain))
    application.add_handler(CommandHandler("reloadmodel", admin_handler.reload_model))
    application.add_handler(CommandHandler("helpml", admin_handler.help_admin))
    application.add_handler(CommandHandler("help", admin_handler.help_admin))
    
    # Set bot commands menu (appears when user types /)
    async def post_init(app: Application) -> None:
        commands = [
            BotCommand("help", "Tampilkan bantuan"),
            BotCommand("stats", "Statistik ML hari ini"),
            BotCommand("report", "Generate laporan"),
            BotCommand("modelstatus", "Status model ML"),
            BotCommand("pendingreview", "Lihat pending review"),
            BotCommand("retrainstatus", "Cek status retrain"),
            BotCommand("retrain", "Retrain model ML"),
            BotCommand("reloadmodel", "Reload model terbaru"),
            BotCommand("updatestats", "Update statistik manual"),
        ]
        await app.bot.set_my_commands(commands)
        _LOGGER.info("Bot commands menu set: %d commands", len(commands))
    
    application.post_init = post_init
    
    _LOGGER.info("Reporting bot application built with %d handlers", 
                len(application.handlers.get(0, [])))
    
    return application

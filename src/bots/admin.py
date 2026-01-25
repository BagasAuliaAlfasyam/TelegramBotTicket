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
    
    async def tiket_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /tiketreport [monthly|quarterly] [bulan/quarter] [tahun] - Generate ticket & SLA report from Logs sheet
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        
        # Parse arguments
        args = context.args or []
        period = args[0].lower() if args else "monthly"
        
        if period not in ["monthly", "quarterly"]:
            await update.message.reply_text(
                "📋 Penggunaan:\n"
                "  /tiketreport monthly [bulan] [tahun]\n"
                "  /tiketreport quarterly [quarter] [tahun]\n\n"
                "Contoh:\n"
                "  /tiketreport monthly → bulan ini\n"
                "  /tiketreport monthly 12 2025 → Desember 2025\n"
                "  /tiketreport quarterly → quarter ini\n"
                "  /tiketreport quarterly 4 2025 → Q4 2025"
            )
            return
        
        await update.message.reply_text("⏳ Generating report from Logs sheet...")
        
        try:
            from datetime import timedelta
            from collections import Counter
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Connect to Logs sheet
            cred_file = self._config.google_service_account_json
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
            client = gspread.authorize(credentials)
            
            spreadsheet = client.open(self._config.google_spreadsheet_name)
            logs_sheet = spreadsheet.worksheet("Logs")
            
            all_data = logs_sheet.get_all_values()
            if len(all_data) <= 1:
                await update.message.reply_text("❌ No data in Logs sheet")
                return
            
            headers = all_data[0]
            
            # Find column indices
            date_col = headers.index("Ticket Date") if "Ticket Date" in headers else -1
            sla_time_col = headers.index("SLA Response Time") if "SLA Response Time" in headers else -1
            sla_status_col = headers.index("SLA Status") if "SLA Status" in headers else -1
            symtomps_col = headers.index("Symtomps") if "Symtomps" in headers else -1
            
            if date_col == -1:
                await update.message.reply_text("❌ Ticket Date column not found")
                return
            
            # Calculate date range
            today = datetime.now(TZ).date()
            nama_bulan = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                          "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
            
            if period == "monthly":
                # Parse optional month and year
                if len(args) >= 3:
                    try:
                        month = int(args[1])
                        year = int(args[2])
                        if month < 1 or month > 12:
                            await update.message.reply_text("❌ Bulan harus 1-12")
                            return
                        start_date = datetime(year, month, 1).date()
                        # End date is last day of month
                        if month == 12:
                            end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
                        else:
                            end_date = datetime(year, month + 1, 1).date() - timedelta(days=1)
                    except ValueError:
                        await update.message.reply_text("❌ Format: /tiketreport monthly [bulan] [tahun]")
                        return
                else:
                    # Current month
                    start_date = today.replace(day=1)
                    end_date = today
                    month = today.month
                    year = today.year
                period_label = f"{nama_bulan[month]} {year}"
            else:  # quarterly
                # Parse optional quarter and year
                if len(args) >= 3:
                    try:
                        quarter = int(args[1])
                        year = int(args[2])
                        if quarter < 1 or quarter > 4:
                            await update.message.reply_text("❌ Quarter harus 1-4")
                            return
                        quarter = quarter - 1  # Convert to 0-indexed
                        start_month = quarter * 3 + 1
                        start_date = datetime(year, start_month, 1).date()
                        # End date is last day of quarter
                        end_month = start_month + 2
                        if end_month == 12:
                            end_date = datetime(year + 1, 1, 1).date() - timedelta(days=1)
                        else:
                            end_date = datetime(year, end_month + 1, 1).date() - timedelta(days=1)
                    except ValueError:
                        await update.message.reply_text("❌ Format: /tiketreport quarterly [quarter] [tahun]")
                        return
                else:
                    # Current quarter
                    quarter = (today.month - 1) // 3
                    start_month = quarter * 3 + 1
                    start_date = today.replace(month=start_month, day=1)
                    end_date = today
                    year = today.year
                quarter_names = ["Q1", "Q2", "Q3", "Q4"]
                # Get months in this quarter
                q_months = [nama_bulan[start_month], nama_bulan[start_month + 1], nama_bulan[start_month + 2]]
                period_label = f"{quarter_names[quarter]} {year} ({', '.join(q_months)})"
            
            # Filter and analyze data
            total_tickets = 0
            sla_times = []
            sla_met_count = 0
            sla_breach_count = 0
            symtomps_counter = Counter()
            
            for row in all_data[1:]:
                if len(row) <= date_col:
                    continue
                
                # Parse date
                date_str = row[date_col]
                try:
                    # Try common date formats
                    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                        try:
                            ticket_date = datetime.strptime(date_str.split()[0], fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        continue  # Skip if no format matches
                except Exception:
                    continue
                
                # Check if in period
                if ticket_date < start_date or ticket_date > end_date:
                    continue
                
                total_tickets += 1
                
                # SLA Time
                if sla_time_col != -1 and len(row) > sla_time_col:
                    try:
                        sla_time = float(row[sla_time_col]) if row[sla_time_col] else 0
                        if sla_time > 0:
                            sla_times.append(sla_time)
                    except ValueError:
                        pass
                
                # SLA Status
                if sla_status_col != -1 and len(row) > sla_status_col:
                    status = row[sla_status_col].strip().upper()
                    if status in ["MET", "OK", "WITHIN SLA"]:
                        sla_met_count += 1
                    elif status in ["BREACH", "BREACHED", "OVER SLA", "LATE", "TERLAMBAT"]:
                        sla_breach_count += 1
                
                # Symtomps
                if symtomps_col != -1 and len(row) > symtomps_col:
                    symtomps = row[symtomps_col].strip()
                    if symtomps:
                        symtomps_counter[symtomps] += 1
            
            if total_tickets == 0:
                await update.message.reply_text(
                    f"📊 **Laporan Tiket - {period_label}**\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                    parse_mode="Markdown"
                )
                return
            
            # Calculate averages
            avg_sla = sum(sla_times) / len(sla_times) if sla_times else 0
            sla_compliance = (sla_met_count / (sla_met_count + sla_breach_count) * 100) if (sla_met_count + sla_breach_count) > 0 else 0
            
            # Top symtomps
            top_symtomps = symtomps_counter.most_common(10)
            
            message = (
                f"📊 **Laporan Tiket - {period_label}**\n"
                f"Dibuat: {datetime.now(TZ).strftime('%d %b %Y %H:%M')}\n\n"
                f"**Ringkasan:**\n"
                f"  📋 Total Tiket: {total_tickets:,}\n"
                f"  ⏱️ Rata-rata SLA: {avg_sla:.1f} menit\n"
                f"  ✅ SLA Tercapai: {sla_met_count:,}\n"
                f"  ❌ SLA Terlambat: {sla_breach_count:,}\n"
                f"  📈 SLA Compliance: {sla_compliance:.1f}%\n\n"
            )
            
            if top_symtomps:
                message += "**Top 10 Symtomps:**\n"
                for i, (sym, count) in enumerate(top_symtomps, 1):
                    pct = (count / total_tickets * 100)
                    message += f"  {i}. {sym}: {count} ({pct:.1f}%)\n"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error generating ticket report: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def trend_bulan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /trendbulan [MIT|MIS] [bulan] [tahun] - Trend tiket bulanan per aplikasi
        
        Contoh:
            /trendbulan MIT           -> MIT bulan ini
            /trendbulan MIS 12 2025   -> MIS Desember 2025
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        
        args = context.args or []
        
        # Parse arguments
        if not args:
            await update.message.reply_text(
                "❌ **Penggunaan:**\n"
                "`/trendbulan [MIT|MIS] [bulan] [tahun]`\n\n"
                "**Contoh:**\n"
                "• `/trendbulan MIT` → MyTech bulan ini\n"
                "• `/trendbulan MIS` → MyStaff bulan ini\n"
                "• `/trendbulan MIT 12 2025` → MyTech Des 2025\n\n"
                "**Keterangan:**\n"
                "• MIT = MyTech\n"
                "• MIS = MyStaff",
                parse_mode="Markdown"
            )
            return
        
        # Validate app type
        app_type = args[0].upper()
        if app_type not in ["MIT", "MIS"]:
            await update.message.reply_text(
                f"❌ Aplikasi '{args[0]}' tidak valid.\n\n"
                "Gunakan **MIT** (MyTech) atau **MIS** (MyStaff)",
                parse_mode="Markdown"
            )
            return
        
        app_name = "MyTech" if app_type == "MIT" else "MyStaff"
        
        # Parse bulan & tahun (default: bulan ini)
        today = datetime.now(TZ).date()
        try:
            bulan = int(args[1]) if len(args) > 1 else today.month
            tahun = int(args[2]) if len(args) > 2 else today.year
            
            if bulan < 1 or bulan > 12:
                await update.message.reply_text("❌ Bulan harus 1-12")
                return
            if tahun < 2020 or tahun > 2030:
                await update.message.reply_text("❌ Tahun tidak valid")
                return
        except ValueError:
            await update.message.reply_text("❌ Format bulan/tahun tidak valid. Gunakan angka.")
            return
        
        await update.message.reply_text(f"⏳ Mengambil data {app_name} bulan {bulan}/{tahun}...")
        
        try:
            from collections import Counter
            import calendar
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Connect to Logs sheet
            cred_file = self._config.google_service_account_json
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
            client = gspread.authorize(credentials)
            
            spreadsheet = client.open(self._config.google_spreadsheet_name)
            logs_sheet = spreadsheet.worksheet("Logs")
            
            all_data = logs_sheet.get_all_values()
            if len(all_data) <= 1:
                await update.message.reply_text("❌ Tidak ada data di sheet Logs")
                return
            
            headers = all_data[0]
            
            # Column indices
            date_col = headers.index("Ticket Date") if "Ticket Date" in headers else 1
            app_col = headers.index("App") if "App" in headers else 13
            symtomps_col = headers.index("Symtomps") if "Symtomps" in headers else 19
            
            # Date range
            start_date = datetime(tahun, bulan, 1).date()
            last_day = calendar.monthrange(tahun, bulan)[1]
            end_date = datetime(tahun, bulan, last_day).date()
            
            # Filter and count
            total_tickets = 0
            symtomps_counter = Counter()
            
            for row in all_data[1:]:
                if len(row) <= max(date_col, app_col, symtomps_col):
                    continue
                
                # Check app type
                row_app = row[app_col].strip().upper() if len(row) > app_col else ""
                if row_app != app_type:
                    continue
                
                # Parse date
                date_str = row[date_col] if len(row) > date_col else ""
                ticket_date = None
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        ticket_date = datetime.strptime(date_str.split()[0], fmt).date()
                        break
                    except ValueError:
                        continue
                
                if not ticket_date:
                    continue
                
                # Check date range
                if ticket_date < start_date or ticket_date > end_date:
                    continue
                
                total_tickets += 1
                
                # Count symtomps
                symtomps = row[symtomps_col].strip() if len(row) > symtomps_col else ""
                if symtomps:
                    symtomps_counter[symtomps] += 1
            
            # Generate report
            nama_bulan = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                          "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
            
            if total_tickets == 0:
                await update.message.reply_text(
                    f"📊 **Trend Tiket {app_name} ({app_type})**\n"
                    f"📅 Periode: {nama_bulan[bulan]} {tahun}\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                    parse_mode="Markdown"
                )
                return
            
            # Top 10 symtomps
            top_symtomps = symtomps_counter.most_common(10)
            
            message = (
                f"📊 **Trend Tiket {app_name} ({app_type})**\n"
                f"📅 Periode: {nama_bulan[bulan]} {tahun}\n"
                f"📋 Total: {total_tickets:,} tiket\n\n"
                f"**Top 10 Symtomps:**\n"
            )
            
            for i, (sym, count) in enumerate(top_symtomps, 1):
                pct = (count / total_tickets * 100)
                message += f"{i}. {sym} - {count} ({pct:.1f}%)\n"
            
            if len(symtomps_counter) > 10:
                others = sum(c for s, c in symtomps_counter.items() if s not in dict(top_symtomps))
                message += f"\n_+{len(symtomps_counter) - 10} symtomps lainnya ({others} tiket)_"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error generating trend bulan: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def trend_mingguan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        /trendmingguan [minggu] [bulan] [tahun] [MIT|MIS] - Top 5 tiket per minggu
        
        Contoh:
            /trendmingguan                    -> Minggu ini, semua app
            /trendmingguan 2                  -> Minggu ke-2 bulan ini
            /trendmingguan 1 12 2025 MIT      -> Minggu 1 Des 2025 MyTech
        """
        if not update.effective_user or not self._is_admin(update.effective_user.id):
            return
        
        args = context.args or []
        today = datetime.now(TZ).date()
        
        # Determine current week (1-5)
        current_week = min((today.day - 1) // 7 + 1, 5)
        
        # Parse arguments
        try:
            minggu = int(args[0]) if len(args) > 0 else current_week
            bulan = int(args[1]) if len(args) > 1 else today.month
            tahun = int(args[2]) if len(args) > 2 else today.year
            app_type = args[3].upper() if len(args) > 3 else None
            
            if minggu < 1 or minggu > 5:
                await update.message.reply_text(
                    "❌ Minggu harus 1-5\n\n"
                    "• Minggu 1: tanggal 1-7\n"
                    "• Minggu 2: tanggal 8-14\n"
                    "• Minggu 3: tanggal 15-21\n"
                    "• Minggu 4: tanggal 22-28\n"
                    "• Minggu 5: tanggal 29-31"
                )
                return
            if bulan < 1 or bulan > 12:
                await update.message.reply_text("❌ Bulan harus 1-12")
                return
            if tahun < 2020 or tahun > 2030:
                await update.message.reply_text("❌ Tahun tidak valid")
                return
            if app_type and app_type not in ["MIT", "MIS"]:
                await update.message.reply_text(
                    f"❌ Aplikasi '{args[3]}' tidak valid.\n\n"
                    "Gunakan **MIT** (MyTech) atau **MIS** (MyStaff)\n"
                    "Atau kosongkan untuk semua aplikasi.",
                    parse_mode="Markdown"
                )
                return
        except ValueError:
            await update.message.reply_text(
                "❌ **Penggunaan:**\n"
                "`/trendmingguan [minggu] [bulan] [tahun] [MIT|MIS]`\n\n"
                "**Contoh:**\n"
                "• `/trendmingguan` → Minggu ini\n"
                "• `/trendmingguan 2` → Minggu ke-2 bulan ini\n"
                "• `/trendmingguan 1 1 2026 MIT` → Minggu 1 Jan 2026 MyTech\n\n"
                "**Keterangan Minggu:**\n"
                "• Minggu 1: tanggal 1-7\n"
                "• Minggu 2: tanggal 8-14\n"
                "• Minggu 3: tanggal 15-21\n"
                "• Minggu 4: tanggal 22-28\n"
                "• Minggu 5: tanggal 29-31",
                parse_mode="Markdown"
            )
            return
        
        app_label = ""
        if app_type:
            app_label = " MyTech" if app_type == "MIT" else " MyStaff"
        
        await update.message.reply_text(f"⏳ Mengambil data minggu ke-{minggu}{app_label} bulan {bulan}/{tahun}...")
        
        try:
            from collections import Counter, defaultdict
            import calendar
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Connect to Logs sheet
            cred_file = self._config.google_service_account_json
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_file(str(cred_file), scopes=scopes)
            client = gspread.authorize(credentials)
            
            spreadsheet = client.open(self._config.google_spreadsheet_name)
            logs_sheet = spreadsheet.worksheet("Logs")
            
            all_data = logs_sheet.get_all_values()
            if len(all_data) <= 1:
                await update.message.reply_text("❌ Tidak ada data di sheet Logs")
                return
            
            headers = all_data[0]
            
            # Column indices
            date_col = headers.index("Ticket Date") if "Ticket Date" in headers else 1
            app_col = headers.index("App") if "App" in headers else 13
            symtomps_col = headers.index("Symtomps") if "Symtomps" in headers else 19
            raw_text_col = 6  # tech raw text column
            
            # Calculate date range for the week
            last_day_of_month = calendar.monthrange(tahun, bulan)[1]
            if minggu == 1:
                start_day, end_day = 1, 7
            elif minggu == 2:
                start_day, end_day = 8, 14
            elif minggu == 3:
                start_day, end_day = 15, 21
            elif minggu == 4:
                start_day, end_day = 22, 28
            else:  # minggu 5
                start_day, end_day = 29, last_day_of_month
            
            # Adjust end_day if it exceeds month
            end_day = min(end_day, last_day_of_month)
            
            if start_day > last_day_of_month:
                await update.message.reply_text(
                    f"❌ Bulan {bulan}/{tahun} tidak memiliki minggu ke-{minggu}\n"
                    f"(Bulan ini hanya sampai tanggal {last_day_of_month})"
                )
                return
            
            start_date = datetime(tahun, bulan, start_day).date()
            end_date = datetime(tahun, bulan, end_day).date()
            
            # Filter and collect data
            total_tickets = 0
            symtomps_counter = Counter()
            ticket_examples = defaultdict(list)  # symtomps -> list of ticket names
            
            for row in all_data[1:]:
                if len(row) <= max(date_col, app_col, symtomps_col, raw_text_col):
                    continue
                
                # Check app type if specified
                if app_type:
                    row_app = row[app_col].strip().upper() if len(row) > app_col else ""
                    if row_app != app_type:
                        continue
                
                # Parse date
                date_str = row[date_col] if len(row) > date_col else ""
                ticket_date = None
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                    try:
                        ticket_date = datetime.strptime(date_str.split()[0], fmt).date()
                        break
                    except ValueError:
                        continue
                
                if not ticket_date:
                    continue
                
                # Check date range
                if ticket_date < start_date or ticket_date > end_date:
                    continue
                
                total_tickets += 1
                
                # Count symtomps
                symtomps = row[symtomps_col].strip() if len(row) > symtomps_col else ""
                raw_text = row[raw_text_col].strip() if len(row) > raw_text_col else ""
                
                if symtomps:
                    symtomps_counter[symtomps] += 1
                    # Keep ticket name example (max 5 per symtomps)
                    if len(ticket_examples[symtomps]) < 5:
                        # Truncate to 50 chars
                        ticket_name = raw_text[:50] + "..." if len(raw_text) > 50 else raw_text
                        if ticket_name:
                            ticket_examples[symtomps].append(ticket_name)
            
            # Generate report
            nama_bulan = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
                          "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
            
            if total_tickets == 0:
                await update.message.reply_text(
                    f"📅 **Trend Mingguan{app_label}**\n"
                    f"Minggu ke-{minggu} ({start_day}-{end_day} {nama_bulan[bulan]} {tahun})\n\n"
                    f"Tidak ada tiket ditemukan untuk periode ini.",
                    parse_mode="Markdown"
                )
                return
            
            # Top 5 symtomps
            top_symtomps = symtomps_counter.most_common(5)
            
            message = (
                f"📅 **Trend Mingguan{app_label}**\n"
                f"Minggu ke-{minggu} ({start_day}-{end_day} {nama_bulan[bulan]} {tahun})\n"
                f"📋 Total: {total_tickets:,} tiket\n\n"
                f"**Top 5 Symtomps:**\n"
            )
            
            for i, (sym, count) in enumerate(top_symtomps, 1):
                pct = (count / total_tickets * 100)
                message += f"\n{i}. **{sym}** - {count} tiket ({pct:.1f}%)\n"
                
                # Show example tickets
                examples = ticket_examples.get(sym, [])
                for ex in examples[:3]:  # Max 3 examples
                    message += f"   └ _{ex}_\n"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            _LOGGER.error("Error generating trend mingguan: %s", e)
            await update.message.reply_text(f"❌ Error: {e}")
    
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
                f"📝 [Open Google Sheets to Review](https://docs.google.com/spreadsheets/d/1rGvoDwBFA038N2jIYoFCcndFAFs9m9lDJ4YfMtZ6Ugs/edit?usp=sharing)"
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
            
            "**/reloadmodel**\n"
            "└ Load ulang model tanpa restart\n"
            "└ Contoh: `/reloadmodel v2` untuk versi spesifik\n\n"
            
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "📈 **LAPORAN TIKET**\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n\n"
            
            "**/tiketreport**\n"
            "└ Laporan tiket & SLA per bulan\n"
            "└ Contoh: `/tiketreport monthly` atau `/tiketreport quarterly`\n\n"
            
            "**/trendbulan**\n"
            "└ Trend Top 10 Symtomps per bulan\n"
            "└ Contoh: `/trendbulan MIT` atau `/trendbulan MIS 12 2025`\n"
            "└ MIT = MyTech, MIS = MyStaff\n\n"
            
            "**/trendmingguan**\n"
            "└ Top 5 tiket per minggu + contoh nama tiket\n"
            "└ Contoh: `/trendmingguan 2` (minggu ke-2)\n"
            "└ Contoh: `/trendmingguan 1 1 2026 MIT`\n\n"
            
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
                if self._ml_classifier:
                    metadata = self._ml_classifier.get_metadata()
                    existing = set(metadata.get('classes', []))
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
            
            # Wait for process to complete with 3-hour timeout
            RETRAIN_TIMEOUT = 3 * 60 * 60  # 3 hours in seconds
            try:
                await read_output()
                await asyncio.wait_for(process.wait(), timeout=RETRAIN_TIMEOUT)
            except asyncio.TimeoutError:
                process.kill()
                progress_task.cancel()
                await update.message.reply_text(
                    "❌ **Retrain Timeout!**\n\n"
                    "Training exceeded 3 hours and was terminated.\n"
                    "This may indicate a problem with the training data or server resources.",
                    parse_mode="Markdown"
                )
                return
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
    application.add_handler(CommandHandler("tiketreport", admin_handler.tiket_report))
    application.add_handler(CommandHandler("trendbulan", admin_handler.trend_bulan))
    application.add_handler(CommandHandler("trendmingguan", admin_handler.trend_mingguan))
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
            BotCommand("report", "Generate laporan ML"),
            BotCommand("tiketreport", "Laporan tiket & SLA"),
            BotCommand("trendbulan", "Trend tiket bulanan MIT/MIS"),
            BotCommand("trendmingguan", "Top tiket per minggu"),
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

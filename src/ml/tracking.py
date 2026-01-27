"""
ML Tracking & Monitoring Module
================================

Modul ini menangani logging prediksi ML ke sheet ML_Tracking dan 
stats monitoring ke sheet Monitoring di Google Sheets.

Sheet ML_Tracking digunakan untuk:
- Menyimpan audit trail setiap prediksi ML
- Review dan koreksi prediksi oleh admin
- Menyediakan data training untuk retrain model

Sheet Monitoring digunakan untuk:
- Stats agregat per jam untuk monitoring performa
- Tracking akurasi dan confidence model

Author: Bagas Aulia Alfasyam
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

import gspread
from gspread.exceptions import APIError, GSpreadException

if TYPE_CHECKING:
    from src.core.config import Config
    from src.ml.classifier import PredictionResult

_LOGGER = logging.getLogger(__name__)

# Sheet names
ML_TRACKING_SHEET = "ML_Tracking"
MONITORING_SHEET = "Monitoring"


class MLTrackingClient:
    """
    Client untuk logging prediksi ML ke Google Sheets.
    
    Sheets yang dikelola:
        - ML_Tracking: Audit trail setiap prediksi untuk review & retrain.
                      Struktur 7 kolom: tech_message_id, tech_raw_text, solving,
                      Symtomps, ml_confidence, review_status, created_at
        - Monitoring: Stats agregat per jam untuk monitoring performa model.
    
    Flow review_status:
        - pending: Menunggu review admin (untuk HIGH/MEDIUM/MANUAL)
        - auto_approved: Auto disetujui karena confidence >= 80% (AUTO)
        - APPROVED: Admin menyetujui prediksi ML sudah benar
        - CORRECTED: Admin mengoreksi prediksi yang salah (edit Symtomps)
        - TRAINED: Sudah digunakan untuk training model
    
    Attributes:
        _config: Konfigurasi aplikasi
        _spreadsheet: Instance gspread Spreadsheet
        _tracking_sheet: Worksheet ML_Tracking
        _monitoring_sheet: Worksheet Monitoring
        _tz: Timezone untuk timestamp (default Asia/Jakarta)
    """
    
    def __init__(self, config: "Config", spreadsheet: Optional[gspread.Spreadsheet] = None) -> None:
        self._config = config
        self._spreadsheet = spreadsheet
        self._tracking_sheet = None
        self._monitoring_sheet = None
        self._tz = ZoneInfo(config.timezone) if config.timezone else ZoneInfo("Asia/Jakarta")
        self._connect()
    
    def _connect(self) -> None:
        """
        Koneksi ke Google Spreadsheet dan setup sheets.
        
        Proses:
            1. Gunakan spreadsheet yang sudah ada atau buat koneksi baru
            2. Get atau create sheet ML_Tracking dengan 7 kolom
            3. Get atau create sheet Monitoring (optional, tidak fatal jika gagal)
        
        Raises:
            GSpreadException: Jika gagal koneksi ke spreadsheet
            APIError: Jika API Google Sheets error
        """
        try:
            # Reuse spreadsheet if provided, otherwise create new connection
            if self._spreadsheet is None:
                client = gspread.service_account(
                    filename=str(self._config.google_service_account_json)
                )
                self._spreadsheet = client.open(self._config.google_spreadsheet_name)
            
            # Get or create ML_Tracking sheet
            # 7 columns: match with actual sheet structure
            # 0=tech message id, 1=tech raw text, 2=solving, 3=Symtomps, 
            # 4=ml_confidence, 5=review_status, 6=Timestamps
            self._tracking_sheet = self._get_or_create_sheet(
                ML_TRACKING_SHEET,
                headers=[
                    "tech message id", "tech raw text", "solving",
                    "Symtomps", "ml_confidence", "review_status", "Timestamps"
                ]
            )
            
            # Get or create Monitoring sheet (optional, may timeout)
            # Using datetime_hour for hourly granularity
            try:
                self._monitoring_sheet = self._get_or_create_sheet(
                    MONITORING_SHEET,
                    headers=[
                        "datetime_hour", "total_predictions", "avg_confidence",
                        "auto_count", "high_count", "medium_count", "manual_count",
                        "reviewed_count", "accuracy", "model_version"
                    ]
                )
            except Exception as exc:
                _LOGGER.warning("Could not access Monitoring sheet (optional): %s", exc)
                self._monitoring_sheet = None
            
            _LOGGER.info("ML Tracking connected to spreadsheet '%s'", 
                        self._config.google_spreadsheet_name)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Unable to connect ML Tracking: %s", exc)
            raise
    
    def _get_or_create_sheet(self, sheet_name: str, headers: list[str]) -> gspread.Worksheet:
        """
        Ambil sheet yang sudah ada atau buat baru jika belum ada.
        
        Args:
            sheet_name: Nama sheet yang dicari/dibuat
            headers: List header kolom untuk sheet baru
            
        Returns:
            gspread.Worksheet: Instance worksheet yang berhasil diakses/dibuat
        """
        try:
            worksheet = self._spreadsheet.worksheet(sheet_name)
            _LOGGER.debug("Found existing sheet: %s", sheet_name)
            return worksheet
        except gspread.WorksheetNotFound:
            _LOGGER.info("Creating new sheet: %s", sheet_name)
            worksheet = self._spreadsheet.add_worksheet(
                title=sheet_name,
                rows=1000,
                cols=len(headers)
            )
            worksheet.append_row(headers, value_input_option='RAW')
            return worksheet
    
    def log_prediction(
        self,
        tech_message_id: int,
        tech_raw_text: str,
        solving: str,
        prediction_result: "PredictionResult",
    ) -> None:
        """
        Simpan prediksi ML ke sheet ML_Tracking untuk audit trail.
        
        Semua prediksi di-log dengan aturan review_status:
            - AUTO (confidence >= 80%): review_status = "auto_approved"
              Tidak perlu review karena confidence tinggi
            - HIGH/MEDIUM/MANUAL: review_status = "pending"
              Perlu review admin untuk memastikan prediksi benar
        
        Kolom yang disimpan:
            0. tech_message_id: ID pesan Telegram teknisi
            1. tech_raw_text: Teks mentah dari teknisi (max 500 char)
            2. solving: Teks solving dari ops (max 500 char)
            3. Symtomps: Hasil prediksi ML (bisa di-edit admin jika salah)
            4. ml_confidence: Nilai confidence 0.0 - 1.0
            5. review_status: Status review (pending/auto_approved)
            6. created_at: Timestamp kapan data dibuat
        
        Args:
            tech_message_id: Telegram message ID dari pesan teknisi
            tech_raw_text: Teks mentah reply teknisi
            solving: Teks solving yang diekstrak dari pesan ops
            prediction_result: Hasil prediksi dari MLClassifier.predict()
        
        Note:
            Method ini dipanggil SETELAH data berhasil ditulis ke Logs sheet
            untuk menjaga konsistensi data (tidak ada orphan di ML_Tracking).
        
        Raises:
            RuntimeError: Jika tracking sheet tidak terkoneksi
            GSpreadException: Jika gagal menulis ke Google Sheets
            APIError: Jika API Google error
        """
        if not self._tracking_sheet:
            _LOGGER.error("Tracking sheet not connected!")
            raise RuntimeError("ML_Tracking sheet not connected")
        
        # AUTO predictions are auto-approved, others need review
        if prediction_result.prediction_status == "AUTO":
            review_status = "auto_approved"
        else:
            review_status = "pending"
        
        try:
            # Get current timestamp in configured timezone
            now = datetime.now(self._tz)
            created_at = now.strftime("%Y-%m-%d %H:%M:%S")
            
            row = [
                str(tech_message_id),                              # 0: tech_message_id
                tech_raw_text[:500] if tech_raw_text else "",       # 1: tech_raw_text
                solving[:500] if solving else "",                   # 2: solving
                prediction_result.predicted_symtomps,               # 3: Symtomps (edit directly if wrong)
                prediction_result.ml_confidence,                    # 4: ml_confidence
                review_status,                                      # 5: review_status
                created_at,                                         # 6: created_at
            ]
            
            self._tracking_sheet.append_row(row, value_input_option='RAW')
            _LOGGER.info("Logged prediction for message %s to ML_Tracking", tech_message_id)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.warning("ML_Tracking write failed, attempting reconnect: %s", exc)
            # Coba reconnect dan retry sekali
            try:
                self._reconnect_tracking_sheet()
                row = [
                    str(tech_message_id),
                    tech_raw_text[:500] if tech_raw_text else "",
                    solving[:500] if solving else "",
                    prediction_result.predicted_symtomps,
                    prediction_result.ml_confidence,
                    review_status,
                    created_at,
                ]
                self._tracking_sheet.append_row(row, value_input_option='RAW')
                _LOGGER.info("Logged prediction for message %s after reconnect", tech_message_id)
            except Exception as retry_exc:
                _LOGGER.exception("Failed to log prediction after reconnect: %s", retry_exc)
                raise  # PENTING: Re-raise agar retry di collector.py berjalan!
    
    def _reconnect_tracking_sheet(self) -> None:
        """
        Reconnect ke ML_Tracking sheet saat koneksi terputus.
        
        Dipanggil secara otomatis saat append_row gagal dengan GSpreadException.
        Membuat koneksi baru ke Google Sheets dan refresh worksheet reference.
        
        Raises:
            GSpreadException: Jika reconnect gagal
        """
        _LOGGER.info("Reconnecting to ML_Tracking sheet...")
        try:
            # Buat koneksi baru
            client = gspread.service_account(
                filename=str(self._config.google_service_account_json)
            )
            self._spreadsheet = client.open(self._config.google_spreadsheet_name)
            self._tracking_sheet = self._spreadsheet.worksheet(ML_TRACKING_SHEET)
            _LOGGER.info("Successfully reconnected to ML_Tracking sheet")
        except Exception as e:
            _LOGGER.exception("Reconnect failed: %s", e)
            raise
    
    def update_hourly_stats(
        self,
        datetime_hour: str,
        model_version: str,
        total_predictions: int,
        avg_confidence: float,
        auto_count: int,
        high_count: int,
        medium_count: int,
        manual_count: int,
        reviewed_count: int = 0,
        accuracy: float = 0.0,
    ) -> None:
        """
        Update atau insert stats per jam ke Monitoring sheet.
        
        Jika sudah ada row untuk jam tersebut, update row yang ada.
        Jika belum ada, append row baru.
        
        Args:
            datetime_hour: Jam dalam format "YYYY-MM-DD HH:00" (contoh: "2026-01-23 14:00")
            model_version: Versi model yang sedang digunakan (contoh: "v1")
            total_predictions: Total prediksi dalam jam tersebut
            avg_confidence: Rata-rata confidence (0-100%)
            auto_count: Jumlah prediksi AUTO (confidence >= 80%)
            high_count: Jumlah prediksi HIGH (confidence 70-80%)
            medium_count: Jumlah prediksi MEDIUM (confidence 50-70%)
            manual_count: Jumlah prediksi MANUAL (confidence < 50%)
            reviewed_count: Jumlah yang sudah di-review
            accuracy: Persentase akurasi (0-100%)
        """
        if not self._monitoring_sheet:
            _LOGGER.warning("Monitoring sheet not connected, skipping stats")
            return
        
        try:
            # Cari row untuk jam ini
            try:
                cell = self._monitoring_sheet.find(datetime_hour, in_column=1)
                row_idx = cell.row
            except Exception:
                row_idx = None
            
            row_data = [
                datetime_hour,
                total_predictions,
                round(avg_confidence, 4),
                auto_count,
                high_count,
                medium_count,
                manual_count,
                reviewed_count,
                round(accuracy, 4),
                model_version,
            ]
            
            if row_idx:
                # Update existing row
                self._monitoring_sheet.update(
                    f'A{row_idx}:J{row_idx}',
                    [row_data],
                    value_input_option='RAW'
                )
            else:
                # Append new row
                self._monitoring_sheet.append_row(row_data, value_input_option='RAW')
            
            _LOGGER.debug("Updated hourly stats for %s", datetime_hour)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to update hourly stats: %s", exc)
    
    # Keep backward compatibility alias
    def update_daily_stats(
        self,
        model_version: str,
        total_predictions: int,
        avg_confidence: float,
        auto_count: int,
        high_count: int,
        medium_count: int,
        manual_count: int,
        reviewed_count: int = 0,
        accuracy: float = 0.0,
    ) -> None:
        """
        Wrapper backward compatibility - sekarang update per jam.
        
        Alias ini dipertahankan untuk kompatibilitas dengan kode lama.
        Secara internal memanggil update_hourly_stats() dengan jam saat ini.
        """
        now = datetime.now(self._tz)
        datetime_hour = now.strftime("%Y-%m-%d %H:00")
        self.update_hourly_stats(
            datetime_hour=datetime_hour,
            model_version=model_version,
            total_predictions=total_predictions,
            avg_confidence=avg_confidence,
            auto_count=auto_count,
            high_count=high_count,
            medium_count=medium_count,
            manual_count=manual_count,
            reviewed_count=reviewed_count,
            accuracy=accuracy,
        )
    
    def get_today_stats(self) -> dict:
        """
        Ambil stats agregat untuk hari ini dari Monitoring sheet.
        
        Menghitung total semua row per jam untuk tanggal hari ini.
        Confidence dihitung weighted average berdasarkan jumlah prediksi.
        
        Returns:
            dict dengan keys:
                - date: Tanggal hari ini (YYYY-MM-DD)
                - total_predictions: Total prediksi hari ini
                - avg_confidence: Rata-rata confidence (weighted)
                - auto_count: Jumlah AUTO
                - high_count: Jumlah HIGH
                - medium_count: Jumlah MEDIUM
                - manual_count: Jumlah MANUAL
                - reviewed_count: Jumlah yang sudah di-review
                - accuracy: Persentase akurasi
                - model_version: Versi model terakhir
        """
        if not self._monitoring_sheet:
            return {}
        
        try:
            today_str = date.today().isoformat()  # "2026-01-23"
            
            all_data = self._monitoring_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}
            
            # Aggregate all hourly rows for today
            total_predictions = 0
            confidence_weighted_sum = 0.0
            auto_count = 0
            high_count = 0
            medium_count = 0
            manual_count = 0
            reviewed_count = 0
            model_version = ""
            
            for row in all_data[1:]:
                if len(row) < 10:
                    continue
                
                # Check if datetime_hour starts with today's date
                datetime_hour = row[0]
                if not datetime_hour.startswith(today_str):
                    continue
                
                try:
                    predictions = int(row[1]) if row[1] else 0
                    total_predictions += predictions
                    
                    if row[2] and predictions > 0:
                        confidence_weighted_sum += float(row[2]) * predictions
                    
                    auto_count += int(row[3]) if row[3] else 0
                    high_count += int(row[4]) if row[4] else 0
                    medium_count += int(row[5]) if row[5] else 0
                    manual_count += int(row[6]) if row[6] else 0
                    reviewed_count += int(row[7]) if row[7] else 0
                    model_version = row[9] if row[9] else model_version
                except (ValueError, IndexError):
                    continue
            
            if total_predictions == 0:
                return {}
            
            return {
                "date": today_str,
                "total_predictions": total_predictions,
                "avg_confidence": confidence_weighted_sum / total_predictions,
                "auto_count": auto_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "manual_count": manual_count,
                "reviewed_count": reviewed_count,
                "accuracy": (reviewed_count / total_predictions * 100) if total_predictions > 0 else 0.0,
                "model_version": model_version,
            }
                
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get today stats: %s", exc)
            return {}
    
    def get_current_hour_stats(self) -> dict:
        """
        Ambil stats untuk jam saat ini dari Monitoring sheet.
        
        Mencari row dengan datetime_hour yang cocok dengan jam sekarang.
        
        Returns:
            dict dengan stats jam ini, atau {} jika tidak ada data.
            Keys: datetime_hour, total_predictions, avg_confidence, 
                  auto_count, high_count, medium_count, manual_count,
                  reviewed_count, accuracy, model_version
        """
        if not self._monitoring_sheet:
            return {}
        
        try:
            now = datetime.now(self._tz)
            datetime_hour = now.strftime("%Y-%m-%d %H:00")
            
            try:
                cell = self._monitoring_sheet.find(datetime_hour, in_column=1)
                row_values = self._monitoring_sheet.row_values(cell.row)
                
                return {
                    "datetime_hour": row_values[0] if len(row_values) > 0 else "",
                    "total_predictions": int(row_values[1]) if len(row_values) > 1 else 0,
                    "avg_confidence": float(row_values[2]) if len(row_values) > 2 else 0.0,
                    "auto_count": int(row_values[3]) if len(row_values) > 3 else 0,
                    "high_count": int(row_values[4]) if len(row_values) > 4 else 0,
                    "medium_count": int(row_values[5]) if len(row_values) > 5 else 0,
                    "manual_count": int(row_values[6]) if len(row_values) > 6 else 0,
                    "reviewed_count": int(row_values[7]) if len(row_values) > 7 else 0,
                    "accuracy": float(row_values[8]) if len(row_values) > 8 else 0.0,
                    "model_version": row_values[9] if len(row_values) > 9 else "",
                }
            except Exception:
                return {}
                
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get current hour stats: %s", exc)
            return {}
    
    def get_realtime_stats(self) -> dict:
        """
        Hitung stats real-time langsung dari ML_Tracking sheet.
        
        Berbeda dengan get_today_stats() yang membaca Monitoring sheet,
        method ini menghitung langsung dari data mentah ML_Tracking.
        Lebih akurat untuk data terkini tapi lebih lambat.
        
        Threshold confidence level:
            - auto_count: >= 90% confidence
            - high_count: 70-90% confidence  
            - medium_count: 50-70% confidence
            - manual_count: < 50% confidence
        
        Returns:
            dict dengan keys:
                - total_predictions: Total semua prediksi
                - avg_confidence: Rata-rata confidence (0-100%)
                - auto_count, high_count, medium_count, manual_count
                - reviewed_count: Sudah APPROVED/CORRECTED/TRAINED/auto_approved
                - pending_count: Masih menunggu review
        """
        if not self._tracking_sheet:
            return {}
        
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}
            
            # 7-column structure:
            # 0=tech_message_id, 1=tech_raw_text, 2=solving, 3=Symtomps, 4=ml_confidence, 5=review_status, 6=created_at
            total_predictions = len(all_data) - 1  # exclude header
            confidence_sum = 0.0
            auto_count = 0      # >= 90%
            high_count = 0      # 70-90%
            medium_count = 0    # 50-70%
            manual_count = 0    # < 50%
            reviewed_count = 0
            pending_count = 0
            
            for row in all_data[1:]:
                if len(row) < 6:
                    continue
                
                # Confidence (index 4) - handle comma as decimal separator
                try:
                    conf_str = row[4].replace(",", ".") if row[4] else "0.0"
                    confidence = float(conf_str)
                    confidence_sum += confidence
                    
                    # Count by confidence level
                    if confidence >= 0.90:
                        auto_count += 1
                    elif confidence >= 0.70:
                        high_count += 1
                    elif confidence >= 0.50:
                        medium_count += 1
                    else:
                        manual_count += 1
                except ValueError:
                    manual_count += 1
                
                # Review status (index 5)
                review_status = row[5]
                if review_status in ("APPROVED", "CORRECTED", "TRAINED", "auto_approved"):
                    reviewed_count += 1
                elif review_status == "pending":
                    pending_count += 1
            
            avg_confidence = (confidence_sum / total_predictions * 100) if total_predictions > 0 else 0.0
            
            return {
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "auto_count": auto_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "manual_count": manual_count,
                "reviewed_count": reviewed_count,
                "pending_count": pending_count,
            }
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get realtime stats: %s", exc)
            return {}
    
    def get_pending_review_count(self) -> dict:
        """
        Hitung jumlah prediksi yang masih menunggu review.
        
        Review diperlukan untuk memastikan prediksi ML benar sebelum
        data digunakan untuk training ulang model.
        
        Returns:
            dict dengan keys:
                - total_pending: Total semua yang pending
                - manual_pending: Pending untuk MANUAL
                - high_review_pending: Pending untuk HIGH
                - medium_review_pending: Pending untuk MEDIUM
        """
        if not self._tracking_sheet:
            return {"total_pending": 0, "manual_pending": 0, "high_review_pending": 0, "medium_review_pending": 0}
        
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return {"total_pending": 0, "manual_pending": 0, "high_review_pending": 0, "medium_review_pending": 0}
            
            manual_pending = 0
            high_pending = 0
            medium_pending = 0
            
            for row in all_data[1:]:
                if len(row) >= 6:
                    # Simple 6-column: 5=review_status
                    review_status = row[5]
                    
                    # Pending = review_status == "pending"
                    if review_status == "pending":
                        # Count all pending as one bucket
                        manual_pending += 1
            
            return {
                "total_pending": manual_pending + high_pending + medium_pending,
                "manual_pending": manual_pending,
                "high_review_pending": high_pending,
                "medium_review_pending": medium_pending,
            }
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to count pending reviews: %s", exc)
            return {"total_pending": 0, "manual_pending": 0, "high_review_pending": 0, "medium_review_pending": 0}
    
    def get_weekly_stats(self, days: int = 7) -> dict:
        """
        Ambil stats agregat untuk N hari terakhir.
        
        Args:
            days: Jumlah hari ke belakang (default 7 hari)
            
        Returns:
            dict dengan stats agregat dari Monitoring sheet
        """
        return self._get_aggregated_stats(days)
    
    def get_monthly_stats(self) -> dict:
        """
        Ambil stats agregat untuk 30 hari terakhir.
        
        Returns:
            dict dengan stats agregat dari Monitoring sheet
        """
        return self._get_aggregated_stats(30)
    
    def _get_aggregated_stats(self, days: int) -> dict:
        """
        Internal method untuk menghitung stats agregat N hari terakhir.
        
        Mengambil N row terakhir dari Monitoring sheet dan menghitung:
            - Total prediksi
            - Weighted average confidence
            - Count per kategori (AUTO/HIGH/MEDIUM/MANUAL)
            - Average accuracy
        
        Args:
            days: Jumlah hari/row terakhir yang diambil
            
        Returns:
            dict dengan keys: days, total_predictions, avg_confidence,
            auto_count, high_review_count, medium_review_count,
            manual_count, reviewed_count, accuracy
        """
        if not self._monitoring_sheet:
            return {}
        
        try:
            all_data = self._monitoring_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}
            
            recent = all_data[1:][-days:]
            
            if not recent:
                return {}
            
            total_predictions = 0
            total_confidence_sum = 0.0
            auto_count = 0
            high_count = 0
            medium_count = 0
            manual_count = 0
            reviewed_count = 0
            accuracy_sum = 0.0
            accuracy_days = 0
            
            for row in recent:
                if len(row) >= 10 and row[1]:
                    day_total = int(row[1]) if row[1] else 0
                    total_predictions += day_total
                    
                    if row[2] and day_total > 0:
                        total_confidence_sum += float(row[2]) * day_total
                    
                    auto_count += int(row[3]) if row[3] else 0
                    high_count += int(row[4]) if row[4] else 0
                    medium_count += int(row[5]) if row[5] else 0
                    manual_count += int(row[6]) if row[6] else 0
                    reviewed_count += int(row[7]) if row[7] else 0
                    
                    if row[8] and float(row[8]) > 0:
                        accuracy_sum += float(row[8])
                        accuracy_days += 1
            
            avg_confidence = (total_confidence_sum / total_predictions) if total_predictions > 0 else 0.0
            avg_accuracy = (accuracy_sum / accuracy_days) if accuracy_days > 0 else 0.0
            
            return {
                "days": days,
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "auto_count": auto_count,
                "high_review_count": high_count,
                "medium_review_count": medium_count,
                "manual_count": manual_count,
                "reviewed_count": reviewed_count,
                "accuracy": avg_accuracy,
            }
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get aggregated stats: %s", exc)
            return {}
    
    def get_reviewed_count(self) -> int:
        """
        Hitung jumlah prediksi yang sudah di-review dan siap training.
        
        Status yang dihitung: APPROVED dan CORRECTED.
        Data dengan status ini bisa digunakan untuk retrain model.
        
        Returns:
            int: Jumlah row yang siap untuk training
        """
        if not self._tracking_sheet:
            return 0
        
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return 0
            
            # Find review_status column from header
            headers = all_data[0]
            review_col = headers.index("review_status") if "review_status" in headers else -1
            if review_col == -1:
                _LOGGER.warning("review_status column not found in ML_Tracking")
                return 0
            
            # Count APPROVED and CORRECTED (ready for training)
            count = sum(
                1 for row in all_data[1:] 
                if len(row) > review_col and row[review_col] in ("APPROVED", "CORRECTED")
            )
            return count
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get reviewed count: %s", exc)
            return 0
    
    def get_reviewed_class_distribution(self) -> dict[str, int]:
        """
        Hitung distribusi kelas untuk prediksi yang siap training.
        
        Menghitung berapa banyak data per kategori Symtomps yang sudah
        di-review (APPROVED/CORRECTED) dan siap digunakan untuk training.
        
        Berguna untuk melihat apakah ada class imbalance sebelum retrain.
        
        Returns:
            dict: {nama_kelas: jumlah} contoh {"FO Cut": 15, "Modem Rusak": 8}
        """
        if not self._tracking_sheet:
            return {}
        
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}
            
            # Find columns from header
            headers = all_data[0]
            symtomps_col = headers.index("Symtomps") if "Symtomps" in headers else -1
            review_col = headers.index("review_status") if "review_status" in headers else -1
            
            if symtomps_col == -1 or review_col == -1:
                _LOGGER.warning("Required columns not found in ML_Tracking")
                return {}
            
            # Count by class for APPROVED/CORRECTED
            class_counts = {}
            for row in all_data[1:]:
                if len(row) > review_col and row[review_col] in ("APPROVED", "CORRECTED"):
                    symtomps = row[symtomps_col].strip() if len(row) > symtomps_col else ""
                    if symtomps:
                        class_counts[symtomps] = class_counts.get(symtomps, 0) + 1
            
            return class_counts
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get class distribution: %s", exc)
            return {}
    
    def get_trained_count(self) -> int:
        """
        Hitung jumlah total data yang sudah digunakan untuk training.
        
        Sumber data training:
            1. ML_Tracking rows dengan status TRAINED
            2. Logs rows yang kolom Symtomps-nya terisi (data historis)
        
        Returns:
            int: Total jumlah data training yang sudah digunakan
        """
        count = 0
        
        # Count TRAINED from ML_Tracking
        if self._tracking_sheet:
            try:
                all_data = self._tracking_sheet.get_all_values()
                if len(all_data) > 1:
                    headers = all_data[0]
                    review_col = headers.index("review_status") if "review_status" in headers else -1
                    if review_col != -1:
                        count += sum(
                            1 for row in all_data[1:] 
                            if len(row) > review_col and row[review_col] == "TRAINED"
                        )
            except (GSpreadException, APIError) as exc:
                _LOGGER.exception("Failed to get trained count from ML_Tracking: %s", exc)
        
        # Count from Logs where Symtomps is filled (historical training data)
        try:
            logs_sheet = self._spreadsheet.worksheet("Logs")
            logs_data = logs_sheet.get_all_values()
            if len(logs_data) > 1:
                headers = logs_data[0]
                symtomps_col = headers.index("Symtomps") if "Symtomps" in headers else -1
                if symtomps_col != -1:
                    count += sum(
                        1 for row in logs_data[1:]
                        if len(row) > symtomps_col and row[symtomps_col].strip() != ''
                    )
        except (GSpreadException, APIError, gspread.WorksheetNotFound) as exc:
            _LOGGER.warning("Could not count from Logs sheet: %s", exc)
        
        return count
    
    def mark_as_trained(self, row_indices: list[int] = None) -> int:
        """
        Tandai row yang sudah di-review sebagai TRAINED setelah training berhasil.
        
        Dipanggil setelah retrain model sukses untuk menandai data mana saja
        yang sudah digunakan untuk training. Ini mencegah data yang sama
        digunakan ulang di training berikutnya.
        
        Args:
            row_indices: Indeks row spesifik yang akan ditandai, atau None
                        untuk menandai semua APPROVED/CORRECTED
        
        Returns:
            int: Jumlah row yang berhasil ditandai TRAINED
        
        Note:
            Update dilakukan dalam batch 100 row untuk menghindari rate limit.
        """
        if not self._tracking_sheet:
            return 0
        
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return 0
            
            # Find review_status column
            headers = all_data[0]
            review_col = headers.index("review_status") if "review_status" in headers else -1
            if review_col == -1:
                _LOGGER.warning("review_status column not found")
                return 0
            
            # Find rows to update
            updates = []
            for i, row in enumerate(all_data[1:], start=2):  # 1-indexed, skip header
                if len(row) > review_col and row[review_col] in ("APPROVED", "CORRECTED"):
                    # gspread uses A1 notation
                    cell = gspread.utils.rowcol_to_a1(i, review_col + 1)
                    updates.append({
                        'range': cell,
                        'values': [['TRAINED']]
                    })
            
            if updates:
                # Batch update in chunks
                chunk_size = 100
                for i in range(0, len(updates), chunk_size):
                    chunk = updates[i:i+chunk_size]
                    self._tracking_sheet.batch_update(chunk, value_input_option='RAW')
                
                _LOGGER.info("Marked %d rows as TRAINED", len(updates))
            
            return len(updates)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to mark rows as trained: %s", exc)
            return 0

    def calculate_and_update_hourly_stats(self, model_version: str = "unknown") -> dict:
        """
        Hitung stats per jam dari ML_Tracking dan update ke Monitoring sheet.
        
        Method ini:
            1. Baca semua data dari ML_Tracking
            2. Filter hanya row untuk jam saat ini (berdasarkan created_at)
            3. Hitung total, confidence, dan distribusi status
            4. Update/insert ke Monitoring sheet
        
        Threshold confidence level:
            - auto_count: >= 90% confidence
            - high_count: 70-90% confidence
            - medium_count: 50-70% confidence
            - manual_count: < 50% confidence (pending)
        
        Args:
            model_version: Versi model yang digunakan (contoh: "v1")
        
        Returns:
            dict dengan stats jam ini termasuk:
                datetime_hour, total_predictions, avg_confidence,
                auto_count, high_count, medium_count, manual_count,
                reviewed_count, accuracy, model_version
        """
        if not self._tracking_sheet:
            _LOGGER.warning("Tracking sheet not connected, cannot calculate stats")
            return {}
        
        try:
            now = datetime.now(self._tz)
            datetime_hour = now.strftime("%Y-%m-%d %H:00")
            hour_prefix = now.strftime("%Y-%m-%d %H:")  # For matching "2026-01-23 14:XX"
            
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                _LOGGER.info("No data in ML_Tracking to calculate stats")
                return {}
            
            # 7-column structure:
            # 0=tech_message_id, 1=tech_raw_text, 2=solving, 3=Symtomps, 4=ml_confidence, 5=review_status, 6=created_at
            total_predictions = 0
            confidence_sum = 0.0
            auto_count = 0
            high_count = 0
            medium_count = 0
            pending_count = 0
            reviewed_count = 0
            
            for row in all_data[1:]:
                if len(row) < 7:
                    continue
                
                # Filter by current hour using created_at (index 6)
                created_at = row[6] if len(row) > 6 else ""
                if not created_at.startswith(hour_prefix):
                    continue
                
                total_predictions += 1
                
                # Confidence (index 4) - handle comma as decimal separator
                try:
                    conf_str = row[4].replace(",", ".") if row[4] else "0.0"
                    confidence = float(conf_str)
                    confidence_sum += confidence
                    
                    # Count by confidence level
                    if confidence >= 0.90:
                        auto_count += 1
                    elif confidence >= 0.70:
                        high_count += 1
                    elif confidence >= 0.50:
                        medium_count += 1
                except ValueError:
                    pass
                
                # Review status (index 5)
                review_status = row[5]
                if review_status == "pending":
                    pending_count += 1
                elif review_status in ("APPROVED", "CORRECTED", "TRAINED", "auto_approved"):
                    reviewed_count += 1
            
            if total_predictions == 0:
                _LOGGER.debug("No predictions for hour %s", datetime_hour)
                return {}
            
            avg_confidence = confidence_sum / total_predictions
            
            stats = {
                "datetime_hour": datetime_hour,
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "auto_count": auto_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "manual_count": pending_count,
                "reviewed_count": reviewed_count,
                "accuracy": reviewed_count / total_predictions if total_predictions > 0 else 0.0,
                "model_version": model_version,
            }
            
            # Update Monitoring sheet
            self.update_hourly_stats(
                datetime_hour=datetime_hour,
                model_version=model_version,
                total_predictions=total_predictions,
                avg_confidence=avg_confidence * 100,
                auto_count=auto_count,
                high_count=high_count,
                medium_count=medium_count,
                manual_count=pending_count,
                reviewed_count=reviewed_count,
                accuracy=stats["accuracy"] * 100,
            )
            
            _LOGGER.info(
                "Hourly stats updated for %s: %d predictions, %.1f%% avg confidence",
                datetime_hour, total_predictions, avg_confidence * 100
            )
            
            return stats
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to calculate hourly stats: %s", exc)
            return {}
    
    # Backward compatibility alias
    def calculate_and_update_daily_stats(self, model_version: str = "unknown") -> dict:
        """
        Alias backward compatibility - sekarang menghitung per jam.
        
        Dipertahankan untuk kompatibilitas kode lama yang memanggil method ini.
        """
        return self.calculate_and_update_hourly_stats(model_version)

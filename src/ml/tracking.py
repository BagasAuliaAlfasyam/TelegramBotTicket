"""
ML Tracking & Monitoring Module
================================
Handle logging ke ML_Tracking sheet dan Monitoring stats.
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
    Client untuk logging ML predictions ke sheet terpisah.
    
    Sheets:
    - ML_Tracking: Audit trail setiap prediksi (untuk review & retrain)
    - Monitoring: Stats harian agregat
    """
    
    def __init__(self, config: "Config", spreadsheet: Optional[gspread.Spreadsheet] = None) -> None:
        self._config = config
        self._spreadsheet = spreadsheet
        self._tracking_sheet = None
        self._monitoring_sheet = None
        self._tz = ZoneInfo(config.timezone) if config.timezone else ZoneInfo("Asia/Jakarta")
        self._connect()
    
    def _connect(self) -> None:
        """Connect ke spreadsheet dan setup sheets."""
        try:
            # Reuse spreadsheet if provided, otherwise create new connection
            if self._spreadsheet is None:
                client = gspread.service_account(
                    filename=str(self._config.google_service_account_json)
                )
                self._spreadsheet = client.open(self._config.google_spreadsheet_name)
            
            # Get or create ML_Tracking sheet
            # SIMPLE: 6 columns only
            # 0=tech_message_id, 1=tech_raw_text, 2=solving, 3=Symtomps, 4=ml_confidence, 5=review_status
            self._tracking_sheet = self._get_or_create_sheet(
                ML_TRACKING_SHEET,
                headers=[
                    "tech_message_id", "tech_raw_text", "solving",
                    "Symtomps", "ml_confidence", "review_status"
                ]
            )
            
            # Get or create Monitoring sheet (optional, may timeout)
            try:
                self._monitoring_sheet = self._get_or_create_sheet(
                    MONITORING_SHEET,
                    headers=[
                        "date", "total_predictions", "avg_confidence",
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
        """Get existing sheet atau create baru dengan headers."""
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
        Log prediksi ke ML_Tracking sheet.
        
        All predictions are logged:
        - AUTO: review_status = "auto_approved" (no review needed)
        - HIGH/MEDIUM/MANUAL: review_status = "pending" (needs review)
        
        Args:
            tech_message_id: Telegram message ID dari teknisi
            tech_raw_text: Raw text dari teknisi
            solving: Text solving dari ops
            prediction_result: Hasil prediksi dari MLClassifier
        """
        if not self._tracking_sheet:
            _LOGGER.warning("Tracking sheet not connected, skipping log")
            return
        
        # AUTO predictions are auto-approved, others need review
        if prediction_result.prediction_status == "AUTO":
            review_status = "auto_approved"
        else:
            review_status = "pending"
        
        try:
            row = [
                str(tech_message_id),                              # 0: tech_message_id
                tech_raw_text[:500] if tech_raw_text else "",       # 1: tech_raw_text
                solving[:500] if solving else "",                   # 2: solving
                prediction_result.predicted_symtomps,               # 3: Symtomps (edit directly if wrong)
                prediction_result.ml_confidence,                    # 4: ml_confidence
                review_status,                                      # 5: review_status
            ]
            
            self._tracking_sheet.append_row(row, value_input_option='RAW')
            _LOGGER.debug("Logged prediction for message %s", tech_message_id)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to log prediction: %s", exc)
    
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
        Update atau insert stats harian ke Monitoring sheet.
        """
        if not self._monitoring_sheet:
            _LOGGER.warning("Monitoring sheet not connected, skipping stats")
            return
        
        try:
            today_str = date.today().isoformat()
            
            # Cari row untuk hari ini
            try:
                cell = self._monitoring_sheet.find(today_str, in_column=1)
                row_idx = cell.row
            except Exception:
                row_idx = None
            
            row_data = [
                today_str,
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
            
            _LOGGER.debug("Updated daily stats for %s", today_str)
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to update daily stats: %s", exc)
    
    def get_today_stats(self) -> dict:
        """Get stats untuk hari ini dari Monitoring sheet."""
        if not self._monitoring_sheet:
            return {}
        
        try:
            today_str = date.today().isoformat()
            
            try:
                cell = self._monitoring_sheet.find(today_str, in_column=1)
                row_values = self._monitoring_sheet.row_values(cell.row)
                
                return {
                    "date": row_values[0] if len(row_values) > 0 else "",
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
            _LOGGER.exception("Failed to get today stats: %s", exc)
            return {}
    
    def get_pending_review_count(self) -> dict:
        """Hitung jumlah rows yang butuh review (belum APPROVED/CORRECTED)."""
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
        """Get aggregated stats untuk N hari terakhir."""
        return self._get_aggregated_stats(days)
    
    def get_monthly_stats(self) -> dict:
        """Get aggregated stats untuk 30 hari terakhir."""
        return self._get_aggregated_stats(30)
    
    def _get_aggregated_stats(self, days: int) -> dict:
        """Internal method to get aggregated stats for N days."""
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
        """Get count of reviewed predictions ready for training (APPROVED + CORRECTED)."""
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
    
    def get_trained_count(self) -> int:
        """
        Get count of already trained data.
        
        Includes:
        - ML_Tracking rows with TRAINED status
        - Logs rows where Symtomps is filled (historical training data)
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
        Mark reviewed rows as TRAINED after successful training.
        
        Args:
            row_indices: Specific row indices to mark, or None to mark all APPROVED/CORRECTED
            
        Returns:
            Number of rows marked as TRAINED
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

    def calculate_and_update_daily_stats(self, model_version: str = "unknown") -> dict:
        """
        Hitung stats dari ML_Tracking sheet dan update ke Monitoring sheet.
        
        Returns:
            dict dengan stats
        """
        if not self._tracking_sheet:
            _LOGGER.warning("Tracking sheet not connected, cannot calculate stats")
            return {}
        
        try:
            today = date.today()
            today_str = today.isoformat()
            
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                _LOGGER.info("No data in ML_Tracking to calculate stats")
                return {}
            
            # Simple 6-column structure:
            # 0=tech_message_id, 1=tech_raw_text, 2=solving, 3=Symtomps, 4=ml_confidence, 5=review_status
            total_predictions = len(all_data) - 1  # exclude header
            confidence_sum = 0.0
            auto_count = 0
            pending_count = 0
            reviewed_count = 0
            
            for row in all_data[1:]:
                if len(row) < 6:
                    continue
                
                # Confidence (index 4)
                try:
                    confidence = float(row[4]) if row[4] else 0.0
                    confidence_sum += confidence
                    
                    # Count by confidence level
                    if confidence >= 0.90:
                        auto_count += 1
                except ValueError:
                    pass
                
                # Review status (index 5)
                review_status = row[5]
                if review_status == "auto_approved":
                    pass  # already counted in auto_count
                elif review_status == "pending":
                    pending_count += 1
                elif review_status in ("APPROVED", "CORRECTED", "TRAINED"):
                    reviewed_count += 1
            
            avg_confidence = confidence_sum / total_predictions if total_predictions > 0 else 0.0
            
            stats = {
                "date": today_str,
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "auto_count": auto_count,
                "high_count": 0,
                "medium_count": 0,
                "manual_count": pending_count,
                "reviewed_count": reviewed_count,
                "accuracy": reviewed_count / total_predictions if total_predictions > 0 else 0.0,
                "model_version": model_version,
            }
            
            # Update Monitoring sheet
            self.update_daily_stats(
                model_version=model_version,
                total_predictions=total_predictions,
                avg_confidence=avg_confidence * 100,
                auto_count=auto_count,
                high_count=0,
                medium_count=0,
                manual_count=pending_count,
                reviewed_count=reviewed_count,
                accuracy=stats["accuracy"] * 100,
            )
            
            _LOGGER.info(
                "Daily stats updated: %d predictions, %.1f%% avg confidence, %d reviewed",
                total_predictions, avg_confidence * 100, reviewed_count
            )
            
            return stats
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to calculate daily stats: %s", exc)
            return {}

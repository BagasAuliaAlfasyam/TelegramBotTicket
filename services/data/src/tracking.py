"""
ML Tracking Client (microservice version)
===========================================
Manages ML_Tracking and Monitoring sheets.
"""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import gspread
from gspread.exceptions import APIError, GSpreadException

from services.shared.config import DataServiceConfig

_LOGGER = logging.getLogger(__name__)

ML_TRACKING_SHEET = "ML_Tracking"
MONITORING_SHEET = "Monitoring"


class MLTrackingClient:
    """Manages ML_Tracking and Monitoring sheets."""

    def __init__(self, config: DataServiceConfig, spreadsheet: Optional[gspread.Spreadsheet] = None):
        self._config = config
        self._spreadsheet = spreadsheet
        self._tracking_sheet = None
        self._monitoring_sheet = None
        self._tz = ZoneInfo(config.timezone)
        self._connect()

    def _connect(self) -> None:
        if self._spreadsheet is None:
            client = gspread.service_account(filename=str(self._config.google_service_account_json))
            self._spreadsheet = client.open(self._config.google_spreadsheet_name)

        self._tracking_sheet = self._get_or_create_sheet(
            ML_TRACKING_SHEET,
            ["tech message id", "tech raw text", "solving", "Symtomps",
             "ml_confidence", "review_status", "Timestamps", "prediction_source"]
        )
        try:
            self._monitoring_sheet = self._get_or_create_sheet(
                MONITORING_SHEET,
                ["datetime_hour", "total_predictions", "avg_confidence",
                 "auto_count", "high_count", "medium_count", "manual_count",
                 "reviewed_count", "accuracy", "model_version"]
            )
        except Exception:
            self._monitoring_sheet = None

    def _get_or_create_sheet(self, name: str, headers: list[str]) -> gspread.Worksheet:
        try:
            return self._spreadsheet.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = self._spreadsheet.add_worksheet(title=name, rows=1000, cols=len(headers))
            ws.append_row(headers, value_input_option='RAW')
            return ws

    def log_prediction(
        self,
        tech_message_id: int,
        tech_raw_text: str,
        solving: str,
        predicted_symtomps: str,
        ml_confidence: float,
        prediction_status: str,
        source: str = "lightgbm",
    ) -> None:
        """Log prediction to ML_Tracking sheet."""
        if not self._tracking_sheet:
            raise RuntimeError("ML_Tracking sheet not connected")

        review_status = "auto_approved" if prediction_status == "AUTO" else "pending"
        now = datetime.now(self._tz)

        row = [
            str(tech_message_id),
            tech_raw_text[:500] if tech_raw_text else "",
            solving[:500] if solving else "",
            predicted_symtomps,
            ml_confidence,
            review_status,
            now.strftime("%Y-%m-%d %H:%M:%S"),
            source,
        ]

        try:
            self._tracking_sheet.append_row(row, value_input_option='RAW')
        except (GSpreadException, APIError):
            _LOGGER.warning("Reconnecting to ML_Tracking...")
            self._reconnect()
            self._tracking_sheet.append_row(row, value_input_option='RAW')

    def _reconnect(self):
        client = gspread.service_account(filename=str(self._config.google_service_account_json))
        self._spreadsheet = client.open(self._config.google_spreadsheet_name)
        self._tracking_sheet = self._spreadsheet.worksheet(ML_TRACKING_SHEET)

    def get_realtime_stats(self) -> dict:
        """Stats from ML_Tracking sheet."""
        if not self._tracking_sheet:
            return {}
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}

            total = len(all_data) - 1
            conf_sum = 0.0
            auto_count = high_count = medium_count = manual_count = 0
            reviewed_count = pending_count = 0
            gemini_count = hybrid_count = 0

            for row in all_data[1:]:
                if len(row) < 6:
                    continue
                try:
                    conf = float(row[4].replace(",", ".")) if row[4] else 0.0
                    conf_sum += conf
                    if conf >= 0.90: auto_count += 1
                    elif conf >= 0.70: high_count += 1
                    elif conf >= 0.50: medium_count += 1
                    else: manual_count += 1
                except ValueError:
                    manual_count += 1

                status = row[5]
                if status in ("APPROVED", "CORRECTED", "TRAINED", "auto_approved"):
                    reviewed_count += 1
                elif status == "pending":
                    pending_count += 1

                # Track prediction source (column 7 if exists)
                if len(row) > 7 and row[7]:
                    if row[7] == "gemini": gemini_count += 1
                    elif row[7] == "hybrid": hybrid_count += 1

            return {
                "total_predictions": total,
                "avg_confidence": (conf_sum / total * 100) if total > 0 else 0.0,
                "auto_count": auto_count, "high_count": high_count,
                "medium_count": medium_count, "manual_count": manual_count,
                "reviewed_count": reviewed_count, "pending_count": pending_count,
                "gemini_count": gemini_count, "hybrid_count": hybrid_count,
            }
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get stats: %s", exc)
            return {}

    def get_weekly_stats(self, days: int = 7) -> dict:
        return self._get_aggregated_stats(days)

    def get_monthly_stats(self) -> dict:
        return self._get_aggregated_stats(30)

    def _get_aggregated_stats(self, days: int) -> dict:
        if not self._monitoring_sheet:
            return {}
        try:
            all_data = self._monitoring_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}
            recent = all_data[1:][-days:]
            if not recent:
                return {}

            total = 0
            conf_sum = 0.0
            auto_c = high_c = medium_c = manual_c = reviewed_c = 0

            for row in recent:
                if len(row) >= 10 and row[1]:
                    day_total = int(row[1]) if row[1] else 0
                    total += day_total
                    if row[2] and day_total > 0: conf_sum += float(row[2]) * day_total
                    auto_c += int(row[3]) if row[3] else 0
                    high_c += int(row[4]) if row[4] else 0
                    medium_c += int(row[5]) if row[5] else 0
                    manual_c += int(row[6]) if row[6] else 0
                    reviewed_c += int(row[7]) if row[7] else 0

            return {
                "days": days, "total_predictions": total,
                "avg_confidence": (conf_sum / total) if total > 0 else 0.0,
                "auto_count": auto_c, "high_review_count": high_c,
                "medium_review_count": medium_c, "manual_count": manual_c,
                "reviewed_count": reviewed_c,
            }
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get aggregated stats: %s", exc)
            return {}

    def calculate_and_update_hourly_stats(self, model_version: str = "unknown") -> dict:
        """Calculate stats for previous hour and update Monitoring sheet."""
        if not self._tracking_sheet:
            return {}

        try:
            now = datetime.now(self._tz)
            previous_hour = now - timedelta(hours=1)
            datetime_hour = previous_hour.strftime("%Y-%m-%d %H:00")
            hour_prefix = previous_hour.strftime("%Y-%m-%d %H:")

            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return {}

            total = 0
            conf_sum = 0.0
            auto_c = high_c = medium_c = pending_c = reviewed_c = 0

            for row in all_data[1:]:
                if len(row) < 7:
                    continue
                created_at = row[6] if len(row) > 6 else ""
                if not created_at.startswith(hour_prefix):
                    continue

                total += 1
                try:
                    conf = float(row[4].replace(",", ".")) if row[4] else 0.0
                    conf_sum += conf
                    if conf >= 0.90: auto_c += 1
                    elif conf >= 0.70: high_c += 1
                    elif conf >= 0.50: medium_c += 1
                except ValueError:
                    pass

                if row[5] == "pending": pending_c += 1
                elif row[5] in ("APPROVED", "CORRECTED", "TRAINED", "auto_approved"):
                    reviewed_c += 1

            if total == 0:
                return {}

            avg_conf = conf_sum / total

            if self._monitoring_sheet:
                row_data = [datetime_hour, total, round(avg_conf * 100, 2),
                           auto_c, high_c, medium_c, pending_c,
                           reviewed_c, round(reviewed_c / total * 100, 2), model_version]
                try:
                    cell = self._monitoring_sheet.find(datetime_hour, in_column=1)
                    self._monitoring_sheet.update(f'A{cell.row}:J{cell.row}', [row_data], value_input_option='RAW')
                except Exception:
                    self._monitoring_sheet.append_row(row_data, value_input_option='RAW')

            return {
                "datetime_hour": datetime_hour, "total_predictions": total,
                "avg_confidence": avg_conf, "auto_count": auto_c,
                "high_count": high_c, "medium_count": medium_c,
                "manual_count": pending_c, "model_version": model_version,
            }
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed hourly stats: %s", exc)
            return {}

    def get_training_data(self) -> list[dict]:
        """Get approved/corrected data for training."""
        if not self._tracking_sheet:
            return []
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return []
            training = []
            for row in all_data[1:]:
                if len(row) >= 6 and row[5] in ("APPROVED", "CORRECTED", "TRAINED", "auto_approved"):
                    training.append({
                        "tech_message_id": row[0],
                        "tech_raw_text": row[1],
                        "solving": row[2],
                        "symtomps": row[3],
                        "ml_confidence": row[4],
                        "review_status": row[5],
                    })
            return training
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed training data: %s", exc)
            return []

    def mark_as_trained(self) -> int:
        """Mark APPROVED/CORRECTED as TRAINED."""
        if not self._tracking_sheet:
            return 0
        try:
            all_data = self._tracking_sheet.get_all_values()
            if len(all_data) <= 1:
                return 0
            updates = []
            for i, row in enumerate(all_data[1:], start=2):
                if len(row) >= 6 and row[5] in ("APPROVED", "CORRECTED"):
                    cell = gspread.utils.rowcol_to_a1(i, 6)
                    updates.append({'range': cell, 'values': [['TRAINED']]})
            if updates:
                for i in range(0, len(updates), 100):
                    self._tracking_sheet.batch_update(updates[i:i+100], value_input_option='RAW')
            return len(updates)
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed mark trained: %s", exc)
            return 0

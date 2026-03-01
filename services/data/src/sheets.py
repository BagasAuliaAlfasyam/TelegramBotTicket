"""
Google Sheets Client (microservice version)
=============================================
Centralized Sheets access — all services go through Data API.

Retry Policy (per operation):
  - Attempt 1 → langsung
  - Attempt 2 → tunggu 2s   (setelah error 500/503)
  - Attempt 3 → tunggu 5s
  - Masih gagal → raise → main.py tangkap → masuk DLQ
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Optional

import gspread
from gspread.exceptions import APIError, GSpreadException
from services.shared.config import DataServiceConfig

_LOGGER = logging.getLogger(__name__)

# Retry delays (detik) untuk operasi Sheets yang dapat 500/503
_SHEETS_RETRY_DELAYS = (2.0, 5.0)
# HTTP status yang layak di-retry (server-side transient)
_RETRYABLE_STATUS = {500, 503}


def _is_retryable(exc: Exception) -> bool:
    """Cek apakah APIError dari gspread bisa di-retry."""
    if isinstance(exc, APIError):
        try:
            return exc.response.status_code in _RETRYABLE_STATUS
        except Exception:
            return True  # Kalau gak bisa cek → retry saja
    return False


def _sheets_call(fn, *args, **kwargs):
    """
    Wrapper untuk call ke gspread dengan auto-retry + backoff.
    Attempt 1 → langsung, Attempt 2 → +2s, Attempt 3 → +5s.
    Raise exception terakhir jika semua attempt gagal.
    """
    last_exc: Exception | None = None
    delays = (None, *_SHEETS_RETRY_DELAYS)
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            _LOGGER.warning(
                "Sheets retry attempt %d/%d (backoff %.0fs) after: %s",
                attempt, len(delays), delay, last_exc,
            )
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except (APIError, GSpreadException) as exc:
            last_exc = exc
            if not _is_retryable(exc):
                # 400, 403, 404 → gak perlu retry
                raise
    raise last_exc  # type: ignore[misc]


class GoogleSheetsClient:
    """CRUD operations on Logs worksheet."""

    def __init__(self, config: DataServiceConfig) -> None:
        self._config = config
        self._spreadsheet = None
        self._worksheet = self._connect()

    @property
    def spreadsheet(self) -> gspread.Spreadsheet:
        return self._spreadsheet

    def _connect(self) -> gspread.Worksheet:
        client = gspread.service_account(
            filename=str(self._config.google_service_account_json)
        )
        self._spreadsheet = client.open(self._config.google_spreadsheet_name)
        worksheet = self._spreadsheet.worksheet(self._config.google_worksheet_name)
        _LOGGER.info(
            "Connected to '%s' / '%s'",
            self._config.google_spreadsheet_name,
            self._config.google_worksheet_name,
        )
        return worksheet

    def append_log_row(self, row: Sequence, *, return_row_index: bool = False) -> int | None:
        """Append row ke Logs sheet, dengan retry otomatis saat 500/503."""
        resp = _sheets_call(
            self._worksheet.append_row, list(row), value_input_option="RAW"
        )
        row_index: int | None = None
        if return_row_index:
            try:
                updated_range = resp.get("updates", {}).get("updatedRange", "")
                # e.g. "Logs!A5215:T5215" → 5215
                row_index = int(updated_range.split(":")[-1].lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            except Exception:
                _LOGGER.warning("Could not parse row_index from response: %s", resp)
        _LOGGER.info(
            "Appended log row for tech_message_id=%s at row %s",
            row[3] if len(row) > 3 else "?",
            row_index,
        )
        return row_index

    def update_log_row(self, row_index: int, row: Sequence) -> None:
        """Update row di Sheets, dengan retry otomatis saat 500/503."""
        end_col = "T" if len(row) >= 20 else "S"
        _sheets_call(
            self._worksheet.update,
            range_name=f"A{row_index}:{end_col}{row_index}",
            values=[list(row)],
            value_input_option="RAW",
        )
        _LOGGER.info("Updated row %s", row_index)

    def get_row(self, row_index: int) -> list[str]:
        try:
            return _sheets_call(self._worksheet.row_values, row_index)
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to fetch row %s", row_index)
            return []

    def find_row_index_by_tech_message_id(self, tech_message_id: str) -> int | None:
        try:
            # in_column=4 → column D only (tech_message_id), prevents false matches
            cell = _sheets_call(
                self._worksheet.find, str(tech_message_id), in_column=4
            )
        except (GSpreadException, APIError):
            _LOGGER.exception("Search failed for tech_message_id=%s", tech_message_id)
            return None
        if not cell:
            return None
        return cell.row

    def get_all_logs_data(self) -> list[list[str]]:
        try:
            return _sheets_call(self._worksheet.get_all_values)
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get all logs: %s", exc)
            return []

    def get_logs_for_training(self) -> list[dict]:
        try:
            all_data = _sheets_call(self._worksheet.get_all_values)
            if len(all_data) <= 1:
                return []
            training_data = []
            for row in all_data[1:]:
                if len(row) >= 20 and row[19]:
                    training_data.append({
                        "tech_message_id": row[3] if len(row) > 3 else "",
                        "tech_raw_text": row[6] if len(row) > 6 else "",
                        "solving": row[11] if len(row) > 11 else "",
                        "symtomps": row[19],
                    })
            return training_data
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get training data: %s", exc)
            return []

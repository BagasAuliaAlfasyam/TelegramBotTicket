"""
Google Sheets Client (microservice version)
=============================================
Centralized Sheets access — all services go through Data API.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Optional

import gspread
from gspread.exceptions import APIError, GSpreadException
from services.shared.config import DataServiceConfig

_LOGGER = logging.getLogger(__name__)


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
        _LOGGER.info("Connected to '%s' / '%s'",
                      self._config.google_spreadsheet_name,
                      self._config.google_worksheet_name)
        return worksheet

    def append_log_row(self, row: Sequence, *, return_row_index: bool = False) -> int | None:
        row_index = None
        if return_row_index:
            row_index = len(self._worksheet.get_all_values()) + 1
        self._worksheet.append_row(list(row), value_input_option="RAW")
        _LOGGER.info("Appended log row for tech_message_id=%s", row[3] if len(row) > 3 else "?")
        return row_index

    def update_log_row(self, row_index: int, row: Sequence) -> None:
        end_col = "T" if len(row) >= 20 else "S"
        self._worksheet.update(
            range_name=f"A{row_index}:{end_col}{row_index}",
            values=[list(row)],
            value_input_option="RAW",
        )
        _LOGGER.info("Updated row %s", row_index)

    def get_row(self, row_index: int) -> list[str]:
        try:
            return self._worksheet.row_values(row_index)
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to fetch row %s", row_index)
            return []

    def find_row_index_by_tech_message_id(self, tech_message_id: str) -> int | None:
        try:
            cell = self._worksheet.find(str(tech_message_id))
        except (GSpreadException, APIError):
            _LOGGER.exception("Search failed for tech_message_id=%s", tech_message_id)
            return None
        if not cell:
            return None
        return cell.row

    def get_all_logs_data(self) -> list[list[str]]:
        try:
            return self._worksheet.get_all_values()
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get all logs: %s", exc)
            return []

    def get_logs_for_training(self) -> list[dict]:
        try:
            all_data = self._worksheet.get_all_values()
            if len(all_data) <= 1:
                return []
            training_data = []
            for row in all_data[1:]:
                if len(row) >= 20 and row[19]:
                    training_data.append({
                        "tech_raw_text": row[6] if len(row) > 6 else "",
                        "solving": row[11] if len(row) > 11 else "",
                        "symtomps": row[19],
                    })
            return training_data
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get training data: %s", exc)
            return []

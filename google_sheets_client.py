"""Wrapper around gspread to append log rows into Google Sheets."""
from __future__ import annotations

import logging
from typing import Sequence

import gspread
from gspread.exceptions import APIError, GSpreadException

from config import Config

_LOGGER = logging.getLogger(__name__)


class GoogleSheetsClient:
    """Simple helper for appending rows to the configured worksheet."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._worksheet = self._connect()

    def _connect(self):
        try:
            client = gspread.service_account(filename=str(self._config.google_service_account_json))
            spreadsheet = client.open(self._config.google_spreadsheet_name)
            worksheet = spreadsheet.worksheet(self._config.google_worksheet_name)
            _LOGGER.info(
                "Connected to spreadsheet '%s' worksheet '%s'",
                self._config.google_spreadsheet_name,
                self._config.google_worksheet_name,
            )
            return worksheet
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Unable to connect to Google Sheets: %s", exc)
            raise

    def append_log_row(self, row: Sequence[str | int | None], *, return_row_index: bool = False) -> int | None:
        """Append a row to the worksheet using raw value input.

        When return_row_index=True, returns the 1-based row index where the row was written.
        """
        row_index = None
        if return_row_index:
            # Approximate next row by counting existing non-empty rows.
            row_index = len(self._worksheet.get_all_values()) + 1
        try:
            self._worksheet.append_row(list(row), value_input_option="RAW")
            _LOGGER.info("Appended log row for tech_message_id=%s", row[3] if len(row) > 3 else "?")
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to append row to Google Sheets: %s", exc)
            raise
        return row_index

    def update_log_row(self, row_index: int, row: Sequence[str | int | None]) -> None:
        """Update an existing row in the worksheet."""
        try:
            self._worksheet.update(
                range_name=f"A{row_index}:S{row_index}",
                values=[list(row)],
                value_input_option="RAW",
            )
            _LOGGER.info("Updated log row %s for tech_message_id=%s", row_index, row[3] if len(row) > 3 else "?")
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to update row %s in Google Sheets: %s", row_index, exc)
            raise

    def get_row(self, row_index: int) -> list[str]:
        """Fetch a single row by index (1-based)."""
        try:
            values = self._worksheet.row_values(row_index)
            return values
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to fetch row %s", row_index)
            return []

    def find_row_index_by_tech_message_id(self, tech_message_id: str) -> int | None:
        """Locate the row index for a given technician message id.

        Uses worksheet.find which searches the whole sheet; assumes tech_message_id
        is unique per sheet (one ticket per technician message).
        """
        try:
            cell = self._worksheet.find(str(tech_message_id))
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to search for tech_message_id=%s", tech_message_id)
            return None

        if not cell:
            return None
        return cell.row

"""
Google Sheets Service
======================

Modul ini menyediakan wrapper untuk gspread agar mudah menulis
dan membaca data dari Google Sheets.

Digunakan untuk:
- Append row log tiket ke sheet Logs
- Update row yang sudah ada
- Membaca data untuk training ML

Kolom sheet Logs:
    A-R: Berbagai field tiket
    S: Solving (dari ops)
    T: Symtomps (hasil prediksi ML atau manual)

Author: Bagas Aulia Alfasyam
"""
from __future__ import annotations

import logging
from typing import Sequence, TYPE_CHECKING

import gspread
from gspread.exceptions import APIError, GSpreadException

if TYPE_CHECKING:
    from src.core.config import Config

_LOGGER = logging.getLogger(__name__)


class GoogleSheetsClient:
    """
    Helper untuk operasi read/write ke Google Sheets worksheet.
    
    Client ini menangani koneksi ke Google Sheets menggunakan service account
    dan menyediakan method untuk append, update, dan query data.
    
    Attributes:
        _config: Konfigurasi aplikasi (credentials, spreadsheet name, dll)
        _spreadsheet: Instance gspread.Spreadsheet
        _worksheet: Instance gspread.Worksheet untuk sheet Logs
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._spreadsheet = None
        self._worksheet = self._connect()

    @property
    def spreadsheet(self) -> gspread.Spreadsheet:
        """
        Getter untuk spreadsheet object.
        
        Berguna untuk reuse koneksi oleh service lain seperti MLTrackingClient
        agar tidak perlu buat koneksi baru.
        
        Returns:
            gspread.Spreadsheet: Instance spreadsheet yang terkoneksi
        """
        return self._spreadsheet

    def _connect(self) -> gspread.Worksheet:
        """
        Koneksi ke Google Sheets dan return worksheet yang dikonfigurasi.
        
        Proses:
            1. Buat client dengan service account credentials
            2. Buka spreadsheet berdasarkan nama di config
            3. Akses worksheet (sheet) sesuai config
        
        Returns:
            gspread.Worksheet: Worksheet yang siap digunakan
            
        Raises:
            GSpreadException: Jika gagal koneksi
            APIError: Jika API Google Sheets error
        """
        try:
            client = gspread.service_account(
                filename=str(self._config.google_service_account_json)
            )
            self._spreadsheet = client.open(self._config.google_spreadsheet_name)
            worksheet = self._spreadsheet.worksheet(self._config.google_worksheet_name)
            _LOGGER.info(
                "Connected to spreadsheet '%s' worksheet '%s'",
                self._config.google_spreadsheet_name,
                self._config.google_worksheet_name,
            )
            return worksheet
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Unable to connect to Google Sheets: %s", exc)
            raise

    def append_log_row(
        self, 
        row: Sequence[str | int | None], 
        *, 
        return_row_index: bool = False
    ) -> int | None:
        """
        Tambah baris baru ke worksheet menggunakan RAW value input.
        
        Baris ditambahkan di akhir sheet. Method ini digunakan untuk
        menyimpan data tiket baru ke Logs sheet.
        
        Args:
            row: Sequence data untuk satu baris (list/tuple)
            return_row_index: Jika True, return index baris yang ditulis
        
        Returns:
            int atau None: Index baris (1-based) jika return_row_index=True
            
        Raises:
            GSpreadException: Jika gagal append
            APIError: Jika API quota exceeded atau error lain
        """
        row_index = None
        if return_row_index:
            # Approximate next row by counting existing non-empty rows.
            row_index = len(self._worksheet.get_all_values()) + 1
        try:
            self._worksheet.append_row(list(row), value_input_option="RAW")
            _LOGGER.info(
                "Appended log row for tech_message_id=%s", 
                row[3] if len(row) > 3 else "?"
            )
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to append row to Google Sheets: %s", exc)
            raise
        return row_index

    def update_log_row(
        self, 
        row_index: int, 
        row: Sequence[str | int | None]
    ) -> None:
        """
        Update baris yang sudah ada di worksheet.
        
        Digunakan untuk update data tiket yang sudah di-append sebelumnya,
        misalnya ketika ada data tambahan atau koreksi.
        
        Args:
            row_index: Index baris yang akan di-update (1-based)
            row: Data baris baru untuk menggantikan data lama
            
        Note:
            Range update otomatis menyesuaikan: A-T jika ada 20+ kolom,
            A-S jika kurang.
        """
        try:
            # Determine range based on row length (support kolom Symtomps di T)
            end_col = "T" if len(row) >= 20 else "S"
            self._worksheet.update(
                range_name=f"A{row_index}:{end_col}{row_index}",
                values=[list(row)],
                value_input_option="RAW",
            )
            _LOGGER.info(
                "Updated log row %s for tech_message_id=%s", 
                row_index, 
                row[3] if len(row) > 3 else "?"
            )
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to update row %s in Google Sheets: %s", row_index, exc)
            raise

    def get_row(self, row_index: int) -> list[str]:
        """
        Ambil satu baris berdasarkan index.
        
        Args:
            row_index: Index baris (1-based, baris 1 = header)
            
        Returns:
            list[str]: List nilai dari baris tersebut, atau [] jika gagal
        """
        try:
            values = self._worksheet.row_values(row_index)
            return values
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to fetch row %s", row_index)
            return []

    def find_row_index_by_tech_message_id(self, tech_message_id: str) -> int | None:
        """
        Cari index baris berdasarkan tech_message_id.
        
        Menggunakan worksheet.find() untuk mencari di seluruh sheet.
        Asumsi: tech_message_id unik per sheet (satu tiket per pesan teknisi).
        
        Args:
            tech_message_id: ID pesan Telegram teknisi yang dicari
            
        Returns:
            int atau None: Index baris jika ditemukan, None jika tidak ada
        """
        try:
            cell = self._worksheet.find(str(tech_message_id))
        except (GSpreadException, APIError):
            _LOGGER.exception("Failed to search for tech_message_id=%s", tech_message_id)
            return None

        if not cell:
            return None
        return cell.row
    
    def get_all_logs_data(self) -> list[list[str]]:
        """
        Ambil semua data dari Logs sheet.
        
        Digunakan untuk retraining model ML.
        
        Returns:
            list[list[str]]: Semua baris termasuk header
        """
        try:
            return self._worksheet.get_all_values()
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get all logs data: %s", exc)
            return []
    
    def get_logs_for_training(self) -> list[dict]:
        """
        Ambil data Logs yang sudah dilabeli untuk training.
        
        Hanya return baris yang kolom Symtomps (kolom T) sudah terisi.
        Data ini digunakan sebagai training data untuk model ML.
        
        Returns:
            list[dict]: List dict dengan keys:
                - tech_raw_text: Teks dari teknisi (kolom M)
                - solving: Teks solving dari ops (kolom P)
                - symtomps: Label kategori (kolom T)
        """
        try:
            all_data = self._worksheet.get_all_values()
            if len(all_data) <= 1:
                return []
            
            training_data = []
            for row in all_data[1:]:  # Skip header
                if len(row) >= 20 and row[19]:  # Column T (index 19)
                    training_data.append({
                        "tech_raw_text": row[12] if len(row) > 12 else "",  # Column M
                        "solving": row[15] if len(row) > 15 else "",  # Column P
                        "symtomps": row[19],  # Column T
                    })
            
            _LOGGER.info("Found %d labeled rows from Logs", len(training_data))
            return training_data
            
        except (GSpreadException, APIError) as exc:
            _LOGGER.exception("Failed to get logs for training: %s", exc)
            return []

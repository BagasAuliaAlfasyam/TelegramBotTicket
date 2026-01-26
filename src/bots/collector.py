"""
Collecting Bot Handler
=======================
Telegram bot logic for collecting Ops replies and logging them to Google Sheets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

from telegram import Bot, Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.bots.parsers import parse_ops_message
from src.bots.sla import compute_sla

if TYPE_CHECKING:
    from src.core.config import Config
    from src.services.sheets import GoogleSheetsClient
    from src.services.storage import S3Uploader
    from src.ml.classifier import MLClassifier, PredictionResult
    from src.ml.tracking import MLTrackingClient

_LOGGER = logging.getLogger(__name__)

# Constants
_ALLOWED_APPS = {"MIT", "MIS"}
_SOLVER_NAME_MAP = {
    "-bg": "Bagas",
    "-dm": "Damas",
    "-dvd": "David",
    "-fr": "Fairuz",
}
_DEFAULT_MIME_BY_TYPE = {
    "photo": "image/jpeg",
    "document": "application/octet-stream",
    "video": "video/mp4",
    "animation": "video/mp4",
    "audio": "audio/mpeg",
}
_EXTENSION_BY_MIME = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "video/mp4": "mp4",
    "video/quicktime": "mov",
    "application/pdf": "pdf",
    "audio/mpeg": "mp3",
}
_EXTENSION_BY_TYPE = {
    "photo": "jpg",
    "document": "bin",
    "video": "mp4",
    "animation": "mp4",
    "audio": "mp3",
}


@dataclass
class MediaInfo:
    """
    Container untuk informasi file media (foto/video/dokumen).
    
    Digunakan untuk menyimpan detail media attachment dari message teknisi
    sebelum di-upload ke S3 storage.
    
    Attributes:
        media_type: Jenis media (photo/video/document/audio/animation)
        file_id: ID unik dari Telegram untuk download file
        mime_type: Tipe MIME file (contoh: image/jpeg, video/mp4)
        file_name: Nama file asli (hanya untuk document)
    """
    media_type: str
    file_id: str
    mime_type: str | None
    file_name: str | None


class PersistentState:
    """
    Penyimpanan cache ke file JSON yang tetap ada meskipun bot restart.
    
    Fungsi utama:
    - Menyimpan waktu "oncek" (acknowledgment) per tiket
    - Menyimpan row index di Google Sheets per tiket
    - Data tidak hilang saat bot restart/crash
    
    Cara kerja:
    - Saat bot start: load data dari state_cache.json
    - Saat ada update: auto-save ke state_cache.json
    
    Key format: (chat_id, tech_message_id) -> value
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._ack_cache: dict[tuple[str, str], datetime] = {}
        self._ack_row_index: dict[tuple[str, str], int] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except Exception:
            _LOGGER.warning("Failed to load state file, starting fresh", exc_info=True)
            return
        for key_str, ts in raw.get("ack_cache", {}).items():
            parts = key_str.split("::", 1)
            if len(parts) != 2:
                continue
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                continue
            self._ack_cache[(parts[0], parts[1])] = dt
        for key_str, idx in raw.get("ack_row_index", {}).items():
            parts = key_str.split("::", 1)
            if len(parts) != 2:
                continue
            if isinstance(idx, int):
                self._ack_row_index[(parts[0], parts[1])] = idx

    def _dump(self) -> None:
        try:
            data = {
                "ack_cache": {
                    f"{k[0]}::{k[1]}": v.isoformat() for k, v in self._ack_cache.items()
                },
                "ack_row_index": {
                    f"{k[0]}::{k[1]}": v for k, v in self._ack_row_index.items()
                },
            }
            self._path.write_text(json.dumps(data))
        except Exception:
            _LOGGER.warning("Failed to persist state", exc_info=True)

    def get_ack_dt(self, key: tuple[str, str]) -> datetime | None:
        return self._ack_cache.get(key)

    def set_ack_dt(self, key: tuple[str, str], value: datetime) -> None:
        self._ack_cache[key] = value
        self._dump()

    def get_row_idx(self, key: tuple[str, str]) -> int | None:
        return self._ack_row_index.get(key)

    def set_row_idx(self, key: tuple[str, str], value: int) -> None:
        self._ack_row_index[key] = value
        self._dump()


class OpsCollector:
    """
    Handler utama untuk memproses reply Ops dan menyimpan ke berbagai sistem.
    
    Ini adalah class inti yang menangani seluruh flow collecting tiket:
    1. Validasi sender (hanya user authorized yang bisa input)
    2. Parsing format MIT/MIS dari text message
    3. Extract info media (foto/video) dari message teknisi
    4. Upload media ke S3/MinIO
    5. Prediksi Symtomps menggunakan ML model
    6. Simpan ke Logs sheet (primary data)
    7. Simpan ke ML_Tracking sheet (audit trail ML)
    8. Kirim notifikasi ke admin
    
    Attributes:
        _config: Konfigurasi aplikasi (token, timezone, dll)
        _sheets: Client untuk operasi Google Sheets
        _s3_uploader: Client untuk upload ke S3/MinIO
        _ml_classifier: Model ML untuk prediksi Symtomps
        _ml_tracking: Client untuk logging ke ML_Tracking sheet
        _reporting_bot: Bot terpisah untuk kirim notifikasi
        _state: Cache persistent untuk data oncek
        _tz: Timezone untuk formatting waktu (Asia/Jakarta)
    """

    def __init__(
        self,
        config: "Config",
        sheets_client: "GoogleSheetsClient",
        s3_uploader: "S3Uploader",
        ml_classifier: Optional["MLClassifier"] = None,
        ml_tracking: Optional["MLTrackingClient"] = None,
    ) -> None:
        self._config = config
        self._sheets = sheets_client
        self._s3_uploader = s3_uploader
        self._ml_classifier = ml_classifier
        self._ml_tracking = ml_tracking
        
        # Reporting bot is optional - use for notifications only if configured
        if config.telegram_reporting_bot_token:
            self._reporting_bot = Bot(token=config.telegram_reporting_bot_token)
        else:
            self._reporting_bot = None
            _LOGGER.warning("Reporting bot token not configured, notifications disabled")
        
        self._state = PersistentState(Path("state_cache.json"))
        try:
            self._tz = ZoneInfo(config.timezone)
        except Exception:
            _LOGGER.warning("Invalid timezone '%s', falling back to UTC", config.timezone)
            self._tz = timezone.utc

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """\n        Health check endpoint untuk monitoring.\n        \n        Trigger: Command /health atau /ping\n        Response: "OK {timestamp}"\n        \n        Berguna untuk:\n        - Cek apakah bot masih hidup\n        - Debugging saat deployment\n        - Monitoring uptime\n        """
        if not update.message:
            return
        await update.message.reply_text(f"OK {datetime.now(timezone.utc).isoformat()}")

    async def handle_ops_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """\n        Proses reply dari Ops, validasi format, dan simpan ke Google Sheets.\n        \n        Ini adalah FUNGSI UTAMA yang dipanggil setiap kali ada message masuk.\n        \n        Flow:\n        1. Validasi: Ada message? Ada reply? Sender authorized?\n        2. Detect type: "oncek" (acknowledgment) atau solusi (MIT/MIS)?\n        3. Extract info: message teknisi, media, timestamps\n        4. Upload media ke S3 (kalau ada foto/video)\n        5. Build row data untuk Sheets\n        6. ML Prediction (kalau bukan oncek):\n           - Prediksi kategori Symtomps\n           - Confidence >= 80%: isi otomatis di Logs\n           - Confidence < 80%: kosongkan (perlu review manual)\n        7. Simpan ke Logs sheet (update atau append)\n        8. Simpan ke ML_Tracking sheet (retry 3x kalau gagal)\n        9. Kirim notifikasi ke admin\n        \n        Format message yang diterima:\n        - Oncek: "oncek" (case insensitive)\n        - Solusi: "MIT [solving] -bg" atau "MIS [solving] -dm"\n        \n        Kalau Logs gagal: return early, tidak lanjut\n        Kalau ML_Tracking gagal: Logs tetap aman, notify admin\n        """
        message = update.effective_message
        if not message or not message.text or not message.reply_to_message:
            return

        sender_id = message.from_user.id if message.from_user else None
        if self._config.admin_user_ids and (
            sender_id is None or sender_id not in self._config.admin_user_ids
        ):
            _LOGGER.info("Ignoring message from unauthorized user %s", sender_id)
            return

        now_dt = _to_utc_datetime(message.date) or datetime.now(timezone.utc)

        is_ack = "oncek" in message.text.lower()
        parsed = parse_ops_message(message.text, _ALLOWED_APPS) if not is_ack else None
        if not parsed and not is_ack:
            return

        tech_message = message.reply_to_message
        media_info = _extract_media_info(tech_message)
        tech_message_date = ""
        tech_message_time = ""
        tech_message_dt = _to_utc_datetime(tech_message.date)
        
        # Detect forwarded messages
        if getattr(tech_message, "forward_date", None) or \
           getattr(tech_message, "forward_origin", None) or \
           getattr(tech_message, "is_automatic_forward", False):
            tech_message_dt = now_dt
            
        tech_local_dt = _to_local_datetime(tech_message_dt, self._tz)
        if tech_local_dt:
            tech_message_date = tech_local_dt.date().isoformat()
            tech_message_time = tech_local_dt.strftime("%H:%M:%S")

        now_local_dt = _to_local_datetime(now_dt, self._tz)
        ticket_date = now_local_dt.date().isoformat() if now_local_dt else ""
        response_at = now_local_dt.strftime("%H:%M:%S") if now_local_dt else ""
        tech_raw_text = tech_message.caption or tech_message.text or ""

        # Handle media upload
        media_url = ""
        media_upload_failed = False
        if media_info.file_id and media_info.media_type:
            try:
                media_url = await self._upload_media_to_s3(
                    bot=context.bot,
                    info=media_info,
                    chat_id=message.chat.id,
                    message_id=tech_message.message_id,
                )
            except Exception:
                _LOGGER.exception("Failed to upload media to S3")
                media_upload_failed = True

        chat_id_str = str(message.chat.id)
        group_label = message.chat.title or message.chat.username or chat_id_str
        tech_message_id_str = str(tech_message.message_id)
        notify_text = ''
        ack_key = (chat_id_str, tech_message_id_str)
        ack_idx = self._state.get_row_idx(ack_key)
        
        if not ack_idx:
            ack_idx = self._sheets.find_row_index_by_tech_message_id(tech_message_id_str)
            if ack_idx:
                self._state.set_row_idx(ack_key, ack_idx)
        
        duplicate_ack = False
        row: list[str | int | float] | None = None

        if is_ack:
            existing_ack_dt = self._state.get_ack_dt(ack_key)
            existing_row = self._sheets.get_row(ack_idx) if ack_idx else []
            has_solution = bool(existing_row and len(existing_row) > 10 and any(existing_row[9:12]))

            if has_solution:
                duplicate_ack = True
                notify_text = f"Tiket sudah punya solusi, oncek diabaikan untuk grup: {group_label}."
            elif existing_ack_dt or ack_idx:
                ack_dt = existing_ack_dt or now_dt
                self._state.set_ack_dt(ack_key, ack_dt)
                sla_response_min, sla_status, sla_remaining_min = compute_sla(tech_message_dt, ack_dt, self._tz)
                row = [
                    group_label, ticket_date, response_at, tech_message_id_str,
                    tech_message_date, tech_message_time, tech_raw_text,
                    media_info.media_type, media_url,
                    '', '', '', '', '', '',
                    'true', sla_response_min, sla_status, sla_remaining_min,
                ]
                notify_text = f"Respon awal (oncek) diperbarui untuk grup: {group_label}."
            else:
                ack_dt = now_dt
                self._state.set_ack_dt(ack_key, ack_dt)
                sla_response_min, sla_status, sla_remaining_min = compute_sla(tech_message_dt, ack_dt, self._tz)
                row = [
                    group_label, ticket_date, response_at, tech_message_id_str,
                    tech_message_date, tech_message_time, tech_raw_text,
                    media_info.media_type, media_url,
                    '', '', '', '', '', '',
                    'true', sla_response_min, sla_status, sla_remaining_min,
                ]
                notify_text = f"Respon awal (oncek) sudah dicatat untuk grup: {group_label}."
        else:
            solver_key = parsed["initials"].lower()
            solver_name = _SOLVER_NAME_MAP.get(solver_key)
            if not solver_name:
                _LOGGER.info("Ignoring ops reply from unknown solver '%s'", parsed["initials"])
                return

            ack_dt = self._state.get_ack_dt(ack_key)
            sla_response_min, sla_status, sla_remaining_min = compute_sla(tech_message_dt, ack_dt or now_dt, self._tz)
            solve_timestamp = now_local_dt.strftime("%H:%M:%S") if now_local_dt else ""
            is_oncek_flag = 'true' if ack_dt or self._state.get_row_idx(ack_key) else 'false'
            row = [
                group_label, ticket_date, response_at, tech_message_id_str,
                tech_message_date, tech_message_time, tech_raw_text,
                media_info.media_type, media_url,
                str(message.message_id), message.text, parsed["solving"],
                solve_timestamp, parsed["app"], solver_name,
                is_oncek_flag, sla_response_min, sla_status, sla_remaining_min,
            ]
            notify_text = f"Laporan dan solusi sudah dicatat oleh {solver_name} untuk grup: {group_label}."
        
        # ============ ML PREDICTION ============
        ml_prediction: "PredictionResult" | None = None
        symtomps_label = ""
        
        if self._ml_classifier and self._ml_classifier.is_loaded and not is_ack:
            solving_text = parsed["solving"] if parsed else ""
            ml_prediction = self._ml_classifier.predict(tech_raw_text, solving_text)
            
            # Hanya isi Symtomps jika confidence >= 80%
            # Di bawah 80% tetap log ke Logs tapi Symtomps kosong
            if ml_prediction.ml_confidence >= 0.80:
                symtomps_label = ml_prediction.predicted_symtomps
            else:
                symtomps_label = ""  # Kosongkan untuk review manual
            
            # NOTE: log_prediction ke ML_Tracking dipindah ke SETELAH Logs sukses
            # untuk menghindari inkonsistensi data (ML_Tracking ada, Logs kosong)
            
            _LOGGER.info(
                "ML Prediction: %s (%.1f%%) - %s -> Logs Symtomps: %s",
                ml_prediction.predicted_symtomps,
                ml_prediction.ml_confidence * 100,
                ml_prediction.prediction_status,
                symtomps_label or "(empty - below 80%)"
            )
            
            # Update notify_text dengan hasil prediksi ML
            confidence_pct = ml_prediction.ml_confidence * 100
            if ml_prediction.ml_confidence >= 0.80:
                notify_text += f"\n\n🤖 Symtomps: <b>{ml_prediction.predicted_symtomps}</b> ({confidence_pct:.0f}%)"
            else:
                notify_text += f"\n\n🤖 Symtomps: <b>{ml_prediction.predicted_symtomps}</b> ({confidence_pct:.0f}%) ⚠️ <i>perlu review</i>"
        
        # Tambah kolom Symtomps ke row (kolom T, index 19)
        if row is not None:
            row.append(symtomps_label)
        # ============ END ML PREDICTION ============
        
        notification_chat_id = self._config.telegram_admin_chat_id or message.chat.id
        can_reply_in_place = notification_chat_id == message.chat.id
        link_text = _build_message_link(message.chat.id, message.message_id) if not can_reply_in_place else None

        try:
            ack_idx = self._state.get_row_idx((chat_id_str, tech_message_id_str))
            if not duplicate_ack and row is not None:
                if is_ack:
                    if ack_idx:
                        self._sheets.update_log_row(ack_idx, row)
                        self._state.set_row_idx((chat_id_str, tech_message_id_str), ack_idx)
                    else:
                        maybe_idx = self._sheets.append_log_row(row, return_row_index=True)
                        if maybe_idx:
                            self._state.set_row_idx((chat_id_str, tech_message_id_str), maybe_idx)
                else:
                    if ack_idx:
                        self._sheets.update_log_row(ack_idx, row)
                        self._state.set_row_idx((chat_id_str, tech_message_id_str), ack_idx)
                    else:
                        self._sheets.append_log_row(row)
            
            # Log ke ML_Tracking SETELAH Logs sukses (untuk konsistensi data)
            # Retry 3x dengan delay, notify admin kalau tetap gagal
            if ml_prediction and self._ml_tracking:
                solving_text = parsed["solving"] if parsed else ""
                ml_tracking_success = False
                
                for attempt in range(3):
                    try:
                        self._ml_tracking.log_prediction(
                            tech_message_id=int(tech_message_id_str),
                            tech_raw_text=tech_raw_text,
                            solving=solving_text,
                            prediction_result=ml_prediction,
                        )
                        ml_tracking_success = True
                        break
                    except Exception as e:
                        _LOGGER.warning(
                            "ML_Tracking attempt %d/3 failed: %s", 
                            attempt + 1, str(e)
                        )
                        if attempt < 2:  # Not last attempt
                            import time
                            time.sleep(0.5)  # 500ms delay before retry
                
                if not ml_tracking_success:
                    _LOGGER.exception("ML_Tracking failed after 3 attempts (Logs already saved)")
                    # Notify admin about ML_Tracking failure
                    await self._safe_notify(
                        chat_id=notification_chat_id,
                        text=f"⚠️ ML_Tracking gagal setelah 3x retry.\n"
                             f"Data Logs sudah tersimpan, tapi ML_Tracking perlu dicek manual.\n"
                             f"Message ID: {tech_message_id_str}",
                        reply_to_id=None,
                    )
                    
        except Exception:
            _LOGGER.exception("Failed to append row to Google Sheets")
            await self._safe_notify(
                chat_id=notification_chat_id,
                text="❌ Gagal mencatat laporan ke Google Sheet.",
                reply_to_id=message.message_id if can_reply_in_place else None,
            )
            return

        if media_upload_failed:
            notify_text = f"{notify_text}\nLampiran gagal diunggah, tiket tetap direkap tanpa lampiran."

        if link_text:
            notify_text = f"{notify_text}\nLihat pesan: {link_text}"

        await self._safe_notify(
            chat_id=notification_chat_id,
            text=notify_text,
            reply_to_id=message.message_id if can_reply_in_place else None,
        )

    async def _upload_media_to_s3(
        self,
        *,
        bot: Bot,
        info: MediaInfo,
        chat_id: int,
        message_id: int,
    ) -> str:
        """
        Download media dari Telegram dan upload ke S3/MinIO storage.
        
        Flow:
        1. Tentukan MIME type dan extension file
        2. Generate key path: {YYYY}/{MM}/{DD}/{chat_id}/{message_id}_{timestamp}.{ext}
        3. Download file dari Telegram ke memory
        4. Upload ke S3/MinIO
        5. Return URL publik
        
        Args:
            bot: Telegram Bot instance untuk download file
            info: MediaInfo berisi file_id dan metadata
            chat_id: ID chat untuk path naming
            message_id: ID message untuk path naming
            
        Returns:
            URL publik dari file yang diupload
        """
        mime_type = info.mime_type or _DEFAULT_MIME_BY_TYPE.get(info.media_type, "application/octet-stream")
        extension = _guess_extension(info, mime_type)
        timestamp = datetime.now(timezone.utc)
        key = (
            f"{timestamp:%Y/%m/%d}/"
            f"{chat_id}/{message_id}_{timestamp:%Y%m%dT%H%M%SZ}.{extension}"
        )

        telegram_file = await bot.get_file(info.file_id)
        buffer = BytesIO()
        await telegram_file.download_to_memory(out=buffer)
        return self._s3_uploader.upload_bytes(
            key=key,
            data=buffer.getvalue(),
            content_type=mime_type,
        )

    async def _safe_notify(self, chat_id: int, text: str, reply_to_id: int | None) -> None:
        """
        Kirim notifikasi via reporting bot tanpa crash kalau gagal.
        
        Safety wrapper untuk send_message yang:
        - Skip kalau reporting bot tidak dikonfigurasi
        - Catch exception dan log warning (tidak raise)
        - Gunakan HTML parse mode untuk formatting
        
        Args:
            chat_id: ID chat tujuan notifikasi
            text: Pesan yang akan dikirim (HTML supported)
            reply_to_id: ID message untuk reply (opsional)
        """
        if not self._reporting_bot:
            _LOGGER.debug("Reporting bot not configured, skipping notification")
            return
        
        try:
            await self._reporting_bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_id,
                disable_notification=False,
                parse_mode="HTML",
            )
        except Exception:
            _LOGGER.warning("Reporting bot failed to send notification", exc_info=True)


# ============ Helper Functions ============

def _build_message_link(chat_id: int, message_id: int) -> str | None:
    """
    Buat deep link t.me untuk message di supergroup.
    
    Format output: https://t.me/c/{short_id}/{message_id}
    
    Catatan:
    - Hanya untuk supergroup (chat_id negatif)
    - Private chat tidak support deep link
    - short_id = chat_id tanpa prefix "100"
    
    Args:
        chat_id: ID chat (harus negatif untuk supergroup)
        message_id: ID message yang akan di-link
        
    Returns:
        URL deep link atau None jika bukan supergroup
    """
    if chat_id >= 0:
        return None
    abs_id = str(abs(chat_id))
    if abs_id.startswith("100"):
        short_id = abs_id[3:]
    else:
        short_id = abs_id
    return f"https://t.me/c/{short_id}/{message_id}"


def _extract_media_info(message: Message) -> MediaInfo:
    """
    Extract informasi media dari message teknisi.
    
    Cek secara berurutan: photo -> document -> video -> animation -> audio
    Return MediaInfo kosong jika tidak ada media.
    
    Untuk photo: ambil ukuran terbesar (index -1)
    Untuk document: include file_name original
    
    Args:
        message: Telegram Message object
        
    Returns:
        MediaInfo dengan detail media atau kosong jika tidak ada
    """
    if message.photo:
        return MediaInfo("photo", message.photo[-1].file_id, "image/jpeg", None)
    if message.document:
        return MediaInfo("document", message.document.file_id, message.document.mime_type, message.document.file_name)
    if message.video:
        return MediaInfo("video", message.video.file_id, message.video.mime_type, message.video.file_name)
    if message.animation:
        return MediaInfo("animation", message.animation.file_id, message.animation.mime_type, message.animation.file_name)
    if message.audio:
        return MediaInfo("audio", message.audio.file_id, message.audio.mime_type, message.audio.file_name)
    return MediaInfo("", "", None, None)


def _guess_extension(info: MediaInfo, mime_type: str) -> str:
    """
    Tebak file extension berdasarkan info media.
    
    Prioritas:
    1. Ambil dari file_name asli (kalau ada)
    2. Mapping dari MIME type (image/jpeg -> jpg)
    3. Mapping dari media type (photo -> jpg)
    4. Default: "bin"
    
    Args:
        info: MediaInfo dengan file_name
        mime_type: MIME type file
        
    Returns:
        File extension tanpa titik (contoh: jpg, mp4, pdf)
    """
    if info.file_name:
        suffix = Path(info.file_name).suffix
        if suffix:
            return suffix.lstrip(".").lower()
    return _EXTENSION_BY_MIME.get(mime_type, _EXTENSION_BY_TYPE.get(info.media_type, "bin"))


def _to_utc_datetime(value: datetime | None) -> datetime | None:
    """
    Konversi datetime ke UTC timezone.
    
    Kalau datetime tidak punya timezone info (naive),
    anggap sudah UTC dan tambahkan tzinfo.
    
    Args:
        value: Datetime object atau None
        
    Returns:
        Datetime dalam UTC atau None
    """
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_local_datetime(value: datetime | None, tz: ZoneInfo) -> datetime | None:
    """
    Konversi datetime UTC ke timezone lokal yang dikonfigurasi.
    
    Digunakan untuk formatting waktu yang ditampilkan ke user
    dan disimpan ke Sheets (biasanya Asia/Jakarta).
    
    Args:
        value: Datetime dalam UTC
        tz: Target timezone (contoh: Asia/Jakarta)
        
    Returns:
        Datetime dalam timezone lokal atau None
    """
    if not value:
        return None
    return value.astimezone(tz)


# ============ Application Builder ============

def build_collecting_application(
    config: "Config",
    sheets_client: "GoogleSheetsClient",
    s3_uploader: "S3Uploader",
    ml_classifier: Optional["MLClassifier"] = None,
    ml_tracking: Optional["MLTrackingClient"] = None,
) -> Application:
    """
    Buat dan konfigurasi Telegram Application untuk collecting bot.
    
    Fungsi ini:
    1. Membuat instance OpsCollector dengan semua dependencies
    2. Build Telegram Application dengan token bot
    3. Register handlers:
       - /health, /ping: Health check
       - Text message: handle_ops_reply (proses tiket)
       - Edited message: handle_ops_reply (update tiket)
    
    Args:
        config: Konfigurasi aplikasi
        sheets_client: Client Google Sheets
        s3_uploader: Client S3/MinIO
        ml_classifier: Model ML (opsional)
        ml_tracking: Client ML_Tracking (opsional)
        
    Returns:
        Telegram Application yang siap dijalankan
    """
    collector = OpsCollector(
        config, sheets_client, s3_uploader,
        ml_classifier=ml_classifier,
        ml_tracking=ml_tracking
    )
    
    application = ApplicationBuilder().token(config.telegram_collecting_bot_token).build()
    
    # Health check command
    application.add_handler(CommandHandler(["health", "ping"], collector.health))
    
    # Message handlers
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), collector.handle_ops_reply)
    )
    application.add_handler(
        MessageHandler(
            filters.UpdateType.EDITED_MESSAGE & filters.TEXT & (~filters.COMMAND),
            collector.handle_ops_reply,
        )
    )
    
    return application

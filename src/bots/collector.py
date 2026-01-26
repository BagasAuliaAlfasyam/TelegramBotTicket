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
    """Container for media file information."""
    media_type: str
    file_id: str
    mime_type: str | None
    file_name: str | None


class PersistentState:
    """Lightweight JSON-backed store for ack caches across restarts."""

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
    """Stateful handler that encapsulates collecting logic."""

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
        """Simple health endpoint useful for debugging deployments."""
        if not update.message:
            return
        await update.message.reply_text(f"OK {datetime.now(timezone.utc).isoformat()}")

    async def handle_ops_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process Ops replies, validate format, and log them to Google Sheets."""
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
            
            # Log ke ML_Tracking untuk audit trail (semua prediksi)
            if self._ml_tracking:
                self._ml_tracking.log_prediction(
                    tech_message_id=int(tech_message_id_str),
                    tech_raw_text=tech_raw_text,
                    solving=solving_text,
                    prediction_result=ml_prediction,
                )
            
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
        """Upload media file to S3 and return URL."""
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
        """Send a notification using the reporting bot without crashing on failure."""
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
    """Build a t.me deep link for supergroups when possible."""
    if chat_id >= 0:
        return None
    abs_id = str(abs(chat_id))
    if abs_id.startswith("100"):
        short_id = abs_id[3:]
    else:
        short_id = abs_id
    return f"https://t.me/c/{short_id}/{message_id}"


def _extract_media_info(message: Message) -> MediaInfo:
    """Derive media details from a technician message."""
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
    """Guess file extension from media info."""
    if info.file_name:
        suffix = Path(info.file_name).suffix
        if suffix:
            return suffix.lstrip(".").lower()
    return _EXTENSION_BY_MIME.get(mime_type, _EXTENSION_BY_TYPE.get(info.media_type, "bin"))


def _to_utc_datetime(value: datetime | None) -> datetime | None:
    """Normalize a datetime to UTC."""
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_local_datetime(value: datetime | None, tz: ZoneInfo) -> datetime | None:
    """Convert a UTC datetime to the configured local timezone."""
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
    """Wire handlers into the telegram Application instance."""
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

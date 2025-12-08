"""Telegram bot logic for collecting Ops replies and logging them to Google Sheets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from telegram import Bot, Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Config
from google_sheets_client import GoogleSheetsClient
from ops_parser import parse_ops_message
from s3_uploader import S3Uploader

_LOGGER = logging.getLogger(__name__)
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
    media_type: str
    file_id: str
    mime_type: str | None
    file_name: str | None


class OpsCollector:
    """Stateful handler that encapsulates collecting logic."""

    def __init__(
        self,
        config: Config,
        sheets_client: GoogleSheetsClient,
        s3_uploader: S3Uploader,
    ) -> None:
        self._config = config
        self._sheets = sheets_client
        self._s3_uploader = s3_uploader
        self._reporting_bot = Bot(token=config.telegram_bot_token_reporting)
        self._ack_cache: dict[tuple[str, str], datetime] = {}
        self._ack_row_index: dict[tuple[str, str], int] = {}

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Simple health endpoint useful for debugging deployments."""
        if not update.message:
            return
        await update.message.reply_text(f"OK {datetime.now(timezone.utc).isoformat()}")

    async def handle_ops_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process Ops replies, validate format, and log them to Google Sheets."""

        message = update.message
        if not message or not message.text or not message.reply_to_message:
            return

        if self._config.target_group_collecting and (
            message.chat.id != self._config.target_group_collecting
        ):
            return

        is_ack = "oncek" in message.text.lower()
        parsed = parse_ops_message(message.text, _ALLOWED_APPS) if not is_ack else None
        if not parsed and not is_ack:
            return

        tech_message = message.reply_to_message
        media_info = _extract_media_info(tech_message)
        tech_message_date = ""
        tech_message_time = ""
        tech_message_dt = _to_utc_datetime(tech_message.date)
        if tech_message_dt:
            tech_message_date = tech_message_dt.date().isoformat()
            tech_message_time = tech_message_dt.strftime("%H:%M:%S")

        now_dt = _to_utc_datetime(message.date) or datetime.now(timezone.utc)
        ticket_date = now_dt.date().isoformat()
        response_at = now_dt.strftime("%H:%M:%S")
        tech_raw_text = (
            tech_message.caption
            or tech_message.text
            or ""
        )

        media_url = ""
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
                notification_chat_id = self._config.target_group_reporting or message.chat.id
                can_reply_in_place = notification_chat_id == message.chat.id
                await self._safe_notify(
                    chat_id=notification_chat_id,
                    text="❌ Gagal mengunggah lampiran, tiket tidak direkap.",
                    reply_to_id=message.message_id if can_reply_in_place else None,
                )
                return

        chat_id_str = str(message.chat.id)
        group_label = message.chat.title or message.chat.username or chat_id_str
        tech_message_id_str = str(tech_message.message_id)
        notify_text = ''
        ack_key = (chat_id_str, tech_message_id_str)
        ack_idx = self._ack_row_index.get(ack_key)
        duplicate_ack = False
        row: list[str | int | float] | None = None

        if is_ack:
            existing_ack_dt = self._ack_cache.get(ack_key)
            if existing_ack_dt or ack_idx:
                duplicate_ack = True
                notify_text = f"Tiket sudah dicek (oncek) untuk grup: {group_label}, belum ada solusi."
            else:
                ack_dt = now_dt
                self._ack_cache[ack_key] = ack_dt
                sla_response_min, sla_status, sla_remaining_min = _compute_sla(tech_message_dt, ack_dt)
                row = [
                    group_label,        # Informasi Tiket
                    ticket_date,
                    response_at,
                    tech_message_id_str,  # Informasi Teknisi
                    tech_message_date,
                    tech_message_time,
                    tech_raw_text,
                    media_info.media_type,
                    media_url,
                    '',  # ops_message_id
                    '',  # ops_message_text
                    '',  # solving
                    '',  # solve_timestamp
                    '',  # app
                    '',  # solver_name
                    'true',  # isOncek
                    sla_response_min,
                    sla_status,
                    sla_remaining_min,
                ]
                notify_text = f"Respon awal (oncek) sudah dicatat untuk grup: {group_label}."
        else:
            solver_key = parsed["initials"].lower()
            solver_name = _SOLVER_NAME_MAP.get(solver_key)
            if not solver_name:
                _LOGGER.info("Ignoring ops reply from unknown solver '%s'", parsed["initials"])
                return

            ack_dt = self._ack_cache.get(ack_key, None)
            sla_response_min, sla_status, sla_remaining_min = _compute_sla(tech_message_dt, ack_dt or now_dt)
            solve_timestamp = now_dt.strftime("%H:%M:%S")
            is_oncek_flag = 'true' if ack_dt or self._ack_row_index.get(ack_key) else 'false'
            row = [
                group_label,
                ticket_date,
                response_at,
                tech_message_id_str,
                tech_message_date,
                tech_message_time,
                tech_raw_text,
                media_info.media_type,
                media_url,
                str(message.message_id),
                message.text,
                parsed["solving"],
                solve_timestamp,
                parsed["app"],
                solver_name,
                is_oncek_flag,
                sla_response_min,
                sla_status,
                sla_remaining_min,
            ]
            notify_text = f"Laporan dan solusi sudah dicatat oleh {solver_name} untuk grup: {group_label}."
        notification_chat_id = self._config.target_group_reporting or message.chat.id
        can_reply_in_place = notification_chat_id == message.chat.id

        try:
            ack_idx = self._ack_row_index.get((chat_id_str, tech_message_id_str))
            if not duplicate_ack and row is not None:
                if is_ack:
                    if ack_idx:
                        self._sheets.update_log_row(ack_idx, row)
                    else:
                        maybe_idx = self._sheets.append_log_row(row, return_row_index=True)
                        if maybe_idx:
                            self._ack_row_index[(chat_id_str, tech_message_id_str)] = maybe_idx
                else:
                    if ack_idx:
                        self._sheets.update_log_row(ack_idx, row)
                    else:
                        self._sheets.append_log_row(row)
        except Exception:  # pragma: no cover - log and notify failure
            _LOGGER.exception("Failed to append row to Google Sheets")
            await self._safe_notify(
                chat_id=notification_chat_id,
                text="❌ Gagal mencatat laporan ke Google Sheet.",
                reply_to_id=message.message_id if can_reply_in_place else None,
            )
            return

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
        try:
            await self._reporting_bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_id,
                disable_notification=False,
            )
        except Exception:  # pragma: no cover - avoid crashing bot when notify fails
            _LOGGER.warning("Reporting bot failed to send notification", exc_info=True)


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


def _compute_sla(tech_dt: datetime | None, response_dt: datetime | None) -> tuple[str | float, str, str | float]:
    """Compute SLA metrics between technician message and response time."""
    if not tech_dt or not response_dt:
        return "", "", ""
    delta_seconds = max((response_dt - tech_dt).total_seconds(), 0)
    minutes = round(delta_seconds / 60, 2)
    remaining = round(max(15 - minutes, 0), 2)
    status = "OK" if minutes <= 15 else "TERLAMBAT"
    return minutes, status, remaining


def build_collecting_application(
    config: Config,
    sheets_client: GoogleSheetsClient,
    s3_uploader: S3Uploader,
) -> Application:
    """Wire handlers into the telegram Application instance."""

    collector = OpsCollector(config, sheets_client, s3_uploader)
    application = ApplicationBuilder().token(config.telegram_bot_token_collecting).build()
    application.add_handler(CommandHandler(["health", "ping"], collector.health))
    application.add_handler(
        MessageHandler(filters.TEXT & (~filters.COMMAND), collector.handle_ops_reply)
    )
    return application

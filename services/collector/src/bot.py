"""
Collector Bot — Microservice Version
======================================
Telegram bot that:
  - Collects Ops replies
  - Calls Prediction API for ML classification (LightGBM + Gemini cascade)
  - Calls Data API for Sheets CRUD + S3 upload + ML Tracking
  - Decoupled from any direct database/sheets/ML access
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
from telegram import Bot, Message, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.collector.src.parsers import parse_ops_message
from services.collector.src.sla import compute_sla
from services.shared.config import CollectorBotConfig, setup_logging

_LOGGER = logging.getLogger(__name__)

_ALLOWED_APPS = {"MIT", "MIS"}
_SOLVER_NAME_MAP = {"-bg": "Bagas", "-dm": "Damas", "-dvd": "David", "-fr": "Fairuz"}
_DEFAULT_MIME_BY_TYPE = {
    "photo": "image/jpeg", "document": "application/octet-stream",
    "video": "video/mp4", "animation": "video/mp4", "audio": "audio/mpeg",
}
_EXTENSION_BY_MIME = {
    "image/jpeg": "jpg", "image/png": "png", "video/mp4": "mp4",
    "video/quicktime": "mov", "application/pdf": "pdf", "audio/mpeg": "mp3",
}
_EXTENSION_BY_TYPE = {"photo": "jpg", "document": "bin", "video": "mp4", "animation": "mp4", "audio": "mp3"}


@dataclass
class MediaInfo:
    media_type: str
    file_id: str
    mime_type: str | None
    file_name: str | None


class PersistentState:
    """JSON-backed cache for oncek timestamps and row indices."""

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
            return
        for k, ts in raw.get("ack_cache", {}).items():
            parts = k.split("::", 1)
            if len(parts) == 2:
                try:
                    self._ack_cache[(parts[0], parts[1])] = datetime.fromisoformat(ts)
                except Exception:
                    pass
        for k, idx in raw.get("ack_row_index", {}).items():
            parts = k.split("::", 1)
            if len(parts) == 2 and isinstance(idx, int):
                self._ack_row_index[(parts[0], parts[1])] = idx

    def _dump(self) -> None:
        try:
            data = {
                "ack_cache": {f"{k[0]}::{k[1]}": v.isoformat() for k, v in self._ack_cache.items()},
                "ack_row_index": {f"{k[0]}::{k[1]}": v for k, v in self._ack_row_index.items()},
            }
            self._path.write_text(json.dumps(data))
        except Exception:
            _LOGGER.warning("Failed to persist state", exc_info=True)

    def get_ack_dt(self, key):
        return self._ack_cache.get(key)

    def set_ack_dt(self, key, value):
        self._ack_cache[key] = value
        self._dump()

    def get_row_idx(self, key):
        return self._ack_row_index.get(key)

    def set_row_idx(self, key, value):
        self._ack_row_index[key] = value
        self._dump()


class OpsCollector:
    """
    Microservice version of OpsCollector.
    Instead of direct Sheets/ML calls, uses HTTP clients to call:
      - Prediction API (POST /predict)
      - Data API (POST /logs/append, PUT /logs/{row}, POST /tracking/log, POST /media/upload)
    """

    def __init__(self, config: CollectorBotConfig) -> None:
        self._config = config
        self._http = httpx.AsyncClient(timeout=30.0)
        self._prediction_url = config.prediction_api_url.rstrip("/")
        self._data_url = config.data_api_url.rstrip("/")

        if config.telegram_reporting_bot_token:
            self._reporting_bot = Bot(token=config.telegram_reporting_bot_token)
        else:
            self._reporting_bot = None

        self._state = PersistentState(Path("/app/state/state_cache.json"))
        try:
            self._tz = ZoneInfo(config.timezone)
        except Exception:
            self._tz = UTC

    async def health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message:
            await update.message.reply_text(f"OK {datetime.now(UTC).isoformat()}")

    async def handle_ops_reply(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text or not message.reply_to_message:
            return

        sender_id = message.from_user.id if message.from_user else None
        if self._config.admin_user_ids and (
            sender_id is None or sender_id not in self._config.admin_user_ids
        ):
            return

        now_dt = _to_utc(message.date) or datetime.now(UTC)
        is_ack = "oncek" in message.text.lower()
        parsed = parse_ops_message(message.text, _ALLOWED_APPS) if not is_ack else None
        if not parsed and not is_ack:
            return

        tech_message = message.reply_to_message
        media_info = _extract_media_info(tech_message)
        tech_message_dt = _to_utc(tech_message.date)

        if getattr(tech_message, "forward_date", None) or \
           getattr(tech_message, "forward_origin", None) or \
           getattr(tech_message, "is_automatic_forward", False):
            tech_message_dt = now_dt

        tech_local = _to_local(tech_message_dt, self._tz)
        now_local = _to_local(now_dt, self._tz)
        tech_raw_text = tech_message.caption or tech_message.text or ""

        # --- Media upload via Data API ---
        media_url = ""
        media_upload_failed = False
        if media_info.file_id and media_info.media_type:
            try:
                media_url = await self._upload_media(context.bot, media_info, message.chat.id, tech_message.message_id)
            except Exception:
                _LOGGER.exception("Media upload failed")
                media_upload_failed = True

        chat_id_str = str(message.chat.id)
        group_label = message.chat.title or message.chat.username or chat_id_str
        tech_mid = str(tech_message.message_id)
        ack_key = (chat_id_str, tech_mid)
        ack_idx = self._state.get_row_idx(ack_key)

        if not ack_idx:
            try:
                resp = await self._http.get(f"{self._data_url}/logs/find/{tech_mid}")
                data = resp.json()
                if data.get("found"):
                    ack_idx = data["row_index"]
                    self._state.set_row_idx(ack_key, ack_idx)
            except Exception:
                pass

        notify_text = ""
        row_data = None

        if is_ack:
            existing_ack_dt = self._state.get_ack_dt(ack_key)
            ack_dt = existing_ack_dt or now_dt
            self._state.set_ack_dt(ack_key, ack_dt)
            sla_min, sla_status, sla_remain = compute_sla(tech_message_dt, ack_dt, self._tz)
            row_data = self._build_row(
                group_label, now_local, tech_mid, tech_local, tech_raw_text,
                media_info, media_url, "", "", "", "", "", "true",
                sla_min, sla_status, sla_remain, "",
            )
            notify_text = f"Respon awal (oncek) dicatat untuk grup: {group_label}."
        else:
            solver_key = parsed["initials"].lower()
            solver_name = _SOLVER_NAME_MAP.get(solver_key)
            if not solver_name:
                return

            ack_dt = self._state.get_ack_dt(ack_key)
            sla_min, sla_status, sla_remain = compute_sla(tech_message_dt, ack_dt or now_dt, self._tz)
            solve_ts = now_local.strftime("%H:%M:%S") if now_local else ""
            is_oncek = "true" if ack_dt or ack_idx else "false"

            # --- ML Prediction via Prediction API ---
            symtomps_label = ""
            ml_source = "lightgbm"
            ml_confidence = 0.0
            ml_predicted = ""
            ml_status = "MANUAL"

            try:
                resp = await self._http.post(
                    f"{self._prediction_url}/predict",
                    json={"tech_raw_text": tech_raw_text, "solving": parsed["solving"]},
                    timeout=15.0,
                )
                if resp.status_code == 200:
                    pred = resp.json()
                    ml_predicted = pred.get("predicted_symtomps", "")
                    ml_confidence = pred.get("ml_confidence", 0.0)
                    ml_source = pred.get("source", "lightgbm")
                    ml_status = pred.get("prediction_status", "MANUAL")
                    if ml_confidence >= 0.80:
                        symtomps_label = ml_predicted
            except Exception:
                _LOGGER.warning("Prediction API unavailable, continuing without ML")

            row_data = self._build_row(
                group_label, now_local, tech_mid, tech_local, tech_raw_text,
                media_info, media_url,
                str(message.message_id), message.text, parsed["solving"],
                solve_ts, parsed["app"], solver_name,
                is_oncek, sla_min, sla_status, sla_remain, symtomps_label,
            )
            notify_text = f"Laporan dicatat oleh {solver_name} untuk grup: {group_label}."

            if ml_predicted:
                pct = ml_confidence * 100
                if ml_confidence >= 0.80:
                    notify_text += f"\n\n🤖 Symtomps: <b>{ml_predicted}</b> ({pct:.0f}%)"
                    if ml_source in ("gemini", "hybrid"):
                        notify_text += f" [{ml_source}]"
                else:
                    notify_text += f"\n\n🤖 Symtomps: <b>{ml_predicted}</b> ({pct:.0f}%) ⚠️ <i>perlu review</i>"

        # --- Save to Data API ---
        notification_chat_id = self._config.telegram_admin_chat_id or message.chat.id
        can_reply = notification_chat_id == message.chat.id

        try:
            if row_data is not None:
                if ack_idx:
                    await self._http.put(f"{self._data_url}/logs/{ack_idx}", json=row_data)
                else:
                    resp = await self._http.post(f"{self._data_url}/logs/append", json=row_data)
                    if resp.status_code == 200:
                        new_idx = resp.json().get("row_index")
                        if new_idx:
                            self._state.set_row_idx(ack_key, new_idx)

            # Log ML prediction to tracking (after logs success)
            if not is_ack and ml_predicted:
                for attempt in range(3):
                    try:
                        await self._http.post(f"{self._data_url}/tracking/log", json={
                            "tech_message_id": int(tech_mid),
                            "tech_raw_text": tech_raw_text,
                            "solving": parsed["solving"] if parsed else "",
                            "predicted_symtomps": ml_predicted,
                            "ml_confidence": ml_confidence,
                            "prediction_status": ml_status,
                            "source": ml_source,
                        })
                        break
                    except Exception:
                        if attempt < 2:
                            await asyncio.sleep(0.5)
                        else:
                            _LOGGER.exception("ML_Tracking failed after 3 attempts")

        except Exception:
            _LOGGER.exception("Failed to save to Data API")
            await self._safe_notify(notification_chat_id, "❌ Gagal menyimpan ke Data API.", message.message_id if can_reply else None)
            return

        if media_upload_failed:
            notify_text += "\nLampiran gagal diunggah, tiket tetap direkap tanpa lampiran."

        if not can_reply:
            link = _build_message_link(message.chat.id, message.message_id)
            if link:
                notify_text += f"\nLihat pesan: {link}"

        await self._safe_notify(
            notification_chat_id, notify_text,
            message.message_id if can_reply else None,
        )

    def _build_row(self, group_label, now_local, tech_mid, tech_local, tech_raw_text,
                   media_info, media_url, ops_mid, ops_text, solving,
                   solve_ts, app_code, solver_name, is_oncek,
                   sla_min, sla_status, sla_remain, symtomps):
        return {
            "group_label": group_label,
            "ticket_date": now_local.date().isoformat() if now_local else "",
            "response_at": now_local.strftime("%H:%M:%S") if now_local else "",
            "tech_message_id": tech_mid,
            "tech_message_date": tech_local.date().isoformat() if tech_local else "",
            "tech_message_time": tech_local.strftime("%H:%M:%S") if tech_local else "",
            "tech_raw_text": tech_raw_text,
            "media_type": media_info.media_type,
            "media_url": media_url,
            "ops_message_id": ops_mid,
            "ops_text": ops_text,
            "solving": solving,
            "solve_timestamp": solve_ts,
            "app_code": app_code if isinstance(app_code, str) else "",
            "solver_name": solver_name if isinstance(solver_name, str) else "",
            "is_oncek": is_oncek,
            "sla_response_min": sla_min,
            "sla_status": sla_status,
            "sla_remaining_min": sla_remain,
            "symtomps": symtomps,
        }

    async def _upload_media(self, bot: Bot, info: MediaInfo, chat_id: int, message_id: int) -> str:
        """Download from Telegram, upload via Data API."""
        mime = info.mime_type or _DEFAULT_MIME_BY_TYPE.get(info.media_type, "application/octet-stream")
        ext = _guess_extension(info, mime)
        ts = datetime.now(UTC)
        key = f"{ts:%Y/%m/%d}/{chat_id}/{message_id}_{ts:%Y%m%dT%H%M%SZ}.{ext}"

        tg_file = await bot.get_file(info.file_id)
        buf = BytesIO()
        await tg_file.download_to_memory(out=buf)

        resp = await self._http.post(
            f"{self._data_url}/media/upload",
            files={"file": (f"media.{ext}", buf.getvalue(), mime)},
            data={"key": key},
        )
        resp.raise_for_status()
        return resp.json().get("url", "")

    async def _safe_notify(self, chat_id, text, reply_to_id=None):
        if not self._reporting_bot:
            return
        try:
            await self._reporting_bot.send_message(
                chat_id=chat_id, text=text,
                reply_to_message_id=reply_to_id,
                parse_mode="HTML",
            )
        except Exception:
            _LOGGER.warning("Notification failed", exc_info=True)


# ---- Helpers ----

def _extract_media_info(msg: Message) -> MediaInfo:
    if msg.photo:
        return MediaInfo("photo", msg.photo[-1].file_id, "image/jpeg", None)
    if msg.document:
        return MediaInfo("document", msg.document.file_id, msg.document.mime_type, msg.document.file_name)
    if msg.video:
        return MediaInfo("video", msg.video.file_id, msg.video.mime_type, msg.video.file_name)
    if msg.animation:
        return MediaInfo("animation", msg.animation.file_id, msg.animation.mime_type, msg.animation.file_name)
    if msg.audio:
        return MediaInfo("audio", msg.audio.file_id, msg.audio.mime_type, msg.audio.file_name)
    return MediaInfo("", "", None, None)


def _guess_extension(info: MediaInfo, mime: str) -> str:
    if info.file_name:
        s = Path(info.file_name).suffix
        if s:
            return s.lstrip(".").lower()
    return _EXTENSION_BY_MIME.get(mime, _EXTENSION_BY_TYPE.get(info.media_type, "bin"))


def _to_utc(v):
    if not v:
        return None
    if v.tzinfo is None:
        return v.replace(tzinfo=UTC)
    return v.astimezone(UTC)


def _to_local(v, tz):
    return v.astimezone(tz) if v else None


def _build_message_link(chat_id: int, message_id: int):
    if chat_id >= 0:
        return None
    a = str(abs(chat_id))
    short = a[3:] if a.startswith("100") else a
    return f"https://t.me/c/{short}/{message_id}"

"""
Data & Analytics API — FastAPI Service
========================================
Centralized data access for Google Sheets, S3, and ML Tracking.

All services access data through this API instead of directly.
This enables future migration from Google Sheets to PostgreSQL.

Endpoints:
    POST   /logs/append        - Append row to Logs sheet (202 jika DLQ)
    PUT    /logs/{row_index}   - Update existing row
    GET    /logs/find/{id}     - Find row by tech_message_id
    GET    /logs/row/{idx}     - Get specific row
    POST   /tracking/log       - Log ML prediction to ML_Tracking
    GET    /stats/realtime     - Realtime stats from ML_Tracking
    GET    /stats/weekly       - Weekly aggregated stats
    GET    /stats/monthly      - Monthly aggregated stats
    POST   /stats/hourly       - Trigger hourly stats calculation
    GET    /training/data      - Get training data (Logs + ML_Tracking)
    POST   /training/mark      - Mark reviewed data as TRAINED
    POST   /media/upload       - Upload media to S3
    GET    /queue/status       - Status antrian DLQ
    GET    /health             - Health check
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from services.data.src.sheets import GoogleSheetsClient
from services.data.src.storage import S3Uploader
from services.data.src.tracking import MLTrackingClient
from services.shared.config import DataServiceConfig, setup_logging, trace_id_var
from services.shared.models import (
    FindRowResponse,
    HealthResponse,
    LogRowRequest,
    TrackingLogRequest,
    TrainingDataResponse,
    TrainingMarkRequest,
    TrainingMarkResponse,
    UploadMediaResponse,
)
from services.shared.telegram_alerter import TelegramAlerter

_LOGGER = logging.getLogger(__name__)

config = DataServiceConfig.from_env()
setup_logging(config.debug, "data-api")

# ============ Global Clients ============
sheets_client: GoogleSheetsClient | None = None
tracking_client: MLTrackingClient | None = None
s3_uploader: S3Uploader | None = None
alerter: TelegramAlerter | None = None

# Persistent prediction gauge — restored from Google Sheets on startup.
_predictions_gauge = Gauge(
    "ticket_predictions_persisted_total",
    "Total ML predictions ever logged (restored from ML_Tracking sheet on startup)",
    ["status"],
)
_label_gauge = Gauge(
    "ticket_predictions_by_label_total",
    "Total ML predictions per label (restored from ML_Tracking sheet on startup)",
    ["label"],
)

# ============ Dead Letter Queue ============

_DLQ_LOCK = asyncio.Lock()
_DLQ_RETRY_INTERVAL = 30   # detik antar retry worker
_DLQ_MAX_BATCH = 50         # maks item di-replay per siklus


def _dlq_load() -> list[dict]:
    """Baca DLQ dari file JSON. Return list kosong jika belum ada."""
    path: Path = config.dlq_path
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOGGER.warning("DLQ file corrupt, reset: %s", exc)
        return []


def _dlq_save(items: list[dict]) -> None:
    """Tulis DLQ ke file JSON (atomic via tmp file)."""
    path: Path = config.dlq_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        _LOGGER.error("Failed to persist DLQ: %s", exc)


async def dlq_enqueue(row_data: dict[str, Any]) -> int:
    """Masukkan satu item ke DLQ (thread-safe). Return ukuran antrian setelah insert."""
    async with _DLQ_LOCK:
        items = _dlq_load()
        items.append({
            "id": uuid.uuid4().hex,
            "queued_at": datetime.now(UTC).isoformat(),
            "row_data": row_data,
        })
        _dlq_save(items)
        size = len(items)
    _LOGGER.warning("DLQ enqueued, queue_size=%d", size)
    return size


async def dlq_size() -> int:
    async with _DLQ_LOCK:
        return len(_dlq_load())


async def _dlq_worker() -> None:
    """
    Background task: tiap 30 detik coba replay semua item di DLQ ke Sheets.
    Jika berhasil → kirim notif sukses ke Telegram dan hapus dari antrian.
    """
    _LOGGER.info("DLQ worker started (interval=%ds)", _DLQ_RETRY_INTERVAL)
    while True:
        await asyncio.sleep(_DLQ_RETRY_INTERVAL)

        async with _DLQ_LOCK:
            items = _dlq_load()

        if not items:
            continue

        if not sheets_client:
            _LOGGER.warning("DLQ worker: Sheets client not ready, skip replay")
            continue

        _LOGGER.info("DLQ worker: attempting replay of %d item(s)", len(items))
        batch = items[:_DLQ_MAX_BATCH]
        replayed: list[str] = []
        failed: list[dict] = []

        for item in batch:
            try:
                row_data: dict = item["row_data"]
                row = [
                    row_data.get("group_label", ""),
                    row_data.get("ticket_date", ""),
                    row_data.get("response_at", ""),
                    row_data.get("tech_message_id", ""),
                    row_data.get("tech_message_date", ""),
                    row_data.get("tech_message_time", ""),
                    row_data.get("tech_raw_text", ""),
                    row_data.get("media_type", ""),
                    row_data.get("media_url", ""),
                    row_data.get("ops_message_id", ""),
                    row_data.get("ops_text", ""),
                    row_data.get("solving", ""),
                    row_data.get("solve_timestamp", ""),
                    row_data.get("app_code", ""),
                    row_data.get("solver_name", ""),
                    row_data.get("is_oncek", ""),
                    row_data.get("sla_response_min", ""),
                    row_data.get("sla_status", ""),
                    row_data.get("sla_remaining_min", ""),
                    row_data.get("symtomps", ""),
                ]
                # Jalankan sinkron di executor agar tidak blok event loop
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda r=row: sheets_client.append_log_row(r)
                )
                replayed.append(item["id"])
                _LOGGER.info("DLQ replayed item id=%s", item["id"])
            except Exception as exc:
                _LOGGER.warning("DLQ replay failed for item id=%s: %s", item["id"], exc)
                failed.append(item)
                # Jika Sheets masih down, hentikan batch ini agar tidak spam retry
                break

        if not replayed:
            continue

        # Hapus yang berhasil dari DLQ
        replayed_set = set(replayed)
        remaining = [i for i in items if i["id"] not in replayed_set]
        async with _DLQ_LOCK:
            _dlq_save(remaining)

        # Kirim notif sukses ke Telegram
        count = len(replayed)
        _LOGGER.info("DLQ replay success: %d item(s) restored to Sheets", count)
        if alerter and alerter.is_ready:
            msg = (
                f"✅ *[Data API] Antrian Berhasil Disimpan*\n\n"
                f"{count} tiket dari antrian berhasil disimpan ke Google Sheets.\n"
                f"Sisa antrian: {len(remaining)} tiket."
            )
            alerter._send_with_cooldown("dlq_replay_success", msg)


# ============ Lifespan ============

@asynccontextmanager
async def lifespan(application: FastAPI):
    global sheets_client, tracking_client, s3_uploader, alerter

    _LOGGER.info("Starting Data API...")

    # Telegram Alerter untuk notif DLQ
    if config.telegram_bot_token_reporting and config.telegram_admin_chat_id:
        alerter = TelegramAlerter(
            bot_token=config.telegram_bot_token_reporting,
            chat_ids=[config.telegram_admin_chat_id],
            cooldown_seconds=60,  # notif DLQ cooldown 1 menit
        )
        _LOGGER.info("TelegramAlerter ready for DLQ notifications (chat_id=%d)", config.telegram_admin_chat_id)
    else:
        _LOGGER.warning("TELEGRAM_BOT_TOKEN_REPORTING / TELEGRAM_ADMIN_CHAT_ID not set — DLQ Telegram alerts disabled")

    # Google Sheets — graceful startup
    try:
        sheets_client = GoogleSheetsClient(config)
        _LOGGER.info("Google Sheets connected: %s", config.google_spreadsheet_name)
    except Exception as e:
        _LOGGER.warning(
            "Google Sheets unavailable at startup (endpoints will return 503): %s. "
            "Ensure spreadsheet '%s' is shared with the service account in service_account.json.",
            e,
            config.google_spreadsheet_name,
        )
        sheets_client = None

    try:
        tracking_client = MLTrackingClient(
            config, spreadsheet=sheets_client.spreadsheet if sheets_client else None
        )
    except Exception as e:
        _LOGGER.warning("MLTrackingClient init failed: %s", e)
        tracking_client = None

    s3_uploader = S3Uploader(config)

    # Restore prediction gauge
    try:
        if tracking_client:
            counts = tracking_client.get_prediction_counts_by_status()
            for status, count in counts.items():
                _predictions_gauge.labels(status=status).set(count)
            label_counts = tracking_client.get_prediction_counts_by_label()
            for label, count in label_counts.items():
                _label_gauge.labels(label=label).set(count)
            total = sum(counts.values())
            _LOGGER.info("Restored predictions gauge: total=%s breakdown=%s", total, counts)
    except Exception as e:
        _LOGGER.warning("Could not restore predictions gauge from sheet: %s", e)

    # Cek DLQ saat startup — kalau ada sisa, log warning
    existing_dlq = _dlq_load()
    if existing_dlq:
        _LOGGER.warning(
            "DLQ has %d item(s) from previous session — worker will replay them",
            len(existing_dlq),
        )

    # Start background worker
    worker_task = asyncio.create_task(_dlq_worker(), name="dlq-worker")

    sheets_status = "connected" if sheets_client else "UNAVAILABLE"
    _LOGGER.info("Data API ready — Sheets: %s [%s]", config.google_spreadsheet_name, sheets_status)

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    _LOGGER.info("Data API shutting down")


# ============ App ============

app = FastAPI(
    title="Ticket Classifier - Data API",
    description="Centralized data access for Google Sheets, S3, and ML Tracking",
    version="2.1.0",
    root_path="/data",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def _trace_middleware(request: Request, call_next):
    """Attach X-Trace-ID to every request and propagate trace_id into log records."""
    trace_id = request.headers.get("X-Trace-ID") or uuid.uuid4().hex[:8]
    token = trace_id_var.set(trace_id)
    try:
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        trace_id_var.reset(token)


# ============ Logs Endpoints ============

@app.get("/logs/all")
async def logs_all():
    """Get all rows from Logs sheet (headers + data)."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")
    try:
        data = sheets_client.get_all_logs_data()
        return {"rows": data, "total": max(len(data) - 1, 0)}
    except Exception as e:
        _LOGGER.exception("Failed to get all logs: %s", e)
        raise HTTPException(500, f"Failed: {e}")


@app.post("/logs/append", status_code=200)
async def logs_append(request: LogRowRequest):
    """
    Append a log row to the Logs sheet.

    Returns:
      - 200: berhasil disimpan ke Sheets
      - 202: Sheets down → data masuk DLQ, akan di-retry otomatis
      - 503: Sheets client belum siap
    """
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    row = [
        request.group_label, request.ticket_date, request.response_at,
        request.tech_message_id, request.tech_message_date, request.tech_message_time,
        request.tech_raw_text, request.media_type, request.media_url,
        request.ops_message_id, request.ops_text, request.solving,
        request.solve_timestamp, request.app_code, request.solver_name,
        request.is_oncek, request.sla_response_min, request.sla_status,
        request.sla_remaining_min, request.symtomps,
    ]

    try:
        # Run sync sheets call di thread executor agar tidak blok event loop
        row_index = await asyncio.get_event_loop().run_in_executor(
            None, lambda: sheets_client.append_log_row(row, return_row_index=True)
        )
        return {"success": True, "row_index": row_index, "queued": False}

    except Exception as e:
        _LOGGER.exception("Failed to append log after retries: %s", e)

        # Masukkan ke DLQ
        row_data = request.model_dump()
        queue_size = await dlq_enqueue(row_data)

        # Kirim alert ke Telegram
        if alerter and alerter.is_ready:
            msg = (
                f"⚠️ *[Data API] Data Dimasukkan ke Antrian*\n\n"
                f"Google Sheets tidak dapat dijangkau (error: `{type(e).__name__}`).\n"
                f"Data tiket *tech\\_id={request.tech_message_id}* telah diamankan di antrian.\n\n"
                f"📥 *Ukuran antrian saat ini: {queue_size} tiket*\n"
                f"🔄 Akan otomatis disimpan ke Sheets saat koneksi pulih."
            )
            alerter._send_with_cooldown(f"dlq_enqueue_{queue_size}", msg)

        # Return 202 Accepted — bukan error, data aman di DLQ
        return JSONResponse(
            status_code=202,
            content={
                "success": False,
                "queued": True,
                "queue_size": queue_size,
                "detail": "Sheets unavailable — data queued for retry",
            },
        )


@app.put("/logs/{row_index}")
async def logs_update(row_index: int, request: LogRowRequest):
    """Update an existing row in the Logs sheet."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    row = [
        request.group_label, request.ticket_date, request.response_at,
        request.tech_message_id, request.tech_message_date, request.tech_message_time,
        request.tech_raw_text, request.media_type, request.media_url,
        request.ops_message_id, request.ops_text, request.solving,
        request.solve_timestamp, request.app_code, request.solver_name,
        request.is_oncek, request.sla_response_min, request.sla_status,
        request.sla_remaining_min, request.symtomps,
    ]

    try:
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: sheets_client.update_log_row(row_index, row)
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to update: {e}")


@app.get("/logs/find/{tech_message_id}")
async def logs_find(tech_message_id: str):
    """Find row index by tech_message_id."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    idx = await asyncio.get_event_loop().run_in_executor(
        None, lambda: sheets_client.find_row_index_by_tech_message_id(tech_message_id)
    )
    return FindRowResponse(row_index=idx, found=idx is not None)


@app.get("/logs/row/{row_index}")
async def logs_get_row(row_index: int):
    """Get a specific row by index."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    row = await asyncio.get_event_loop().run_in_executor(
        None, lambda: sheets_client.get_row(row_index)
    )
    return {"row": row, "found": bool(row)}


# ============ ML Tracking Endpoints ============

@app.post("/tracking/log")
async def tracking_log(request: TrackingLogRequest):
    """Log ML prediction to ML_Tracking sheet."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")

    try:
        tracking_client.log_prediction(
            tech_message_id=request.tech_message_id,
            tech_raw_text=request.tech_raw_text,
            solving=request.solving,
            predicted_symtomps=request.predicted_symtomps,
            ml_confidence=request.ml_confidence,
            prediction_status=request.prediction_status,
            source=request.source,
        )
        _predictions_gauge.labels(status=request.prediction_status).inc()
        if request.predicted_symtomps:
            _label_gauge.labels(label=request.predicted_symtomps).inc()
            _LOGGER.info("Label gauge incremented: label=%s", request.predicted_symtomps)
        else:
            _LOGGER.warning("tracking/log received empty predicted_symtomps (status=%s)", request.prediction_status)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to log: {e}")


# ============ Stats Endpoints ============

@app.get("/stats/realtime")
async def stats_realtime():
    """Get realtime stats from ML_Tracking."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    return tracking_client.get_realtime_stats()


@app.get("/stats/weekly")
async def stats_weekly():
    """Get weekly aggregated stats."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    try:
        return tracking_client.get_weekly_stats() or {}
    except Exception as e:
        _LOGGER.error("stats_weekly error: %s", e)
        return {}


@app.get("/stats/monthly")
async def stats_monthly():
    """Get monthly aggregated stats."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    try:
        return tracking_client.get_monthly_stats() or {}
    except Exception as e:
        _LOGGER.error("stats_monthly error: %s", e)
        return {}


@app.post("/stats/hourly")
async def stats_hourly_update(model_version: str = "unknown"):
    """Trigger hourly stats calculation."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    result = tracking_client.calculate_and_update_hourly_stats(model_version)
    return {"success": True, "stats": result or {}, "empty": not bool(result)}


# ============ Training Data Endpoints ============

@app.get("/training/data")
async def training_data():
    """Get combined training data from Logs + ML_Tracking."""
    if not sheets_client or not tracking_client:
        raise HTTPException(503, "Not connected")

    logs_data = sheets_client.get_logs_for_training()
    tracking_snapshot = tracking_client.get_training_snapshot()
    tracking_data = tracking_snapshot.get("training_data", [])

    dataset_fingerprint = hashlib.sha256(
        json.dumps(
            {"logs_data": logs_data, "tracking_data": tracking_data},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return TrainingDataResponse(
        logs_data=logs_data,
        tracking_data=tracking_data,
        total_samples=len(logs_data) + len(tracking_data),
        dataset_fingerprint=dataset_fingerprint,
        mark_token=tracking_snapshot.get("mark_token"),
        snapshot_generated_at=tracking_snapshot.get("snapshot_generated_at"),
        mark_candidates_count=int(tracking_snapshot.get("mark_candidates_count", 0)),
    )


@app.post("/training/mark", response_model=TrainingMarkResponse)
async def training_mark(request: TrainingMarkRequest | None = None):
    """Mark reviewed data as TRAINED after successful training."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")

    expected_token = request.mark_token if request else None
    result = tracking_client.mark_as_trained(expected_mark_token=expected_token)

    if expected_token is not None and not result.get("token_matched", False):
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Mark token mismatch. Snapshot changed since training data fetch.",
                "current_mark_token": result.get("current_mark_token"),
            },
        )

    return TrainingMarkResponse(
        success=True,
        marked_count=int(result.get("marked_count", 0)),
        token_matched=bool(result.get("token_matched", True)),
        current_mark_token=result.get("current_mark_token"),
    )


# ============ Media Upload ============

@app.post("/media/upload", response_model=UploadMediaResponse)
async def media_upload(
    file: UploadFile = File(...),
    key: str = Form(...),
):
    """Upload media file to S3."""
    if not s3_uploader:
        raise HTTPException(503, "S3 not connected")

    data = await file.read()
    content_type = file.content_type or "application/octet-stream"

    try:
        url = s3_uploader.upload_bytes(key=key, data=data, content_type=content_type)
        return UploadMediaResponse(url=url)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")


# ============ DLQ Status Endpoint ============

@app.get("/queue/status")
async def queue_status():
    """Status Dead Letter Queue — berapa item yang menunggu replay ke Sheets."""
    size = await dlq_size()
    items = _dlq_load()
    oldest = items[0]["queued_at"] if items else None
    return {
        "queue_size": size,
        "oldest_item_at": oldest,
        "status": "empty" if size == 0 else "pending",
    }


# ============ Health ============

@app.get("/health", response_model=HealthResponse)
async def health():
    queue_size = await dlq_size()
    status = "ok" if sheets_client else "degraded"
    if queue_size > 0:
        status = "degraded"
    return HealthResponse(
        service="data-api",
        status=status,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=False,
    )

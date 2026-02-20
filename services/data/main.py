"""
Data & Analytics API — FastAPI Service
========================================
Centralized data access for Google Sheets, S3, and ML Tracking.

All services access data through this API instead of directly.
This enables future migration from Google Sheets to PostgreSQL.

Endpoints:
    POST   /logs/append        - Append row to Logs sheet
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
    GET    /health             - Health check
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from services.data.src.sheets import GoogleSheetsClient
from services.data.src.storage import S3Uploader
from services.data.src.tracking import MLTrackingClient
from services.shared.config import DataServiceConfig, setup_logging, trace_id_var
from services.shared.models import (
    FindRowResponse,
    HealthResponse,
    LogRowRequest,
    StatsResponse,
    TrackingLogRequest,
    TrainingDataResponse,
    TrainingMarkRequest,
    TrainingMarkResponse,
    UploadMediaResponse,
)

_LOGGER = logging.getLogger(__name__)

config = DataServiceConfig.from_env()
setup_logging(config.debug, "data-api")

# Global clients
sheets_client: GoogleSheetsClient | None = None
tracking_client: MLTrackingClient | None = None
s3_uploader: S3Uploader | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global sheets_client, tracking_client, s3_uploader
    _LOGGER.info("Starting Data API...")

    sheets_client = GoogleSheetsClient(config)
    tracking_client = MLTrackingClient(config, spreadsheet=sheets_client.spreadsheet)
    s3_uploader = S3Uploader(config)

    _LOGGER.info("Data API ready — Sheets: %s", config.google_spreadsheet_name)
    yield
    _LOGGER.info("Data API shutting down")


app = FastAPI(
    title="Ticket Classifier - Data API",
    description="Centralized data access for Google Sheets, S3, and ML Tracking",
    version="2.0.0",
    root_path="/data",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus HTTP metrics (auto-instruments all endpoints → /metrics)
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

@app.post("/logs/append")
async def logs_append(request: LogRowRequest):
    """Append a log row to the Logs sheet."""
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
        row_index = sheets_client.append_log_row(row, return_row_index=True)
        return {"success": True, "row_index": row_index}
    except Exception as e:
        _LOGGER.exception("Failed to append log: %s", e)
        raise HTTPException(500, f"Failed to append: {e}")


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
        sheets_client.update_log_row(row_index, row)
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, f"Failed to update: {e}")


@app.get("/logs/find/{tech_message_id}")
async def logs_find(tech_message_id: str):
    """Find row index by tech_message_id."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    idx = sheets_client.find_row_index_by_tech_message_id(tech_message_id)
    return FindRowResponse(row_index=idx, found=idx is not None)


@app.get("/logs/row/{row_index}")
async def logs_get_row(row_index: int):
    """Get a specific row by index."""
    if not sheets_client:
        raise HTTPException(503, "Sheets not connected")

    row = sheets_client.get_row(row_index)
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
    return tracking_client.get_weekly_stats()


@app.get("/stats/monthly")
async def stats_monthly():
    """Get monthly aggregated stats."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    return tracking_client.get_monthly_stats()


@app.post("/stats/hourly")
async def stats_hourly_update(model_version: str = "unknown"):
    """Trigger hourly stats calculation."""
    if not tracking_client:
        raise HTTPException(503, "Tracking not connected")
    result = tracking_client.calculate_and_update_hourly_stats(model_version)
    return {"success": bool(result), "stats": result}


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
            {
                "logs_data": logs_data,
                "tracking_data": tracking_data,
            },
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


# ============ Health ============

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        service="data-api",
        status="ok" if sheets_client else "degraded",
    )


if __name__ == "__main__":
    # Use app object (not string) to prevent double-import that causes
    # Prometheus 'Duplicated timeseries' ValueError.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=False,
    )

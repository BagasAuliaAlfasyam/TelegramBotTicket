"""
Training Service — FastAPI Entry Point
========================================
Endpoints:
    POST /train      - Trigger retraining
    GET  /status     - Get training status
    GET  /health     - Health check
"""
from __future__ import annotations

import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.shared.config import TrainingServiceConfig, setup_logging
from services.shared.models import HealthResponse
from services.training.src.retrain import RetrainPipeline

_LOGGER = logging.getLogger(__name__)
_train_lock = threading.Lock()

config = TrainingServiceConfig.from_env()
setup_logging(config.debug, "training-api")

pipeline: RetrainPipeline | None = None
MIN_TUNE_TRIALS = 1
MAX_TUNE_TRIALS = 200


@asynccontextmanager
async def lifespan(application: FastAPI):
    global pipeline
    pipeline = RetrainPipeline(config)
    _LOGGER.info("Training service ready")
    yield
    _LOGGER.info("Training service shutting down")


app = FastAPI(
    title="Ticket Classifier - Training Service",
    version="2.0.0",
    root_path="/training",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/train")
async def train(body: dict | None = None):
    """Trigger retraining. Runs synchronously (or in background thread)."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")

    body = body or {}

    if not _train_lock.acquire(blocking=False):
        return {"success": False, "message": "Training already in progress", "status": "running"}

    force = bool(body.get("force", False))
    tune = bool(body.get("tune", False))

    raw_tune_trials = body.get("tune_trials", 50)
    try:
        tune_trials = int(raw_tune_trials)
    except (TypeError, ValueError):
        _train_lock.release()
        raise HTTPException(400, "tune_trials must be an integer")

    tune_trials = max(MIN_TUNE_TRIALS, min(tune_trials, MAX_TUNE_TRIALS))

    # Run in background thread to not block the API
    def _run():
        try:
            pipeline.run(force=force, tune=tune, tune_trials=tune_trials)
        finally:
            _train_lock.release()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "success": True,
        "message": "Training started",
        "status": "running",
        "params": {"force": force, "tune": tune, "tune_trials": tune_trials},
    }


@app.get("/status")
async def status():
    if not pipeline:
        return {"status": "not_initialized"}
    return {
        "status": pipeline.status,
        "last_trained": pipeline.last_trained,
        "last_result": pipeline.last_result,
        "progress": pipeline.progress,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        service="training-api",
        status="ok" if pipeline else "degraded",
    )


if __name__ == "__main__":
    uvicorn.run(
        "services.training.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )

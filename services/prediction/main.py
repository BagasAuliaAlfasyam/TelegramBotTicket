"""
Prediction API — FastAPI Service
==================================
ML Prediction microservice with LightGBM + Gemini cascade.

Endpoints:
    POST /predict         - Single text prediction
    POST /predict/batch   - Batch prediction (up to 100)
    GET  /model/info      - Model metadata & status
    POST /model/reload    - Hot reload model from MLflow
    GET  /health          - Health check
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from services.prediction.src.hybrid import HybridClassifier
from services.shared.config import PredictionServiceConfig, setup_logging, trace_id_var
from services.shared.models import (
    BatchPredictionRequest,
    BatchPredictionResult,
    HealthResponse,
    ModelInfoResponse,
    ModelReloadRequest,
    ModelReloadResponse,
    PredictionRequest,
    PredictionResult,
)

_LOGGER = logging.getLogger(__name__)

# ============ App Setup ============

config = PredictionServiceConfig.from_env()
setup_logging(config.debug, "prediction-api")

# Global classifier instance
classifier: HybridClassifier | None = None

# ============ Custom Business Metrics ============

_predictions_counter = Counter(
    "ticket_predictions_total",
    "Total ticket predictions",
    ["status", "source", "label"],
)
_confidence_histogram = Histogram(
    "ticket_prediction_confidence",
    "ML prediction confidence score (0-1)",
    buckets=[0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
)
_gemini_counter = Counter(
    "ticket_gemini_calls_total",
    "Total Gemini cascade calls",
    ["outcome"],  # success | error
)


# ============ Startup/Shutdown ============

@asynccontextmanager
async def lifespan(application: FastAPI):
    global classifier
    _LOGGER.info("Starting Prediction API...")
    _LOGGER.info("MLflow URI: %s", config.mlflow_tracking_uri)
    _LOGGER.info("Gemini enabled: %s", config.gemini_enabled)

    classifier = HybridClassifier(config)

    if not classifier.load_model():
        _LOGGER.error("Failed to load model from MLflow!")
        # Don't crash — service starts but returns REVIEW for all requests
    else:
        _LOGGER.info("Model loaded successfully: %s", classifier._lgbm.model_version)
    yield
    _LOGGER.info("Prediction API shutting down")


app = FastAPI(
    title="Ticket Classifier - Prediction API",
    description="LightGBM + Gemini cascade prediction service",
    version="2.0.0",
    root_path="/prediction",
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


# ============ Endpoints ============

@app.post("/predict", response_model=PredictionResult)
async def predict(request: PredictionRequest):
    """
    Predict Symtomps category for a ticket.

    Uses LightGBM as primary classifier.
    If confidence < cascade_threshold, falls back to Gemini.
    """
    if not classifier or not classifier.is_loaded:
        return PredictionResult(
            predicted_symtomps="",
            ml_confidence=0.0,
            prediction_status="REVIEW",
            inference_time_ms=0.0,
            source="lightgbm",
        )

    result = classifier.predict(request.tech_raw_text, request.solving)

    # Record business metrics
    label = result.predicted_symtomps or "unknown"
    _predictions_counter.labels(
        status=result.prediction_status,
        source=result.source,
        label=label,
    ).inc()
    _confidence_histogram.observe(result.ml_confidence)
    if result.source in ("gemini", "hybrid"):
        _gemini_counter.labels(outcome="success").inc()

    return result


@app.post("/predict/batch", response_model=BatchPredictionResult)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction for multiple tickets."""
    if not classifier or not classifier.is_loaded:
        raise HTTPException(503, "Model not loaded")

    start = time.time()
    results = []
    for item in request.items:
        result = classifier.predict(item.tech_raw_text, item.solving)
        label = result.predicted_symtomps or "unknown"
        _predictions_counter.labels(
            status=result.prediction_status,
            source=result.source,
            label=label,
        ).inc()
        _confidence_histogram.observe(result.ml_confidence)
        if result.source in ("gemini", "hybrid"):
            _gemini_counter.labels(outcome="success").inc()
        results.append(result)

    return BatchPredictionResult(
        results=results,
        total_time_ms=round((time.time() - start) * 1000, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata, version, classes, and thresholds."""
    if not classifier:
        raise HTTPException(503, "Classifier not initialized")

    info = classifier.get_info()
    return ModelInfoResponse(
        version=info.get("version", "unknown"),
        loaded_stage=info.get("loaded_stage"),
        model_run_id=info.get("model_run_id"),
        is_loaded=info.get("is_loaded", False),
        num_classes=info.get("num_classes", 0),
        classes=info.get("classes", []),
        thresholds=info.get("thresholds", {}),
        loaded_from_mlflow=True,
        gemini_enabled=info.get("gemini_enabled", False),
        training_samples=info.get("training_samples"),
        training_accuracy=info.get("training_accuracy"),
        training_f1_macro=info.get("training_f1_macro"),
        trained_at=info.get("trained_at"),
    )


@app.post("/model/reload", response_model=ModelReloadResponse)
async def model_reload(request: ModelReloadRequest = ModelReloadRequest()):
    """Hot reload model from MLflow without restart."""
    if not classifier:
        raise HTTPException(503, "Classifier not initialized")

    success, old_ver, new_ver = classifier.reload_model(request.stage)

    return ModelReloadResponse(
        success=success,
        old_version=old_ver,
        new_version=new_ver,
        message=f"Reloaded from {old_ver} to {new_ver}" if success else "Reload failed",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        service="prediction-api",
        status="ok" if (classifier and classifier.is_loaded) else "degraded",
    )


# ============ MLflow Registry Endpoints ============

@app.get("/mlflow/status")
async def mlflow_status():
    """Get MLflow registry status and model versions."""
    if not classifier or not classifier._mlflow_mgr:
        raise HTTPException(503, "MLflow not configured")

    mlflow_mgr = classifier._mlflow_mgr
    if not mlflow_mgr.is_enabled:
        return {"enabled": False, "message": "MLflow not enabled"}

    if not mlflow_mgr._initialized:
        mlflow_mgr.init()

    status = mlflow_mgr.get_status()
    versions = mlflow_mgr.get_model_versions(limit=10)
    status["versions"] = versions
    return status


@app.post("/mlflow/promote")
async def mlflow_promote(version: str, stage: str = "Production"):
    """Promote a model version to Production/Staging."""
    if not classifier or not classifier._mlflow_mgr:
        raise HTTPException(503, "MLflow not configured")

    mlflow_mgr = classifier._mlflow_mgr
    if not mlflow_mgr.is_enabled or not mlflow_mgr._initialized:
        raise HTTPException(503, "MLflow not initialized")

    result = mlflow_mgr.transition_model_stage(version, stage)
    if not result["success"]:
        raise HTTPException(400, result["message"])
    return result


# ============ Entry Point ============

if __name__ == "__main__":
    # Use app object (not string) to prevent double-import that causes
    # Prometheus 'Duplicated timeseries' ValueError.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=False,  # reload=True requires string ref; disabled to avoid double-import
    )

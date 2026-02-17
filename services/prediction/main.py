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
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.prediction.src.hybrid import HybridClassifier
from services.shared.config import PredictionServiceConfig, setup_logging
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
        # Don't crash — service starts but returns MANUAL for all requests
    else:
        _LOGGER.info("Model loaded successfully: %s", classifier._lgbm.model_version)
    yield
    _LOGGER.info("Prediction API shutting down")


app = FastAPI(
    title="Ticket Classifier - Prediction API",
    description="LightGBM + Gemini cascade prediction service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            prediction_status="MANUAL",
            inference_time_ms=0.0,
            source="lightgbm",
        )

    result = classifier.predict(request.tech_raw_text, request.solving)
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
        is_loaded=info.get("is_loaded", False),
        num_classes=info.get("num_classes", 0),
        classes=info.get("classes", []),
        thresholds=info.get("thresholds", {}),
        loaded_from_mlflow=True,
        gemini_enabled=info.get("gemini_enabled", False),
        training_samples=info.get("training_samples"),
        training_accuracy=info.get("training_accuracy"),
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


# ============ Entry Point ============

if __name__ == "__main__":
    uvicorn.run(
        "services.prediction.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )

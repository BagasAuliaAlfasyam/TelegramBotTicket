"""
Shared Data Models
===================
Pydantic models used as API contracts between microservices.
Single source of truth for request/response schemas.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ============ Enums ============

class PredictionStatus(str, Enum):
    AUTO = "AUTO"
    HIGH_REVIEW = "HIGH_REVIEW"
    MEDIUM_REVIEW = "MEDIUM_REVIEW"
    MANUAL = "MANUAL"


class PredictionSource(str, Enum):
    LIGHTGBM = "lightgbm"
    GEMINI = "gemini"
    HYBRID = "hybrid"


class ReviewStatus(str, Enum):
    PENDING = "pending"
    AUTO_APPROVED = "auto_approved"
    APPROVED = "APPROVED"
    CORRECTED = "CORRECTED"
    TRAINED = "TRAINED"


# ============ Prediction Service Models ============

class PredictionRequest(BaseModel):
    """Request body for /predict endpoint."""
    tech_raw_text: str = Field(..., description="Raw text dari teknisi")
    solving: str = Field("", description="Text solving dari ops")


class PredictionResult(BaseModel):
    """Response from /predict endpoint."""
    predicted_symtomps: str = Field(..., description="Label kategori yang diprediksi")
    ml_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    prediction_status: PredictionStatus = Field(..., description="Status berdasarkan threshold")
    inference_time_ms: float = Field(..., description="Waktu prediksi dalam milliseconds")
    source: PredictionSource = Field(PredictionSource.LIGHTGBM, description="Sumber prediksi")
    gemini_label: Optional[str] = Field(None, description="Label dari Gemini (jika cascade)")
    gemini_confidence: Optional[float] = Field(None, description="Confidence Gemini (jika cascade)")


class BatchPredictionRequest(BaseModel):
    """Request body for /predict/batch endpoint."""
    items: list[PredictionRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResult(BaseModel):
    """Response from /predict/batch endpoint."""
    results: list[PredictionResult]
    total_time_ms: float


class ModelInfoResponse(BaseModel):
    """Response from /model/info endpoint."""
    version: str
    is_loaded: bool
    num_classes: int
    classes: list[str] = []
    thresholds: dict[str, float] = {}
    loaded_from_mlflow: bool = False
    gemini_enabled: bool = False
    training_samples: Optional[int] = None
    training_accuracy: Optional[float] = None
    trained_at: Optional[str] = None


class ModelReloadRequest(BaseModel):
    """Request body for /model/reload endpoint."""
    stage: str = Field("Production", description="MLflow stage to load")


class ModelReloadResponse(BaseModel):
    """Response from /model/reload endpoint."""
    success: bool
    old_version: str
    new_version: str
    message: str


# ============ Data Service Models ============

class LogRowRequest(BaseModel):
    """Request to append a row to Logs sheet."""
    group_label: str = ""
    ticket_date: str = ""
    response_at: str = ""
    tech_message_id: str = ""
    tech_message_date: str = ""
    tech_message_time: str = ""
    tech_raw_text: str = ""
    media_type: str = ""
    media_url: str = ""
    ops_message_id: str = ""
    ops_text: str = ""
    solving: str = ""
    solve_timestamp: str = ""
    app_code: str = ""
    solver_name: str = ""
    is_oncek: str = "false"
    sla_response_min: str | float = ""
    sla_status: str = ""
    sla_remaining_min: str | float = ""
    symtomps: str = ""


class LogRowUpdateRequest(BaseModel):
    """Request to update a row in Logs sheet."""
    row_index: int
    row: LogRowRequest


class TrackingLogRequest(BaseModel):
    """Request to log prediction to ML_Tracking sheet."""
    tech_message_id: int
    tech_raw_text: str
    solving: str
    predicted_symtomps: str
    ml_confidence: float
    prediction_status: str
    source: str = "lightgbm"


class StatsResponse(BaseModel):
    """Generic stats response."""
    total_predictions: int = 0
    avg_confidence: float = 0.0
    auto_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    manual_count: int = 0
    reviewed_count: int = 0
    pending_count: int = 0


class UploadMediaRequest(BaseModel):
    """Upload media metadata (actual bytes sent separately)."""
    key: str
    content_type: str


class UploadMediaResponse(BaseModel):
    """Response from media upload."""
    url: str


class FindRowResponse(BaseModel):
    """Response for finding a row by tech_message_id."""
    row_index: Optional[int] = None
    found: bool = False


class TrainingDataResponse(BaseModel):
    """Training data from Logs + ML_Tracking."""
    logs_data: list[dict] = []
    tracking_data: list[dict] = []
    total_samples: int = 0


# ============ Training Service Models ============

class TrainRequest(BaseModel):
    """Request to trigger model training."""
    use_optuna: bool = Field(False, description="Use Optuna hyperparameter tuning")
    target_stage: str = Field("Staging", description="MLflow stage after training")
    min_samples: int = Field(50, description="Minimum samples required")


class TrainStatusResponse(BaseModel):
    """Response for training status."""
    status: str  # idle, training, completed, failed
    progress: Optional[float] = None
    message: str = ""
    model_version: Optional[str] = None
    metrics: Optional[dict] = None


# ============ Health Check ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str
    version: str = "1.0.0"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

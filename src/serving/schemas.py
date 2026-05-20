from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class_id: int = Field(..., description="Predicted class index.")
    predicted_label: str = Field(..., description="Human-readable predicted damage label.")
    confidence: float = Field(..., description="Confidence of the predicted class.")
    probabilities: dict[str, float] = Field(..., description="Per-class probability distribution.")
    model_name: str = Field(..., description="Serving model name.")
    model_version: str = Field(..., description="Serving model version.")


class HealthResponse(BaseModel):
    status: str
    service: str
    model_name: str
    model_version: str
    device: str
    checkpoint_path: str
    model_loaded: bool
    checkpoint_exists: bool
    model_source_used: str | None = None
    model_uri: str | None = None


class ModelInfoResponse(BaseModel):
    model_source_requested: str
    model_source_used: str | None = None
    registered_model_name: str | None = None
    model_stage: str | None = None
    model_alias: str | None = None
    model_uri: str | None = None
    tracking_uri: str | None = None
    checkpoint_path: str | None = None
    checkpoint_path_used: str | None = None
    device: str | None = None
    loaded: bool
    load_error: str | None = None
    fallback_warning: str | None = None

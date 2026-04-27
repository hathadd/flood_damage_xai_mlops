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

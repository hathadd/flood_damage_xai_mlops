from __future__ import annotations

from pydantic import BaseModel, Field


class SceneBuildingPrediction(BaseModel):
    building_index: int
    building_uid: str | None = None
    bbox: list[float] = Field(default_factory=list)
    crop_box: list[int] = Field(default_factory=list)
    polygon: list[list[float]] = Field(default_factory=list)
    true_label: str | None = None
    predicted_class_id: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


class ScenePredictionResponse(BaseModel):
    total_buildings: int
    predictions: list[SceneBuildingPrediction]
    annotated_image_path: str | None = None
    model_name: str
    model_version: str

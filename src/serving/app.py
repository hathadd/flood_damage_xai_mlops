from __future__ import annotations

from fastapi import FastAPI

from src.serving.api import router as crop_router
from src.serving.scene_api import router as scene_router

app = FastAPI(
    title="Flood Damage XAI MLOps Inference API",
    version="0.2.0",
    description=(
        "FastAPI inference service for the final Run B Siamese ResNet18 regularized "
        "flood damage classifier. It supports both crop-level prediction and a Phase A "
        "scene-level prototype using xBD building polygons as a pseudo-detector."
    ),
)

app.include_router(crop_router)
app.include_router(scene_router)

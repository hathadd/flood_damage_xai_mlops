from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.serving.api import router as crop_router
from src.serving.explain_api import router as explain_router
from src.serving.model_loader import preload_model
from src.serving.scene_api import router as scene_router

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        preload_model()
    except Exception as exc:
        LOGGER.warning("Serving startup preload failed: %s", exc)
    yield


app = FastAPI(
    title="Flood Damage XAI MLOps Inference API",
    version="0.4.0",
    description=(
        "FastAPI inference service for the final Run B Siamese ResNet18 regularized "
        "flood damage classifier. It supports crop-level prediction, a Phase A "
        "scene-level prototype using xBD building polygons as a pseudo-detector, "
        "building-specific Grad-CAM explanations, and MLflow Registry loading with "
        "safe local checkpoint fallback."
    ),
    lifespan=lifespan,
)

app.include_router(crop_router)
app.include_router(scene_router)
app.include_router(explain_router)

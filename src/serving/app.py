from __future__ import annotations

from fastapi import FastAPI

from src.serving.api import router

app = FastAPI(
    title="Flood Damage XAI MLOps Inference API",
    version="0.1.0",
    description=(
        "FastAPI inference service for the final Run B Siamese ResNet18 regularized "
        "flood damage classifier. This first deployment version provides prediction "
        "only; Grad-CAM serving will be added later."
    ),
)

app.include_router(router)

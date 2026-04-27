from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.serving.config import settings
from src.serving.inference import predict_damage
from src.serving.model_loader import get_device, load_model
from src.serving.preprocessing import preprocess_pair
from src.serving.schemas import HealthResponse, PredictionResponse

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "flood_damage_xai_mlops-serving",
        "message": "Flood damage prediction API is running.",
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checkpoint_path = Path(settings.checkpoint_path)
    model_loaded = load_model.cache_info().currsize > 0
    return HealthResponse(
        status="ok",
        service="flood_damage_xai_mlops-serving",
        model_name=settings.model_name,
        model_version=settings.model_version,
        device=str(get_device()),
        checkpoint_path=str(checkpoint_path),
        model_loaded=model_loaded,
        checkpoint_exists=checkpoint_path.exists(),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
) -> PredictionResponse:
    if not pre_image.content_type or not pre_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="pre_image must be an image upload.")
    if not post_image.content_type or not post_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="post_image must be an image upload.")

    try:
        pre_bytes = await pre_image.read()
        post_bytes = await post_image.read()
        pre_tensor, post_tensor = preprocess_pair(pre_bytes, post_bytes)
        prediction = predict_damage(pre_tensor, post_tensor)
        return PredictionResponse(**prediction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

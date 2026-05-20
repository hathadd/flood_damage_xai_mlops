from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.monitoring.collect_inference_logs import log_upload_inference
from src.serving.config import settings
from src.serving.inference import predict_damage
from src.serving.model_loader import get_device, get_model_info, get_serving_identity, load_model
from src.serving.preprocessing import preprocess_pair
from src.serving.schemas import HealthResponse, ModelInfoResponse, PredictionResponse

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {
        "service": "flood_damage_xai_mlops-serving",
        "message": "Flood damage prediction API is running.",
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    info = get_model_info()
    model_name, model_version = get_serving_identity()
    checkpoint_path = info.get("checkpoint_path") or settings.checkpoint_path
    checkpoint_exists = Path(checkpoint_path).exists() if checkpoint_path else False
    return HealthResponse(
        status="ok",
        service="flood_damage_xai_mlops-serving",
        model_name=model_name,
        model_version=model_version,
        device=str(get_device()),
        checkpoint_path=str(checkpoint_path),
        model_loaded=bool(info.get("loaded")),
        checkpoint_exists=checkpoint_exists,
        model_source_used=info.get("source_used"),
        model_uri=info.get("model_uri"),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    info = get_model_info()
    return ModelInfoResponse(
        model_source_requested=info.get("source_requested") or settings.model_source,
        model_source_used=info.get("source_used"),
        registered_model_name=info.get("registered_model_name"),
        model_stage=info.get("model_stage"),
        model_alias=info.get("model_alias"),
        model_uri=info.get("model_uri"),
        tracking_uri=info.get("tracking_uri"),
        checkpoint_path=info.get("checkpoint_path"),
        checkpoint_path_used=info.get("checkpoint_path_used"),
        device=info.get("device") or str(get_device()),
        loaded=bool(info.get("loaded")),
        load_error=info.get("load_error"),
        fallback_warning=info.get("fallback_warning"),
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
        try:
            log_upload_inference(
                pre_image_bytes=pre_bytes,
                post_image_bytes=post_bytes,
                prediction=prediction,
                pre_image_path=pre_image.filename,
                post_image_path=post_image.filename,
            )
        except Exception:
            pass
        return PredictionResponse(**prediction)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

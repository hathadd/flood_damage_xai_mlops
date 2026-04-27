from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.scene.polygon_parser import parse_xbd_buildings
from src.scene.scene_inference import predict_scene
from src.scene.schemas import ScenePredictionResponse
from src.scene.visualization import annotate_post_image
from src.serving.config import settings
from src.serving.model_loader import get_device, load_model

router = APIRouter()

OUTPUT_DIR = Path("outputs/serving/scene_predictions")


def _load_rgb_image(image_bytes: bytes, field_name: str) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{field_name} is not a valid image.") from exc


@router.post("/predict-scene", response_model=ScenePredictionResponse)
async def predict_scene_endpoint(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    post_json: UploadFile = File(...),
    context_ratio: float = Form(0.25),
    min_crop_size: int = Form(64),
    save_annotated: bool = Form(True),
) -> ScenePredictionResponse:
    if not pre_image.content_type or not pre_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="pre_image must be an image upload.")
    if not post_image.content_type or not post_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="post_image must be an image upload.")
    if post_json.content_type and "json" not in post_json.content_type.lower():
        if not post_json.filename or not post_json.filename.lower().endswith(".json"):
            raise HTTPException(status_code=400, detail="post_json must be a JSON upload.")
    if context_ratio < 0:
        raise HTTPException(status_code=400, detail="context_ratio must be non-negative.")
    if min_crop_size <= 0:
        raise HTTPException(status_code=400, detail="min_crop_size must be strictly positive.")

    try:
        pre_bytes = await pre_image.read()
        post_bytes = await post_image.read()
        json_bytes = await post_json.read()

        pre_pil = _load_rgb_image(pre_bytes, "pre_image")
        post_pil = _load_rgb_image(post_bytes, "post_image")
        buildings = parse_xbd_buildings(json_bytes)

        model = load_model()
        device = get_device()
        predictions = predict_scene(
            pre_image=pre_pil,
            post_image=post_pil,
            buildings=buildings,
            model=model,
            device=device,
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
        )

        annotated_image_path: str | None = None
        if save_annotated:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = OUTPUT_DIR / f"scene_prediction_{timestamp}.png"
            _, saved_path = annotate_post_image(post_pil, predictions, output_path=output_path)
            if saved_path is not None:
                annotated_image_path = str(saved_path)

        return ScenePredictionResponse(
            total_buildings=len(predictions),
            predictions=predictions,
            annotated_image_path=annotated_image_path,
            model_name=settings.model_name,
            model_version=settings.model_version,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scene prediction failed: {exc}") from exc


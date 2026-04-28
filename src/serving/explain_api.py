from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.serving.config import settings
from src.xai.scene_gradcam import generate_building_gradcam

router = APIRouter()

OUTPUT_DIR = Path("outputs/serving/gradcam")


def _load_rgb_image(image_bytes: bytes, field_name: str) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{field_name} is not a valid image.") from exc


@router.post("/explain-building")
async def explain_building(
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    post_json: UploadFile = File(...),
    building_index: int = Form(...),
    context_ratio: float = Form(0.25),
    min_crop_size: int = Form(64),
) -> dict[str, object]:
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
        post_json_bytes = await post_json.read()

        pre_pil = _load_rgb_image(pre_bytes, "pre_image")
        post_pil = _load_rgb_image(post_bytes, "post_image")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        result = generate_building_gradcam(
            pre_image=pre_pil,
            post_image=post_pil,
            post_json_bytes=post_json_bytes,
            building_index=building_index,
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
            output_dir=OUTPUT_DIR,
        )

        return {
            "building_index": result["building_index"],
            "building_uid": result["building_uid"],
            "bbox": result["bbox"],
            "crop_box": result["crop_box"],
            "true_label": result["true_label"],
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "pre_gradcam_path": result["pre_gradcam_path"],
            "post_gradcam_path": result["post_gradcam_path"],
            "pre_crop_path": result["pre_crop_path"],
            "post_crop_path": result["post_crop_path"],
            "model_name": settings.model_name,
            "model_version": settings.model_version,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Building explanation failed: {exc}") from exc

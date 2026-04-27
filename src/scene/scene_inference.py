from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from src.scene.crop_extraction import crop_with_context
from src.serving.config import settings
from src.serving.preprocessing import EVAL_TRANSFORMS


def _preprocess_crop(crop: Image.Image) -> torch.Tensor:
    crop_np = np.asarray(crop.convert("RGB"))
    transformed = EVAL_TRANSFORMS(image=crop_np)
    return transformed["image"]


def predict_scene(
    pre_image: Image.Image,
    post_image: Image.Image,
    buildings: list[dict[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    if not buildings:
        return predictions

    model.eval()
    for building in buildings:
        bbox = tuple(float(value) for value in building["bbox"])
        pre_crop, crop_box = crop_with_context(
            pre_image,
            bbox,
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
        )
        post_crop, _ = crop_with_context(
            post_image,
            bbox,
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
        )

        pre_tensor = _preprocess_crop(pre_crop).unsqueeze(0).to(device, non_blocking=True)
        post_tensor = _preprocess_crop(post_crop).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(pre_tensor, post_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class_id].item())
        probabilities_dict = {
            settings.label_mapping[index]: float(probabilities[index].item())
            for index in range(settings.num_classes)
        }

        predictions.append(
            {
                "building_index": int(building["building_index"]),
                "building_uid": building.get("building_uid"),
                "bbox": [float(value) for value in bbox],
                "crop_box": [int(value) for value in crop_box],
                "polygon": building.get("polygon", []),
                "true_label": building.get("true_label"),
                "predicted_class_id": predicted_class_id,
                "predicted_label": settings.label_mapping[predicted_class_id],
                "confidence": confidence,
                "probabilities": probabilities_dict,
            }
        )

    return predictions

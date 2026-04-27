from __future__ import annotations

from io import BytesIO

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError

from src.serving.config import settings


def _build_eval_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=settings.image_size, width=settings.image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


EVAL_TRANSFORMS = _build_eval_transforms()


def preprocess_single_image(image_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    image_np = np.asarray(image)
    transformed = EVAL_TRANSFORMS(image=image_np)
    return transformed["image"]


def preprocess_pair(pre_image_bytes: bytes, post_image_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    pre_tensor = preprocess_single_image(pre_image_bytes).unsqueeze(0)
    post_tensor = preprocess_single_image(post_image_bytes).unsqueeze(0)
    return pre_tensor, post_tensor

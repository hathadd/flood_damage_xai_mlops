from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_geometric_transforms(image_size: int = 224) -> A.ReplayCompose:
    return A.ReplayCompose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=0,
                p=0.3,
            ),
        ]
    )


def get_train_photometric_transforms() -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_test_transforms(image_size: int = 224) -> A.Compose:
    return get_eval_transforms(image_size=image_size)


def build_transforms(config: dict[str, Any]) -> dict[str, Any]:
    image_size = config.get("image_size", 224)

    return {
        # Train uses shared geometry for the bi-temporal pair, then
        # branch-specific photometric augmentation for pre and post.
        "train": {
            "joint": get_train_geometric_transforms(image_size=image_size),
            "pre": get_train_photometric_transforms(),
            "post": get_train_photometric_transforms(),
        },
        "val": get_eval_transforms(image_size=image_size),
        "test": get_test_transforms(image_size=image_size),
    }

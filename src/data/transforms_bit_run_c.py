from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_bit_run_c_train_transforms(image_size: int = 224) -> A.ReplayCompose:
    return A.ReplayCompose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.RandomRotate90(p=0.25),
            A.Affine(
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                scale=(0.95, 1.05),
                rotate=(-12, 12),
                p=0.35,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.12,
                contrast_limit=0.12,
                p=0.35,
            ),
            A.GaussNoise(p=0.10),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def get_bit_run_c_eval_transforms(image_size: int = 224) -> A.Compose:
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

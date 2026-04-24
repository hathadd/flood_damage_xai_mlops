from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_run_c_train_transforms(image_size: int = 224) -> A.ReplayCompose:
    return A.ReplayCompose(
        [
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.RandomRotate90(p=0.35),
            A.Affine(
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                scale=(0.90, 1.10),
                rotate=(-20, 20),
                p=0.45,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.18,
                contrast_limit=0.18,
                p=0.45,
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.25,
            ),
            A.GaussNoise(p=0.15),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                p=0.20,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )



def get_run_c_eval_transforms(image_size: int = 224) -> A.Compose:
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

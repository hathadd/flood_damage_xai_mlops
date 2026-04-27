from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServingConfig:
    checkpoint_path: str = os.getenv(
        "FLOOD_DAMAGE_CHECKPOINT_PATH",
        "/content/drive/MyDrive/flood_damage_xai_mlops/outputs/focal_run_b_regularized/checkpoints/best_siamese_resnet18.pt",
    )
    model_name: str = os.getenv("FLOOD_DAMAGE_MODEL_NAME", "run_b_siamese_resnet18_regularized")
    model_version: str = os.getenv("FLOOD_DAMAGE_MODEL_VERSION", "production_candidate_v1")
    image_size: int = int(os.getenv("FLOOD_DAMAGE_IMAGE_SIZE", "224"))
    dropout: float = float(os.getenv("FLOOD_DAMAGE_DROPOUT", "0.4"))
    num_classes: int = int(os.getenv("FLOOD_DAMAGE_NUM_CLASSES", "4"))
    default_device: str = os.getenv("FLOOD_DAMAGE_DEVICE", "auto")
    label_mapping: dict[int, str] = field(
        default_factory=lambda: {
            0: "no-damage",
            1: "minor-damage",
            2: "major-damage",
            3: "destroyed",
        }
    )


settings = ServingConfig()

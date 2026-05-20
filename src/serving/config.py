from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SERVING_CONFIG_PATH = os.getenv("FLOOD_DAMAGE_SERVING_CONFIG", "configs/serving.yaml")


def _load_yaml_config(config_path: str | Path = DEFAULT_SERVING_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _env_or_config(env_key: str, config: dict[str, Any], config_key: str, default: Any) -> Any:
    env_value = os.getenv(env_key)
    if env_value not in {None, ""}:
        return env_value
    return config.get(config_key, default)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ServingConfig:
    model_source: str
    mlflow_tracking_uri: str
    registered_model_name: str | None
    model_stage: str | None
    model_alias: str | None
    direct_model_uri: str | None
    fallback_checkpoint_path: str
    model_name: str
    model_version: str
    image_size: int
    dropout: float
    num_classes: int
    default_device: str
    startup_preload: bool
    config_path: str = DEFAULT_SERVING_CONFIG_PATH
    label_mapping: dict[int, str] = field(
        default_factory=lambda: {
            0: "no-damage",
            1: "minor-damage",
            2: "major-damage",
            3: "destroyed",
        }
    )

    @property
    def checkpoint_path(self) -> str:
        return self.fallback_checkpoint_path


def load_serving_settings(config_path: str | Path = DEFAULT_SERVING_CONFIG_PATH) -> ServingConfig:
    config = _load_yaml_config(config_path)
    return ServingConfig(
        model_source=str(_env_or_config("FLOOD_DAMAGE_MODEL_SOURCE", config, "model_source", "local_checkpoint")),
        mlflow_tracking_uri=str(_env_or_config("MLFLOW_TRACKING_URI", config, "mlflow_tracking_uri", "./mlruns")),
        registered_model_name=_normalize_optional_text(
            _env_or_config("FLOOD_DAMAGE_REGISTERED_MODEL_NAME", config, "registered_model_name", None)
        ),
        model_stage=_normalize_optional_text(_env_or_config("FLOOD_DAMAGE_MODEL_STAGE", config, "model_stage", None)),
        model_alias=_normalize_optional_text(_env_or_config("FLOOD_DAMAGE_MODEL_ALIAS", config, "model_alias", None)),
        direct_model_uri=_normalize_optional_text(_env_or_config("FLOOD_DAMAGE_MODEL_URI", config, "direct_model_uri", None)),
        fallback_checkpoint_path=str(
            _env_or_config(
                "FLOOD_DAMAGE_CHECKPOINT_PATH",
                config,
                "fallback_checkpoint_path",
                "outputs/focal_run_b_regularized/checkpoints/best_siamese_resnet18.pt",
            )
        ),
        model_name=str(_env_or_config("FLOOD_DAMAGE_MODEL_NAME", config, "model_name", "run_b_siamese_resnet18_regularized")),
        model_version=str(_env_or_config("FLOOD_DAMAGE_MODEL_VERSION", config, "model_version", "production_candidate_v1")),
        image_size=int(_env_or_config("FLOOD_DAMAGE_IMAGE_SIZE", config, "image_size", 224)),
        dropout=float(_env_or_config("FLOOD_DAMAGE_DROPOUT", config, "dropout", 0.4)),
        num_classes=int(_env_or_config("FLOOD_DAMAGE_NUM_CLASSES", config, "num_classes", 4)),
        default_device=str(_env_or_config("FLOOD_DAMAGE_DEVICE", config, "device", "auto")),
        startup_preload=_to_bool(
            _env_or_config("FLOOD_DAMAGE_STARTUP_PRELOAD", config, "startup_preload", True),
            True,
        ),
        config_path=str(config_path),
    )


settings = load_serving_settings()

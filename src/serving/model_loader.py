from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.models.siamese_model import SiameseResNet18
from src.serving.config import settings


def _resolve_device() -> torch.device:
    requested = settings.default_device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
            return _normalize_state_dict_keys(checkpoint_obj["model_state_dict"])
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return _normalize_state_dict_keys(checkpoint_obj["state_dict"])
        if checkpoint_obj and all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return _normalize_state_dict_keys(checkpoint_obj)
    raise ValueError("Unsupported checkpoint format. Expected raw state_dict or dict containing model_state_dict/state_dict.")


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return _resolve_device()


@lru_cache(maxsize=1)
def load_model() -> SiameseResNet18:
    checkpoint_path = Path(settings.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    model = SiameseResNet18(
        num_classes=settings.num_classes,
        pretrained=False,
        dropout=settings.dropout,
    )
    model.load_state_dict(_extract_state_dict(checkpoint_obj), strict=True)
    model.to(device)
    model.eval()
    return model

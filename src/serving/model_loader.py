from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from src.models.siamese_model import SiameseResNet18
from src.serving.config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelLoadState:
    source_requested: str
    source_used: str | None = None
    tracking_uri: str | None = None
    registered_model_name: str | None = None
    model_stage: str | None = None
    model_alias: str | None = None
    model_uri: str | None = None
    checkpoint_path: str | None = None
    checkpoint_path_used: str | None = None
    device: str | None = None
    loaded: bool = False
    load_error: str | None = None
    fallback_warning: str | None = None


_MODEL_STATE = ModelLoadState(
    source_requested=settings.model_source,
    tracking_uri=settings.mlflow_tracking_uri,
    registered_model_name=settings.registered_model_name,
    model_stage=settings.model_stage,
    model_alias=settings.model_alias,
    checkpoint_path=settings.fallback_checkpoint_path,
)


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


def _reset_model_state() -> None:
    global _MODEL_STATE
    _MODEL_STATE = ModelLoadState(
        source_requested=settings.model_source,
        tracking_uri=settings.mlflow_tracking_uri,
        registered_model_name=settings.registered_model_name,
        model_stage=settings.model_stage,
        model_alias=settings.model_alias,
        checkpoint_path=settings.fallback_checkpoint_path,
    )


def _update_model_state(**kwargs: Any) -> None:
    global _MODEL_STATE
    state_dict = asdict(_MODEL_STATE)
    state_dict.update(kwargs)
    _MODEL_STATE = ModelLoadState(**state_dict)


def _build_registry_model_uri() -> str:
    if settings.direct_model_uri:
        return settings.direct_model_uri
    if not settings.registered_model_name:
        raise ValueError("registered_model_name must be configured for MLflow Registry loading.")
    if settings.model_alias:
        return f"models:/{settings.registered_model_name}@{settings.model_alias}"
    if settings.model_stage:
        return f"models:/{settings.registered_model_name}/{settings.model_stage}"
    raise ValueError(
        "MLflow Registry loading requires one of: direct_model_uri, model_alias, or model_stage."
    )


def _load_from_registry(device: torch.device) -> torch.nn.Module:
    try:
        import mlflow
        import mlflow.pytorch
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed in the current environment.") from exc

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = _build_registry_model_uri()
    model = mlflow.pytorch.load_model(model_uri)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"MLflow Registry model at '{model_uri}' is not a torch.nn.Module.")

    model.to(device)
    model.eval()
    _update_model_state(
        source_used="mlflow_registry",
        model_uri=model_uri,
        device=str(device),
        loaded=True,
        load_error=None,
        fallback_warning=None,
        checkpoint_path_used=None,
    )
    return model


def _load_from_local_checkpoint(device: torch.device, fallback_warning: str | None = None) -> SiameseResNet18:
    checkpoint_path = Path(settings.fallback_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    model = SiameseResNet18(
        num_classes=settings.num_classes,
        pretrained=False,
        dropout=settings.dropout,
    )
    model.load_state_dict(_extract_state_dict(checkpoint_obj), strict=True)
    model.to(device)
    model.eval()
    _update_model_state(
        source_used="local_checkpoint",
        model_uri=None,
        checkpoint_path_used=str(checkpoint_path),
        device=str(device),
        loaded=True,
        load_error=None,
        fallback_warning=fallback_warning,
    )
    return model


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return _resolve_device()


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    _reset_model_state()
    device = get_device()
    requested_source = settings.model_source.strip().lower()
    _update_model_state(device=str(device))

    if requested_source in {"mlflow_registry", "auto"}:
        try:
            return _load_from_registry(device)
        except Exception as exc:
            message = (
                f"Failed to load model from MLflow Registry: {exc}. "
                f"Falling back to local checkpoint at {settings.fallback_checkpoint_path}."
            )
            warnings.warn(message, RuntimeWarning)
            LOGGER.warning(message)
            _update_model_state(load_error=str(exc), fallback_warning=message)
            return _load_from_local_checkpoint(device, fallback_warning=message)

    if requested_source in {"local_checkpoint", "checkpoint", "local"}:
        return _load_from_local_checkpoint(device)

    raise ValueError(
        f"Unsupported model_source='{settings.model_source}'. Expected mlflow_registry, auto, or local_checkpoint."
    )


def preload_model() -> dict[str, Any]:
    try:
        load_model()
    except Exception as exc:
        LOGGER.warning("Model preload failed: %s", exc)
        _update_model_state(loaded=False, load_error=str(exc), device=str(get_device()))
    return get_model_info()


def clear_model_cache() -> None:
    load_model.cache_clear()
    get_device.cache_clear()
    _reset_model_state()


def get_model_info() -> dict[str, Any]:
    info = asdict(_MODEL_STATE)
    info["loaded"] = bool(_MODEL_STATE.loaded or load_model.cache_info().currsize > 0)
    return info


def get_serving_identity() -> tuple[str, str]:
    info = get_model_info()
    if info.get("source_used") == "mlflow_registry":
        model_name = info.get("registered_model_name") or settings.model_name
        model_version = info.get("model_alias") or info.get("model_stage") or info.get("model_uri") or settings.model_version
        return str(model_name), str(model_version)
    return settings.model_name, settings.model_version

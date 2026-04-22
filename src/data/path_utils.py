from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

DEFAULT_DATA_CONFIG_PATH = "configs/data.yaml"
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def load_data_config(config_path: str | Path = DEFAULT_DATA_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_path_text(path_value: str | Path) -> str:
    return str(path_value).replace("\\", "/")


def is_windows_absolute_path(path_value: str | Path) -> bool:
    return bool(WINDOWS_ABSOLUTE_PATH_PATTERN.match(normalize_path_text(path_value)))


def convert_windows_path_to_wsl(path_value: str | Path) -> Path:
    normalized_path = normalize_path_text(path_value)
    drive = normalized_path[0].lower()
    relative_path = normalized_path[3:].lstrip("/")
    return Path(f"/mnt/{drive}/{relative_path}")


def resolve_dataset_root(
    dataset_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> Path | None:
    if dataset_root is None:
        if config is None:
            config = load_data_config(config_path)
        dataset_root = config.get("dataset", {}).get("root_dir")

    if dataset_root in {None, ""}:
        return None

    normalized_root = normalize_path_text(dataset_root)
    if is_windows_absolute_path(normalized_root):
        return convert_windows_path_to_wsl(normalized_root)

    return Path(normalized_root)


def resolve_data_path(
    path_value: str | Path,
    dataset_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> Path:
    normalized_path = normalize_path_text(path_value)
    if is_windows_absolute_path(normalized_path):
        return convert_windows_path_to_wsl(normalized_path)

    path = Path(normalized_path)
    if path.is_absolute():
        return path

    resolved_root = resolve_dataset_root(
        dataset_root=dataset_root,
        config_path=config_path,
        config=config,
    )
    if resolved_root is None:
        return path

    return resolved_root / path


def resolve_configured_data_path(
    config: dict[str, Any],
    path_key: str,
    dataset_root: str | Path | None = None,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
) -> Path:
    return resolve_data_path(
        config["paths"][path_key],
        dataset_root=dataset_root,
        config_path=config_path,
        config=config,
    )


def make_relative_to_dataset_root(
    path_value: str | Path,
    dataset_root: str | Path | None,
    config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
    config: dict[str, Any] | None = None,
) -> str:
    resolved_root = resolve_dataset_root(
        dataset_root=dataset_root,
        config_path=config_path,
        config=config,
    )
    if resolved_root is None:
        raise ValueError("dataset_root must be provided to create relative metadata paths.")

    absolute_path = resolve_data_path(
        path_value,
        dataset_root=dataset_root,
        config_path=config_path,
        config=config,
    )

    try:
        relative_path = absolute_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Path '{absolute_path}' is outside dataset_root '{resolved_root}'."
        ) from exc

    return PurePosixPath(relative_path.as_posix()).as_posix()

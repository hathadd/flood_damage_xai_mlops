from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.dataset import expand_bbox, load_rgb_image, parse_wkt_polygon, polygon_to_bbox
from src.data.path_utils import resolve_dataset_root
from src.monitoring import DEFAULT_MONITORING_CONFIG_PATH, PROJECT_ROOT

LABEL_ORDER = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def load_monitoring_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load monitoring YAML configuration."""
    final_path = resolve_project_path(config_path or DEFAULT_MONITORING_CONFIG_PATH)
    with open(final_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_project_path(path_value: str | Path | None) -> Path:
    """Resolve a project-relative or absolute path."""
    if path_value is None:
        raise ValueError("Path value cannot be None.")

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_parent_dir(path_value: str | Path) -> Path:
    """Create parent directories for an output path."""
    path = resolve_project_path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_table(path_value: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet table."""
    path = resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(df: pd.DataFrame, path_value: str | Path) -> Path:
    """Save a dataframe as CSV or Parquet."""
    path = ensure_parent_dir(path_value)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def write_json(payload: dict[str, Any], path_value: str | Path) -> Path:
    """Persist a JSON payload."""
    path = ensure_parent_dir(path_value)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def slugify_label(label: str) -> str:
    """Convert a label into a safe column suffix."""
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def probability_columns(probabilities: dict[str, float] | None) -> dict[str, float | None]:
    """Normalize probability keys into flat columns."""
    values = probabilities or {}
    return {
        f"prob_{slugify_label(label)}": float(values.get(label)) if label in values else None
        for label in LABEL_ORDER
    }


def extract_image_statistics(image_array: np.ndarray) -> dict[str, float]:
    """Compute simple RGB and luminance statistics."""
    rgb = image_array.astype(np.float32)
    channel_mean = rgb.mean(axis=(0, 1))
    channel_std = rgb.std(axis=(0, 1))
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return {
        "rgb_mean_r": float(channel_mean[0]),
        "rgb_mean_g": float(channel_mean[1]),
        "rgb_mean_b": float(channel_mean[2]),
        "rgb_std_r": float(channel_std[0]),
        "rgb_std_g": float(channel_std[1]),
        "rgb_std_b": float(channel_std[2]),
        "luminance_mean": float(luminance.mean()),
    }


def crop_array(image_array: np.ndarray, crop_box: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop a numpy image if a crop box is provided."""
    if crop_box is None:
        return image_array
    x1, y1, x2, y2 = crop_box
    return image_array[y1:y2, x1:x2]


def compute_reference_row_features(
    row: dict[str, Any] | pd.Series,
    dataset_root: str | Path | None = None,
    config_path: str | Path = "configs/data.yaml",
    include_image_statistics: bool = True,
) -> dict[str, Any]:
    """Build tabular monitoring features from one metadata row."""
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)

    wkt = row_dict.get("wkt")
    bbox = None
    crop_box = None
    bbox_width = None
    bbox_height = None
    bbox_area = None
    crop_width = None
    crop_height = None
    crop_area = None
    if isinstance(wkt, str) and wkt.strip():
        bbox = polygon_to_bbox(parse_wkt_polygon(wkt))
        bbox_width = float(max(bbox[2] - bbox[0], 0.0))
        bbox_height = float(max(bbox[3] - bbox[1], 0.0))
        bbox_area = float(bbox_width * bbox_height)
        crop_box = expand_bbox(
            bbox=bbox,
            image_width=int(row_dict["image_width"]),
            image_height=int(row_dict["image_height"]),
            context_ratio=0.25,
            min_crop_size=64,
        )
        crop_width = int(max(crop_box[2] - crop_box[0], 0))
        crop_height = int(max(crop_box[3] - crop_box[1], 0))
        crop_area = int(crop_width * crop_height)

    record: dict[str, Any] = {
        "timestamp": None,
        "source_dataset": "reference",
        "request_source": "reference_dataset",
        "sample_id": row_dict.get("sample_id"),
        "building_uid": row_dict.get("building_uid"),
        "split": row_dict.get("split"),
        "disaster": row_dict.get("disaster"),
        "disaster_type": row_dict.get("disaster_type"),
        "sensor": row_dict.get("sensor"),
        "capture_date": row_dict.get("capture_date"),
        "pre_image_path": row_dict.get("pre_image_path"),
        "post_image_path": row_dict.get("post_image_path"),
        "true_label": row_dict.get("damage_class"),
        "true_class_id": row_dict.get("class_id"),
        "predicted_label": None,
        "predicted_class_id": None,
        "confidence": None,
        "model_name": None,
        "model_version": None,
        "image_width": row_dict.get("image_width"),
        "image_height": row_dict.get("image_height"),
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "crop_x1": crop_box[0] if crop_box else None,
        "crop_y1": crop_box[1] if crop_box else None,
        "crop_x2": crop_box[2] if crop_box else None,
        "crop_y2": crop_box[3] if crop_box else None,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "crop_area": crop_area,
    }
    record.update(probability_columns(None))
    record.update(
        {
            "pre_rgb_mean_r": None,
            "pre_rgb_mean_g": None,
            "pre_rgb_mean_b": None,
            "pre_rgb_std_r": None,
            "pre_rgb_std_g": None,
            "pre_rgb_std_b": None,
            "pre_luminance_mean": None,
            "post_rgb_mean_r": None,
            "post_rgb_mean_g": None,
            "post_rgb_mean_b": None,
            "post_rgb_std_r": None,
            "post_rgb_std_g": None,
            "post_rgb_std_b": None,
            "post_luminance_mean": None,
            "delta_luminance_mean": None,
            "delta_rgb_mean_abs": None,
        }
    )

    if not include_image_statistics:
        return record

    try:
        resolved_root = resolve_dataset_root(dataset_root=dataset_root, config_path=config_path)
        pre_image = load_rgb_image(row_dict["pre_image_path"], dataset_root=resolved_root)
        post_image = load_rgb_image(row_dict["post_image_path"], dataset_root=resolved_root)
        pre_crop = crop_array(pre_image, crop_box)
        post_crop = crop_array(post_image, crop_box)
        pre_stats = extract_image_statistics(pre_crop)
        post_stats = extract_image_statistics(post_crop)
        record.update(
            {
                "pre_rgb_mean_r": pre_stats["rgb_mean_r"],
                "pre_rgb_mean_g": pre_stats["rgb_mean_g"],
                "pre_rgb_mean_b": pre_stats["rgb_mean_b"],
                "pre_rgb_std_r": pre_stats["rgb_std_r"],
                "pre_rgb_std_g": pre_stats["rgb_std_g"],
                "pre_rgb_std_b": pre_stats["rgb_std_b"],
                "pre_luminance_mean": pre_stats["luminance_mean"],
                "post_rgb_mean_r": post_stats["rgb_mean_r"],
                "post_rgb_mean_g": post_stats["rgb_mean_g"],
                "post_rgb_mean_b": post_stats["rgb_mean_b"],
                "post_rgb_std_r": post_stats["rgb_std_r"],
                "post_rgb_std_g": post_stats["rgb_std_g"],
                "post_rgb_std_b": post_stats["rgb_std_b"],
                "post_luminance_mean": post_stats["luminance_mean"],
                "delta_luminance_mean": post_stats["luminance_mean"] - pre_stats["luminance_mean"],
                "delta_rgb_mean_abs": float(
                    np.abs(
                        np.array(
                            [
                                post_stats["rgb_mean_r"] - pre_stats["rgb_mean_r"],
                                post_stats["rgb_mean_g"] - pre_stats["rgb_mean_g"],
                                post_stats["rgb_mean_b"] - pre_stats["rgb_mean_b"],
                            ],
                            dtype=np.float32,
                        )
                    ).mean()
                ),
            }
        )
    except Exception:
        pass

    return record


def import_evidently_components() -> dict[str, Any]:
    """Import Evidently symbols with compatibility fallbacks."""
    try:
        from evidently import Report
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is not installed in the active environment. "
            "Install it with `pip install evidently` before generating monitoring reports."
        ) from exc

    try:
        from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset
    except ImportError:
        try:
            from evidently.metric_preset import (  # type: ignore[attr-defined]
                ClassificationPreset,
                DataDriftPreset,
                DataQualityPreset as DataSummaryPreset,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Unable to import Evidently presets from the installed version. "
                "Please install a recent Evidently release."
            ) from exc

    return {
        "Report": Report,
        "DataDriftPreset": DataDriftPreset,
        "DataSummaryPreset": DataSummaryPreset,
        "ClassificationPreset": ClassificationPreset,
    }


def report_to_dict(report: Any) -> dict[str, Any]:
    """Serialize an Evidently report to a Python dictionary."""
    if hasattr(report, "as_dict"):
        return report.as_dict()
    if hasattr(report, "dict"):
        return report.dict()
    if hasattr(report, "json"):
        return json.loads(report.json())
    raise TypeError("Unsupported Evidently report object: cannot export to dict.")


def choose_feature_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ignore_columns: list[str] | None = None,
    preferred_columns: list[str] | None = None,
) -> list[str]:
    """Select the common columns to compare in monitoring reports."""
    ignored = set(ignore_columns or [])
    common_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns
        and column not in ignored
        and not reference_df[column].dropna().empty
        and not current_df[column].dropna().empty
    ]
    if preferred_columns:
        ordered = [column for column in preferred_columns if column in common_columns]
        extras = [column for column in common_columns if column not in ordered]
        return ordered + extras
    return common_columns


def append_rows_to_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Append records to a CSV file while preserving all columns."""
    path = ensure_parent_dir(output_path)
    new_df = pd.DataFrame(records)
    if path.exists():
        existing_df = pd.read_csv(path)
        all_columns = list(dict.fromkeys([*existing_df.columns.tolist(), *new_df.columns.tolist()]))
        combined = pd.concat(
            [
                existing_df.reindex(columns=all_columns),
                new_df.reindex(columns=all_columns),
            ],
            ignore_index=True,
        )
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    return path


def pil_from_bytes(image_bytes: bytes) -> Image.Image:
    """Open uploaded image bytes as RGB PIL image."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")

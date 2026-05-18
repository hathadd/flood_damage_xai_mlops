from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.monitoring.utils import (
    LABEL_ORDER,
    append_rows_to_csv,
    compute_reference_row_features,
    extract_image_statistics,
    load_monitoring_config,
    pil_from_bytes,
    probability_columns,
    save_table,
)


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_common_prediction_fields(
    prediction: dict[str, Any],
    true_label: str | None = None,
    true_class_id: int | None = None,
) -> dict[str, Any]:
    probabilities = prediction.get("probabilities", {})
    return {
        "true_label": true_label,
        "true_class_id": true_class_id,
        "predicted_label": prediction.get("predicted_label"),
        "predicted_class_id": prediction.get("predicted_class_id"),
        "confidence": prediction.get("confidence"),
        "model_name": prediction.get("model_name"),
        "model_version": prediction.get("model_version"),
        **probability_columns(probabilities),
    }


def _stats_to_prefixed(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {
        f"{prefix}_rgb_mean_r": stats["rgb_mean_r"],
        f"{prefix}_rgb_mean_g": stats["rgb_mean_g"],
        f"{prefix}_rgb_mean_b": stats["rgb_mean_b"],
        f"{prefix}_rgb_std_r": stats["rgb_std_r"],
        f"{prefix}_rgb_std_g": stats["rgb_std_g"],
        f"{prefix}_rgb_std_b": stats["rgb_std_b"],
        f"{prefix}_luminance_mean": stats["luminance_mean"],
    }


def build_upload_inference_record(
    pre_image_bytes: bytes,
    post_image_bytes: bytes,
    prediction: dict[str, Any],
    *,
    sample_id: str | None = None,
    building_uid: str | None = None,
    pre_image_path: str | None = None,
    post_image_path: str | None = None,
    request_source: str = "predict",
    true_label: str | None = None,
    true_class_id: int | None = None,
) -> dict[str, Any]:
    """Build one monitoring log record from uploaded images."""
    pre_image = pil_from_bytes(pre_image_bytes)
    post_image = pil_from_bytes(post_image_bytes)
    pre_array = np.asarray(pre_image)
    post_array = np.asarray(post_image)
    pre_stats = extract_image_statistics(pre_array)
    post_stats = extract_image_statistics(post_array)

    record = {
        "timestamp": _current_timestamp(),
        "source_dataset": "inference",
        "request_source": request_source,
        "sample_id": sample_id,
        "building_uid": building_uid,
        "split": None,
        "disaster": None,
        "disaster_type": None,
        "sensor": None,
        "capture_date": None,
        "pre_image_path": pre_image_path,
        "post_image_path": post_image_path,
        "image_width": post_image.width,
        "image_height": post_image.height,
        "bbox_width": None,
        "bbox_height": None,
        "bbox_area": None,
        "crop_x1": 0,
        "crop_y1": 0,
        "crop_x2": post_image.width,
        "crop_y2": post_image.height,
        "crop_width": post_image.width,
        "crop_height": post_image.height,
        "crop_area": post_image.width * post_image.height,
    }
    record.update(_build_common_prediction_fields(prediction, true_label=true_label, true_class_id=true_class_id))
    record.update(_stats_to_prefixed("pre", pre_stats))
    record.update(_stats_to_prefixed("post", post_stats))
    record["delta_luminance_mean"] = post_stats["luminance_mean"] - pre_stats["luminance_mean"]
    record["delta_rgb_mean_abs"] = float(
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
    )
    return record


def build_scene_inference_records(
    pre_image: Image.Image,
    post_image: Image.Image,
    predictions: list[dict[str, Any]],
    *,
    pre_image_path: str | None = None,
    post_image_path: str | None = None,
    request_source: str = "predict-scene",
) -> list[dict[str, Any]]:
    """Build one monitoring log record per predicted building in a scene."""
    pre_array = np.asarray(pre_image.convert("RGB"))
    post_array = np.asarray(post_image.convert("RGB"))
    records: list[dict[str, Any]] = []

    for prediction in predictions:
        crop_box = prediction.get("crop_box") or [0, 0, post_image.width, post_image.height]
        x1, y1, x2, y2 = [int(value) for value in crop_box]
        pre_crop = pre_array[y1:y2, x1:x2]
        post_crop = post_array[y1:y2, x1:x2]
        pre_stats = extract_image_statistics(pre_crop)
        post_stats = extract_image_statistics(post_crop)
        bbox = prediction.get("bbox") or [None, None, None, None]
        bbox_width = None
        bbox_height = None
        bbox_area = None
        if bbox[0] is not None:
            bbox_width = float(max(float(bbox[2]) - float(bbox[0]), 0.0))
            bbox_height = float(max(float(bbox[3]) - float(bbox[1]), 0.0))
            bbox_area = float(bbox_width * bbox_height)

        record = {
            "timestamp": _current_timestamp(),
            "source_dataset": "inference",
            "request_source": request_source,
            "sample_id": None,
            "building_uid": prediction.get("building_uid"),
            "split": None,
            "disaster": None,
            "disaster_type": None,
            "sensor": None,
            "capture_date": None,
            "pre_image_path": pre_image_path,
            "post_image_path": post_image_path,
            "image_width": post_image.width,
            "image_height": post_image.height,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "bbox_area": bbox_area,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "crop_width": x2 - x1,
            "crop_height": y2 - y1,
            "crop_area": (x2 - x1) * (y2 - y1),
        }
        record.update(
            _build_common_prediction_fields(
                prediction,
                true_label=prediction.get("true_label"),
                true_class_id=None,
            )
        )
        record.update(_stats_to_prefixed("pre", pre_stats))
        record.update(_stats_to_prefixed("post", post_stats))
        record["delta_luminance_mean"] = post_stats["luminance_mean"] - pre_stats["luminance_mean"]
        record["delta_rgb_mean_abs"] = float(
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
        )
        records.append(record)

    return records


def append_inference_records(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Append monitoring records to the current inference log."""
    if not records:
        raise ValueError("No records provided for inference logging.")
    return append_rows_to_csv(records, output_path)


def log_upload_inference(
    pre_image_bytes: bytes,
    post_image_bytes: bytes,
    prediction: dict[str, Any],
    *,
    pre_image_path: str | None = None,
    post_image_path: str | None = None,
    config_path: str = "configs/monitoring.yaml",
) -> Path:
    """Append one `/predict` inference row using the configured output path."""
    config = load_monitoring_config(config_path)
    output_path = config["paths"]["current_data_path"]
    record = build_upload_inference_record(
        pre_image_bytes=pre_image_bytes,
        post_image_bytes=post_image_bytes,
        prediction=prediction,
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
    )
    return append_inference_records([record], output_path)


def log_scene_inference(
    pre_image: Image.Image,
    post_image: Image.Image,
    predictions: list[dict[str, Any]],
    *,
    pre_image_path: str | None = None,
    post_image_path: str | None = None,
    config_path: str = "configs/monitoring.yaml",
) -> Path:
    """Append `/predict-scene` rows using the configured output path."""
    config = load_monitoring_config(config_path)
    output_path = config["paths"]["current_data_path"]
    records = build_scene_inference_records(
        pre_image=pre_image,
        post_image=post_image,
        predictions=predictions,
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
    )
    return append_inference_records(records, output_path)


def import_evaluation_predictions(
    predictions_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    data_config_path: str = "configs/data.yaml",
    request_source: str = "offline_prediction_import",
) -> Path:
    """Build an example current inference dataset from offline evaluation outputs."""
    predictions_df = pd.read_csv(predictions_path)
    metadata_df = pd.read_csv(metadata_path)
    merged_df = predictions_df.merge(metadata_df, on=["sample_id", "building_uid"], how="left")

    records: list[dict[str, Any]] = []
    for _, row in merged_df.iterrows():
        base_record = compute_reference_row_features(
            row=row,
            dataset_root=dataset_root,
            config_path=data_config_path,
            include_image_statistics=True,
        )
        base_record.update(
            {
                "timestamp": _current_timestamp(),
                "source_dataset": "inference",
                "request_source": request_source,
                "predicted_class_id": row.get("pred_label"),
                "predicted_label": row.get("pred_class"),
                "true_class_id": row.get("true_label"),
                "true_label": row.get("true_class"),
                "confidence": row.get("confidence"),
                "model_name": "run_b_siamese_resnet18_regularized",
                "model_version": "offline_evaluation_import",
            }
        )
        probability_payload = {
            label: row.get(f"prob_{label.replace('-', '_')}")
            for label in LABEL_ORDER
        }
        base_record.update(probability_columns(probability_payload))
        records.append(base_record)

    output = save_table(pd.DataFrame(records), output_path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or simulate inference logs for monitoring.")
    parser.add_argument("--config-path", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--predictions-path", type=str, default="reports/evaluation/run_b/predictions.csv")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--data-config-path", type=str, default="configs/data.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_monitoring_config(args.config_path)
    metadata_path = args.metadata_path or config["paths"]["metadata_path"]
    output_path = args.output_path or config["paths"]["current_data_path"]
    saved_path = import_evaluation_predictions(
        predictions_path=args.predictions_path,
        metadata_path=metadata_path,
        output_path=output_path,
        dataset_root=args.dataset_root,
        data_config_path=args.data_config_path,
    )
    print(f"Current inference monitoring dataset saved to {Path(saved_path).as_posix()}.")


if __name__ == "__main__":
    main()

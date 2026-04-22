from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.path_utils import (
    load_data_config,
    make_relative_to_dataset_root,
    resolve_configured_data_path,
    resolve_data_path,
    resolve_dataset_root,
)


def load_config(config_path: str = "configs/data.yaml") -> dict[str, Any]:
    return load_data_config(config_path)


def derive_pre_image_name(post_image_name: str) -> str:
    return post_image_name.replace("_post_disaster.png", "_pre_disaster.png")


def derive_sample_id(post_image_name: str) -> str:
    return post_image_name.replace("_post_disaster.png", "")


def parse_single_label_file(
    json_path: Path,
    images_pre_dir: Path,
    images_post_dir: Path,
    keep_classes: set[str],
    ignore_classes: set[str],
    dataset_root: str | Path,
) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    metadata = data.get("metadata", {})
    features_root = data.get("features", {})
    xy_features = features_root.get("xy", [])

    post_image_name = metadata.get("img_name")
    if not post_image_name:
        return []

    pre_image_name = derive_pre_image_name(post_image_name)
    sample_id = derive_sample_id(post_image_name)

    pre_image_path = images_pre_dir / pre_image_name
    post_image_path = images_post_dir / post_image_name

    rows: list[dict[str, Any]] = []

    for feature in xy_features:
        properties = feature.get("properties", {})
        feature_type = properties.get("feature_type")
        damage_class = properties.get("subtype")
        building_uid = properties.get("uid")
        wkt_polygon = feature.get("wkt")

        if feature_type != "building":
            continue

        if damage_class in ignore_classes:
            continue

        if damage_class not in keep_classes:
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "building_uid": building_uid,
                "disaster": metadata.get("disaster"),
                "disaster_type": metadata.get("disaster_type"),
                "capture_date": metadata.get("capture_date"),
                "sensor": metadata.get("sensor"),
                "image_width": metadata.get("width"),
                "image_height": metadata.get("height"),
                "pre_image_name": pre_image_name,
                "post_image_name": post_image_name,
                "pre_image_path": make_relative_to_dataset_root(pre_image_path, dataset_root=dataset_root),
                "post_image_path": make_relative_to_dataset_root(post_image_path, dataset_root=dataset_root),
                "label_json_path": make_relative_to_dataset_root(json_path, dataset_root=dataset_root),
                "damage_class": damage_class,
                "wkt": wkt_polygon,
            }
        )

    return rows


def build_metadata_dataframe(config_path: str = "configs/data.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    dataset_root = resolve_dataset_root(config=config, config_path=config_path)
    if dataset_root is None:
        raise ValueError("dataset.root_dir must be configured to build metadata.")

    images_pre_dir = resolve_configured_data_path(config, "images_pre", dataset_root=dataset_root, config_path=config_path)
    images_post_dir = resolve_configured_data_path(config, "images_post", dataset_root=dataset_root, config_path=config_path)
    labels_post_dir = resolve_configured_data_path(config, "labels_post", dataset_root=dataset_root, config_path=config_path)

    keep_classes = set(config["classes"]["keep"])
    ignore_classes = set(config["classes"]["ignore"])

    all_rows: list[dict[str, Any]] = []

    json_files = sorted(labels_post_dir.glob("*.json"))
    for json_path in json_files:
        rows = parse_single_label_file(
            json_path=json_path,
            images_pre_dir=images_pre_dir,
            images_post_dir=images_post_dir,
            keep_classes=keep_classes,
            ignore_classes=ignore_classes,
            dataset_root=dataset_root,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df["class_id"] = df["damage_class"].map(config["label_map"])
        df["pre_exists"] = df["pre_image_path"].apply(lambda value: resolve_data_path(value, dataset_root=dataset_root).exists())
        df["post_exists"] = df["post_image_path"].apply(lambda value: resolve_data_path(value, dataset_root=dataset_root).exists())
        df["label_exists"] = df["label_json_path"].apply(lambda value: resolve_data_path(value, dataset_root=dataset_root).exists())

    return df


def main() -> None:
    config = load_config("configs/data.yaml")
    output_path = Path(config["paths"]["metadata_out"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_metadata_dataframe("configs/data.yaml")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Metadata saved to: {output_path}")
    print(f"Number of building samples: {len(df)}")

    if not df.empty:
        print("\nClass distribution:")
        print(df["damage_class"].value_counts(dropna=False))

        print("\nFile existence checks:")
        print("pre_exists all true :", bool(df["pre_exists"].all()))
        print("post_exists all true:", bool(df["post_exists"].all()))
        print("label_exists all true:", bool(df["label_exists"].all()))
    else:
        print("Warning: metadata dataframe is empty.")


if __name__ == "__main__":
    main()

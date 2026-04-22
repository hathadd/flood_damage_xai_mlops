from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders, build_datasets
from src.data.path_utils import resolve_data_path

DEFAULT_METADATA_PATH = "data/interim/metadata.csv"
DEFAULT_SPLITS_PATH = "data/splits/metadata_splits.csv"
REQUIRED_METADATA_COLUMNS = {
    "sample_id",
    "building_uid",
    "pre_image_path",
    "post_image_path",
    "label_json_path",
    "damage_class",
    "class_id",
    "wkt",
    "image_width",
    "image_height",
}
REQUIRED_SPLIT_COLUMNS = REQUIRED_METADATA_COLUMNS | {"split"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata, splits, dataset, and dataloaders.")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--splits", type=str, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def require_file(path_value: str | Path, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        fail(f"Missing {label}: {path}")
    return path


def validate_columns(df: pd.DataFrame, required_columns: set[str], label: str) -> None:
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        fail(f"{label} is missing required columns: {missing_columns}")


def validate_path_columns(df: pd.DataFrame, dataset_root: str | None, path_columns: tuple[str, ...]) -> None:
    missing_counts: dict[str, int] = {}
    for column in path_columns:
        exists_mask = df[column].apply(lambda value: resolve_data_path(value, dataset_root=dataset_root).exists())
        missing_count = int((~exists_mask).sum())
        if missing_count:
            missing_counts[column] = missing_count
    if missing_counts:
        fail(f"Path validation failed: {json.dumps(missing_counts, sort_keys=True)}")


def validate_split_overlap(split_df: pd.DataFrame) -> None:
    sample_overlap: dict[str, int] = {}
    building_overlap: dict[str, int] = {}
    for left in ("train", "val", "test"):
        left_samples = set(split_df.loc[split_df["split"] == left, "sample_id"])
        left_buildings = set(split_df.loc[split_df["split"] == left, "building_uid"])
        for right in ("train", "val", "test"):
            if left >= right:
                continue
            right_samples = set(split_df.loc[split_df["split"] == right, "sample_id"])
            right_buildings = set(split_df.loc[split_df["split"] == right, "building_uid"])
            sample_overlap[f"{left}-{right}"] = len(left_samples & right_samples)
            building_overlap[f"{left}-{right}"] = len(left_buildings & right_buildings)

    if any(sample_overlap.values()):
        fail(f"sample_id leakage detected: {json.dumps(sample_overlap, sort_keys=True)}")
    if any(building_overlap.values()):
        fail(f"building_uid leakage detected: {json.dumps(building_overlap, sort_keys=True)}")


def validate_runtime(args: argparse.Namespace) -> None:
    datasets = build_datasets(
        split_metadata_path=args.splits,
        image_size=args.image_size,
        return_metadata=True,
        dataset_root=args.dataset_root,
    )

    for split_name in ("train", "val", "test"):
        dataset = datasets[split_name]
        indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
        for index in indices:
            item = dataset[index]
            if tuple(item["pre_image"].shape) != (3, args.image_size, args.image_size):
                fail(f"Unexpected pre_image shape for {split_name}[{index}]: {tuple(item['pre_image'].shape)}")
            if tuple(item["post_image"].shape) != (3, args.image_size, args.image_size):
                fail(f"Unexpected post_image shape for {split_name}[{index}]: {tuple(item['post_image'].shape)}")
            label_value = int(item["label"].item())
            if label_value not in {0, 1, 2, 3}:
                fail(f"Unexpected label value for {split_name}[{index}]: {label_value}")

    dataloaders = build_dataloaders(
        split_metadata_path=args.splits,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        return_metadata=True,
        dataset_root=args.dataset_root,
    )
    for split_name, loader in (("train", dataloaders.train_loader), ("val", dataloaders.val_loader), ("test", dataloaders.test_loader)):
        batch = next(iter(loader))
        if tuple(batch["pre_image"].shape[1:]) != (3, args.image_size, args.image_size):
            fail(f"Unexpected {split_name} batch pre_image shape: {tuple(batch['pre_image'].shape)}")
        if tuple(batch["post_image"].shape[1:]) != (3, args.image_size, args.image_size):
            fail(f"Unexpected {split_name} batch post_image shape: {tuple(batch['post_image'].shape)}")


def main() -> None:
    args = parse_args()
    metadata_path = require_file(args.metadata, "metadata.csv")
    splits_path = require_file(args.splits, "metadata_splits.csv")

    metadata_df = pd.read_csv(metadata_path)
    split_df = pd.read_csv(splits_path)

    validate_columns(metadata_df, REQUIRED_METADATA_COLUMNS, "metadata.csv")
    validate_columns(split_df, REQUIRED_SPLIT_COLUMNS, "metadata_splits.csv")

    if int(metadata_df.duplicated(subset=["sample_id", "building_uid"]).sum()) != 0:
        fail("metadata.csv contains duplicate (sample_id, building_uid) rows")
    if int(split_df.duplicated(subset=["sample_id", "building_uid"]).sum()) != 0:
        fail("metadata_splits.csv contains duplicate (sample_id, building_uid) rows")

    validate_split_overlap(split_df)
    validate_path_columns(metadata_df, args.dataset_root, ("pre_image_path", "post_image_path", "label_json_path"))
    validate_path_columns(split_df, args.dataset_root, ("pre_image_path", "post_image_path", "label_json_path"))
    validate_runtime(args)

    print("Data foundation check passed.")
    print(f"metadata_rows={len(metadata_df)}")
    print(f"split_rows={len(split_df)}")
    print(json.dumps(split_df["split"].value_counts().sort_index().to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()

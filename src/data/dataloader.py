from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import XBDPairBuildingDataset
from src.data.transforms import build_transforms

DEFAULT_DATA_CONFIG_PATH = "configs/data.yaml"
DEFAULT_SPLIT_METADATA_PATH = "data/splits/metadata_splits.csv"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0
DEFAULT_PIN_MEMORY = False
DEFAULT_RANDOM_STATE = 42


@dataclass
class DataLoadersBundle:
    train_dataset: XBDPairBuildingDataset
    val_dataset: XBDPairBuildingDataset
    test_dataset: XBDPairBuildingDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_sampler: WeightedRandomSampler
    class_weights: torch.Tensor


def load_split_metadata(split_metadata_path: str | Path = DEFAULT_SPLIT_METADATA_PATH) -> pd.DataFrame:
    split_metadata_path = Path(split_metadata_path)
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")

    df = pd.read_csv(split_metadata_path)
    required_columns = {"split", "class_id", "damage_class", "sample_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in split metadata: {missing}")

    return df


def filter_split_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split_df = df.loc[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No samples found for split='{split}'")

    return split_df.reset_index(drop=True)


def build_split_dataset(
    split_df: pd.DataFrame,
    metadata_csv: str | Path,
    split: str,
    transforms: Any,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
    return_metadata: bool = True,
    dataset_root: str | Path | None = None,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
) -> XBDPairBuildingDataset:
    dataset = XBDPairBuildingDataset(
        metadata_csv=metadata_csv,
        split=split,
        transforms=transforms,
        context_ratio=context_ratio,
        min_crop_size=min_crop_size,
        return_metadata=return_metadata,
        dataset_root=dataset_root,
        config_path=data_config_path,
    )
    dataset.df = split_df.reset_index(drop=True)
    return dataset


def compute_class_weights_from_dataframe(
    df: pd.DataFrame,
    label_column: str = "class_id",
    num_classes: int | None = None,
) -> torch.Tensor:
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataframe.")

    label_counts = df[label_column].value_counts().sort_index()
    if label_counts.empty:
        raise ValueError("Cannot compute class weights from an empty dataframe.")

    if num_classes is None:
        num_classes = int(label_counts.index.max()) + 1

    missing_classes = [class_index for class_index in range(num_classes) if class_index not in label_counts.index]
    if missing_classes:
        raise ValueError(
            f"Train split is missing classes required for weighting: {missing_classes}"
        )

    total_samples = int(label_counts.sum())
    class_weights = torch.zeros(num_classes, dtype=torch.float32)

    for class_index in range(num_classes):
        class_count = int(label_counts.loc[class_index])
        class_weights[class_index] = total_samples / (num_classes * class_count)

    return class_weights


def build_weighted_random_sampler(
    train_df: pd.DataFrame,
    class_weights: torch.Tensor,
    label_column: str = "class_id",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> WeightedRandomSampler:
    sample_weights = class_weights[train_df[label_column].to_numpy()].double()
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def build_datasets(
    split_metadata_path: str | Path = DEFAULT_SPLIT_METADATA_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
    return_metadata: bool = True,
    dataset_root: str | Path | None = None,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
) -> dict[str, XBDPairBuildingDataset]:
    split_df = load_split_metadata(split_metadata_path)
    transforms = build_transforms({"image_size": image_size})

    datasets: dict[str, XBDPairBuildingDataset] = {}
    for split_name in ("train", "val", "test"):
        current_split_df = filter_split_dataframe(split_df, split_name)
        datasets[split_name] = build_split_dataset(
            split_df=current_split_df,
            metadata_csv=split_metadata_path,
            split=split_name,
            transforms=transforms[split_name],
            context_ratio=context_ratio,
            min_crop_size=min_crop_size,
            return_metadata=return_metadata,
            dataset_root=dataset_root,
            data_config_path=data_config_path,
        )

    return datasets


def build_dataloaders(
    split_metadata_path: str | Path = DEFAULT_SPLIT_METADATA_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    pin_memory: bool = DEFAULT_PIN_MEMORY,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
    return_metadata: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
    dataset_root: str | Path | None = None,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG_PATH,
) -> DataLoadersBundle:
    datasets = build_datasets(
        split_metadata_path=split_metadata_path,
        image_size=image_size,
        context_ratio=context_ratio,
        min_crop_size=min_crop_size,
        return_metadata=return_metadata,
        dataset_root=dataset_root,
        data_config_path=data_config_path,
    )

    full_split_df = load_split_metadata(split_metadata_path)
    num_classes = int(full_split_df["class_id"].max()) + 1
    class_weights = compute_class_weights_from_dataframe(
        datasets["train"].df,
        label_column="class_id",
        num_classes=num_classes,
    )
    train_sampler = build_weighted_random_sampler(
        datasets["train"].df,
        class_weights=class_weights,
        label_column="class_id",
        random_state=random_state,
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoadersBundle(
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
        test_dataset=datasets["test"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        class_weights=class_weights,
    )

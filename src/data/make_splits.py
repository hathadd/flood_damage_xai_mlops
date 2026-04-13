from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DEFAULT_CONFIG_PATH = "configs/data.yaml"
DEFAULT_OUTPUT_PATH = "data/splits/metadata_splits.csv"
DEFAULT_RANDOM_STATE = 42


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/val/test splits grouped by sample_id."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the project data config.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Optional override for the input metadata CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the split metadata CSV will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of sample_ids assigned to train.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of sample_ids assigned to validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of sample_ids assigned to test.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="sample_id",
        help="Grouping column used to prevent leakage across related samples.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="class_id",
        help="Label column used to derive a scene-level stratification target.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_metadata_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.metadata_path is not None:
        return Path(args.metadata_path)

    metadata_path = config.get("paths", {}).get("metadata_out")
    if not metadata_path:
        raise ValueError("Could not resolve metadata path from arguments or config file.")

    return Path(metadata_path)


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, but received {total:.6f}."
        )

    for split_name, split_ratio in {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }.items():
        if split_ratio <= 0:
            raise ValueError(f"{split_name} ratio must be strictly positive.")


def build_group_metadata(
    df: pd.DataFrame,
    group_column: str,
    label_column: str,
) -> pd.DataFrame:
    required_columns = {group_column, label_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for split generation: {missing}")

    group_df = (
        df.groupby(group_column)
        .agg(
            group_label=(label_column, "max"),
            building_count=(group_column, "size"),
        )
        .reset_index()
    )

    return group_df


def maybe_get_stratify_labels(labels: pd.Series) -> pd.Series | None:
    label_counts = labels.value_counts()
    if label_counts.empty:
        return None

    if int(label_counts.min()) < 2:
        return None

    return labels


def split_groups(
    group_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    stratify_labels = maybe_get_stratify_labels(group_df["group_label"])

    train_groups, temp_groups = train_test_split(
        group_df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True,
        stratify=stratify_labels,
    )

    temp_stratify = maybe_get_stratify_labels(temp_groups["group_label"])
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)

    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        shuffle=True,
        stratify=temp_stratify,
    )

    return {
        "train": train_groups.reset_index(drop=True),
        "val": val_groups.reset_index(drop=True),
        "test": test_groups.reset_index(drop=True),
    }


def attach_split_column(
    df: pd.DataFrame,
    grouped_splits: dict[str, pd.DataFrame],
    group_column: str,
) -> pd.DataFrame:
    split_frames: list[pd.DataFrame] = []
    for split_name, split_df in grouped_splits.items():
        split_frames.append(split_df[[group_column]].assign(split=split_name))

    split_lookup = pd.concat(split_frames, ignore_index=True)
    merged_df = df.merge(split_lookup, on=group_column, how="left", validate="many_to_one")

    if merged_df["split"].isna().any():
        missing_groups = merged_df.loc[merged_df["split"].isna(), group_column].unique()
        raise ValueError(
            f"Some groups were not assigned to a split: {missing_groups[:5].tolist()}"
        )

    return merged_df


def validate_split_integrity(
    split_df: pd.DataFrame,
    group_column: str,
) -> None:
    split_sets = {
        split_name: set(split_part[group_column].unique())
        for split_name, split_part in split_df.groupby("split")
    }

    expected_splits = {"train", "val", "test"}
    if set(split_sets) != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, found {set(split_sets)}")

    for left_name in expected_splits:
        for right_name in expected_splits:
            if left_name >= right_name:
                continue
            overlap = split_sets[left_name] & split_sets[right_name]
            if overlap:
                raise ValueError(
                    f"Data leakage detected: {left_name} and {right_name} share groups {sorted(overlap)[:5]}"
                )


def print_split_summary(
    split_df: pd.DataFrame,
    group_column: str,
) -> None:
    row_summary = split_df["split"].value_counts().reindex(["train", "val", "test"])
    group_summary = (
        split_df.groupby("split")[group_column]
        .nunique()
        .reindex(["train", "val", "test"])
    )
    class_summary = (
        split_df.groupby(["split", "damage_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )

    print("Samples per split:")
    print(row_summary.to_string())
    print()

    print(f"Unique {group_column} per split:")
    print(group_summary.to_string())
    print()

    print("Class distribution per split:")
    print(class_summary.to_string())


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    config = load_config(args.config)
    metadata_path = resolve_metadata_path(args, config)
    output_path = Path(args.output_path)

    df = pd.read_csv(metadata_path)
    group_df = build_group_metadata(
        df=df,
        group_column=args.group_column,
        label_column=args.label_column,
    )

    grouped_splits = split_groups(
        group_df=group_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    split_df = attach_split_column(
        df=df,
        grouped_splits=grouped_splits,
        group_column=args.group_column,
    )
    validate_split_integrity(split_df, group_column=args.group_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved split metadata to: {output_path}")
    print_split_summary(split_df, group_column=args.group_column)


if __name__ == "__main__":
    main()

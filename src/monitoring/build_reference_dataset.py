from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.monitoring.utils import compute_reference_row_features, load_monitoring_config, save_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the tabular reference dataset for monitoring.")
    parser.add_argument("--config-path", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--data-config-path", type=str, default="configs/data.yaml")
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-image-statistics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_monitoring_config(args.config_path)
    metadata_path = args.metadata_path or config["paths"]["metadata_path"]
    output_path = args.output_path or config["paths"]["reference_data_path"]
    selected_splits = args.splits or config.get("reference", {}).get("splits", ["train", "test"])
    include_image_statistics = not args.skip_image_statistics and config.get("reference", {}).get(
        "include_image_statistics",
        True,
    )

    metadata_df = pd.read_csv(metadata_path)
    filtered_df = metadata_df[metadata_df["split"].isin(selected_splits)].copy()
    if args.max_rows is not None:
        filtered_df = filtered_df.head(args.max_rows)

    records = [
        compute_reference_row_features(
            row=row,
            dataset_root=args.dataset_root,
            config_path=args.data_config_path,
            include_image_statistics=include_image_statistics,
        )
        for _, row in filtered_df.iterrows()
    ]
    reference_df = pd.DataFrame(records)
    saved_path = save_table(reference_df, output_path)
    print(
        "Reference monitoring dataset saved to "
        f"{Path(saved_path).as_posix()} "
        f"with {len(reference_df)} rows from splits {selected_splits}."
    )


if __name__ == "__main__":
    main()

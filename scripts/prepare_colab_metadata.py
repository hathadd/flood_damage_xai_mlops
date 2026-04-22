from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_PATH = "data/splits/metadata_splits.csv"
DEFAULT_OUTPUT_PATH = "data/splits/metadata_splits_colab.csv"
PATH_COLUMNS = ("pre_image_path", "post_image_path", "label_json_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an optional Colab-friendly copy of metadata_splits.csv. "
            "Portable relative-path metadata can be used directly on Colab with "
            "--dataset-root, so this helper is only needed for legacy absolute-path CSVs."
        )
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Original dataset root prefix stored in a legacy absolute-path CSV.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=None,
        help="Target dataset root, for example a Drive mount on Colab.",
    )
    return parser.parse_args()


def normalize_prefix(path_text: str) -> str:
    return str(path_text).replace("\\", "/").rstrip("/")


def is_absolute_path(path_text: str) -> bool:
    normalized_path = normalize_prefix(path_text)
    return normalized_path.startswith("/") or (
        len(normalized_path) >= 3 and normalized_path[1] == ":" and normalized_path[2] == "/"
    )


def remap_path(path_text: str, source_root: str | None, target_root: str | None) -> str:
    normalized_path = normalize_prefix(path_text)
    if not is_absolute_path(normalized_path):
        return normalized_path

    if source_root is None or target_root is None:
        return normalized_path

    normalized_source = normalize_prefix(source_root)
    normalized_target = normalize_prefix(target_root)
    if normalized_path.startswith(normalized_source):
        relative_path = normalized_path[len(normalized_source) :].lstrip("/")
        return str(Path(normalized_target) / relative_path)

    return normalized_path


def main() -> None:
    args = parse_args()
    if (args.source_root is None) ^ (args.target_root is None):
        raise ValueError("--source-root and --target-root must be provided together.")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata file not found: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = set(PATH_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing path columns in metadata: {missing_columns}")

    for column in PATH_COLUMNS:
        df[column] = df[column].apply(
            lambda value: remap_path(
                path_text=str(value),
                source_root=args.source_root,
                target_root=args.target_root,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved Colab metadata copy to: {output_path}")
    print(f"Rows: {len(df)}")
    print("Updated columns:", ", ".join(PATH_COLUMNS))
    if args.source_root is None:
        print("Paths were already portable; the CSV was copied unchanged.")


if __name__ == "__main__":
    main()

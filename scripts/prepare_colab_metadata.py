from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_PATH = "data/splits/metadata_splits.csv"
DEFAULT_OUTPUT_PATH = "data/splits/metadata_splits_colab.csv"
PATH_COLUMNS = ("pre_image_path", "post_image_path", "label_json_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Colab-friendly copy of metadata_splits.csv with Drive-mounted dataset paths."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Original root prefix currently stored in the CSV, for example C:/Users/.../flooding_dataset.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        required=True,
        help="Dataset root as mounted in Colab/Drive, for example /content/drive/MyDrive/flooding_dataset/flooding_dataset.",
    )
    return parser.parse_args()


def normalize_prefix(path_text: str) -> str:
    return str(path_text).replace("\\", "/").rstrip("/")


def remap_path(path_text: str, source_root: str, target_root: str) -> str:
    normalized_path = normalize_prefix(path_text)
    normalized_source = normalize_prefix(source_root)
    normalized_target = normalize_prefix(target_root)

    if normalized_path.startswith(normalized_source):
        relative_path = normalized_path[len(normalized_source) :].lstrip("/")
        return str(Path(normalized_target) / relative_path)

    return path_text


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

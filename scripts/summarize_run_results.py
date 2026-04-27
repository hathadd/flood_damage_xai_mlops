from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_RUNS = [
    ("run_a", "reports/evaluation/run_a/metrics.json"),
    ("run_b", "reports/evaluation/run_b/metrics.json"),
    ("run_c_bit", "reports/evaluation/run_c_bit/metrics.json"),
]
OUTPUT_COLUMNS = [
    "run_name",
    "model_type",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics across known runs.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--output-path", type=str, default="reports/evaluation/comparison_summary.csv")
    return parser.parse_args()


def build_row(run_key: str, metrics_path: Path) -> dict[str, str]:
    row = {column: "missing" for column in OUTPUT_COLUMNS}
    row["run_name"] = run_key

    if not metrics_path.exists():
        return row

    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return row

    row["run_name"] = str(payload.get("run_name", run_key))
    row["model_type"] = str(payload.get("model_type", "missing"))
    for column in OUTPUT_COLUMNS[2:]:
        value = payload.get(column, "missing")
        row[column] = str(value)
    return row


def print_table(rows: list[dict[str, str]]) -> None:
    widths: dict[str, int] = {}
    for column in OUTPUT_COLUMNS:
        widths[column] = max(len(column), *(len(row.get(column, "")) for row in rows))

    header = " | ".join(column.ljust(widths[column]) for column in OUTPUT_COLUMNS)
    separator = "-+-".join("-" * widths[column] for column in OUTPUT_COLUMNS)
    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(row.get(column, "").ljust(widths[column]) for column in OUTPUT_COLUMNS))


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = (project_root / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)

    rows: list[dict[str, str]] = []
    for run_key, relative_metrics_path in EXPECTED_RUNS:
        metrics_path = project_root / relative_metrics_path
        rows.append(build_row(run_key, metrics_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print(f"\nSaved comparison summary to: {output_path}")


if __name__ == "__main__":
    main()

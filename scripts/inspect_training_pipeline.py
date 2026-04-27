from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

KNOWN_CHECKPOINTS = {
    "run_a": "outputs/focal/checkpoints/best_siamese_resnet18.pt",
    "run_b": "outputs/focal_run_b_regularized/checkpoints/best_siamese_resnet18.pt",
    "run_c_bit": "outputs/focal_run_c_bit_transformer/checkpoints/best_bit_run_c.pt",
}
KNOWN_OUTPUT_DIRS = {
    "run_a": "outputs/focal",
    "run_b": "outputs/focal_run_b_regularized",
    "run_c_bit": "outputs/focal_run_c_bit_transformer",
}
KNOWN_EVAL_REPORTS = {
    "run_a": "reports/evaluation/run_a/metrics.json",
    "run_b": "reports/evaluation/run_b/metrics.json",
    "run_c_bit": "reports/evaluation/run_c_bit/metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the project training pipeline without training.")
    parser.add_argument("--split-metadata-path", type=str, default="data/splits/metadata_splits.csv")
    parser.add_argument("--project-root", type=str, default=".")
    return parser.parse_args()


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def list_files(paths: Iterable[Path], root: Path) -> None:
    paths = list(paths)
    if not paths:
        print("missing")
        return
    for path in paths:
        print(f"- {relpath(path, root)}")


def load_split_summary(split_path: Path) -> dict[str, object] | None:
    if not split_path.exists():
        return None

    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        split_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        class_by_split: dict[str, dict[str, int]] = {}
        row_count = 0

        for row in reader:
            row_count += 1
            split = row.get("split", "") or "missing"
            damage_class = row.get("damage_class", "") or "missing"
            split_counts[split] = split_counts.get(split, 0) + 1
            class_counts[damage_class] = class_counts.get(damage_class, 0) + 1
            split_bucket = class_by_split.setdefault(split, {})
            split_bucket[damage_class] = split_bucket.get(damage_class, 0) + 1

    return {
        "row_count": row_count,
        "columns": columns,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "class_by_split": class_by_split,
    }


def print_json_file_summary(path: Path, root: Path) -> None:
    if not path.exists():
        print(f"- {relpath(path, root)}: missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metric_keys = [
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1",
            "test_macro_precision",
            "test_macro_recall",
            "test_loss",
        ]
        metrics = {key: payload.get(key, "missing") for key in metric_keys}
        print(f"- {relpath(path, root)}: {metrics}")
    except Exception as exc:
        print(f"- {relpath(path, root)}: unreadable ({exc})")


def detect_model_registry_artifacts(mlruns_root: Path) -> list[Path]:
    if not mlruns_root.exists():
        return []

    matches: list[Path] = []
    direct_models_dir = mlruns_root / "models"
    if direct_models_dir.exists():
        matches.append(direct_models_dir)

    for pattern in ("**/MLmodel", "**/artifacts/model", "**/registered_model_meta.yaml"):
        for match in mlruns_root.glob(pattern):
            matches.append(match)
    deduped = sorted({path.resolve() for path in matches})
    return [Path(path) for path in deduped]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    split_path = (project_root / args.split_metadata_path).resolve() if not Path(args.split_metadata_path).is_absolute() else Path(args.split_metadata_path)

    print_section("Project Root")
    print(project_root)

    print_section("Available Run Files")
    training_scripts = sorted((project_root / "src" / "training").glob("train*.py"))
    list_files(training_scripts, project_root)

    print_section("Detected Transform Files")
    transform_files = sorted((project_root / "src" / "data").glob("transforms*.py"))
    list_files(transform_files, project_root)

    print_section("Detected Model Files")
    model_files = sorted((project_root / "src" / "models").glob("*.py"))
    list_files(model_files, project_root)

    print_section("Known Checkpoints")
    for run_name, relative_path in KNOWN_CHECKPOINTS.items():
        checkpoint_path = project_root / relative_path
        status = relpath(checkpoint_path, project_root) if checkpoint_path.exists() else "missing"
        print(f"- {run_name}: {status}")

    print_section("Known Evaluation Reports")
    for run_name, relative_path in KNOWN_EVAL_REPORTS.items():
        report_path = project_root / relative_path
        print_json_file_summary(report_path, project_root)

    print_section("Metadata Split Summary")
    summary = load_split_summary(split_path)
    if summary is None:
        print(f"- split file missing: {split_path}")
    else:
        print(f"- split file: {relpath(split_path, project_root)}")
        print(f"- total rows: {summary['row_count']}")
        print(f"- columns: {', '.join(summary['columns'])}")
        print(f"- split counts: {summary['split_counts']}")
        print(f"- class counts: {summary['class_counts']}")
        print("- class counts by split:")
        for split_name, counts in summary["class_by_split"].items():
            print(f"  - {split_name}: {counts}")

    print_section("Known Output Paths")
    for run_name, relative_path in KNOWN_OUTPUT_DIRS.items():
        output_path = project_root / relative_path
        status = relpath(output_path, project_root) if output_path.exists() else "missing"
        print(f"- {run_name}: {status}")

    print_section("Grad-CAM Outputs")
    gradcam_root = project_root / "reports" / "xai" / "gradcam" / "run_b"
    gradcam_summary = gradcam_root / "gradcam_summary.csv"
    gradcam_report = gradcam_root / "selected_samples_report.txt"
    print(f"- summary csv: {relpath(gradcam_summary, project_root) if gradcam_summary.exists() else 'missing'}")
    print(f"- selected samples report: {relpath(gradcam_report, project_root) if gradcam_report.exists() else 'missing'}")

    print_section("MLflow / Registry Artifacts")
    mlruns_root = project_root / "mlruns"
    if not mlruns_root.exists():
        print("- mlruns: missing")
    else:
        artifacts = detect_model_registry_artifacts(mlruns_root)
        if not artifacts:
            print("- no explicit registry/model artifacts detected under mlruns")
        else:
            print("- detected registry/model artifacts:")
            for artifact in artifacts[:20]:
                print(f"  - {relpath(artifact, project_root)}")
            if len(artifacts) > 20:
                print(f"  - ... {len(artifacts) - 20} more")


if __name__ == "__main__":
    main()

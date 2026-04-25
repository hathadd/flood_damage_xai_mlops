from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.models.siamese_model import SiameseResNet18

MODEL_METADATA_TAGS = {
    "task": "flood_damage_classification",
    "model_family": "siamese_resnet18",
    "selected_run": "run_b",
    "dataset": "xbd_flooding_subset",
    "input_type": "pre_post_satellite_crops",
    "classes": "no-damage, minor-damage, major-damage, destroyed",
    "final_model": "true",
}
SUPPORTED_STAGES = {"none", "staging", "production"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the final Run B model in MLflow Model Registry.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="run_b_siamese_resnet18_regularized")
    parser.add_argument("--registered-model-name", type=str, default="flood_damage_siamese_resnet18")
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, default="model")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--stage", type=str, default="None")
    parser.add_argument("--test-metrics-path", type=str, default=None)
    return parser.parse_args()


def normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported --stage value: {stage}. Expected one of: None, Staging, Production.")
    return normalized


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def extract_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
            return normalize_state_dict_keys(checkpoint_obj["model_state_dict"])
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return normalize_state_dict_keys(checkpoint_obj["state_dict"])
        if checkpoint_obj and all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return normalize_state_dict_keys(checkpoint_obj)
    raise ValueError("Unsupported checkpoint format. Expected raw state_dict or dict containing model_state_dict/state_dict.")


def load_model(checkpoint_path: str | Path, num_classes: int, dropout: float) -> SiameseResNet18:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    model = SiameseResNet18(num_classes=num_classes, pretrained=False, dropout=dropout)
    model.load_state_dict(extract_state_dict(checkpoint_obj), strict=True)
    model.eval()
    return model


def load_test_metrics(test_metrics_path: str | Path | None) -> dict[str, float]:
    if test_metrics_path is None:
        return {}

    metrics_path = Path(test_metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Test metrics file not found: {metrics_path}")

    metrics_obj = json.loads(metrics_path.read_text(encoding="utf-8"))
    metric_keys = [
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_macro_precision",
        "test_macro_recall",
        "test_loss",
    ]
    metrics: dict[str, float] = {}
    for key in metric_keys:
        if key in metrics_obj:
            metrics[key] = float(metrics_obj[key])
    return metrics


def ensure_registered_model(client: MlflowClient, registered_model_name: str) -> None:
    try:
        client.create_registered_model(registered_model_name)
    except MlflowException as exc:
        message = str(exc).lower()
        if "already exists" not in message and "resource already exists" not in message:
            raise


def find_latest_model_version(
    client: MlflowClient,
    registered_model_name: str,
    run_id: str,
) -> str:
    versions = client.search_model_versions(f"name = '{registered_model_name}'")
    matching_versions = [version for version in versions if version.run_id == run_id]
    if not matching_versions:
        raise RuntimeError(
            f"Could not locate a registered model version for name='{registered_model_name}' and run_id='{run_id}'."
        )
    latest = max(matching_versions, key=lambda item: int(item.version))
    return str(latest.version)


def apply_stage_or_tag(
    client: MlflowClient,
    registered_model_name: str,
    model_version: str,
    stage: str,
) -> str:
    if stage == "none":
        client.set_model_version_tag(registered_model_name, model_version, "lifecycle_stage", "none")
        return "tag:lifecycle_stage=none"

    desired_stage = stage.capitalize()
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version,
            stage=desired_stage,
            archive_existing_versions=False,
        )
        client.set_model_version_tag(registered_model_name, model_version, "lifecycle_stage", stage)
        return desired_stage
    except Exception:
        client.set_model_version_tag(registered_model_name, model_version, "lifecycle_stage", stage)
        return f"tag:lifecycle_stage={stage}"


def main() -> None:
    args = parse_args()
    stage = normalize_stage(args.stage)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=args.mlflow_tracking_uri)

    model = load_model(
        checkpoint_path=args.checkpoint_path,
        num_classes=args.num_classes,
        dropout=args.dropout,
    )
    test_metrics = load_test_metrics(args.test_metrics_path)

    ensure_registered_model(client, args.registered_model_name)

    with mlflow.start_run(run_name=args.model_name) as run:
        run_id = run.info.run_id
        mlflow.set_tags(
            {
                **MODEL_METADATA_TAGS,
                "model_name": args.model_name,
                "registered_model_name": args.registered_model_name,
                "artifact_path": args.artifact_path,
                "checkpoint_path": args.checkpoint_path,
            }
        )
        mlflow.log_params(
            {
                "num_classes": args.num_classes,
                "dropout": args.dropout,
                "checkpoint_path": args.checkpoint_path,
                "registered_model_name": args.registered_model_name,
                "artifact_path": args.artifact_path,
                "requested_stage": stage,
            }
        )
        if test_metrics:
            mlflow.log_metrics(test_metrics)

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=args.artifact_path,
            registered_model_name=args.registered_model_name,
        )

    model_version = find_latest_model_version(
        client=client,
        registered_model_name=args.registered_model_name,
        run_id=run_id,
    )

    for key, value in MODEL_METADATA_TAGS.items():
        client.set_model_version_tag(args.registered_model_name, model_version, key, value)
    client.set_model_version_tag(args.registered_model_name, model_version, "model_name", args.model_name)
    if test_metrics:
        for metric_name, metric_value in test_metrics.items():
            client.set_model_version_tag(args.registered_model_name, model_version, metric_name, str(metric_value))

    stage_result = apply_stage_or_tag(
        client=client,
        registered_model_name=args.registered_model_name,
        model_version=model_version,
        stage=stage,
    )

    print(f"run_id: {run_id}")
    print(f"registered_model_name: {args.registered_model_name}")
    print(f"model_version: {model_version}")
    print(f"stage_or_tag_applied: {stage_result}")
    print(f"artifact_uri: {model_info.model_uri}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import XBDPairBuildingDataset
from src.models.bit_transformer_run_c import BITTransformerRunC
from src.models.siamese_model import SiameseResNet18

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
SUPPORTED_MODEL_TYPES = {"siamese_resnet18", "bit_transformer_run_c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one trained run on the test split only.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-type", type=str, choices=sorted(SUPPORTED_MODEL_TYPES), required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split-metadata-path", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def load_split_metadata(split_metadata_path: str | Path) -> pd.DataFrame:
    split_metadata_path = Path(split_metadata_path)
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")

    df = pd.read_csv(split_metadata_path)
    required_columns = {
        "split",
        "sample_id",
        "building_uid",
        "class_id",
        "damage_class",
        "pre_image_path",
        "post_image_path",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in split metadata: {missing}")
    return df


def build_test_dataset(
    split_metadata_path: str | Path,
    dataset_root: str | Path,
    image_size: int,
) -> XBDPairBuildingDataset:
    split_df = load_split_metadata(split_metadata_path)
    test_df = split_df.loc[split_df["split"] == "test"].copy().reset_index(drop=True)
    if test_df.empty:
        raise ValueError("No samples found for split='test'.")

    dataset = XBDPairBuildingDataset(
        metadata_csv=split_metadata_path,
        split="test",
        transforms=build_eval_transforms(image_size=image_size),
        return_metadata=True,
        dataset_root=dataset_root,
    )
    dataset.df = test_df
    return dataset


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "siamese_resnet18":
        return SiameseResNet18(
            num_classes=args.num_classes,
            pretrained=False,
            dropout=args.dropout,
        )

    if args.model_type == "bit_transformer_run_c":
        return BITTransformerRunC(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=3,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
        )

    raise ValueError(f"Unsupported model_type: {args.model_type}")


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
        if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], dict):
            return normalize_state_dict_keys(checkpoint_obj["model"])
        if checkpoint_obj and all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return normalize_state_dict_keys(checkpoint_obj)
    raise ValueError("Unsupported checkpoint format. Expected a raw state_dict or a dict containing model_state_dict/state_dict.")


def load_model_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any] | None:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint_obj if isinstance(checkpoint_obj, dict) else None


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    prediction_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        pre_image = batch["pre_image"].to(device, non_blocking=True)
        post_image = batch["post_image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(pre_image, post_image)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1).values

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        labels_np = labels.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()
        probabilities_np = probabilities.detach().cpu().numpy()
        confidences_np = confidences.detach().cpu().numpy()

        all_targets.extend(labels_np.tolist())
        all_predictions.extend(predictions_np.tolist())
        all_probabilities.extend(probabilities_np.tolist())

        for idx in range(batch_size):
            true_label = int(labels_np[idx])
            pred_label = int(predictions_np[idx])
            probs = probabilities_np[idx]
            prediction_rows.append(
                {
                    "sample_id": batch["sample_id"][idx],
                    "building_uid": batch["building_uid"][idx],
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "pred_label": pred_label,
                    "pred_class": CLASS_NAMES[pred_label],
                    "confidence": float(confidences_np[idx]),
                    "prob_no_damage": float(probs[0]),
                    "prob_minor_damage": float(probs[1]),
                    "prob_major_damage": float(probs[2]),
                    "prob_destroyed": float(probs[3]),
                }
            )

    if total_samples == 0:
        raise ValueError("No samples were evaluated from the test split.")

    y_true = np.asarray(all_targets, dtype=np.int64)
    y_pred = np.asarray(all_predictions, dtype=np.int64)
    confusion = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        zero_division=0,
    )

    metrics = {
        "test_loss": total_loss / total_samples,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": report_dict,
        "classification_report_text": report_text,
    }

    predictions_df = pd.DataFrame(prediction_rows)
    return metrics, predictions_df, confusion


def save_confusion_matrix_figure(matrix: np.ndarray, output_path: str | Path, class_names: list[str]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, col_index] > threshold else "black"
            ax.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_outputs(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    predictions_df: pd.DataFrame,
    confusion: np.ndarray,
) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    report_csv_path = output_dir / "classification_report.csv"
    report_txt_path = output_dir / "classification_report.txt"
    confusion_path = output_dir / "confusion_matrix.png"
    predictions_path = output_dir / "predictions.csv"

    serializable_metrics = {
        "run_name": args.run_name,
        "model_type": args.model_type,
        "checkpoint_path": args.checkpoint_path,
        "split": "test",
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["test_accuracy"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_weighted_f1": metrics["test_weighted_f1"],
        "test_macro_precision": metrics["test_macro_precision"],
        "test_macro_recall": metrics["test_macro_recall"],
        "per_class": metrics["per_class"],
    }
    metrics_path.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")

    report_df = pd.DataFrame(metrics["per_class"]).transpose()
    report_df.to_csv(report_csv_path, index=True)
    report_txt_path.write_text(metrics["classification_report_text"], encoding="utf-8")
    predictions_df.to_csv(predictions_path, index=False)
    save_confusion_matrix_figure(confusion, confusion_path, CLASS_NAMES[: args.num_classes])

    return {
        "metrics": metrics_path,
        "classification_report_csv": report_csv_path,
        "classification_report_txt": report_txt_path,
        "confusion_matrix": confusion_path,
        "predictions": predictions_path,
    }


def print_summary(metrics: dict[str, Any], output_paths: dict[str, Path]) -> None:
    print("Evaluation completed on split=test")
    print(f"test_loss: {metrics['test_loss']:.6f}")
    print(f"test_accuracy: {metrics['test_accuracy']:.6f}")
    print(f"test_macro_f1: {metrics['test_macro_f1']:.6f}")
    print(f"test_weighted_f1: {metrics['test_weighted_f1']:.6f}")
    print(f"test_macro_precision: {metrics['test_macro_precision']:.6f}")
    print(f"test_macro_recall: {metrics['test_macro_recall']:.6f}")
    print("Saved outputs:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    dataset = build_test_dataset(
        split_metadata_path=args.split_metadata_path,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args).to(device)
    load_model_checkpoint(model=model, checkpoint_path=args.checkpoint_path, device=device)

    metrics, predictions_df, confusion = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=args.num_classes,
    )
    output_paths = save_outputs(args, metrics, predictions_df, confusion)
    print_summary(metrics, output_paths)


if __name__ == "__main__":
    main()

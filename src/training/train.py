from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloaders
from src.evaluation.metrics import accuracy_score, confusion_matrix, logits_to_predictions, macro_f1_score
from src.models.siamese_model import SiameseResNet18
from src.training.losses import build_loss


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Siamese ResNet18 flood damage classifier.")
    parser.add_argument(
        "--split-metadata-path",
        "--metadata-csv",
        dest="split_metadata_path",
        type=str,
        default="data/splits/metadata_splits.csv",
        help="Path to metadata_splits.csv. --metadata-csv is kept as a Colab-friendly alias.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--history-path", type=str, default="outputs/history/training_history.csv")
    parser.add_argument("--figures-dir", type=str, default="outputs/figures")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="mlruns")
    parser.add_argument("--mlflow-experiment-name", type=str, default="flood_damage_xai_mlops")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_mlflow_tracking_uri(tracking_uri: str) -> str:
    if "://" in tracking_uri or tracking_uri.startswith("databricks"):
        return tracking_uri
    return Path(tracking_uri).resolve().as_uri()


def build_mlflow_params(args: argparse.Namespace, device: torch.device) -> dict[str, str | int | float | bool]:
    return {
        "split_metadata_path": args.split_metadata_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "pretrained": args.pretrained,
        "num_classes": args.num_classes,
        "random_state": args.random_state,
        "device": str(device),
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
    }


def log_epoch_metrics_to_mlflow(epoch: int, train_result: EpochResult, val_result: EpochResult) -> None:
    mlflow.log_metrics(
        {
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_macro_f1": train_result.macro_f1,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_macro_f1": val_result.macro_f1,
        },
        step=epoch,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_image = batch["pre_image"].to(device, non_blocking=True)
    post_image = batch["post_image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return pre_image, post_image, labels


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        pre_image, post_image, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pre_image, post_image)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_predictions.append(logits_to_predictions(logits).detach().cpu())
        all_targets.append(labels.detach().cpu())

    return summarize_epoch(total_loss, total_samples, all_predictions, all_targets, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        pre_image, post_image, labels = move_batch_to_device(batch, device)
        logits = model(pre_image, post_image)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_predictions.append(logits_to_predictions(logits).detach().cpu())
        all_targets.append(labels.detach().cpu())

    return summarize_epoch(total_loss, total_samples, all_predictions, all_targets, num_classes)


def summarize_epoch(
    total_loss: float,
    total_samples: int,
    all_predictions: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    num_classes: int,
) -> EpochResult:
    if total_samples == 0:
        raise ValueError("No samples were processed during the epoch.")

    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    return EpochResult(
        loss=total_loss / total_samples,
        accuracy=accuracy_score(predictions, targets),
        macro_f1=macro_f1_score(predictions, targets, num_classes=num_classes),
        confusion_matrix=confusion_matrix(predictions, targets, num_classes=num_classes),
    )


def build_history_row(epoch: int, train_result: EpochResult, val_result: EpochResult) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": train_result.loss,
        "train_accuracy": train_result.accuracy,
        "train_macro_f1": train_result.macro_f1,
        "val_loss": val_result.loss,
        "val_accuracy": val_result.accuracy,
        "val_macro_f1": val_result.macro_f1,
    }


def save_history_csv(history: list[dict[str, float | int]], history_path: str | Path) -> Path:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]

    with open(history_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return history_path


def plot_metric_curve(
    history: list[dict[str, float | int]],
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_values = [float(row[train_key]) for row in history]
    val_values = [float(row[val_key]) for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label="train")
    plt.plot(epochs, val_values, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_training_curves(history: list[dict[str, float | int]], figures_dir: str | Path) -> dict[str, Path]:
    if not history:
        raise ValueError("Cannot save training curves from an empty history.")

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "loss": plot_metric_curve(
            history=history,
            train_key="train_loss",
            val_key="val_loss",
            ylabel="Loss",
            title="Training and Validation Loss",
            output_path=figures_dir / "loss_curve.png",
        ),
        "macro_f1": plot_metric_curve(
            history=history,
            train_key="train_macro_f1",
            val_key="val_macro_f1",
            ylabel="Macro F1-score",
            title="Training and Validation Macro F1-score",
            output_path=figures_dir / "f1_curve.png",
        ),
        "accuracy": plot_metric_curve(
            history=history,
            train_key="train_accuracy",
            val_key="val_accuracy",
            ylabel="Accuracy",
            title="Training and Validation Accuracy",
            output_path=figures_dir / "accuracy_curve.png",
        ),
    }


def save_checkpoint(
    output_dir: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_result: EpochResult,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_siamese_resnet18.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_macro_f1": val_result.macro_f1,
            "val_accuracy": val_result.accuracy,
            "args": vars(args),
        },
        checkpoint_path,
    )
    return checkpoint_path


def fit(args: argparse.Namespace) -> None:
    set_seed(args.random_state)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(resolve_mlflow_tracking_uri(args.mlflow_tracking_uri))
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        active_run = mlflow.active_run()
        if active_run is not None:
            print(f"MLflow run_id: {active_run.info.run_id}")

        mlflow.log_params(build_mlflow_params(args, device))

        dataloaders = build_dataloaders(
            split_metadata_path=args.split_metadata_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            random_state=args.random_state,
        )

        model = SiameseResNet18(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            dropout=args.dropout,
        ).to(device)
        criterion = build_loss(
            loss_type=args.loss_type,
            class_weights=dataloaders.class_weights.to(device),
            gamma=args.focal_gamma,
        )
        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        history: list[dict[str, float | int]] = []
        best_val_macro_f1 = -1.0
        best_val_accuracy = 0.0
        best_checkpoint_path: Path | None = None

        for epoch in range(1, args.epochs + 1):
            train_result = train_one_epoch(
                model=model,
                dataloader=dataloaders.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=args.num_classes,
                max_batches=args.max_train_batches,
            )
            val_result = evaluate(
                model=model,
                dataloader=dataloaders.val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                max_batches=args.max_val_batches,
            )

            history.append(build_history_row(epoch, train_result, val_result))
            log_epoch_metrics_to_mlflow(epoch, train_result, val_result)

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_result.loss:.4f} train_accuracy={train_result.accuracy:.4f} "
                f"train_macro_f1={train_result.macro_f1:.4f} | "
                f"val_loss={val_result.loss:.4f} val_accuracy={val_result.accuracy:.4f} "
                f"val_macro_f1={val_result.macro_f1:.4f}"
            )

            if val_result.macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_result.macro_f1
                best_val_accuracy = val_result.accuracy
                best_checkpoint_path = save_checkpoint(
                    output_dir=args.output_dir,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    val_result=val_result,
                    args=args,
                )
                print(f"Saved best checkpoint: {best_checkpoint_path}")

        history_path = save_history_csv(history, args.history_path)
        curve_paths = save_training_curves(history, args.figures_dir)

        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_accuracy": best_val_accuracy,
            }
        )
        mlflow.log_artifact(str(history_path), artifact_path="history")
        for curve_path in curve_paths.values():
            mlflow.log_artifact(str(curve_path), artifact_path="figures")
        if best_checkpoint_path is not None:
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

        print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
        if best_checkpoint_path is not None:
            print(f"Best checkpoint path: {best_checkpoint_path}")
        print(f"Final history CSV: {history_path}")
        print("Final training curves:", ", ".join(str(path) for path in curve_paths.values()))
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")


def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()

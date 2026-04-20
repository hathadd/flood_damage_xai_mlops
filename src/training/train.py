from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any  # IMP-5

import matplotlib
import mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW  # IMP-1
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR  # FIX-3 # IMP-2
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


class EarlyStopping:  # IMP-3
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:  # IMP-3
        self.patience = max(0, patience)  # IMP-3
        self.min_delta = min_delta  # IMP-3
        self.best_val_macro_f1 = -1.0  # IMP-3
        self.best_val_loss = float("inf")  # IMP-3
        self.counter = 0  # IMP-3
        self.should_stop = False  # IMP-3

    def is_improvement(self, val_macro_f1: float, val_loss: float) -> bool:  # IMP-3
        if val_macro_f1 > self.best_val_macro_f1 + self.min_delta:  # IMP-3
            return True  # IMP-3
        if abs(val_macro_f1 - self.best_val_macro_f1) <= self.min_delta and val_loss < self.best_val_loss:  # IMP-5
            return True  # IMP-5
        return False  # IMP-3

    def step(self, val_macro_f1: float, val_loss: float) -> bool:  # IMP-3
        improved = self.is_improvement(val_macro_f1=val_macro_f1, val_loss=val_loss)  # IMP-3
        if improved:  # IMP-3
            self.best_val_macro_f1 = val_macro_f1  # IMP-3
            self.best_val_loss = val_loss  # IMP-3
            self.counter = 0  # IMP-3
            self.should_stop = False  # IMP-3
            return True  # IMP-3

        self.counter += 1  # IMP-3
        self.should_stop = self.counter >= self.patience  # IMP-3
        return False  # IMP-3

    def state_dict(self) -> dict[str, float | int | bool]:  # IMP-5
        return {  # IMP-5
            "patience": self.patience,  # IMP-5
            "min_delta": self.min_delta,  # IMP-5
            "best_val_macro_f1": self.best_val_macro_f1,  # IMP-5
            "best_val_loss": self.best_val_loss,  # IMP-5
            "counter": self.counter,  # IMP-5
            "should_stop": self.should_stop,  # IMP-5
        }  # IMP-5

    def load_state_dict(self, state_dict: dict[str, float | int | bool]) -> None:  # IMP-5
        self.patience = int(state_dict.get("patience", self.patience))  # IMP-5
        self.min_delta = float(state_dict.get("min_delta", self.min_delta))  # IMP-5
        self.best_val_macro_f1 = float(state_dict.get("best_val_macro_f1", self.best_val_macro_f1))  # IMP-5
        self.best_val_loss = float(state_dict.get("best_val_loss", self.best_val_loss))  # IMP-5
        self.counter = int(state_dict.get("counter", self.counter))  # IMP-5
        self.should_stop = bool(state_dict.get("should_stop", self.should_stop))  # IMP-5


class LRSchedulerController:
    def __init__(
        self,
        name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        warmup_scheduler: LinearLR | None = None,
        warmup_epochs: int = 0,
    ) -> None:
        self.name = name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.step_count = 0

    def step(self, val_loss: float | None = None) -> None:
        if self.name == "none":
            return

        if self.name == "plateau":
            if self.warmup_scheduler is not None and self.step_count < self.warmup_epochs:
                self.warmup_scheduler.step()
            elif self.scheduler is not None:
                if val_loss is None:
                    raise ValueError("val_loss is required when using ReduceLROnPlateau.")
                self.scheduler.step(val_loss)
            self.step_count += 1
            return

        if self.scheduler is not None:
            self.scheduler.step()
        self.step_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_count": self.step_count,
            "warmup_epochs": self.warmup_epochs,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict() if self.warmup_scheduler is not None else None,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "name" not in state_dict:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state_dict)
            return

        self.step_count = int(state_dict.get("step_count", 0))
        self.warmup_epochs = int(state_dict.get("warmup_epochs", self.warmup_epochs))
        scheduler_state_dict = state_dict.get("scheduler_state_dict")
        warmup_scheduler_state_dict = state_dict.get("warmup_scheduler_state_dict")
        if self.scheduler is not None and scheduler_state_dict is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)
        if self.warmup_scheduler is not None and warmup_scheduler_state_dict is not None:
            self.warmup_scheduler.load_state_dict(warmup_scheduler_state_dict)


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
    parser.add_argument("--lr-scheduler", type=str, choices=["none", "cosine", "plateau"], default="cosine")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=2)  # IMP-2
    parser.add_argument("--early-stopping-patience", type=int, default=5)  # IMP-3
    parser.add_argument("--mixed-precision", action="store_true")  # IMP-4
    parser.add_argument("--resume-from", type=str, default=None)  # IMP-5
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
        "lr_scheduler": args.lr_scheduler,
        "min_learning_rate": args.min_learning_rate,
        "plateau_factor": args.plateau_factor,
        "plateau_patience": args.plateau_patience,
        "warmup_epochs": args.warmup_epochs,  # IMP-2
        "early_stopping_patience": args.early_stopping_patience,  # IMP-3
        "mixed_precision": args.mixed_precision,  # IMP-4
        "resume_from": args.resume_from or "",  # IMP-5
    }


def log_epoch_metrics_to_mlflow(epoch: int, train_result: EpochResult, val_result: EpochResult, learning_rate: float) -> None:  # FIX-3
    mlflow.log_metrics(
        {
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_macro_f1": train_result.macro_f1,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_macro_f1": val_result.macro_f1,
            "lr": learning_rate,
            "learning_rate": learning_rate,  # FIX-3
        },
        step=epoch,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_image = batch["pre_image"].to(device, non_blocking=True)
    post_image = batch["post_image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return pre_image, post_image, labels


def train_one_epoch(
    model: nn.Module,  # IMP-5
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,  # IMP-5
    device: torch.device,  # IMP-5
    num_classes: int,
    scaler: torch.cuda.amp.GradScaler | None = None,  # IMP-4
    max_batches: int | None = None,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    use_amp = scaler is not None and scaler.is_enabled()  # IMP-4

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        pre_image, post_image, labels = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):  # IMP-4
            logits = model(pre_image, post_image)  # IMP-4
            loss = criterion(logits, labels)  # IMP-4

        if use_amp:  # IMP-4
            scaler.scale(loss).backward()  # IMP-4
            scaler.unscale_(optimizer)  # IMP-4
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # FIX-2
            scaler.step(optimizer)  # IMP-4
            scaler.update()  # IMP-4
        else:  # IMP-4
            loss.backward()  # FIX-2
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # FIX-2
            optimizer.step()  # IMP-4

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_predictions.append(logits_to_predictions(logits).detach().cpu())
        all_targets.append(labels.detach().cpu())

    return summarize_epoch(total_loss, total_samples, all_predictions, all_targets, num_classes)


@torch.no_grad()
def evaluate(
    model: nn.Module,  # IMP-5
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,  # IMP-5
    num_classes: int,
    mixed_precision: bool = False,  # IMP-4
    max_batches: int | None = None,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    use_amp = mixed_precision and device.type == "cuda"  # IMP-4

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        pre_image, post_image, labels = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=use_amp):  # IMP-4
            logits = model(pre_image, post_image)  # IMP-4
            loss = criterion(logits, labels)  # IMP-4

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


def build_history_row(epoch: int, train_result: EpochResult, val_result: EpochResult, learning_rate: float) -> dict[str, float | int]:  # FIX-3
    return {
        "epoch": epoch,
        "lr": learning_rate,
        "learning_rate": learning_rate,  # FIX-3
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
        "lr",
        "learning_rate",  # FIX-3
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
    history: list[dict[str, float | int]],  # IMP-5
    train_key: str,
    val_key: str,
    ylabel: str,
    title: str,
    output_path: str | Path,  # IMP-6
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


def build_class_names(num_classes: int) -> list[str]:  # IMP-6
    default_class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]  # IMP-6
    if num_classes == len(default_class_names):  # IMP-6
        return default_class_names  # IMP-6
    return [f"class_{class_index}" for class_index in range(num_classes)]  # IMP-6


def save_confusion_matrix_figure(  # IMP-6
    matrix: torch.Tensor,  # IMP-6
    class_names: list[str],  # IMP-6
    output_path: str | Path,  # IMP-6
) -> Path:  # IMP-6
    output_path = Path(output_path)  # IMP-6
    output_path.parent.mkdir(parents=True, exist_ok=True)  # IMP-6
    matrix_np = matrix.cpu().numpy().astype(np.int64)  # IMP-6

    fig, ax = plt.subplots(figsize=(7, 6))  # IMP-6
    image = ax.imshow(matrix_np, interpolation="nearest", cmap="Blues")  # IMP-6
    fig.colorbar(image, ax=ax)  # IMP-6
    ax.set_xticks(np.arange(len(class_names)))  # IMP-6
    ax.set_yticks(np.arange(len(class_names)))  # IMP-6
    ax.set_xticklabels(class_names, rotation=45, ha="right")  # IMP-6
    ax.set_yticklabels(class_names)  # IMP-6
    ax.set_xlabel("Predicted label")  # IMP-6
    ax.set_ylabel("True label")  # IMP-6
    ax.set_title("Validation Confusion Matrix")  # IMP-6

    threshold = matrix_np.max() / 2.0 if matrix_np.size else 0.0  # IMP-6
    for row_index in range(matrix_np.shape[0]):  # IMP-6
        for col_index in range(matrix_np.shape[1]):  # IMP-6
            color = "white" if matrix_np[row_index, col_index] > threshold else "black"  # IMP-6
            ax.text(col_index, row_index, str(matrix_np[row_index, col_index]), ha="center", va="center", color=color)  # IMP-6

    fig.tight_layout()  # IMP-6
    fig.savefig(output_path, dpi=150)  # IMP-6
    plt.close(fig)  # IMP-6
    return output_path  # IMP-6


def validate_scheduler_args(
    scheduler_name: str,
    min_learning_rate: float,
    plateau_factor: float,
    plateau_patience: int,
) -> None:
    if min_learning_rate < 0.0:
        raise ValueError("--min-learning-rate must be non-negative.")
    if scheduler_name == "plateau" and not 0.0 < plateau_factor < 1.0:
        raise ValueError("--plateau-factor must be in the interval (0, 1).")
    if plateau_patience < 0:
        raise ValueError("--plateau-patience must be non-negative.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    min_learning_rate: float,
    plateau_factor: float,
    plateau_patience: int,
) -> LRSchedulerController:
    validate_scheduler_args(
        scheduler_name=scheduler_name,
        min_learning_rate=min_learning_rate,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
    )
    if scheduler_name == "none":
        return LRSchedulerController(name="none", optimizer=optimizer)

    warmup_epochs = max(0, min(warmup_epochs, max(0, epochs - 1)))
    if scheduler_name == "cosine":
        cosine_epochs = max(1, epochs - warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_learning_rate)
        if warmup_epochs == 0:
            return LRSchedulerController(name="cosine", optimizer=optimizer, scheduler=cosine_scheduler)

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        sequential_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        return LRSchedulerController(name="cosine", optimizer=optimizer, scheduler=sequential_scheduler)

    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=plateau_factor,
        patience=plateau_patience,
        min_lr=min_learning_rate,
    )
    return LRSchedulerController(
        name="plateau",
        optimizer=optimizer,
        scheduler=plateau_scheduler,
        warmup_scheduler=warmup_scheduler,
        warmup_epochs=warmup_epochs,
    )

def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:  # FIX-3
    return float(optimizer.param_groups[0]["lr"])  # FIX-3


def save_checkpoint(
    output_dir: str | Path,
    checkpoint_name: str,  # IMP-5
    epoch: int,
    model: nn.Module,  # IMP-5
    optimizer: torch.optim.Optimizer,  # IMP-5
    scheduler: Any,  # IMP-5
    val_result: EpochResult,
    args: argparse.Namespace,
    history: list[dict[str, float | int]],  # IMP-5
    best_val_macro_f1: float,  # IMP-5
    best_val_accuracy: float,  # IMP-5
    best_val_loss: float,  # IMP-5
    early_stopping: EarlyStopping,  # IMP-5
) -> Path:  # IMP-5
    output_dir = Path(output_dir)  # IMP-5
    output_dir.mkdir(parents=True, exist_ok=True)  # IMP-5
    checkpoint_path = output_dir / checkpoint_name  # IMP-5

    torch.save(  # IMP-5
        {  # IMP-5
            "epoch": epoch,  # IMP-5
            "model_state_dict": model.state_dict(),  # IMP-5
            "optimizer_state_dict": optimizer.state_dict(),  # IMP-5
            "scheduler_state_dict": scheduler.state_dict(),  # IMP-5
            "val_macro_f1": val_result.macro_f1,  # IMP-5
            "val_accuracy": val_result.accuracy,  # IMP-5
            "val_loss": val_result.loss,  # IMP-5
            "best_val_macro_f1": best_val_macro_f1,  # IMP-5
            "best_val_accuracy": best_val_accuracy,  # IMP-5
            "best_val_loss": best_val_loss,  # IMP-5
            "history": history,  # IMP-5
            "early_stopping_state": early_stopping.state_dict(),  # IMP-5
            "args": vars(args),  # IMP-5
        },  # IMP-5
        checkpoint_path,  # IMP-5
    )  # IMP-5
    return checkpoint_path  # IMP-5


def load_training_checkpoint(  # IMP-5
    checkpoint_path: str | Path,  # IMP-5
    model: nn.Module,  # IMP-5
    optimizer: torch.optim.Optimizer,  # IMP-5
    scheduler: Any,  # IMP-5
    early_stopping: EarlyStopping,  # IMP-5
    device: torch.device,  # IMP-5
) -> dict[str, object]:  # IMP-5
    checkpoint_path = Path(checkpoint_path)  # IMP-5
    if not checkpoint_path.exists():  # IMP-5
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")  # IMP-5

    checkpoint = torch.load(checkpoint_path, map_location=device)  # IMP-5
    model.load_state_dict(checkpoint["model_state_dict"])  # IMP-5
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # IMP-5
    if "scheduler_state_dict" in checkpoint:  # IMP-5
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # IMP-5
    if "early_stopping_state" in checkpoint:  # IMP-5
        early_stopping.load_state_dict(checkpoint["early_stopping_state"])  # IMP-5
    return checkpoint  # IMP-5


def restore_model_weights(checkpoint_path: str | Path, model: nn.Module, device: torch.device) -> None:  # IMP-3
    checkpoint = torch.load(checkpoint_path, map_location=device)  # IMP-3
    model.load_state_dict(checkpoint["model_state_dict"])  # IMP-3


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
        optimizer = AdamW(  # IMP-1
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=args.lr_scheduler,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_learning_rate=args.min_learning_rate,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
        )
        use_mixed_precision = args.mixed_precision and device.type == "cuda"  # IMP-4
        scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)  # IMP-4
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)  # IMP-3

        history: list[dict[str, float | int]] = []
        start_epoch = 1  # IMP-5
        best_val_macro_f1 = -1.0
        best_val_accuracy = 0.0
        best_val_loss = float("inf")  # IMP-5
        best_checkpoint_path: Path | None = None
        last_checkpoint_path = Path(args.output_dir) / "last_epoch.pt"  # IMP-5

        if args.resume_from is not None:  # IMP-5
            checkpoint = load_training_checkpoint(  # IMP-5
                checkpoint_path=args.resume_from,  # IMP-5
                model=model,  # IMP-5
                optimizer=optimizer,  # IMP-5
                scheduler=scheduler,  # IMP-5
                early_stopping=early_stopping,  # IMP-5
                device=device,  # IMP-5
            )  # IMP-5
            history = list(checkpoint.get("history", []))  # IMP-5
            start_epoch = int(checkpoint.get("epoch", 0)) + 1  # IMP-5
            best_val_macro_f1 = float(checkpoint.get("best_val_macro_f1", checkpoint.get("val_macro_f1", best_val_macro_f1)))  # IMP-5
            best_val_accuracy = float(checkpoint.get("best_val_accuracy", checkpoint.get("val_accuracy", best_val_accuracy)))  # IMP-5
            best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss)))  # IMP-5
            best_checkpoint_path = Path(args.output_dir) / "best_siamese_resnet18.pt"  # IMP-5
            print(f"Resumed training from checkpoint: {args.resume_from}")  # IMP-5

        class_names = build_class_names(args.num_classes)  # IMP-6
        stopped_early = False  # IMP-3

        for epoch in range(start_epoch, args.epochs + 1):  # IMP-5
            train_result = train_one_epoch(
                model=model,
                dataloader=dataloaders.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_classes=args.num_classes,
                scaler=scaler,  # IMP-4
                max_batches=args.max_train_batches,
            )
            val_result = evaluate(
                model=model,
                dataloader=dataloaders.val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                mixed_precision=use_mixed_precision,  # IMP-4
                max_batches=args.max_val_batches,
            )

            confusion_matrix_path = save_confusion_matrix_figure(  # IMP-6
                matrix=val_result.confusion_matrix,  # IMP-6
                class_names=class_names,  # IMP-6
                output_path=Path(args.figures_dir) / "confusion_matrix" / f"epoch_{epoch:03d}.png",  # IMP-6
            )  # IMP-6
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="figures/confusion_matrix")  # IMP-6
            learning_rate = get_current_learning_rate(optimizer)  # FIX-3

            history.append(build_history_row(epoch, train_result, val_result, learning_rate))  # FIX-3
            log_epoch_metrics_to_mlflow(epoch, train_result, val_result, learning_rate)  # FIX-3
            scheduler.step(val_loss=val_result.loss)  # FIX-3

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"lr={learning_rate:.8f} "  # FIX-3
                f"train_loss={train_result.loss:.4f} train_accuracy={train_result.accuracy:.4f} "
                f"train_macro_f1={train_result.macro_f1:.4f} | "
                f"val_loss={val_result.loss:.4f} val_accuracy={val_result.accuracy:.4f} "
                f"val_macro_f1={val_result.macro_f1:.4f}"
            )

            improved = early_stopping.step(val_macro_f1=val_result.macro_f1, val_loss=val_result.loss)  # IMP-3
            if improved:  # IMP-5
                best_val_macro_f1 = val_result.macro_f1  # IMP-5
                best_val_accuracy = val_result.accuracy  # IMP-5
                best_val_loss = val_result.loss  # IMP-5
                best_checkpoint_path = save_checkpoint(  # IMP-5
                    output_dir=args.output_dir,  # IMP-5
                    checkpoint_name="best_siamese_resnet18.pt",  # IMP-5
                    epoch=epoch,  # IMP-5
                    model=model,  # IMP-5
                    optimizer=optimizer,  # IMP-5
                    scheduler=scheduler,  # IMP-5
                    val_result=val_result,  # IMP-5
                    args=args,  # IMP-5
                    history=history,  # IMP-5
                    best_val_macro_f1=best_val_macro_f1,  # IMP-5
                    best_val_accuracy=best_val_accuracy,  # IMP-5
                    best_val_loss=best_val_loss,  # IMP-5
                    early_stopping=early_stopping,  # IMP-5
                )  # IMP-5
                print(f"Saved best checkpoint: {best_checkpoint_path}")  # IMP-5

            last_checkpoint_path = save_checkpoint(  # IMP-5
                output_dir=args.output_dir,  # IMP-5
                checkpoint_name="last_epoch.pt",  # IMP-5
                epoch=epoch,  # IMP-5
                model=model,  # IMP-5
                optimizer=optimizer,  # IMP-5
                scheduler=scheduler,  # IMP-5
                val_result=val_result,  # IMP-5
                args=args,  # IMP-5
                history=history,  # IMP-5
                best_val_macro_f1=best_val_macro_f1,  # IMP-5
                best_val_accuracy=best_val_accuracy,  # IMP-5
                best_val_loss=best_val_loss,  # IMP-5
                early_stopping=early_stopping,  # IMP-5
            )  # IMP-5

            if early_stopping.should_stop:  # IMP-3
                stopped_early = True  # IMP-3
                print(f"Early stopping triggered at epoch {epoch:03d}.")  # IMP-3
                if best_checkpoint_path is not None and best_checkpoint_path.exists():  # IMP-3
                    restore_model_weights(best_checkpoint_path, model, device)  # IMP-3
                break  # IMP-3

        history_path = save_history_csv(history, args.history_path)
        curve_paths = save_training_curves(history, args.figures_dir)

        mlflow.log_metrics(
            {
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": best_val_loss,  # IMP-5
                "stopped_early": float(stopped_early),  # IMP-3
            }
        )
        mlflow.log_artifact(str(history_path), artifact_path="history")
        for curve_path in curve_paths.values():
            mlflow.log_artifact(str(curve_path), artifact_path="figures")
        if best_checkpoint_path is not None and best_checkpoint_path.exists():  # IMP-5
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        if last_checkpoint_path.exists():  # IMP-5
            mlflow.log_artifact(str(last_checkpoint_path), artifact_path="checkpoints")  # IMP-5

        print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
        if best_checkpoint_path is not None:
            print(f"Best checkpoint path: {best_checkpoint_path}")
        print(f"Last checkpoint path: {last_checkpoint_path}")  # IMP-5
        print(f"Final history CSV: {history_path}")
        print("Final training curves:", ", ".join(str(path) for path in curve_paths.values()))
        print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")


def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()

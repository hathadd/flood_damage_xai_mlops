from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.dataloader import build_dataloaders
from src.evaluation.metrics import accuracy_score, confusion_matrix, logits_to_predictions, macro_f1_score
from src.models.siamese_model import SiameseResNet18
from src.training.losses import build_weighted_cross_entropy_loss


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Siamese ResNet18 flood damage classifier.")
    parser.add_argument("--split-metadata-path", type=str, default="data/splits/metadata_splits.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    criterion = build_weighted_cross_entropy_loss(dataloaders.class_weights.to(device))
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_macro_f1 = -1.0
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

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.4f} "
            f"train_macro_f1={train_result.macro_f1:.4f} | "
            f"val_loss={val_result.loss:.4f} val_acc={val_result.accuracy:.4f} "
            f"val_macro_f1={val_result.macro_f1:.4f}"
        )

        if val_result.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_result.macro_f1
            best_checkpoint_path = save_checkpoint(
                output_dir=args.output_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_result=val_result,
                args=args,
            )
            print(f"Saved best checkpoint: {best_checkpoint_path}")

    print(f"Best validation macro F1: {best_val_macro_f1:.4f}")
    if best_checkpoint_path is not None:
        print(f"Best checkpoint path: {best_checkpoint_path}")


def main() -> None:
    args = parse_args()
    fit(args)


if __name__ == "__main__":
    main()

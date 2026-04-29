from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.dataset import XBDBuildingSegmentationDataset
from src.segmentation.model_unet import UNet
from src.segmentation.transforms import get_eval_transforms, get_train_transforms


@dataclass
class SegmentationMetrics:
    loss: float
    dice: float
    iou: float
    pixel_accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight U-Net for building segmentation.")
    parser.add_argument("--split-metadata-path", type=str, default="data/splits/metadata_splits.csv")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/segmentation_unet")
    parser.add_argument("--mask-cache-dir", type=str, default="data/segmentation_masks_cache")
    parser.add_argument("--image-type", type=str, default="post_image", choices=["post_image", "pre_image"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--use-dice-loss", action="store_true")
    parser.add_argument("--selection-metric", type=str, default="dice", choices=["dice", "iou"])
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_batch_statistics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> tuple[float, float, float]:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    target_area = targets.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection

    dice = ((2.0 * intersection + eps) / (pred_area + target_area + eps)).mean().item()
    iou = ((intersection + eps) / (union + eps)).mean().item()
    pixel_accuracy = preds.eq(targets).float().mean(dim=(1, 2, 3)).mean().item()
    return dice, iou, pixel_accuracy


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = XBDBuildingSegmentationDataset(
        split_metadata_path=args.split_metadata_path,
        split="train",
        dataset_root=args.dataset_root,
        image_type=args.image_type,
        transforms=get_train_transforms(args.image_size),
        mask_cache_dir=args.mask_cache_dir,
    )
    val_dataset = XBDBuildingSegmentationDataset(
        split_metadata_path=args.split_metadata_path,
        split="val",
        dataset_root=args.dataset_root,
        image_type=args.image_type,
        transforms=get_eval_transforms(args.image_size),
        mask_cache_dir=args.mask_cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW | None,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_mixed_precision: bool,
    use_dice_loss: bool,
) -> SegmentationMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pixel_accuracy = 0.0
    num_batches = 0

    for images, masks in tqdm(dataloader, leave=False):
        images = images.to(device)
        masks = masks.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                logits = model(images)
                loss = criterion(logits, masks)
                if use_dice_loss:
                    loss = loss + dice_loss(logits, masks)

            if is_train and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        dice, iou, pixel_accuracy = compute_batch_statistics(logits.detach(), masks.detach())
        total_loss += float(loss.detach().item())
        total_dice += dice
        total_iou += iou
        total_pixel_accuracy += pixel_accuracy
        num_batches += 1

    if num_batches == 0:
        return SegmentationMetrics(loss=math.inf, dice=0.0, iou=0.0, pixel_accuracy=0.0)

    return SegmentationMetrics(
        loss=total_loss / num_batches,
        dice=total_dice / num_batches,
        iou=total_iou / num_batches,
        pixel_accuracy=total_pixel_accuracy / num_batches,
    )


def save_history(history: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else []
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: SegmentationMetrics,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics.__dict__,
            "args": vars(args),
            "history": history,
        },
        output_path,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.random_state)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
    use_mixed_precision = args.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_metric_name = f"val_{args.selection_metric}"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
            use_dice_loss=args.use_dice_loss,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_mixed_precision=use_mixed_precision,
            use_dice_loss=args.use_dice_loss,
        )
        scheduler.step()

        learning_rate = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics.loss,
            "train_dice": train_metrics.dice,
            "train_iou": train_metrics.iou,
            "train_pixel_accuracy": train_metrics.pixel_accuracy,
            "val_loss": val_metrics.loss,
            "val_dice": val_metrics.dice,
            "val_iou": val_metrics.iou,
            "val_pixel_accuracy": val_metrics.pixel_accuracy,
        }
        history.append(history_row)

        print(
            f"U-Net Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr={learning_rate:.7f} "
            f"train_loss={train_metrics.loss:.4f} train_dice={train_metrics.dice:.4f} train_iou={train_metrics.iou:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_dice={val_metrics.dice:.4f} val_iou={val_metrics.iou:.4f}"
        )

        save_checkpoint(
            output_dir / "last_epoch.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            val_metrics,
            args,
            history,
        )

        current_score = getattr(val_metrics, args.selection_metric)
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                output_dir / "best_unet.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                args,
                history,
            )

    save_history(history, output_dir / "training_history.csv")
    summary = {
        "best_metric_name": best_metric_name,
        "best_metric_value": best_score,
        "epochs": args.epochs,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

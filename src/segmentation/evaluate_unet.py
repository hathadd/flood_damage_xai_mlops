from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.dataset import XBDBuildingSegmentationDataset
from src.segmentation.model_unet import UNet
from src.segmentation.postprocessing import threshold_mask
from src.segmentation.train_unet import compute_batch_statistics, resolve_device
from src.segmentation.transforms import get_eval_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the experimental U-Net segmentation model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split-metadata-path", type=str, default="data/splits/metadata_splits.csv")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/segmentation_unet")
    parser.add_argument("--mask-cache-dir", type=str, default="data/segmentation_masks_cache")
    parser.add_argument("--image-type", type=str, default="post_image", choices=["post_image", "pre_image"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-visualizations", type=int, default=8)
    return parser.parse_args()


def load_checkpoint_state(checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def save_visualization(
    image: torch.Tensor,
    true_mask: torch.Tensor,
    predicted_mask: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip((image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406]), 0.0, 1.0)
    true_mask_np = true_mask.detach().cpu().squeeze(0).numpy()

    overlay = image_np.copy()
    overlay[predicted_mask > 0] = np.array([1.0, 0.0, 0.0]) * 0.5 + overlay[predicted_mask > 0] * 0.5

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[1].imshow(true_mask_np, cmap="gray")
    axes[1].set_title("True Mask")
    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = XBDBuildingSegmentationDataset(
        split_metadata_path=args.split_metadata_path,
        split="test",
        dataset_root=args.dataset_root,
        image_type=args.image_type,
        transforms=get_eval_transforms(args.image_size),
        mask_cache_dir=args.mask_cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(load_checkpoint_state(args.checkpoint_path))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pixel_accuracy = 0.0
    num_batches = 0
    saved_visualizations = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, leave=False)):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            dice, iou, pixel_accuracy = compute_batch_statistics(logits, masks)

            total_loss += float(loss.item())
            total_dice += dice
            total_iou += iou
            total_pixel_accuracy += pixel_accuracy
            num_batches += 1

            probs = torch.sigmoid(logits)
            preds = threshold_mask(probs.detach().cpu().numpy())
            for item_idx in range(images.size(0)):
                if saved_visualizations >= args.num_visualizations:
                    break
                save_visualization(
                    image=images[item_idx].cpu(),
                    true_mask=masks[item_idx].cpu(),
                    predicted_mask=preds[item_idx, 0],
                    output_path=output_dir / "samples" / f"sample_{batch_idx:03d}_{item_idx:02d}.png",
                )
                saved_visualizations += 1

    metrics = {
        "test_loss": total_loss / max(num_batches, 1),
        "test_dice": total_dice / max(num_batches, 1),
        "test_iou": total_iou / max(num_batches, 1),
        "test_pixel_accuracy": total_pixel_accuracy / max(num_batches, 1),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(json.dumps(metrics, indent=2))


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()

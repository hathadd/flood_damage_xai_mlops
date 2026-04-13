from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.siamese_model import SiameseResNet18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a Siamese ResNet18 checkpoint can be reloaded.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})

    model = SiameseResNet18(
        num_classes=int(checkpoint_args.get("num_classes", 4)),
        pretrained=False,
        dropout=float(checkpoint_args.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_size = int(checkpoint_args.get("image_size", 224))
    dummy_pre = torch.randn(1, 3, image_size, image_size, device=device)
    dummy_post = torch.randn_like(dummy_pre)

    with torch.no_grad():
        logits = model(dummy_pre, dummy_post)

    print(f"Checkpoint loaded successfully: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Validation macro F1: {checkpoint.get('val_macro_f1')}")
    print(f"Logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

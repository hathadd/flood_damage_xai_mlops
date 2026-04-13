from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import build_dataloaders
from src.models.siamese_model import SiameseResNet18
from src.training.losses import build_weighted_cross_entropy_loss
from src.training.train import train_one_epoch


def test_siamese_resnet18_forward_and_one_train_batch() -> None:
    device = torch.device("cpu")
    dataloaders = build_dataloaders(
        image_size=64,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        return_metadata=True,
    )
    model = SiameseResNet18(num_classes=4, pretrained=False).to(device)
    criterion = build_weighted_cross_entropy_loss(dataloaders.class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=1e-4)

    batch = next(iter(dataloaders.train_loader))
    logits = model(batch["pre_image"].to(device), batch["post_image"].to(device))

    assert logits.shape == (batch["label"].shape[0], 4)

    result = train_one_epoch(
        model=model,
        dataloader=dataloaders.train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=4,
        max_batches=1,
    )

    assert result.loss > 0.0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.confusion_matrix.shape == (4, 4)

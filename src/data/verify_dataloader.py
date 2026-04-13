from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler

from src.data.dataloader import build_dataloaders
from src.training.losses import build_weighted_cross_entropy_loss


def main() -> None:
    bundle = build_dataloaders(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    train_batch = next(iter(bundle.train_loader))

    assert isinstance(bundle.train_loader.sampler, WeightedRandomSampler)
    assert train_batch["pre_image"].shape == train_batch["post_image"].shape
    assert train_batch["pre_image"].ndim == 4
    assert train_batch["label"].ndim == 1
    assert train_batch["pre_image"].shape[0] == train_batch["label"].shape[0]
    assert torch.all(train_batch["label"] >= 0)
    assert torch.all(train_batch["label"] < len(bundle.class_weights))

    criterion = build_weighted_cross_entropy_loss(bundle.class_weights)

    print("Train batch checks passed.")
    print(f"pre_image shape: {tuple(train_batch['pre_image'].shape)}")
    print(f"post_image shape: {tuple(train_batch['post_image'].shape)}")
    print(f"label shape: {tuple(train_batch['label'].shape)}")
    print(f"train sampler: {type(bundle.train_loader.sampler).__name__}")
    print(f"val sampler: {type(bundle.val_loader.sampler).__name__}")
    print(f"test sampler: {type(bundle.test_loader.sampler).__name__}")
    print(f"class weights: {bundle.class_weights.tolist()}")
    print(f"loss module: {criterion}")


if __name__ == "__main__":
    main()

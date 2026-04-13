from __future__ import annotations

import torch
from torch import nn


def compute_class_weights_from_counts(
    class_counts: torch.Tensor,
) -> torch.Tensor:
    if class_counts.ndim != 1:
        raise ValueError("class_counts must be a 1D tensor.")

    if torch.any(class_counts <= 0):
        raise ValueError("All classes must have a strictly positive count.")

    total_samples = float(class_counts.sum().item())
    num_classes = int(class_counts.numel())
    return total_samples / (num_classes * class_counts.float())


def compute_class_weights_from_labels(
    labels: torch.Tensor,
    num_classes: int | None = None,
) -> torch.Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor.")

    if labels.numel() == 0:
        raise ValueError("Cannot compute class weights from empty labels.")

    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    class_counts = torch.bincount(labels.long(), minlength=num_classes)
    return compute_class_weights_from_counts(class_counts)


def build_weighted_cross_entropy_loss(
    class_weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(
        weight=class_weights.float(),
        label_smoothing=label_smoothing,
    )

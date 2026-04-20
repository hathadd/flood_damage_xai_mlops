from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'.")

        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must have shape (batch_size, num_classes).")

        if targets.ndim != 1:
            raise ValueError("targets must have shape (batch_size,).")

        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs
        alpha_t: torch.Tensor | None = None  # FIX-1

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
            alpha_t = alpha.gather(dim=0, index=targets)  # FIX-1
            loss = loss * alpha_t  # FIX-1

        if self.reduction == "mean":
            if alpha_t is not None:  # FIX-1
                return loss.sum() / alpha_t.sum().clamp_min(torch.finfo(loss.dtype).eps)  # FIX-1
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


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


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> nn.Module:
    loss_type = loss_type.lower()

    if loss_type == "ce":
        return build_weighted_cross_entropy_loss(class_weights=class_weights)

    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)

    raise ValueError(f"Unsupported loss_type: {loss_type}. Expected 'ce' or 'focal'.")  # FIX-1

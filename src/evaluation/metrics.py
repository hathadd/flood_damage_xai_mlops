from __future__ import annotations

import torch


def logits_to_predictions(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes).")
    return torch.argmax(logits, dim=1)


def accuracy_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = predictions.detach().cpu().long()
    targets = targets.detach().cpu().long()

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")

    if targets.numel() == 0:
        return 0.0

    return float((predictions == targets).float().mean().item())


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    predictions = predictions.detach().cpu().long().view(-1)
    targets = targets.detach().cpu().long().view(-1)

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    valid_mask = (targets >= 0) & (targets < num_classes) & (predictions >= 0) & (predictions < num_classes)

    for target, prediction in zip(targets[valid_mask], predictions[valid_mask], strict=True):
        matrix[target, prediction] += 1

    return matrix


def macro_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> float:
    matrix = confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
    ).float()

    true_positive = torch.diag(matrix)
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive

    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1_per_class = 2 * precision * recall / (precision + recall + eps)

    return float(f1_per_class.mean().item())


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> dict[str, float | torch.Tensor]:
    predictions = logits_to_predictions(logits)
    return {
        "accuracy": accuracy_score(predictions, targets),
        "macro_f1": macro_f1_score(predictions, targets, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(predictions, targets, num_classes=num_classes),
    }

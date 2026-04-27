from __future__ import annotations

from typing import Any

import torch

from src.serving.config import settings
from src.serving.model_loader import get_device, load_model


def predict_damage(pre_tensor: torch.Tensor, post_tensor: torch.Tensor) -> dict[str, Any]:
    model = load_model()
    device = get_device()

    if pre_tensor.ndim == 3:
        pre_tensor = pre_tensor.unsqueeze(0)
    if post_tensor.ndim == 3:
        post_tensor = post_tensor.unsqueeze(0)

    pre_tensor = pre_tensor.to(device, non_blocking=True)
    post_tensor = post_tensor.to(device, non_blocking=True)

    with torch.no_grad():
        logits = model(pre_tensor, post_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_class_id = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class_id].item())
    probabilities_dict = {
        settings.label_mapping[index]: float(probabilities[index].item())
        for index in range(settings.num_classes)
    }

    return {
        "predicted_class_id": predicted_class_id,
        "predicted_label": settings.label_mapping[predicted_class_id],
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
    }

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn

matplotlib.use("Agg")

from src.scene.crop_extraction import crop_with_context
from src.scene.polygon_parser import parse_xbd_buildings
from src.serving.config import settings
from src.serving.model_loader import get_device, load_model
from src.serving.preprocessing import EVAL_TRANSFORMS

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class SceneSiameseGradCAM:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.target_layer = model.backbone.layer4[-1].conv2
        self.current_branch: str | None = None
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        branch_name = self.current_branch
        if branch_name is None:
            return
        self.activations[branch_name] = output.detach()

        def _save_gradients(grad: torch.Tensor, branch: str = branch_name) -> None:
            self.gradients[branch] = grad.detach()

        output.register_hook(_save_gradients)

    def clear(self) -> None:
        self.activations = {}
        self.gradients = {}
        self.current_branch = None

    def remove(self) -> None:
        self.forward_handle.remove()


def _sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def _overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_float + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255.0).round().astype(np.uint8)


def _save_rgb_image(image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)
    return output_path


def _preprocess_crop(crop: Image.Image) -> torch.Tensor:
    crop_np = np.asarray(crop.convert("RGB"))
    transformed = EVAL_TRANSFORMS(image=crop_np)
    return transformed["image"]


def _compute_gradcam_maps(
    model: nn.Module,
    gradcam: SceneSiameseGradCAM,
    pre_tensor: torch.Tensor,
    post_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[dict[str, np.ndarray], int, float, np.ndarray]:
    model.zero_grad(set_to_none=True)
    gradcam.clear()

    gradcam.current_branch = "pre"
    pre_features = model.extract_features(pre_tensor)
    gradcam.current_branch = "post"
    post_features = model.extract_features(post_tensor)
    gradcam.current_branch = None

    fused_features = model.fuse_features(pre_features, post_features)
    logits = model.classifier(fused_features)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_id = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(torch.max(probabilities, dim=1).values.item())
    class_index = predicted_class_id if target_class is None else target_class

    logits[0, class_index].backward()

    heatmaps: dict[str, np.ndarray] = {}
    for branch in ("pre", "post"):
        activations = gradcam.activations.get(branch)
        gradients = gradcam.gradients.get(branch)
        if activations is None or gradients is None:
            raise RuntimeError(f"Missing activations or gradients for branch '{branch}'.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=pre_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmaps[branch] = cam.astype(np.float32)

    return heatmaps, predicted_class_id, confidence, probabilities[0].detach().cpu().numpy()


def generate_building_gradcam(
    pre_image: Image.Image,
    post_image: Image.Image,
    post_json_bytes: bytes,
    building_index: int,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
    output_dir: str | Path = "outputs/serving/gradcam",
) -> dict[str, Any]:
    if min_crop_size <= 0:
        raise ValueError("min_crop_size must be strictly positive.")
    if context_ratio < 0:
        raise ValueError("context_ratio must be non-negative.")

    buildings = parse_xbd_buildings(post_json_bytes)
    selected_building = next((building for building in buildings if int(building["building_index"]) == int(building_index)), None)
    if selected_building is None:
        raise ValueError(f"Building with building_index={building_index} not found in post_json.")

    bbox = tuple(float(value) for value in selected_building["bbox"])
    pre_crop, crop_box = crop_with_context(
        pre_image,
        bbox,
        context_ratio=context_ratio,
        min_crop_size=min_crop_size,
    )
    post_crop, _ = crop_with_context(
        post_image,
        bbox,
        context_ratio=context_ratio,
        min_crop_size=min_crop_size,
    )

    device = get_device()
    model = load_model()
    gradcam = SceneSiameseGradCAM(model)

    try:
        pre_tensor = _preprocess_crop(pre_crop).unsqueeze(0).to(device, non_blocking=True)
        post_tensor = _preprocess_crop(post_crop).unsqueeze(0).to(device, non_blocking=True)
        heatmaps, predicted_class_id, confidence, probabilities = _compute_gradcam_maps(
            model=model,
            gradcam=gradcam,
            pre_tensor=pre_tensor,
            post_tensor=post_tensor,
            target_class=None,
        )
    finally:
        gradcam.remove()

    pre_crop_rgb = _tensor_to_rgb_image(pre_tensor[0].cpu())
    post_crop_rgb = _tensor_to_rgb_image(post_tensor[0].cpu())
    pre_overlay = _overlay_heatmap_on_image(pre_crop_rgb, heatmaps["pre"])
    post_overlay = _overlay_heatmap_on_image(post_crop_rgb, heatmaps["post"])

    building_uid = selected_building.get("building_uid")
    true_label = selected_building.get("true_label")
    predicted_label = settings.label_mapping[predicted_class_id]
    probabilities_dict = {
        settings.label_mapping[index]: float(probabilities[index])
        for index in range(settings.num_classes)
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder_name = _sanitize_name(f"building_{building_index}_{building_uid or 'no_uid'}_{timestamp}")
    sample_dir = Path(output_dir) / folder_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    pre_crop_path = _save_rgb_image(pre_crop_rgb, sample_dir / "pre_crop.png")
    post_crop_path = _save_rgb_image(post_crop_rgb, sample_dir / "post_crop.png")
    pre_gradcam_path = _save_rgb_image(pre_overlay, sample_dir / "pre_gradcam_overlay.png")
    post_gradcam_path = _save_rgb_image(post_overlay, sample_dir / "post_gradcam_overlay.png")

    return {
        "building_index": int(selected_building["building_index"]),
        "building_uid": building_uid,
        "bbox": [float(value) for value in bbox],
        "crop_box": [int(value) for value in crop_box],
        "true_label": true_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "pre_gradcam_path": str(pre_gradcam_path),
        "post_gradcam_path": str(post_gradcam_path),
        "pre_crop_path": str(pre_crop_path),
        "post_crop_path": str(post_crop_path),
    }

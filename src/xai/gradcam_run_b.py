from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")

from src.data.dataset import XBDPairBuildingDataset
from src.models.siamese_model import SiameseResNet18

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_SELECTION_PREDICTIONS_PATH = Path("reports/evaluation/run_b/predictions.csv")
INFERENCE_BATCH_SIZE = 32


class SiameseGradCAM:
    def __init__(self, model: SiameseResNet18) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for the final Run B model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--split-metadata-path", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--max-samples-per-category", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def build_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=tuple(IMAGENET_MEAN.tolist()),
                std=tuple(IMAGENET_STD.tolist()),
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )


def load_test_dataset(
    split_metadata_path: str | Path,
    dataset_root: str | Path,
    image_size: int,
) -> XBDPairBuildingDataset:
    split_metadata_path = Path(split_metadata_path)
    if not split_metadata_path.exists():
        raise FileNotFoundError(f"Split metadata file not found: {split_metadata_path}")

    dataset = XBDPairBuildingDataset(
        metadata_csv=split_metadata_path,
        split="test",
        transforms=build_eval_transforms(image_size=image_size),
        return_metadata=True,
        dataset_root=dataset_root,
    )
    split_df = dataset.df.loc[dataset.df["split"] == "test"].copy().reset_index(drop=True)
    if split_df.empty:
        raise ValueError("No samples found for split='test'.")
    dataset.df = split_df
    return dataset


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def extract_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
            return normalize_state_dict_keys(checkpoint_obj["model_state_dict"])
        if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
            return normalize_state_dict_keys(checkpoint_obj["state_dict"])
        if checkpoint_obj and all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return normalize_state_dict_keys(checkpoint_obj)
    raise ValueError("Unsupported checkpoint format. Expected raw state_dict or dict containing model_state_dict/state_dict.")


def load_model(checkpoint_path: str | Path, device: torch.device, num_classes: int) -> SiameseResNet18:
    model = SiameseResNet18(num_classes=num_classes, pretrained=False, dropout=0.5)
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint_obj), strict=True)
    model.to(device)
    model.eval()
    return model


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def tensor_to_rgb_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * IMAGENET_STD) + IMAGENET_MEAN
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def overlay_heatmap_on_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    colored_heatmap = plt.get_cmap("jet")(heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_float + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return (overlay * 255.0).round().astype(np.uint8)


def save_rgb_image(image: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)
    return output_path


def compute_predictions_dataframe(
    model: SiameseResNet18,
    dataset: XBDPairBuildingDataset,
    device: torch.device,
    num_classes: int,
) -> pd.DataFrame:
    dataloader = DataLoader(
        dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pre_image = batch["pre_image"].to(device, non_blocking=True)
            post_image = batch["post_image"].to(device, non_blocking=True)
            logits = model(pre_image, post_image)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1).values
            labels = batch["label"].detach().cpu().numpy()
            predictions_np = predictions.detach().cpu().numpy()
            confidences_np = confidences.detach().cpu().numpy()

            for index in range(len(labels)):
                true_label = int(labels[index])
                pred_label = int(predictions_np[index])
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "building_uid": str(batch["building_uid"][index]),
                        "true_label": true_label,
                        "true_class": CLASS_NAMES[true_label],
                        "pred_label": pred_label,
                        "pred_class": CLASS_NAMES[pred_label],
                        "confidence": float(confidences_np[index]),
                    }
                )
    return pd.DataFrame(rows)


def load_or_build_predictions(
    model: SiameseResNet18,
    dataset: XBDPairBuildingDataset,
    device: torch.device,
    num_classes: int,
) -> pd.DataFrame:
    predictions_path = DEFAULT_SELECTION_PREDICTIONS_PATH
    required_columns = {"sample_id", "building_uid", "true_label", "pred_label", "confidence"}
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        if required_columns <= set(predictions_df.columns):
            predictions_df["sample_id"] = predictions_df["sample_id"].astype(str)
            predictions_df["building_uid"] = predictions_df["building_uid"].astype(str)
            return predictions_df
        print(f"Warning: {predictions_path} is missing required columns. Falling back to internal inference.")
    else:
        print(f"Warning: {predictions_path} not found. Falling back to internal inference.")

    return compute_predictions_dataframe(model=model, dataset=dataset, device=device, num_classes=num_classes)


def select_representative_samples(
    predictions_df: pd.DataFrame,
    max_samples_per_category: int,
    random_state: int,
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    category_specs: list[tuple[str, pd.DataFrame]] = [
        (
            "correct_no_damage",
            predictions_df.loc[(predictions_df["true_label"] == 0) & (predictions_df["pred_label"] == 0)].copy(),
        ),
        (
            "correct_major_damage",
            predictions_df.loc[(predictions_df["true_label"] == 2) & (predictions_df["pred_label"] == 2)].copy(),
        ),
        (
            "correct_destroyed",
            predictions_df.loc[(predictions_df["true_label"] == 3) & (predictions_df["pred_label"] == 3)].copy(),
        ),
    ]

    failed_minor_exact = predictions_df.loc[
        (predictions_df["true_label"] == 1) & (predictions_df["pred_label"] == 0)
    ].copy()
    if failed_minor_exact.empty:
        print("Warning: no true minor-damage predicted as no-damage samples found. Falling back to any failed minor-damage samples.")
        failed_minor_exact = predictions_df.loc[
            (predictions_df["true_label"] == 1) & (predictions_df["pred_label"] != 1)
        ].copy()
        category_specs.append(("failed_minor_damage_other", failed_minor_exact))
    else:
        category_specs.append(("failed_minor_damage_pred_no_damage", failed_minor_exact))

    for category_name, category_df in category_specs:
        if category_df.empty:
            print(f"Warning: no samples found for category '{category_name}'.")
            continue
        sample_count = min(max_samples_per_category, len(category_df))
        chosen = category_df.sample(n=sample_count, random_state=random_state)
        for row in chosen.to_dict(orient="records"):
            row["category"] = category_name
            selections.append(row)
    return selections


def build_dataset_index(dataset: XBDPairBuildingDataset) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for idx, row in dataset.df.reset_index(drop=True).iterrows():
        index[(str(row["sample_id"]), str(row["building_uid"]))] = idx
    return index


def generate_gradcam_maps(
    model: SiameseResNet18,
    gradcam: SiameseGradCAM,
    pre_tensor: torch.Tensor,
    post_tensor: torch.Tensor,
    target_class: int | None,
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
    predicted_label = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(torch.max(probabilities, dim=1).values.item())
    class_index = predicted_label if target_class is None else target_class

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
        cam = F.interpolate(
            cam,
            size=pre_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        heatmaps[branch] = cam.astype(np.float32)

    return heatmaps, predicted_label, confidence, probabilities[0].detach().cpu().numpy()


def save_combined_figure(
    pre_original: np.ndarray,
    post_original: np.ndarray,
    pre_overlay: np.ndarray,
    post_overlay: np.ndarray,
    output_path: str | Path,
    sample_id: str,
    building_uid: str,
    true_class: str,
    pred_class: str,
    confidence: float,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(pre_original)
    axes[0, 0].set_title("Original pre image")
    axes[0, 1].imshow(post_original)
    axes[0, 1].set_title("Original post image")
    axes[1, 0].imshow(pre_overlay)
    axes[1, 0].set_title("Pre-disaster Grad-CAM")
    axes[1, 1].imshow(post_overlay)
    axes[1, 1].set_title("Post-disaster Grad-CAM")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"sample_id={sample_id} | building_uid={building_uid}\n"
        f"True class: {true_class} | Predicted class: {pred_class} | Confidence: {confidence:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_gradcam_outputs(
    dataset: XBDPairBuildingDataset,
    model: SiameseResNet18,
    gradcam: SiameseGradCAM,
    selected_samples: list[dict[str, Any]],
    dataset_index: dict[tuple[str, str], int],
    output_dir: str | Path,
    device: torch.device,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for sample in selected_samples:
        sample_id = str(sample["sample_id"])
        building_uid = str(sample["building_uid"])
        dataset_key = (sample_id, building_uid)
        if dataset_key not in dataset_index:
            print(f"Warning: sample {dataset_key} not found in dataset index. Skipping.")
            continue

        item = dataset[dataset_index[dataset_key]]
        pre_tensor = item["pre_image"].unsqueeze(0).to(device)
        post_tensor = item["post_image"].unsqueeze(0).to(device)
        true_label = int(item["label"].item())
        true_class = CLASS_NAMES[true_label]

        heatmaps, pred_label, confidence, probabilities = generate_gradcam_maps(
            model=model,
            gradcam=gradcam,
            pre_tensor=pre_tensor,
            post_tensor=post_tensor,
            target_class=None,
        )

        pre_original = tensor_to_rgb_image(item["pre_image"])
        post_original = tensor_to_rgb_image(item["post_image"])
        pre_overlay = overlay_heatmap_on_image(pre_original, heatmaps["pre"])
        post_overlay = overlay_heatmap_on_image(post_original, heatmaps["post"])
        pred_class = CLASS_NAMES[pred_label]

        sample_dir = output_dir / sample["category"] / sanitize_name(f"{sample_id}_{building_uid}")
        sample_dir.mkdir(parents=True, exist_ok=True)

        pre_original_path = save_rgb_image(pre_original, sample_dir / "pre_original.png")
        post_original_path = save_rgb_image(post_original, sample_dir / "post_original.png")
        pre_overlay_path = save_rgb_image(pre_overlay, sample_dir / "pre_gradcam_overlay.png")
        post_overlay_path = save_rgb_image(post_overlay, sample_dir / "post_gradcam_overlay.png")
        combined_path = save_combined_figure(
            pre_original=pre_original,
            post_original=post_original,
            pre_overlay=pre_overlay,
            post_overlay=post_overlay,
            output_path=sample_dir / "combined_explanation.png",
            sample_id=sample_id,
            building_uid=building_uid,
            true_class=true_class,
            pred_class=pred_class,
            confidence=confidence,
        )

        summary_rows.append(
            {
                "sample_id": sample_id,
                "building_uid": building_uid,
                "true_label": true_label,
                "true_class": true_class,
                "pred_label": pred_label,
                "pred_class": pred_class,
                "confidence": confidence,
                "category": sample["category"],
                "pre_overlay_path": str(pre_overlay_path),
                "post_overlay_path": str(post_overlay_path),
                "combined_path": str(combined_path),
                "pre_original_path": str(pre_original_path),
                "post_original_path": str(post_original_path),
                "probabilities": json.dumps(probabilities.tolist()),
            }
        )

    return pd.DataFrame(summary_rows)


def print_selection_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    print(f"Grad-CAM outputs saved under: {output_dir}")
    if summary_df.empty:
        print("No Grad-CAM samples were generated.")
        return
    print(f"Generated Grad-CAM explanations for {len(summary_df)} samples.")
    print(summary_df[["category", "sample_id", "building_uid", "true_class", "pred_class", "confidence"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.max_samples_per_category <= 0:
        raise ValueError("--max-samples-per-category must be strictly positive.")

    device = resolve_device(args.device)
    dataset = load_test_dataset(
        split_metadata_path=args.split_metadata_path,
        dataset_root=args.dataset_root,
        image_size=args.image_size,
    )
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        num_classes=args.num_classes,
    )

    predictions_df = load_or_build_predictions(
        model=model,
        dataset=dataset,
        device=device,
        num_classes=args.num_classes,
    )
    selected_samples = select_representative_samples(
        predictions_df=predictions_df,
        max_samples_per_category=args.max_samples_per_category,
        random_state=args.random_state,
    )
    dataset_index = build_dataset_index(dataset)

    gradcam = SiameseGradCAM(model)
    try:
        summary_df = save_gradcam_outputs(
            dataset=dataset,
            model=model,
            gradcam=gradcam,
            selected_samples=selected_samples,
            dataset_index=dataset_index,
            output_dir=args.output_dir,
            device=device,
        )
    finally:
        gradcam.remove()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "gradcam_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not summary_df.empty:
        report = classification_report(
            summary_df["true_label"].astype(int),
            summary_df["pred_label"].astype(int),
            labels=list(range(args.num_classes)),
            target_names=CLASS_NAMES[: args.num_classes],
            zero_division=0,
        )
        (output_dir / "selected_samples_report.txt").write_text(report, encoding="utf-8")

    print_selection_summary(summary_df, output_dir)
    print(f"Summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()

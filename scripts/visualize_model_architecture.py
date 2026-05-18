from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.models.siamese_model import SiameseResNet18

try:
    from torchinfo import summary as torchinfo_summary
except ImportError:  # pragma: no cover - runtime dependency check
    torchinfo_summary = None


DEFAULT_OUTPUT_DIR = Path("reports/model_architecture")
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "siamese_resnet18_summary.txt"
DEFAULT_DIAGRAM_PATH = DEFAULT_OUTPUT_DIR / "siamese_resnet18_architecture.png"
DEFAULT_PUBLICATION_PNG_PATH = DEFAULT_OUTPUT_DIR / "siamese_resnet18_publication.png"
DEFAULT_PUBLICATION_PDF_PATH = DEFAULT_OUTPUT_DIR / "siamese_resnet18_publication.pdf"
DEFAULT_EXPLANATIONS_PATH = DEFAULT_OUTPUT_DIR / "architecture_explanations.md"


PALETTE = {
    "pre": "#E8F1FB",
    "post": "#FDEDEC",
    "shared": "#E9EEF5",
    "feature": "#EDF7ED",
    "difference": "#FFF4D6",
    "fusion": "#EDE9FE",
    "classifier": "#FDE2E4",
    "output": "#EEF2FF",
    "accent": "#DCE7F7",
    "edge": "#24324B",
    "arrow": "#3A4A66",
    "text": "#172033",
    "muted": "#5B6475",
}


RESNET18_STAGE_SPECS = [
    ("Conv1", "7x7 conv, 64, stride 2", "64 x 112 x 112"),
    ("MaxPool", "3x3 max pool, stride 2", "64 x 56 x 56"),
    ("Layer1", "2 x BasicBlock, 64", "64 x 56 x 56"),
    ("Layer2", "2 x BasicBlock, 128, stride 2", "128 x 28 x 28"),
    ("Layer3", "2 x BasicBlock, 256, stride 2", "256 x 14 x 14"),
    ("Layer4", "2 x BasicBlock, 512, stride 2", "512 x 7 x 7"),
    ("GlobalAvgPool", "adaptive average pooling", "512"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scientific architecture assets for the final Run B Siamese ResNet18 model."
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--diagram-path", type=str, default=str(DEFAULT_DIAGRAM_PATH))
    parser.add_argument("--publication-png-path", type=str, default=str(DEFAULT_PUBLICATION_PNG_PATH))
    parser.add_argument("--publication-pdf-path", type=str, default=str(DEFAULT_PUBLICATION_PDF_PATH))
    parser.add_argument("--explanations-path", type=str, default=str(DEFAULT_EXPLANATIONS_PATH))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(num_classes: int, dropout: float) -> tuple[SiameseResNet18, bool]:
    try:
        model = SiameseResNet18(
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
        )
        pretrained_loaded = True
    except Exception:
        model = SiameseResNet18(
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )
        pretrained_loaded = False
    return model, pretrained_loaded


def format_classifier_layers(layers: Iterable[torch.nn.Module]) -> str:
    lines = []
    for index, layer in enumerate(layers):
        lines.append(f"  [{index}] {layer}")
    return "\n".join(lines)


def build_summary_text(
    model: SiameseResNet18,
    pretrained_loaded: bool,
    image_size: int,
    stats_text: str,
) -> str:
    device = next(model.parameters()).device
    with torch.no_grad():
        pre_image = torch.randn(1, 3, image_size, image_size, device=device)
        post_image = torch.randn(1, 3, image_size, image_size, device=device)
        pre_features = model.extract_features(pre_image)
        post_features = model.extract_features(post_image)
        fused_features = model.fuse_features(pre_features, post_features)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    backbone_params = sum(parameter.numel() for parameter in model.backbone.parameters())
    classifier_params = sum(parameter.numel() for parameter in model.classifier.parameters())

    header = [
        "Siamese ResNet18 - Final Run B Architecture Summary",
        "=" * 60,
        "",
        "Configuration",
        "-" * 13,
        f"Model class: {model.__class__.__name__}",
        "num_classes: 4",
        "dropout: 0.4",
        "pretrained_requested: True",
        f"pretrained_loaded: {pretrained_loaded}",
        f"input_pre_shape: [1, 3, {image_size}, {image_size}]",
        f"input_post_shape: [1, 3, {image_size}, {image_size}]",
        "",
        "Feature Geometry",
        "-" * 16,
        f"pre_features shape: {list(pre_features.shape)}",
        f"post_features shape: {list(post_features.shape)}",
        f"backbone output dimension: {pre_features.shape[1]}",
        f"fused feature shape: {list(fused_features.shape)}",
        f"fused feature dimension: {fused_features.shape[1]}",
        "",
        "Classifier Head",
        "-" * 15,
        format_classifier_layers(model.classifier),
        "",
        "Parameter Counts",
        "-" * 16,
        f"backbone parameters: {backbone_params:,}",
        f"classifier parameters: {classifier_params:,}",
        f"total trainable parameters: {trainable_params:,}",
        f"total parameters: {total_params:,}",
        "",
        "Detailed torchinfo Summary",
        "-" * 26,
        stats_text,
    ]
    return "\n".join(header)


def build_torchinfo_text(model: SiameseResNet18, image_size: int, device: torch.device) -> str:
    if torchinfo_summary is None:
        raise ModuleNotFoundError(
            "torchinfo is required. Install it with: pip install torchinfo"
        )

    stats = torchinfo_summary(
        model,
        input_data=(
            torch.randn(1, 3, image_size, image_size, device=device),
            torch.randn(1, 3, image_size, image_size, device=device),
        ),
        depth=5,
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "trainable"),
        device=str(device),
    )
    return str(stats)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: int = 12,
    linewidth: float = 1.8,
    subtitle: str | None = None,
) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=PALETTE["edge"],
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.60,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="semibold",
        color=PALETTE["text"],
        wrap=True,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.24,
            subtitle,
            ha="center",
            va="center",
            fontsize=max(fontsize - 2, 8),
            color=PALETTE["muted"],
            wrap=True,
        )


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3,rad=0.0",
    linewidth: float = 1.8,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=linewidth,
        color=PALETTE["arrow"],
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_publication_diagram(png_path: Path, pdf_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    draw_box(ax, (0.03, 0.62), 0.13, 0.16, "PRE image", PALETTE["pre"], subtitle="[1, 3, 224, 224]")
    draw_box(ax, (0.03, 0.22), 0.13, 0.16, "POST image", PALETTE["post"], subtitle="[1, 3, 224, 224]")
    draw_box(ax, (0.24, 0.40), 0.17, 0.20, "Shared ResNet18", PALETTE["shared"], fontsize=13, subtitle="single backbone reused for both branches")

    draw_box(ax, (0.49, 0.66), 0.13, 0.12, "pre_features", PALETTE["feature"], subtitle="512-D")
    draw_box(ax, (0.49, 0.22), 0.13, 0.12, "post_features", PALETTE["feature"], subtitle="512-D")
    draw_box(ax, (0.66, 0.42), 0.14, 0.16, "|post - pre|", PALETTE["difference"], fontsize=13, subtitle="absolute temporal difference\n512-D")
    draw_box(ax, (0.66, 0.68), 0.16, 0.13, "Feature fusion", PALETTE["fusion"], fontsize=13, subtitle="concat(pre, post, diff)\n1536-D")
    draw_box(ax, (0.85, 0.39), 0.12, 0.28, "Classifier head", PALETTE["classifier"], fontsize=13, subtitle="Linear(1536, 512)\nReLU\nDropout(0.4)\nLinear(512, 4)")

    ax.text(
        0.915,
        0.18,
        "4 output classes\nno-damage | minor-damage\nmajor-damage | destroyed",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["text"],
        fontweight="semibold",
    )
    ax.text(0.325, 0.64, "shared weights", fontsize=11, color=PALETTE["muted"], ha="center")

    draw_arrow(ax, (0.16, 0.70), (0.24, 0.54))
    draw_arrow(ax, (0.16, 0.30), (0.24, 0.46))
    draw_arrow(ax, (0.41, 0.55), (0.49, 0.72))
    draw_arrow(ax, (0.41, 0.45), (0.49, 0.28))
    draw_arrow(ax, (0.62, 0.72), (0.66, 0.745))
    draw_arrow(ax, (0.62, 0.28), (0.66, 0.745), connectionstyle="arc3,rad=-0.18")
    draw_arrow(ax, (0.62, 0.72), (0.66, 0.50))
    draw_arrow(ax, (0.62, 0.28), (0.66, 0.50))
    draw_arrow(ax, (0.82, 0.745), (0.85, 0.53))
    draw_arrow(ax, (0.97, 0.53), (0.97, 0.24))

    ax.set_title(
        "Final Run B Architecture: Siamese ResNet18 for Bi-Temporal Flood Damage Classification",
        fontsize=17,
        fontweight="bold",
        color=PALETTE["text"],
        pad=16,
    )
    ax.text(
        0.5,
        0.03,
        "Shared CNN features are extracted independently from pre- and post-disaster crops, fused with an absolute difference vector,\nthen classified into four building damage categories.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    fig.tight_layout(pad=1.0)
    save_figure(fig, png_path, pdf_path)


def draw_resnet18_backbone_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 4.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    widths = [0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.11]
    x = 0.02
    y = 0.38
    h = 0.22
    centers = []
    for idx, (name, op, dim) in enumerate(RESNET18_STAGE_SPECS):
        w = widths[idx]
        draw_box(ax, (x, y), w, h, name, PALETTE["accent"], fontsize=12, subtitle=f"{op}\n{dim}")
        centers.append((x + w / 2, y + h / 2))
        if idx < len(RESNET18_STAGE_SPECS) - 1:
            draw_arrow(ax, (x + w, y + h / 2), (x + w + 0.03, y + h / 2))
        x += w + 0.03

    skip_specs = [
        (2, 2, "skip x2"),
        (3, 3, "skip x2"),
        (4, 4, "skip x2"),
        (5, 5, "skip x2"),
    ]
    for idx, _, label in skip_specs:
        cx, cy = centers[idx]
        ax.text(cx, 0.70, label, ha="center", va="center", fontsize=10.5, color=PALETTE["muted"])
        draw_arrow(ax, (cx - 0.045, 0.64), (cx + 0.045, 0.64), connectionstyle="arc3,rad=0.8", linewidth=1.4)

    ax.text(0.5, 0.90, "Detailed ResNet18 Backbone Used in the Siamese Damage Classifier", ha="center", fontsize=17, fontweight="bold", color=PALETTE["text"])
    ax.text(0.5, 0.14, "Each residual stage contains BasicBlocks with identity skip connections. The final 512-D embedding is obtained after global average pooling.", ha="center", fontsize=10.5, color=PALETTE["muted"])

    fig.tight_layout(pad=1.0)
    save_figure(fig, output_dir / "resnet18_backbone_detailed.png", output_dir / "resnet18_backbone_detailed.pdf")


def draw_residual_block_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.40), 0.14, 0.18, "Input x", PALETTE["accent"], fontsize=13)
    draw_box(ax, (0.28, 0.58), 0.16, 0.14, "3x3 Conv + BN + ReLU", PALETTE["shared"], fontsize=12)
    draw_box(ax, (0.50, 0.58), 0.16, 0.14, "3x3 Conv + BN", PALETTE["shared"], fontsize=12)
    draw_box(ax, (0.73, 0.40), 0.14, 0.18, "Add + ReLU", PALETTE["feature"], fontsize=13)
    draw_box(ax, (0.88, 0.40), 0.10, 0.18, "Output", PALETTE["fusion"], fontsize=13)

    draw_arrow(ax, (0.19, 0.49), (0.28, 0.65))
    draw_arrow(ax, (0.44, 0.65), (0.50, 0.65))
    draw_arrow(ax, (0.66, 0.65), (0.73, 0.49))
    draw_arrow(ax, (0.87, 0.49), (0.88, 0.49))
    draw_arrow(ax, (0.19, 0.49), (0.73, 0.49), connectionstyle="arc3,rad=-0.28")
    ax.text(0.46, 0.28, "identity shortcut / skip connection", ha="center", fontsize=11, color=PALETTE["muted"])

    ax.set_title("Residual BasicBlock and Residual Learning Mechanism", fontsize=16, fontweight="bold", color=PALETTE["text"], pad=14)
    ax.text(0.5, 0.08, "The shortcut path propagates the input directly to the block output, improving gradient flow and enabling deeper optimization.", ha="center", fontsize=10.5, color=PALETTE["muted"])

    fig.tight_layout(pad=1.0)
    save_figure(fig, output_dir / "residual_block_publication.png", output_dir / "residual_block_publication.pdf")


def draw_transfer_learning_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.04, 0.38), 0.18, 0.20, "ImageNet-pretrained\nResNet18 backbone", PALETTE["shared"], fontsize=13, subtitle="generic visual features")
    draw_box(ax, (0.32, 0.38), 0.16, 0.20, "Backbone reuse\ninside Siamese model", PALETTE["feature"], fontsize=13, subtitle="shared for PRE and POST")
    draw_box(ax, (0.58, 0.38), 0.16, 0.20, "Bi-temporal fusion\n+ classifier head", PALETTE["fusion"], fontsize=13, subtitle="task-specific adaptation")
    draw_box(ax, (0.83, 0.38), 0.13, 0.20, "Flood damage\nclassification", PALETTE["classifier"], fontsize=13, subtitle="4 classes")

    draw_arrow(ax, (0.22, 0.48), (0.32, 0.48))
    draw_arrow(ax, (0.48, 0.48), (0.58, 0.48))
    draw_arrow(ax, (0.74, 0.48), (0.83, 0.48))

    ax.set_title("Transfer Learning Pipeline for the Final Run B Classifier", fontsize=16, fontweight="bold", color=PALETTE["text"], pad=14)
    ax.text(0.5, 0.13, "Pretrained convolutional filters provide general low-level and mid-level visual representations, which are adapted to the xBD flood damage task through supervised fine-tuning.", ha="center", fontsize=10.5, color=PALETTE["muted"])

    fig.tight_layout(pad=1.0)
    save_figure(fig, output_dir / "transfer_learning_pipeline.png", output_dir / "transfer_learning_pipeline.pdf")


def draw_fine_tuning_figure(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(ax, (0.05, 0.58), 0.18, 0.16, "Backbone\ntrainable", PALETTE["shared"], fontsize=13, subtitle="all ResNet18 layers updated")
    draw_box(ax, (0.31, 0.58), 0.18, 0.16, "Classifier head\ntrainable", PALETTE["classifier"], fontsize=13, subtitle="task-specific damage head")
    draw_box(ax, (0.57, 0.58), 0.18, 0.16, "Regularization", PALETTE["difference"], fontsize=13, subtitle="dropout=0.4\nweight decay=5e-3")
    draw_box(ax, (0.31, 0.26), 0.18, 0.16, "Optimization", PALETTE["feature"], fontsize=13, subtitle="AdamW + cosine LR\nwarmup + mixed precision")
    draw_box(ax, (0.57, 0.26), 0.18, 0.16, "Generalization control", PALETTE["fusion"], fontsize=13, subtitle="focal loss\nweighted sampling\nearly stopping")

    draw_arrow(ax, (0.23, 0.66), (0.31, 0.66))
    draw_arrow(ax, (0.49, 0.66), (0.57, 0.66))
    draw_arrow(ax, (0.40, 0.58), (0.40, 0.42))
    draw_arrow(ax, (0.66, 0.58), (0.66, 0.42))
    draw_arrow(ax, (0.49, 0.34), (0.57, 0.34))

    ax.set_title("Fine-Tuning Strategy Used in Run B", fontsize=16, fontweight="bold", color=PALETTE["text"], pad=14)
    ax.text(0.5, 0.10, "Run B kept the pretrained backbone trainable while increasing classifier regularization and weight decay to reduce overfitting on the small, highly imbalanced xBD subset.", ha="center", fontsize=10.5, color=PALETTE["muted"])

    fig.tight_layout(pad=1.0)
    save_figure(fig, output_dir / "fine_tuning_strategy.png", output_dir / "fine_tuning_strategy.pdf")


def build_markdown_explanations(model: SiameseResNet18, pretrained_loaded: bool) -> str:
    backbone_params = sum(parameter.numel() for parameter in model.backbone.parameters())
    classifier_params = sum(parameter.numel() for parameter in model.classifier.parameters())
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    classifier_description = "\n".join(f"- {layer}" for layer in model.classifier)

    return f"""# Scientific Architecture Notes: Final Run B Siamese ResNet18

## 1. ResNet18 Backbone
The final classifier uses a ResNet18 backbone originally designed for large-scale natural image recognition. In this project, the final fully connected classification layer of the original network is removed and replaced by an identity mapping, so the backbone acts strictly as a feature extractor.

For an input crop of size `3 x 224 x 224`, the backbone produces a `512`-dimensional embedding after the following stages:

- Conv1: `7x7`, 64 filters, stride 2 -> `64 x 112 x 112`
- MaxPool: `3x3`, stride 2 -> `64 x 56 x 56`
- Layer1: 2 residual BasicBlocks -> `64 x 56 x 56`
- Layer2: 2 residual BasicBlocks -> `128 x 28 x 28`
- Layer3: 2 residual BasicBlocks -> `256 x 14 x 14`
- Layer4: 2 residual BasicBlocks -> `512 x 7 x 7`
- Global average pooling -> `512`

This design offers a good compromise between representational capacity and computational efficiency for medium-size satellite building crops.

## 2. Residual Learning
Residual learning is the central mechanism of ResNet18. Instead of learning a direct mapping `H(x)`, each residual block learns a residual function `F(x) = H(x) - x`, and the block output becomes `y = F(x) + x`.

This skip-connection design improves gradient propagation, stabilizes optimization, and reduces degradation when the network depth increases. In practice, it enables deeper feature extraction while preserving low-level information useful for structural building patterns.

## 3. Siamese Network Design
The model follows a Siamese architecture because each sample contains two temporally related building crops:

- a pre-disaster crop
- a post-disaster crop

The same ResNet18 backbone is reused for both branches, meaning the feature extractor has shared weights. This ensures that both inputs are embedded in the same representation space, which is crucial for meaningful temporal comparison.

The resulting branch features are:

- `pre_features`: `512-D`
- `post_features`: `512-D`

A temporal difference vector is then computed as `abs(post_features - pre_features)`.

## 4. Transfer Learning
The backbone is initialized with ImageNet-pretrained weights when available.
Pretrained weights were requested in the visualization build: `{pretrained_loaded}`.

Transfer learning is useful in this project because the xBD flood subset is relatively small and heavily imbalanced. Starting from pretrained low-level and mid-level filters helps the network learn more stable spatial representations than full random initialization.

## 5. Fine-Tuning Strategy
The final Run B strategy fine-tunes the backbone and the classifier jointly rather than freezing the CNN.
The main regularization and optimization choices were:

- dropout increased to `0.4`
- weight decay increased to `5e-3`
- focal loss used for class imbalance
- weighted random sampling during training
- cosine learning-rate schedule with warmup
- early stopping based on validation macro-F1
- mixed precision enabled on GPU

This strategy aimed to improve generalization while retaining the benefits of pretrained convolutional features.

## 6. Feature Fusion
Feature fusion is one of the key scientific choices of the final model. The classifier does not rely only on independent pre and post features; it explicitly models temporal change through an absolute difference vector.

The fusion vector is:

- `pre_features` (`512-D`)
- `post_features` (`512-D`)
- `abs(post_features - pre_features)` (`512-D`)

These three vectors are concatenated to form a `1536-D` representation. This fused representation is passed through the classifier head:

{classifier_description}

This fusion design is simple, interpretable, and effective for change-aware building damage classification.

## 7. Parameter Summary
- Backbone parameters: `{backbone_params:,}`
- Classifier parameters: `{classifier_params:,}`
- Total trainable parameters: `{trainable_params:,}`
- Total parameters: `{total_params:,}`

The parameter budget remains moderate for a deep convolutional model, which is appropriate for Colab-scale experimentation and deployment-oriented prototyping.
"""


def save_architecture_diagram(output_path: Path) -> None:
    legacy_pdf = output_path.with_suffix(".pdf")
    draw_publication_diagram(output_path, legacy_pdf)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    diagram_path = Path(args.diagram_path)
    publication_png_path = Path(args.publication_png_path)
    publication_pdf_path = Path(args.publication_pdf_path)
    explanations_path = Path(args.explanations_path)
    output_dir = summary_path.parent

    for path in (summary_path, diagram_path, publication_png_path, publication_pdf_path, explanations_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, pretrained_loaded = build_model(
        num_classes=args.num_classes,
        dropout=args.dropout,
    )
    model = model.to(device)
    model.eval()

    stats_text = build_torchinfo_text(model=model, image_size=args.image_size, device=device)
    summary_text = build_summary_text(
        model=model,
        pretrained_loaded=pretrained_loaded,
        image_size=args.image_size,
        stats_text=stats_text,
    )

    summary_path.write_text(summary_text, encoding="utf-8")
    explanations_path.write_text(build_markdown_explanations(model, pretrained_loaded), encoding="utf-8")

    save_architecture_diagram(diagram_path)
    draw_publication_diagram(publication_png_path, publication_pdf_path)
    draw_resnet18_backbone_figure(output_dir)
    draw_residual_block_figure(output_dir)
    draw_transfer_learning_figure(output_dir)
    draw_fine_tuning_figure(output_dir)

    print(summary_text)
    print()
    print(f"Saved text summary to: {summary_path}")
    print(f"Saved explanations markdown to: {explanations_path}")
    print(f"Saved standard architecture diagram to: {diagram_path}")
    print(f"Saved publication PNG to: {publication_png_path}")
    print(f"Saved publication PDF to: {publication_pdf_path}")
    print(f"Saved ResNet18 backbone figure to: {output_dir / 'resnet18_backbone_detailed.png'}")
    print(f"Saved residual block figure to: {output_dir / 'residual_block_publication.png'}")
    print(f"Saved transfer learning figure to: {output_dir / 'transfer_learning_pipeline.png'}")
    print(f"Saved fine-tuning figure to: {output_dir / 'fine_tuning_strategy.png'}")


if __name__ == "__main__":
    main()

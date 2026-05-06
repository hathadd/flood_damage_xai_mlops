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


PALETTE = {
    "pre": "#E8F1FB",
    "post": "#FDEDEC",
    "shared": "#E9EEF5",
    "feature": "#EDF7ED",
    "difference": "#FFF4D6",
    "fusion": "#EDE9FE",
    "classifier": "#FDE2E4",
    "output": "#EEF2FF",
    "edge": "#24324B",
    "arrow": "#3A4A66",
    "text": "#172033",
    "muted": "#5B6475",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the final Run B Siamese ResNet18 architecture."
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--diagram-path", type=str, default=str(DEFAULT_DIAGRAM_PATH))
    parser.add_argument("--publication-png-path", type=str, default=str(DEFAULT_PUBLICATION_PNG_PATH))
    parser.add_argument("--publication-pdf-path", type=str, default=str(DEFAULT_PUBLICATION_PDF_PATH))
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
        f"num_classes: 4",
        f"dropout: 0.4",
        f"pretrained_requested: True",
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
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=PALETTE["arrow"],
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def draw_publication_diagram(png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    draw_arrow(ax, (0.62, 0.72), (0.66, 0.50), connectionstyle="arc3,rad=0.0")
    draw_arrow(ax, (0.62, 0.28), (0.66, 0.50), connectionstyle="arc3,rad=0.0")
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
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_architecture_diagram(output_path: Path) -> None:
    legacy_pdf = output_path.with_suffix(".pdf")
    draw_publication_diagram(output_path, legacy_pdf)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    diagram_path = Path(args.diagram_path)
    publication_png_path = Path(args.publication_png_path)
    publication_pdf_path = Path(args.publication_pdf_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    diagram_path.parent.mkdir(parents=True, exist_ok=True)
    publication_png_path.parent.mkdir(parents=True, exist_ok=True)
    publication_pdf_path.parent.mkdir(parents=True, exist_ok=True)

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
    save_architecture_diagram(diagram_path)
    draw_publication_diagram(publication_png_path, publication_pdf_path)

    print(summary_text)
    print()
    print(f"Saved text summary to: {summary_path}")
    print(f"Saved standard architecture diagram to: {diagram_path}")
    print(f"Saved publication PNG to: {publication_png_path}")
    print(f"Saved publication PDF to: {publication_pdf_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SiameseResNet18(nn.Module):
    """Bi-temporal Siamese ResNet18 baseline for building-level damage classification."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone, feature_dim = self._build_backbone(weights=weights)
        fused_feature_dim = feature_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    @staticmethod
    def _build_backbone(weights: ResNet18_Weights | None = None) -> tuple[nn.Module, int]:
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def fuse_features(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor,
    ) -> torch.Tensor:
        temporal_difference = torch.abs(post_features - pre_features)
        return torch.cat([pre_features, post_features, temporal_difference], dim=1)

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        pre_features = self.extract_features(pre_image)
        post_features = self.extract_features(post_image)
        fused_features = self.fuse_features(pre_features, post_features)
        return self.classifier(fused_features)


def build_siamese_resnet18(
    num_classes: int = 4,
    pretrained: bool = False,
    dropout: float = 0.2,
    **_: Any,
) -> SiameseResNet18:
    return SiameseResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

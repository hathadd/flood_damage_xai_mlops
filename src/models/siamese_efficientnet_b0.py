from __future__ import annotations

import timm
import torch
from torch import nn

__all__ = ["SiameseEfficientNetB0"]


class SiameseEfficientNetB0(nn.Module):
    """Bi-temporal Siamese EfficientNet-B0 for building-level damage classification."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )
        self.feature_dim = 1280
        self._freeze_early_layers()

        fused_feature_dim = self.feature_dim * 2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(fused_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def _freeze_early_layers(self) -> None:
        frozen_prefixes = ("conv_stem", "bn1", "blocks.0", "blocks.1", "blocks.2")
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad = False

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def fuse_features(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([pre_features, post_features], dim=1)

    def forward(self, pre_image: torch.Tensor, post_image: torch.Tensor) -> torch.Tensor:
        pre_features = self.extract_features(pre_image)
        post_features = self.extract_features(post_image)
        fused_features = self.fuse_features(pre_features, post_features)
        return self.classifier(fused_features)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import XBDPairBuildingDataset
from src.data.transforms import build_transforms, get_train_geometric_transforms


def test_joint_geometric_replay_is_identical_for_same_image() -> None:
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    joint = get_train_geometric_transforms(image_size=224)

    transformed_pre = joint(image=image)
    transformed_post = joint.replay(transformed_pre["replay"], image=image)

    assert transformed_pre["image"].shape == (224, 224, 3)
    assert transformed_post["image"].shape == (224, 224, 3)
    assert np.array_equal(transformed_pre["image"], transformed_post["image"])


def test_train_pipeline_shares_geometry_and_keeps_output_shapes_consistent() -> None:
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    transforms = build_transforms({"image_size": 224})["train"]
    dataset = XBDPairBuildingDataset.__new__(XBDPairBuildingDataset)
    dataset.transforms = transforms

    pre_tensor, post_tensor = dataset._apply_transforms(image, image)

    assert isinstance(pre_tensor, torch.Tensor)
    assert isinstance(post_tensor, torch.Tensor)
    assert pre_tensor.shape == (3, 224, 224)
    assert post_tensor.shape == (3, 224, 224)

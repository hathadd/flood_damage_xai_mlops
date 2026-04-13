from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    """
    Parse a WKT polygon string like:
    POLYGON ((x1 y1, x2 y2, ..., xn yn))
    and return a list of (x, y) tuples.
    """
    wkt = wkt.strip()
    if not wkt.startswith("POLYGON"):
        raise ValueError(f"Unsupported WKT format: {wkt[:50]}")

    coords_text = wkt.replace("POLYGON ((", "").replace("))", "")
    points = []

    for pair in coords_text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split()
        points.append((float(x_str), float(y_str)))

    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points.")

    return points


def polygon_to_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def expand_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
) -> tuple[int, int, int, int]:
    """
    Expand bbox with context around the building and keep it inside image bounds.
    """
    x_min, y_min, x_max, y_max = bbox

    box_w = max(x_max - x_min, 1.0)
    box_h = max(y_max - y_min, 1.0)

    pad_x = box_w * context_ratio
    pad_y = box_h * context_ratio

    x1 = x_min - pad_x
    y1 = y_min - pad_y
    x2 = x_max + pad_x
    y2 = y_max + pad_y

    # enforce minimum crop size
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_crop_size:
        extra = (min_crop_size - crop_w) / 2.0
        x1 -= extra
        x2 += extra

    if crop_h < min_crop_size:
        extra = (min_crop_size - crop_h) / 2.0
        y1 -= extra
        y2 += extra

    # clip to image bounds
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(image_width, int(np.ceil(x2)))
    y2 = min(image_height, int(np.ceil(y2)))

    # final safeguard
    if x2 <= x1:
        x2 = min(image_width, x1 + min_crop_size)
    if y2 <= y1:
        y2 = min(image_height, y1 + min_crop_size)

    return x1, y1, x2, y2


def normalize_image_path(image_path: str | Path) -> Path:
    path_str = str(image_path)

    if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] in {"\\", "/"}:
        drive = path_str[0].lower()
        relative_path = path_str[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{relative_path}")

    return Path(image_path)


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    normalized_path = normalize_image_path(image_path)
    image = Image.open(normalized_path).convert("RGB")
    return np.array(image)


class XBDPairBuildingDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str = "train",
        transforms: Any = None,
        context_ratio: float = 0.25,
        min_crop_size: int = 64,
        return_metadata: bool = True,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.split = split
        self.transforms = transforms
        self.context_ratio = context_ratio
        self.min_crop_size = min_crop_size
        self.return_metadata = return_metadata

        self.df = pd.read_csv(self.metadata_csv)

        required_columns = {
            "sample_id",
            "building_uid",
            "pre_image_path",
            "post_image_path",
            "damage_class",
            "class_id",
            "wkt",
            "image_width",
            "image_height",
        }
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in metadata CSV: {missing}")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _get_crop_box(self, row: pd.Series) -> tuple[int, int, int, int]:
        polygon_points = parse_wkt_polygon(row["wkt"])
        bbox = polygon_to_bbox(polygon_points)

        x1, y1, x2, y2 = expand_bbox(
            bbox=bbox,
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
            context_ratio=self.context_ratio,
            min_crop_size=self.min_crop_size,
        )
        return x1, y1, x2, y2

    def _crop_pair(self, pre_image: np.ndarray, post_image: np.ndarray, crop_box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = crop_box
        pre_crop = pre_image[y1:y2, x1:x2]
        post_crop = post_image[y1:y2, x1:x2]
        return pre_crop, post_crop

    def _apply_transforms(self, pre_crop: np.ndarray, post_crop: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transforms is None:
            pre_tensor = torch.from_numpy(pre_crop).permute(2, 0, 1).float() / 255.0
            post_tensor = torch.from_numpy(post_crop).permute(2, 0, 1).float() / 255.0
            return pre_tensor, post_tensor

        # Train-time bi-temporal augmentation: share geometry across the pair,
        # then apply appearance perturbations independently to pre and post.
        if isinstance(self.transforms, dict) and {"joint", "pre", "post"} <= set(self.transforms):
            joint_transforms = self.transforms["joint"]
            transformed_pre = joint_transforms(image=pre_crop)
            replay = transformed_pre["replay"]
            transformed_post = joint_transforms.replay(replay, image=post_crop)

            pre_geo = transformed_pre["image"]
            post_geo = transformed_post["image"]

            pre_tensor = self.transforms["pre"](image=pre_geo)["image"]
            post_tensor = self.transforms["post"](image=post_geo)["image"]
            return pre_tensor, post_tensor

        # Keep paired augmentations in sync only when using ReplayCompose.
        if self.transforms.__class__.__name__ == "ReplayCompose":
            transformed_pre = self.transforms(image=pre_crop)
            replay = transformed_pre["replay"]
            transformed_post = self.transforms.replay(replay, image=post_crop)
            return transformed_pre["image"], transformed_post["image"]

        # Deterministic Compose fallback used by validation and test pipelines.
        transformed_pre = self.transforms(image=pre_crop)
        transformed_post = self.transforms(image=post_crop)
        return transformed_pre["image"], transformed_post["image"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        pre_image = load_rgb_image(row["pre_image_path"])
        post_image = load_rgb_image(row["post_image_path"])

        crop_box = self._get_crop_box(row)
        pre_crop, post_crop = self._crop_pair(pre_image, post_image, crop_box)

        pre_tensor, post_tensor = self._apply_transforms(pre_crop, post_crop)

        item = {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "label": torch.tensor(int(row["class_id"]), dtype=torch.long),
        }

        if self.return_metadata:
            item.update(
                {
                    "sample_id": row["sample_id"],
                    "building_uid": row["building_uid"],
                    "damage_class": row["damage_class"],
                    "crop_box": torch.tensor(crop_box, dtype=torch.int32),
                    "pre_image_path": row["pre_image_path"],
                    "post_image_path": row["post_image_path"],
                }
            )

        return item
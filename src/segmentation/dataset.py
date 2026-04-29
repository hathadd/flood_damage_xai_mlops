from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.path_utils import resolve_data_path
from src.segmentation.mask_utils import rasterize_building_mask, save_mask


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: Path
    label_json_path: Path
    image_width: int
    image_height: int
    mask_cache_path: Path


class XBDBuildingSegmentationDataset(Dataset):
    def __init__(
        self,
        split_metadata_path: str | Path,
        split: str,
        dataset_root: str | Path | None = None,
        image_type: str = "post_image",
        transforms: Any | None = None,
        mask_cache_dir: str | Path = "data/segmentation_masks_cache",
        save_generated_masks: bool = True,
    ) -> None:
        if image_type not in {"post_image", "pre_image"}:
            raise ValueError("image_type must be either 'post_image' or 'pre_image'.")

        self.split_metadata_path = Path(split_metadata_path)
        self.split = split
        self.dataset_root = dataset_root
        self.image_type = image_type
        self.transforms = transforms
        self.mask_cache_dir = Path(mask_cache_dir)
        self.save_generated_masks = save_generated_masks

        dataframe = pd.read_csv(self.split_metadata_path)
        split_df = dataframe.loc[dataframe["split"] == split].copy()
        if split_df.empty:
            raise ValueError(f"No samples found for split '{split}'.")

        image_column = f"{image_type}_path"
        required_columns = ["sample_id", image_column, "label_json_path", "image_width", "image_height"]
        missing_columns = [column for column in required_columns if column not in split_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in split metadata: {missing_columns}")

        dedup_columns = ["sample_id", image_column, "label_json_path"]
        split_df = split_df.drop_duplicates(subset=dedup_columns).reset_index(drop=True)

        self.df = split_df
        self.samples = [self._build_sample(row) for _, row in self.df.iterrows()]

    def _build_sample(self, row: pd.Series) -> SegmentationSample:
        image_column = f"{self.image_type}_path"
        image_path = resolve_data_path(row[image_column], dataset_root=self.dataset_root)
        label_json_path = resolve_data_path(row["label_json_path"], dataset_root=self.dataset_root)
        relative_mask_name = f"{Path(str(row['sample_id'])).stem}.png"
        mask_cache_path = self.mask_cache_dir / self.image_type / self.split / relative_mask_name

        return SegmentationSample(
            sample_id=str(row["sample_id"]),
            image_path=image_path,
            label_json_path=label_json_path,
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
            mask_cache_path=mask_cache_path,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_or_create_mask(self, sample: SegmentationSample) -> np.ndarray:
        if sample.mask_cache_path.exists():
            return (np.asarray(Image.open(sample.mask_cache_path).convert("L")) > 0).astype(np.uint8)

        mask = rasterize_building_mask(
            json_path=sample.label_json_path,
            image_width=sample.image_width,
            image_height=sample.image_height,
        )
        if self.save_generated_masks:
            save_mask(mask, sample.mask_cache_path)
        return mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_array = np.asarray(image)
        mask_array = self._load_or_create_mask(sample)

        if self.transforms is not None:
            transformed = self.transforms(image=image_array, mask=mask_array)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
        else:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_array)

        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.as_tensor(mask_tensor)
        mask_tensor = mask_tensor.float()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            mask_tensor = mask_tensor[:1]

        return image_tensor, (mask_tensor > 0).float()

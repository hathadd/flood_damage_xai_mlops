from __future__ import annotations

import numpy as np
from scipy import ndimage


def threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask, dtype=np.float32) >= float(threshold)).astype(np.uint8)


def filter_small_components(mask: np.ndarray, min_area: int = 32) -> np.ndarray:
    binary_mask = threshold_mask(mask)
    labeled_mask, num_components = ndimage.label(binary_mask)
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for component_id in range(1, num_components + 1):
        component = labeled_mask == component_id
        if int(component.sum()) >= int(min_area):
            filtered_mask[component] = 1
    return filtered_mask


def connected_components_to_bboxes(mask: np.ndarray, min_area: int = 32) -> list[tuple[int, int, int, int]]:
    filtered_mask = filter_small_components(mask, min_area=min_area)
    labeled_mask, num_components = ndimage.label(filtered_mask)
    bboxes: list[tuple[int, int, int, int]] = []

    for component_id in range(1, num_components + 1):
        ys, xs = np.where(labeled_mask == component_id)
        if xs.size == 0 or ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

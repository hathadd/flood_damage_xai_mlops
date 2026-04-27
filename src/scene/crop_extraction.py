from __future__ import annotations

from typing import Iterable

from PIL import Image


def polygon_to_bbox(polygon: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    points = list(polygon)
    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points.")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_with_context(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    context_ratio: float = 0.25,
    min_crop_size: int = 64,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if min_crop_size <= 0:
        raise ValueError("min_crop_size must be strictly positive.")

    image_width, image_height = image.size
    x_min, y_min, x_max, y_max = bbox

    box_w = max(x_max - x_min, 1.0)
    box_h = max(y_max - y_min, 1.0)

    pad_x = box_w * context_ratio
    pad_y = box_h * context_ratio

    x1 = x_min - pad_x
    y1 = y_min - pad_y
    x2 = x_max + pad_x
    y2 = y_max + pad_y

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

    x1 = max(0, int(x1 // 1))
    y1 = max(0, int(y1 // 1))
    x2 = min(image_width, int(-(-x2 // 1)))
    y2 = min(image_height, int(-(-y2 // 1)))

    if x2 <= x1:
        x2 = min(image_width, x1 + min_crop_size)
    if y2 <= y1:
        y2 = min(image_height, y1 + min_crop_size)

    final_crop_box = (x1, y1, x2, y2)
    crop = image.crop(final_crop_box)
    return crop, final_crop_box

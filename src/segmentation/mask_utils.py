from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _parse_wkt_polygon(wkt: str) -> list[tuple[float, float]]:
    text = wkt.strip()
    if not text.startswith("POLYGON"):
        raise ValueError(f"Unsupported WKT format: {text[:50]}")

    coords_text = text.replace("POLYGON ((", "").replace("))", "")
    points: list[tuple[float, float]] = []
    for pair in coords_text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split()
        if len(parts) < 2:
            continue
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points.")
    return points


def _extract_polygon(feature: dict[str, Any]) -> list[tuple[float, float]] | None:
    if isinstance(feature.get("wkt"), str):
        try:
            return _parse_wkt_polygon(feature["wkt"])
        except ValueError:
            return None

    properties = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
    if isinstance(properties.get("wkt"), str):
        try:
            return _parse_wkt_polygon(properties["wkt"])
        except ValueError:
            return None

    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
        coordinates = geometry.get("coordinates", [])
        if coordinates and isinstance(coordinates[0], list):
            ring = coordinates[0]
            points = []
            for point in ring:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                points.append((float(point[0]), float(point[1])))
            if len(points) >= 3:
                return points
    return None


def parse_polygons_from_xbd_json(json_path: str | Path) -> list[list[tuple[float, float]]]:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    features_root = payload.get("features", {})
    xy_features = features_root.get("xy", [])
    if not isinstance(xy_features, list):
        return []

    polygons: list[list[tuple[float, float]]] = []
    for feature in xy_features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        feature_type = properties.get("feature_type")
        if feature_type not in {None, "building"}:
            continue
        polygon = _extract_polygon(feature)
        if polygon is None:
            continue
        polygons.append(polygon)
    return polygons


def rasterize_building_mask(
    json_path: str | Path,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    mask_image = Image.new("L", (int(image_width), int(image_height)), color=0)
    draw = ImageDraw.Draw(mask_image)

    for polygon in parse_polygons_from_xbd_json(json_path):
        try:
            clipped_polygon = [
                (
                    min(max(float(x), 0.0), float(image_width - 1)),
                    min(max(float(y), 0.0), float(image_height - 1)),
                )
                for x, y in polygon
            ]
            if len(clipped_polygon) >= 3:
                draw.polygon(clipped_polygon, outline=1, fill=1)
        except (TypeError, ValueError):
            continue

    return np.asarray(mask_image, dtype=np.uint8)


def save_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(output)
    return output

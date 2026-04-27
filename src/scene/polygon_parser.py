from __future__ import annotations

import json
from typing import Any

from src.scene.crop_extraction import polygon_to_bbox


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
        x_str, y_str = pair.split()
        points.append((float(x_str), float(y_str)))
    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points.")
    return points


def _extract_polygon(feature: dict[str, Any]) -> list[tuple[float, float]] | None:
    if isinstance(feature.get("wkt"), str):
        return _parse_wkt_polygon(feature["wkt"])

    properties = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
    if isinstance(properties.get("wkt"), str):
        return _parse_wkt_polygon(properties["wkt"])

    geometry = feature.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
        coordinates = geometry.get("coordinates", [])
        if coordinates and isinstance(coordinates[0], list):
            ring = coordinates[0]
            points = [(float(point[0]), float(point[1])) for point in ring if len(point) >= 2]
            if len(points) >= 3:
                return points

    return None


def parse_xbd_buildings(json_bytes: bytes) -> list[dict[str, Any]]:
    payload = json.loads(json_bytes.decode("utf-8"))
    features_root = payload.get("features", {})
    xy_features = features_root.get("xy", [])
    if not isinstance(xy_features, list):
        raise ValueError("Invalid xBD JSON: expected features.xy to be a list.")

    buildings: list[dict[str, Any]] = []
    for index, feature in enumerate(xy_features):
        if not isinstance(feature, dict):
            continue

        properties = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
        feature_type = properties.get("feature_type")
        if feature_type not in {None, "building"}:
            continue

        polygon = _extract_polygon(feature)
        if polygon is None:
            continue

        buildings.append(
            {
                "building_index": index,
                "building_uid": properties.get("uid"),
                "true_label": properties.get("subtype"),
                "polygon": [[float(x), float(y)] for x, y in polygon],
                "bbox": list(polygon_to_bbox(polygon)),
            }
        )

    return buildings

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "no-damage": "#16a34a",
    "minor-damage": "#eab308",
    "major-damage": "#f97316",
    "destroyed": "#dc2626",
}


def annotate_post_image(
    post_image: Image.Image,
    predictions: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> tuple[Image.Image, Path | None]:
    annotated = post_image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for prediction in predictions:
        predicted_label = str(prediction.get("predicted_label", "unknown"))
        confidence = float(prediction.get("confidence", 0.0))
        color = CLASS_COLORS.get(predicted_label, "#2563eb")

        polygon = prediction.get("polygon")
        if polygon:
            points = [tuple(point) for point in polygon]
            draw.polygon(points, outline=color, width=3)
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            x_min, y_min = min(xs), min(ys)
        else:
            bbox = prediction.get("bbox", prediction.get("crop_box"))
            if not bbox:
                continue
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)

        text = f"{predicted_label} ({confidence:.2f})"
        text_x = int(x_min)
        text_y = max(0, int(y_min) - 14)
        draw.rectangle((text_x, text_y, text_x + max(80, len(text) * 7), text_y + 14), fill=color)
        draw.text((text_x + 2, text_y + 1), text, fill="black", font=font)

    saved_path: Path | None = None
    if output_path is not None:
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(saved_path)

    return annotated, saved_path

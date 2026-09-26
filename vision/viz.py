"""Draw Grok boxes and labels on an image for debugging."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision.schemas import IndexedObject, norm_box_to_pixels

GROK_COLOR = (32, 140, 64)
TEXT_COLOR = (255, 255, 255)


def annotate_image(
    source: Path,
    objects: list[IndexedObject],
    dest: Path,
) -> Path:
    """Save a copy of source with a green box and label for each Grok object."""
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    width, height = image.size

    for obj in objects:
        x1, y1, x2, y2 = norm_box_to_pixels(obj.box, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=GROK_COLOR, width=3)
        caption = _caption(obj)
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = y1 - text_height - 6
        if top < 0:
            top = y1
        draw.rectangle(
            [x1, top, min(width - 1, x1 + text_width + 8), top + text_height + 4],
            fill=GROK_COLOR,
        )
        draw.text((x1 + 4, top + 2), caption, fill=TEXT_COLOR, font=font)

    legend = "Grok"
    legend_box = draw.textbbox((0, 0), legend, font=font)
    legend_w = legend_box[2] - legend_box[0]
    legend_h = legend_box[3] - legend_box[1]
    draw.rectangle([8, 8, 16 + legend_w, 14 + legend_h], fill=GROK_COLOR)
    draw.text((12, 10), legend, fill=TEXT_COLOR, font=font)

    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest


def _caption(obj: IndexedObject) -> str:
    parts = [obj.label]
    if obj.count != 1:
        parts.append(f"x{obj.count}")
    if obj.drug_name:
        parts.append(obj.drug_name)
    text = " · ".join(parts)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

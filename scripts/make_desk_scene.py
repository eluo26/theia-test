"""Draw the synthetic desk scene used by geometry-aware tests.

The pictures are fixtures: colored shapes at known azimuth and elevation.
They are not photographs and they are not Grok output. Regenerate after
changing hfov_deg or vfov_deg in config.yaml:

    python scripts/make_desk_scene.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vision.config import load_settings
from vision.geometry import direction_to_pixel

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "test_scenes" / "desk"
WIDTH = 640
HEIGHT = 480
TIMESTAMP = "2026-09-26T15:40:00"

VIEWS = [(-40, 0), (-20, 0), (0, 0), (20, 0), (40, 0), (0, -15)]

OBJECTS = [
    {"label": "yellow lamp", "az": -22.0, "el": 6.0, "half_w": 6.0, "half_h": 8.0, "kind": "lamp"},
    {"label": "green mug", "az": 36.0, "el": -6.0, "half_w": 4.5, "half_h": 4.5, "kind": "mug"},
    {"label": "Jardiance", "az": 2.0, "el": -4.0, "half_w": 6.0, "half_h": 5.0, "kind": "jardiance"},
    {"label": "blue water bottle", "az": 18.0, "el": -8.0, "half_w": 3.5, "half_h": 9.0, "kind": "bottle"},
]

GROUND_TRUTH = [
    {
        "query": "where's my blue water bottle?",
        "expected_label": "blue water bottle",
        "az": 18.0,
        "el": -8.0,
    },
    {
        "query": "where's the Jardiance starter pack?",
        "expected_label": "Jardiance",
        "az": 2.0,
        "el": -4.0,
    },
    {
        "query": "where is the yellow lamp?",
        "expected_label": "yellow lamp",
        "az": -22.0,
        "el": 6.0,
    },
    {
        "query": "where is the green mug?",
        "expected_label": "green mug",
        "az": 36.0,
        "el": -6.0,
    },
]


def _fmt(value: float) -> str:
    number = int(round(value))
    if abs(value - number) > 1e-6:
        return f"{value:.1f}"
    sign = "-" if number < 0 else ""
    return f"{sign}{abs(number):03d}"


def frame_name(pan: float, tilt: float) -> str:
    return f"pan{_fmt(pan)}_tilt{_fmt(tilt)}.png"


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _projected_box(obj, pan, tilt, settings):
    corners = []
    for daz in (-obj["half_w"], obj["half_w"]):
        for del_el in (-obj["half_h"], obj["half_h"]):
            u, v, z = direction_to_pixel(
                obj["az"] + daz,
                obj["el"] + del_el,
                pan,
                tilt,
                WIDTH,
                HEIGHT,
                settings,
            )
            if z <= 0:
                return None
            corners.append((u, v))
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    box = (min(xs), min(ys), max(xs), max(ys))
    x1, y1, x2, y2 = box
    if x2 <= 0 or y2 <= 0 or x1 >= WIDTH or y1 >= HEIGHT:
        return None
    return box


def _draw_object(draw: ImageDraw.ImageDraw, kind: str, box) -> None:
    x1, y1, x2, y2 = box
    if kind == "bottle":
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=(38, 104, 186))
    elif kind == "lamp":
        draw.polygon(
            [(x1, y2), (x2, y2), (x1 + (x2 - x1) * 0.8, y1), (x1 + (x2 - x1) * 0.2, y1)],
            fill=(240, 196, 64),
        )
    elif kind == "mug":
        draw.ellipse([x1, y1, x2, y2], fill=(46, 130, 78))
    elif kind == "jardiance":
        draw.rectangle([x1, y1, x2, y2], fill=(248, 248, 246), outline=(180, 40, 40))
        size = max(12, int((y2 - y1) * 0.28))
        draw.text((x1 + 6, y1 + 6), "JARDIANCE", fill=(20, 20, 20), font=_font(size))
        draw.text((x1 + 6, y1 + 8 + size), "EXP 2027-04", fill=(90, 40, 40), font=_font(max(10, size - 4)))


def render(pan: float, tilt: float, settings) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (232, 224, 208))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(HEIGHT * 0.72), WIDTH, HEIGHT], fill=(186, 154, 112))
    for obj in OBJECTS:
        box = _projected_box(obj, pan, tilt, settings)
        if box is not None:
            _draw_object(draw, obj["kind"], box)
    return image


def main() -> None:
    settings = load_settings(load_env=False)
    DEST.mkdir(parents=True, exist_ok=True)
    frames = []
    for pan, tilt in VIEWS:
        name = frame_name(pan, tilt)
        render(pan, tilt, settings).save(DEST / name)
        frames.append({"file": name, "pan": pan, "tilt": tilt, "timestamp": TIMESTAMP})
    manifest = {
        "note": "Synthetic fixture angles for the desk scene. Not a Grok response.",
        "frames": frames,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (DEST / "ground_truth.json").write_text(json.dumps(GROUND_TRUTH, indent=2) + "\n", encoding="utf-8")
    print(DEST)


if __name__ == "__main__":
    main()

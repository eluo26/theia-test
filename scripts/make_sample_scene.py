"""Draw a small synthetic scene so milestone 1 has an image to send to Grok.

This is a stand-in until real angle-tagged photos exist in data/test_scenes/.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "test_scenes" / "m1" / "scene.png"


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def main() -> None:
    image = Image.new("RGB", (960, 640), (232, 224, 208))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 430, 960, 640], fill=(186, 154, 112))
    draw.rectangle([70, 150, 190, 470], fill=(38, 104, 186))
    draw.ellipse([95, 130, 165, 190], fill=(38, 104, 186))
    draw.rectangle([250, 250, 470, 430], fill=(248, 248, 246), outline=(180, 40, 40), width=8)
    draw.text((275, 300), "JARDIANCE", fill=(20, 20, 20), font=_font(28))
    draw.text((275, 345), "EXP 2027-04", fill=(90, 40, 40), font=_font(20))
    draw.rectangle([560, 80, 590, 300], fill=(90, 90, 90))
    draw.polygon([(500, 300), (700, 300), (650, 120), (540, 120)], fill=(240, 196, 64))
    draw.ellipse([760, 280, 900, 430], fill=(46, 130, 78))
    draw.rectangle([800, 400, 860, 470], fill=(46, 130, 78))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    image.save(DEST)
    print(DEST)


if __name__ == "__main__":
    main()

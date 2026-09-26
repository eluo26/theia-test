from pathlib import Path

from PIL import Image

from vision.schemas import IndexedObject
from vision.viz import GROK_COLOR, annotate_image


def test_annotate_draws_a_green_box(tmp_path: Path):
    source = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), (240, 240, 240)).save(source)
    obj = IndexedObject.model_validate(
        {
            "label": "mug",
            "description": "green mug",
            "box": [10, 10, 80, 70],
            "count": 1,
            "drug_name": None,
            "expiry_text": None,
        }
    )
    dest = tmp_path / "out" / "scene_annotated.png"
    annotate_image(source, [obj], dest)
    painted = Image.open(dest).convert("RGB")
    assert painted.getpixel((20, 12)) == GROK_COLOR
    assert painted.getpixel((50, 70)) == (240, 240, 240)

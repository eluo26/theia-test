from pathlib import Path

from PIL import Image

import numpy as np

from vision.schemas import Detection, IndexedObject
from vision.viz import DETECTOR_COLOR, GROK_COLOR, annotate_detections, annotate_image


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


def test_debug_image_uses_a_second_color_for_detector_boxes(tmp_path: Path):
    image = np.full((80, 100, 3), 240, dtype=np.uint8)
    detection = Detection(
        label="mug",
        description="green mug",
        bbox_px=[10, 10, 80, 70],
        grok_bbox_px=[8, 8, 84, 74],
        count=1,
        confidence=0.9,
        box_source="detector",
        frame_file="pan000_tilt000.png",
        pan_deg=0,
        tilt_deg=0,
        azimuth_deg=0,
        elevation_deg=0,
        image_width=100,
        image_height=80,
    )
    dest = tmp_path / "debug.png"
    annotate_detections(image, [detection], dest, ids={(10, 10, 80, 70): "obj_001"})
    painted = np.asarray(Image.open(dest).convert("RGB"))
    colors = {tuple(int(channel) for channel in pixel) for pixel in painted.reshape(-1, 3)}
    assert DETECTOR_COLOR in colors
    assert GROK_COLOR in colors

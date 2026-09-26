import json
from pathlib import Path

from PIL import Image

from vision.config import load_settings
from vision.geometry import direction_to_pixel
from vision.ingest import ingest_directory

DESK = Path("data/test_scenes/desk")


def test_desk_scene_manifest_matches_filenames():
    manifest = json.loads((DESK / "manifest.json").read_text(encoding="utf-8"))
    assert "fixture" in manifest["note"].lower() or "synthetic" in manifest["note"].lower()
    frames = ingest_directory(DESK)
    by_name = {frame.source_file: frame for frame in frames}
    assert len(by_name) == len(manifest["frames"])
    for entry in manifest["frames"]:
        frame = by_name[entry["file"]]
        assert frame.pan_deg == entry["pan"]
        assert frame.tilt_deg == entry["tilt"]
        assert frame.image.shape[0] > 0


def test_bottle_is_drawn_at_the_geometry_pixel():
    settings = load_settings(load_env=False).model_copy(
        update={"fx": None, "fy": None, "cx": None, "cy": None, "dist_coeffs": None}
    )
    image = Image.open(DESK / "pan020_tilt000.png").convert("RGB")
    width, height = image.size
    u, v, z = direction_to_pixel(18.0, -8.0, 20.0, 0.0, width, height, settings)
    assert z > 0
    pixel = image.getpixel((int(round(u)), int(round(v))))
    assert pixel[2] > pixel[0]
    assert pixel[2] > 150

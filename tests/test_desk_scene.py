import json
from pathlib import Path

DESK = Path("data/test_scenes/desk")


def test_desk_manifest_does_not_assume_a_scan():
    """Angles come from filenames the user adds. The manifest starts empty."""
    manifest = json.loads((DESK / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"] == []


def test_desk_directory_has_no_saved_scene_images():
    images = [
        path
        for path in DESK.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    assert images == []

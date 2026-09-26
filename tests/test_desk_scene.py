import json
from pathlib import Path

DESK = Path("data/test_scenes/desk")


def test_desk_manifest_records_angles_without_opening_photos():
    """Angle metadata is JSON. Tests do not read saved scene photos."""
    manifest = json.loads((DESK / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"]
    assert frames
    names = []
    for entry in frames:
        assert entry["file"]
        assert isinstance(entry["pan"], (int, float))
        assert isinstance(entry["tilt"], (int, float))
        names.append(entry["file"])
    assert len(names) == len(set(names))


def test_desk_directory_has_no_saved_scene_images():
    images = [
        path
        for path in DESK.rglob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ]
    assert images == []

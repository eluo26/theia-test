import json
from pathlib import Path

import pytest
from PIL import Image

from vision.config import load_settings
from vision.detector import DetectorHit, NullDetector, ScriptedDetector
from vision.index import build_catalog, catalog_for_query, merge_detections, write_catalog
from vision.schemas import Catalog, Detection, IndexResponse


def _api_key() -> str:
    return "sk-" + "test" + "notalivekeyvalue123456"


def _settings(tmp_path: Path, **updates):
    settings = load_settings(load_env=False)
    payload = {
        "cache_dir": str(tmp_path / "cache"),
        "out_dir": str(tmp_path / "out"),
        "tile_grid": (1, 1),
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "dist_coeffs": None,
    }
    payload.update(updates)
    return settings.model_copy(update=payload)


def _detection(**overrides) -> Detection:
    payload = {
        "label": "blue water bottle",
        "description": "blue bottle",
        "bbox_px": [90, 40, 110, 60],
        "count": 1,
        "drug_name": None,
        "expiry_text": None,
        "confidence": 0.45,
        "box_source": "grok",
        "frame_file": "center.png",
        "pan_deg": 12.0,
        "tilt_deg": 0.0,
        "azimuth_deg": 12.0,
        "elevation_deg": 0.0,
        "image_width": 200,
        "image_height": 100,
        "timestamp": "2026-09-26T15:40:00",
    }
    payload.update(overrides)
    return Detection.model_validate(payload)


def test_dedupe_writes_a_catalog(tmp_path: Path):
    settings = _settings(tmp_path)
    edge = _detection(
        frame_file="edge.png",
        bbox_px=[0, 0, 10, 10],
        azimuth_deg=10.0,
        elevation_deg=1.0,
        pan_deg=10.0,
        description="near the edge",
    )
    center = _detection(
        frame_file="center.png",
        bbox_px=[90, 40, 110, 60],
        azimuth_deg=12.0,
        elevation_deg=0.0,
        description="near the middle",
    )
    lamp = _detection(
        label="yellow lamp",
        description="yellow lamp",
        frame_file="lamp.png",
        bbox_px=[80, 30, 120, 70],
        azimuth_deg=40.0,
        elevation_deg=5.0,
        pan_deg=40.0,
    )
    far = _detection(
        frame_file="far.png",
        azimuth_deg=80.0,
        elevation_deg=0.0,
        pan_deg=80.0,
        description="same words, other side of the room",
    )
    objects = merge_detections([edge, center, lamp, far], settings, created_at="2026-09-26T16:00:00")
    assert [obj.label for obj in objects] == [
        "blue water bottle",
        "yellow lamp",
        "blue water bottle",
    ]
    bottle = objects[0]
    assert bottle.object_id == "obj_001"
    assert bottle.frame_file == "center.png"
    assert bottle.last_seen == "2026-09-26T15:40:00"

    catalog = Catalog(objects=objects, created_at="2026-09-26T16:00:00", scan_dir=str(tmp_path))
    path = write_catalog(catalog, tmp_path / "out" / "catalog.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["objects"][0]["object_id"] == "obj_001"
    assert payload["objects"][0]["frame_file"] == "center.png"
    assert "range_m" not in payload["objects"][0]


def test_build_catalog_merges_two_views_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    for name, pan in (("pan010_tilt000.png", 10), ("pan012_tilt000.png", 12)):
        Image.new("RGB", (200, 100), (255, 255, 255)).save(tmp_path / name)
    settings = _settings(tmp_path)

    def fake_image(image_bytes, mime, settings, *, prompt, response_model, model, **kwargs):
        assert model == settings.fast_model
        assert response_model is IndexResponse
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "blue water bottle",
                            "description": "blue bottle on the desk",
                            "box": [400, 300, 600, 700],
                            "count": 1,
                            "drug_name": None,
                            "expiry_text": None,
                        }
                    ]
                }
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    catalog = build_catalog(tmp_path, settings=settings, detector=NullDetector(), save_debug=True)
    assert len(catalog.objects) == 1
    assert catalog.objects[0].object_id == "obj_001"
    assert catalog.objects[0].azimuth_deg == pytest.approx(10.0, abs=1e-6)
    assert catalog.objects[0].elevation_deg == pytest.approx(0.0, abs=1e-6)
    written = json.loads((tmp_path / "out" / "catalog.json").read_text(encoding="utf-8"))
    assert written["objects"][0]["label"] == "blue water bottle"
    assert (tmp_path / "out" / "debug" / "pan010_tilt000.png").is_file()


def test_detector_box_is_kept_when_iou_is_high(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    Image.new("RGB", (100, 80), (255, 255, 255)).save(tmp_path / "pan000_tilt000.png")
    settings = _settings(tmp_path)

    def fake_image(*args, **kwargs):
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "green mug",
                            "description": "green mug",
                            "box": [100, 100, 900, 900],
                            "count": 1,
                            "drug_name": None,
                            "expiry_text": None,
                        }
                    ]
                }
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    detector = ScriptedDetector(
        {"green mug": [DetectorHit(bbox_px=[20, 16, 80, 64], score=0.91)]}
    )
    catalog = build_catalog(tmp_path, settings=settings, detector=detector)
    obj = catalog.objects[0]
    assert obj.box_source == "detector"
    assert obj.confidence == 0.91
    assert obj.bbox_px == [20, 16, 80, 64]
    assert detector.calls == ["green mug"]


def test_invalid_tile_is_skipped(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    Image.new("RGB", (32, 32), (255, 255, 255)).save(tmp_path / "pan000_tilt000.png")
    settings = _settings(tmp_path, validation_retries=1)

    class Boom:
        def __init__(self):
            self.calls = 0

        def parse(self, **kwargs):
            self.calls += 1
            message = type("M", (), {"parsed": None, "content": "not-json", "refusal": None})()
            choice = type("C", (), {"message": message})()
            return type("R", (), {"choices": [choice]})()

    boom = Boom()

    def factory(**kwargs):
        chat = type("Chat", (), {})()
        chat.completions = boom
        return type("Client", (), {"chat": chat})()

    import logging

    with caplog.at_level(logging.WARNING):
        catalog = build_catalog(tmp_path, settings=settings, client_factory=factory, detector=NullDetector())
    assert catalog.objects == []
    assert boom.calls == 2
    assert "skipping tile" in caplog.text


def test_downscaled_photo_keeps_the_center_on_pan(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    Image.new("RGB", (1600, 800), (255, 255, 255)).save(tmp_path / "pan020_tilt000.jpg", quality=85)
    settings = _settings(tmp_path, api_max_edge=100)

    def fake_image(*args, **kwargs):
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "blue water bottle",
                            "description": "centered bottle",
                            "box": [400, 300, 600, 700],
                            "count": 1,
                            "drug_name": None,
                            "expiry_text": None,
                        }
                    ]
                }
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    catalog = build_catalog(tmp_path, settings=settings, detector=NullDetector())
    obj = catalog.objects[0]
    assert obj.image_width == 1600
    assert obj.image_height == 800
    assert obj.azimuth_deg == pytest.approx(20.0, abs=1e-6)
    assert obj.elevation_deg == pytest.approx(0.0, abs=1e-6)


def test_second_query_reuses_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    Image.new("RGB", (80, 40), (255, 255, 255)).save(tmp_path / "pan000_tilt000.png")
    settings = _settings(tmp_path)
    calls = {"n": 0}

    def fake_image(*args, **kwargs):
        calls["n"] += 1
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "green mug",
                            "description": "green mug",
                            "box": [100, 100, 400, 800],
                            "count": 1,
                            "drug_name": None,
                            "expiry_text": None,
                        }
                    ]
                }
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    first = catalog_for_query(tmp_path, settings=settings, detector=NullDetector())
    second = catalog_for_query(tmp_path, settings=settings, detector=NullDetector())
    assert calls["n"] == 1
    assert second.objects[0].object_id == first.objects[0].object_id
    assert second.fingerprint == first.fingerprint

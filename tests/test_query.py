import json
from pathlib import Path

from PIL import Image

from vision.config import load_settings
from vision.detector import DetectorHit, NullDetector, ScriptedDetector
from vision.query import catalog_text, locate, public_query_dict
from vision.schemas import Catalog, CatalogObject, QueryDecision, VerifyResponse


def _api_key() -> str:
    return "xai-" + "testkeyvalue123456"


def _object(frame: str) -> CatalogObject:
    return CatalogObject(
        object_id="obj_007",
        label="blue water bottle",
        description="blue bottle next to the lamp",
        count=1,
        drug_name=None,
        expiry_text=None,
        azimuth_deg=30.0,
        elevation_deg=-10.0,
        confidence=0.45,
        frame_file=frame,
        bbox_px=[80, 30, 120, 70],
        box_source="grok",
        last_seen="2026-09-26T15:40:00",
        pan_deg=30.0,
        tilt_deg=-10.0,
        image_width=200,
        image_height=100,
    )


def _settings(tmp_path: Path, **updates):
    settings = load_settings(load_env=False)
    payload = {
        "cache_dir": str(tmp_path / "cache"),
        "out_dir": str(tmp_path / "out"),
        "clinic_mode": False,
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "dist_coeffs": None,
    }
    payload.update(updates)
    return settings.model_copy(update=payload)


def test_locate_returns_the_json_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    frame = "pan030_tilt-10.png"
    Image.new("RGB", (200, 100), (240, 240, 240)).save(tmp_path / frame)
    catalog = Catalog(
        objects=[_object(frame)],
        created_at="2026-09-26T16:00:00",
        scan_dir=str(tmp_path),
    )
    settings = _settings(tmp_path)
    seen = {}

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        seen["model"] = model
        seen["prompt"] = prompt
        assert response_model is QueryDecision
        assert "azimuth" not in prompt
        assert "obj_007" in prompt
        assert "blue water bottle" in prompt
        return (
            QueryDecision(
                status="found",
                object_id="obj_007",
                confidence=0.86,
                reason="Matched the blue bottle next to the lamp",
                candidates=[],
            ),
            "miss",
        )

    def fake_image(image_bytes, mime, settings, *, prompt, response_model, model, **kwargs):
        assert response_model is VerifyResponse
        assert model == settings.grok_fast_model
        assert kwargs.get("client_factory") is None or True
        assert "reasoning_effort" not in kwargs
        return (
            VerifyResponse(is_match=True, box=[115, 115, 885, 885], reason="Yes, tight box."),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    result = locate(
        "where's my blue water bottle?",
        catalog,
        settings=settings,
        detector=NullDetector(),
    )
    payload = public_query_dict(result)
    for key in (
        "status",
        "object_id",
        "label",
        "azimuth_deg",
        "elevation_deg",
        "range_m",
        "confidence",
        "frame_file",
        "bbox_px",
        "candidates",
        "metadata",
        "reason",
    ):
        assert key in payload
    assert payload["range_m"] is None
    assert payload["status"] == "found"
    assert payload["object_id"] == "obj_007"
    assert payload["label"] == "blue water bottle"
    assert payload["azimuth_deg"] == pytest_close(30.0)
    assert payload["elevation_deg"] == pytest_close(-10.0)
    assert payload["metadata"]["count"] == 1
    assert payload["metadata"]["last_seen"] == "2026-09-26T15:40:00"
    assert "inventory" not in payload["metadata"]
    assert seen["model"] == settings.grok_reasoning_model
    assert "drug_name" in catalog_text(catalog)
    json.dumps(payload)


def test_clinic_mode_attaches_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    frame = "pan000_tilt000.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(tmp_path / frame)
    obj = _object(frame).model_copy(
        update={"drug_name": "Jardiance starter pack", "label": "Jardiance box", "pan_deg": 0, "tilt_deg": 0}
    )
    catalog = Catalog(objects=[obj], created_at="2026-09-26T16:00:00", scan_dir=str(tmp_path))
    settings = _settings(tmp_path, clinic_mode=True)

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        return (
            QueryDecision(
                status="found",
                object_id="obj_007",
                confidence=0.8,
                reason="Matched the medication package",
                candidates=[],
            ),
            "miss",
        )

    def fake_image(*args, **kwargs):
        return (VerifyResponse(is_match=True, box=[115, 115, 885, 885], reason="yes"), "miss")

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    result = locate("where's the Jardiance starter pack?", catalog, settings=settings, detector=NullDetector())
    payload = public_query_dict(result)
    assert payload["metadata"]["inventory"]["name"] == "Jardiance"
    assert payload["metadata"]["inventory"]["count"] == 2
    assert payload["metadata"]["inventory"]["resource_links"]
    assert payload["range_m"] is None


def test_tighter_detector_box_on_the_crop(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    frame = "pan000_tilt000.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(tmp_path / frame)
    catalog = Catalog(
        objects=[_object(frame).model_copy(update={"pan_deg": 0.0, "tilt_deg": 0.0, "azimuth_deg": 0, "elevation_deg": 0})],
        created_at="2026-09-26T16:00:00",
        scan_dir=str(tmp_path),
    )
    settings = _settings(tmp_path)

    def fake_text(*args, **kwargs):
        return (
            QueryDecision(status="found", object_id="obj_007", confidence=0.7, reason="found", candidates=[]),
            "miss",
        )

    def fake_image(*args, **kwargs):
        return (VerifyResponse(is_match=True, box=[0, 0, 1000, 1000], reason="wide"), "miss")

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    # Crop of [80, 30, 120, 70] with pad 0.15 is origin (74, 24) and size 52x52.
    # A tight detector box around the original object is smaller than the full crop.
    detector = ScriptedDetector({"blue water bottle": [DetectorHit(bbox_px=[6, 6, 46, 46], score=0.88)]})
    result = locate("where is the bottle?", catalog, settings=settings, detector=detector)
    assert result.box_source if False else result.bbox_px == [80, 30, 120, 70]
    assert result.range_m is None
    assert result.azimuth_deg == pytest_close(0.0)
    assert result.elevation_deg == pytest_close(0.0)


def pytest_close(value, tolerance=1e-6):
    class _Close:
        def __eq__(self, other):
            return other is not None and abs(other - value) <= tolerance

    return _Close()

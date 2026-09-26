import io
import json

import pytest
from PIL import Image

from vision.aim import aim
from vision.config import MissingAPIKeyError, load_settings
from vision.detector import NullDetector
from vision.ingest import IngestError
from vision.schemas import IndexResponse, QueryDecision


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), (255, 255, 255)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _settings(tmp_path):
    settings = load_settings(load_env=False)
    return settings.model_copy(
        update={
            "cache_dir": str(tmp_path / "cache"),
            "out_dir": str(tmp_path / "out"),
            "tile_grid": (1, 1),
            "fx": None,
            "fy": None,
            "cx": None,
            "cy": None,
            "dist_coeffs": None,
            "verify_match": False,
        }
    )


def _index():
    return (
        IndexResponse.model_validate(
            {
                "objects": [
                    {
                        "label": "lamp",
                        "description": "yellow lamp",
                        "box": [450, 450, 550, 550],
                        "count": 1,
                        "drug_name": None,
                        "expiry_text": None,
                    }
                ]
            }
        ),
        "miss",
    )


def test_missing_key_points_at_env_and_does_not_call_the_api(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("vision.aim.load_settings", lambda: _settings(tmp_path))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("the API must not be called without a key")

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fail_if_called)
    with pytest.raises(MissingAPIKeyError, match=r"\.env") as caught:
        aim([{"image": _jpeg(), "pan": 0, "tilt": 0}], "where is the lamp?")
    assert "XAI_API_KEY" in str(caught.value)


def test_aim_returns_the_short_laser_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-" + "test" + "notalivekeyvalue123456")
    monkeypatch.setattr("vision.grok_client.parse_image_bytes", lambda *args, **kwargs: _index())
    monkeypatch.setattr(
        "vision.grok_client.parse_text",
        lambda *args, **kwargs: (
            QueryDecision(
                status="not_found",
                object_id=None,
                confidence=0,
                reason="unused",
                candidates=[],
            ),
            "miss",
        ),
    )
    payload = aim(
        [{"image": _jpeg(), "pan": 30, "tilt": -5, "timestamp": "2026-09-26T12:00:00"}],
        "where is the lamp?",
        settings=_settings(tmp_path),
        detector=NullDetector(),
    )
    assert set(payload) == {"fire_laser", "aim", "items"}
    assert payload["fire_laser"] is True
    assert payload["aim"]["azimuth_deg"] == pytest.approx(30, abs=1)
    assert payload["aim"]["elevation_deg"] == pytest.approx(-5, abs=1)
    assert set(payload["items"][0]) == {
        "id",
        "label",
        "azimuth_deg",
        "elevation_deg",
        "confidence",
    }
    assert payload["items"][0]["label"] == "lamp"
    encoded = json.dumps(payload)
    assert "bbox" not in encoded
    assert "description" not in encoded
    assert "image_width" not in encoded
    assert "notalivekeyvalue" not in encoded

    missed = aim(
        [{"image": _jpeg(), "pan": 0, "tilt": 0}],
        "where is the open laptop?",
        settings=_settings(tmp_path),
        detector=NullDetector(),
    )
    assert missed["fire_laser"] is False
    assert missed["aim"] is None
    assert missed["items"][0]["label"] == "lamp"


def test_bad_frames_fail_before_the_api():
    with pytest.raises(IngestError, match="JPEG"):
        aim([{"image": b"not-a-jpeg", "pan": 0, "tilt": 0}], "lamp")
    with pytest.raises(IngestError, match="pan"):
        aim([{"image": _jpeg(), "tilt": 0}], "lamp")

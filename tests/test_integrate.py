import json
from pathlib import Path

from PIL import Image

from vision.config import load_settings
from vision.detector import NullDetector
from vision.integrate import answer
from vision.schemas import IndexResponse, QueryDecision


def _api_key() -> str:
    return "sk-" + "test" + "notalivekeyvalue123456"


def _settings(tmp_path: Path):
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


def _index_response(objects):
    return (
        IndexResponse.model_validate({"objects": objects}),
        "miss",
    )


def _obj(label, box):
    return {
        "label": label,
        "description": label,
        "box": box,
        "count": 1,
        "drug_name": None,
        "expiry_text": None,
    }


def test_answer_fires_only_when_the_query_is_found(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", _api_key())
    Image.new("RGB", (200, 100), (255, 255, 255)).save(tmp_path / "pan000_tilt000.png")
    settings = _settings(tmp_path)
    decisions = []

    def fake_image(image_bytes, mime, settings, *, prompt, response_model, model, **kwargs):
        return _index_response(
            [
                _obj("laptop", [100, 200, 400, 800]),
                _obj("water bottle", [600, 200, 900, 800]),
            ]
        )

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        decisions.append(prompt)
        return (
            QueryDecision(
                status="ambiguous",
                object_id=None,
                confidence=0.4,
                reason="Two different objects",
                candidates=[
                    {
                        "object_id": "obj_001",
                        "label": "laptop",
                        "confidence": 0.9,
                        "reason": "laptop",
                    },
                    {
                        "object_id": "obj_002",
                        "label": "water bottle",
                        "confidence": 0.8,
                        "reason": "bottle",
                    },
                ],
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)

    found = answer(
        tmp_path,
        "where is the laptop?",
        settings=settings,
        detector=NullDetector(),
    )
    assert found["fire_laser"] is True
    assert set(found) == {"fire_laser", "aim", "result", "items"}
    assert found["aim"]["azimuth_deg"] == found["result"]["azimuth_deg"]
    assert found["aim"]["elevation_deg"] == found["result"]["elevation_deg"]
    assert found["result"]["status"] == "found"
    assert found["result"]["range_m"] is None
    assert found["items"]
    assert found["items"][0]["id"] == found["result"]["object_id"]
    for key in ("label", "description", "azimuth_deg", "elevation_deg", "bbox_px", "frame_file"):
        assert key in found["items"][0]
    json.dumps(found)

    conflict = answer(
        tmp_path,
        "which one?",
        settings=settings,
        detector=NullDetector(),
    )
    assert conflict["fire_laser"] is False
    assert conflict["aim"] is None
    assert conflict["result"]["status"] == "ambiguous"
    assert len(conflict["items"]) == len(found["items"])

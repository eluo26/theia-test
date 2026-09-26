import json
from pathlib import Path

import pytest
from PIL import Image

from vision.cli import main
from vision.detector import NullDetector
from vision.schemas import IndexResponse, QueryDecision, VerifyResponse


def _api_key() -> str:
    # Split so a contiguous xai- token is not staged. The pre-commit hook blocks those.
    return "xai-" + "testkeyvalue123456"


def test_check_config_hides_a_present_key(capsys, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    assert main(["check-config"]) == 0
    out = capsys.readouterr().out
    assert "grok_fast_model:" in out
    assert "XAI_API_KEY: present" in out
    assert _api_key() not in out
    assert "testkeyvalue" not in out


def test_check_config_reports_a_missing_key(capsys, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert main(["check-config"]) == 0
    assert "XAI_API_KEY: missing" in capsys.readouterr().out


def test_describe_requires_a_key(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    image = tmp_path / "scene.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(image)
    assert main(["describe", str(image), "--out-dir", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert "XAI_API_KEY" in err
    assert not list((tmp_path / "out").glob("*"))


def test_describe_writes_json_and_an_annotated_image(tmp_path, monkeypatch, capsys):
    image = tmp_path / "scene.png"
    Image.new("RGB", (80, 60), (255, 255, 255)).save(image)

    def fake_describe(path, settings, use_cache=True):
        assert Path(path) == image
        assert use_cache is True
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "lamp",
                            "description": "yellow lamp",
                            "box": [10, 10, 40, 40],
                            "count": 1,
                            "drug_name": None,
                            "expiry_text": None,
                        }
                    ]
                }
            ),
            "miss",
        )

    monkeypatch.setattr("vision.cli.describe_image", fake_describe)
    out_dir = tmp_path / "out"
    assert main(["describe", str(image), "--out-dir", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert '"label": "lamp"' in captured.out
    assert (out_dir / "scene.json").is_file()
    assert (out_dir / "scene_annotated.png").is_file()
    assert "wrote" in captured.err


def test_index_and_ask_fail_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert main(["index", str(tmp_path)]) == 1
    assert "XAI_API_KEY" in capsys.readouterr().err
    assert main(["ask", str(tmp_path), "where is the bottle?"]) == 1
    err = capsys.readouterr().err
    assert "XAI_API_KEY" in err
    assert "console.x.ai" in err


def test_index_and_ask_succeed_when_grok_is_mocked(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    Image.new("RGB", (200, 100), (255, 255, 255)).save(tmp_path / "pan030_tilt-10.png")
    settings = __import__("vision.config", fromlist=["load_settings"]).load_settings(load_env=False)
    settings = settings.model_copy(
        update={
            "cache_dir": str(tmp_path / "cache"),
            "out_dir": str(tmp_path / "out"),
            "tile_grid": (1, 1),
            "fx": None,
            "fy": None,
            "cx": None,
            "cy": None,
            "dist_coeffs": None,
        }
    )
    monkeypatch.setattr("vision.cli.load_settings", lambda *args, **kwargs: settings)

    def fake_image(image_bytes, mime, settings, *, prompt, response_model, model, **kwargs):
        if response_model is VerifyResponse:
            return (VerifyResponse(is_match=True, box=[400, 300, 600, 700], reason="yes"), "miss")
        return (
            IndexResponse.model_validate(
                {
                    "objects": [
                        {
                            "label": "blue water bottle",
                            "description": "blue bottle",
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

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        assert model == settings.grok_reasoning_model
        assert response_model is QueryDecision
        return (
            QueryDecision(
                status="found",
                object_id="obj_001",
                confidence=0.86,
                reason="Matched the blue bottle",
                candidates=[],
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    monkeypatch.setattr("vision.detector.build_detector", lambda _settings: NullDetector())

    assert main(["index", str(tmp_path)]) == 0
    indexed = capsys.readouterr()
    assert "blue water bottle" in indexed.out
    assert _api_key() not in indexed.out + indexed.err
    assert (tmp_path / "out" / "catalog.json").is_file()

    assert main(["ask", str(tmp_path), "where's my blue water bottle?"]) == 0
    asked = capsys.readouterr()
    payload = json.loads(asked.out)
    assert payload["status"] == "found"
    assert payload["range_m"] is None
    assert payload["object_id"] == "obj_001"
    assert _api_key() not in asked.out + asked.err

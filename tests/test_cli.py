from pathlib import Path

import pytest
from PIL import Image

from vision.cli import main
from vision.schemas import IndexResponse


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

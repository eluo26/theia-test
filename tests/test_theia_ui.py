import importlib.util
from pathlib import Path

import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


speak = _load("theia_ui_speak", "theia-ui/speak.py")
server = _load("theia_ui_server", "theia-ui/server.py")


def test_found_answer_is_a_sentence():
    text = speak.speak(
        {
            "fire_laser": True,
            "aim": {"azimuth_deg": 95.2, "elevation_deg": 3.4},
            "result": {"status": "found", "label": "blue water bottle", "reason": "matched"},
        }
    )
    assert "blue water bottle" in text
    assert "95 degrees to the right" in text
    assert "3 degrees up" in text
    assert "laser would point there" in text
    assert "{" not in text


def test_missing_object_stays_off():
    text = speak.speak(
        {
            "fire_laser": False,
            "aim": None,
            "result": {
                "status": "not_found",
                "label": None,
                "reason": "No open laptop is listed in the catalog.",
            },
        }
    )
    assert text.startswith("I don't see that")
    assert "laser stays off" in text
    assert "open laptop" in text


def test_inventory_lists_each_label_once():
    text = speak.inventory_sentence(
        {
            "items": [
                {"label": "desk lamp"},
                {"label": "Desk lamp"},
                {"label": "blue water bottle"},
            ]
        }
    )
    assert text == "Also in view: desk lamp, blue water bottle."


def test_camera_address_must_be_http():
    assert server.normalize_base("192.168.1.20:4747") == "http://192.168.1.20:4747"
    with pytest.raises(server.UiError):
        server.normalize_base("file:///C:/photos")

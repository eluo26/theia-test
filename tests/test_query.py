import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from vision.grok_client import parse_text

from vision.config import load_settings
from vision.detector import DetectorHit, NullDetector, ScriptedDetector
from vision.matching import same_label_family, token_ratio
from vision.query import catalog_text, locate, public_query_dict
from vision.schemas import Candidate, Catalog, CatalogObject, QueryDecision, VerifyResponse


def _api_key() -> str:
    return "sk-" + "test" + "notalivekeyvalue123456"


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
        assert model == settings.fast_model
        assert "reasoning_effort" not in kwargs
        return (
            VerifyResponse(is_match=True, box=[115, 115, 885, 885], reason="Yes, tight box."),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    monkeypatch.setattr("vision.grok_client.parse_image_bytes", fake_image)
    result = locate(
        "what is next to the lamp?",
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
    assert seen["model"] == settings.active_query_model
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
    settings = _settings(tmp_path, verify_match=True)

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
    assert result.bbox_px == [80, 30, 120, 70]
    assert result.range_m is None
    assert result.azimuth_deg == pytest_close(0.0)
    assert result.elevation_deg == pytest_close(0.0)


def test_ambiguous_same_label_aims_at_the_higher_confidence_view(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    left = _object("pan000_tilt000.png").model_copy(
        update={
            "object_id": "obj_001",
            "label": "laptop",
            "description": "open laptop",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "pan_deg": 0.0,
            "tilt_deg": 0.0,
            "confidence": 0.45,
        }
    )
    right = _object("pan030_tilt000.png").model_copy(
        update={
            "object_id": "obj_002",
            "label": "laptop",
            "description": "the same laptop from another photo",
            "azimuth_deg": 30.0,
            "elevation_deg": 1.0,
            "pan_deg": 30.0,
            "tilt_deg": 0.0,
            "confidence": 0.45,
        }
    )
    catalog = Catalog(
        objects=[left, right],
        created_at="2026-09-26T16:00:00",
        scan_dir=str(tmp_path),
    )
    settings = _settings(tmp_path)

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        return (
            QueryDecision(
                status="ambiguous",
                object_id=None,
                confidence=0.5,
                reason="Two views of a laptop",
                candidates=[
                    Candidate(object_id="obj_001", label="laptop", confidence=0.40, reason="left view"),
                    Candidate(object_id="obj_002", label="laptop", confidence=0.92, reason="clearer view"),
                ],
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    result = locate("which view should we use?", catalog, settings=settings, detector=NullDetector())
    assert result.status == "found"
    assert result.object_id == "obj_002"
    assert result.label == "laptop"
    assert result.azimuth_deg == 30.0
    assert result.elevation_deg == 1.0
    assert result.confidence == 0.92


def test_ambiguous_different_labels_stay_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    laptop = _object("pan000_tilt000.png").model_copy(
        update={
            "object_id": "obj_001",
            "label": "laptop",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "pan_deg": 0.0,
            "tilt_deg": 0.0,
        }
    )
    bottle = _object("pan030_tilt000.png").model_copy(
        update={
            "object_id": "obj_002",
            "label": "water bottle",
            "description": "clear bottle",
            "azimuth_deg": 30.0,
            "elevation_deg": -2.0,
            "pan_deg": 30.0,
            "tilt_deg": 0.0,
        }
    )
    catalog = Catalog(
        objects=[laptop, bottle],
        created_at="2026-09-26T16:00:00",
        scan_dir=str(tmp_path),
    )
    settings = _settings(tmp_path)
    assert token_ratio("laptop", "water bottle") < settings.label_sim

    def fake_text(prompt, settings, *, response_model, model, **kwargs):
        return (
            QueryDecision(
                status="ambiguous",
                object_id=None,
                confidence=0.5,
                reason="Two different objects",
                candidates=[
                    Candidate(object_id="obj_001", label="laptop", confidence=0.99, reason="laptop"),
                    Candidate(object_id="obj_002", label="water bottle", confidence=0.80, reason="bottle"),
                ],
            ),
            "miss",
        )

    monkeypatch.setattr("vision.grok_client.parse_text", fake_text)
    result = locate("where is it?", catalog, settings=settings, detector=NullDetector())
    assert result.status == "ambiguous"


def test_text_query_does_not_send_reasoning_effort(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    calls = []

    class Completions:
        def parse(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(
                parsed=QueryDecision(
                    status="not_found",
                    object_id=None,
                    confidence=0.0,
                    reason="none",
                    candidates=[],
                ),
                content=None,
                refusal=None,
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def factory(**kwargs):
        chat = SimpleNamespace(completions=Completions())
        return SimpleNamespace(chat=chat)

    parse_text(
        "where is it?",
        settings,
        response_model=QueryDecision,
        model=settings.reasoning_model,
        client_factory=factory,
        use_cache=False,
    )
    assert set(calls[0]) == {"model", "messages", "response_format"}
    assert calls[0]["model"] == settings.reasoning_model
    assert calls[0]["response_format"] is QueryDecision
    assert isinstance(calls[0]["messages"][0]["content"], str)


def _counting_client(parsed):
    calls = {"n": 0}

    class Completions:
        def parse(self, **kwargs):
            calls["n"] += 1
            message = SimpleNamespace(parsed=parsed, content=None, refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def factory(**kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    return factory, calls


def test_direct_label_skips_the_text_model_and_aims_at_the_best_view(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    frame = "pan090_tilt000.png"
    Image.new("RGB", (200, 100), (240, 240, 240)).save(tmp_path / frame)
    left = _object(frame).model_copy(
        update={
            "object_id": "obj_037",
            "label": "left elevator doors",
            "description": "elevator doors from the left side of the frame",
            "azimuth_deg": 91.2,
            "elevation_deg": 10.3,
            "confidence": 0.40,
            "pan_deg": 90.0,
            "tilt_deg": 0.0,
        }
    )
    right = _object(frame).model_copy(
        update={
            "object_id": "obj_038",
            "label": "right elevator doors",
            "description": "elevator doors from the right side of the frame",
            "azimuth_deg": 109.4,
            "elevation_deg": 9.6,
            "confidence": 0.92,
            "pan_deg": 90.0,
            "tilt_deg": 0.0,
        }
    )
    catalog = Catalog(objects=[left, right], created_at="2026-09-26T16:00:00", scan_dir=str(tmp_path))
    settings = _settings(tmp_path, verify_match=True)
    assert token_ratio("left elevator doors", "right elevator doors") < settings.label_sim
    assert same_label_family("left elevator doors", "right elevator doors", settings.label_sim)
    factory, calls = _counting_client(
        QueryDecision(status="not_found", object_id=None, confidence=0.0, reason="should not run", candidates=[])
    )
    result = locate(
        "where are the elevator doors?",
        catalog,
        settings=settings,
        client_factory=factory,
        detector=NullDetector(),
        use_cache=False,
    )
    assert calls["n"] == 0
    assert result.status == "found"
    assert result.object_id == "obj_038"
    assert result.label == "right elevator doors"
    assert result.confidence == 0.92
    assert result.azimuth_deg == 109.4
    payload = public_query_dict(result)
    assert payload["status"] == "found"


def test_relational_question_still_calls_the_text_model(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    lamp = _object("pan000_tilt000.png").model_copy(
        update={"object_id": "obj_001", "label": "lamp", "confidence": 0.4, "azimuth_deg": 0.0}
    )
    mug = _object("pan030_tilt000.png").model_copy(
        update={"object_id": "obj_002", "label": "mug", "confidence": 0.95, "azimuth_deg": 30.0}
    )
    catalog = Catalog(objects=[lamp, mug], created_at="2026-09-26T16:00:00", scan_dir=str(tmp_path))
    settings = _settings(tmp_path, verify_match=False)
    factory, calls = _counting_client(
        QueryDecision(
            status="found",
            object_id="obj_001",
            confidence=0.77,
            reason="The lamp is left of the mug",
            candidates=[],
        )
    )
    result = locate(
        "what is left of the mug?",
        catalog,
        settings=settings,
        client_factory=factory,
        detector=NullDetector(),
        use_cache=False,
    )
    assert calls["n"] == 1
    assert result.status == "found"
    assert result.object_id == "obj_001"
    assert result.label == "lamp"


def test_two_different_objects_stay_ambiguous_without_a_text_call(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    cola = _object("pan000_tilt000.png").model_copy(
        update={
            "object_id": "obj_016",
            "label": "red Coca-Cola can",
            "confidence": 0.95,
            "azimuth_deg": -11.6,
        }
    )
    other = _object("pan000_tilt000.png").model_copy(
        update={
            "object_id": "obj_012",
            "label": "red beverage can",
            "confidence": 0.40,
            "azimuth_deg": -22.2,
        }
    )
    catalog = Catalog(objects=[cola, other], created_at="2026-09-26T16:00:00", scan_dir=str(tmp_path))
    settings = _settings(tmp_path, verify_match=True)
    assert token_ratio("red Coca-Cola can", "red beverage can") < settings.label_sim
    assert not same_label_family("red Coca-Cola can", "red beverage can", settings.label_sim)
    factory, calls = _counting_client(
        QueryDecision(status="found", object_id="obj_016", confidence=0.99, reason="should not run", candidates=[])
    )
    result = locate(
        "red can",
        catalog,
        settings=settings,
        client_factory=factory,
        detector=NullDetector(),
        use_cache=False,
    )
    assert calls["n"] == 0
    assert result.status == "ambiguous"


def pytest_close(value, tolerance=1e-6):
    class _Close:
        def __eq__(self, other):
            return other is not None and abs(other - value) <= tolerance

    return _Close()

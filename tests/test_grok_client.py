import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vision.config import load_settings
from vision.grok_client import GrokCallError, describe_image, redact, response_cache_key
from vision.schemas import IndexResponse


def _settings(tmp_path: Path):
    settings = load_settings(load_env=False)
    return settings.model_copy(
        update={"cache_dir": str(tmp_path / "cache"), "out_dir": str(tmp_path / "out")}
    )


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "scene.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    return path


def _completion(parsed=None, content=None):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client_factory(completions):
    def factory(**kwargs):
        chat = SimpleNamespace(completions=completions)
        return SimpleNamespace(chat=chat, kwargs=kwargs)

    return factory


def _api_key() -> str:
    # Split so a contiguous sk- token is not staged. The pre-commit hook blocks those.
    return "sk-" + "test" + "notalivekeyvalue123456"


def _ok_response():
    return IndexResponse.model_validate(
        {
            "objects": [
                {
                    "label": "blue bottle",
                    "description": "blue cylinder on the left",
                    "box": [70, 130, 190, 470],
                    "count": 1,
                    "drug_name": None,
                    "expiry_text": None,
                }
            ]
        }
    )


def test_describe_uses_the_model_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    completions = _FakeCompletions([_completion(_ok_response())])
    result, cache_state = describe_image(
        _image(tmp_path),
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    assert cache_state == "miss"
    assert result.objects[0].label == "blue bottle"
    assert completions.calls[0]["model"] == settings.fast_model
    assert completions.calls[0]["model"] == settings.grok_fast_model
    assert completions.calls[0]["response_format"] is IndexResponse
    sent = completions.calls[0]["messages"][0]["content"][0]["image_url"]["detail"]
    assert sent == settings.image_detail


def test_second_call_is_a_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    completions = _FakeCompletions([_completion(_ok_response())])
    image = _image(tmp_path)
    factory = _client_factory(completions)
    describe_image(image, settings, client_factory=factory, sleeper=lambda _delay: None)
    result, cache_state = describe_image(
        image, settings, client_factory=factory, sleeper=lambda _delay: None
    )
    assert cache_state == "hit"
    assert len(completions.calls) == 1
    assert result.objects[0].label == "blue bottle"


def test_invalid_json_is_retried_once(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path).model_copy(update={"validation_retries": 1})
    completions = _FakeCompletions(
        [
            _completion(parsed=None, content="{"),
            _completion(_ok_response()),
        ]
    )
    result, cache_state = describe_image(
        _image(tmp_path),
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    assert cache_state == "miss"
    assert len(completions.calls) == 2
    assert result.objects[0].label == "blue bottle"


def test_repeated_invalid_json_skips_the_image(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path).model_copy(update={"validation_retries": 1})
    completions = _FakeCompletions(
        [
            _completion(parsed=None, content="not-json"),
            _completion(parsed=None, content="still-not-json"),
        ]
    )
    with pytest.raises(GrokCallError, match="schema validation"):
        describe_image(
            _image(tmp_path),
            settings,
            client_factory=_client_factory(completions),
            sleeper=lambda _delay: None,
        )


def test_http_400_is_not_retried_and_points_at_the_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    error = RuntimeError("bad request involving " + _api_key())
    error.status_code = 400
    completions = _FakeCompletions([error, _completion(_ok_response())])
    sleeps = []
    with pytest.raises(GrokCallError, match="docs.x.ai") as caught:
        describe_image(
            _image(tmp_path),
            settings,
            client_factory=_client_factory(completions),
            sleeper=sleeps.append,
        )
    assert len(completions.calls) == 1
    assert sleeps == []
    assert _api_key() not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_rate_limit_uses_backoff_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path).model_copy(update={"retry_base_delay_s": 0.5})
    limited = RuntimeError("slow down")
    limited.status_code = 429
    completions = _FakeCompletions([limited, _completion(_ok_response())])
    sleeps = []
    result, _cache = describe_image(
        _image(tmp_path),
        settings,
        client_factory=_client_factory(completions),
        sleeper=sleeps.append,
    )
    assert sleeps == [0.5]
    assert result.objects[0].label == "blue bottle"


def test_unsupported_image_type(tmp_path):
    settings = _settings(tmp_path)
    image = tmp_path / "frame.bmp"
    image.write_bytes(b"bmp")
    with pytest.raises(GrokCallError, match="Unsupported image type"):
        describe_image(image, settings, client_factory=_client_factory(_FakeCompletions([])))


def _write_heic(path: Path) -> None:
    """Write a tiny HEIC. Skip only when no encoder can produce one."""
    errors: list[str] = []
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        Image.new("RGB", (8, 8), (9, 8, 7)).save(path, format="HEIF")
        if path.is_file() and path.stat().st_size > 0:
            return
        errors.append("pillow-heif encode produced an empty file")
    except Exception as exc:
        errors.append(f"pillow-heif encode failed: {exc}")
    pytest.skip("writing HEIC is impossible: " + "; ".join(errors))


def test_heic_is_sent_as_jpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    path = tmp_path / "IMG_0205.HEIC"
    _write_heic(path)
    completions = _FakeCompletions([_completion(_ok_response())])
    describe_image(
        path,
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    url = completions.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert "image/heic" not in url
    sent = _sent_image_bytes(completions.calls[0])
    assert sent.startswith(b"\xff\xd8")
    assert b"ftyp" not in sent[:16]
    with Image.open(io.BytesIO(sent)) as image:
        assert image.format == "JPEG"


def test_webp_suffix_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    image = tmp_path / "frame.webp"
    image.write_bytes(b"RIFFxxxxWEBP")
    completions = _FakeCompletions([_completion(_ok_response())])
    result, _cache = describe_image(
        image,
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    sent = completions.calls[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert sent.startswith("data:image/webp;base64,")
    assert result.objects[0].label == "blue bottle"


def _sent_image_bytes(call: dict) -> bytes:
    url = call["messages"][0]["content"][0]["image_url"]["url"]
    return base64.b64decode(url.split(",", 1)[1])


def test_upright_jpeg_is_sent_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    path = tmp_path / "pan000_tilt0.jpg"
    Image.new("RGB", (40, 12), (20, 40, 60)).save(path, format="JPEG", quality=90)
    original = path.read_bytes()
    completions = _FakeCompletions([_completion(_ok_response())])
    describe_image(
        path,
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    assert _sent_image_bytes(completions.calls[0]) == original


def test_sideways_phone_jpeg_is_sent_upright(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    path = tmp_path / "pan000_tilt0.jpg"
    image = Image.new("RGB", (40, 12), (200, 10, 10))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif, quality=95)
    completions = _FakeCompletions([_completion(_ok_response())])
    describe_image(
        path,
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    sent = Image.open(io.BytesIO(_sent_image_bytes(completions.calls[0])))
    assert sent.size == (12, 40)
    assert sent.getexif().get(274) in (None, 1)


def test_cache_key_changes_with_model_and_image():
    first = response_cache_key(
        image_bytes=b"a", prompt="p", model="one", image_detail="high"
    )
    second = response_cache_key(
        image_bytes=b"b", prompt="p", model="one", image_detail="high"
    )
    third = response_cache_key(
        image_bytes=b"a", prompt="p", model="two", image_detail="high"
    )
    assert len({first, second, third}) == 3


def test_redact_removes_key_material():
    leaked = "xai-" + "secretvalue12"
    assert "secretvalue" not in redact(f"token {leaked} in the log")
    openai_leaked = "sk-" + ("b" * 24)
    assert openai_leaked not in redact(f"token {openai_leaked} in the log")
    header = "Authorization: Bearer " + openai_leaked
    cleaned = redact(header)
    assert openai_leaked not in cleaned
    assert "Bearer [REDACTED]" in cleaned


def test_client_source_does_not_hardcode_a_model_id():
    source = Path("vision/grok_client.py").read_text(encoding="utf-8")
    config_source = Path("vision/config.py").read_text(encoding="utf-8")
    assert "grok-" not in source
    assert "gpt-" not in source
    assert "grok-" not in config_source
    assert "gpt-" not in config_source


def test_cached_file_is_json(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", _api_key())
    settings = _settings(tmp_path)
    completions = _FakeCompletions([_completion(_ok_response())])
    describe_image(
        _image(tmp_path),
        settings,
        client_factory=_client_factory(completions),
        sleeper=lambda _delay: None,
    )
    files = list((tmp_path / "cache").glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["model"] == settings.fast_model
    assert "xai-" not in files[0].read_text(encoding="utf-8")

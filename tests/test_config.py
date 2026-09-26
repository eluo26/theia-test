from pathlib import Path

import pytest
from pydantic import ValidationError

from vision.config import (
    PLACEHOLDER_API_KEY,
    ConfigError,
    MissingAPIKeyError,
    Settings,
    api_key_is_present,
    load_settings,
    require_api_key,
)


def test_repo_config_loads():
    settings = load_settings(load_env=False)
    assert settings.provider == "openai"
    assert settings.fast_model == settings.openai_fast_model
    assert settings.reasoning_model == settings.openai_reasoning_model
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.api_key_env == "OPENAI_API_KEY"
    assert settings.grok_fast_model
    assert settings.grok_reasoning_model
    assert settings.xai_base_url == "https://api.x.ai/v1"
    assert settings.xai_api_key_env == "XAI_API_KEY"
    assert settings.image_detail in {"auto", "low", "high", "original"}
    assert settings.cache_path.name == "cache"
    assert settings.out_path.name == "out"


def test_xai_provider_uses_grok_settings():
    settings = load_settings(load_env=False).model_copy(update={"provider": "xai"})
    assert settings.fast_model == settings.grok_fast_model
    assert settings.reasoning_model == settings.grok_reasoning_model
    assert settings.base_url == "https://api.x.ai/v1"
    assert settings.api_key_env == "XAI_API_KEY"


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="Missing config file"):
        load_settings(tmp_path / "nope.yaml", load_env=False)


def test_invalid_config_reports_the_field(tmp_path: Path):
    source = Path("config.yaml").read_text(encoding="utf-8")
    broken = source.replace("image_detail: low", "image_detail: ultra")
    path = tmp_path / "config.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError, match="image_detail"):
        load_settings(path, load_env=False)


def test_settings_reject_unknown_keys():
    settings = load_settings(load_env=False)
    payload = settings.model_dump()
    payload["invented_model"] = "not-a-real-model"
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert api_key_is_present() is False
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        require_api_key()


def test_placeholder_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_API_KEY)
    assert api_key_is_present() is False
    with pytest.raises(MissingAPIKeyError):
        require_api_key()


def test_xai_provider_requires_its_own_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-unused")
    settings = load_settings(load_env=False).model_copy(update={"provider": "xai"})
    assert api_key_is_present(settings) is False
    with pytest.raises(MissingAPIKeyError, match="XAI_API_KEY"):
        require_api_key(settings)


def test_dotenv_file_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-the-env-file\n", encoding="utf-8")
    monkeypatch.setattr("vision.config.REPO_ROOT", tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(Path("config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    load_settings(config)
    assert require_api_key() == "from-the-env-file"

"""Load config.yaml and .env. The API key stays out of the settings object."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_API_KEY = "your-key-here"


class ConfigError(RuntimeError):
    """config.yaml is missing or does not match the settings schema."""


class MissingAPIKeyError(ConfigError):
    """XAI_API_KEY is unset or still the placeholder from .env.example."""


class Settings(BaseModel):
    """Values from config.yaml. Model ids are never defaulted in code."""

    model_config = ConfigDict(extra="forbid")

    grok_fast_model: str = Field(min_length=1)
    grok_reasoning_model: str = Field(min_length=1)
    xai_base_url: str = Field(min_length=1)
    image_detail: Literal["auto", "low", "high"]
    request_timeout_s: float = Field(gt=0)
    max_image_bytes: int = Field(gt=0)

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    hfov_deg: float = Field(gt=0, lt=180)
    vfov_deg: float = Field(gt=0, lt=180)
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    dist_coeffs: list[float] | None = None

    tile_grid: tuple[int, int]
    tile_overlap: float = Field(ge=0, lt=1)
    max_concurrency: int = Field(ge=1)
    refine_iou: float = Field(ge=0, le=1)
    label_sim: float = Field(ge=0, le=100)
    merge_deg: float = Field(gt=0)
    video_sample_every_s: float = Field(gt=0)
    blur_threshold: float = Field(ge=0)
    crop_pad: float = Field(ge=0)
    clinic_mode: bool

    detector: Literal["owlv2", "grounding_dino"]
    owlv2_model: str = Field(min_length=1)
    grounding_dino_model: str = Field(min_length=1)

    cache_dir: str = Field(min_length=1)
    out_dir: str = Field(min_length=1)

    max_transport_retries: int = Field(ge=0)
    retry_base_delay_s: float = Field(gt=0)
    validation_retries: int = Field(ge=0)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    @property
    def cache_path(self) -> Path:
        return self.resolve_path(self.cache_dir)

    @property
    def out_path(self) -> Path:
        return self.resolve_path(self.out_dir)


def load_settings(config_path: Path | None = None, *, load_env: bool = True) -> Settings:
    """Load settings from config.yaml. A .env file is read when it exists."""
    if load_env:
        env_path = REPO_ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)

    path = config_path if config_path is not None else REPO_ROOT / "config.yaml"
    if not path.is_file():
        raise ConfigError(f"Missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a YAML mapping of settings.")

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file {path}:\n{exc}") from exc


def api_key_is_present() -> bool:
    """True when XAI_API_KEY is set to something other than the example placeholder."""
    key = os.environ.get("XAI_API_KEY", "").strip()
    return bool(key) and key != PLACEHOLDER_API_KEY


def require_api_key() -> str:
    """Return the xAI key from the environment. Never log the return value."""
    key = os.environ.get("XAI_API_KEY", "").strip()
    if not key or key == PLACEHOLDER_API_KEY:
        raise MissingAPIKeyError(
            "XAI_API_KEY is not set. Copy .env.example to .env in the project root "
            "and paste the key you created at https://console.x.ai. "
            "Confirm .env is gitignored before you commit (run git status)."
        )
    return key

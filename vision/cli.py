"""Command line for the vision module.

Milestone 1 commands:
  python -m vision.cli check-config
  python -m vision.cli describe path/to/image.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vision.config import MissingAPIKeyError, api_key_is_present, load_settings
from vision.grok_client import GrokCallError, describe_image, redact
from vision.viz import annotate_image

logger = logging.getLogger("vision.cli")


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(item) if isinstance(item, str) else item for item in record.args
            )
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RedactFilter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vision.cli",
        description="Object localization for the laser-pointing turret.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-config", help="Print loaded settings. Does not print the API key.")

    describe = sub.add_parser(
        "describe",
        help="Run one Grok vision call on one image and write an annotated debug image.",
    )
    describe.add_argument("image", type=Path, help="A .jpg, .jpeg, or .png file")
    describe.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the JSON and annotated image (default: data/out)",
    )
    describe.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the on-disk response cache and call the API",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except Exception as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1

    if args.command == "check-config":
        return _check_config(settings)
    if args.command == "describe":
        return _describe(settings, args.image, args.out_dir, use_cache=not args.no_cache)
    parser.error(f"Unknown command {args.command}")
    return 2


def _check_config(settings) -> int:
    key_state = "present" if api_key_is_present() else "missing"
    lines = [
        f"grok_fast_model: {settings.grok_fast_model}",
        f"grok_reasoning_model: {settings.grok_reasoning_model}",
        f"xai_base_url: {settings.xai_base_url}",
        f"image_detail: {settings.image_detail}",
        f"request_timeout_s: {settings.request_timeout_s}",
        f"XAI_API_KEY: {key_state}",
        f"cache_dir: {settings.cache_path}",
        f"out_dir: {settings.out_path}",
        f"detector: {settings.detector}",
    ]
    print("\n".join(lines))
    return 0


def _describe(settings, image: Path, out_dir: Path | None, *, use_cache: bool) -> int:
    destination_dir = out_dir if out_dir is not None else settings.out_path
    try:
        result, cache_state = describe_image(image, settings, use_cache=use_cache)
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except GrokCallError as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1

    destination_dir.mkdir(parents=True, exist_ok=True)
    json_path = destination_dir / f"{image.stem}.json"
    image_path = destination_dir / f"{image.stem}_annotated.png"
    payload = result.model_dump()
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    annotate_image(image, result.objects, image_path)

    print(json.dumps(payload, indent=2))
    print(f"cache: {cache_state}", file=sys.stderr)
    print(f"wrote {json_path}", file=sys.stderr)
    print(f"wrote {image_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line for the vision module.

  python -m vision.cli check-config
  python -m vision.cli describe path/to/image.jpg
  python -m vision.cli index SCAN_DIR
  python -m vision.cli index video.mp4 --sweep START_PAN END_PAN TILT
  python -m vision.cli index video.mp4 --angles-csv angles.csv
  python -m vision.cli ask SCAN_DIR "where's my blue water bottle?"
  python -m vision.cli eval desk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vision.config import MissingAPIKeyError, api_key_is_present, load_settings, require_api_key
from vision.eval import EvalError, evaluate_scene
from vision.grok_client import GrokCallError, describe_image, redact
from vision.index import build_catalog
from vision.ingest import IngestError
from vision.query import locate, public_query_dict
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

    index = sub.add_parser(
        "index",
        help="Build a catalog from angle-tagged images or a sweep video.",
    )
    index.add_argument("scan_dir", type=Path, help="Image folder or a video file")
    index.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START_PAN", "END_PAN", "TILT"),
        help="Constant-speed video sweep: start pan, end pan, and a fixed tilt",
    )
    index.add_argument(
        "--angles-csv",
        type=Path,
        help="Video sidecar CSV with columns timestamp_s,pan,tilt",
    )
    index.add_argument("--out-dir", type=Path, default=None)
    index.add_argument("--no-cache", action="store_true")

    ask = sub.add_parser("ask", help="Index a scan and print a QueryResult.")
    ask.add_argument("scan_dir", type=Path, help="Image folder or a video file")
    ask.add_argument("query", type=str, help='Question, for example "where is the blue water bottle?"')
    ask.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START_PAN", "END_PAN", "TILT"),
    )
    ask.add_argument("--angles-csv", type=Path)
    ask.add_argument("--out-dir", type=Path, default=None)
    ask.add_argument("--no-cache", action="store_true")

    evaluate = sub.add_parser(
        "eval",
        help="Score a scene's ground_truth.json. Pass a directory or a name under data/test_scenes.",
    )
    evaluate.add_argument("scene", type=str)
    evaluate.add_argument("--out-dir", type=Path, default=None)
    evaluate.add_argument("--no-cache", action="store_true")
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

    out_dir = getattr(args, "out_dir", None)
    if out_dir is not None:
        settings = settings.model_copy(update={"out_dir": str(out_dir)})

    try:
        if args.command == "check-config":
            return _check_config(settings)
        if args.command == "describe":
            return _describe(settings, args.image, args.out_dir, use_cache=not args.no_cache)
        if args.command == "index":
            return _index(settings, args)
        if args.command == "ask":
            return _ask(settings, args)
        if args.command == "eval":
            return _eval_scene(settings, args)
    except MissingAPIKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (GrokCallError, IngestError, EvalError) as exc:
        print(redact(str(exc)), file=sys.stderr)
        return 1
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


def _video_kwargs(args) -> dict:
    sweep = tuple(args.sweep) if getattr(args, "sweep", None) else None
    return {
        "sweep": sweep,
        "angles_csv": getattr(args, "angles_csv", None),
        "use_cache": not args.no_cache,
    }


def _index(settings, args) -> int:
    require_api_key()
    catalog = build_catalog(args.scan_dir, settings=settings, **_video_kwargs(args))
    print(catalog.model_dump_json(indent=2))
    print(f"wrote {settings.out_path / 'catalog.json'}", file=sys.stderr)
    return 0


def _ask(settings, args) -> int:
    require_api_key()
    catalog = build_catalog(args.scan_dir, settings=settings, **_video_kwargs(args))
    result = locate(args.query, catalog, settings=settings, use_cache=not args.no_cache)
    print(json.dumps(public_query_dict(result), indent=2))
    return 0


def _eval_scene(settings, args) -> int:
    require_api_key()
    report = evaluate_scene(args.scene, settings=settings, use_cache=not args.no_cache)
    print(report.model_dump_json(indent=2))
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

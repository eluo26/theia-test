# What this module can do

Operator setup is [README.md](README.md). This file is the full behavior of the vision package. [PLAN.md](PLAN.md) is the original milestone spec. There is no web framework. Teammates call `build_catalog` and `locate`, or `python -m vision.cli`.

Two uses share one pipeline: finding a belonging at home, and a clinic sample closet (count, expiry, reorder metadata). The clinic path is off until `clinic_mode` is true.

## Library

```python
from vision import build_catalog, locate

catalog = build_catalog("data/test_scenes/<scene>")
result = locate("where is the blue water bottle?", catalog)
```

`build_catalog` writes `data/out/catalog.json`. `locate` returns the turret JSON. `range_m` is always null because the camera is monocular.

```json
{
  "status": "found",
  "object_id": "obj_007",
  "label": "blue water bottle",
  "azimuth_deg": 34.2,
  "elevation_deg": -12.5,
  "range_m": null,
  "confidence": 0.86,
  "frame_file": "pan30_tilt0.jpg",
  "bbox_px": [812, 440, 960, 720],
  "candidates": [],
  "metadata": {
    "count": 1,
    "drug_name": null,
    "expiry_text": null,
    "last_seen": "2026-09-26T15:40:00"
  },
  "reason": "Matched the blue bottle next to the lamp"
}
```

`status` is `found`, `ambiguous`, or `not_found`. `candidates` holds up to three `{object_id, label, confidence, reason}` entries. `metadata` holds `count`, `drug_name`, `expiry_text`, and `last_seen`. With clinic mode on, `metadata.inventory` is added when a drug name matches `data/inventory.json`.

Azimuth 0 is pan home. Positive azimuth is right, clockwise from above. Elevation 0 is horizontal. Positive elevation is up. Degrees.

## Commands

Run from the repo root with the venv active. On Windows, call `.\.venv\Scripts\python.exe` directly.

| Command | What it does |
| --- | --- |
| `check-config` | Prints provider, models, and whether the key is present. Never prints the key. |
| `describe path\to\photo.jpg` | One vision call. Writes JSON and an annotated PNG under `data/out/`. |
| `index SCAN_DIR` | Builds `data/out/catalog.json` from a folder of stills. |
| `ask SCAN_DIR "question"` | Reuses the catalog when the photos are unchanged, otherwise indexes, then prints one JSON result. |
| `eval SCENE` | Scores `ground_truth.json`. Exits when `queries` is empty, before any API call. |

`SCENE` may be a directory or a name under `data/test_scenes/`.

Shared flags:

- `--out-dir DIR` redirects `catalog.json` and debug images.
- `--no-cache` ignores `data/cache/` and, for `ask`, rebuilds the catalog.
- `--debug` on `index` and `ask` writes annotated images under `data/out/debug/`. The same switch is `save_debug` in `config.yaml`. It is off by default.

Video, passed to `index` or `ask`:

```bash
python -m vision.cli index sweep.mp4 --sweep -40 40 0
python -m vision.cli index sweep.mp4 --angles-csv angles.csv
```

`--sweep START_PAN END_PAN TILT` treats the clip as a constant-speed pan at a fixed tilt. The CSV header is `timestamp_s,pan,tilt`. Frames are taken every `video_sample_every_s` (0.5 s). Frames whose Laplacian variance is below `blur_threshold` (100) are dropped. The sharpest frame in each 1° pan/tilt bucket is kept.

Accepted stills are PNG, JPEG, WEBP, and non-animated GIF. HEIC is ignored. A sideways phone JPEG is rotated from its EXIF tag before the model sees it, so boxes match the upright picture.

## Photos and angles

`pan30_tilt-10.jpg` means pan 30°, tilt −10°. A `manifest.json` entry `{file, pan, tilt, timestamp}` overrides the filename when that file is present. The desk manifest ships with `"frames": []`.

`ground_truth.json` is only for `eval`. Each row is `{query, expected_label, az, el}`. The committed file has `"queries": []`. Fill it after you measure a real scene. `describe`, `index`, and `ask` do not read object names from either file.

The repo does not ship sample photos. Do not generate stand-ins. Shoot the phone main lens, not ultrawide, as JPEG. Objects at about 2–3 m and a photo about every 20° of pan are enough for a first scene.

## Pipeline

1. Load stills or sample a video. The working image sent to the model is capped at `api_max_edge` (768 px on the long side). Angles use the original photo size.
2. Undistort only when `fx`, `fy`, `cx`, and `cy` are all set and `dist_coeffs` is set. Those intrinsics replace the field-of-view model.
3. Split the frame with `tile_grid` `[columns, rows]` and `tile_overlap`. The current config is `[1, 1]`, so one call per photo. `[2, 2]` is available when small label text matters. Tile boxes are mapped back to full-frame pixels.
4. The fast model returns objects as structured JSON: label, description, a 0–1000 box, count, drug name, and expiry text. Calls run at most `max_concurrency` at once, with exponential backoff. At most `max_objects_per_image` objects are kept per image (12). Invalid JSON is retried once, then that tile is skipped.
5. If the optional local detector is installed, it searches the tile for that label. When intersection-over-union is at least `refine_iou` (0.3), the detector box is kept (`box_source` `detector`). Otherwise the vision box is kept at lower confidence. A missing detector does not drop the vision box.
6. The box center becomes a camera ray, rotated by tilt about x and then pan about vertical. This is the full rotation, not a small-angle shortcut. Image center returns `(pan, tilt)`. The right edge at tilt 0 returns `pan + hfov/2`.
7. Detections merge when the label token ratio is at least `label_sim` (85) and the angular separation is at most `merge_deg` (8°). The representative is the detection closest to its frame center. Ids are `obj_001`, `obj_002`, ... in a stable order.
8. The question plus catalog text (id, label, description, count, drug name) goes to `query_model`. `fast` uses the fast model. `reasoning` uses the flagship model. It returns status, object id, up to three candidates, confidence, and a reason.
9. When `verify_match` is true, the chosen object is cropped with `crop_pad`, the fast model returns a tight box, and the detector runs on the crop. The tighter agreeing box is kept and the angles are recomputed. This is off by default so a question does not wait on a second image call.
10. With `clinic_mode: true`, `drug_name` is fuzzy-matched against `data/inventory.json`. That file is a mock list. It is not a live pharmacy system.

`ask` reuses `catalog.json` when the photo bytes and the indexing settings are unchanged. `--debug` and `--no-cache` force a rebuild. Response cache files live in `data/cache/`. The cache key is the SHA-256 of the image bytes, prompt, model, and image detail.

Debug images use green for the vision model and blue for the detector. The green legend is the active provider name (`OpenAI` or `xAI`). `describe` always writes an annotated image. `index` and `ask` write them only with `--debug` or `save_debug: true`.

Image-conditioned search from `data/references/` is implemented as `vision.references.image_guided_boxes`. `index` and `ask` do not call it. It needs the detector weights.

## Provider

`provider: openai` in `config.yaml`. Base URL `https://api.openai.com/v1`. Key variable `OPENAI_API_KEY`.

Model ids are read from config only. They were copied from the docs on 2026-09-26. If a call fails, change the id in `config.yaml` from the docs. Do not invent a replacement in code.

- Fast / vision / indexing: `openai_fast_model` `gpt-6-luna`. https://developers.openai.com/api/docs/models/gpt-6-luna
- Reasoning: `openai_reasoning_model` `gpt-6-astra`. https://developers.openai.com/api/docs/models/gpt-6-astra
- Questions currently use the fast model (`query_model: fast`).

Calls use `client.chat.completions.parse(..., response_format=<pydantic model>)` with an `image_url` data URL. `image_detail` may be `low`, `high`, `original`, or `auto`. Config uses `low`. https://platform.openai.com/docs/guides/images-vision

xAI is a config switch. Set `provider: xai`, then `grok-4.3`, `grok-4.7`, base URL `https://api.x.ai/v1`, and `XAI_API_KEY`. https://docs.x.ai/developers/models/grok-4.3 and https://docs.x.ai/developers/models/grok-4.7

`reasoning.effort` is a Responses API field. It is not sent on these Chat Completions calls.

## Detector

Default backend is OWLv2 `google/owlv2-base-patch16-ensemble`. Set `detector: grounding_dino` to use `IDEA-Research/grounding-dino-tiny`. Weights are not in git. Install with `pip install -r requirements-detector.txt` (about 1 GB on first use). Tests inject a fake detector and do not download weights.

## Camera

`image_width`, `image_height`, `hfov_deg`, and `vfov_deg` in `config.yaml` are placeholders until the lens is measured. Optional `fx`, `fy`, `cx`, `cy`, and `dist_coeffs` override that model when set. Agree pan-home and the sign convention with the hardware teammate before a demo.

## Secrets and retries

The key is read from the environment. A repo-root `.env` is loaded when it exists. An environment variable already set is left alone. The placeholder `your-key-here` is rejected.

`.gitignore` covers `.env`, `data/cache/`, `data/out/`, video files, virtualenvs, and model weights. `.githooks/pre-commit` blocks a staged `.env` and added lines that look like `sk-` or `xai-` keys. Enable it with `git config core.hooksPath .githooks`. Logs pass through a redactor.

Transport retries cover connection failures, timeouts, and HTTP 408, 409, 429, and 5xx. HTTP 400 and 401 are not retried. A 400 points at the provider docs. A 401 or 403 points at the key page. Schema mismatches retry once, then that image is skipped. `request_timeout_s` is 3600 because a reasoning call can be slow.

## What this module does not do

- It does not draw a UI or move the laser.
- It does not estimate distance.
- It does not read HEIC.
- It does not invent a model id when a call fails.
- It does not search `data/references/` during `index` or `ask`.
- `pytest` mocks the API. A live call needs a real photo and a key in `.env`.

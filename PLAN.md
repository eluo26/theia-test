# Vision module plan

Python package for a HackGT laser-pointing turret. The configured vision provider does scene understanding, and a local open-vocabulary detector tightens boxes. No web framework. The command line is `python -m vision.cli`.

This file is the spec for milestones M1–M6. M1 through M6 are implemented. The operator guide is README.md. The geometry below is the full pinhole rotation in `vision/geometry.py`, not a small-angle fraction of the FOV.

`provider` in `config.yaml` is `xai` by default. Model ids live only in that file.

OpenAI ids were copied from the docs on 2026-09-26 ([catalog](https://developers.openai.com/api/docs/models)):

- `openai_fast_model`: `gpt-6-luna` ([docs](https://developers.openai.com/api/docs/models/gpt-6-luna)). Image input, structured outputs, Chat Completions. High-volume model used for indexing.
- `openai_reasoning_model`: `gpt-6-astra` ([docs](https://developers.openai.com/api/docs/models/gpt-6-astra)). Image input, structured outputs, Chat Completions. Flagship reasoning model used for the text query.
- Base URL `https://api.openai.com/v1`. Key env `OPENAI_API_KEY`.

`provider: xai` uses the Grok settings already in that file:

- `grok_fast_model`: `grok-4.3` ([docs](https://docs.x.ai/developers/models/grok-4.3)).
- `grok_reasoning_model`: `grok-4.7` ([docs](https://docs.x.ai/developers/models/grok-4.7)).
- Base URL `https://api.x.ai/v1`. Key env `XAI_API_KEY`.

If a call fails, update those names from the docs. Do not invent a replacement id in Python.

The vision call uses the OpenAI SDK: `client.chat.completions.parse` with `response_format` set to the Pydantic model, and `image_url` detail taken from `config.yaml` (`auto`, `low`, `high`, or `original`). https://developers.openai.com/api/docs/guides/structured-outputs and https://platform.openai.com/docs/guides/images-vision . Image input allows PNG, JPEG, WEBP, and non-animated GIF. A local cap is `max_image_bytes`.

## Coordinate convention

- Image origin is the top-left. X increases to the right. Y increases downward.
- Grok boxes are `[x1, y1, x2, y2]` normalized to 0–1000 on the image or tile that was sent. Every coordinate is inside that range, `x2 > x1`, and `y2 > y1`.
- `norm_box_to_pixels` maps a box onto pixels with `round(coord / 1000 * size)`, clamps each edge to `[0, size - 1]`, and expands a collapsed edge by one pixel so the rectangle stays visible.
- `bbox_px` is that box in full-frame pixels after any tile is mapped back (tile mapping is M3).
- `pan_deg` and `tilt_deg` are the turret angles of the capture, in degrees. Filenames may encode them (`pan030_tilt-10.jpg`).
- `azimuth_deg` and `elevation_deg` are where to point the laser, in degrees. Positive azimuth is to the right of the turret zero. Positive elevation is up. Because image Y grows downward, a pixel below the principal point contributes a negative elevation offset.
- Geometry: box center `(u, v)`. Intrinsics from calibration when `fx`, `fy`, `cx`, and `cy` are set, otherwise `fx = (W/2)/tan(hfov/2)`, `fy = (H/2)/tan(vfov/2)`, `cx = W/2`, `cy = H/2`. Camera ray `d = [(u-cx)/fx, -(v-cy)/fy, 1]` (x right, y up, z forward). Rotate by tilt about x, then pan about vertical. `az = atan2(dx, dz)`, `el = atan2(dy, sqrt(dx^2+dz^2))`. Full rotation, not a small-angle model. `dist_coeffs` with a complete intrinsic set undistorts before tiling.
- `range_m` is always null. This module does not estimate distance.

## JSON contract

`QueryResult` is what the turret teammate consumes. `locate()` fills it in M6. `range_m` stays null.

```json
{
  "status": "found",
  "object_id": "obj_007",
  "label": "blue water bottle",
  "azimuth_deg": 34.2,
  "elevation_deg": -12.5,
  "range_m": null,
  "confidence": 0.86,
  "frame_file": "pan030_tilt-10.jpg",
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

- `status` is `found`, `ambiguous`, or `not_found`.
- `candidates` items are `{object_id, label, confidence, reason}` and are used when the query matches more than one object.
- `metadata` carries `count`, `drug_name`, `expiry_text`, and `last_seen`.
- Unknown fields are rejected.

M1 asks Grok for `IndexResponse` instead. One object:

```json
{
  "label": "blue water bottle",
  "description": "color, brand, readable text, and neighboring objects",
  "box": [100, 200, 300, 800],
  "count": 1,
  "drug_name": null,
  "expiry_text": "EXP 2027-04"
}
```

`drug_name` is the medication name when the object is a drug package, otherwise null. `expiry_text` is visible expiry text, otherwise null. The prompt skips empty wall, floor, and bare tabletop, and it must not invent hidden objects.

Internal models already defined for later milestones, so the contract does not drift:

- `Frame`: `source_file`, `pan_deg`, `tilt_deg`, `timestamp`, `width`, `height`. The image array is attached by ingest and is not stored in JSON.
- `Detection`: one observation in full-frame pixels before catalog dedupe. Includes `bbox_px`, `confidence`, `box_source` (`grok` or `detector`), the frame file, pan/tilt, and optional azimuth/elevation.
- `CatalogObject` / `Catalog`: deduped objects written to `data/out/catalog.json`, with a stable `object_id`.

## Secrets

- The active provider's key (`OPENAI_API_KEY` or `XAI_API_KEY`) is read only from the environment. A repo-root `.env` is loaded when it exists (`override` is false, so a real environment variable wins).
- Missing key, empty key, or the placeholder `your-key-here` raises a clear error that points at the provider's key page and tells the operator to copy `.env.example` to `.env`.
- Commit `.env.example` with `OPENAI_API_KEY=your-key-here` and a commented `XAI_API_KEY=your-key-here`. Gitignore `.env`, `.env.*` (except the example), `data/cache/`, `data/out/`, video files, virtualenvs, Python caches, and model-weight files.
- `.githooks/pre-commit` blocks staged env files other than `.env.example`, and blocks added lines that contain an `xai-` token or an `sk-` token. Install with `git config core.hooksPath .githooks`.
- Logs and exceptions pass through a redactor. `check-config` prints `present` or `missing`, never the key.

## Cache, retries, logging

- Cache key is the SHA-256 of image bytes, prompt, model, and image detail. The plan's key is image bytes + prompt + model; image detail is included so a config change does not reuse boxes from another detail level.
- Cache files are `data/cache/<sha256>.json` with `{"model", "response"}`. A model mismatch or invalid JSON is ignored. Only a schema-valid response is cached.
- Transport retries (`max_transport_retries`, exponential backoff from `retry_base_delay_s`) cover connection failures, timeouts, and HTTP 408, 409, 429, and 5xx. HTTP 400 and 401 are not retried. The SDK client is created with `max_retries=0` so retries stay in this policy. A 400 tells the operator to check the docs rather than changing the model in code. A 401 or 403 tells them to check or revoke the key.
- Schema mismatches retry `validation_retries` times, then the image is skipped.
- Logs include model, latency, cache hit or miss, and validation pass or fail.

## Milestones

### M1 — Config, secrets, one Grok call

Implemented.

- `config.yaml` loaded by Pydantic. Unknown keys and invalid values fail with the field name. Model ids have no defaults in Python.
- `.env` loading, `.gitignore`, `.env.example`, and the pre-commit hook.
- `python -m vision.cli check-config` and `python -m vision.cli describe <image>`.
- One image, one call to the configured fast model, response validated as `IndexResponse`.
- Annotated debug image: green box and label per object, legend "Grok".
- Tests mock the client. No live call in M1, because the key is not available in this environment.
- No synthetic scene photos are stored in the repo. Geometry and schema tests use in-memory numbers.

### M2 — Ingest

Implemented in `vision/ingest.py`. Video uses `--sweep` or a `timestamp_s,pan,tilt` CSV. Soft frames are dropped. The sharpest frame in each 1-degree bucket is kept.

- Read angle-tagged stills from `data/test_scenes/` into `Frame` records. Pan and tilt come from a sidecar or from the filename.
- Optional video: sample a frame every `video_sample_every_s` (0.5 s). Drop frames whose blur score is below `blur_threshold` (variance of Laplacian, threshold 100).
- OpenCV is added in this milestone. The image array stays on the frame object and is not written into JSON.

### M3 — Tiled indexing

Implemented. `tile_grid` is `[columns, rows]`. Tile failures are logged and skipped.

- Split each frame into `tile_grid` (`[nx, ny]`, columns then rows) with `tile_overlap` (fraction of the tile shared with its neighbor).
- Call the configured fast model on each tile, at most `max_concurrency` calls at once. Same prompt and schema as M1.
- Map each tile's 0–1000 box back to full-frame `bbox_px`.
- Use the M1 cache. Skip a tile after validation retries fail, and continue with the rest of the frame.

### M4 — Camera geometry

Implemented in `vision/geometry.py`. `range_m` stays null.

- Fill `azimuth_deg` and `elevation_deg` from pan, tilt, and the box center, using the coordinate convention above.
- `image_width`, `image_height`, `hfov_deg`, and `vfov_deg` in `config.yaml` are placeholders until a human measures the lens. Optional `fx`, `fy`, `cx`, `cy`, and `dist_coeffs` override the FOV model.
- Leave `range_m` null.

### M5 — Detector refinement and catalog

Implemented.

- Local open-vocabulary detector selected by `detector`: `owlv2` (`owlv2_model`) or `grounding_dino` (`grounding_dino_model`). Torch and transformers are in `requirements-detector.txt`. The first run downloads about 1 GB of weights into a gitignored cache. Tests inject a fake detector.
- For each Grok detection, the detector runs on the tile with the label as the text query. If IoU is at least `refine_iou`, the detector box is used (`box_source` `detector`). Otherwise the Grok box is kept at lower confidence. Debug images draw detector boxes in blue.
- Dedupe when the rapidfuzz token ratio is at least `label_sim` and the angular separation is at most `merge_deg`. The representative is the member closest to its frame center. Write `data/out/catalog.json`.
- `clinic_mode` fuzzy-matches `drug_name` against `data/inventory.json` at query time and attaches the match to metadata.

### M6 — Query

Implemented. `locate(query, catalog)` sends catalog text (id, label, description, count, drug name) to the configured reasoning model, then crops the chosen frame with `crop_pad`, asks for a tight box, and runs the detector on the crop. The tighter agreeing box is mapped back and azimuth/elevation are recomputed.

- `found` when one object wins. `ambiguous` with `candidates` when more than one is plausible. `not_found` with a `reason` when nothing matches.
- `range_m` remains null. `python -m vision.cli ask` prints the JSON. `python -m vision.cli eval` reports correct-object rate and mean angular error. Still no web framework.

## [HUMAN]

- Create an xAI API key at https://console.x.ai. It may be shown only once.
- Store it in `.env` as `XAI_API_KEY`. Never commit `.env`. Confirm `git status` before committing. If a key leaks, revoke it and create a new one.
- Confirm `grok_fast_model` and `grok_reasoning_model` in `config.yaml` against the docs before the demo. To use OpenAI later, set `provider: openai` and `OPENAI_API_KEY`.
- Measure the camera field of view, width, and height later, and replace the placeholders. Add intrinsics if the lens needs them.
- Shoot real angle-tagged photos into `data/test_scenes/<scene>/`. This repo does not include synthetic stand-in images.

# Vision module plan

Python package for a HackGT laser-pointing turret. Grok (xAI, OpenAI-compatible base URL `https://api.x.ai/v1`) does scene understanding, and a local open-vocabulary detector tightens boxes in a later milestone. No web framework. The command line is `python -m vision.cli`.

This file is the spec for milestones M1–M6. M1 is implemented. Do not start M2 until M1 is accepted.

Model ids live only in `config.yaml`. Chosen from the xAI docs on 2026-09-26:

- `grok_fast_model`: `grok-4.3` ([docs](https://docs.x.ai/developers/models/grok-4.3)). Image input and structured outputs. This is the fast model. Older grok-4-fast aliases resolve to it.
- `grok_reasoning_model`: `grok-4.7` ([docs](https://docs.x.ai/developers/models/grok-4.7)). Image input, structured outputs, reasoning default high.

If a call fails, update those names from the docs. Do not invent a replacement id in Python.

The vision call uses the OpenAI SDK: `client.beta.chat.completions.parse` with `response_format` set to the Pydantic model, and `image_url.detail` taken from `config.yaml` (`auto`, `low`, or `high`). Images are jpg, jpeg, or png, at most `max_image_bytes` (20 MiB).

## Coordinate convention

- Image origin is the top-left. X increases to the right. Y increases downward.
- Grok boxes are `[x1, y1, x2, y2]` normalized to 0–1000 on the image or tile that was sent. Every coordinate is inside that range, `x2 > x1`, and `y2 > y1`.
- `norm_box_to_pixels` maps a box onto pixels with `round(coord / 1000 * size)`, clamps each edge to `[0, size - 1]`, and expands a collapsed edge by one pixel so the rectangle stays visible.
- `bbox_px` is that box in full-frame pixels after any tile is mapped back (tile mapping is M3).
- `pan_deg` and `tilt_deg` are the turret angles of the capture, in degrees. Filenames may encode them (`pan030_tilt-10.jpg`).
- `azimuth_deg` and `elevation_deg` are where to point the laser, in degrees. Positive azimuth is to the right of the turret zero. Positive elevation is up. Because image Y grows downward, a pixel below the principal point contributes a negative elevation offset.
- Geometry (M4): with only the FOV, the object center's horizontal fraction of `hfov_deg` is added to `pan_deg`, and the vertical fraction of `vfov_deg` is applied to `tilt_deg` with image-down as negative elevation. When `fx`, `fy`, `cx`, and `cy` are set, they override that FOV model. `dist_coeffs`, when set, undistort the pixel before the conversion.
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

- `XAI_API_KEY` is read only from the environment. A repo-root `.env` is loaded when it exists (`override` is false, so a real environment variable wins).
- Missing key, empty key, or the placeholder `your-key-here` raises a clear error that points at https://console.x.ai and tells the operator to copy `.env.example` to `.env`.
- Commit `.env.example` with `XAI_API_KEY=your-key-here`. Gitignore `.env`, `.env.*` (except the example), `data/cache/`, `data/out/`, video files, virtualenvs, Python caches, and model-weight files.
- `.githooks/pre-commit` blocks staged env files other than `.env.example`, and blocks added lines that contain an `xai-` token of 10 or more alphanumeric characters. Install with `git config core.hooksPath .githooks`.
- Logs and exceptions pass through a redactor. `check-config` prints `present` or `missing`, never the key.

## Cache, retries, logging

- Cache key is the SHA-256 of image bytes, prompt, model, and image detail. The plan's key is image bytes + prompt + model; image detail is included so a config change does not reuse boxes from another detail level.
- Cache files are `data/cache/<sha256>.json` with `{"model", "response"}`. A model mismatch or invalid JSON is ignored. Only a schema-valid response is cached.
- Transport retries (`max_transport_retries`, exponential backoff from `retry_base_delay_s`) cover connection failures, timeouts, and HTTP 408, 409, 429, and 5xx. HTTP 400 and 401 are not retried. The SDK client is created with `max_retries=0` so retries stay in this policy. A 400 tells the operator to check the docs rather than changing the model in code. A 401 or 403 tells them to check or revoke the key.
- Schema mismatches retry `validation_retries` times, then the image is skipped.
- Logs include model, latency, cache hit or miss, and validation pass or fail.

## Milestones

### M1 — Config, secrets, one Grok call

Implemented. Stop here.

- `config.yaml` loaded by Pydantic. Unknown keys and invalid values fail with the field name. Model ids have no defaults in Python.
- `.env` loading, `.gitignore`, `.env.example`, and the pre-commit hook.
- `python -m vision.cli check-config` and `python -m vision.cli describe <image>`.
- One image, one call to `grok_fast_model`, response validated as `IndexResponse`.
- Annotated debug image: green box and label per object, legend "Grok".
- Tests mock the client. No live call in M1, because the key is not available in this environment.
- Synthetic scene: `data/test_scenes/m1/scene.png`.

### M2 — Ingest

Not started.

- Read angle-tagged stills from `data/test_scenes/` into `Frame` records. Pan and tilt come from a sidecar or from the filename.
- Optional video: sample a frame every `video_sample_every_s` (0.5 s). Drop frames whose blur score is below `blur_threshold` (variance of Laplacian, threshold 100).
- OpenCV is added in this milestone. The image array stays on the frame object and is not written into JSON.

### M3 — Tiled indexing

Not started.

- Split each frame into `tile_grid` (`[nx, ny]`, columns then rows) with `tile_overlap` (fraction of the tile shared with its neighbor).
- Call `grok_fast_model` on each tile, at most `max_concurrency` calls at once. Same prompt and schema as M1.
- Map each tile's 0–1000 box back to full-frame `bbox_px`.
- Use the M1 cache. Skip a tile after validation retries fail, and continue with the rest of the frame.

### M4 — Camera geometry

Not started.

- Fill `azimuth_deg` and `elevation_deg` from pan, tilt, and the box center, using the coordinate convention above.
- `image_width`, `image_height`, `hfov_deg`, and `vfov_deg` in `config.yaml` are placeholders until a human measures the lens. Optional `fx`, `fy`, `cx`, `cy`, and `dist_coeffs` override the FOV model.
- Leave `range_m` null.

### M5 — Detector refinement and catalog

Not started.

- Local open-vocabulary detector selected by `detector`: `owlv2` (`owlv2_model`) or `grounding_dino` (`grounding_dino_model`). Add torch and transformers here. The first run downloads the weights (on the order of 1 GB) into a gitignored cache.
- Run the detector on a crop of each Grok box expanded by `crop_pad`. Keep the tighter box when its IoU with the Grok box is at least `refine_iou`. Record `box_source` as `grok` or `detector`. Draw detector boxes in a color other than the M1 green.
- Dedupe observations into `Catalog` when label similarity (0–100) is at least `label_sim` and the angular separation is at most `merge_deg`. Write `data/out/catalog.json`.
- `clinic_mode`: when true, medication packages stay first-class in the catalog (`drug_name`, `expiry_text`). The M1 prompt already asks for those fields.

### M6 — Query

Not started.

- `locate(query)` sends the catalog, and any needed crop, to `grok_reasoning_model` and returns `QueryResult`.
- `found` when one object wins. `ambiguous` with `candidates` when more than one is plausible. `not_found` with a `reason` when nothing matches.
- `range_m` remains null. A CLI command can print the JSON for the turret teammate. Still no web framework.

## [HUMAN]

- Create an API key at https://console.x.ai. It may be shown only once.
- Store it in `.env` as `XAI_API_KEY`. Never commit `.env`. Confirm `git status` before committing. If a key leaks, revoke it and create a new one.
- Confirm `grok_fast_model` and `grok_reasoning_model` in `config.yaml` against the docs before the demo.
- Measure the camera field of view, width, and height later, and replace the placeholders. Add intrinsics if the lens needs them.
- Shoot real angle-tagged test scenes later. `data/test_scenes/m1/scene.png` is a synthetic stand-in, not a turret photo.

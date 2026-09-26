# Turret vision
We begin with the computer vision aspect of the project. This pipeline is designed to take in images from the camera as .jpg files, and interpolate it through an agent that recognizes objects, and also take in user inquiries as data. The output is data in spherical coordinates, along with other metadata that would allow the UI and pointer elements of the project to find the object given my data.


Python module for a HackGT tabletop laser turret. A camera scans in angle-tagged steps. This package turns those pictures into a spherical direction: azimuth, elevation, a bounding box, and metadata. A teammate owns the UI and the laser loop. There is no web framework.

Two framings use the same pipeline: finding an object at home ("where's my blue water bottle?") and finding a package in a clinic sample closet ("where's the Jardiance starter pack?").

OpenAI is the current vision provider (`provider: openai` in `config.yaml`, base URL `https://api.openai.com/v1`). xAI stays available by setting `provider: xai`. Model names are read from config, so that switch does not need a code change. A local open-vocabulary detector (OWLv2 by default, or Grounding DINO) can tighten boxes. The command line is `python -m vision.cli`.

The GitHub repo name for this project is `theia-test` (`https://github.com/eluo26/theia-test`). This checkout already has a git history and an `origin` remote. Do not run a "first commit" snippet that calls `git init` or replaces `origin`.

## What was verified here

- `pytest -q` passes. The API client is mocked. Tests do not read saved scene photos.
- `python -m vision.cli check-config` prints `present` or `missing` for the active key. It does not print the key.

Not verified here:

- A live vision call on a real photo. This repo does not contain one. A models list or a tiny text call can check that `OPENAI_API_KEY` works.
- OWLv2 or Grounding DINO weights (about 1 GB). Tests inject a fake detector and do not download them.
- A push to `https://github.com/eluo26/theia-test` if the remote add or push in the manual steps failed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git config core.hooksPath .githooks
```

`scripts/install-hooks.sh` sets that hooks path and marks `.githooks/pre-commit` executable. The hook refuses a staged `.env` (`.env.example` is allowed) and any added line that looks like an `xai-` key or an OpenAI-style `sk-` key.

Optional detector, only when you want local box refinement:

```bash
pip install -r requirements-detector.txt
```

## Commands

Check the loaded settings. This does not print the API key:

```bash
python -m vision.cli check-config
```

Describe one real photo. Needs a key in `.env`. This repo does not include a sample image:

```bash
python -m vision.cli describe path/to/photo.jpg
```

Index a folder of angle-tagged photos. Writes `data/out/catalog.json` and debug images under `data/out/debug/`:

```bash
python -m vision.cli index data/test_scenes/desk
```

Ask where something is. This indexes the folder, then prints one `QueryResult` JSON object. `range_m` is always null:

```bash
python -m vision.cli ask data/test_scenes/desk "where's my blue water bottle?"
```

Score a scene against `ground_truth.json` after you have measured it. Pass a directory or a name under `data/test_scenes/`. The desk sheet ships with an empty `queries` list, and `eval` exits until you add rows. The report is the correct-object rate and the mean angular error in degrees:

```bash
python -m vision.cli eval desk
```

Video ingest. Sample every `video_sample_every_s`, drop frames whose Laplacian variance is below `blur_threshold`, and keep the sharpest frame in each 1-degree pan/tilt bucket.

Constant-speed sweep (pan moves from start to end, tilt is fixed):

```bash
python -m vision.cli index sweep.mp4 --sweep -40 40 0
```

Or a sidecar CSV with header `timestamp_s,pan,tilt`. Pan and tilt are interpolated at each sampled timestamp:

```bash
python -m vision.cli index sweep.mp4 --angles-csv angles.csv
```

`ask` accepts the same `--sweep` and `--angles-csv` flags. `--out-dir` redirects `catalog.json` and debug images. `--no-cache` ignores `data/cache/`.

Without `OPENAI_API_KEY` (or `XAI_API_KEY` when `provider` is `xai`), `index`, `ask`, and `eval` exit with a message that points at the key page and `.env`. They do not invent a model name. If a live call fails, the error points at the provider docs.

## Library

```python
from vision import build_catalog, locate

catalog = build_catalog("data/test_scenes/desk")
result = locate("where's my blue water bottle?", catalog)
```

`catalog.json` is written to `data/out/`. `locate` returns the turret JSON: `status` (`found`, `ambiguous`, or `not_found`), `object_id`, `label`, `azimuth_deg`, `elevation_deg`, `range_m` (null), `confidence`, `frame_file`, `bbox_px`, `candidates`, `metadata`, and `reason`.

## Pipeline

1. Ingest stills. `pan060_tilt-10.jpg` supplies pan and tilt. `manifest.json` entries `{file, pan, tilt, timestamp}` win over the filename. Full resolution is kept.
2. Undistort when `fx`, `fy`, `cx`, `cy`, and `dist_coeffs` are all set. Those intrinsics also override the FOV model. Tile with `tile_grid` `[columns, rows]` and `tile_overlap`.
3. Each tile goes to the configured fast vision model (structured vision). Calls run with `max_concurrency` and exponential backoff. Invalid JSON is retried once, then that tile is logged and skipped. The cache key is the SHA-256 of the image bytes, prompt, model, and image detail.
4. For each vision detection, the local detector runs on that tile with the label as the text query. If IoU is at least `refine_iou`, the detector box is kept (`box_source` `detector`). Otherwise the vision box is kept and confidence is lower.
5. The box center becomes a camera ray, rotated by tilt about x and then pan about vertical. Azimuth is `atan2(dx, dz)`. Elevation is `atan2(dy, sqrt(dx^2+dz^2))`. This is the full rotation, not a small-angle shortcut.
6. Detections merge when the rapidfuzz token ratio is at least `label_sim` and the angular separation is at most `merge_deg`. The representative view is the one closest to its frame center. Ids are `obj_001`, `obj_002`, ... in a stable order.
7. The question plus catalog text (id, label, description, count, drug name only) goes to the configured reasoning model. It returns status, object id, up to three candidates, confidence, and a reason.
8. The chosen object is cropped with `crop_pad`. The fast vision model is asked whether the crop is that label and to return a tight box. The detector runs on the crop too. The tighter box is kept when the two agree, then azimuth and elevation are recomputed.
9. With `clinic_mode: true`, `drug_name` is fuzzy-matched against `data/inventory.json` and the match is attached to `metadata.inventory`.

Debug images use green for vision-model boxes and blue for detector boxes, with labels and ids. The green legend uses the active provider name from `config.yaml` (`OpenAI` or `xAI`).

Image-conditioned OWLv2 search from `data/references/` is implemented in `vision/references.py` (`image_guided_boxes`) and is not called by `index` or `ask`. It needs the detector weights and was not run here. See the manual steps.

## Photos

Do not generate stand-in pictures. Put real photos in `data/test_scenes/<scene>/` when you have them. A filename such as `pan060_tilt-10.jpg` carries pan and tilt. `manifest.json` starts with an empty `frames` list. An entry `{file, pan, tilt, timestamp}` is optional and wins over the filename when you add one. `ground_truth.json` starts with an empty `queries` list. `describe`, `index`, and `ask` do not read object names from either file. The image files themselves are not in this repo.

## Config

`provider` defaults to `openai`. Model ids live only in `config.yaml`. The OpenAI ids were copied from the docs on 2026-09-26:

- `openai_fast_model`: `gpt-6-luna` for indexing and vision. https://developers.openai.com/api/docs/models/gpt-6-luna
- `openai_reasoning_model`: `gpt-6-astra` for the text query. https://developers.openai.com/api/docs/models/gpt-6-astra
- Catalog: https://developers.openai.com/api/docs/models

Set `provider: xai` to use the existing Grok settings instead:

- `grok_fast_model`: `grok-4.3`. https://docs.x.ai/developers/models/grok-4.3
- `grok_reasoning_model`: `grok-4.7`. https://docs.x.ai/developers/models/grok-4.7
- Base URL `https://api.x.ai/v1`, key env `XAI_API_KEY`.

Vision calls use `client.chat.completions.parse(..., response_format=<pydantic model>)` with an `image_url` data URL and `detail` from config. https://developers.openai.com/api/docs/guides/structured-outputs and https://platform.openai.com/docs/guides/images-vision

OpenAI image input allows PNG, JPEG, WEBP, and non-animated GIF. `image_detail` may be `low`, `high`, `original`, or `auto`. The local size cap is `max_image_bytes`. Reasoning effort is a Responses API field (`reasoning.effort`). It is not sent on Chat Completions.

`detector` is `owlv2` (`google/owlv2-base-patch16-ensemble`) or `grounding_dino` (`IDEA-Research/grounding-dino-tiny`).

Azimuth 0 is pan home. Positive azimuth is to the right, clockwise from above. Elevation 0 is horizontal. Positive elevation is up. Degrees. `range_m` is always null.

## MANUAL STEPS

- Create the private GitHub repo https://github.com/eluo26/theia-test if needed and push (commands below). Do not run a snippet that does `git init` or that replaces the existing `origin` remote.
- Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Create the key at https://platform.openai.com/api-keys. `git status` must not show `.env`.
- For xAI later, set `provider: xai` and add `XAI_API_KEY` from https://console.x.ai. Share keys via DM. Each teammate uses their own key. If a key is leaked, revoke it and create a new one. Deleting a commit is not enough.
- Confirm the model names in the docs still match `config.yaml`: https://developers.openai.com/api/docs/models/gpt-6-luna and https://developers.openai.com/api/docs/models/gpt-6-astra. If a live call fails, update the ids from those pages. Do not invent a replacement id in code.
- Create a venv and `pip install -r requirements.txt`. Detector weights are about 1 GB on the first real detector run (`pip install -r requirements-detector.txt`).
- Shoot a real staged scene later: phone main lens, 12 MP, objects at 2–3 m, photos every ~20 degrees, angles in the filenames, one slow sweep video, and a hand-written `ground_truth.json`.
- Measure the camera FOV and, if you can, do a checkerboard calibration. Replace the placeholder width, height, FOV, and intrinsics in `config.yaml`.
- Agree pan-home and the sign convention with the hardware teammate before the demo. Azimuth 0 is pan home, positive is right (clockwise from above). Elevation 0 is horizontal, positive is up.

GitHub, from this repo, without removing `origin`:

```bash
git remote add github https://github.com/eluo26/theia-test.git
git push -u github main
```

If `github` already exists, skip the `remote add`. If the push fails because the repo is missing or the account is not logged in, create the private repo on GitHub and authenticate, then run the push again. Do not delete `origin`.

Copy `.env.example` to `.env` and paste the key:

```bash
cp .env.example .env
git status
```

`git status` should not list `.env`. The placeholder `your-key-here` is rejected.

Set `clinic_mode: true` in `config.yaml` when you want inventory matches from `data/inventory.json`. That file is a mock list (counts, expiry, resource links).

To try image-conditioned search later, put reference photos in `data/references/` and call `vision.references.image_guided_boxes` after the detector weights are installed. `index` and `ask` do not call it.

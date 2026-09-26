# Turret vision

Python vision module for a HackGT laser-pointing turret. Grok (xAI) does scene understanding through the OpenAI-compatible API at `https://api.x.ai/v1`. There is no web framework. The command line is `python -m vision.cli`.

Milestone 1 is the current stop line. It loads `config.yaml` and `.env`, makes one Grok vision call on one image, validates the JSON, and writes an annotated debug image. Later milestones (ingest, tiling, camera geometry, the local detector, the catalog, and query) are specified in [PLAN.md](PLAN.md) and are not implemented.

A live Grok call was not run. `XAI_API_KEY` is not in this environment, and the tests mock the client.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git config core.hooksPath .githooks
```

`scripts/install-hooks.sh` runs that `git config` command and marks the hook executable. The hook refuses commits that stage `.env` (`.env.example` is allowed) or lines that contain an `xai-` API-key-like token.

## Check the config

```bash
python -m vision.cli check-config
```

This prints model names, the base URL, and whether `XAI_API_KEY` is present or missing. It does not print the key.

## Tests

```bash
pytest -q
```

Tests mock the OpenAI client. They do not call xAI.

## Describe one image

After the human steps below, with a real key in `.env`:

```bash
python -m vision.cli describe data/test_scenes/m1/scene.png
```

Stdout is the validated object JSON. The command also writes `data/out/scene.json` and `data/out/scene_annotated.png`. Those output paths are gitignored. Add `--no-cache` to ignore `data/cache/`, or `--out-dir` to write somewhere else.

`data/test_scenes/m1/scene.png` is a synthetic stand-in (a bottle, a medication box, a lamp, and a plant). It is not a photo from the turret camera.

Model ids are read from `config.yaml` (`grok_fast_model`, `grok_reasoning_model`). They are not hardcoded in Python. If a live call fails, check the xAI docs and update `config.yaml`. Do not swap in a different model name from code.

## [HUMAN]

- Create an API key at [console.x.ai](https://console.x.ai). It may be shown only once.
- Copy `.env.example` to `.env` and set `XAI_API_KEY`. The placeholder `your-key-here` is rejected.
- Never commit `.env`. Confirm `git status` before every commit. If a key leaks, revoke it at the console and create a new one.
- Confirm the model names in `config.yaml` against the current docs before a demo: [grok-4.3](https://docs.x.ai/developers/models/grok-4.3) and [grok-4.7](https://docs.x.ai/developers/models/grok-4.7).
- Measure the camera field of view later and replace the placeholder width, height, and FOV in `config.yaml`.
- Shoot real angle-tagged test scenes later. The synthetic image in `data/test_scenes/m1/` is only for milestone 1.

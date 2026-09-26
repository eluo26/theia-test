# Turret vision

This is the vision piece of the tabletop laser turret. It reads camera JPEGs, finds objects, and answers a question such as "where is the blue water bottle?". The result is a direction in degrees, a box, and metadata for the UI and the pointer. It does not own the UI, the laser, or a web app.

The full list of commands, settings, and pipeline steps is in [GUIDE.md](GUIDE.md).

## Setup

From the repo root. Each person uses their own key. Put it only in `.env`. `.env.example` stays `OPENAI_API_KEY=your-key-here`.

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
git config core.hooksPath .githooks
```

macOS or Linux: `python3 -m venv .venv`, then `source .venv/bin/activate` and `pip install -r requirements.txt`.

`git status` must not list `.env`. If a key leaks, revoke it at https://platform.openai.com/api-keys and create a new one.

## Photos

Put JPEGs in `data/test_scenes/<scene>/`. The filename is the camera angle: `pan0_tilt0.jpg` is home and level, `pan30_tilt0.jpg` is 30° to the right, `pan-30_tilt0.jpg` is 30° to the left. Positive pan is right. Tilt 0 is level. Positive tilt is up.

Use JPEG from the phone's main lens. HEIC files are ignored. `manifest.json` and `ground_truth.json` start empty. `describe`, `index`, and `ask` do not read object names from them.

## Run

```powershell
.\.venv\Scripts\python.exe -m vision.cli check-config
.\.venv\Scripts\python.exe -m vision.cli describe data\test_scenes\<scene>\pan0_tilt0.jpg
.\.venv\Scripts\python.exe -m vision.cli index data\test_scenes\<scene>
.\.venv\Scripts\python.exe -m vision.cli ask data\test_scenes\<scene> "where is the blue water bottle?"
```

Replace `<scene>` with the folder you added, and ask about an object that is actually in those photos. `check-config` prints `present` or `missing`. It does not print the key. A later ask reuses `data/out/catalog.json` when those photos have not changed. Add `--debug` on `index` or `ask` when you want box images under `data/out/debug/`.

`describe` writes `data/out/<name>.json` and `data/out/<name>_annotated.png`. `ask` prints one JSON object. Azimuth 0 is pan home, positive is right. Elevation 0 is level, positive is up. `range_m` is always null.

Offline check, with the API mocked:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If a live call fails because of a model name, change that name in `config.yaml` from the provider docs. Do not put a new model id in Python.

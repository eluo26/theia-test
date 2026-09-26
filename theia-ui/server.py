"""Local test page for DroidCam.

Run from the repo root:

    .venv\\Scripts\\python.exe theia-ui\\server.py

Then open http://127.0.0.1:8765
The page shows a sentence, not the raw aim JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_speak_spec = importlib.util.spec_from_file_location(
    "theia_ui_speak", Path(__file__).with_name("speak.py")
)
_speak = importlib.util.module_from_spec(_speak_spec)
assert _speak_spec.loader is not None
_speak_spec.loader.exec_module(_speak)
inventory_sentence = _speak.inventory_sentence
speak = _speak.speak

from vision.integrate import answer  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765
SHOT_DIR = ROOT / "data" / "test_scenes" / "droidcam"
PHOTO_PATHS = ("/photo.jpg", "/shot.jpg", "/photo", "/cam/1/frame.jpg")
STREAM_PATHS = ("/video", "/mjpegfeed")
MAX_PHOTO_BYTES = 20_000_000

_camera_base = ""


def main() -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    print("Enter the DroidCam address shown on the phone, such as http://192.168.1.20:4747")
    server.serve_forever()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            return
        if path == "/preview.jpg":
            self._preview()
            return
        if path == "/api/state":
            self._json(200, state())
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/camera":
                set_camera(str(body.get("url") or ""))
                self._json(200, state())
                return
            if path == "/api/capture":
                name, pan = capture_photo()
                payload = state()
                if pan == 0:
                    sentence = f"Saved {name}. This photo is straight ahead."
                else:
                    sentence = f"Saved {name}. This photo is {pan:.0f} degrees to the right of the first shot."
                payload["text"] = sentence
                self._json(200, payload)
                return
            if path == "/api/clear":
                clear_photos()
                payload = state()
                payload["text"] = "Cleared the photos. Take a new set from DroidCam."
                self._json(200, payload)
                return
            if path == "/api/ask":
                question = str(body.get("question") or "").strip()
                self._json(200, ask(question))
                return
        except UiError as exc:
            self._json(400, {"text": str(exc), "seen": ""})
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def log_message(self, fmt: str, *args) -> None:
        return

    def _preview(self) -> None:
        try:
            image = fetch_droidcam_jpeg(_camera_base)
        except UiError:
            self._send(404, "text/plain; charset=utf-8", b"No camera")
            return
        self._send(200, "image/jpeg", image)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise UiError("The page sent something that is not a question.") from exc
        if not isinstance(payload, dict):
            raise UiError("The page sent something that is not a question.")
        return payload

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class UiError(Exception):
    """A problem the page can show as a sentence."""


def set_camera(url: str) -> None:
    global _camera_base
    _camera_base = normalize_base(url)


def normalize_base(url: str) -> str:
    text = url.strip().rstrip("/")
    if not text:
        raise UiError("Enter the DroidCam address from the phone.")
    parsed = urlparse(text if "://" in text else "http://" + text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UiError("The DroidCam address should look like http://192.168.1.20:4747")
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def capture_photo() -> tuple[str, float]:
    image = fetch_droidcam_jpeg(_camera_base)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    number = _next_shot_number()
    name = f"shot{number}.jpg"
    (SHOT_DIR / name).write_bytes(image)
    ordered = sorted(SHOT_DIR.glob("shot*.jpg"), key=_shot_sort)
    index = next(i for i, path in enumerate(ordered) if path.name == name)
    return name, index * 30.0


def clear_photos() -> None:
    if not SHOT_DIR.is_dir():
        return
    for path in SHOT_DIR.glob("shot*.jpg"):
        path.unlink()


def ask(question: str) -> dict:
    if not question:
        raise UiError("Type a question first.")
    shots = sorted(SHOT_DIR.glob("shot*.jpg")) if SHOT_DIR.is_dir() else []
    if not shots:
        raise UiError("Take at least one photo from DroidCam before asking.")
    payload = answer(SHOT_DIR, question, save_debug=False)
    return {"text": speak(payload), "seen": inventory_sentence(payload)}


def state() -> dict:
    shots = []
    if SHOT_DIR.is_dir():
        for index, path in enumerate(sorted(SHOT_DIR.glob("shot*.jpg"), key=_shot_sort)):
            shots.append({"name": path.name, "pan": index * 30})
    return {"camera": _camera_base, "shots": shots, "text": "", "seen": ""}


def fetch_droidcam_jpeg(base: str) -> bytes:
    if not base:
        raise UiError("Enter the DroidCam address from the phone.")
    errors: list[str] = []
    for suffix in PHOTO_PATHS:
        try:
            return _read_url(base + suffix)
        except UiError as exc:
            errors.append(str(exc))
    for suffix in STREAM_PATHS:
        try:
            return _read_jpeg_frame(base + suffix)
        except UiError as exc:
            errors.append(str(exc))
    raise UiError(
        "Could not get a photo from DroidCam. Check that the phone and this computer "
        "are on the same Wi-Fi, and that the address matches the app."
    )


def _read_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "theia-ui"})
    try:
        with urlopen(request, timeout=8) as response:
            data = response.read(MAX_PHOTO_BYTES + 1)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise UiError(f"DroidCam did not answer at {url}.") from exc
    if len(data) > MAX_PHOTO_BYTES:
        raise UiError("The photo from DroidCam is too large.")
    if not data.startswith(b"\xff\xd8"):
        raise UiError("DroidCam did not return a JPEG.")
    return data


def _read_jpeg_frame(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "theia-ui"})
    try:
        with urlopen(request, timeout=8) as response:
            data = b""
            while len(data) < MAX_PHOTO_BYTES:
                chunk = response.read(65536)
                if not chunk:
                    break
                data += chunk
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    return data[start : end + 2]
    except (HTTPError, URLError, TimeoutError) as exc:
        raise UiError(f"DroidCam did not answer at {url}.") from exc
    raise UiError("DroidCam did not return a JPEG frame.")


def _next_shot_number() -> int:
    numbers = []
    for path in SHOT_DIR.glob("shot*.jpg"):
        digits = "".join(character for character in path.stem if character.isdigit())
        if digits:
            numbers.append(int(digits))
    return max(numbers, default=0) + 1


def _shot_sort(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Theia local test</title>
<style>
  body { margin: 0; font-family: Georgia, serif; background: #f4f1ea; color: #1c1915; }
  main { max-width: 720px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-weight: normal; font-size: 1.6rem; margin: 0 0 8px; }
  p.lead { margin: 0 0 24px; color: #4d473f; }
  label { display: block; margin: 16px 0 6px; font-family: sans-serif; font-size: 0.85rem; }
  input { width: 100%; box-sizing: border-box; font: 1rem sans-serif; padding: 10px 12px; border: 1px solid #c8c0b4; background: white; }
  button { font: 1rem sans-serif; margin: 12px 8px 0 0; padding: 10px 14px; background: #1c1915; color: white; border: 0; cursor: pointer; }
  button.secondary { background: transparent; color: #1c1915; border: 1px solid #1c1915; }
  img { width: 100%; margin-top: 16px; background: #ddd; min-height: 180px; object-fit: contain; }
  #answer { font-size: 1.45rem; line-height: 1.45; margin: 28px 0 8px; }
  #seen { font-family: sans-serif; color: #4d473f; }
  #shots { font-family: sans-serif; color: #4d473f; }
</style>
</head>
<body>
<main>
  <h1>Theia local test</h1>
  <p class="lead">Point DroidCam at the room, take a few photos, then ask in plain words.</p>
  <label for="camera">DroidCam address</label>
  <input id="camera" placeholder="http://192.168.1.20:4747" autocomplete="off">
  <button id="connect" type="button">Connect</button>
  <img id="preview" alt="DroidCam preview">
  <button id="capture" type="button">Take photo</button>
  <button id="clear" class="secondary" type="button">Clear photos</button>
  <p id="shots">No photos yet.</p>
  <label for="question">Question</label>
  <input id="question" placeholder="where is the blue water bottle?">
  <button id="ask" type="button">Ask</button>
  <p id="answer">Take a photo, then ask a question.</p>
  <p id="seen"></p>
</main>
<script>
const answer = document.getElementById("answer");
const seen = document.getElementById("seen");
const shots = document.getElementById("shots");
const preview = document.getElementById("preview");

function show(payload) {
  if (payload.text) answer.textContent = payload.text;
  if ("seen" in payload) seen.textContent = payload.seen || "";
  const count = (payload.shots || []).length;
  shots.textContent = count
    ? count + (count === 1 ? " photo saved." : " photos saved.") + " The first is straight ahead. Each next photo is 30 degrees to the right."
    : "No photos yet.";
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  const payload = await response.json();
  show(payload);
}

document.getElementById("connect").onclick = () => post("/api/camera", { url: document.getElementById("camera").value });
document.getElementById("capture").onclick = () => post("/api/capture", {});
document.getElementById("clear").onclick = () => post("/api/clear", {});
document.getElementById("ask").onclick = () => {
  answer.textContent = "Looking...";
  seen.textContent = "";
  post("/api/ask", { question: document.getElementById("question").value });
};

setInterval(() => {
  preview.src = "/preview.jpg?t=" + Date.now();
}, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

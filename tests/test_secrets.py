import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"


def _run_hook(repo: Path) -> subprocess.CompletedProcess[str]:
    """Git for Windows runs this bash hook. Invoke bash explicitly on Windows."""
    if os.name == "nt":
        bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        command = [str(bash) if bash.is_file() else "bash", str(HOOK)]
    else:
        command = [str(HOOK)]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True)


def test_gitignore_covers_the_required_paths():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (
        ".env",
        ".env.*",
        "!.env.example",
        "data/cache/",
        "data/out/",
        "*.mp4",
        "*.mov",
        "__pycache__/",
        ".venv/",
        ".cache/",
        "*.safetensors",
    ):
        assert entry in text


def test_env_example_has_no_live_key():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=your-key-here" in text
    assert "XAI_API_KEY=your-key-here" in text
    assert "sk-" not in text
    assert "xai-" not in text


def test_hook_blocks_an_env_file_and_a_key(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    leak = repo / "notes.txt"
    # Built at runtime so this source file does not itself contain a key-like token.
    leaked = "XAI_API_KEY=" + "xai-" + "thisisatestkey12345\n"
    leak.write_text(leaked, encoding="utf-8")
    subprocess.run(["git", "add", "notes.txt"], cwd=repo, check=True, capture_output=True)
    blocked = _run_hook(repo)
    assert blocked.returncode != 0
    assert "key-like" in blocked.stdout + blocked.stderr

    subprocess.run(["git", "rm", "--cached", "notes.txt"], cwd=repo, check=True, capture_output=True)
    openai_leak = repo / "openai.txt"
    openai_leak.write_text("OPENAI_API_KEY=" + "sk-" + ("a" * 24) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "openai.txt"], cwd=repo, check=True, capture_output=True)
    blocked_openai = _run_hook(repo)
    assert blocked_openai.returncode != 0
    assert "key-like" in blocked_openai.stdout + blocked_openai.stderr

    subprocess.run(["git", "rm", "--cached", "openai.txt"], cwd=repo, check=True, capture_output=True)
    env_file = repo / ".env"
    env_file.write_text("OPENAI_API_KEY=your-key-here\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, check=True, capture_output=True)
    blocked_env = _run_hook(repo)
    assert blocked_env.returncode != 0
    assert ".env" in blocked_env.stdout + blocked_env.stderr


def test_hook_allows_a_normal_file(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "readme.txt").write_text("no secrets here\n", encoding="utf-8")
    subprocess.run(["git", "add", "readme.txt"], cwd=repo, check=True, capture_output=True)
    allowed = _run_hook(repo)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr


def test_hook_is_executable():
    assert os.access(HOOK, os.X_OK)
    assert HOOK.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")

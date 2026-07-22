"""Create/finalize a reproducibility manifest for a pipeline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def tree_sha256(path: Path) -> dict:
    """Hash a file or a directory tree, including relative filenames."""
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "files": 0}
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.name if path.is_file() else file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(file_path).encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(path.resolve()), "exists": True,
        "sha256": digest.hexdigest(), "files": len(files),
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "finish"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status", choices=["running", "complete", "failed"], default="running")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--command", default="")
    parser.add_argument("--data-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "manifest.json"
    if args.action == "start":
        import torch

        weights = args.weights.resolve() if args.weights else None
        data = {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current")},
            "command": args.command,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "nvidia_driver": command_output([
                    "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
                ]),
                "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]),
            },
            "checkpoint": {
                "path": str(weights) if weights else None,
                "sha256": sha256(weights) if weights and weights.is_file() else None,
            },
            "parameters": {k: v for k, v in os.environ.items() if k.startswith("LTSR_")},
            "data_fingerprints": [tree_sha256(path) for path in args.data_path],
            "git_dirty": bool(git("status", "--porcelain")),
        }
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["status"] = args.status
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

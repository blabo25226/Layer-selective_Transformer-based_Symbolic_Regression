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


def record_stage(manifest_path: Path, stage: str, status: str) -> dict | None:
    """Record the outcome of a post-pipeline stage (validation, publication, ...).

    The pipeline itself finishes long before the run is checked and published, so a
    later failure must still be visible in the manifest. A failed stage moves the
    top-level status to ``<stage>_failed``; a successful one restores ``complete``.
    """
    if not manifest_path.is_file():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = data.setdefault("stages", {})
    stages[stage] = {"status": status, "at_utc": datetime.now(timezone.utc).isoformat()}
    if status == "failed":
        data["status"] = f"{stage}_failed"
    elif data.get("status", "").endswith("_failed"):
        still_failed = [name for name, info in stages.items() if info.get("status") == "failed"]
        data["status"] = f"{still_failed[0]}_failed" if still_failed else "complete"
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "finish", "stage", "resume"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status", choices=["running", "complete", "failed"], default="running")
    parser.add_argument("--stage", help="post-pipeline stage name, e.g. validation or publication")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--command", default="")
    parser.add_argument("--data-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "stage" and not args.stage:
        parser.error("--stage is required for the stage action")
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
    elif args.action == "stage":
        if record_stage(path, args.stage, "failed" if args.status == "failed" else "complete") is None:
            print(f"no manifest to annotate: {path}", file=sys.stderr)
        return 0
    elif args.action == "resume":
        # Re-entering an existing run (RESUME=1). Preserve the original start
        # provenance and append a resume breadcrumb (time + commit) so the
        # manifest records that later phases may run at a different commit.
        if not path.is_file():
            print(f"no manifest to resume: {path}", file=sys.stderr)
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("resumes", []).append({
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "git_dirty": bool(git("status", "--porcelain")),
        })
        data["status"] = "running"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["status"] = args.status
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

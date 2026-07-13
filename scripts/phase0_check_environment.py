"""Phase 0 environment and artifact checks."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = {
    "NSRS/weights/10M.ckpt": ROOT / "NSRS" / "weights" / "10M.ckpt",
    "TPSR/symbolicregression/weights/model1.pt": ROOT
    / "TPSR"
    / "symbolicregression"
    / "weights"
    / "model1.pt",
    "TPSR/nesymres/weights/10M.ckpt": ROOT / "TPSR" / "nesymres" / "weights" / "10M.ckpt",
}


def check_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - diagnostic script
        return f"FAIL ({exc})"
    version = getattr(module, "__version__", "ok")
    return str(version)


def main() -> int:
    print("=== Phase 0 environment check ===")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Executable: {sys.executable}")

    for pkg in ("torch", "pytorch_lightning", "sympy", "numpy", "omegaconf"):
        print(f"{pkg}: {check_import(pkg)}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as exc:
        print(f"CUDA check failed: {exc}")

    print("\n=== Checkpoint files ===")
    ok = True
    for label, path in CHECKPOINTS.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"OK  {label} ({size_mb:.1f} MB)")
        else:
            ok = False
            print(f"MISSING  {label}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

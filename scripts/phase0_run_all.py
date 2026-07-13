"""Run all Phase 0 Python smoke tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "phase0_check_environment.py",
    "phase0_nesymres_smoke.py",
    "phase0_pysr_smoke.py",
    "phase0_tpsr_smoke.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    failed = []

    for name in SCRIPTS:
        path = root / name
        print(f"\n{'=' * 60}\nRUN {name}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(path)], cwd=root.parent)
        if result.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All Phase 0 smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

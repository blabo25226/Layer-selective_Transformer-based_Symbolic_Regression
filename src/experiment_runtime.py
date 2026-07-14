"""Paths and metadata shared by reproducible experiment scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


def phase_output_paths(
    root: Path, phase: str, legacy_report_name: str
) -> Tuple[Path, Path]:
    """Return run-scoped paths when LTSR_RUN_DIR is set, else legacy paths."""
    run_dir = os.environ.get("LTSR_RUN_DIR")
    tag = os.environ.get("LTSR_PHASE_TAG", "").strip()
    if tag:
        phase = f"{phase}_{tag}"
        report_path = Path(legacy_report_name)
        legacy_report_name = f"{report_path.stem}_{tag}{report_path.suffix}"
    if run_dir:
        base = Path(run_dir).expanduser().resolve()
        out, report = base / phase, base / "reports" / legacy_report_name
        report.parent.mkdir(parents=True, exist_ok=True)
        return out, report
    return (
        root / "results" / "phase_results" / phase,
        root / "results" / "phase_results" / legacy_report_name,
    )

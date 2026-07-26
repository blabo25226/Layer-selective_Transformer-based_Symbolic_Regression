"""Paths and metadata shared by reproducible experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


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


def load_phase4_seed_checkpoint(
    out_dir: Path, seed: int
) -> tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Any]],
] | None:
    """Load a complete Phase-4 seed checkpoint, or return ``None``.

    All six audit files must exist and parse before a Colab resume may skip the
    seed. The caller rebuilds cross-seed aggregates from the returned records.
    """
    paths = {
        "contrib": out_dir / f"contrib_seed{seed}.json",
        "raw_scores": out_dir / f"raw_scores_seed{seed}.json",
        "absolute": out_dir / f"absolute_improvements_seed{seed}.json",
        "status": out_dir / f"contribution_status_seed{seed}.json",
        "tuning": out_dir / f"tuning_seed{seed}.json",
        "equations": out_dir / f"equations_seed{seed}.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        payloads = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
    except (OSError, json.JSONDecodeError):
        return None
    if not all(isinstance(payloads[name], dict) for name in paths):
        return None
    absolute = {
        metric: {
            layer: value for layer, value in table.items() if layer != "all_params"
        }
        for metric, table in payloads["absolute"].items()
        if isinstance(table, dict)
    }
    return payloads["contrib"], absolute, payloads["status"]

"""Aggregate Phase 6 noise-sweep metrics across paired seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.generalization import _ci95  # noqa: E402


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        path = args.run_dir / f"phase6_noise_seed{seed}" / "noise_sweep.json"
        if not path.is_file():
            parser.error(f"missing Phase 6 result: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    noises = sorted(set.intersection(*(set(run) for run in runs)), key=float)
    cells = sorted(set.intersection(*(set(run[noises[0]]) for run in runs)))
    summary = {}
    for noise in noises:
        summary[noise] = {}
        for cell in cells:
            summary[noise][cell] = {}
            for metric in ("penalized_nmse", "valid_rate", "complexity", "sym_rate"):
                values = [float(run[noise][cell][metric]) for run in runs]
                summary[noise][cell][metric] = {**_ci95(values), "values": values}
    out = {"seeds": args.seeds, "summary": summary}
    out_dir = args.run_dir / "phase6_noise_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(safe(out), indent=2), encoding="utf-8")
    lines = ["# Phase 6 multi-seed noise summary", "", f"- Seeds: {args.seeds}", ""]
    for noise in noises:
        lines += [f"## Noise {noise}", "",
                  "| condition | penalized NMSE | 95% t-CI | valid | complexity |",
                  "|---|---:|---:|---:|---:|"]
        for cell in cells:
            s = summary[noise][cell]
            lines.append(
                f"| `{cell}` | {s['penalized_nmse']['mean']:.4g} | "
                f"±{s['penalized_nmse']['ci95']:.4g} | {s['valid_rate']['mean']:.3f} | "
                f"{s['complexity']['mean']:.3f} |"
            )
        lines.append("")
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase6_noise_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

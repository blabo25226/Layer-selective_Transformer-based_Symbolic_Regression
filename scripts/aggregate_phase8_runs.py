"""Aggregate LODO application results across fine-tuning/decode seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.generalization import _ci95  # noqa: E402


def safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        path = args.run_dir / f"phase8_lodo_seed{seed}" / "lodo_results.json"
        if not path.is_file():
            parser.error(f"missing Phase 8 result: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    methods = sorted(set.intersection(*(set(run["aggregate"]) for run in runs)))
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in ("mean_in", "mean_hold", "gap_mean"):
            values = [
                float(run["aggregate"][method][metric])
                if run["aggregate"][method].get(metric) is not None else float("nan")
                for run in runs
            ]
            summary[method][metric] = {**_ci95(values), "values": values}
        for metric in ("hold_valid_rate", "hold_near_singularity", "hold_extrapolation_valid"):
            values = []
            for run in runs:
                fold_values = []
                for fold in run["per_fold"]:
                    if method not in fold:
                        continue
                    value = fold[method].get(metric)
                    fold_values.append(float(value) if value is not None else float("nan"))
                finite = [v for v in fold_values if math.isfinite(v)]
                values.append(sum(finite) / len(finite) if finite else float("nan"))
            summary[method][metric] = {**_ci95(values), "values": values}
    output = {"seeds": args.seeds, "summary": summary}
    out_dir = args.run_dir / "phase8_lodo_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(safe(output), indent=2), encoding="utf-8")
    lines = [
        "# Phase 8 LODO multi-seed summary", "", f"- Seeds: {args.seeds}",
        "- Donor folds are averaged within each seed; the t-CI is across training/decode seeds.", "",
        "| method | holdout NMSE | 95% t-CI | gap | valid | singularity rate | extrapolation valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| `{method}` | {row['mean_hold']['mean']:.4g} | ±{row['mean_hold']['ci95']:.4g} | "
            f"{row['gap_mean']['mean']:.4g} | {row['hold_valid_rate']['mean']:.3f} | "
            f"{row['hold_near_singularity']['mean']:.3f} | {row['hold_extrapolation_valid']['mean']:.3f} |"
        )
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase8_lodo_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

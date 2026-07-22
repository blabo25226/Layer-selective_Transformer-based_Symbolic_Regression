"""Aggregate DREAM4 Size10/100 results across paired training/split seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.generalization import _ci95  # noqa: E402


def finite_mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


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
    output = {"seeds": args.seeds, "sizes": {}}
    lines = ["# Phase 7 DREAM4 multi-seed summary", "", f"- Seeds: {args.seeds}", ""]
    for size in (10, 100):
        runs = []
        for seed in args.seeds:
            path = args.run_dir / f"phase7_dream4_size{size}_seed{seed}" / f"size{size}_results.json"
            if not path.is_file():
                parser.error(f"missing Phase 7 result: {path}")
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        conditions = sorted(set.intersection(*(
            {condition for net in run["sr"].values() for condition in net}
            for run in runs
        )))
        metrics = (
            "penalized_nmse", "valid_rate", "complexity",
            "near_singularity_mean", "extrapolation_valid_mean",
        )
        summary = {}
        for condition in conditions:
            summary[condition] = {}
            for metric in metrics:
                seed_means = [finite_mean(
                    net[condition]["aggregate"].get(metric)
                    for net in run["sr"].values() if condition in net
                ) for run in runs]
                summary[condition][metric] = {**_ci95(seed_means), "values": seed_means}
        selection = {}
        selection_methods = sorted(set.intersection(*(
            {method for net in run["selection"].values() for method in net}
            for run in runs
        )))
        for method in selection_methods:
            seed_means = [finite_mean(
                net[method]["edge_recovery"]["f1"]
                for net in run["selection"].values() if method in net
            ) for run in runs]
            selection[method] = {**_ci95(seed_means), "values": seed_means}
        output["sizes"][str(size)] = {"sr": summary, "selection_edge_f1": selection}
        lines += [
            f"## Size {size}", "",
            "SR values first average over the five fixed networks within each seed; the t-CI is then across seeds.",
            "",
            "| condition | penalized NMSE | 95% t-CI | valid | complexity | singularity rate | extrapolation valid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for condition, row in summary.items():
            lines.append(
                f"| `{condition}` | {row['penalized_nmse']['mean']:.4g} | ±{row['penalized_nmse']['ci95']:.4g} | "
                f"{row['valid_rate']['mean']:.3f} | {row['complexity']['mean']:.3f} | "
                f"{row['near_singularity_mean']['mean']:.3f} | {row['extrapolation_valid_mean']['mean']:.3f} |"
            )
        lines += ["", "| selector | mean edge F1 | 95% t-CI |", "|---|---:|---:|"]
        for method, stats in selection.items():
            lines.append(f"| `{method}` | {stats['mean']:.4g} | ±{stats['ci95']:.4g} |")
        lines.append("")
    out_dir = args.run_dir / "phase7_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(safe(output), indent=2), encoding="utf-8")
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase7_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

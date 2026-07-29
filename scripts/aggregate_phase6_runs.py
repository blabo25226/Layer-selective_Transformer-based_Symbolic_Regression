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


def _num(value):
    """Coerce to float, or None for missing/non-finite metrics.

    A cell with no valid predictions reports metrics such as ``complexity`` as
    null; _ci95 already drops None/non-finite entries, so keep them as None here
    instead of crashing on float(None).
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


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
            for metric in (
                "penalized_nmse", "valid_rate", "complexity", "sym_rate",
                "elapsed_sec", "near_singularity_mean", "extrapolation_valid_mean",
            ):
                values = [_num(run[noise][cell].get(metric)) for run in runs]
                summary[noise][cell][metric] = {**_ci95(values), "values": values}
    effects = {}
    required = {
        "pretrained_beam", "pretrained_tpsr", "selective_beam", "selective_tpsr"
    }
    if required.issubset(cells):
        for noise in noises:
            per_seed = []
            for run in runs:
                score = lambda cell: float(run[noise][cell]["penalized_nmse"])
                tpsr_pre = score("pretrained_beam") - score("pretrained_tpsr")
                tpsr_sel = score("selective_beam") - score("selective_tpsr")
                per_seed.append({
                    "ft_effect_beam": score("pretrained_beam") - score("selective_beam"),
                    "ft_effect_tpsr": score("pretrained_tpsr") - score("selective_tpsr"),
                    "tpsr_effect_pretrained": tpsr_pre,
                    "tpsr_effect_selective": tpsr_sel,
                    "interaction": tpsr_sel - tpsr_pre,
                })
            effects[noise] = {
                key: {**_ci95([row[key] for row in per_seed]), "values": [row[key] for row in per_seed]}
                for key in per_seed[0]
            }
    out = {"seeds": args.seeds, "summary": summary, "paired_effects": effects}
    out_dir = args.run_dir / "phase6_noise_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(safe(out), indent=2), encoding="utf-8")
    lines = ["# Phase 6 multi-seed noise summary", "", f"- Seeds: {args.seeds}", ""]
    for noise in noises:
        lines += [f"## Noise {noise}", "",
                  "| condition | penalized NMSE | 95% t-CI | valid | complexity | time (s) | singularity | extrapolation valid |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cell in cells:
            s = summary[noise][cell]
            lines.append(
                f"| `{cell}` | {s['penalized_nmse']['mean']:.4g} | "
                f"±{s['penalized_nmse']['ci95']:.4g} | {s['valid_rate']['mean']:.3f} | "
                f"{s['complexity']['mean']:.3f} | {s['elapsed_sec']['mean']:.1f} | "
                f"{s['near_singularity_mean']['mean']:.3f} | "
                f"{s['extrapolation_valid_mean']['mean']:.3f} |"
            )
        lines.append("")
        if noise in effects:
            lines += [
                "Paired effects use positive values for improvement in failure-penalized NMSE.",
                "",
                "| effect | mean | 95% t-CI |",
                "|---|---:|---:|",
            ]
            for name, stats in effects[noise].items():
                lines.append(
                    f"| `{name}` | {stats['mean']:.4g} | ±{stats['ci95']:.4g} |"
                )
            lines.append("")
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase6_noise_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

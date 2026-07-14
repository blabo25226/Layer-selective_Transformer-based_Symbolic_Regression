"""Aggregate Phase 5 condition scores across paired training seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.generalization import _ci95  # noqa: E402


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    per_seed = []
    for seed in args.seeds:
        path = args.run_dir / f"phase5_seed{seed}" / "selective_results.json"
        if not path.is_file():
            parser.error(f"missing Phase 5 result: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        per_seed.append({r["condition"]: r for r in rows})

    conditions = sorted(set.intersection(*(set(run) for run in per_seed)))
    summary = {}
    for condition in conditions:
        summary[condition] = {}
        for metric in ("penalized_nmse", "penalized_r2", "valid_rate", "complexity"):
            values = [float(run[condition]["eval"][metric]) for run in per_seed]
            stats = _ci95(values)
            stats["values"] = values
            summary[condition][metric] = stats

    top, random = f"top_{args.k}", f"random_{args.k}"
    paired_delta = []
    if top in summary and random in summary:
        paired_delta = [
            per_seed[i][random]["eval"]["penalized_nmse"]
            - per_seed[i][top]["eval"]["penalized_nmse"]
            for i in range(len(per_seed))
        ]
    output = {
        "seeds": args.seeds,
        "conditions": summary,
        "top_minus_random_improvement": {**_ci95(paired_delta), "values": paired_delta},
    }
    out_dir = args.run_dir / "phase5_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(json_safe(output), indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Phase 5 multi-seed summary", "", f"- Seeds: {args.seeds}",
        "- Primary metric: failure-penalized NMSE", "",
        "| condition | mean | 95% t-CI | valid rate | complexity |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        s = summary[condition]
        lines.append(
            f"| `{condition}` | {s['penalized_nmse']['mean']:.4g} | "
            f"±{s['penalized_nmse']['ci95']:.4g} | {s['valid_rate']['mean']:.3f} | "
            f"{s['complexity']['mean']:.3f} |"
        )
    delta = output["top_minus_random_improvement"]
    lines += ["", f"Paired random-minus-top NMSE improvement: "
                    f"{delta['mean']:.4g} ± {delta['ci95']:.4g} (95% t-CI)."]
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase5_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

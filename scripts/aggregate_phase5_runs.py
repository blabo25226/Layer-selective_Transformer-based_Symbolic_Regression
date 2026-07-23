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


def paired_effect(differences, *, direction: str):
    stats = _ci95(differences)
    half = stats["ci95"]
    stats.update({
        "values": differences,
        "direction": direction,
        "ci_low": stats["mean"] - half if math.isfinite(half) else float("nan"),
        "ci_high": stats["mean"] + half if math.isfinite(half) else float("nan"),
    })
    return stats


def equivalence_decision(top_minus_full, margin: float):
    effect = paired_effect(
        top_minus_full,
        direction="top penalized NMSE minus full-FT penalized NMSE; negative favors top",
    )
    low, high = effect["ci_low"], effect["ci_high"]
    finite = math.isfinite(low) and math.isfinite(high)
    equivalent = finite and low > -margin and high < margin
    noninferior = finite and high <= margin
    if equivalent:
        conclusion = "equivalent_within_margin"
    elif finite and high < 0:
        conclusion = "top_superior"
    elif finite and low > 0:
        conclusion = "full_superior"
    elif noninferior:
        conclusion = "top_noninferior_but_not_equivalent"
    else:
        conclusion = "inconclusive"
    return {
        **effect,
        "margin": margin,
        "equivalent": equivalent,
        "top_noninferior": noninferior,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--nmse-equivalence-margin", type=float, default=0.05,
        help="Predeclared absolute margin on failure-penalized NMSE",
    )
    args = parser.parse_args()
    if args.nmse_equivalence_margin <= 0:
        parser.error("--nmse-equivalence-margin must be positive")
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
        for metric in (
            "penalized_nmse", "penalized_r2", "valid_rate", "complexity",
            "near_singularity_mean", "extrapolation_valid_mean",
        ):
            values = [float(run[condition]["eval"][metric]) for run in per_seed]
            stats = _ci95(values)
            stats["values"] = values
            summary[condition][metric] = stats
        for metric in ("elapsed_sec", "peak_mem_mb"):
            values = [float(run[condition][metric]) for run in per_seed]
            summary[condition][metric] = {**_ci95(values), "values": values}

    random_conditions = sorted(
        name for name in conditions
        if name == f"random_{args.k}" or name.startswith(f"random_{args.k}_seed")
    )
    random_mean_by_seed = [
        sum(float(run[name]["eval"]["penalized_nmse"]) for name in random_conditions)
        / len(random_conditions)
        for run in per_seed
    ] if random_conditions else []
    top_conditions = sorted(name for name in conditions if name.startswith("top_"))
    paired_comparisons = {}
    equivalence = {}
    for top in top_conditions:
        paired_comparisons[top] = {}
        comparators = [
            name for name in ("pretrained", "all_params", f"middle_{args.k}", f"bottom_{args.k}")
            if name in conditions
        ]
        for comparator in comparators:
            differences = [
                float(per_seed[i][top]["eval"]["penalized_nmse"])
                - float(per_seed[i][comparator]["eval"]["penalized_nmse"])
                for i in range(len(per_seed))
            ]
            paired_comparisons[top][comparator] = paired_effect(
                differences,
                direction="top minus comparator failure-penalized NMSE; negative favors top",
            )
        if random_mean_by_seed:
            differences = [
                float(per_seed[i][top]["eval"]["penalized_nmse"]) - random_mean_by_seed[i]
                for i in range(len(per_seed))
            ]
            paired_comparisons[top]["random_set_mean"] = paired_effect(
                differences,
                direction="top minus within-training-seed mean random-set NMSE; negative favors top",
            )
        if "all_params" in conditions:
            equivalence[top] = equivalence_decision(
                [
                    float(per_seed[i][top]["eval"]["penalized_nmse"])
                    - float(per_seed[i]["all_params"]["eval"]["penalized_nmse"])
                    for i in range(len(per_seed))
                ],
                args.nmse_equivalence_margin,
            )
    primary_top = f"top_{args.k}"
    paired_delta = []
    if primary_top in paired_comparisons and "random_set_mean" in paired_comparisons[primary_top]:
        paired_delta = [
            -value for value in paired_comparisons[primary_top]["random_set_mean"]["values"]
        ]
    metric_directions = {
        "penalized_nmse": "top minus comparator; negative favors top",
        "penalized_r2": "top minus comparator; positive favors top",
        "valid_rate": "top minus comparator; positive favors top",
        "complexity": "top minus comparator; negative means a simpler top-layer equation",
    }
    paired_metric_comparisons = {}
    for top in top_conditions:
        paired_metric_comparisons[top] = {}
        comparator_names = [
            name for name in ("pretrained", "all_params", f"middle_{args.k}", f"bottom_{args.k}")
            if name in conditions
        ]
        if random_conditions:
            comparator_names.append("random_set_mean")
        for comparator in comparator_names:
            paired_metric_comparisons[top][comparator] = {}
            for metric, direction in metric_directions.items():
                if comparator == "random_set_mean":
                    comparator_values = [
                        sum(float(per_seed[i][name]["eval"][metric]) for name in random_conditions)
                        / len(random_conditions)
                        for i in range(len(per_seed))
                    ]
                else:
                    comparator_values = [
                        float(per_seed[i][comparator]["eval"][metric])
                        for i in range(len(per_seed))
                    ]
                differences = [
                    float(per_seed[i][top]["eval"][metric]) - comparator_values[i]
                    for i in range(len(per_seed))
                ]
                paired_metric_comparisons[top][comparator][metric] = paired_effect(
                    differences, direction=direction
                )
    output = {
        "seeds": args.seeds,
        "conditions": summary,
        "random_layer_conditions": random_conditions,
        "random_mean_nmse_by_training_seed": random_mean_by_seed,
        "paired_comparisons": paired_comparisons,
        "paired_metric_comparisons": paired_metric_comparisons,
        "top_vs_full_equivalence": equivalence,
        "nmse_equivalence_margin": args.nmse_equivalence_margin,
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
        "| condition | mean | 95% t-CI | valid | complexity | time (s) | peak MB | singularity | extrapolation valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        s = summary[condition]
        lines.append(
            f"| `{condition}` | {s['penalized_nmse']['mean']:.4g} | "
            f"±{s['penalized_nmse']['ci95']:.4g} | {s['valid_rate']['mean']:.3f} | "
            f"{s['complexity']['mean']:.3f} | {s['elapsed_sec']['mean']:.1f} | "
            f"{s['peak_mem_mb']['mean']:.1f} | {s['near_singularity_mean']['mean']:.3f} | "
            f"{s['extrapolation_valid_mean']['mean']:.3f} |"
        )
    delta = output["top_minus_random_improvement"]
    lines += ["", f"Paired random-minus-top NMSE improvement: "
                    f"{delta['mean']:.4g} ± {delta['ci95']:.4g} (95% t-CI)."]
    lines += [
        "",
        "## Paired top-layer comparisons",
        "",
        f"Predeclared equivalence margin: ±{args.nmse_equivalence_margin:g} failure-penalized NMSE.",
        "",
        "| top condition | comparator | mean(top-comparator) | 95% t-CI | conclusion |",
        "|---|---|---:|---:|---|",
    ]
    for top, comparisons in paired_comparisons.items():
        for comparator, effect in comparisons.items():
            conclusion = (
                equivalence.get(top, {}).get("conclusion", "")
                if comparator == "all_params" else ""
            )
            lines.append(
                f"| `{top}` | `{comparator}` | {effect['mean']:.4g} | "
                f"[{effect['ci_low']:.4g}, {effect['ci_high']:.4g}] | {conclusion} |"
            )
    reports = args.run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "phase5_multiseed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-seed Phase 4: layer contribution with mean ± CI across seeds (A-1).

Single-seed layer contribution on a handful of equations cannot support claims
of layer selectivity (reviewer note A-1). This runs the Phase 4 scan for several
seeds — each seed reshuffles training, resamples points, and re-draws dropout —
and reports, per metric and per layer, the mean contribution, its 95% CI, and a
ranking-stability score (how often the layer lands in the metric's top-3).

Intended to run in the NeSymReS-compatible environment (Colab), pointed at the
diverse suite:

    python scripts/generate_diverse_suite.py --n-per-skeleton 8
    python scripts/phase4_multiseed.py --data-dir results/synthetic/diverse_v1 \
        --seeds 0 1 2 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.finetune_dataset import (  # noqa: E402
    GRNFinetuneDataset,
    collate_finetune,
    load_split_problems,
)
from data.splits import split_synthetic_train_validation  # noqa: E402
from evaluation.generalization import _ci95  # noqa: E402
from evaluation.layer_contribution import (  # noqa: E402
    absolute_improvements,
    compute_contributions,
    rank_by_contribution,
    reference_improves,
)
from models.nesymres_adapter import load_nesymres  # noqa: E402
from training.single_layer import clone_model  # noqa: E402
from training.tuning import build_config_grid, seed_everything, tune_selective  # noqa: E402
from experiment_runtime import phase_output_paths  # noqa: E402

# Reuse the single-seed Phase 4 building blocks.
from phase4_layer_contribution import (  # noqa: E402
    WEIGHTS,
    CONFIG,
    EQ_SETTING,
    build_phase4_conditions,
    eval_ce_loss,
    eval_problems,
    make_eval_fit_params,
)

OUT_DIR, REPORT = phase_output_paths(ROOT, "phase4_multiseed", "phase4_multiseed_report.md")

# Metrics: (key, higher_is_better)
METRICS = [
    ("val_ce", False),
    ("penalized_nmse", False),
    ("penalized_r2", True),
    ("var_f1", True),
    ("sym_rate", True),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x: float, d: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.{d}g}"


def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def run_one_seed(
    base_model,
    fit_eval,
    word2id,
    train_problems,
    validation_problems,
    args,
    seed: int,
    device,
) -> Dict[str, Any]:
    """Return raw scores, guarded contributions, and tuning records for one seed."""
    train_ds = GRNFinetuneDataset(train_problems, word2id, max_points=args.max_points, seed=seed)
    val_ds = GRNFinetuneDataset(validation_problems, word2id, max_points=args.max_points, seed=seed + 1000)
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(len(val_ds), 1)),
        shuffle=False,
        collate_fn=collate_finetune,
    )

    conditions = build_phase4_conditions(base_model)
    scores_by_metric: Dict[str, Dict[str, float]] = {m: {} for m, _ in METRICS}
    tuning_by_condition: Dict[str, Dict[str, object]] = {}
    configs = build_config_grid(
        args.lr_grid or [args.lr],
        args.epoch_grid or [args.epochs],
        patience=args.patience,
        min_delta=args.min_delta,
    )

    for name, layers in conditions.items():
        def train_loader_factory():
            return DataLoader(
                train_ds,
                batch_size=min(args.batch_size, len(train_ds)),
                shuffle=True,
                collate_fn=collate_finetune,
                generator=torch.Generator().manual_seed(seed),
            )

        if name == "pretrained" or layers == []:
            seed_everything(seed)
            model = clone_model(base_model)
        else:
            model, tuning = tune_selective(
                base_model,
                train_loader_factory,
                val_loader,
                layers,
                configs,
                device=device,
                seed=seed,
            )
            tuning_by_condition[name] = tuning
        model.eval()
        val_ce = eval_ce_loss(model, val_loader, device) if len(val_ds) else float("nan")
        seed_everything(seed + 10_000)
        agg = eval_problems(model, fit_eval, validation_problems)["aggregate"]
        agg["val_ce"] = val_ce
        for m, _ in METRICS:
            scores_by_metric[m][name] = float(agg.get(m, float("nan")))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    contrib: Dict[str, Dict[str, float]] = {}
    improvements: Dict[str, Dict[str, float]] = {}
    statuses: Dict[str, Dict[str, object]] = {}
    for m, higher in METRICS:
        scores = scores_by_metric[m]
        improvements[m] = absolute_improvements(scores, higher_is_better=higher)
        base = float(scores.get("pretrained", float("nan")))
        full = float(scores.get("all_params", float("nan")))
        valid_reference = reference_improves(base, full, higher_is_better=higher)
        statuses[m] = {
            "base": base,
            "full": full,
            "higher_is_better": higher,
            "full_improves_base": valid_reference,
        }
        try:
            contrib[m] = compute_contributions(scores, higher_is_better=higher)
        except KeyError:
            contrib[m] = {}
    return {
        "raw_scores": scores_by_metric,
        "absolute_improvements": improvements,
        "contributions": contrib,
        "contribution_status": statuses,
        "tuning": tuning_by_condition,
    }


def aggregate(per_seed: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """{metric: {layer: {mean, std, sem, ci95, n, top3_frac}}}."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    metrics = per_seed[0].keys() if per_seed else []
    for m in metrics:
        layers = sorted({l for s in per_seed for l in s.get(m, {})})
        # top-3 membership per seed for stability
        top3_counts = {l: 0 for l in layers}
        for s in per_seed:
            ranked = [n for n, _ in rank_by_contribution(s.get(m, {}))][:3]
            for l in ranked:
                top3_counts[l] = top3_counts.get(l, 0) + 1
        stats: Dict[str, Dict[str, float]] = {}
        for l in layers:
            vals = [
                s[m][l]
                for s in per_seed
                if l in s.get(m, {}) and s[m][l] is not None and np.isfinite(s[m][l])
            ]
            if not vals:
                stats[l] = {"mean": float("nan"), "std": float("nan"), "sem": float("nan"),
                            "ci95": float("nan"), "n": 0.0, "top3_frac": 0.0}
                continue
            arr = np.array(vals, dtype=float)
            n = len(arr)
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            sem = std / math.sqrt(n) if n > 0 else float("nan")
            ci = _ci95(arr.tolist())
            stats[l] = {
                "mean": float(arr.mean()),
                "std": std,
                "sem": float(sem),
                "ci95": float(ci["ci95"]),
                "ci_method": "student_t",
                "n": float(n),
                "top3_frac": top3_counts.get(l, 0) / len(per_seed),
            }
        out[m] = stats
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-dir", default=str(ROOT / "results" / "synthetic" / "diverse_v1"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-grid", type=float, nargs="+", default=None)
    parser.add_argument("--epoch-grid", type=int, nargs="+", default=None)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    log(f"Device: {device} | data: {data_dir} | seeds: {args.seeds}")

    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size)
    fit_eval = make_eval_fit_params(params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time)
    word2id = json.loads(EQ_SETTING.read_text(encoding="utf-8"))["word2id"]

    all_train_problems = load_split_problems(data_dir, "train")
    train_problems, validation_problems = split_synthetic_train_validation(
        all_train_problems,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    if args.eval_limit > 0:
        validation_problems = validation_problems[: args.eval_limit]
    log(f"train={len(train_problems)} validation={len(validation_problems)} (test unused)")

    per_seed: List[Dict[str, Dict[str, float]]] = []
    status_per_seed: List[Dict[str, Dict[str, object]]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        log(f"\n=== seed {seed} ===")
        run = run_one_seed(
            base_model, fit_eval, word2id, train_problems, validation_problems, args, seed, device
        )
        contrib = run["contributions"]
        per_seed.append(contrib)
        status_per_seed.append(run["contribution_status"])
        (OUT_DIR / f"contrib_seed{seed}.json").write_text(
            json.dumps(_sanitize(contrib), indent=2), encoding="utf-8"
        )
        for key, filename in (
            ("raw_scores", f"raw_scores_seed{seed}.json"),
            ("absolute_improvements", f"absolute_improvements_seed{seed}.json"),
            ("contribution_status", f"contribution_status_seed{seed}.json"),
            ("tuning", f"tuning_seed{seed}.json"),
        ):
            (OUT_DIR / filename).write_text(
                json.dumps(_sanitize(run[key]), indent=2), encoding="utf-8"
            )
        log(f"  seed {seed} done ({time.time() - t0:.1f}s)")

    agg = aggregate(per_seed)
    reference_summary = {
        metric: {
            "valid_seed_count": sum(
                bool(status.get(metric, {}).get("full_improves_base", False))
                for status in status_per_seed
            ),
            "seed_count": len(status_per_seed),
        }
        for metric, _ in METRICS
    }
    # Do not select layers from a normalized metric unless full FT is an
    # improving reference in every paired seed. Partial-seed ranking would be
    # biased toward seeds where the denominator happened to be valid.
    for metric, summary in reference_summary.items():
        if summary["valid_seed_count"] != summary["seed_count"]:
            for stats in agg.get(metric, {}).values():
                stats.update({
                    "mean": float("nan"),
                    "std": float("nan"),
                    "sem": float("nan"),
                    "ci95": float("nan"),
                    "n": 0.0,
                    "top3_frac": 0.0,
                })
    (OUT_DIR / "contrib_aggregate.json").write_text(
        json.dumps(_sanitize(agg), indent=2), encoding="utf-8"
    )
    (OUT_DIR / "contribution_status_aggregate.json").write_text(
        json.dumps(reference_summary, indent=2), encoding="utf-8"
    )

    # Report
    lines = [
        "# Phase 4 (multi-seed): layer contribution with CI",
        "",
        f"- Data: `{data_dir.as_posix()}` (train {len(train_problems)} / validation {len(validation_problems)})",
        "- Validation is held out by motif; the test split is not used for layer selection.",
        f"- Seeds: {args.seeds}",
        f"- Validation tuning grid: lr={args.lr_grid or [args.lr]}, "
        f"epochs={args.epoch_grid or [args.epochs]}, patience={args.patience}",
        f"- Device: `{device}`",
        "",
        "Contribution `C=1` means the single layer recovers the full-FT gain; `C=0` "
        "means no better than pretrained. **A layer is only 'high-contribution' if its "
        "CI stays clearly above the random/other layers across seeds.**",
        "If full FT does not improve on pretrained for a metric, normalized contribution "
        "is recorded as undefined; use the saved absolute improvements instead.",
        "",
    ]
    lines += [
        "## Full-FT reference validity",
        "",
        "| metric | seeds where full improves base | total seeds |",
        "|---|---:|---:|",
    ]
    for metric, _ in METRICS:
        summary = reference_summary[metric]
        lines.append(
            f"| `{metric}` | {summary['valid_seed_count']} | {summary['seed_count']} |"
        )
    lines.append("")
    for m, higher in METRICS:
        stats = agg.get(m, {})
        # order by mean desc (NaN last)
        order = sorted(
            stats.items(),
            key=lambda kv: (1, 0.0) if math.isnan(kv[1]["mean"]) else (0, -kv[1]["mean"]),
        )
        lines += [
            f"## {m}",
            "",
            "| rank | layer | mean C | 95% CI | std | top-3 stability |",
            "|------|-------|--------|--------|-----|-----------------|",
        ]
        for i, (layer, s) in enumerate(order, 1):
            lines.append(
                f"| {i} | `{layer}` | {fmt(s['mean'])} | ±{fmt(s['ci95'])} | "
                f"{fmt(s['std'])} | {s['top3_frac']*100:.0f}% |"
            )
        lines.append("")

    lines += [
        "## How to read this",
        "",
        "- **Overlapping CIs** between the top layer and mid/random layers ⇒ H2 "
        "(layer selectivity) is **not** supported for that metric.",
        "- **top-3 stability < ~66%** ⇒ the 'high-contribution' layer is seed-dependent, "
        "not a property of the network.",
        "- Compare the CE ranking (decoder-heavy) vs prediction rankings (encoder-heavy): "
        "if they disagree, report them separately (plan §Phase 4).",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {OUT_DIR / 'contrib_aggregate.json'}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

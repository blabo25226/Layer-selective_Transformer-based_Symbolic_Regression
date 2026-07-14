"""Phase 8 deep-dive: leave-one-donor-out generalization (reviewer note).

The headline Phase 8 result — selective-FT NeSymReS holding up on a held-out donor
while PySR overfits in-donor — was based on a *single* holdout donor. This runs
leave-one-donor-out (LODO): each donor is held out in turn, and the
**generalization gap** (holdout NMSE − in-donor NMSE) is aggregated across folds
with a 95% CI, so the claim rests on cross-validation rather than one split.

The dreamlike selective fine-tune is donor-independent, so it is done once and
reused across folds. Runs in the NeSymReS environment (Colab):

    python scripts/phase8_lodo.py --epochs 5            # NeSymReS methods only
    python scripts/phase8_lodo.py --epochs 5 --with-pysr --pysr-iters 12
"""

from __future__ import annotations

import argparse
import json
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

from data.dreamlike_grn import build_local_problem  # noqa: E402
from data.finetune_dataset import GRNFinetuneDataset, collate_finetune  # noqa: E402
from data.human import build_human_local_problems, prepare_gse112372  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from evaluation.generalization import aggregate_lodo, rank_by_generalization  # noqa: E402
from models.nesymres_adapter import load_nesymres  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402
from experiment_runtime import phase_output_paths  # noqa: E402

# Reuse Phase 8 building blocks unchanged.
from phase8_run_human import (  # noqa: E402
    WEIGHTS,
    CONFIG,
    EQ_SETTING,
    DATA_DIR,
    HIGH_CONTRIB,
    build_dreamlike_ft,
    eval_holdout_donor,
    eval_sr,
    fmt,
    make_light_fit,
    run_pysr,
    stack_donor_derivatives,
)

OUT_DIR, REPORT = phase_output_paths(ROOT, "phase8_lodo", "phase8_lodo_report.md")


def log(msg: str) -> None:
    print(msg, flush=True)


def pysr_fold(prior_probs, prior_sel, panel, X_te, Y_te, max_vars, niters) -> Dict[str, float]:
    """PySR in-donor + holdout NMSE for one fold (mirrors phase8 main)."""
    network = panel.as_grn_like()
    in_rows, hold_rows = [], []
    for ds in prior_probs:
        try:
            expr = run_pysr(ds.X, ds.y, ds.spec.variable_names, niters)
        except Exception:
            expr = ""
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(ds.y, y_hat, expr, ds.spec.variable_names, true_expr="")
        in_rows.append(sc)
        motif = ds.spec.motif or ""
        tname = motif.split("target=")[-1].split(";")[0] if "target=" in motif else ""
        if tname in panel.gene_names:
            t = panel.gene_index(tname)
            te = build_local_problem(
                network, X_te, Y_te[:, t], t, prior_sel.get(t, []),
                eq_id=f"{ds.spec.eq_id}_holdout", split="holdout",
                max_vars=max_vars, selection_method="prior",
            )
            y_hat_te = eval_expression(expr, te.X, te.spec.variable_names) if expr else None
            sc_te = score_prediction(te.y, y_hat_te, expr, te.spec.variable_names, true_expr="")
            hold_rows.append(sc_te)
    in_agg = aggregate_prediction_scores(in_rows)
    hold_agg = aggregate_prediction_scores(hold_rows)
    return {
        "in": in_agg["penalized_nmse"],
        "hold": hold_agg["penalized_nmse"],
        "in_valid_rate": in_agg["valid_rate"],
        "hold_valid_rate": hold_agg["valid_rate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--derivative", default="smooth_fd", choices=["smooth_fd", "fd", "spline"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-vars", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--with-pysr", action="store_true")
    parser.add_argument("--pysr-iters", type=int, default=12)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    panel = prepare_gse112372(args.data_dir, force_download=args.force_download)
    donors = sorted(panel.X_donors.keys())
    log(f"donors={donors} (LODO: {len(donors)} folds)")

    # Donor-independent selective FT (done once).
    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=1)
    fit = make_light_fit(params_fit)
    base_model.to(device)
    word2id = json.loads(EQ_SETTING.read_text(encoding="utf-8"))["word2id"]
    dl_ds = GRNFinetuneDataset(build_dreamlike_ft(k=2), word2id, max_points=args.max_points, seed=0)
    dl_loader = DataLoader(
        dl_ds, batch_size=min(args.batch_size, max(len(dl_ds), 1)),
        shuffle=True, collate_fn=collate_finetune,
    )
    dreamlike_model = clone_model(base_model)
    train_selective(dreamlike_model, dl_loader, HIGH_CONTRIB, epochs=args.epochs, lr=args.lr, device=device)
    log(f"Selective FT layers: {HIGH_CONTRIB}")

    folds: List[Dict[str, Dict[str, float]]] = []
    per_fold_detail: List[Dict[str, Any]] = []
    for hold in donors:
        t0 = time.time()
        train_donors = [d for d in donors if d != hold]
        X_tr, Y_tr = stack_donor_derivatives(panel, args.derivative, train_donors)
        X_te, Y_te = stack_donor_derivatives(panel, args.derivative, [hold])
        prior_probs, prior_sel, _ = build_human_local_problems(
            panel, X_tr, Y_tr, method="prior", k=args.k, max_vars=args.max_vars, split="train"
        )

        fold: Dict[str, Dict[str, float]] = {}
        for name, model in [
            ("pretrained_beam", clone_model(base_model)),
            ("selective_beam", dreamlike_model),
        ]:
            model.eval()
            inn = eval_sr(model, fit, prior_probs, decode="beam")
            hd = eval_holdout_donor(model, fit, panel, prior_probs, prior_sel, X_te, Y_te, decode="beam")
            fold[name] = {
                "in": inn["aggregate"]["penalized_nmse"],
                "hold": hd["aggregate"]["penalized_nmse"],
                "in_valid_rate": inn["aggregate"]["valid_rate"],
                "hold_valid_rate": hd["aggregate"]["valid_rate"],
            }

        if args.with_pysr:
            fold["pysr"] = pysr_fold(prior_probs, prior_sel, panel, X_te, Y_te, args.max_vars, args.pysr_iters)

        folds.append(fold)
        per_fold_detail.append({"holdout": hold, "n_targets": len(prior_probs), **fold})
        log(
            f"  holdout={hold}: "
            + "  ".join(f"{m}(in={fmt(v['in'])},hold={fmt(v['hold'])})" for m, v in fold.items())
            + f"  ({time.time()-t0:.1f}s)"
        )

    agg = aggregate_lodo(folds, metric="nmse", lower_better=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lodo_results.json").write_text(
        json.dumps({"per_fold": per_fold_detail, "aggregate": agg}, indent=2, default=lambda x: None),
        encoding="utf-8",
    )

    order = rank_by_generalization(agg, lower_better=True)
    lines = [
        "# Phase 8 LODO — cross-donor generalization",
        "",
        f"- Donors (folds): {donors}",
        f"- Derivative: `{args.derivative}`  |  selective layers: `{', '.join(HIGH_CONTRIB)}`",
        f"- PySR included: {args.with_pysr}",
        "",
        "Generalization gap = median holdout NMSE − median in-donor NMSE, averaged "
        "over LODO folds (positive = overfits to training donors).",
        "",
        "| method | mean in-NMSE | mean hold-NMSE | hold 95% CI | gap | gap 95% CI | folds |",
        "|--------|--------------|----------------|-------------|-----|------------|-------|",
    ]
    for m in order:
        s = agg[m]
        lines.append(
            f"| `{m}` | {fmt(s['mean_in'])} | {fmt(s['mean_hold'])} | ±{fmt(s['hold_ci95'])} | "
            f"{fmt(s['gap_mean'])} | ±{fmt(s['gap_ci95'])} | {int(s['n_folds'])} |"
        )

    # Head-to-head: selective_beam vs pysr on holdout (the key claim).
    if "pysr" in agg and "selective_beam" in agg:
        sb, ps = agg["selective_beam"], agg["pysr"]
        better = sb["mean_hold"] < ps["mean_hold"]
        lines += [
            "",
            "## Key claim: does selective-FT generalize better than PySR?",
            "",
            f"- PySR: in={fmt(ps['mean_in'])} → hold={fmt(ps['mean_hold'])} (gap {fmt(ps['gap_mean'])})",
            f"- selective_beam: in={fmt(sb['mean_in'])} → hold={fmt(sb['mean_hold'])} (gap {fmt(sb['gap_mean'])})",
            "",
            f"**Verdict:** selective-FT NeSymReS "
            f"{'generalizes better (lower holdout NMSE) than PySR' if better else 'does NOT beat PySR on holdout NMSE'}"
            " across LODO folds. Check whether the holdout-NMSE CIs overlap before "
            "claiming significance.",
        ]
    lines += [
        "",
        "> ⚠️ Derivatives are proxies (`smooth_fd`), not true time derivatives, so "
        "this measures cross-donor consistency of the fitted RHS, not recovery of a "
        "true ODE. Grow the donor/gene panel to tighten the CIs.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {OUT_DIR / 'lodo_results.json'}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 8: human LPS macrophage time-series (GSE112372).

Pipeline:
  1. Load curated inflammatory gene panel (auto-download if needed)
  2. Estimate dx/dt (smoothed FD; spline optional)
  3. Restrict candidates (prior / prior_corr / corr / mi)
  4. Local SR: pretrained / dreamlike-selective / PySR (+ optional TPSR)
  5. Report predictive NMSE and prior-edge consistency (no true-equation claim)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.dreamlike_grn import (  # noqa: E402
    build_local_problem,
    generate_dreamlike_dataset,
    load_expression,
    load_network,
)
from data.finetune_dataset import GRNFinetuneDataset, collate_finetune  # noqa: E402
from data.human import (  # noqa: E402
    build_human_local_problems,
    estimate_derivatives,
    prepare_gse112372,
    prior_edge_recovery,
    save_panel_summary,
)
from data.regulator_selection import oracle_regulators  # noqa: E402
from data.synthetic_grn import SampledDataset  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from evaluation.equation_records import dataset_variable_mapping, make_equation_record  # noqa: E402
from experiment_runtime import phase_output_paths  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from models.tpsr_adapter import predict_equation_tpsr  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DATA_DIR = ROOT / "data" / "human" / "gse112372_lps"
DREAMLIKE = Path(
    os.environ.get(
        "LTSR_DREAMLIKE_DATA",
        str(ROOT / "results" / "synthetic" / "phase7_dreamlike_v1"),
    )
)
# Checkpoint/config env-overridable for GPU runs (e.g. LTSR_WEIGHTS=.../100M.ckpt)
WEIGHTS = Path(os.environ.get("LTSR_WEIGHTS", str(ROOT / "NSRS" / "weights" / "10M.ckpt")))
CONFIG = Path(os.environ.get("LTSR_CONFIG", str(ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml")))
EQ_SETTING = Path(os.environ.get("LTSR_EQ_SETTING", str(ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json")))
OUT_DIR, REPORT = phase_output_paths(ROOT, "phase8", "phase8_report.md")

from training.selective_layers import (  # noqa: E402
    require_live_phase4_ranking,
    resolve_selected_layers,
)

# High-contribution layer set = top-k of the Phase 4 accuracy ranking (principled
# a-priori; NOT the earlier post-hoc middle_3). Falls back to the frozen ranking
# if contributions.json is absent.
_PHASE4_CONTRIB = Path(
    os.environ.get(
        "LTSR_PHASE4_CONTRIB",
        str(ROOT / "results" / "phase_results" / "phase4_multiseed" / "contrib_aggregate.json"),
    )
)
HIGH_CONTRIB, _HC_SOURCE, _HC_RULE = resolve_selected_layers(
    _PHASE4_CONTRIB, mode="accuracy", rule="top", k=3
)
if os.environ.get("LTSR_REQUIRE_LIVE_PHASE4", "0") == "1":
    require_live_phase4_ranking(_HC_SOURCE, _PHASE4_CONTRIB)


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}g}"


def sanitize(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def make_light_fit(params_fit, beam_size=1, n_restarts=1, stop_time=0.5):
    from copy import deepcopy
    from nesymres.dclasses import BFGSParams

    p = deepcopy(params_fit)
    p.beam_size = beam_size
    p.bfgs = BFGSParams(
        activated=True,
        n_restarts=n_restarts,
        add_coefficients_if_not_existing=False,
        normalization_o=False,
        idx_remove=True,
        normalization_type="MSE",
        stop_time=stop_time,
    )
    return p


def stack_donor_derivatives(panel, method: str, donors: Optional[Sequence[str]] = None):
    keys = list(donors) if donors is not None else list(panel.X_donors.keys())
    Xs, Ys = [], []
    for d in keys:
        if d not in panel.X_donors:
            continue
        X, Y = estimate_derivatives(panel.times, panel.X_donors[d], method=method)
        Xs.append(X)
        Ys.append(Y)
    if not Xs:
        raise RuntimeError("No donor trajectories for derivative stacking")
    return np.vstack(Xs), np.vstack(Ys)


def build_dreamlike_ft(k: int = 2) -> List[SampledDataset]:
    if not (DREAMLIKE / "network.json").exists():
        generate_dreamlike_dataset(DREAMLIKE)
    network = load_network(DREAMLIKE / "network.json")
    expr = load_expression(DREAMLIKE / "expression.npz")
    out = []
    for t in range(min(6, network.n_genes)):
        regs = oracle_regulators(network, t)[:k]
        out.append(
            build_local_problem(
                network,
                expr["X_train"],
                expr["Y_train"][:, t],
                t,
                regs,
                eq_id=f"dreamlike_t{t}",
                split="train",
                max_vars=3,
                selection_method="oracle",
            )
        )
    return out


def eval_sr(
    model, params_fit, problems, decode: str = "beam", tpsr_kw=None, source_names=None
) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    tpsr_kw = tpsr_kw or {}
    rows = []
    for ds in problems:
        expr = ""
        out: Dict[str, Any] = {}
        failure_reason = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if decode == "beam":
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        out = predict_equation(model, params_fit, ds.X, ds.y, quiet=True)
                    expr = out["equation"]
                else:
                    out = predict_equation_tpsr(
                        model, params_fit, ds.X, ds.y, quiet=True, **tpsr_kw
                    )
                    expr = out["equation"]
        except Exception as exc:
            expr = ""
            failure_reason = f"{type(exc).__name__}: {exc}"
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(
            ds.y, y_hat, expr, ds.spec.variable_names, true_expr="", X=ds.X,
            variable_names=ds.spec.variable_names,
        )
        rows.append(make_equation_record(
            eq_id=ds.spec.eq_id,
            predicted_expr=expr,
            variable_names=ds.spec.variable_names,
            mapping=dataset_variable_mapping(ds, source_names),
            scores=sc,
            true_expr="",
            candidate_expressions=out.get("all_preds", [expr] if expr else []),
            decoder="nesymres_beam_bfgs" if decode == "beam" else "tpsr_mcts_bfgs",
            decoder_metadata={
                key: out.get(key)
                for key in ("bfgs_loss", "reward", "mcts_steps", "sample_times", "state_ids")
                if key in out
            },
            failure_reason=failure_reason,
        ))
    return {
        "aggregate": aggregate_prediction_scores(rows),
        "per_problem": rows,
    }


def eval_holdout_donor(
    model,
    params_fit,
    panel,
    train_problems: List[SampledDataset],
    selections: Dict[int, List[int]],
    X_te: np.ndarray,
    Y_te: np.ndarray,
    decode: str = "beam",
    tpsr_kw=None,
) -> Dict[str, Any]:
    """Fit on train local problems; score equations on held-out donor states."""
    import contextlib
    import io
    import warnings

    tpsr_kw = tpsr_kw or {}
    network = panel.as_grn_like()
    rows = []
    for ds in train_problems:
        # recover target index from motif annotation
        motif = ds.spec.motif or ""
        tname = motif.split("target=")[-1].split(";")[0] if "target=" in motif else ""
        if tname not in panel.gene_names:
            continue
        t = panel.gene_index(tname)
        regs = selections.get(t, [])
        te = build_local_problem(
            network,
            X_te,
            Y_te[:, t],
            t,
            regs,
            eq_id=f"{ds.spec.eq_id}_holdout",
            split="holdout",
            include_target=True,
            max_vars=3,
            selection_method=ds.spec.selection_method
            if hasattr(ds.spec, "selection_method")
            else "prior",
        )
        expr = ""
        out: Dict[str, Any] = {}
        failure_reason = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if decode == "beam":
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        out = predict_equation(model, params_fit, ds.X, ds.y, quiet=True)
                    expr = out["equation"]
                else:
                    out = predict_equation_tpsr(
                        model, params_fit, ds.X, ds.y, quiet=True, **tpsr_kw
                    )
                    expr = out["equation"]
        except Exception as exc:
            expr = ""
            failure_reason = f"{type(exc).__name__}: {exc}"
        y_hat = eval_expression(expr, te.X, te.spec.variable_names) if expr else None
        sc = score_prediction(
            te.y, y_hat, expr, te.spec.variable_names, true_expr="", X=te.X,
            variable_names=te.spec.variable_names,
        )
        rows.append(make_equation_record(
            eq_id=te.spec.eq_id,
            predicted_expr=expr,
            variable_names=te.spec.variable_names,
            mapping=dataset_variable_mapping(te, panel.gene_names),
            scores=sc,
            true_expr="",
            candidate_expressions=out.get("all_preds", [expr] if expr else []),
            decoder="nesymres_beam_bfgs" if decode == "beam" else "tpsr_mcts_bfgs",
            decoder_metadata={
                key: out.get(key)
                for key in ("bfgs_loss", "reward", "mcts_steps", "sample_times", "state_ids")
                if key in out
            },
            failure_reason=failure_reason,
            target=tname,
        ))
    return {
        "aggregate": aggregate_prediction_scores(rows),
        "per_problem": rows,
    }


def run_pysr(X, y, variable_names, niterations: int) -> str:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=20,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        random_state=0,
    )
    model.fit(X, y, variable_names=list(variable_names))
    return str(model.get_best()["equation"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--derivative", default="smooth_fd", choices=["smooth_fd", "fd", "spline"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-vars", type=int, default=3)
    parser.add_argument("--holdout-donor", default="11")
    parser.add_argument("--pysr-iters", type=int, default=12)
    parser.add_argument("--skip-pysr", action="store_true")
    parser.add_argument("--with-tpsr", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    # ---- data ----
    log("=== Prepare GSE112372 panel ===")
    panel = prepare_gse112372(args.data_dir, force_download=args.force_download)
    save_panel_summary(panel, args.data_dir / "panel_summary.json")
    log(
        f"genes={len(panel.gene_names)} times={panel.times.tolist()} "
        f"donors={sorted(panel.X_donors)} prior_edges={len(panel.prior_edges)}"
    )

    donors = sorted(panel.X_donors.keys())
    hold = args.holdout_donor if args.holdout_donor in panel.X_donors else donors[-1]
    train_donors = [d for d in donors if d != hold]
    log(f"train donors={train_donors} holdout={hold}")

    X_tr, Y_tr = stack_donor_derivatives(panel, args.derivative, train_donors)
    X_te, Y_te = stack_donor_derivatives(panel, args.derivative, [hold])
    log(f"train FD points={X_tr.shape[0]} holdout points={X_te.shape[0]}")

    # ---- selection ----
    log("=== Regulator selection vs curated prior ===")
    sel_methods = ["prior", "prior_corr", "corr", "mi"]
    selection_summary: Dict[str, Any] = {}
    for method in sel_methods:
        _, selections, rows = build_human_local_problems(
            panel,
            X_tr,
            Y_tr,
            method=method,
            k=args.k,
            max_vars=args.max_vars,
            split="train",
        )
        er = prior_edge_recovery(panel, selections)
        mean_recall = float(np.mean([r["prior_recall"] for r in rows])) if rows else 0.0
        selection_summary[method] = {
            "edge_recovery": er,
            "mean_prior_recall": mean_recall,
            "n_targets": len(rows),
            "rows": rows,
        }
        log(
            f"  {method}: edgeF1={fmt(er['f1'])} P={fmt(er['precision'])} "
            f"R={fmt(er['recall'])} mean_prior_recall={fmt(mean_recall)}"
        )

    prior_probs, prior_sel, prior_rows = build_human_local_problems(
        panel,
        X_tr,
        Y_tr,
        method="prior",
        k=args.k,
        max_vars=args.max_vars,
        split="train",
    )
    log(f"SR problems (prior, k={args.k}): {len(prior_probs)}")

    # ---- models ----
    log("=== Load NeSymReS + dreamlike selective FT ===")
    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=1)
    fit = make_light_fit(params_fit)
    with EQ_SETTING.open(encoding="utf-8") as f:
        word2id = json.load(f)["word2id"]
    base_model.to(device)

    dl_probs = build_dreamlike_ft(k=2)
    dl_ds = GRNFinetuneDataset(dl_probs, word2id, max_points=args.max_points, seed=0)
    dl_loader = DataLoader(
        dl_ds,
        batch_size=min(args.batch_size, max(len(dl_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
    )
    dreamlike_model = clone_model(base_model)
    dl_train = train_selective(
        dreamlike_model,
        dl_loader,
        HIGH_CONTRIB,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    log(f"dreamlike selective FT CE={fmt(dl_train['final_loss'])}")

    tpsr_kw = {
        "rollout": 1,
        "horizon": 20,
        "width": 2,
        "num_beams": 1,
        "bfgs_restarts": 1,
        "bfgs_stop_time": 0.4,
    }

    # ---- SR comparison ----
    log("=== Local SR comparison (fit on train donors) ===")
    compare_in: Dict[str, Any] = {}
    compare_hold: Dict[str, Any] = {}
    cells = [
        ("pretrained_beam", clone_model(base_model), "beam"),
        ("selective_dreamlike_beam", dreamlike_model, "beam"),
    ]
    if args.with_tpsr:
        cells.append(("selective_dreamlike_tpsr", clone_model(dreamlike_model), "tpsr"))

    for name, model, decode in cells:
        log(f"  {name}")
        t0 = time.time()
        model.eval()
        inn = eval_sr(
            model, fit, prior_probs, decode=decode, tpsr_kw=tpsr_kw,
            source_names=panel.gene_names,
        )
        holdout = eval_holdout_donor(
            model,
            fit,
            panel,
            prior_probs,
            prior_sel,
            X_te,
            Y_te,
            decode=decode,
            tpsr_kw=tpsr_kw,
        )
        inn["elapsed_sec"] = time.time() - t0
        compare_in[name] = inn
        compare_hold[name] = holdout
        log(
            f"    in-donors NMSE={fmt(inn['aggregate']['nmse'])}  "
            f"holdout NMSE={fmt(holdout['aggregate']['nmse'])}  "
            f"({inn['elapsed_sec']:.1f}s)"
        )

    if not args.skip_pysr:
        log("  pysr")
        t0 = time.time()
        pysr_rows = []
        hold_rows = []
        network = panel.as_grn_like()
        for ds in prior_probs:
            failure_reason = None
            try:
                expr = run_pysr(ds.X, ds.y, ds.spec.variable_names, args.pysr_iters)
            except Exception as exc:
                log(f"    PySR fail {ds.spec.eq_id}: {exc}")
                expr = ""
                failure_reason = f"{type(exc).__name__}: {exc}"
            y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
            sc = score_prediction(
                ds.y, y_hat, expr, ds.spec.variable_names, true_expr="", X=ds.X,
                variable_names=ds.spec.variable_names,
            )
            pysr_rows.append(make_equation_record(
                eq_id=ds.spec.eq_id,
                predicted_expr=expr,
                variable_names=ds.spec.variable_names,
                mapping=dataset_variable_mapping(ds, panel.gene_names),
                scores=sc,
                true_expr="",
                candidate_expressions=[expr] if expr else [],
                decoder="pysr",
                failure_reason=failure_reason,
            ))

            motif = ds.spec.motif or ""
            tname = motif.split("target=")[-1].split(";")[0] if "target=" in motif else ""
            if tname in panel.gene_names:
                t = panel.gene_index(tname)
                regs = prior_sel.get(t, [])
                te = build_local_problem(
                    network,
                    X_te,
                    Y_te[:, t],
                    t,
                    regs,
                    eq_id=f"{ds.spec.eq_id}_holdout",
                    split="holdout",
                    max_vars=args.max_vars,
                    selection_method="prior",
                )
                y_hat_te = (
                    eval_expression(expr, te.X, te.spec.variable_names) if expr else None
                )
                sc_te = score_prediction(
                    te.y, y_hat_te, expr, te.spec.variable_names, true_expr="", X=te.X,
                    variable_names=te.spec.variable_names,
                )
                hold_rows.append(make_equation_record(
                    eq_id=te.spec.eq_id,
                    predicted_expr=expr,
                    variable_names=te.spec.variable_names,
                    mapping=dataset_variable_mapping(te, panel.gene_names),
                    scores=sc_te,
                    true_expr="",
                    candidate_expressions=[expr] if expr else [],
                    decoder="pysr",
                    failure_reason=failure_reason,
                    target=tname,
                ))

        compare_in["pysr"] = {
            "aggregate": aggregate_prediction_scores(pysr_rows),
            "per_problem": pysr_rows,
            "elapsed_sec": time.time() - t0,
        }
        compare_hold["pysr"] = {
            "aggregate": aggregate_prediction_scores(hold_rows),
            "per_problem": hold_rows,
        }
        log(
            f"    in-donors NMSE={fmt(compare_in['pysr']['aggregate']['nmse'])}  "
            f"holdout NMSE={fmt(compare_hold['pysr']['aggregate']['nmse'])}  "
            f"({compare_in['pysr']['elapsed_sec']:.1f}s)"
        )

    # ---- write ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "source": panel.source,
        "genes": panel.gene_names,
        "times_h": panel.times.tolist(),
        "donors": donors,
        "holdout_donor": hold,
        "train_donors": train_donors,
        "derivative": args.derivative,
        "k": args.k,
        "selected_layers": HIGH_CONTRIB,
        "layer_ranking_source": _HC_SOURCE,
        "layer_selection_rule": _HC_RULE,
        "phase4_contributions": str(_PHASE4_CONTRIB),
        "n_train_points": int(X_tr.shape[0]),
        "n_holdout_points": int(X_te.shape[0]),
        "selection": selection_summary,
        "dreamlike_ft": dl_train,
        "compare_in_donors": compare_in,
        "compare_holdout_donor": compare_hold,
        "caveats": [
            "True ODEs unknown; do not claim equation discovery.",
            "dx/dt is a smoothed finite-difference / spline proxy.",
            "Prior edges are curated literature-style soft gold, not assay-validated for this cohort.",
            "Panel is a 20-gene LPS inflammatory subset of GSE112372.",
        ],
    }
    out_json = OUT_DIR / "phase8_results.json"
    out_json.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")

    lines = [
        "# Phase 8: human LPS macrophage application (GSE112372)",
        "",
        "## Setup",
        "",
        f"- Source: `{panel.source}` TPM subset",
        f"- Genes: {len(panel.gene_names)} ({', '.join(panel.gene_names)})",
        f"- Times (h): {panel.times.tolist()}",
        f"- Train donors: {train_donors}; holdout donor: `{hold}`",
        f"- Derivative: `{args.derivative}` (proxy, not true time derivative)",
        f"- Candidates per target: k={args.k}, max_vars={args.max_vars}",
        f"- Selective FT layers: `{', '.join(HIGH_CONTRIB)}`",
        f"- Layer ranking: `{_HC_SOURCE}` (`{_HC_RULE}`) from `{_PHASE4_CONTRIB.as_posix()}`",
        f"- Results JSON: `{out_json.as_posix()}`",
        "",
        "## Regulator selection vs curated prior",
        "",
        "| method | edge F1 | precision | recall | mean prior recall |",
        "|--------|---------|-----------|--------|-------------------|",
    ]
    for m, s in selection_summary.items():
        er = s["edge_recovery"]
        lines.append(
            f"| `{m}` | {fmt(er['f1'])} | {fmt(er['precision'])} | "
            f"{fmt(er['recall'])} | {fmt(s['mean_prior_recall'])} |"
        )

    lines += [
        "",
        "## Local SR (prior candidates)",
        "",
        "### In-donor (train donors)",
        "",
        "| method | NMSE | R2 | time (s) |",
        "|--------|------|----|----------|",
    ]
    for name, ev in compare_in.items():
        a = ev["aggregate"]
        lines.append(
            f"| `{name}` | {fmt(a['nmse'])} | {fmt(a['r2'])} | "
            f"{fmt(ev.get('elapsed_sec', float('nan')), 3)} |"
        )

    lines += [
        "",
        "### Holdout donor",
        "",
        "| method | NMSE | R2 |",
        "|--------|------|----|",
    ]
    for name, ev in compare_hold.items():
        a = ev["aggregate"]
        lines.append(f"| `{name}` | {fmt(a['nmse'])} | {fmt(a['r2'])} |")

    # example predicted equations
    lines += ["", "## Example predicted equations (prior targets)", ""]
    example_src = compare_hold.get("selective_dreamlike_beam") or next(
        iter(compare_hold.values()), None
    )
    if example_src:
        for row in example_src.get("per_problem", [])[:5]:
            lines.append(
                f"- `{row.get('target', row.get('eq_id'))}`: `{row.get('pred', '')}` "
                f"(holdout NMSE={fmt(row.get('nmse', float('nan')))})"
            )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Do **not** claim recovery of true human regulatory ODEs.",
        "- Evaluate by **held-out donor prediction** and **prior TF consistency** only.",
        "- RNA-seq TPM + FD/spline yields a coarse proxy for dx/dt on 5 time points.",
        "- Curated prior is soft gold for this inflammatory panel.",
        "",
        "## Findings",
        "",
    ]
    best_hold = min(
        compare_hold.items(),
        key=lambda kv: kv[1]["aggregate"]["nmse"]
        if np.isfinite(kv[1]["aggregate"]["nmse"])
        else 1e9,
    )
    prior_f1 = selection_summary["prior"]["edge_recovery"]["f1"]
    corr_f1 = selection_summary["corr"]["edge_recovery"]["f1"]
    lines.append(
        f"1. Prior-constrained selection edge F1={fmt(prior_f1)} vs data-only corr F1={fmt(corr_f1)}."
    )
    lines.append(
        f"2. Best holdout donor NMSE: `{best_hold[0]}` = {fmt(best_hold[1]['aggregate']['nmse'])}."
    )
    lines.append(
        "3. Phase 8 is an application demo; main LTSR claims remain on synthetic + DREAM4."
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

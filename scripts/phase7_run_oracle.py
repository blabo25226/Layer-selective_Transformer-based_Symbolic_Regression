"""Phase 7: regulator preselection + oracle/local NeSymReS evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.dreamlike_grn import (  # noqa: E402
    build_local_problem,
    load_expression,
    load_network,
)
from data.finetune_dataset import GRNFinetuneDataset, collate_finetune, instantiate_expr  # noqa: E402
from data.regulator_selection import (  # noqa: E402
    select_regulators,
    selection_metrics,
    oracle_regulators,
)
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.aggregation import aggregate_prediction_scores, true_variables  # noqa: E402
from evaluation.grn_metrics import (  # noqa: E402
    edge_recovery,
    predicted_edges_from_selections,
)
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase7_dreamlike_v1"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase7"
REPORT = ROOT / "results" / "phase_results" / "phase7_report.md"

from training.selective_layers import resolve_selected_layers  # noqa: E402

# High-contribution layer set = top-k of the Phase 4 accuracy ranking (principled
# a-priori; NOT the earlier post-hoc middle_3). Falls back to the frozen ranking
# if contributions.json is absent.
_PHASE4_CONTRIB = ROOT / "results" / "phase_results" / "phase4" / "contributions.json"
HIGH_CONTRIB, _HC_SOURCE, _HC_RULE = resolve_selected_layers(
    _PHASE4_CONTRIB, mode="accuracy", rule="top", k=3
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}g}"


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


def build_problems_for_method(
    network,
    X,
    Y,
    *,
    method: str,
    k: int,
    split: str,
    max_vars: int,
    target_limit: int,
) -> List:
    problems = []
    selections: Dict[int, List[int]] = {}
    sel_rows = []
    n_targets = network.n_genes if target_limit <= 0 else min(target_limit, network.n_genes)
    for t in range(n_targets):
        y = Y[:, t]
        regs = select_regulators(method, network, X, y, t, k=k)
        selections[t] = regs
        true = oracle_regulators(network, t)
        sm = selection_metrics(true, regs)
        sel_rows.append({"target": t, "true": true, "pred": regs, **sm})
        ds = build_local_problem(
            network,
            X,
            y,
            t,
            regs,
            eq_id=f"{split}_{method}_t{t}",
            split=split,
            include_target=True,
            max_vars=max_vars,
            selection_method=method,
        )
        problems.append(ds)
    return problems, selections, sel_rows


def eval_sr(model, params_fit, problems) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    rows = []
    for ds in problems:
        # Prefer concrete numeric expr for symbolic scoring when possible
        true_expr = ds.spec.target_expr
        try:
            true_expr = instantiate_expr(ds)
        except Exception:
            true_expr = ds.spec.target_expr
        expr = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    out = predict_equation(model, params_fit, ds.X, ds.y, quiet=True)
                expr = out["equation"]
        except Exception:
            expr = ""
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(
            ds.y, y_hat, expr, true_variables(true_expr, ds.spec.variable_names),
            true_expr=true_expr
        )
        rows.append({"eq_id": ds.spec.eq_id, "pred": expr, "true": true_expr, **sc})
    return {
        "aggregate": aggregate_prediction_scores(rows),
        "per_problem": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--k", type=int, default=2, help="Max candidate regulators")
    parser.add_argument("--max-vars", type=int, default=3)
    parser.add_argument("--target-limit", type=int, default=6, help="0=all genes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    args = parser.parse_args()

    net_path = args.data_dir / "network.json"
    exp_path = args.data_dir / "expression.npz"
    if not net_path.exists() or not exp_path.exists():
        log("Dataset missing; generating phase7_dreamlike_v1...")
        from data.dreamlike_grn import generate_dreamlike_dataset

        generate_dreamlike_dataset(args.data_dir)

    network = load_network(net_path)
    expr = load_expression(exp_path)
    X_tr, Y_tr = expr["X_train"], expr["Y_train"]
    X_te, Y_te = expr["X_test"], expr["Y_test"]

    methods = ["oracle", "corr", "mi", "lasso"]
    sel_summary = {}
    n_targets = network.n_genes if args.target_limit <= 0 else min(
        args.target_limit, network.n_genes
    )
    evaluated = set(range(n_targets))
    true_edges = [(r, t) for r, t, _ in network.edges if t in evaluated]

    log(f"Network: {network.n_genes} genes, {len(network.edges)} edges")
    log(f"Edge-recovery scope: {len(true_edges)} edges among {n_targets} targets")
    for method in methods:
        _, selections, sel_rows = build_problems_for_method(
            network,
            X_tr,
            Y_tr,
            method=method,
            k=args.k,
            split="train",
            max_vars=args.max_vars,
            target_limit=args.target_limit,
        )
        # edge recovery from selection on all scored targets
        pred_edges = predicted_edges_from_selections(selections)
        er = edge_recovery(true_edges, pred_edges)
        mean_f1 = float(np.mean([r["f1"] for r in sel_rows])) if sel_rows else 0.0
        sel_summary[method] = {
            "per_target_f1_mean": mean_f1,
            "edge_recovery": er,
            "per_target": sel_rows,
        }
        log(
            f"  select[{method}] target-F1={fmt(mean_f1)}  "
            f"edge-F1={fmt(er['f1'])}  P={fmt(er['precision'])} R={fmt(er['recall'])}"
        )

    # SR on oracle local problems (train FT + test eval)
    train_probs, _, _ = build_problems_for_method(
        network,
        X_tr,
        Y_tr,
        method="oracle",
        k=args.k,
        split="train",
        max_vars=args.max_vars,
        target_limit=args.target_limit,
    )
    test_probs, _, _ = build_problems_for_method(
        network,
        X_te,
        Y_te,
        method="oracle",
        k=args.k,
        split="test",
        max_vars=args.max_vars,
        target_limit=args.target_limit,
    )
    # Also evaluate SR under corr selection (error compounding)
    test_corr, _, _ = build_problems_for_method(
        network,
        X_te,
        Y_te,
        method="corr",
        k=args.k,
        split="test",
        max_vars=args.max_vars,
        target_limit=args.target_limit,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    base_model, params_fit = load_nesymres(
        WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size
    )
    fit = make_light_fit(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )

    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    word2id = eq_setting["word2id"]
    train_ds = GRNFinetuneDataset(
        train_probs, word2id, max_points=args.max_points, seed=0
    )
    log(f"Oracle FT examples tokenized: {len(train_ds)} / {len(train_probs)}")

    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
    )

    log("Fine-tuning selective layers on oracle locals...")
    ft_model = clone_model(base_model)
    if len(train_ds) > 0:
        train_info = train_selective(
            ft_model,
            loader,
            HIGH_CONTRIB,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
    else:
        train_info = {"final_loss": float("nan"), "trainable": 0.0}
    log(f"  trainable={int(train_info.get('trainable', 0)):,}  CE={fmt(train_info.get('final_loss', float('nan')))}")

    sr_results = {}
    for name, model, problems in [
        ("pretrained_oracle", clone_model(base_model), test_probs),
        ("selective_oracle", ft_model, test_probs),
        ("selective_corr", clone_model(ft_model), test_corr),
    ]:
        log(f"\n=== SR {name} | n={len(problems)} ===")
        t0 = time.time()
        model.eval()
        ev = eval_sr(model, fit, problems)
        elapsed = time.time() - t0
        ev["elapsed_sec"] = elapsed
        sr_results[name] = ev
        a = ev["aggregate"]
        log(
            f"  NMSE={fmt(a['nmse'])}  R2={fmt(a['r2'])}  "
            f"varF1={fmt(a['var_f1'])}  sym={fmt(a['sym_rate'])}  ({elapsed:.1f}s)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "network": {"n_genes": network.n_genes, "n_edges": len(network.edges)},
        "selection": sel_summary,
        "sr": sr_results,
        "config": {
            "k": args.k,
            "max_vars": args.max_vars,
            "target_limit": args.target_limit,
            "layers": HIGH_CONTRIB,
            "epochs": args.epochs,
        },
    }
    out_json = OUT_DIR / "dreamlike_results.json"
    out_json.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")

    lines = [
        "# Phase 7: DREAM-like GRN (oracle + variable selection)",
        "",
        f"- Dataset: `{args.data_dir.as_posix()}` (GNW-style synthetic; no Synapse download)",
        f"- Genes: {network.n_genes}, edges: {len(network.edges)}",
        f"- Targets evaluated: {args.target_limit or network.n_genes}, k={args.k} regulators, max_vars={args.max_vars}",
        f"- Selective FT layers: `{', '.join(HIGH_CONTRIB)}`",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## Regulator selection (train expression)",
        "",
        "| method | mean target F1 | edge P | edge R | edge F1 |",
        "|--------|----------------|--------|--------|---------|",
    ]
    for m in methods:
        s = sel_summary[m]
        er = s["edge_recovery"]
        lines.append(
            f"| `{m}` | {fmt(s['per_target_f1_mean'])} | {fmt(er['precision'])} | "
            f"{fmt(er['recall'])} | {fmt(er['f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Local symbolic regression (test RHS)",
            "",
            "| condition | selection | NMSE | R² | var F1 | sym | time (s) |",
            "|-----------|-----------|------|----|--------|-----|----------|",
        ]
    )
    labels = {
        "pretrained_oracle": ("pretrained", "oracle"),
        "selective_oracle": ("selective", "oracle"),
        "selective_corr": ("selective", "corr"),
    }
    for key, (ft, sel) in labels.items():
        a = sr_results[key]["aggregate"]
        lines.append(
            f"| `{key}` | {sel} / {ft} | {fmt(a['nmse'])} | {fmt(a['r2'])} | "
            f"{fmt(a['var_f1'])} | {fmt(a['sym_rate'])} | "
            f"{sr_results[key]['elapsed_sec']:.1f} |"
        )

    a0 = sr_results["pretrained_oracle"]["aggregate"]["nmse"]
    a1 = sr_results["selective_oracle"]["aggregate"]["nmse"]
    a2 = sr_results["selective_corr"]["aggregate"]["nmse"]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"1. **Oracle selection** edge F1 = {fmt(sel_summary['oracle']['edge_recovery']['f1'])} "
            f"(expected high / perfect on true parents).",
            f"2. **Practical selectors** (corr / mi / lasso) edge F1: "
            f"{fmt(sel_summary['corr']['edge_recovery']['f1'])} / "
            f"{fmt(sel_summary['mi']['edge_recovery']['f1'])} / "
            f"{fmt(sel_summary['lasso']['edge_recovery']['f1'])}.",
            f"3. **Selective FT vs pretrained** (oracle locals): "
            f"ΔNMSE = {fmt(a1 - a0)}.",
            f"4. **Selection error compounding**: selective+corr NMSE = {fmt(a2)} "
            f"vs selective+oracle {fmt(a1)} "
            f"(Δ = {fmt(a2 - a1)}).",
            "",
            "## Notes",
            "",
            "- This is a **DREAM4/GNW-style** synthetic substitute until official Synapse dumps are wired in.",
            "- Per-target problems keep ≤3 variables for NeSymReS.",
            "- Next: import real DREAM4 gold standard + steady-state / time series when available.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

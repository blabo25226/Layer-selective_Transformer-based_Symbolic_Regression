"""Phase 7c: official DREAM4 Size100 — selection + transfer SR evaluation."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.dream4 import (  # noqa: E402
    build_dream4_local_problems,
    find_dream4_root,
    list_net_ids,
    load_dream4_expression_bundle,
    trajectory_train_test_split,
    load_dream4_network,
    targets_with_parents,
)
from data.dreamlike_grn import (  # noqa: E402
    build_local_problem,
    generate_dreamlike_dataset,
    load_expression,
    load_network,
)
from data.finetune_dataset import GRNFinetuneDataset, collate_finetune  # noqa: E402
from data.regulator_selection import oracle_regulators  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.grn_metrics import edge_recovery, predicted_edges_from_selections  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DREAM4 = ROOT / "data" / "dream4"
DREAMLIKE = ROOT / "results" / "synthetic" / "phase7_dreamlike_v1"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase7_dream4"
REPORT = ROOT / "results" / "phase_results" / "phase7_dream4_size100_report.md"

HIGH_CONTRIB = ["decoder_0", "decoder_4", "encoder_0"]
SIZE = 100


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


def eval_sr(model, params_fit, problems) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    rows = []
    nmses, r2s, vfs = [], [], []
    for ds in problems:
        true_vars = ds.spec.variable_names
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
        sc = score_prediction(ds.y, y_hat, expr, true_vars, true_expr="")
        rows.append({"eq_id": ds.spec.eq_id, "pred": expr, "motif": ds.spec.motif, **sc})
        if np.isfinite(sc["nmse"]):
            nmses.append(sc["nmse"])
        if np.isfinite(sc["r2"]):
            r2s.append(sc["r2"])
        vfs.append(sc.get("var_f1", 0.0))
    return {
        "aggregate": {
            "n_eval": float(len(problems)),
            "n_valid": float(len(nmses)),
            "nmse": float(np.median(nmses)) if nmses else float("inf"),
            "r2": float(np.median(r2s)) if r2s else float("-inf"),
            "var_f1": float(np.mean(vfs)) if vfs else 0.0,
        },
        "per_problem": rows,
    }


def build_dreamlike_ft_problems(target_limit: int = 6, k: int = 2):
    if not (DREAMLIKE / "network.json").exists():
        generate_dreamlike_dataset(DREAMLIKE)
    network = load_network(DREAMLIKE / "network.json")
    expr = load_expression(DREAMLIKE / "expression.npz")
    problems = []
    for t in range(min(target_limit, network.n_genes)):
        regs = oracle_regulators(network, t)[:k]
        ds = build_local_problem(
            network,
            expr["X_train"],
            expr["Y_train"][:, t],
            t,
            regs,
            eq_id=f"ft_oracle_t{t}",
            split="train",
            max_vars=3,
            selection_method="oracle",
        )
        problems.append(ds)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dream4-root", type=Path, default=DREAM4)
    parser.add_argument("--net-id", type=int, default=1)
    parser.add_argument("--all-nets", action="store_true")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-vars", type=int, default=3)
    parser.add_argument(
        "--sr-targets",
        type=int,
        default=20,
        help="Max #targets with parents used for SR (0=all with parents)",
    )
    parser.add_argument(
        "--select-all",
        action="store_true",
        help="Run selection metrics on all 100 genes (default: parents-only set)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    args = parser.parse_args()

    root = find_dream4_root(args.dream4_root)
    net_ids = list_net_ids(root, SIZE) if args.all_nets else [args.net_id]
    log(f"DREAM4 root: {root}")
    log(f"Size{SIZE} nets: {net_ids}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, params_fit = load_nesymres(
        WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size
    )
    fit = make_light_fit(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )

    log("Building dreamlike oracle FT set (transfer)...")
    ft_problems = build_dreamlike_ft_problems(target_limit=6, k=args.k)
    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    train_ds = GRNFinetuneDataset(
        ft_problems, eq_setting["word2id"], max_points=args.max_points, seed=0
    )
    log(f"FT tokenized: {len(train_ds)}")
    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
    )
    ft_model = clone_model(base_model)
    if len(train_ds):
        train_info = train_selective(
            ft_model, loader, HIGH_CONTRIB, epochs=args.epochs, lr=args.lr, device=device
        )
    else:
        train_info = {"final_loss": float("nan"), "trainable": 0.0}
    log(f"Selective FT done: trainable={int(train_info.get('trainable', 0)):,}")

    methods = ["oracle", "corr", "mi", "lasso"]
    all_sel: Dict[str, Any] = {}
    all_sr: Dict[str, Any] = {}

    for net_id in net_ids:
        log(f"\n======== Size100 Network {net_id} ========")
        network = load_dream4_network(root, SIZE, net_id)
        bundle = load_dream4_expression_bundle(root, SIZE, net_id)
        X, Y = bundle["X_ts"], bundle["Y_ts"]
        log(f"FD samples: {X.shape[0]} x {X.shape[1]} genes; edges={len(network.edges)}")

        parents = targets_with_parents(network)
        if args.select_all:
            sel_targets = list(range(network.n_genes))
        else:
            sel_targets = parents
        sr_targets = list(parents)
        if args.sr_targets > 0:
            sr_targets = sr_targets[: args.sr_targets]
        log(f"Selection targets: {len(sel_targets)}; SR targets: {len(sr_targets)}")

        X_tr, Y_tr, X_te, Y_te = trajectory_train_test_split(
            bundle["times"], bundle["trajectories"], seed=1000 + net_id
        )

        evaluated = set(sel_targets)
        true_edges = [(r, t) for r, t, _ in network.edges if t in evaluated]

        sel_summary = {}
        for method in methods:
            _, selections, sel_rows = build_dream4_local_problems(
                network,
                X_tr,
                Y_tr,
                method=method,
                k=args.k,
                split="train",
                max_vars=args.max_vars,
                target_ids=sel_targets,
                size_tag=SIZE,
            )
            er = edge_recovery(true_edges, predicted_edges_from_selections(selections))
            mean_f1 = float(np.mean([r["f1"] for r in sel_rows])) if sel_rows else 0.0
            sel_summary[method] = {
                "per_target_f1_mean": mean_f1,
                "edge_recovery": er,
                "n_true_edges_scoped": len(true_edges),
                "n_targets": len(sel_targets),
            }
            log(
                f"  select[{method}] F1_target={fmt(mean_f1)}  "
                f"edgeF1={fmt(er['f1'])} P={fmt(er['precision'])} R={fmt(er['recall'])}"
            )
        all_sel[f"net{net_id}"] = sel_summary

        test_oracle, _, _ = build_dream4_local_problems(
            network,
            X_te,
            Y_te,
            method="oracle",
            k=args.k,
            split="test",
            max_vars=args.max_vars,
            target_ids=sr_targets,
            size_tag=SIZE,
        )
        test_corr, _, _ = build_dream4_local_problems(
            network,
            X_te,
            Y_te,
            method="corr",
            k=args.k,
            split="test",
            max_vars=args.max_vars,
            target_ids=sr_targets,
            size_tag=SIZE,
        )

        sr_net = {}
        for name, model, probs in [
            ("pretrained_oracle", clone_model(base_model), test_oracle),
            ("selective_oracle", ft_model, test_oracle),
            ("selective_corr", clone_model(ft_model), test_corr),
        ]:
            log(f"  SR {name} n={len(probs)}")
            t0 = time.time()
            model.eval()
            ev = eval_sr(model, fit, probs)
            ev["elapsed_sec"] = time.time() - t0
            a = ev["aggregate"]
            log(
                f"    NMSE={fmt(a['nmse'])} R2={fmt(a['r2'])} "
                f"varF1={fmt(a['var_f1'])} ({ev['elapsed_sec']:.1f}s)"
            )
            sr_net[name] = ev
        all_sr[f"net{net_id}"] = sr_net

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "dream4_root": str(root),
        "size": SIZE,
        "net_ids": net_ids,
        "selection": all_sel,
        "sr": all_sr,
        "ft": {"layers": HIGH_CONTRIB, "source": "phase7_dreamlike_oracle", **train_info},
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        },
    }
    out_json = OUT_DIR / "size100_results.json"
    out_json.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")

    lines = [
        "# Phase 7c: official DREAM4 Size100",
        "",
        f"- Data root: `{root.as_posix()}`",
        f"- Networks: {net_ids}",
        f"- Supervision: timeseries finite-difference `dx/dt` (70/30 by trajectory; "
        f"~200 rows/net — no multifactorial in Size100 training set)",
        f"- Transfer FT: selective `{', '.join(HIGH_CONTRIB)}` on synthetic dreamlike",
        f"- Selection on {'all genes' if args.select_all else 'genes with parents'}; "
        f"SR on up to {args.sr_targets or 'all'} parent-genes; k={args.k}",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## Regulator selection",
        "",
    ]
    for net_id in net_ids:
        lines.append(f"### Network {net_id}")
        lines.append("")
        lines.append("| method | mean target F1 | edge P | edge R | edge F1 |")
        lines.append("|--------|----------------|--------|--------|---------|")
        for m in methods:
            s = all_sel[f"net{net_id}"][m]
            er = s["edge_recovery"]
            lines.append(
                f"| `{m}` | {fmt(s['per_target_f1_mean'])} | {fmt(er['precision'])} | "
                f"{fmt(er['recall'])} | {fmt(er['f1'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Local SR (FD targets)",
            "",
        ]
    )
    for net_id in net_ids:
        lines.append(f"### Network {net_id}")
        lines.append("")
        lines.append("| condition | NMSE | R² | var F1 | time (s) |")
        lines.append("|-----------|------|----|--------|----------|")
        for key in ("pretrained_oracle", "selective_oracle", "selective_corr"):
            ev = all_sr[f"net{net_id}"][key]
            a = ev["aggregate"]
            lines.append(
                f"| `{key}` | {fmt(a['nmse'])} | {fmt(a['r2'])} | "
                f"{fmt(a['var_f1'])} | {ev['elapsed_sec']:.1f} |"
            )
        lines.append("")

    def mean_metric(cond: str, key: str) -> float:
        vals = [all_sr[f"net{i}"][cond]["aggregate"][key] for i in net_ids]
        return float(np.mean(vals))

    lines.extend(
        [
            "## Aggregate (mean over networks)",
            "",
            "| condition | mean NMSE | mean R² |",
            "|-----------|-----------|---------|",
        ]
    )
    for key in ("pretrained_oracle", "selective_oracle", "selective_corr"):
        lines.append(
            f"| `{key}` | {fmt(mean_metric(key, 'nmse'))} | {fmt(mean_metric(key, 'r2'))} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Size100 has **no multifactorial** file in the main training folder "
            "(unlike Size10); evaluation uses timeseries FD only.",
            "- Gold TSV lists all gene pairs with 0/1; we keep edges with flag=1.",
            "- Oracle edge recall can be <1 when true degree > k.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

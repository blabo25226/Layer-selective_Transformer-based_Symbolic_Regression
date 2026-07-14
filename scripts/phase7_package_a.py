"""Package A: SBML-FT transfer hardening + Size10 multi-net + method comparison.

Guards against SBML-FT overfit:
  - label noise on teacher RHS
  - mix trajectory (protein TF) + random states
  - multi-net training diversity
  - early stopping on held-out token CE
  - report train/val CE gap and transfer NMSE separately
"""

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

from data.dream4 import (  # noqa: E402
    build_dream4_local_problems,
    find_dream4_root,
    list_net_ids,
    load_dream4_expression_bundle,
    load_dream4_network,
)
from data.dream4_sbml import (  # noqa: E402
    expression_string,
    load_nonoise_aligned_fd,
    mix_supervised_and_trajectory,
    parse_sbml_gene_odes,
    sample_supervised_points,
    sbml_path_for,
)
from data.dreamlike_grn import (  # noqa: E402
    build_local_problem,
    generate_dreamlike_dataset,
    load_expression,
    load_network,
)
from data.finetune_dataset import GRNFinetuneDataset, collate_finetune  # noqa: E402
from data.regulator_selection import oracle_regulators  # noqa: E402
from data.synthetic_grn import EquationSpec, SampledDataset  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.grn_metrics import edge_recovery, predicted_edges_from_selections  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from models.tpsr_adapter import predict_equation_tpsr  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DREAM4 = ROOT / "data" / "dream4"
DREAMLIKE = ROOT / "results" / "synthetic" / "phase7_dreamlike_v1"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase7_package_a"
REPORT = ROOT / "results" / "phase_results" / "phase7_package_a_report.md"
HIGH_CONTRIB = ["decoder_0", "decoder_4", "encoder_0"]


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


def eval_ce(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            nums, tokens = batch[0].to(device), batch[1].to(device)
            output, trg = model.forward([nums, tokens])
            losses.append(float(model.compute_loss(output, trg).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def build_sbml_local_problems(
    odes,
    gene_names: Sequence[str],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    max_parents: int,
    net_id: int,
    split: str,
) -> List[SampledDataset]:
    problems = []
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    for g in gene_names:
        ode = odes[g]
        parents = [gene_to_idx[m] for m in ode.modifiers]
        if len(parents) > max_parents:
            continue
        t = ode.gene_idx
        cols = [t] + [r for r in parents if r != t]
        cols = cols[:3]
        local_map = {gene_names[gi]: f"x_{i+1}" for i, gi in enumerate(cols)}
        local_map[g] = local_map.get(g, "x_1")
        numeric = expression_string(ode, local_map=local_map, numeric=True)
        problems.append(
            SampledDataset(
                spec=EquationSpec(
                    eq_id=f"sbml_s10_{net_id}_{split}_{g}",
                    family="dream4_sbml",
                    target_expr=expression_string(ode, local_map=local_map, numeric=False),
                    variable_names=[f"x_{i+1}" for i in range(len(cols))],
                    parameters={"target_gene": float(t), "net_id": float(net_id)},
                    split=split,
                    motif=numeric,
                ),
                X=X[:, cols].astype(float),
                y=Y[:, t].astype(float),
                noise_std=0.0,
            )
        )
    return problems


def eval_sr(model, params_fit, problems, decode: str = "beam", tpsr_kw=None) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    tpsr_kw = tpsr_kw or {}
    nmses, r2s, rows = [], [], []
    for ds in problems:
        expr = ""
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
        except Exception:
            expr = ""
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(ds.y, y_hat, expr, ds.spec.variable_names, true_expr="")
        rows.append({"eq_id": ds.spec.eq_id, "pred": expr, **sc})
        if np.isfinite(sc["nmse"]):
            nmses.append(sc["nmse"])
        if np.isfinite(sc["r2"]):
            r2s.append(sc["r2"])
    return {
        "aggregate": {
            "n_eval": float(len(problems)),
            "n_valid": float(len(nmses)),
            "nmse": float(np.median(nmses)) if nmses else float("inf"),
            "r2": float(np.median(r2s)) if r2s else float("-inf"),
        },
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dream4-root", type=Path, default=DREAM4)
    parser.add_argument("--nets", default="1,2,3,4,5")
    parser.add_argument("--max-parents", type=int, default=2)
    parser.add_argument("--n-random", type=int, default=150)
    parser.add_argument("--label-noise", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--compare-net", type=int, default=1, help="Net for method table")
    parser.add_argument("--compare-targets", type=int, default=8)
    parser.add_argument("--pysr-iters", type=int, default=15)
    parser.add_argument("--skip-pysr", action="store_true")
    parser.add_argument("--with-tpsr", action="store_true")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    args = parser.parse_args()

    root = find_dream4_root(args.dream4_root)
    net_ids = [int(x) for x in args.nets.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}; nets={net_ids}")

    base_model, params_fit = load_nesymres(
        WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size
    )
    fit = make_light_fit(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )
    with EQ_SETTING.open(encoding="utf-8") as f:
        word2id = json.load(f)["word2id"]

    # ----- A1: multi-net SBML FT with overfit guards -----
    log("\n=== A1: SBML-FT (multi-net, noise, early-stop) ===")
    train_probs: List[SampledDataset] = []
    val_probs: List[SampledDataset] = []
    for net_id in net_ids:
        xml = sbml_path_for(root, 10, net_id)
        odes = parse_sbml_gene_odes(xml)
        genes = [f"G{i}" for i in range(1, 11)]
        try:
            _, Xm, Xp, _ = load_nonoise_aligned_fd(root, 10, net_id)
            X, Y = mix_supervised_and_trajectory(
                odes,
                genes,
                Xm,
                Xp,
                n_random=args.n_random,
                seed=10 * net_id,
                label_noise_std=args.label_noise,
            )
            log(f"  net{net_id}: mix traj+random X={X.shape}")
        except Exception as exc:
            log(f"  net{net_id}: traj mix failed ({exc}); random only")
            X, Y = sample_supervised_points(
                odes,
                genes,
                n_points=args.n_random,
                seed=10 * net_id,
                label_noise_std=args.label_noise,
            )
        rng = np.random.default_rng(net_id)
        idx = rng.permutation(X.shape[0])
        n_tr = int(0.8 * len(idx))
        tr, va = idx[:n_tr], idx[n_tr:]
        train_probs.extend(
            build_sbml_local_problems(
                odes, genes, X[tr], Y[tr], max_parents=args.max_parents, net_id=net_id, split="train"
            )
        )
        val_probs.extend(
            build_sbml_local_problems(
                odes, genes, X[va], Y[va], max_parents=args.max_parents, net_id=net_id, split="val"
            )
        )

    train_ds = GRNFinetuneDataset(train_probs, word2id, max_points=args.max_points, seed=0)
    val_ds = GRNFinetuneDataset(val_probs, word2id, max_points=args.max_points, seed=1)
    log(f"FT tokenized train/val: {len(train_ds)} / {len(val_ds)}")
    train_loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(len(val_ds), 1)),
        shuffle=False,
        collate_fn=collate_finetune,
    )

    sbml_model = clone_model(base_model)
    sbml_train = train_selective(
        sbml_model,
        train_loader,
        HIGH_CONTRIB,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        val_loader=val_loader if len(val_ds) else None,
        patience=args.patience if len(val_ds) else 0,
    )
    train_ce = eval_ce(sbml_model, train_loader, device)
    val_ce = eval_ce(sbml_model, val_loader, device) if len(val_ds) else float("nan")
    overfit_gap = (
        float(val_ce - train_ce)
        if np.isfinite(train_ce) and np.isfinite(val_ce)
        else float("nan")
    )
    log(
        f"SBML-FT: stopped_epoch={sbml_train.get('stopped_epoch')}  "
        f"train_CE={fmt(train_ce)} val_CE={fmt(val_ce)} gap={fmt(overfit_gap)}"
    )

    # dreamlike selective (transfer baseline)
    log("\n=== dreamlike selective FT ===")
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
        epochs=5,
        lr=1e-4,
        device=device,
    )
    log(f"dreamlike FT CE={fmt(dl_train['final_loss'])}")

    # ----- A3: Size10 selection aggregate -----
    log("\n=== A3: Size10 selection across nets ===")
    methods = ["oracle", "corr", "mi", "lasso"]
    sel_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for net_id in net_ids:
        network = load_dream4_network(root, 10, net_id)
        bundle = load_dream4_expression_bundle(root, 10, net_id)
        X, Y = bundle["X_ts"], bundle["Y_ts"]
        true_edges = [(r, t) for r, t, _ in network.edges]
        for method in methods:
            _, selections, sel_rows = build_dream4_local_problems(
                network,
                X,
                Y,
                method=method,
                k=2,
                split="all",
                max_vars=3,
                target_limit=10,
                size_tag=10,
            )
            er = edge_recovery(true_edges, predicted_edges_from_selections(selections))
            sel_by_method[method].append(er["f1"])
            log(f"  net{net_id} {method} edgeF1={fmt(er['f1'])}")

    sel_summary = {
        m: {
            "edge_f1_mean": float(np.mean(v)),
            "edge_f1_std": float(np.std(v)),
            "per_net": v,
        }
        for m, v in sel_by_method.items()
    }

    # ----- A2: method comparison on compare-net -----
    log(f"\n=== A2: method comparison on Size10 net{args.compare_net} ===")
    network = load_dream4_network(root, 10, args.compare_net)
    bundle = load_dream4_expression_bundle(root, 10, args.compare_net)
    Xd, Yd = bundle["X_ts"], bundle["Y_ts"]
    n = Xd.shape[0]
    rng = np.random.default_rng(args.compare_net)
    idx = rng.permutation(n)
    te = idx[int(0.7 * n) :]
    if len(te) == 0:
        te = idx
    X_te, Y_te = Xd[te], Yd[te]
    oracle_probs, _, _ = build_dream4_local_problems(
        network,
        X_te,
        Y_te,
        method="oracle",
        k=2,
        split="test",
        max_vars=3,
        target_limit=args.compare_targets,
        size_tag=10,
    )
    log(f"Compare problems: {len(oracle_probs)}")

    tpsr_kw = {
        "rollout": 1,
        "horizon": 20,
        "width": 2,
        "num_beams": 1,
        "bfgs_restarts": 1,
        "bfgs_stop_time": 0.4,
    }
    compare: Dict[str, Any] = {}

    cells = [
        ("pretrained_beam", clone_model(base_model), "beam"),
        ("selective_dreamlike_beam", dreamlike_model, "beam"),
        ("sbml_ft_beam", sbml_model, "beam"),
    ]
    if args.with_tpsr:
        cells.append(("sbml_ft_tpsr", clone_model(sbml_model), "tpsr"))

    for name, model, decode in cells:
        log(f"  SR {name}")
        t0 = time.time()
        model.eval()
        ev = eval_sr(model, fit, oracle_probs, decode=decode, tpsr_kw=tpsr_kw)
        ev["elapsed_sec"] = time.time() - t0
        a = ev["aggregate"]
        log(f"    NMSE={fmt(a['nmse'])} R2={fmt(a['r2'])} ({ev['elapsed_sec']:.1f}s)")
        compare[name] = ev

    if not args.skip_pysr:
        log("  SR pysr")
        t0 = time.time()
        pysr_rows = []
        nmses, r2s = [], []
        for ds in oracle_probs:
            try:
                expr = run_pysr(ds.X, ds.y, ds.spec.variable_names, args.pysr_iters)
            except Exception as exc:
                log(f"    PySR fail {ds.spec.eq_id}: {exc}")
                expr = ""
            y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
            sc = score_prediction(ds.y, y_hat, expr, ds.spec.variable_names, true_expr="")
            pysr_rows.append({"eq_id": ds.spec.eq_id, "pred": expr, **sc})
            if np.isfinite(sc["nmse"]):
                nmses.append(sc["nmse"])
            if np.isfinite(sc["r2"]):
                r2s.append(sc["r2"])
        compare["pysr"] = {
            "aggregate": {
                "n_eval": float(len(oracle_probs)),
                "n_valid": float(len(nmses)),
                "nmse": float(np.median(nmses)) if nmses else float("inf"),
                "r2": float(np.median(r2s)) if r2s else float("-inf"),
            },
            "per_problem": pysr_rows,
            "elapsed_sec": time.time() - t0,
        }
        log(
            f"    NMSE={fmt(compare['pysr']['aggregate']['nmse'])} "
            f"R2={fmt(compare['pysr']['aggregate']['r2'])} "
            f"({compare['pysr']['elapsed_sec']:.1f}s)"
        )

    # SBML in-distribution holdout check (overfit diagnostic)
    holdout_probs = []
    for net_id in net_ids[:1]:
        odes = parse_sbml_gene_odes(sbml_path_for(root, 10, net_id))
        genes = [f"G{i}" for i in range(1, 11)]
        Xh, Yh = sample_supervised_points(
            odes, genes, n_points=120, seed=123, label_noise_std=0.0
        )
        holdout_probs = build_sbml_local_problems(
            odes, genes, Xh, Yh, max_parents=args.max_parents, net_id=net_id, split="holdout"
        )
    hold_pre = eval_sr(clone_model(base_model), fit, holdout_probs)
    hold_ft = eval_sr(sbml_model, fit, holdout_probs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "sbml_ft": {
            **sbml_train,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "overfit_gap_ce": overfit_gap,
            "label_noise": args.label_noise,
            "nets": net_ids,
        },
        "dreamlike_ft": dl_train,
        "selection_size10": sel_summary,
        "compare_net": args.compare_net,
        "compare": compare,
        "sbml_holdout": {
            "pretrained": hold_pre["aggregate"],
            "sbml_ft": hold_ft["aggregate"],
        },
    }
    out_json = OUT_DIR / "package_a_results.json"
    out_json.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")

    lines = [
        "# Package A: transfer hardening + Size10 multi-net + comparison",
        "",
        "## Overfit-aware SBML-FT",
        "",
        f"- Train nets: {net_ids}",
        f"- Label noise std (rel): {args.label_noise}",
        f"- Mix: nonoise protein-TF trajectory + random SBML RHS",
        f"- Early stop patience={args.patience}, lr={args.lr}",
        f"- train_CE={fmt(train_ce)}, val_CE={fmt(val_ce)}, gap={fmt(overfit_gap)}",
        f"- stopped_epoch={sbml_train.get('stopped_epoch')}",
        "",
        "| eval | pretrained NMSE | SBML-FT NMSE |",
        "|------|-----------------|--------------|",
        f"| SBML holdout (clean teacher) | {fmt(hold_pre['aggregate']['nmse'])} | "
        f"{fmt(hold_ft['aggregate']['nmse'])} |",
        f"| DREAM FD transfer (net{args.compare_net}, oracle locals) | "
        f"{fmt(compare['pretrained_beam']['aggregate']['nmse'])} | "
        f"{fmt(compare['sbml_ft_beam']['aggregate']['nmse'])} |",
        "",
        "## Size10 regulator selection (mean edge F1 over nets)",
        "",
        "| method | mean edge F1 | std |",
        "|--------|--------------|-----|",
    ]
    for m in methods:
        s = sel_summary[m]
        lines.append(f"| `{m}` | {fmt(s['edge_f1_mean'])} | {fmt(s['edge_f1_std'])} |")

    lines.extend(
        [
            "",
            f"## Method comparison (Size10 net{args.compare_net}, oracle locals, noisy FD)",
            "",
            "| method | NMSE | R2 | time (s) |",
            "|--------|------|----|----------|",
        ]
    )
    order = [
        "pysr",
        "pretrained_beam",
        "selective_dreamlike_beam",
        "sbml_ft_beam",
        "sbml_ft_tpsr",
    ]
    for key in order:
        if key not in compare:
            continue
        a = compare[key]["aggregate"]
        lines.append(
            f"| `{key}` | {fmt(a['nmse'])} | {fmt(a['r2'])} | "
            f"{compare[key]['elapsed_sec']:.1f} |"
        )

    # gap caution
    transfer_gain = (
        compare["pretrained_beam"]["aggregate"]["nmse"]
        - compare["sbml_ft_beam"]["aggregate"]["nmse"]
    )
    hold_gain = hold_pre["aggregate"]["nmse"] - hold_ft["aggregate"]["nmse"]
    lines.extend(
        [
            "",
            "## Reading the overfit risk",
            "",
            f"- Holdout gain (clean SBML): {fmt(hold_gain)} NMSE drop",
            f"- Transfer gain (noisy DREAM FD): {fmt(transfer_gain)} NMSE drop",
            "- If holdout >> transfer, SBML-FT still mostly memorizes teacher domain.",
            "- CE gap (val-train) should stay modest; large gap => reduce epochs / raise noise.",
            "",
            f"- Results JSON: `{out_json.as_posix()}`",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

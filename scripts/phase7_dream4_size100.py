"""Phase 7c: official DREAM4 Size100 — selection + transfer SR evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

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
from evaluation.equation_records import dataset_variable_mapping, make_equation_record  # noqa: E402
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from evaluation.decode_timeout import decode_time_limit  # noqa: E402
from evaluation.grn_metrics import edge_recovery, predicted_edges_from_selections  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from resumable_evaluation import (  # noqa: E402
    TargetEvaluationBudget,
    TargetEvaluationBudgetReached,
    completed_prefix,
    load_target_checkpoint,
    restore_rng_state,
    save_target_checkpoint,
)
from training.single_layer import clone_model, train_selective  # noqa: E402
from training.selective_layers import require_live_phase4_ranking, resolve_selected_layers  # noqa: E402
from experiment_runtime import phase_output_paths  # noqa: E402

DREAM4 = ROOT / "data" / "dream4"
DREAMLIKE = Path(os.environ.get("LTSR_DREAMLIKE_DATA", str(ROOT / "results" / "synthetic" / "phase7_dreamlike_v1")))
WEIGHTS = Path(os.environ.get("LTSR_WEIGHTS", str(ROOT / "NSRS" / "weights" / "10M.ckpt")))
CONFIG = Path(os.environ.get("LTSR_CONFIG", str(ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml")))
EQ_SETTING = Path(os.environ.get("LTSR_EQ_SETTING", str(ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json")))
OUT_DIR, REPORT = phase_output_paths(ROOT, "phase7_dream4_size100", "phase7_dream4_size100_report.md")
PHASE4_CONTRIB = Path(os.environ.get("LTSR_PHASE4_CONTRIB", str(ROOT / "results" / "phase_results" / "phase4_multiseed" / "contrib_aggregate.json")))
HIGH_CONTRIB, LAYER_SOURCE, LAYER_RULE = resolve_selected_layers(PHASE4_CONTRIB, mode="accuracy", rule="top", k=3)
if os.environ.get("LTSR_REQUIRE_LIVE_PHASE4", "0") == "1":
    require_live_phase4_ranking(LAYER_SOURCE, PHASE4_CONTRIB)
SIZE = 100
DECODE_TIMEOUT_SEC = float(os.environ.get("LTSR_DECODE_TIMEOUT_SEC", "240"))


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


def eval_sr(
    model,
    params_fit,
    problems,
    source_names=None,
    *,
    completed_rows: Sequence[Mapping[str, Any]] = (),
    on_target_complete: Callable[[list[dict[str, Any]]], None] | None = None,
) -> Dict[str, Any]:
    """Evaluate problems, atomically checkpointing after every completed target."""
    import contextlib
    import io
    import warnings

    expected_ids = [str(ds.spec.eq_id) for ds in problems]
    start = completed_prefix(completed_rows, expected_ids)
    rows = [dict(row) for row in completed_rows]
    for index, ds in enumerate(problems[start:], start=start):
        true_vars = ds.spec.variable_names
        expr = ""
        out: Dict[str, Any] = {}
        failure_reason = None
        try:
            with warnings.catch_warnings(), decode_time_limit(DECODE_TIMEOUT_SEC):
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    out = predict_equation(model, params_fit, ds.X, ds.y, quiet=True)
                expr = out["equation"]
        except Exception as exc:
            expr = ""
            failure_reason = f"{type(exc).__name__}: {exc}"
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(
            ds.y, y_hat, expr, true_vars, true_expr="", X=ds.X,
            variable_names=ds.spec.variable_names,
        )
        target_index = int(ds.spec.parameters.get("target_gene", -1))
        target_name = (
            str(source_names[target_index])
            if source_names is not None and 0 <= target_index < len(source_names)
            else None
        )
        rows.append(
            make_equation_record(
                eq_id=ds.spec.eq_id,
                predicted_expr=expr,
                variable_names=ds.spec.variable_names,
                mapping=dataset_variable_mapping(ds, source_names),
                scores=sc,
                true_expr="",
                candidate_expressions=out.get("all_preds", []),
                decoder="nesymres_beam_bfgs",
                decoder_metadata={"bfgs_loss": out.get("bfgs_loss")},
                failure_reason=failure_reason,
                motif=ds.spec.motif,
                target_gene=target_name,
            )
        )
        if on_target_complete is not None:
            on_target_complete(rows)
        log(f"      target checkpoint {index + 1}/{len(problems)}")
        del out, y_hat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "aggregate": aggregate_prediction_scores(rows),
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
    parser.add_argument(
        "--target-eval-budget",
        type=int,
        default=0,
        help="Exit 75 after this many newly checkpointed targets (0=unbounded)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    target_budget = TargetEvaluationBudget(args.target_eval_budget)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = find_dream4_root(args.dream4_root)
    net_ids = list_net_ids(root, SIZE) if args.all_nets else [args.net_id]
    log(f"DREAM4 root: {root}")
    log(f"Size{SIZE} nets: {net_ids}")
    checkpoint_identity = {
        "size": SIZE,
        "net_ids": net_ids,
        "seed": args.seed,
        "k": args.k,
        "max_vars": args.max_vars,
        "sr_targets": args.sr_targets,
        "select_all": args.select_all,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_points": args.max_points,
        "beam_size": args.beam_size,
        "bfgs_restarts": args.bfgs_restarts,
        "bfgs_stop_time": args.bfgs_stop_time,
        "layers": list(HIGH_CONTRIB),
        "ranking_source": str(LAYER_SOURCE),
    }
    checkpoint_path = OUT_DIR / "target_checkpoint.pt"
    saved_checkpoint = load_target_checkpoint(
        checkpoint_path, expected_identity=checkpoint_identity
    )
    progress: dict[str, Any] = (
        saved_checkpoint["progress"] if saved_checkpoint is not None else {"sr": {}}
    )
    progress.setdefault("sr", {})

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
        ft_problems, eq_setting["word2id"], max_points=args.max_points, seed=args.seed
    )
    log(f"FT tokenized: {len(train_ds)}")
    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ft_model = clone_model(base_model)
    if len(train_ds):
        train_info = train_selective(
            ft_model, loader, HIGH_CONTRIB, epochs=args.epochs, lr=args.lr, device=device
        )
    else:
        train_info = {"final_loss": float("nan"), "trainable": 0.0}
    log(f"Selective FT done: trainable={int(train_info.get('trainable', 0)):,}")
    if saved_checkpoint is not None:
        restore_rng_state(saved_checkpoint["rng_state"])
        saved_targets = sum(
            len(condition.get("per_problem", []))
            for network in progress["sr"].values()
            for condition in network.values()
        )
        log(f"Resuming target checkpoint: {saved_targets} target evaluations complete")

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
            bundle["times"], bundle["trajectories"], seed=args.seed * 1000 + net_id
        )

        evaluated = set(sel_targets)
        true_edges = [(r, t) for r, t, _ in network.edges if t in evaluated]

        sel_summary = {}
        train_selections = {}
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
            train_selections[method] = selections
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
            fixed_selections=train_selections["corr"],
        )

        sr_net = {}
        network_progress = progress["sr"].setdefault(f"net{net_id}", {})
        # Evaluation does not mutate model weights. Reusing these two instances
        # avoids retaining two extra 100M-parameter clones per seed.
        for name, model, probs in [
            ("pretrained_oracle", base_model, test_oracle),
            ("selective_oracle", ft_model, test_oracle),
            ("selective_corr", ft_model, test_corr),
        ]:
            condition_progress = network_progress.get(name, {})
            completed_rows = condition_progress.get("per_problem", [])
            elapsed_before = float(condition_progress.get("elapsed_sec", 0.0))
            completed_prefix(
                completed_rows, [str(problem.spec.eq_id) for problem in probs]
            )
            log(
                f"  SR {name} n={len(probs)} "
                f"resume={len(completed_rows)}/{len(probs)}"
            )
            t0 = time.time()
            model.eval()

            def checkpoint_rows(
                rows: list[dict[str, Any]],
                *,
                condition: str = name,
                prior_elapsed: float = elapsed_before,
                started_at: float = t0,
            ) -> None:
                network_progress[condition] = {
                    "per_problem": list(rows),
                    "elapsed_sec": prior_elapsed + time.time() - started_at,
                }
                save_target_checkpoint(
                    checkpoint_path,
                    identity=checkpoint_identity,
                    progress=progress,
                )
                target_budget.record_checkpoint()

            had_work = len(completed_rows) < len(probs)
            ev = eval_sr(
                model,
                fit,
                probs,
                network.gene_names,
                completed_rows=completed_rows,
                on_target_complete=checkpoint_rows,
            )
            ev["elapsed_sec"] = elapsed_before + (
                time.time() - t0 if had_work else 0.0
            )
            network_progress[name] = {
                "per_problem": ev["per_problem"],
                "elapsed_sec": ev["elapsed_sec"],
            }
            save_target_checkpoint(
                checkpoint_path,
                identity=checkpoint_identity,
                progress=progress,
            )
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
        "ft": {"layers": HIGH_CONTRIB, "ranking_source": LAYER_SOURCE, "rule": LAYER_RULE, "source": "phase7_dreamlike_oracle", **train_info},
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        },
    }
    out_json = OUT_DIR / "size100_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    partial_out = out_json.with_name(out_json.name + ".partial")
    partial_out.write_text(
        json.dumps(sanitize(out), indent=2), encoding="utf-8"
    )
    os.replace(partial_out, out_json)

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
    try:
        exit_code = main()
    except TargetEvaluationBudgetReached as exc:
        log(f"[target budget] {exc}")
        exit_code = 75
    raise SystemExit(exit_code)

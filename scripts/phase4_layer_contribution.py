"""Phase 4: formal layer contribution across accuracy / symbolic / variable metrics."""

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

from data.finetune_dataset import (  # noqa: E402
    GRNFinetuneDataset,
    collate_finetune,
    instantiate_expr,
    load_split_problems,
)
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.layer_contribution import (  # noqa: E402
    compute_contributions,
    rank_by_contribution,
)
from models.layer_selector import get_layer_registry  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase4"
REPORT = ROOT / "results" / "phase_results" / "phase4_report.md"


def build_phase4_conditions(model) -> Dict[str, Optional[List[str]]]:
    """Core contribution set: base, head, each encoder/decoder layer, full."""
    reg = get_layer_registry(model)
    enc = [n for n in reg if n.startswith("encoder_") and n != "encoder_pma"]
    dec = [n for n in reg if n.startswith("decoder_")]
    cond: Dict[str, Optional[List[str]]] = {
        "pretrained": [],
        "output_head": ["output_head"],
    }
    for name in enc:
        cond[name] = [name]
    for name in dec:
        cond[name] = [name]
    cond["all_params"] = None
    return cond


def eval_ce_loss(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            nums, tokens = batch[0].to(device), batch[1].to(device)
            output, trg = model.forward([nums, tokens])
            losses.append(float(model.compute_loss(output, trg).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def make_eval_fit_params(params_fit, beam_size: int, n_restarts: int, stop_time: float):
    """Copy FitParams with capped BFGS for contribution scans."""
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


def log(msg: str) -> None:
    print(msg, flush=True)


def eval_problems(
    model,
    params_fit,
    problems,
) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    per: List[Dict[str, Any]] = []
    keys = [
        "nmse",
        "nmse_var",
        "r2",
        "var_f1",
        "var_precision",
        "var_recall",
        "sym_recovery",
        "sym_skeleton",
        "complexity",
        "valid_pred",
    ]
    buckets: Dict[str, List[float]] = {k: [] for k in keys}

    for ds in problems:
        true_expr = instantiate_expr(ds)
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
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names)
        sc = score_prediction(
            ds.y, y_hat, expr, ds.spec.variable_names, true_expr=true_expr
        )
        row = {"eq_id": ds.spec.eq_id, "pred": expr, "true": true_expr, **sc}
        per.append(row)
        for k in keys:
            v = sc[k]
            if np.isfinite(v):
                buckets[k].append(float(v))

    agg: Dict[str, float] = {
        "n_eval": float(len(problems)),
        "n_valid": float(len(buckets["nmse"])),
    }
    for k, vals in buckets.items():
        if not vals:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_median"] = float("nan")
            continue
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_median"] = float(np.median(vals))
    # rates: mean of binary / continuous recovery metrics
    agg["sym_rate"] = agg.get("sym_recovery_mean", float("nan"))
    agg["var_f1"] = agg.get("var_f1_mean", float("nan"))
    agg["r2"] = agg.get("r2_median", float("nan"))
    agg["nmse"] = agg.get("nmse_median", float("nan"))
    return {"aggregate": agg, "per_problem": per}


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 = all test problems")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument(
        "--bfgs-stop-time",
        type=float,
        default=0.5,
        help="BFGS wall budget per equation (seconds)",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated subset (empty = Phase-4 default set)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size)
    fit_eval = make_eval_fit_params(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )

    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    word2id = eq_setting["word2id"]

    train_problems = load_split_problems(DATA_DIR, "train")
    test_problems = load_split_problems(DATA_DIR, "test")
    if args.eval_limit > 0:
        test_problems = test_problems[: args.eval_limit]

    train_ds = GRNFinetuneDataset(
        train_problems, word2id, max_points=args.max_points, seed=0
    )
    val_ds = GRNFinetuneDataset(
        test_problems, word2id, max_points=args.max_points, seed=1
    )
    log(f"Train FT examples: {len(train_ds)} / {len(train_problems)}")
    log(f"Eval problems: {len(test_problems)}; val CE examples: {len(val_ds)}")
    if len(train_ds) == 0:
        log("No tokenizable train equations.")
        return 1

    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=collate_finetune,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, max(len(val_ds), 1)),
        shuffle=False,
        collate_fn=collate_finetune,
    )

    conditions = build_phase4_conditions(base_model)
    if args.conditions.strip():
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = {k: v for k, v in conditions.items() if k in wanted}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for name, layers in conditions.items():
        log(f"\n=== Condition: {name} | layers={layers} ===")
        t0 = time.time()
        model = clone_model(base_model)

        if name == "pretrained" or layers == []:
            train_info = {
                "final_loss": float("nan"),
                "trainable": 0.0,
                "total": float(sum(p.numel() for p in model.parameters())),
                "epochs": 0.0,
                "trainable_fraction": 0.0,
            }
        else:
            train_info = train_selective(
                model,
                loader,
                layers,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )

        model.eval()
        val_ce = eval_ce_loss(model, val_loader, device) if len(val_ds) else float("nan")
        decoded = eval_problems(model, fit_eval, test_problems)
        agg = decoded["aggregate"]
        agg["val_ce"] = val_ce
        elapsed = time.time() - t0
        row = {
            "condition": name,
            "layers": layers,
            "train": train_info,
            "eval": agg,
            "per_problem": decoded["per_problem"],
            "elapsed_sec": elapsed,
        }
        results.append(row)
        log(
            f"  val_CE={fmt(val_ce)}  NMSE_med={fmt(agg['nmse'])}  "
            f"R2_med={fmt(agg['r2'])}  varF1={fmt(agg['var_f1'])}  "
            f"sym={fmt(agg['sym_rate'])}  ({elapsed:.1f}s)"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_json = OUT_DIR / "layer_contribution.json"
    # JSON-safe: replace inf/-inf in nested structures via default
    def _sanitize(obj):
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    out_json.write_text(json.dumps(_sanitize(results), indent=2), encoding="utf-8")

    # Build metric score maps
    metric_specs = [
        ("val_ce", False, "Cross-entropy (token teacher-forcing)"),
        ("nmse", False, "Prediction NMSE (median, lower better)"),
        ("r2", True, "Prediction R² (median, higher better)"),
        ("var_f1", True, "Variable recovery F1 (mean)"),
        ("sym_rate", True, "Symbolic recovery rate (mean)"),
    ]

    contrib_tables: Dict[str, Dict[str, float]] = {}
    for key, higher, _desc in metric_specs:
        scores = {}
        for r in results:
            scores[r["condition"]] = float(r["eval"].get(key, float("nan")))
        if "pretrained" not in scores or "all_params" not in scores:
            continue
        try:
            contrib_tables[key] = compute_contributions(
                scores, higher_is_better=higher
            )
        except KeyError:
            continue

    contrib_path = OUT_DIR / "contributions.json"
    contrib_path.write_text(
        json.dumps(_sanitize(contrib_tables), indent=2), encoding="utf-8"
    )

    # Report
    lines = [
        "# Phase 4: layer contribution",
        "",
        f"- Train FT examples: {len(train_ds)} (Phase 1 train)",
        f"- Eval problems: {len(test_problems)} test",
        f"- Epochs: {args.epochs}, lr: {args.lr}",
        f"- Decode: beam={args.beam_size}, BFGS restarts={args.bfgs_restarts}, "
        f"stop_time={args.bfgs_stop_time}s",
        f"- Device: `{device}`",
        f"- Raw results: `{out_json.as_posix()}`",
        f"- Contributions: `{contrib_path.as_posix()}`",
        "",
        "## Raw scores",
        "",
        "| condition | trainable | val CE | NMSE med | R² med | var F1 | sym rate | time (s) |",
        "|-----------|-----------|--------|----------|--------|--------|----------|----------|",
    ]
    for r in results:
        e = r["eval"]
        lines.append(
            f"| `{r['condition']}` | {int(r['train']['trainable']):,} | "
            f"{fmt(e['val_ce'])} | {fmt(e['nmse'])} | {fmt(e['r2'])} | "
            f"{fmt(e['var_f1'])} | {fmt(e['sym_rate'])} | {r['elapsed_sec']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Layer contribution (separate metrics; plan §Phase 4)",
            "",
            "Formulas:",
            "",
            "- Higher-better: `C = (S_k - S_base) / (S_full - S_base)`",
            "- Lower-better: `C = (L_base - L_k) / (L_base - L_full)`",
            "",
            "`S_base` / `L_base` = `pretrained`, `S_full` / `L_full` = `all_params`.",
            "",
        ]
    )

    for key, higher, desc in metric_specs:
        if key not in contrib_tables:
            continue
        ranked = rank_by_contribution(contrib_tables[key], descending=True)
        direction = "higher better raw → C above" if higher else "lower better raw → C above"
        lines.append(f"### {key} — {desc}")
        lines.append("")
        lines.append(f"({direction})")
        lines.append("")
        lines.append("| rank | condition | C |")
        lines.append("|------|-----------|---|")
        for i, (name, c) in enumerate(ranked, 1):
            lines.append(f"| {i} | `{name}` | {fmt(c)} |")
        lines.append("")

    # Consensus ranking: average rank across available C tables
    layer_names = [
        r["condition"]
        for r in results
        if r["condition"] not in ("pretrained", "all_params")
    ]
    rank_sums: Dict[str, float] = {n: 0.0 for n in layer_names}
    rank_counts: Dict[str, int] = {n: 0 for n in layer_names}
    for key, table in contrib_tables.items():
        ranked = rank_by_contribution(table)
        for i, (name, _) in enumerate(ranked, 1):
            if name in rank_sums:
                rank_sums[name] += i
                rank_counts[name] += 1
    consensus = []
    for name in layer_names:
        if rank_counts[name]:
            consensus.append((name, rank_sums[name] / rank_counts[name]))
    consensus.sort(key=lambda x: x[1])

    lines.extend(
        [
            "## Consensus ranking (mean rank across metrics)",
            "",
            "| rank | condition | mean metric-rank |",
            "|------|-----------|------------------|",
        ]
    )
    for i, (name, mean_r) in enumerate(consensus, 1):
        lines.append(f"| {i} | `{name}` | {mean_r:.2f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Primary Phase-4 claim uses **separate** contribution tables (accuracy / "
            "symbolic / variable), not a single weighted composite.",
            "- Symbolic recovery = max(exact, skeleton-equivalent after constant→c, sympy equiv).",
            "- Variable recovery = presence of true `x_i` names in predicted string (F1).",
            "- Decode BFGS defaults are light (`beam=1`, `restarts=1`, `stop_time=0.5`) "
            "for scan throughput; raise `--bfgs-restarts` / `--bfgs-stop-time` for paper numbers.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {contrib_path}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

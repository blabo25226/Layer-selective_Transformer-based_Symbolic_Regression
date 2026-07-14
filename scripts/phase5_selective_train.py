"""Phase 5: train only high-contribution layers vs mid / random / bottom / full."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.selective_layers import (  # noqa: E402
    build_phase5_conditions,
    load_phase4_ranking,
)
from training.single_layer import clone_model, train_selective  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
# Checkpoint/config env-overridable for GPU runs (e.g. LTSR_WEIGHTS=.../100M.ckpt)
WEIGHTS = Path(os.environ.get("LTSR_WEIGHTS", str(ROOT / "NSRS" / "weights" / "10M.ckpt")))
CONFIG = Path(os.environ.get("LTSR_CONFIG", str(ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml")))
EQ_SETTING = Path(os.environ.get("LTSR_EQ_SETTING", str(ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json")))
OUT_DIR = ROOT / "results" / "phase_results" / "phase5"
REPORT = ROOT / "results" / "phase_results" / "phase5_report.md"
PHASE4_CONTRIB = ROOT / "results" / "phase_results" / "phase4" / "contributions.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}g}"


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


def eval_problems(model, params_fit, problems) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    keys = [
        "nmse",
        "r2",
        "var_f1",
        "sym_recovery",
        "complexity",
        "valid_pred",
    ]
    buckets: Dict[str, List[float]] = {k: [] for k in keys}
    per: List[Dict[str, Any]] = []

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
        per.append({"eq_id": ds.spec.eq_id, "pred": expr, "true": true_expr, **sc})
        for k in keys:
            v = sc[k]
            if np.isfinite(v):
                buckets[k].append(float(v))

    agg: Dict[str, float] = {
        "n_eval": float(len(problems)),
        "n_valid": float(len(buckets["nmse"])),
        "nmse": float(np.median(buckets["nmse"])) if buckets["nmse"] else float("inf"),
        "r2": float(np.median(buckets["r2"])) if buckets["r2"] else float("-inf"),
        "var_f1": float(np.mean(buckets["var_f1"])) if buckets["var_f1"] else float("nan"),
        "sym_rate": float(np.mean(buckets["sym_recovery"]))
        if buckets["sym_recovery"]
        else 0.0,
        "complexity": float(np.mean(buckets["complexity"]))
        if buckets["complexity"]
        else float("nan"),
    }
    return {"aggregate": agg, "per_problem": per}


def peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    # CPU: process RSS if available
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024**2))
    except Exception:
        return float("nan")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 = all test")
    parser.add_argument("--k", type=int, default=3, help="k for mid/random/bottom")
    parser.add_argument(
        "--ranking",
        choices=["accuracy", "ce"],
        default="accuracy",
        help="Layer order for top/mid/bottom (Phase 4)",
    )
    parser.add_argument(
        "--contributions",
        default=str(PHASE4_CONTRIB),
        help="Phase 4 contributions.json to derive the ranking from "
        "(falls back to frozen constant if missing)",
    )
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    parser.add_argument("--conditions", default="", help="Comma subset (empty=all)")
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Suite dir (default phase1_v1; use diverse_v1 for A-1 sample size)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | data: {data_dir}")

    ranking, ranking_source = load_phase4_ranking(Path(args.contributions), args.ranking)
    if ranking_source == "phase4":
        log(f"Ranking ({args.ranking}) from Phase 4 `{args.contributions}`: {ranking}")
    else:
        log(
            f"WARNING: Phase 4 contributions not found/invalid at "
            f"`{args.contributions}`; using frozen fallback ranking ({args.ranking}). "
            f"Run scripts/phase4_layer_contribution.py first for live rankings."
        )
        log(f"Ranking ({args.ranking}, fallback): {ranking}")

    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size)
    fit_eval = make_eval_fit_params(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )

    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    word2id = eq_setting["word2id"]

    train_problems = load_split_problems(data_dir, "train")
    test_problems = load_split_problems(data_dir, "test")
    if args.eval_limit > 0:
        test_problems = test_problems[: args.eval_limit]

    train_ds = GRNFinetuneDataset(train_problems, word2id, max_points=args.max_points, seed=0)
    val_ds = GRNFinetuneDataset(test_problems, word2id, max_points=args.max_points, seed=1)
    log(f"Train FT: {len(train_ds)}; eval test: {len(test_problems)}")
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

    conditions = build_phase5_conditions(ranking, k=args.k, random_seed=0)
    if args.conditions.strip():
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = {k: v for k, v in conditions.items() if k in wanted}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for name, layers in conditions.items():
        log(f"\n=== {name} | layers={layers} ===")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
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
        train_ce = (
            float("nan")
            if name == "pretrained"
            else eval_ce_loss(model, loader, device)
        )
        val_ce = eval_ce_loss(model, val_loader, device) if len(val_ds) else float("nan")
        decoded = eval_problems(model, fit_eval, test_problems)
        agg = decoded["aggregate"]
        elapsed = time.time() - t0
        gap = (
            float("nan")
            if (np.isnan(train_ce) or np.isnan(val_ce))
            else float(val_ce - train_ce)
        )
        row = {
            "condition": name,
            "layers": layers,
            "ranking": args.ranking,
            "train": train_info,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "overfit_gap": gap,
            "eval": agg,
            "per_problem": decoded["per_problem"],
            "elapsed_sec": elapsed,
            "peak_mem_mb": peak_mem_mb(device),
        }
        results.append(row)
        log(
            f"  params={int(train_info['trainable']):,}  "
            f"train_CE={fmt(train_ce)}  val_CE={fmt(val_ce)}  "
            f"gap={fmt(gap)}  NMSE={fmt(agg['nmse'])}  R2={fmt(agg['r2'])}  "
            f"sym={fmt(agg['sym_rate'])}  ({elapsed:.1f}s, mem={fmt(row['peak_mem_mb'])}MB)"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_json = OUT_DIR / "selective_results.json"
    out_json.write_text(json.dumps(sanitize(results), indent=2), encoding="utf-8")

    # Efficiency: recovery toward all_params NMSE / CE with fewer params
    by_name = {r["condition"]: r for r in results}
    base = by_name.get("pretrained")
    full = by_name.get("all_params")

    lines = [
        "# Phase 5: high-contribution selective fine-tuning",
        "",
        f"- Ranking mode: `{args.ranking}` "
        f"({'Phase 4 contributions.json' if ranking_source == 'phase4' else 'FROZEN FALLBACK'})",
        f"- Order: `{', '.join(ranking)}`",
        f"- k={args.k} for middle / random / bottom",
        f"- Train FT: {len(train_ds)}; test eval: {len(test_problems)}",
        f"- Epochs: {args.epochs}, lr: {args.lr}",
        f"- Decode: beam={args.beam_size}, BFGS restarts={args.bfgs_restarts}, "
        f"stop_time={args.bfgs_stop_time}s",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## Conditions",
        "",
        "| condition | layers |",
        "|-----------|--------|",
    ]
    for r in results:
        layers = r["layers"]
        layer_s = "ALL" if layers is None else (", ".join(layers) if layers else "—")
        lines.append(f"| `{r['condition']}` | {layer_s} |")

    lines.extend(
        [
            "",
            "## Scores",
            "",
            "| condition | trainable | frac | train CE | val CE | gap | "
            "NMSE | R² | sym | time (s) | mem (MB) |",
            "|-----------|-----------|------|----------|--------|-----|"
            "------|----|-----|----------|----------|",
        ]
    )
    for r in results:
        e = r["eval"]
        tr = r["train"]
        lines.append(
            f"| `{r['condition']}` | {int(tr['trainable']):,} | "
            f"{fmt(tr.get('trainable_fraction', 0))} | "
            f"{fmt(r['train_ce'])} | {fmt(r['val_ce'])} | {fmt(r['overfit_gap'])} | "
            f"{fmt(e['nmse'])} | {fmt(e['r2'])} | {fmt(e['sym_rate'])} | "
            f"{r['elapsed_sec']:.1f} | {fmt(r['peak_mem_mb'])} |"
        )

    if base and full:
        l_base = float(base["val_ce"])
        l_full = float(full["val_ce"])
        n_base = float(base["eval"]["nmse"])
        n_full = float(full["eval"]["nmse"])
        lines.extend(
            [
                "",
                "## Recovery vs full FT (efficiency)",
                "",
                f"- L_base / L_full val CE = {fmt(l_base)} / {fmt(l_full)}",
                f"- NMSE_base / NMSE_full = {fmt(n_base)} / {fmt(n_full)}",
                "",
                "| condition | C_CE | C_NMSE | trainable % |",
                "|-----------|------|--------|-------------|",
            ]
        )
        for r in results:
            if r["condition"] in ("pretrained", "all_params"):
                continue
            lk = float(r["val_ce"])
            nk = float(r["eval"]["nmse"])
            c_ce = (
                (l_base - lk) / (l_base - l_full) if abs(l_base - l_full) > 1e-12 else float("nan")
            )
            c_nmse = (
                (n_base - nk) / (n_base - n_full) if abs(n_base - n_full) > 1e-12 else float("nan")
            )
            frac = float(r["train"].get("trainable_fraction", 0.0))
            lines.append(
                f"| `{r['condition']}` | {fmt(c_ce)} | {fmt(c_nmse)} | {fmt(100 * frac)}% |"
            )

    # Rank by val CE then NMSE
    ranked = sorted(
        [r for r in results if r["condition"] != "pretrained"],
        key=lambda r: (r["val_ce"], r["eval"]["nmse"]),
    )
    lines.extend(
        [
            "",
            "## Ranking (val CE, then NMSE)",
            "",
            "| rank | condition | val CE | NMSE | trainable |",
            "|------|-----------|--------|------|-----------|",
        ]
    )
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{r['condition']}` | {fmt(r['val_ce'])} | "
            f"{fmt(r['eval']['nmse'])} | {int(r['train']['trainable']):,} |"
        )

    # --- H2 honest check: does the selected top-k actually beat the random control? ---
    def _cond_nmse(cname: str) -> float:
        r = by_name.get(cname)
        return float(r["eval"]["nmse"]) if r else float("nan")

    def _cond_ce(cname: str) -> float:
        r = by_name.get(cname)
        return float(r["val_ce"]) if r else float("nan")

    top_name = f"top_{args.k}"
    rand_name = f"random_{args.k}"
    mid_name = f"middle_{args.k}"
    top_nmse, rand_nmse = _cond_nmse(top_name), _cond_nmse(rand_name)
    top_ce, rand_ce = _cond_ce(top_name), _cond_ce(rand_name)
    lines.extend(
        [
            "",
            "## H2 check: selected layers vs random control",
            "",
            f"- `{top_name}` NMSE={fmt(top_nmse)}, val CE={fmt(top_ce)}",
            f"- `{rand_name}` NMSE={fmt(rand_nmse)}, val CE={fmt(rand_ce)} "
            "(random now excludes the top-3 layers — A-3 fix)",
            f"- `{mid_name}` NMSE={fmt(_cond_nmse(mid_name))}, val CE={fmt(_cond_ce(mid_name))}",
            "",
            (
                f"**Verdict:** top-k {'beats' if (np.isfinite(top_nmse) and np.isfinite(rand_nmse) and top_nmse < rand_nmse) else 'does NOT beat'} "
                "the random control on prediction NMSE"
                + (
                    f" (Δ={fmt(rand_nmse - top_nmse)})."
                    if np.isfinite(top_nmse) and np.isfinite(rand_nmse)
                    else "."
                )
            ),
            "",
            "> ⚠️ **Statistical caveat (A-1):** this run uses a small train/eval set "
            f"({len(train_ds)} train / {len(test_problems)} eval equations, single seed). "
            "A top≈random gap of this size is NOT evidence for H2. Re-run "
            "`scripts/phase4_multiseed.py` + this script across ≥3 seeds and compare "
            "distributions (mean ± CI) before claiming layer selectivity.",
            "",
            "## Notes",
            "",
            "- Transfer proxy: Phase 1 train/test split uses OOD parameter ranges "
            "(same equation families).",
            "- Overfit gap = val_CE − train_CE (larger → more overfit).",
            "- Layer order derived from Phase 4 `contributions.json` (mode "
            f"`{args.ranking}`); `random_k` seed=0 excludes top-3 for a fair control.",
            "- Light BFGS decode; raise flags for paper-quality decode metrics.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

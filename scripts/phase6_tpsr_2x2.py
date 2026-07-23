"""Phase 6: 2x2 experiment — {pretrained, high-contrib FT} x {beam, TPSR}."""

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
from evaluation.aggregation import aggregate_prediction_scores, true_variables  # noqa: E402
from evaluation.equation_records import dataset_variable_mapping, make_equation_record  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from models.tpsr_adapter import predict_equation_tpsr  # noqa: E402
from training.selective_layers import resolve_selected_layers  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402
from experiment_runtime import phase_output_paths  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
# Checkpoint/config env-overridable for GPU runs (e.g. LTSR_WEIGHTS=.../100M.ckpt)
WEIGHTS = Path(os.environ.get("LTSR_WEIGHTS", str(ROOT / "NSRS" / "weights" / "10M.ckpt")))
CONFIG = Path(os.environ.get("LTSR_CONFIG", str(ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml")))
EQ_SETTING = Path(os.environ.get("LTSR_EQ_SETTING", str(ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json")))
OUT_DIR, REPORT = phase_output_paths(ROOT, "phase6", "phase6_report.md")

PHASE4_CONTRIB = ROOT / "results" / "phase_results" / "phase4_multiseed" / "contrib_aggregate.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{digits}g}"


def make_beam_fit_params(params_fit, beam_size: int, n_restarts: int, stop_time: float):
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


def eval_one(
    model,
    params_fit,
    problems,
    *,
    decode: str,
    tpsr_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    tpsr_kwargs = tpsr_kwargs or {}
    rows = []
    times = []

    for ds in problems:
        true_expr = instantiate_expr(ds)
        t0 = time.time()
        expr = ""
        meta: Dict[str, Any] = {}
        candidates: List[str] = []
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
                    candidates = out.get("all_preds", [])
                    meta = {"bfgs_loss": out.get("bfgs_loss")}
                else:
                    out = predict_equation_tpsr(
                        model, params_fit, ds.X, ds.y, quiet=True, **tpsr_kwargs
                    )
                    expr = out["equation"]
                    candidates = [expr] if expr else []
                    meta = {
                        "bfgs_loss": out.get("bfgs_loss"),
                        "reward": out.get("reward"),
                        "mcts_steps": out.get("mcts_steps"),
                        "sample_times": out.get("sample_times"),
                    }
        except Exception as exc:
            expr = ""
            failure_reason = f"{type(exc).__name__}: {exc}"
            meta = {"error": failure_reason}
        elapsed = time.time() - t0
        times.append(elapsed)
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        sc = score_prediction(
            ds.y, y_hat, expr, true_variables(true_expr, ds.spec.variable_names),
            true_expr=true_expr, X=ds.X, variable_names=ds.spec.variable_names,
        )
        rows.append(make_equation_record(
            eq_id=ds.spec.eq_id,
            predicted_expr=expr,
            variable_names=ds.spec.variable_names,
            mapping=dataset_variable_mapping(ds),
            scores=sc,
            true_expr=true_expr,
            candidate_expressions=candidates,
            decoder="nesymres_beam_bfgs" if decode == "beam" else "tpsr_mcts_bfgs",
            decoder_metadata=meta,
            failure_reason=failure_reason,
            elapsed_sec=elapsed,
        ))
    agg = aggregate_prediction_scores(rows)
    agg["mean_time_sec"] = float(np.mean(times)) if times else float("nan")
    agg["total_time_sec"] = float(np.sum(times)) if times else float("nan")
    return {"aggregate": agg, "per_problem": rows}


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
    parser.add_argument("--eval-limit", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    parser.add_argument("--rollout", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--layers",
        default="",
        help="Explicit comma layer set (overrides the Phase 4 top-k rule)",
    )
    parser.add_argument("--layer-rule", choices=["top", "middle", "bottom"], default="top")
    parser.add_argument("--layer-mode", choices=["accuracy", "ce"], default="accuracy")
    parser.add_argument("--layer-k", type=int, default=3)
    parser.add_argument("--contributions", default=str(PHASE4_CONTRIB))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explicit = [x.strip() for x in args.layers.split(",") if x.strip()] or None
    layers, layer_src, layer_rule = resolve_selected_layers(
        Path(args.contributions),
        mode=args.layer_mode,
        rule=args.layer_rule,
        k=args.layer_k,
        explicit=explicit,
    )
    log(f"Device: {device}")
    log(f"High-contrib layers ({layer_rule}, source={layer_src}): {layers}")

    base_model, params_fit = load_nesymres(
        WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size
    )
    beam_params = make_beam_fit_params(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )
    tpsr_kwargs = {
        "rollout": args.rollout,
        "horizon": args.horizon,
        "width": args.width,
        "num_beams": args.num_beams,
        "bfgs_restarts": args.bfgs_restarts,
        "bfgs_stop_time": args.bfgs_stop_time,
    }

    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    word2id = eq_setting["word2id"]

    train_problems = load_split_problems(DATA_DIR, "train")
    test_problems = load_split_problems(DATA_DIR, "test")
    if args.eval_limit > 0:
        test_problems = test_problems[: args.eval_limit]
    log(f"Train FT: {len(train_problems)}; eval: {len(test_problems)}")

    train_ds = GRNFinetuneDataset(
        train_problems, word2id, max_points=args.max_points, seed=0
    )
    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
    )

    # Fine-tune high-contrib model once
    log(f"\nFine-tuning selective layers ({args.epochs} epochs)...")
    ft_model = clone_model(base_model)
    train_info = train_selective(
        ft_model,
        loader,
        layers,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    log(
        f"  trainable={int(train_info['trainable']):,}  "
        f"final_CE={fmt(train_info['final_loss'])}"
    )

    cells = [
        ("pretrained_beam", clone_model(base_model), "beam"),
        ("pretrained_tpsr", clone_model(base_model), "tpsr"),
        ("selective_beam", ft_model, "beam"),
        ("selective_tpsr", clone_model(ft_model), "tpsr"),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for name, model, decode in cells:
        log(f"\n=== {name} ({decode}) ===")
        t0 = time.time()
        model.eval()
        ev = eval_one(
            model,
            beam_params if decode == "beam" else params_fit,
            test_problems,
            decode=decode,
            tpsr_kwargs=tpsr_kwargs if decode == "tpsr" else None,
        )
        elapsed = time.time() - t0
        agg = ev["aggregate"]
        row = {
            "condition": name,
            "finetune": "none" if name.startswith("pretrained") else "selective",
            "decode": decode,
            "layers": [] if name.startswith("pretrained") else layers,
            "train": train_info if not name.startswith("pretrained") else {
                "trainable": 0.0,
                "trainable_fraction": 0.0,
            },
            "eval": agg,
            "per_problem": ev["per_problem"],
            "elapsed_sec": elapsed,
        }
        results.append(row)
        log(
            f"  NMSE={fmt(agg['nmse'])}  R2={fmt(agg['r2'])}  "
            f"sym={fmt(agg['sym_rate'])}  "
            f"time/eq={fmt(agg['mean_time_sec'])}s  total={elapsed:.1f}s"
        )

    out_json = OUT_DIR / "tpsr_2x2.json"
    out_json.write_text(json.dumps(sanitize(results), indent=2), encoding="utf-8")

    # Interaction summary
    by = {r["condition"]: r for r in results}
    lines = [
        "# Phase 6: TPSR 2×2 (fine-tune × decode)",
        "",
        f"- High-contrib layers ({layer_rule}, source=`{layer_src}`): `{', '.join(layers)}`",
        f"- Train FT examples: {len(train_ds)}; eval test: {len(test_problems)}",
        f"- Epochs: {args.epochs}, lr: {args.lr}",
        f"- Beam BFGS: beam={args.beam_size}, restarts={args.bfgs_restarts}, "
        f"stop={args.bfgs_stop_time}s",
        f"- TPSR: rollout={args.rollout}, horizon={args.horizon}, "
        f"width={args.width}, num_beams={args.num_beams}",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## 2×2 results",
        "",
        "| Fine-tune | Decode | NMSE med | R² med | var F1 | sym | "
        "time/eq (s) | total (s) |",
        "|-----------|--------|----------|--------|--------|-----|"
        "------------|----------|",
    ]
    order = [
        "pretrained_beam",
        "pretrained_tpsr",
        "selective_beam",
        "selective_tpsr",
    ]
    labels = {
        "pretrained_beam": ("none", "beam"),
        "pretrained_tpsr": ("none", "TPSR"),
        "selective_beam": ("selective", "beam"),
        "selective_tpsr": ("selective", "TPSR"),
    }
    for key in order:
        r = by[key]
        e = r["eval"]
        ft, dec = labels[key]
        lines.append(
            f"| {ft} | {dec} | {fmt(e['nmse'])} | {fmt(e['r2'])} | "
            f"{fmt(e['var_f1'])} | {fmt(e['sym_rate'])} | "
            f"{fmt(e['mean_time_sec'])} | {r['elapsed_sec']:.1f} |"
        )

    # Delta table: FT effect, TPSR effect, interaction
    def nmse(key: str) -> float:
        return float(by[key]["eval"]["penalized_nmse"])

    def r2(key: str) -> float:
        return float(by[key]["eval"]["r2"])

    lines.extend(
        [
            "",
            "## Effect decomposition (NMSE ↓ better)",
            "",
            f"- Δ FT | beam: NMSE(selective_beam) − NMSE(pretrained_beam) = "
            f"{fmt(nmse('selective_beam') - nmse('pretrained_beam'))}",
            f"- Δ TPSR | pretrained: NMSE(pretrained_tpsr) − NMSE(pretrained_beam) = "
            f"{fmt(nmse('pretrained_tpsr') - nmse('pretrained_beam'))}",
            f"- Δ TPSR | selective: NMSE(selective_tpsr) − NMSE(selective_beam) = "
            f"{fmt(nmse('selective_tpsr') - nmse('selective_beam'))}",
            f"- Interaction (NMSE): "
            f"[selective_tpsr − selective_beam] − [pretrained_tpsr − pretrained_beam] = "
            f"{fmt((nmse('selective_tpsr') - nmse('selective_beam')) - (nmse('pretrained_tpsr') - nmse('pretrained_beam')))}",
            "",
            "## Effect decomposition (R² ↑ better)",
            "",
            f"- Δ FT | beam: {fmt(r2('selective_beam') - r2('pretrained_beam'))}",
            f"- Δ TPSR | pretrained: {fmt(r2('pretrained_tpsr') - r2('pretrained_beam'))}",
            f"- Δ TPSR | selective: {fmt(r2('selective_tpsr') - r2('selective_beam'))}",
            "",
            "## Notes",
            "",
            "- Plan Phase 6: separate FT gain, MCTS gain, and interaction.",
            "- TPSR uses NeSymReS backbone + UCT (not E2E).",
            "- Light MCTS/BFGS budgets for CPU; raise `--rollout` / `--horizon` for paper runs.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

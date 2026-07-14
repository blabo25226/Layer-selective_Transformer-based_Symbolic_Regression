"""Phase 3: selective-layer fine-tuning scan on NeSymReS."""

from __future__ import annotations

import argparse
import json
import random
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
    load_split_problems,
)
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from models.layer_selector import get_layer_registry  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase3"
REPORT = ROOT / "results" / "phase_results" / "phase3_report.md"


def build_conditions(model) -> Dict[str, Optional[List[str]]]:
    reg = get_layer_registry(model)
    enc = [n for n in reg if n.startswith("encoder_") and n != "encoder_pma"]
    dec = [n for n in reg if n.startswith("decoder_")]
    cond: Dict[str, Optional[List[str]]] = {
        "pretrained": [],  # no training
        "output_head": ["output_head"],
    }
    for name in enc:
        cond[name] = [name]
    for name in dec:
        cond[name] = [name]
    # middle encoder / decoder
    if enc:
        mid_e = enc[len(enc) // 2]
        cond["middle_encoder"] = [mid_e]
    if dec:
        mid_d = dec[len(dec) // 2]
        cond["middle_decoder"] = [mid_d]
    # random one encoder + one decoder (fixed seed)
    rng = random.Random(0)
    if enc and dec:
        cond["random_one_each"] = [rng.choice(enc), rng.choice(dec)]
    cond["all_encoder_blocks"] = enc
    cond["all_decoder_blocks"] = dec
    cond["all_params"] = None  # None => unfreeze all
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


def make_light_fit_params(params_fit):
    """Copy FitParams with cheap BFGS for scan-time NMSE."""
    from copy import deepcopy
    from nesymres.dclasses import BFGSParams

    p = deepcopy(params_fit)
    p.beam_size = 1
    p.bfgs = BFGSParams(
        activated=True,
        n_restarts=1,
        add_coefficients_if_not_existing=False,
        normalization_o=False,
        idx_remove=True,
        normalization_type="MSE",
        stop_time=0.5,
    )
    return p


def eval_nmse_on_problems(model, params_fit, problems) -> Dict[str, float]:
    import contextlib
    import io
    import warnings

    light = make_light_fit_params(params_fit)
    nmses = []
    r2s = []
    for ds in problems:
        expr = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    out = predict_equation(model, light, ds.X, ds.y, quiet=True)
                expr = out["equation"]
        except Exception:
            expr = ""
        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names)
        sc = score_prediction(ds.y, y_hat, expr, ds.spec.variable_names)
        if np.isfinite(sc["nmse"]):
            nmses.append(sc["nmse"])
        if np.isfinite(sc["r2"]):
            r2s.append(sc["r2"])
    return {
        "nmse_median": float(np.median(nmses)) if nmses else float("inf"),
        "nmse_mean": float(np.mean(nmses)) if nmses else float("inf"),
        "r2_median": float(np.median(r2s)) if r2s else float("-inf"),
        "n_eval": float(len(problems)),
        "n_valid": float(len(nmses)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=4, help="Test problems for NMSE eval")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated subset of condition names (empty=all)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=2)
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
    print(f"Train FT examples: {len(train_ds)} / {len(train_problems)}")
    print(f"Val CE examples: {len(val_ds)}")
    if len(train_ds) == 0:
        print("No tokenizable train equations.")
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

    conditions = build_conditions(base_model)
    if args.conditions.strip():
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = {k: v for k, v in conditions.items() if k in wanted}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for name, layers in conditions.items():
        print(f"\n=== Condition: {name} | layers={layers} ===")
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
        metrics = eval_nmse_on_problems(model, params_fit, test_problems)
        metrics["val_ce"] = val_ce
        elapsed = time.time() - t0
        row = {
            "condition": name,
            "layers": layers,
            "train": train_info,
            "eval": metrics,
            "elapsed_sec": elapsed,
        }
        results.append(row)
        print(
            f"  train_CE={train_info['final_loss']:.4g}  "
            f"val_CE={val_ce:.4g}  "
            f"trainable={int(train_info['trainable']):,}  "
            f"NMSE_med={metrics['nmse_median']:.4g}  "
            f"({elapsed:.1f}s)"
        )

        # free memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_json = OUT_DIR / "layer_scan.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Primary ranking: validation CE (lower better)
    ranked = sorted(results, key=lambda r: r["eval"].get("val_ce", float("inf")))
    base = next(r for r in results if r["condition"] == "pretrained")
    base_ce = base["eval"]["val_ce"]
    base_nmse = base["eval"]["nmse_median"]
    full = next((r for r in results if r["condition"] == "all_params"), None)

    lines = [
        "# Phase 3: selective-layer fine-tuning",
        "",
        f"- Train problems: {len(train_ds)} tokenized (from Phase 1 train)",
        f"- Eval problems: {len(test_problems)} test (NMSE); val CE on tokenized test",
        f"- Epochs: {args.epochs}, lr: {args.lr}, max_points: {args.max_points}",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## Ranking by validation CE (lower is better)",
        "",
        "| rank | condition | trainable | train CE | val CE | NMSE med | time (s) |",
        "|------|-----------|-----------|----------|--------|----------|----------|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | `{r['condition']}` | {int(r['train']['trainable']):,} | "
            f"{r['train']['final_loss']:.4g} | {r['eval']['val_ce']:.4g} | "
            f"{r['eval']['nmse_median']:.4g} | {r['elapsed_sec']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Layer contribution preview (CE recovery toward all_params)",
            "",
            f"- L_base (pretrained) val CE = {base_ce:.4g}",
            f"- pretrained NMSE med = {base_nmse:.4g}",
        ]
    )
    if full is not None:
        l_full = full["eval"]["val_ce"]
        lines.append(f"- L_full (all_params) val CE = {l_full:.4g}")
        denom = base_ce - l_full
        lines.append("")
        lines.append("| condition | C_CE = (L_base - L_k) / (L_base - L_full) |")
        lines.append("|-----------|------------------------------------------|")
        for r in results:
            if r["condition"] in ("pretrained", "all_params"):
                continue
            lk = r["eval"]["val_ce"]
            if abs(denom) < 1e-12:
                c = float("nan")
            else:
                c = (base_ce - lk) / denom
            lines.append(f"| `{r['condition']}` | {c:.4g} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Fine-tuning uses teacher-forcing CE on GRN equation skeletons (constants -> `c`).",
            "- Gradients flow through the full model; only selected layers have `requires_grad=True`.",
            "- NMSE uses light BFGS (1 restart, 0.5s) for scan speed; Phase 4 should use fuller decoding.",
            "- LoRA not included in v1.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

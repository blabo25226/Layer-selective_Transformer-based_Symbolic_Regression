"""Phase 2 baselines: pretrained NeSymReS (+beam) and PySR on Phase 1 suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.synthetic_grn import load_problem  # noqa: E402
from evaluation.equation_metrics import (  # noqa: E402
    eval_expression,
    score_prediction,
)
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
OUT_DIR = ROOT / "results" / "phase_results" / "phase2"
REPORT = ROOT / "results" / "phase_results" / "phase2_report.md"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"


def make_extrapolation_X(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample points outside training support [0,3] -> [3,5]."""
    n, d = X.shape
    return rng.uniform(3.0, 5.0, size=(n, d))


def true_y_from_spec(ds, X: np.ndarray) -> np.ndarray:
    """Recompute clean RHS labels for arbitrary X using stored parameters."""
    from data.synthetic_grn import (
        hill_activation,
        hill_repression,
        repressilator_dxi,
        toggle_dx,
    )

    p = ds.spec.parameters
    fam = ds.spec.family
    if fam == "activation":
        return hill_activation(X[:, 0], X[:, 1], p["alpha"], p["K"], p["n"], p["beta"])
    if fam == "repression":
        return hill_repression(X[:, 0], X[:, 1], p["alpha"], p["K"], p["n"], p["beta"])
    if fam == "toggle":
        if ds.spec.motif.endswith("dx"):
            return toggle_dx(X[:, 0], X[:, 1], p["alpha1"], p["n1"], p["beta1"])
        return toggle_dx(X[:, 1], X[:, 0], p["alpha2"], p["n2"], p["beta2"])
    if fam == "repressilator":
        target_idx = int(p["target_idx"])
        repressors = {0: 2, 1: 0, 2: 1}
        return repressilator_dxi(
            X[:, target_idx],
            X[:, repressors[target_idx]],
            p["alpha"],
            p["n"],
            p["beta"],
        )
    raise ValueError(fam)


def run_pysr(X: np.ndarray, y: np.ndarray, variable_names: List[str], niterations: int) -> str:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=25,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        random_state=0,
    )
    model.fit(X, y, variable_names=variable_names)
    return str(model.get_best()["equation"])


def evaluate_method(
    name: str,
    eq_id: str,
    ds,
    expr: str,
    elapsed: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    var_names = ds.spec.variable_names
    y_hat = eval_expression(expr, ds.X, var_names)
    scores_id = score_prediction(ds.y, y_hat, expr, var_names)

    X_ood = make_extrapolation_X(ds.X, rng)
    y_ood = true_y_from_spec(ds, X_ood)
    y_hat_ood = eval_expression(expr, X_ood, var_names)
    scores_ood = score_prediction(y_ood, y_hat_ood, expr, var_names)

    return {
        "method": name,
        "eq_id": eq_id,
        "family": ds.spec.family,
        "split": ds.spec.split,
        "equation": expr,
        "elapsed_sec": elapsed,
        "id": scores_id,
        "ood": scores_ood,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    summary = {}
    for method, items in by_method.items():
        nmse_id = [x["id"]["nmse"] for x in items if np.isfinite(x["id"]["nmse"])]
        nmse_ood = [x["ood"]["nmse"] for x in items if np.isfinite(x["ood"]["nmse"])]
        r2_id = [x["id"]["r2"] for x in items if np.isfinite(x["id"]["r2"])]
        var_f1 = [x["id"]["var_f1"] for x in items]
        valid = [x["id"]["valid_pred"] for x in items]
        summary[method] = {
            "n": len(items),
            "valid_rate": float(np.mean(valid)) if valid else 0.0,
            "nmse_id_median": float(np.median(nmse_id)) if nmse_id else None,
            "nmse_ood_median": float(np.median(nmse_ood)) if nmse_ood else None,
            "r2_id_median": float(np.median(r2_id)) if r2_id else None,
            "var_f1_mean": float(np.mean(var_f1)) if var_f1 else None,
            "elapsed_mean": float(np.mean([x["elapsed_sec"] for x in items])),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--limit", type=int, default=0, help="Max problems (0=all)")
    parser.add_argument("--pysr-iters", type=int, default=20)
    parser.add_argument("--skip-pysr", action="store_true")
    parser.add_argument("--skip-nesymres", action="store_true")
    parser.add_argument("--beam-sizes", default="2,5")
    args = parser.parse_args()

    index = json.loads((DATA_DIR / "index.json").read_text(encoding="utf-8"))
    if args.split != "all":
        index = [x for x in index if x["split"] == args.split]
    if args.limit > 0:
        index = index[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "baseline_results.jsonl"
    summary_path = OUT_DIR / "baseline_summary.json"

    rng = np.random.default_rng(0)
    rows: List[Dict[str, Any]] = []

    nesym_models = {}
    if not args.skip_nesymres:
        for b in [int(x) for x in args.beam_sizes.split(",") if x.strip()]:
            print(f"Loading NeSymReS with beam_size={b}...")
            nesym_models[b] = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=b)

    for item in index:
        ds = load_problem(DATA_DIR / item["file"])
        print(f"\n=== {ds.spec.eq_id} ({ds.spec.family}) ===")

        if not args.skip_nesymres:
            for beam, (model, params) in nesym_models.items():
                method = f"nesymres_beam{beam}"
                t0 = time.time()
                try:
                    out = predict_equation(model, params, ds.X, ds.y, quiet=True)
                    expr = out["equation"]
                except Exception as exc:
                    print(f"  {method} FAILED: {exc}")
                    expr = ""
                elapsed = time.time() - t0
                row = evaluate_method(method, ds.spec.eq_id, ds, expr, elapsed, rng)
                rows.append(row)
                print(
                    f"  {method}: NMSE_id={row['id']['nmse']:.4g} "
                    f"expr={expr[:80]!r} ({elapsed:.1f}s)"
                )

        if not args.skip_pysr:
            method = "pysr"
            t0 = time.time()
            try:
                expr = run_pysr(ds.X, ds.y, ds.spec.variable_names, args.pysr_iters)
            except Exception as exc:
                print(f"  {method} FAILED: {exc}")
                expr = ""
            elapsed = time.time() - t0
            row = evaluate_method(method, ds.spec.eq_id, ds, expr, elapsed, rng)
            rows.append(row)
            print(
                f"  {method}: NMSE_id={row['id']['nmse']:.4g} "
                f"expr={expr[:80]!r} ({elapsed:.1f}s)"
            )

    with results_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Phase 2: baseline evaluation",
        "",
        f"- Suite: `{DATA_DIR.as_posix()}`",
        f"- Split filter: `{args.split}`",
        f"- Problems run: {len(index)}",
        f"- Results: `{results_path.as_posix()}`",
        "",
        "## Methods",
        "",
        "- `nesymres_beam2`: pretrained NeSymReS, beam_size=2 (default)",
        "- `nesymres_beam5`: pretrained NeSymReS, beam_size=5",
        "- `pysr`: PySR baseline",
        "- TPSR: deferred to a follow-up run (heavy; adapter later)",
        "",
        "## Summary",
        "",
        "| method | n | valid | median NMSE (ID) | median NMSE (OOD) | median R² (ID) | mean var-F1 | mean time (s) |",
        "|--------|---|-------|------------------|-------------------|----------------|-------------|---------------|",
    ]
    for method, s in summary.items():
        lines.append(
            f"| `{method}` | {s['n']} | {s['valid_rate']:.2f} | "
            f"{s['nmse_id_median']} | {s['nmse_ood_median']} | "
            f"{s['r2_id_median']} | {s['var_f1_mean']} | {s['elapsed_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Note",
            "",
            "Pretrained NeSymReS is expected to underperform on GRN-style ODEs;",
            "Phase 2 establishes that baseline before layer-selective fine-tuning (Phase 3+).",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

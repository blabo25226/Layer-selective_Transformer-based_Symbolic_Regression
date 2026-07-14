"""Supervised selective FT from DREAM4 SBML-reconstructed teacher ODEs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.dream4 import (  # noqa: E402
    build_dream4_local_problems,
    find_dream4_root,
    finite_difference_rhs,
    load_dream4_expression_bundle,
    load_dream4_network,
    load_timeseries,
)
from data.dream4_sbml import (  # noqa: E402
    expression_string,
    parse_sbml_gene_odes,
    sample_supervised_points,
    sbml_path_for,
)
from data.finetune_dataset import (  # noqa: E402
    GRNFinetuneDataset,
    collate_finetune,
    expression_to_tokens,
    points_to_nesymres_tensor,
)
from data.synthetic_grn import EquationSpec, SampledDataset  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from models.nesymres_adapter import load_nesymres, predict_equation  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

DREAM4 = ROOT / "data" / "dream4"
WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
EQ_SETTING = ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
OUT_DIR = ROOT / "results" / "phase_results" / "phase7_sbml_ft"
REPORT = ROOT / "results" / "phase_results" / "phase7_sbml_ft_report.md"

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


def build_sbml_ft_problems(
    odes,
    gene_names: Sequence[str],
    X: np.ndarray,
    Y: np.ndarray,
    *,
    max_parents: int = 2,
    max_vars: int = 3,
    net_id: int = 1,
) -> List[SampledDataset]:
    """Oracle local problems with SBML numeric teacher expressions as motif."""
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    problems = []
    for g in gene_names:
        ode = odes[g]
        parents = [gene_to_idx[m] for m in ode.modifiers]
        if len(parents) > max_parents:
            continue
        t = ode.gene_idx
        cols: List[int] = [t]
        for r in parents:
            if r not in cols:
                cols.append(r)
        cols = cols[:max_vars]
        local_map = {}
        for i, gi in enumerate(cols):
            local_map[gene_names[gi]] = f"x_{i+1}"
        # Ensure target gene in map
        local_map[g] = local_map.get(g, "x_1")
        numeric = expression_string(ode, local_map=local_map, numeric=True)
        skeleton = expression_string(ode, local_map=local_map, numeric=False)
        Xloc = X[:, cols]
        y = Y[:, t]
        var_names = [f"x_{i+1}" for i in range(len(cols))]
        problems.append(
            SampledDataset(
                spec=EquationSpec(
                    eq_id=f"sbml_s10_{net_id}_{g}",
                    family="dream4_sbml",
                    target_expr=skeleton,
                    variable_names=var_names,
                    parameters={"target_gene": float(t), "net_id": float(net_id)},
                    split="train",
                    motif=numeric,
                ),
                X=Xloc.astype(float),
                y=y.astype(float),
                noise_std=0.0,
            )
        )
    return problems


# extend instantiate for dream4_sbml
def _patch_instantiate():
    from data import finetune_dataset as fd

    _orig = fd.instantiate_expr

    def instantiate_expr(ds: SampledDataset) -> str:
        if ds.spec.family == "dream4_sbml":
            return ds.spec.motif
        return _orig(ds)

    fd.instantiate_expr = instantiate_expr  # type: ignore


def eval_sr(model, params_fit, problems) -> Dict[str, Any]:
    import contextlib
    import io
    import warnings

    rows = []
    for ds in problems:
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
        sc = score_prediction(ds.y, y_hat, expr, ds.spec.variable_names, true_expr="")
        rows.append({"eq_id": ds.spec.eq_id, "pred": expr, **sc})
    return {
        "aggregate": aggregate_prediction_scores(rows),
        "per_problem": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dream4-root", type=Path, default=DREAM4)
    parser.add_argument("--net-id", type=int, default=1)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--max-parents", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-targets", type=int, default=10)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    args = parser.parse_args()

    # instantiate_expr supports dream4_sbml via motif
    root = find_dream4_root(args.dream4_root)
    xml = sbml_path_for(root, 10, args.net_id)
    log(f"SBML: {xml}")
    odes = parse_sbml_gene_odes(xml)
    gene_names = [f"G{i}" for i in range(1, 11)]

    X_sup, Y_sup = sample_supervised_points(
        odes, gene_names, n_points=args.n_points, seed=args.net_id
    )
    ft_problems = build_sbml_ft_problems(
        odes,
        gene_names,
        X_sup,
        Y_sup,
        max_parents=args.max_parents,
        net_id=args.net_id,
    )
    log(f"SBML FT problems (<= {args.max_parents} parents): {len(ft_problems)}")
    for p in ft_problems:
        log(f"  {p.spec.eq_id}: {p.spec.motif[:100]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, params_fit = load_nesymres(
        WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size
    )
    fit = make_light_fit(
        params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time
    )
    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)

    train_ds = GRNFinetuneDataset(
        ft_problems, eq_setting["word2id"], max_points=args.max_points, seed=0
    )
    log(f"Tokenized FT examples: {len(train_ds)} / {len(ft_problems)}")
    if len(train_ds) == 0:
        log("No tokenizable SBML equations.")
        return 1

    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=collate_finetune,
    )
    ft_model = clone_model(base_model)
    train_info = train_selective(
        ft_model,
        loader,
        HIGH_CONTRIB,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    log(
        f"FT done: trainable={int(train_info['trainable']):,}  "
        f"CE={fmt(train_info['final_loss'])}"
    )

    # Hold-out supervised SR (same distribution as FT)
    rng = np.random.default_rng(99)
    X_te, Y_te = sample_supervised_points(
        odes, gene_names, n_points=120, seed=99 + args.net_id
    )
    te_problems = build_sbml_ft_problems(
        odes,
        gene_names,
        X_te,
        Y_te,
        max_parents=args.max_parents,
        net_id=args.net_id,
    )

    # Transfer eval: official noisy DREAM timeseries FD + oracle selection
    network = load_dream4_network(root, 10, args.net_id)
    bundle = load_dream4_expression_bundle(root, 10, args.net_id)
    X_d, Y_d = bundle["X_ts"], bundle["Y_ts"]
    dream_problems, _, _ = build_dream4_local_problems(
        network,
        X_d,
        Y_d,
        method="oracle",
        k=args.k,
        split="test",
        max_vars=3,
        target_limit=args.eval_targets,
        size_tag=10,
    )

    results = {}
    for name, model, probs in [
        ("pretrained_sbml_holdout", clone_model(base_model), te_problems),
        ("sbml_ft_holdout", ft_model, te_problems),
        ("pretrained_dream_fd", clone_model(base_model), dream_problems),
        ("sbml_ft_dream_fd", clone_model(ft_model), dream_problems),
    ]:
        log(f"\n=== {name} n={len(probs)} ===")
        t0 = time.time()
        model.eval()
        ev = eval_sr(model, fit, probs)
        ev["elapsed_sec"] = time.time() - t0
        a = ev["aggregate"]
        log(f"  NMSE={fmt(a['nmse'])}  R2={fmt(a['r2'])}  ({ev['elapsed_sec']:.1f}s)")
        results[name] = ev

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "sbml": str(xml),
        "n_ft_problems": len(ft_problems),
        "n_tokenized": len(train_ds),
        "train": train_info,
        "layers": HIGH_CONTRIB,
        "eval": results,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_json = OUT_DIR / f"size10_net{args.net_id}_sbml_ft.json"
    out_json.write_text(json.dumps(sanitize(out), indent=2), encoding="utf-8")

    lines = [
        "# Phase 7d: SBML-supervised selective fine-tuning (DREAM4 Size10)",
        "",
        f"- SBML: `{xml.as_posix()}`",
        f"- Teacher ODEs reconstructed from GNW parameters "
        f"(Hill modules; protein≈mRNA quasi-steady proxy)",
        f"- FT problems: {len(ft_problems)} genes with ≤{args.max_parents} parents "
        f"({len(train_ds)} tokenized)",
        f"- Layers: `{', '.join(HIGH_CONTRIB)}`, epochs={args.epochs}",
        f"- Device: `{device}`",
        f"- Results: `{out_json.as_posix()}`",
        "",
        "## Results",
        "",
        "| condition | NMSE | R² | time (s) |",
        "|-----------|------|----|----------|",
    ]
    for key in (
        "pretrained_sbml_holdout",
        "sbml_ft_holdout",
        "pretrained_dream_fd",
        "sbml_ft_dream_fd",
    ):
        a = results[key]["aggregate"]
        lines.append(
            f"| `{key}` | {fmt(a['nmse'])} | {fmt(a['r2'])} | "
            f"{results[key]['elapsed_sec']:.1f} |"
        )

    a0 = results["pretrained_sbml_holdout"]["aggregate"]["nmse"]
    a1 = results["sbml_ft_holdout"]["aggregate"]["nmse"]
    b0 = results["pretrained_dream_fd"]["aggregate"]["nmse"]
    b1 = results["sbml_ft_dream_fd"]["aggregate"]["nmse"]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"1. **In-distribution (SBML teacher holdout):** "
            f"ΔNMSE = {fmt(a1 - a0)} (FT − pretrained).",
            f"2. **Transfer to noisy DREAM FD:** "
            f"ΔNMSE = {fmt(b1 - b0)}.",
            "3. SBML files lack MathML; reconstruction is Hill/module based on GNW params.",
            "4. Multi-regulator / constitutive dynamics remain approximate under the proxy.",
            "",
            "## Notes",
            "",
            "- Prefer genes with ≤2 parents for cleaner teacher strings.",
            "- Extend with `--net-id` 2..5 or Size100 SBML similarly.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out_json}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

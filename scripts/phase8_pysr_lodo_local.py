"""Run the Phase 8 PySR LODO baseline locally without importing NeSymReS.

This entry point is intentionally Python-3.12 compatible: the repository's
Hydra-pinned NeSymReS environment remains Python 3.10-only, while PySR and the
human-data/evaluation helpers do not require Hydra.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.pysr_runner import fit_pysr_expression  # noqa: E402
from data.dreamlike_grn import build_local_problem  # noqa: E402
from data.human import (  # noqa: E402
    build_human_local_problems,
    estimate_derivatives,
    prepare_gse112372,
)
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.equation_records import (  # noqa: E402
    dataset_variable_mapping,
    make_equation_record,
)
from evaluation.generalization import aggregate_lodo  # noqa: E402


def stack_donors(panel, donors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for donor in donors:
        X, Y = estimate_derivatives(
            panel.times, panel.X_donors[donor], method="smooth_fd"
        )
        Xs.append(X)
        Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, default=lambda value: None),
        encoding="utf-8",
    )
    os.replace(partial, path)


def run_fold(
    panel,
    *,
    holdout: str,
    seed: int,
    fold_index: int,
    niterations: int,
) -> dict[str, Any]:
    train_donors = [donor for donor in sorted(panel.X_donors) if donor != holdout]
    X_tr, Y_tr = stack_donors(panel, train_donors)
    X_te, Y_te = stack_donors(panel, [holdout])
    problems, selections, _ = build_human_local_problems(
        panel,
        X_tr,
        Y_tr,
        method="prior",
        k=2,
        max_vars=3,
        split="train",
    )
    network = panel.as_grn_like()
    in_rows, hold_rows = [], []
    for target_index, ds in enumerate(problems):
        random_state = seed * 10000 + fold_index * 100 + target_index
        failure_reason = None
        try:
            expr = fit_pysr_expression(
                ds.X,
                ds.y,
                ds.spec.variable_names,
                niterations=niterations,
                random_state=random_state,
            )
        except Exception as exc:
            expr = ""
            failure_reason = f"{type(exc).__name__}: {exc}"

        y_hat = eval_expression(expr, ds.X, ds.spec.variable_names) if expr else None
        scores = score_prediction(
            ds.y,
            y_hat,
            expr,
            ds.spec.variable_names,
            true_expr="",
            X=ds.X,
            variable_names=ds.spec.variable_names,
        )
        in_rows.append(
            make_equation_record(
                eq_id=ds.spec.eq_id,
                predicted_expr=expr,
                variable_names=ds.spec.variable_names,
                mapping=dataset_variable_mapping(ds, panel.gene_names),
                scores=scores,
                decoder="pysr",
                failure_reason=failure_reason,
                decoder_metadata={
                    "niterations": niterations,
                    "random_state": random_state,
                    "execution": "local_cpu",
                },
            )
        )

        motif = ds.spec.motif or ""
        target_name = (
            motif.split("target=")[-1].split(";")[0]
            if "target=" in motif
            else ""
        )
        target = panel.gene_index(target_name)
        heldout = build_local_problem(
            network,
            X_te,
            Y_te[:, target],
            target,
            selections.get(target, []),
            eq_id=f"{ds.spec.eq_id}_holdout",
            split="holdout",
            include_target=True,
            max_vars=3,
            selection_method="prior",
        )
        y_hat_hold = (
            eval_expression(expr, heldout.X, heldout.spec.variable_names)
            if expr
            else None
        )
        hold_scores = score_prediction(
            heldout.y,
            y_hat_hold,
            expr,
            heldout.spec.variable_names,
            true_expr="",
            X=heldout.X,
            variable_names=heldout.spec.variable_names,
        )
        hold_rows.append(
            make_equation_record(
                eq_id=heldout.spec.eq_id,
                predicted_expr=expr,
                variable_names=heldout.spec.variable_names,
                mapping=dataset_variable_mapping(heldout, panel.gene_names),
                scores=hold_scores,
                decoder="pysr",
                failure_reason=failure_reason,
                decoder_metadata={
                    "niterations": niterations,
                    "random_state": random_state,
                    "execution": "local_cpu",
                },
                target=target_name,
                reused_training_decode=True,
            )
        )

    in_aggregate = aggregate_prediction_scores(in_rows)
    hold_aggregate = aggregate_prediction_scores(hold_rows)
    return {
        "in": in_aggregate["penalized_nmse"],
        "hold": hold_aggregate["penalized_nmse"],
        "in_valid_rate": in_aggregate["valid_rate"],
        "hold_valid_rate": hold_aggregate["valid_rate"],
        "hold_near_singularity": hold_aggregate.get(
            "near_singularity_mean", float("nan")
        ),
        "hold_extrapolation_valid": hold_aggregate.get(
            "extrapolation_valid_mean", float("nan")
        ),
        "in_aggregate": in_aggregate,
        "hold_aggregate": hold_aggregate,
        "in_per_problem": in_rows,
        "hold_per_problem": hold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "human" / "gse112372_lps",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--pysr-iters", type=int, default=12)
    args = parser.parse_args()

    panel = prepare_gse112372(args.data_dir)
    donors = sorted(panel.X_donors)
    for seed in args.seeds:
        out_dir = args.run_dir / f"phase8_pysr_seed{seed}"
        checkpoint = out_dir / "fold_checkpoint.json"
        identity = {
            "schema_version": 1,
            "seed": seed,
            "donors": donors,
            "pysr_iterations": args.pysr_iters,
            "execution": "local_cpu",
        }
        if checkpoint.is_file():
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
            if state.get("identity") != identity:
                raise RuntimeError(f"checkpoint configuration mismatch: {checkpoint}")
            details = state.get("per_fold", [])
        else:
            details = []
        completed = {str(row["holdout"]) for row in details}
        for fold_index, holdout in enumerate(donors):
            if holdout in completed:
                print(f"seed={seed} holdout={holdout}: checkpoint complete")
                continue
            print(f"seed={seed} holdout={holdout}: starting", flush=True)
            result = run_fold(
                panel,
                holdout=holdout,
                seed=seed,
                fold_index=fold_index,
                niterations=args.pysr_iters,
            )
            details.append(
                {
                    "holdout": holdout,
                    "n_targets": len(result["in_per_problem"]),
                    "pysr": result,
                }
            )
            atomic_json(
                checkpoint, {"identity": identity, "per_fold": details}
            )

        folds = [{"pysr": row["pysr"]} for row in details]
        output = {
            "seed": seed,
            "execution": "local_cpu",
            "pysr_iterations": args.pysr_iters,
            "per_fold": details,
            "aggregate": aggregate_lodo(
                folds, metric="nmse", lower_better=True
            ),
        }
        atomic_json(out_dir / "pysr_results.json", output)
        print(f"seed={seed}: wrote {out_dir / 'pysr_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

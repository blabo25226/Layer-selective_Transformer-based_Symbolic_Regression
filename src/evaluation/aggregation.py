"""Shared, failure-aware aggregation for symbolic-regression evaluations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .equation_metrics import extract_variables


def true_variables(true_expr: str, fallback: Sequence[str]) -> list[str]:
    """Use variables that actually occur in the ground-truth expression."""
    found = sorted(extract_variables(true_expr)) if true_expr else []
    return found or list(fallback)


def aggregate_prediction_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    failure_nmse: float = 1e6,
    failure_r2: float = -1.0,
) -> Dict[str, float]:
    """Aggregate predictions without hiding failed decodes."""
    n_total = len(rows)
    valid_rows = [r for r in rows if float(r.get("valid_pred", 0.0)) == 1.0]
    n_valid = len(valid_rows)

    def finite_values(key: str, source=valid_rows) -> list[float]:
        values = []
        for row in source:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        return values

    out: Dict[str, float] = {
        "n_eval": float(n_total), "n_total": float(n_total),
        "n_valid": float(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else 0.0,
    }
    for key in ("nmse", "nmse_var", "r2", "complexity"):
        vals = finite_values(key)
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{key}_median"] = float(np.median(vals)) if vals else float("nan")
    for key in ("var_f1", "var_precision", "var_recall", "sym_recovery", "sym_skeleton"):
        vals = finite_values(key, rows)
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
    penalized_nmse = [
        float(r["nmse"]) if float(r.get("valid_pred", 0.0)) == 1.0 and math.isfinite(float(r["nmse"])) else failure_nmse
        for r in rows
    ]
    penalized_r2 = [
        float(r["r2"]) if float(r.get("valid_pred", 0.0)) == 1.0 and math.isfinite(float(r["r2"])) else failure_r2
        for r in rows
    ]
    out["penalized_nmse"] = float(np.median(penalized_nmse)) if penalized_nmse else float("nan")
    out["penalized_r2"] = float(np.median(penalized_r2)) if penalized_r2 else float("nan")
    out["valid_nmse"] = out["nmse_median"]
    out["valid_r2"] = out["r2_median"]
    # Primary aliases include every attempted problem; conditional metrics are
    # explicitly named ``valid_*`` to prevent accidental survivorship bias.
    out["nmse"] = out["penalized_nmse"]
    out["r2"] = out["penalized_r2"]
    out["var_f1"] = out.get("var_f1_mean", float("nan"))
    out["sym_rate"] = out.get("sym_recovery_mean", float("nan"))
    out["complexity"] = out["complexity_mean"]
    return out

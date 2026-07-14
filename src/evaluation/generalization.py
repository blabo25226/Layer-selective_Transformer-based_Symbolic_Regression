"""Cross-donor generalization metrics for Phase 8 (reviewer deep-dive).

The single most publishable Phase 8 signal is that selective-FT NeSymReS keeps its
accuracy on a *held-out donor* while PySR (which wins in-donor) collapses. One
holdout donor is an anecdote; leave-one-donor-out (LODO) turns it into a
cross-validated **generalization gap** = holdout_error − in_donor_error, which
these helpers aggregate across folds with a confidence interval.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


def generalization_gap(in_score: float, hold_score: float, *, lower_better: bool = True) -> float:
    """Positive gap = worse on holdout than in-donor (overfitting).

    For lower-is-better errors (NMSE) the gap is ``hold - in``; for higher-is-better
    scores (R²) it is ``in - hold`` so a positive gap always means "generalizes worse".
    """
    if lower_better:
        return hold_score - in_score
    return in_score - hold_score


_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _ci95(values: Sequence[float]) -> Dict[str, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci95": float("nan"), "n": 0.0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    sem = std / math.sqrt(n)
    critical = _T975.get(n - 1, 1.96) if n > 1 else float("nan")
    return {"mean": mean, "std": std, "sem": sem,
            "ci95": critical * sem if n > 1 else float("nan"),
            "n": float(n), "ci_method": "student_t"}


def aggregate_lodo(
    folds: Sequence[Mapping[str, Mapping[str, float]]],
    *,
    metric: str = "nmse",
    lower_better: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate leave-one-donor-out results per method.

    ``folds[i]`` maps ``method -> {"in": score, "hold": score}`` for the i-th
    held-out donor. Returns ``{method: {mean_in, mean_hold, gap_mean, gap_ci95,
    n_folds}}`` where ``gap`` is the generalization gap (see
    :func:`generalization_gap`).
    """
    methods: List[str] = []
    for fold in folds:
        for m in fold:
            if m not in methods:
                methods.append(m)

    out: Dict[str, Dict[str, float]] = {}
    for m in methods:
        ins, holds, gaps = [], [], []
        for fold in folds:
            if m not in fold:
                continue
            i = fold[m].get("in", float("nan"))
            h = fold[m].get("hold", float("nan"))
            ins.append(i)
            holds.append(h)
            if i is not None and h is not None and math.isfinite(i) and math.isfinite(h):
                gaps.append(generalization_gap(i, h, lower_better=lower_better))
        in_s = _ci95(ins)
        hold_s = _ci95(holds)
        gap_s = _ci95(gaps)
        out[m] = {
            "metric": metric,
            "mean_in": in_s["mean"],
            "mean_hold": hold_s["mean"],
            "hold_ci95": hold_s["ci95"],
            "gap_mean": gap_s["mean"],
            "gap_ci95": gap_s["ci95"],
            "n_folds": gap_s["n"],
        }
    return out


def rank_by_generalization(
    agg: Mapping[str, Mapping[str, float]],
    *,
    lower_better: bool = True,
) -> List[str]:
    """Order methods by mean holdout score (best generalizer first)."""
    def key(m: str):
        v = agg[m].get("mean_hold", float("nan"))
        if v is None or math.isnan(v):
            return (1, 0.0)
        return (0, v if lower_better else -v)

    return sorted(agg, key=key)

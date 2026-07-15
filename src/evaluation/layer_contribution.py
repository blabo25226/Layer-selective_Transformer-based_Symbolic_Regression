"""Layer contribution metrics (Phase 4 / Issue 9)."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional


def reference_improves(
    base: float,
    full: float,
    *,
    higher_is_better: bool,
    min_improvement: float = 1e-12,
) -> bool:
    """Whether full FT improves on pretrained in the metric's direction."""
    if not (math.isfinite(base) and math.isfinite(full)):
        return False
    improvement = full - base if higher_is_better else base - full
    return improvement > min_improvement


def absolute_improvements(
    scores: Mapping[str, float],
    *,
    base_key: str = "pretrained",
    higher_is_better: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Return signed raw improvements over pretrained without full-FT scaling."""
    if base_key not in scores:
        raise KeyError(f"Need '{base_key}' in scores")
    base = float(scores[base_key])
    skip = set(skip_keys) if skip_keys is not None else {base_key}
    out: Dict[str, float] = {}
    for name, value in scores.items():
        if name in skip:
            continue
        value = float(value)
        out[name] = value - base if higher_is_better else base - value
    return out


def contribution_higher_better(s_k: float, s_base: float, s_full: float) -> float:
    """C(k) = (S_k - S_base) / (S_full - S_base) when larger S is better."""
    denom = s_full - s_base
    if abs(denom) < 1e-12:
        return float("nan")
    return (s_k - s_base) / denom


def contribution_lower_better(l_k: float, l_base: float, l_full: float) -> float:
    """C_loss(k) = (L_base - L_k) / (L_base - L_full) when smaller L is better."""
    denom = l_base - l_full
    if abs(denom) < 1e-12:
        return float("nan")
    return (l_base - l_k) / denom


def compute_contributions(
    scores: Mapping[str, float],
    *,
    base_key: str = "pretrained",
    full_key: str = "all_params",
    higher_is_better: bool = True,
    skip_keys: Optional[Iterable[str]] = None,
    require_full_improvement: bool = True,
    min_improvement: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute layer contribution for every condition in `scores`.

    `skip_keys` defaults to {base_key, full_key}.
    """
    if base_key not in scores or full_key not in scores:
        raise KeyError(f"Need '{base_key}' and '{full_key}' in scores")
    skip = set(skip_keys) if skip_keys is not None else {base_key, full_key}
    s_base = float(scores[base_key])
    s_full = float(scores[full_key])
    if require_full_improvement and not reference_improves(
        s_base,
        s_full,
        higher_is_better=higher_is_better,
        min_improvement=min_improvement,
    ):
        return {
            name: float("nan")
            for name in scores
            if name not in skip
        }
    out: Dict[str, float] = {}
    for name, val in scores.items():
        if name in skip:
            continue
        if higher_is_better:
            out[name] = contribution_higher_better(float(val), s_base, s_full)
        else:
            out[name] = contribution_lower_better(float(val), s_base, s_full)
    return out


def rank_by_contribution(
    contributions: Mapping[str, float],
    *,
    descending: bool = True,
) -> List[tuple]:
    """Return [(condition, C), ...] sorted by contribution (NaNs last)."""
    items = list(contributions.items())

    def sort_key(item):
        name, c = item
        if c != c:  # NaN
            return (1, 0.0, name)
        return (0, -c if descending else c, name)

    return sorted(items, key=sort_key)

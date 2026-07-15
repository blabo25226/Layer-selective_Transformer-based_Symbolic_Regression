"""Select high/mid/low/random contribution layers for Phase 5.

The layer rankings that drive Phase 5 are **derived from Phase 4's output**
(``results/phase_results/phase4/contributions.json``) via
:func:`load_phase4_ranking`. The module-level ``PHASE4_*_RANKING`` constants are
only a **frozen fallback** (from one CPU run of 17 train / 4 test problems) used
when that JSON is missing, e.g. in unit tests or a fresh checkout. Prefer the
dynamic loader so re-running Phase 4 actually changes Phase 5.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# Metric keys (as written by scripts/phase4_layer_contribution.py) that define
# each ranking mode. "accuracy" = mean rank across CE + prediction metrics;
# "ce" = teacher-forcing cross-entropy only.
ACCURACY_METRICS: List[str] = ["val_ce", "penalized_nmse", "penalized_r2"]
CE_METRICS: List[str] = ["val_ce"]
RANKING_METRICS: Dict[str, List[str]] = {
    "accuracy": ACCURACY_METRICS,
    "ce": CE_METRICS,
}


# --- Frozen fallback rankings (only used when contributions.json is absent) ---

# Phase 4 accuracy consensus (val_ce + nmse + r2 mean rank), best → worst
PHASE4_ACCURACY_RANKING: List[str] = [
    "encoder_2",
    "encoder_1",
    "encoder_3",
    "decoder_1",
    "decoder_0",
    "decoder_4",
    "encoder_0",
    "decoder_2",
    "decoder_3",
    "encoder_5",
    "encoder_4",
    "output_head",
]

# Phase 3/4 CE ranking (decoder-heavy)
PHASE4_CE_RANKING: List[str] = [
    "decoder_4",
    "decoder_3",
    "decoder_2",
    "decoder_1",
    "encoder_1",
    "encoder_2",
    "decoder_0",
    "encoder_3",
    "encoder_0",
    "encoder_5",
    "encoder_4",
    "output_head",
]

FALLBACK_RANKINGS: Dict[str, List[str]] = {
    "accuracy": PHASE4_ACCURACY_RANKING,
    "ce": PHASE4_CE_RANKING,
}


def usable_ranking_metrics(
    contrib_tables: Mapping[str, Mapping[str, float]], mode: str = "accuracy"
) -> List[str]:
    """Return the finite contribution metrics that can drive a ranking."""
    if mode not in RANKING_METRICS:
        raise ValueError(f"Unknown ranking mode {mode!r}; use {list(RANKING_METRICS)}")

    def _has_finite_score(table: Mapping[str, object]) -> bool:
        for value in table.values():
            if isinstance(value, Mapping):
                value = value.get("mean", float("nan"))
            try:
                if value is not None and math.isfinite(float(value)):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    metrics = [
        m for m in RANKING_METRICS[mode]
        if m in contrib_tables and _has_finite_score(contrib_tables[m])
    ]
    if mode == "accuracy" and len(metrics) == 1 and "val_ce" in metrics:
        legacy = [
            m for m in ("val_ce", "nmse", "r2")
            if m in contrib_tables and _has_finite_score(contrib_tables[m])
        ]
        if len(legacy) > 1:
            metrics = legacy
    return metrics


def ranking_from_contributions(
    contrib_tables: Mapping[str, Mapping[str, float]],
    mode: str = "accuracy",
) -> List[str]:
    """
    Derive a best→worst layer ranking from Phase 4 contribution tables.

    ``contrib_tables`` maps ``metric -> {condition: C}`` (C higher = better;
    NaN = undefined). For a single metric we rank by C descending; for several
    (``accuracy``) we average each layer's per-metric rank (NaN → worst rank)
    and sort ascending mean rank. ``pretrained`` / ``all_params`` are excluded.
    """
    metrics = usable_ranking_metrics(contrib_tables, mode)
    if not metrics:
        raise KeyError(
            f"None of {RANKING_METRICS[mode]} present in contributions "
            f"(have {list(contrib_tables)})"
        )

    # Union of candidate layers across the chosen metrics.
    layers: List[str] = []
    for m in metrics:
        for name in contrib_tables[m]:
            if name in ("pretrained", "all_params"):
                continue
            if name not in layers:
                layers.append(name)

    n = len(layers)
    rank_sum: Dict[str, float] = {name: 0.0 for name in layers}
    for m in metrics:
        table = contrib_tables[m]

        def _score(name: str) -> float:
            c = table.get(name, float("nan"))
            if isinstance(c, Mapping):
                c = c.get("mean", float("nan"))
            return float(c) if c is not None else float("nan")

        def _key(name: str):
            c = _score(name)
            if math.isnan(c):
                return (1, 0.0, name)  # NaN → worst
            return (0, -c, name)

        order = sorted(layers, key=_key)
        for rank, name in enumerate(order, 1):
            # Layers a metric never scored get the worst possible rank.
            c = _score(name)
            rank_sum[name] += rank if not math.isnan(c) else n

    return sorted(layers, key=lambda name: (rank_sum[name] / len(metrics), name))


def resolve_selected_layers(
    contributions_path: Path,
    *,
    mode: str = "accuracy",
    rule: str = "top",
    k: int = 3,
    explicit: Optional[Sequence[str]] = None,
) -> tuple[List[str], str, str]:
    """
    Resolve the single "high-contribution" layer set used by Phases 6–8.

    Prefer a **principled a-priori** rule (``rule="top"`` → top-k of the Phase 4
    ranking) over the earlier post-hoc ``middle_3`` set, which was chosen because
    it happened to win on 4 test problems (reviewer note A-2). Pass ``explicit``
    to force a specific set (still logged as such).

    Returns ``(layers, source, detail)`` where ``source`` is
    ``"explicit" | "phase4" | "fallback"`` and ``detail`` describes the rule.
    """
    if explicit:
        return list(explicit), "explicit", "user-specified"
    ranking, source = load_phase4_ranking(contributions_path, mode)
    rule = rule.lower()
    if rule == "top":
        layers = top_k(ranking, k)
    elif rule in ("mid", "middle"):
        layers = middle_k(ranking, k)
    elif rule == "bottom":
        layers = bottom_k(ranking, k)
    else:
        raise ValueError(f"Unknown rule {rule!r}; use top|middle|bottom")
    return layers, source, f"{rule}_{k} of {mode} ranking"


def load_phase4_ranking(
    contributions_path: Path,
    mode: str = "accuracy",
) -> tuple[List[str], str]:
    """
    Load ranking for ``mode`` from a Phase 4 ``contributions.json``.

    Returns ``(ranking, source)`` where ``source`` is ``"phase4"`` when the file
    was read, or ``"fallback"`` when it was missing/invalid and the frozen
    constant was used instead.
    """
    try:
        tables = json.loads(Path(contributions_path).read_text(encoding="utf-8"))
        ranking = ranking_from_contributions(tables, mode)
        if ranking:
            multiseed = any(
                isinstance(value, Mapping)
                for table in tables.values()
                if isinstance(table, Mapping)
                for value in table.values()
            )
            return ranking, "phase4_multiseed" if multiseed else "phase4"
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return list(FALLBACK_RANKINGS[mode]), "fallback"


def top_k(ranking: Sequence[str], k: int) -> List[str]:
    return list(ranking[: max(0, k)])


def bottom_k(ranking: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    return list(ranking[-k:])


def middle_k(ranking: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    n = len(ranking)
    if k >= n:
        return list(ranking)
    start = max(0, (n - k) // 2)
    return list(ranking[start : start + k])


def random_k(
    ranking: Sequence[str],
    k: int,
    seed: int = 0,
    *,
    exclude: Sequence[str] = (),
) -> List[str]:
    """Sample k layers uniformly, optionally excluding some (e.g. the top-k).

    Excluding the top layers keeps the random condition an honest control: a
    "random" set that happens to contain the #1 layer makes any top-vs-random
    gap disappear for reasons unrelated to the ranking (reviewer note A-3).
    """
    if k <= 0:
        return []
    excl = set(exclude)
    pool = [name for name in ranking if name not in excl]
    rng = random.Random(seed)
    k = min(k, len(pool))
    return rng.sample(pool, k)


def build_phase5_conditions(
    ranking: Sequence[str],
    *,
    k: int = 3,
    random_seed: int = 0,
) -> Dict[str, Optional[List[str]]]:
    """
    Phase 5 comparison sets.

    - pretrained: no training
    - top_1 / top_2 / top_3 / top_k
    - middle_k / random_k / bottom_k
    - all_params: unfreeze everything

    ``random_k`` samples from layers *outside* the top ``max(k, 3)`` so it is a
    genuine "non-top" control rather than one that can re-draw the best layers.
    """
    ranking = list(ranking)
    cond: Dict[str, Optional[List[str]]] = {
        "pretrained": [],
        "top_1": top_k(ranking, 1),
        "top_2": top_k(ranking, 2),
        "top_3": top_k(ranking, 3),
    }
    if k != 3:
        cond[f"top_{k}"] = top_k(ranking, k)
    top_exclude = top_k(ranking, max(k, 3))
    cond[f"middle_{k}"] = middle_k(ranking, k)
    cond[f"random_{k}"] = random_k(ranking, k, seed=random_seed, exclude=top_exclude)
    cond[f"bottom_{k}"] = bottom_k(ranking, k)
    cond["all_params"] = None
    return cond

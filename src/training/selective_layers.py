"""Select high/mid/low/random contribution layers for Phase 5."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence


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


def random_k(ranking: Sequence[str], k: int, seed: int = 0) -> List[str]:
    if k <= 0:
        return []
    pool = list(ranking)
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
    cond[f"middle_{k}"] = middle_k(ranking, k)
    cond[f"random_{k}"] = random_k(ranking, k, seed=random_seed)
    cond[f"bottom_{k}"] = bottom_k(ranking, k)
    cond["all_params"] = None
    return cond

"""Leakage-resistant dataset splitting helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def grouped_train_validation_split(
    items: Sequence[T],
    *,
    validation_fraction: float = 0.2,
    seed: int = 1729,
    group_key: Callable[[T], str],
) -> Tuple[List[T], List[T]]:
    """Split whole groups so related equations never cross train/validation."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not items:
        raise ValueError("cannot split an empty sequence")
    groups = defaultdict(list)
    for item in items:
        groups[str(group_key(item))].append(item)
    names = sorted(groups)
    if len(names) < 2:
        raise ValueError("grouped split requires at least two distinct groups")
    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(names, dtype=object)[rng.permutation(len(names))])
    n_val = min(len(names) - 1, max(1, round(len(names) * validation_fraction)))
    val_names = set(shuffled[:n_val])
    train = [item for name in names if name not in val_names for item in groups[name]]
    validation = [item for name in names if name in val_names for item in groups[name]]
    return train, validation


def split_synthetic_train_validation(
    problems: Sequence[T],
    *,
    validation_fraction: float = 0.2,
    seed: int = 1729,
) -> Tuple[List[T], List[T]]:
    """Group synthetic equations by motif before splitting."""
    return grouped_train_validation_split(
        problems,
        validation_fraction=validation_fraction,
        seed=seed,
        group_key=lambda problem: problem.spec.motif or problem.spec.family,
    )

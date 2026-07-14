"""Tests for DREAM4 Size10 loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dream4 import (  # noqa: E402
    find_dream4_root,
    list_size10_net_ids,
    load_size10_network,
    load_size10_expression_bundle,
    finite_difference_rhs,
    trajectory_train_test_split,
    load_timeseries,
)


def test_trajectory_split_keeps_complete_trajectories_apart():
    times = [np.arange(4, dtype=float) for _ in range(4)]
    xs = [np.full((4, 2), i, dtype=float) for i in range(4)]
    X_tr, _, X_te, _ = trajectory_train_test_split(times, xs, seed=3)
    assert set(np.unique(X_tr[:, 0])).isdisjoint(set(np.unique(X_te[:, 0])))


def test_dream4_size10_present():
    root = find_dream4_root(ROOT / "data" / "dream4")
    ids = list_size10_net_ids(root)
    assert ids == [1, 2, 3, 4, 5]
    net = load_size10_network(root, 1)
    assert net.n_genes == 10
    assert len(net.edges) >= 1
    bundle = load_size10_expression_bundle(root, 1)
    assert bundle["X_multi"].shape[1] == 10
    assert bundle["X_ts"].shape == bundle["Y_ts"].shape
    assert bundle["X_ts"].shape[0] > 10

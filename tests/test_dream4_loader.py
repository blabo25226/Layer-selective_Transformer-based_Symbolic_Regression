"""Tests for DREAM4 Size10 loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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
    build_dream4_local_problems,
    Dream4Size10Network,
)


def test_trajectory_split_keeps_complete_trajectories_apart():
    times = [np.arange(4, dtype=float) for _ in range(4)]
    xs = [np.full((4, 2), i, dtype=float) for i in range(4)]
    X_tr, _, X_te, _ = trajectory_train_test_split(times, xs, seed=3)
    assert set(np.unique(X_tr[:, 0])).isdisjoint(set(np.unique(X_te[:, 0])))


def test_dream4_size10_present():
    data_root = ROOT / "data" / "dream4"
    if not data_root.exists():
        pytest.skip("optional DREAM4 archive is not present; see GPU_RUN.md")
    root = find_dream4_root(data_root)
    ids = list_size10_net_ids(root)
    assert ids == [1, 2, 3, 4, 5]
    net = load_size10_network(root, 1)
    assert net.n_genes == 10
    assert len(net.edges) >= 1
    bundle = load_size10_expression_bundle(root, 1)
    assert bundle["X_multi"].shape[1] == 10
    assert bundle["X_ts"].shape == bundle["Y_ts"].shape
    assert bundle["X_ts"].shape[0] > 10


def test_fixed_train_selection_is_reused_on_test_values(tmp_path):
    network = Dream4Size10Network(
        net_id=1,
        n_genes=3,
        gene_names=["G1", "G2", "G3"],
        edges=[(1, 0, "act")],
        root=tmp_path,
    )
    X_test = np.arange(18, dtype=float).reshape(6, 3)
    Y_test = np.column_stack([X_test[:, 2], X_test[:, 0], X_test[:, 1]])
    _, selections, _ = build_dream4_local_problems(
        network, X_test, Y_test, method="corr", k=1, split="test",
        target_ids=[0], fixed_selections={0: [1]},
    )
    assert selections == {0: [1]}

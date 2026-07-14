"""Tests for Phase 7 dreamlike GRN + regulator selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dreamlike_grn import (  # noqa: E402
    generate_random_grn,
    rhs_for_target,
    sample_expression,
    build_local_problem,
)
from data.regulator_selection import (  # noqa: E402
    oracle_regulators,
    correlation_select,
    selection_metrics,
)
from evaluation.grn_metrics import edge_recovery  # noqa: E402


def test_network_and_rhs():
    net = generate_random_grn(n_genes=6, seed=1)
    assert net.n_genes == 6
    assert len(net.edges) > 0
    rng = np.random.default_rng(0)
    X = sample_expression(net, 50, (0.1, 2.0), rng)
    y = rhs_for_target(net, X, 0)
    assert y.shape == (50,)
    assert np.all(np.isfinite(y))


def test_oracle_and_local():
    net = generate_random_grn(n_genes=6, seed=2)
    rng = np.random.default_rng(0)
    X = sample_expression(net, 40, (0.1, 2.0), rng)
    t = 1
    y = rhs_for_target(net, X, t)
    regs = oracle_regulators(net, t)
    ds = build_local_problem(
        net, X, y, t, regs, eq_id="t1", split="train", max_vars=3
    )
    assert ds.X.shape[1] <= 3
    assert ds.y.shape[0] == 40


def test_selection_metrics():
    m = selection_metrics([0, 1], [1, 2])
    assert 0 < m["f1"] < 1
    assert edge_recovery([(0, 1)], [(0, 1), (2, 1)])["recall"] == 1.0


def test_corr_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = X[:, 2] + 0.1 * rng.normal(size=80)
    regs = correlation_select(X, y, target=0, k=2)
    assert len(regs) == 2

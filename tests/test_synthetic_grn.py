"""Tests for synthetic GRN data generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic_grn import (  # noqa: E402
    build_phase1_suite,
    hill_activation,
    load_problem,
    save_suite,
)


def test_activation_rhs_matches():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 3, 50)
    y = rng.uniform(0, 3, 50)
    out = hill_activation(x, y, alpha=1.0, k=1.0, n=2.0, beta=0.5)
    assert out.shape == (50,)
    assert np.all(np.isfinite(out))


def test_suite_and_io(tmp_path: Path):
    out = tmp_path
    datasets = build_phase1_suite(n_points=20, seed=1)
    assert len(datasets) > 10
    families = {d.spec.family for d in datasets}
    assert families == {"activation", "repression", "toggle", "repressilator"}
    splits = {d.spec.split for d in datasets}
    assert splits == {"train", "test"}

    index_path = save_suite(datasets, Path(out))
    assert index_path.exists()
    first = datasets[0]
    loaded = load_problem(Path(out) / f"{first.spec.eq_id}.npz")
    assert loaded.spec.eq_id == first.spec.eq_id
    np.testing.assert_allclose(loaded.X, first.X)
    np.testing.assert_allclose(loaded.y, first.y)


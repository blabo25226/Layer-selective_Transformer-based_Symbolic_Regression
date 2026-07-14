"""Unit tests for Phase 8 LODO generalization metrics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.generalization import (  # noqa: E402
    aggregate_lodo,
    generalization_gap,
    rank_by_generalization,
)


def test_gap_direction():
    # lower-better: positive gap = worse on holdout
    assert abs(generalization_gap(0.1, 0.5) - 0.4) < 1e-9
    # higher-better (R2): positive gap still = generalizes worse
    assert abs(generalization_gap(0.9, 0.4, lower_better=False) - 0.5) < 1e-9


def test_aggregate_and_rank():
    folds = [
        {"pysr": {"in": 0.01, "hold": 0.55}, "sel": {"in": 0.17, "hold": 0.19}},
        {"pysr": {"in": 0.02, "hold": 0.48}, "sel": {"in": 0.15, "hold": 0.21}},
        {"pysr": {"in": 0.008, "hold": 0.60}, "sel": {"in": 0.18, "hold": 0.18}},
    ]
    agg = aggregate_lodo(folds)
    # PySR overfits: large positive gap; sel generalizes: small gap
    assert agg["pysr"]["gap_mean"] > agg["sel"]["gap_mean"]
    assert agg["pysr"]["n_folds"] == 3
    # best generalizer (lowest holdout NMSE) is sel despite pysr winning in-donor
    assert rank_by_generalization(agg)[0] == "sel"


def test_handles_missing_and_nonfinite():
    folds = [
        {"a": {"in": 0.1, "hold": 0.2}},
        {"a": {"in": float("inf"), "hold": 0.3}},  # inf dropped from stats
        {"b": {"in": 0.4, "hold": 0.5}},  # method only in one fold
    ]
    agg = aggregate_lodo(folds)
    assert agg["a"]["n_folds"] == 1  # only the finite fold counts toward the gap
    assert not math.isnan(agg["b"]["gap_mean"])

"""Unit tests for layer contribution formulas."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.layer_contribution import (  # noqa: E402
    absolute_improvements,
    compute_contributions,
    contribution_higher_better,
    contribution_lower_better,
    rank_by_contribution,
    reference_improves,
)


def test_contribution_formulas():
    assert abs(contribution_higher_better(0.5, 0.0, 1.0) - 0.5) < 1e-9
    assert abs(contribution_lower_better(1.0, 2.0, 0.0) - 0.5) < 1e-9
    assert math.isnan(contribution_higher_better(0.1, 0.5, 0.5))


def test_compute_and_rank():
    scores = {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.7, "encoder_0": 0.1}
    c = compute_contributions(scores, higher_is_better=True)
    assert abs(c["decoder_4"] - 0.7) < 1e-9
    ranked = rank_by_contribution(c)
    assert ranked[0][0] == "decoder_4"


def test_loss_contributions():
    losses = {"pretrained": 2.0, "all_params": 0.2, "decoder_2": 1.0}
    c = compute_contributions(losses, higher_is_better=False)
    assert abs(c["decoder_2"] - (2.0 - 1.0) / (2.0 - 0.2)) < 1e-9


def test_full_must_improve_before_normalization():
    losses = {"pretrained": 1.0, "all_params": 1.2, "decoder_2": 0.5}
    contributions = compute_contributions(losses, higher_is_better=False)
    assert math.isnan(contributions["decoder_2"])
    assert not reference_improves(1.0, 1.2, higher_is_better=False)


def test_absolute_improvement_remains_available_when_full_is_worse():
    losses = {"pretrained": 1.0, "all_params": 1.2, "decoder_2": 0.5}
    improvements = absolute_improvements(losses, higher_is_better=False)
    assert improvements["decoder_2"] == 0.5
    assert abs(improvements["all_params"] + 0.2) < 1e-12

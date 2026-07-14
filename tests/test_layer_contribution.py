"""Unit tests for layer contribution formulas."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.layer_contribution import (  # noqa: E402
    compute_contributions,
    contribution_higher_better,
    contribution_lower_better,
    rank_by_contribution,
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

"""Unit tests for Phase 5 layer selection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.selective_layers import (  # noqa: E402
    PHASE4_ACCURACY_RANKING,
    bottom_k,
    build_phase5_conditions,
    middle_k,
    random_k,
    top_k,
)


def test_top_bottom_middle():
    r = PHASE4_ACCURACY_RANKING
    assert top_k(r, 2) == ["encoder_2", "encoder_1"]
    assert bottom_k(r, 1) == ["output_head"]
    mid = middle_k(r, 3)
    assert len(mid) == 3
    assert mid[0] in r


def test_random_repro():
    r = PHASE4_ACCURACY_RANKING
    a = random_k(r, 3, seed=0)
    b = random_k(r, 3, seed=0)
    assert a == b
    assert len(a) == 3


def test_build_conditions():
    c = build_phase5_conditions(PHASE4_ACCURACY_RANKING, k=3)
    assert "top_1" in c and c["top_1"] == ["encoder_2"]
    assert c["all_params"] is None
    assert len(c["middle_3"]) == 3
    assert len(c["bottom_3"]) == 3

"""Unit tests for Phase 5 layer selection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from training.selective_layers import (  # noqa: E402
    PHASE4_ACCURACY_RANKING,
    bottom_k,
    build_phase5_conditions,
    load_phase4_ranking,
    middle_k,
    random_k,
    ranking_from_contributions,
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


def test_random_excludes_top(tmp_path=None):
    # random_3 must not contain any of the top-3 layers (A-3 control fix).
    c = build_phase5_conditions(PHASE4_ACCURACY_RANKING, k=3)
    top3 = set(top_k(PHASE4_ACCURACY_RANKING, 3))
    assert not (set(c["random_3"]) & top3)
    # explicit exclude on random_k
    r = random_k(PHASE4_ACCURACY_RANKING, 3, seed=0, exclude=top3)
    assert not (set(r) & top3)


def test_ranking_from_contributions():
    # Higher C = better. accuracy = mean rank over val_ce+nmse+r2.
    tables = {
        "val_ce": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.9, "encoder_2": 0.2},
        "nmse": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.1, "encoder_2": 0.8},
        "r2": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.1, "encoder_2": 0.8},
    }
    acc = ranking_from_contributions(tables, "accuracy")
    assert acc[0] == "encoder_2"  # wins 2 of 3 metrics
    ce = ranking_from_contributions(tables, "ce")
    assert ce[0] == "decoder_4"  # wins the single CE metric
    assert "pretrained" not in acc and "all_params" not in acc


def test_load_phase4_ranking_fallback_and_file(tmp_path):
    # Missing file → frozen fallback.
    ranking, source = load_phase4_ranking(tmp_path / "nope.json", "accuracy")
    assert source == "fallback" and ranking == PHASE4_ACCURACY_RANKING
    # Present file → derived from JSON.
    p = tmp_path / "contributions.json"
    p.write_text(
        json.dumps(
            {
                "val_ce": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.9, "encoder_2": 0.2},
                "nmse": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.1, "encoder_2": 0.8},
                "r2": {"pretrained": 0.0, "all_params": 1.0, "decoder_4": 0.1, "encoder_2": 0.8},
            }
        ),
        encoding="utf-8",
    )
    ranking, source = load_phase4_ranking(p, "accuracy")
    assert source == "phase4" and ranking[0] == "encoder_2"

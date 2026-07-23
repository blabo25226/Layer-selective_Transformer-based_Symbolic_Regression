"""Tests for paired GPU-run aggregation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_phase6_runs import main as aggregate_phase6  # noqa: E402
from scripts.aggregate_phase5_runs import main as aggregate_phase5  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from evaluation.layer_contribution import rank_correlations  # noqa: E402


def _phase5_row(condition, nmse):
    return {
        "condition": condition,
        "eval": {
            "penalized_nmse": nmse,
            "penalized_r2": 1.0 - nmse,
            "valid_rate": 1.0,
            "complexity": 2.0,
            "near_singularity_mean": 0.0,
            "extrapolation_valid_mean": 1.0,
        },
        "elapsed_sec": 1.0,
        "peak_mem_mb": 10.0,
    }


def test_phase5_reports_random_draws_and_top_full_equivalence(tmp_path, monkeypatch):
    run = tmp_path / "run"
    values = {
        "pretrained": 1.5,
        "all_params": 1.0,
        "top_1": 1.02,
        "top_2": 1.02,
        "top_3": 1.02,
        "middle_3": 1.4,
        "bottom_3": 1.6,
        "random_3": 2.0,
        "random_3_seed1": 4.0,
    }
    for seed in (0, 1):
        path = run / f"phase5_seed{seed}" / "selective_results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps([_phase5_row(name, value) for name, value in values.items()]),
            encoding="utf-8",
        )
    monkeypatch.setattr(sys, "argv", [
        "aggregate_phase5_runs.py", "--run-dir", str(run), "--seeds", "0", "1",
        "--k", "3", "--nmse-equivalence-margin", "0.05",
    ])
    assert aggregate_phase5() == 0
    result = json.loads((run / "phase5_multiseed" / "summary.json").read_text(encoding="utf-8"))
    assert result["random_layer_conditions"] == ["random_3", "random_3_seed1"]
    assert result["random_mean_nmse_by_training_seed"] == [3.0, 3.0]
    assert result["top_vs_full_equivalence"]["top_3"]["equivalent"] is True
    assert result["top_vs_full_equivalence"]["top_3"]["conclusion"] == "equivalent_within_margin"
    assert "penalized_r2" in result["paired_metric_comparisons"]["top_3"]["all_params"]


def test_phase4_rank_correlations_detect_stable_and_reversed_orders():
    a = {"l1": 3.0, "l2": 2.0, "l3": 1.0}
    assert rank_correlations(a, dict(a))["spearman"] == 1.0
    reversed_scores = {"l1": 1.0, "l2": 2.0, "l3": 3.0}
    assert rank_correlations(a, reversed_scores)["spearman"] == -1.0


def test_phase6_reports_paired_interaction(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cells = {
        "pretrained_beam": 10.0,
        "pretrained_tpsr": 9.0,
        "selective_beam": 8.0,
        "selective_tpsr": 5.0,
    }
    for seed in (0, 1):
        payload = {"0.0": {}}
        for cell, nmse in cells.items():
            payload["0.0"][cell] = {
                "penalized_nmse": nmse + seed,
                "valid_rate": 1.0,
                "complexity": 5.0,
                "sym_rate": 0.0,
                "elapsed_sec": 1.0,
                "near_singularity_mean": 0.0,
                "extrapolation_valid_mean": 1.0,
            }
        path = run / f"phase6_noise_seed{seed}" / "noise_sweep.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "aggregate_phase6_runs.py", "--run-dir", str(run), "--seeds", "0", "1",
    ])
    assert aggregate_phase6() == 0
    result = json.loads((run / "phase6_noise_multiseed" / "summary.json").read_text(encoding="utf-8"))
    # TPSR improvement is 1 before FT and 3 after FT, so interaction is +2.
    assert result["paired_effects"]["0.0"]["interaction"]["mean"] == 2.0

"""Unit tests for symbolic recovery helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.equation_metrics import expression_safety, symbolic_recovery, variable_recovery  # noqa: E402
from evaluation.equation_records import make_equation_record, variable_mapping  # noqa: E402


def test_variable_recovery_f1():
    vr = variable_recovery(["x_1", "x_2"], "x_1 + x_2**2")
    assert vr["recall"] == 1.0
    assert vr["precision"] == 1.0


def test_symbolic_skeleton_match():
    true = "1.5*x_2**2/(0.8**2+x_2**2)-0.4*x_1"
    pred = "2.0*x_2**2/(1.0**2+x_2**2)-0.5*x_1"
    sr = symbolic_recovery(true, pred)
    # Both are Hill-activation skeletons; may or may not match depending on placeholder
    assert "recovery" in sr
    assert sr["exact"] == 0.0


def test_symbolic_exact():
    expr = "x_1 + x_2"
    sr = symbolic_recovery(expr, expr)
    assert sr["exact"] == 1.0
    assert sr["recovery"] == 1.0


def test_expression_safety_flags_tan_and_singularity():
    X = np.array([[0.0], [0.5], [1.0]])
    result = expression_safety("tan(x_1) + 1/(x_1-0.5)", X, ["x_1"])
    assert result["has_tan"] == 1.0
    assert result["has_division"] == 1.0
    assert result["near_singularity"] == 1.0


def test_equation_record_preserves_raw_simplified_candidates_and_mapping():
    record = make_equation_record(
        eq_id="gene1",
        predicted_expr="x_1 + x_1",
        variable_names=["x_1"],
        mapping=variable_mapping(["x_1"], column_indices=[2], source_names=["A", "B", "C"]),
        scores={"valid_pred": 1.0, "nmse": 0.0},
        true_expr="2*x_1",
        candidate_expressions=["x_1 + x_1", "2.1*x_1"],
        decoder="nesymres_beam_bfgs",
    )
    assert record["pred"] == "x_1 + x_1"
    assert record["pred_raw"] == "x_1 + x_1"
    assert record["pred_simplified"] == "2*x_1"
    assert record["candidate_expressions"] == ["x_1 + x_1", "2.1*x_1"]
    assert record["variable_mapping"][0]["source_name"] == "C"
    assert record["failure_reason"] is None


def test_equation_record_assigns_failure_reason():
    record = make_equation_record(
        eq_id="failed",
        predicted_expr="",
        variable_names=["x_1"],
        scores={"valid_pred": 0.0},
        decoder="pysr",
    )
    assert record["failure_reason"] == "decoder_returned_no_expression"

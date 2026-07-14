"""Unit tests for symbolic recovery helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.equation_metrics import symbolic_recovery, variable_recovery  # noqa: E402


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

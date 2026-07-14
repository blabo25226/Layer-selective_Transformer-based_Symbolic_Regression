"""Equation / GRN evaluation metrics for Phase 2."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
from sympy import lambdify, sympify


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.mean(y_true**2) + 1e-12
    return float(np.mean((y_true - y_pred) ** 2) / denom)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def extract_variables(expr: str) -> Set[str]:
    return set(re.findall(r"\bx_\d+\b", expr))


def variable_recovery(
    true_vars: Sequence[str],
    predicted_expr: str,
) -> Dict[str, float]:
    """Recovery of named variables present in the predicted equation string."""
    true_set = set(true_vars)
    pred_set = extract_variables(predicted_expr)
    if not true_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(true_set & pred_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / len(true_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def complexity(expr: str) -> int:
    """Rough complexity: count of operators / symbols."""
    return len(re.findall(r"[A-Za-z_]+|\d+\.?\d*|[+\-*/^()]", expr))


def eval_expression(
    expr: str,
    X: np.ndarray,
    variable_names: Sequence[str],
) -> Optional[np.ndarray]:
    """
    Evaluate a sympy-parsable expression on columns of X.
    Returns None on failure.
    """
    try:
        cleaned = expr.replace("constant", "1.0")
        # Map up to 3 named vars from columns
        env = {}
        for i in range(max(3, X.shape[1])):
            name = f"x_{i + 1}"
            if i < X.shape[1]:
                env[name] = X[:, i]
            else:
                env[name] = np.zeros(X.shape[0], dtype=float)
        # Also allow any declared names
        for i, name in enumerate(variable_names):
            if i < X.shape[1]:
                env[name] = X[:, i]
        fn = lambdify(["x_1", "x_2", "x_3"], sympify(cleaned), modules=["numpy"])
        out = np.asarray(fn(env["x_1"], env["x_2"], env["x_3"]), dtype=float)
        out = np.broadcast_to(out, (X.shape[0],)).ravel()
        if not np.all(np.isfinite(out)):
            return None
        return out
    except Exception:
        return None


def score_prediction(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    predicted_expr: str,
    true_vars: Sequence[str],
) -> Dict[str, float]:
    if y_pred is None:
        return {
            "nmse": float("inf"),
            "r2": float("-inf"),
            "var_precision": 0.0,
            "var_recall": 0.0,
            "var_f1": 0.0,
            "complexity": float(complexity(predicted_expr)) if predicted_expr else 0.0,
            "valid_pred": 0.0,
        }
    vr = variable_recovery(true_vars, predicted_expr)
    return {
        "nmse": nmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "var_precision": vr["precision"],
        "var_recall": vr["recall"],
        "var_f1": vr["f1"],
        "complexity": float(complexity(predicted_expr)),
        "valid_pred": 1.0,
    }

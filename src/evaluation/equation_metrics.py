"""Equation / GRN evaluation metrics for Phase 2–4."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sympy import simplify, sympify
from sympy.core.expr import Expr


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized MSE by E[y^2] (Phase 2 default; stable for near-zero mean)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.mean(y_true**2) + 1e-12
    return float(np.mean((y_true - y_pred) ** 2) / denom)


def nmse_vs_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plan §9.1 NMSE: SSE / sum (y - ybar)^2."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return float(np.sum((y_true - y_pred) ** 2) / denom)


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


def _normalize_expr_str(expr: str) -> str:
    s = (expr or "").strip()
    s = s.replace("constant", "c")
    s = re.sub(r"\bc\d+\b", "c", s)
    return s


def to_skeleton(expr: str) -> Optional[Expr]:
    """Parse expression and replace numeric constants with symbol c (best-effort)."""
    try:
        from nesymres.architectures.data import constants_to_placeholder

        sk = constants_to_placeholder(_normalize_expr_str(expr))
        return simplify(sympify(sk))
    except Exception:
        try:
            cleaned = re.sub(r"\d+\.?\d*(?:[eE][+-]?\d+)?", "c", _normalize_expr_str(expr))
            cleaned = re.sub(r"c+", "c", cleaned)
            return simplify(sympify(cleaned))
        except Exception:
            return None


def symbolic_recovery(
    true_expr: str,
    predicted_expr: str,
) -> Dict[str, float]:
    """
    Symbolic recovery scores vs ground-truth expression.

    - exact: string match after strip
    - skeleton: SymPy-simplified skeletons equal (constants -> c)
    - equiv: difference simplifies to 0 (numeric constants kept when possible)
    """
    pred = (predicted_expr or "").strip()
    true = (true_expr or "").strip()
    exact = 1.0 if pred and pred == true else 0.0

    sk_true = to_skeleton(true)
    sk_pred = to_skeleton(pred)
    skeleton = 0.0
    if sk_true is not None and sk_pred is not None:
        try:
            skeleton = 1.0 if simplify(sk_true - sk_pred) == 0 else 0.0
            if skeleton == 0.0 and sk_true.equals(sk_pred):
                skeleton = 1.0
        except Exception:
            skeleton = 0.0

    equiv = 0.0
    if true and pred:
        try:
            diff = simplify(sympify(true) - sympify(_normalize_expr_str(pred)))
            equiv = 1.0 if diff == 0 else 0.0
        except Exception:
            equiv = skeleton  # fall back

    return {
        "exact": exact,
        "skeleton": skeleton,
        "equiv": equiv,
        "recovery": max(exact, skeleton, equiv),
    }


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
        from sympy import lambdify

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
    true_expr: str = "",
) -> Dict[str, float]:
    if y_pred is None:
        sr = symbolic_recovery(true_expr, predicted_expr) if true_expr else {
            "exact": 0.0,
            "skeleton": 0.0,
            "equiv": 0.0,
            "recovery": 0.0,
        }
        return {
            "nmse": float("inf"),
            "nmse_var": float("inf"),
            "r2": float("-inf"),
            "var_precision": 0.0,
            "var_recall": 0.0,
            "var_f1": 0.0,
            "complexity": float(complexity(predicted_expr)) if predicted_expr else 0.0,
            "valid_pred": 0.0,
            "sym_exact": sr["exact"],
            "sym_skeleton": sr["skeleton"],
            "sym_equiv": sr["equiv"],
            "sym_recovery": sr["recovery"],
        }
    vr = variable_recovery(true_vars, predicted_expr)
    sr = symbolic_recovery(true_expr, predicted_expr) if true_expr else {
        "exact": 0.0,
        "skeleton": 0.0,
        "equiv": 0.0,
        "recovery": 0.0,
    }
    return {
        "nmse": nmse(y_true, y_pred),
        "nmse_var": nmse_vs_variance(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "var_precision": vr["precision"],
        "var_recall": vr["recall"],
        "var_f1": vr["f1"],
        "complexity": float(complexity(predicted_expr)),
        "valid_pred": 1.0,
        "sym_exact": sr["exact"],
        "sym_skeleton": sr["skeleton"],
        "sym_equiv": sr["equiv"],
        "sym_recovery": sr["recovery"],
    }

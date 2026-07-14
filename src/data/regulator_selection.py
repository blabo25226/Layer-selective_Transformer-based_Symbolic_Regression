"""Regulator preselection for Phase 7 local GRN problems."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def oracle_regulators(network, target: int) -> List[int]:
    return [r for r, _ in network.parents(target)]


def correlation_select(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    k: int,
    *,
    exclude_target: bool = True,
) -> List[int]:
    """Top-k genes by |Pearson| with y (dx/dt), excluding target optionally."""
    scores = []
    for g in range(X.shape[1]):
        if exclude_target and g == target:
            continue
        x = np.asarray(X[:, g], dtype=float)
        yy = np.asarray(y, dtype=float)
        x_centered = x - np.mean(x)
        y_centered = yy - np.mean(yy)
        denom = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
        if denom < 1e-12:
            s = 0.0
        else:
            # Avoid np.corrcoef: this direct form is faster for 1-D vectors and
            # avoids known MKL aborts in some Windows NumPy environments.
            s = abs(float(np.dot(x_centered, y_centered) / denom))
            if not np.isfinite(s):
                s = 0.0
        scores.append((s, g))
    scores.sort(reverse=True)
    return [g for _, g in scores[:k]]


def _mutual_info_1d(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Histogram mutual information (nats)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_xy = c_xy + 1e-12
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = float(np.sum(p_xy * np.log(p_xy / (p_x[:, None] * p_y[None, :]))))
    return max(mi, 0.0)


def mi_select(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    k: int,
    *,
    exclude_target: bool = True,
    bins: int = 8,
) -> List[int]:
    scores = []
    for g in range(X.shape[1]):
        if exclude_target and g == target:
            continue
        scores.append((_mutual_info_1d(X[:, g], y, bins=bins), g))
    scores.sort(reverse=True)
    return [g for _, g in scores[:k]]


def lasso_select(
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    k: int,
    *,
    exclude_target: bool = True,
    alpha: float = 0.01,
) -> List[int]:
    """Top-k by |LASSO coefficient| (sklearn if available; else correlation)."""
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        return correlation_select(X, y, target, k, exclude_target=exclude_target)

    genes = [g for g in range(X.shape[1]) if not (exclude_target and g == target)]
    if not genes:
        return []
    Xm = X[:, genes]
    # standardize lightly
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-8)
    yy = (y - y.mean()) / (y.std() + 1e-8)
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(Xm, yy)
    coef = np.abs(model.coef_)
    order = np.argsort(-coef)
    return [genes[i] for i in order[:k]]


def select_regulators(
    method: str,
    network,
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    k: int,
) -> List[int]:
    method = method.lower()
    if method == "oracle":
        regs = oracle_regulators(network, target)
        return regs[:k] if k > 0 else regs
    if method in ("corr", "correlation"):
        return correlation_select(X, y, target, k)
    if method in ("mi", "mutual_info"):
        return mi_select(X, y, target, k)
    if method == "lasso":
        return lasso_select(X, y, target, k)
    raise ValueError(f"Unknown method: {method}")


def selection_metrics(
    true_regs: Sequence[int],
    pred_regs: Sequence[int],
) -> Dict[str, float]:
    true_set = set(true_regs)
    pred_set = set(pred_regs)
    if not true_set and not pred_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not true_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    tp = len(true_set & pred_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / len(true_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

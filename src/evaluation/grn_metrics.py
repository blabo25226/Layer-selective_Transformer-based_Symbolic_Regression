"""GRN structure metrics for Phase 7."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def edge_set(edges: Iterable[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    return {(int(r), int(t)) for r, t in edges}


def edge_recovery(
    true_edges: Sequence[Tuple[int, int]],
    pred_edges: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    t = edge_set(true_edges)
    p = edge_set(pred_edges)
    if not t and not p:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}
    tp = len(t & p)
    fp = len(p - t)
    fn = len(t - p)
    precision = tp / max(len(p), 1)
    recall = tp / max(len(t), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def predicted_edges_from_selections(
    selections: Dict[int, Sequence[int]],
) -> List[Tuple[int, int]]:
    """selections: target -> candidate regulators."""
    edges = []
    for t, regs in selections.items():
        for r in regs:
            edges.append((int(r), int(t)))
    return edges

"""Canonical, JSON-safe records for symbolic-regression outputs."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def simplify_expression(expression: str) -> tuple[str, Optional[str]]:
    """Return a deterministic SymPy simplification and an optional parse error."""
    text = (expression or "").strip()
    if not text:
        return "", None
    try:
        import sympy as sp

        parsed = sp.sympify(text.replace("^", "**"))
        return str(sp.simplify(parsed)), None
    except Exception as exc:
        return text, f"{type(exc).__name__}: {exc}"


def variable_mapping(
    variable_names: Sequence[str],
    *,
    column_indices: Optional[Sequence[Optional[int]]] = None,
    source_names: Optional[Sequence[str]] = None,
) -> list[dict[str, Any]]:
    """Map local SR symbols to source columns and, when known, gene names."""
    rows: list[dict[str, Any]] = []
    for position, symbol in enumerate(variable_names):
        source_index = None
        if column_indices is not None and position < len(column_indices):
            source_index = column_indices[position]
        source_name = str(symbol)
        if source_index is not None and source_names is not None:
            idx = int(source_index)
            if 0 <= idx < len(source_names):
                source_name = str(source_names[idx])
        rows.append(
            {
                "symbol": str(symbol),
                "input_position": position,
                "source_index": None if source_index is None else int(source_index),
                "source_name": source_name,
            }
        )
    return rows


def dataset_variable_mapping(dataset, source_names: Optional[Sequence[str]] = None):
    """Build a mapping from an EquationSpec's gene_col_* metadata."""
    params = getattr(dataset.spec, "parameters", {}) or {}
    indices = []
    for position, _ in enumerate(dataset.spec.variable_names):
        value = params.get(f"gene_col_{position}")
        indices.append(None if value is None else int(value))
    return variable_mapping(
        dataset.spec.variable_names,
        column_indices=indices,
        source_names=source_names,
    )


def make_equation_record(
    *,
    eq_id: str,
    predicted_expr: str,
    variable_names: Sequence[str],
    scores: Mapping[str, Any],
    true_expr: str = "",
    mapping: Optional[Sequence[Mapping[str, Any]]] = None,
    candidate_expressions: Optional[Sequence[str]] = None,
    decoder: str,
    failure_reason: Optional[str] = None,
    decoder_metadata: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create the required per-problem equation record used by Phases 4--8."""
    raw = (predicted_expr or "").strip()
    simplified, simplify_error = simplify_expression(raw)
    candidates = [str(x) for x in (candidate_expressions or []) if str(x).strip()]
    if raw and raw not in candidates:
        candidates.insert(0, raw)

    valid = bool(scores.get("valid_pred", 0.0))
    reason = failure_reason
    if not reason and not raw:
        reason = "decoder_returned_no_expression"
    elif not reason and not valid:
        reason = "expression_evaluation_failed"

    record = {
        "eq_id": str(eq_id),
        "true": true_expr,
        "true_expr": true_expr,
        "pred": raw,
        "pred_raw": raw,
        "pred_simplified": simplified,
        "candidate_expressions": candidates,
        "variable_names": [str(x) for x in variable_names],
        "variable_mapping": list(mapping or variable_mapping(variable_names)),
        "decoder": decoder,
        "decoder_metadata": dict(decoder_metadata or {}),
        "simplification_error": simplify_error,
        "failure_reason": reason,
        **dict(scores),
        **extra,
    }
    return record

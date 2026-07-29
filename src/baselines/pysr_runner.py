"""Small, dependency-isolated PySR adapter used by local CPU runs."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_pysr_expression(
    X: np.ndarray,
    y: np.ndarray,
    variable_names: Sequence[str],
    *,
    niterations: int,
    random_state: int = 0,
    timeout_in_seconds: float | None = None,
) -> str:
    """Fit one bounded PySR problem and return its best expression."""
    from pysr import PySRRegressor

    kwargs = dict(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=20,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        random_state=random_state,
        parallelism="multithreading",
    )
    if timeout_in_seconds is not None:
        kwargs["timeout_in_seconds"] = float(timeout_in_seconds)
    model = PySRRegressor(**kwargs)
    model.fit(X, y, variable_names=list(variable_names))
    return str(model.get_best()["equation"])

"""Phase 0 smoke test: PySR on the same synthetic equation as NeSymReS."""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        print(f"PySR not installed: {exc}")
        return 1

    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, size=200)
    y = x * np.sin(x)

    model = PySRRegressor(
        niterations=20,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        maxsize=20,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
    )
    model.fit(x.reshape(-1, 1), y)
    print("Best equation:", model.get_best()["equation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

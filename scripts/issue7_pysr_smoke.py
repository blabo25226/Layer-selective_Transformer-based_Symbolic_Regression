"""Issue 7 smoke: PySR on one Phase 1 activation equation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic_grn import load_problem  # noqa: E402

DATA_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
OUT = ROOT / "results" / "phase_results" / "issue7_pysr_smoke.json"


def main() -> int:
    index_path = DATA_DIR / "index.json"
    if not index_path.exists():
        print("Missing Phase 1 data. Run scripts/issue6_generate_synthetic.py first.")
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    item = next(x for x in index if x["family"] == "activation" and x["split"] == "train")
    ds = load_problem(DATA_DIR / item["file"])

    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        print(f"PySR not installed: {exc}")
        return 1

    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        maxsize=25,
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        random_state=0,
    )
    # PySR prefers feature names
    model.fit(ds.X, ds.y, variable_names=ds.spec.variable_names)
    best = model.get_best()
    pred = model.predict(ds.X)
    nmse = float(np.mean((ds.y - pred) ** 2) / (np.var(ds.y) + 1e-12))

    payload = {
        "eq_id": ds.spec.eq_id,
        "family": ds.spec.family,
        "true_params": ds.spec.parameters,
        "true_expr_template": ds.spec.target_expr,
        "pysr_equation": str(best["equation"]),
        "pysr_loss": float(best["loss"]) if "loss" in best else None,
        "nmse": nmse,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.pysr_runner import fit_pysr_expression
from scripts.build_colab_notebooks import validate_notebook
from scripts.validate_gpu_run import resolve_dream4_networks


def _method(mean: float) -> dict:
    return {
        "mean_in": mean,
        "mean_hold": mean + 0.1,
        "gap_mean": 0.1,
    }


def _fold_method(value: float) -> dict:
    return {
        "hold_valid_rate": value,
        "hold_near_singularity": 0.0,
        "hold_extrapolation_valid": 1.0,
    }


def test_pysr_adapter_uses_reduced_budget(monkeypatch) -> None:
    captured = {}

    class FakeRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, X, y, variable_names):
            captured["variable_names"] = variable_names

        def get_best(self):
            return {"equation": "x_1"}

    monkeypatch.setitem(
        sys.modules, "pysr", SimpleNamespace(PySRRegressor=FakeRegressor)
    )
    expression = fit_pysr_expression(
        np.ones((2, 1)),
        np.ones(2),
        ["x_1"],
        niterations=12,
        random_state=7,
        timeout_in_seconds=15,
    )
    assert expression == "x_1"
    assert captured["niterations"] == 12
    assert captured["random_state"] == 7
    assert captured["parallelism"] == "multithreading"
    assert captured["timeout_in_seconds"] == 15.0


def test_phase8_aggregate_merges_local_pysr(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    nesymres = run / "phase8_lodo_seed0"
    pysr = run / "phase8_pysr_seed0"
    nesymres.mkdir(parents=True)
    pysr.mkdir(parents=True)
    (nesymres / "lodo_results.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "aggregate": {"pretrained_beam": _method(1.0)},
                "per_fold": [
                    {
                        "holdout": "1",
                        "pretrained_beam": _fold_method(0.8),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (pysr / "pysr_results.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "aggregate": {"pysr": _method(0.5)},
                "per_fold": [
                    {"holdout": "1", "pysr": _fold_method(0.9)}
                ],
            }
        ),
        encoding="utf-8",
    )

    from scripts import aggregate_phase8_runs

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_phase8_runs.py",
            "--run-dir",
            str(run),
            "--seeds",
            "0",
        ],
    )
    assert aggregate_phase8_runs.main() == 0
    summary = json.loads(
        (run / "phase8_lodo_multiseed" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["pysr_included"] is True
    assert set(summary["summary"]) == {"pretrained_beam", "pysr"}


def test_phase9_aggregates_local_pysr_before_validation() -> None:
    payload = validate_notebook(
        run_id="test_run",
        pinned_source_commit="deadbeef",
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )
    aggregate_at = source.index("scripts/aggregate_phase8_runs.py")
    validate_at = source.index("scripts/validate_gpu_run.py")
    assert aggregate_at < validate_at
    assert 'phase8_summary.get("pysr_included")' in source
    assert 'graph_dir = REPO_ROOT / "graphs" / RUN_ID' in source
    assert "graph_dir.mkdir(parents=True, exist_ok=True)" in source


def test_validator_accepts_documented_phase7_curtailment(tmp_path) -> None:
    out = tmp_path / "phase7_multiseed"
    out.mkdir()
    scope = {
        "reason": "compute budget",
        "selection_rule": "largest common completed prefix",
        "planned_networks": [1, 2, 3, 4, 5],
        "included_networks": [1, 2, 3],
    }
    (out / "summary.json").write_text(
        json.dumps({"execution_scope": scope}), encoding="utf-8"
    )
    (out / "curtailment.json").write_text(
        json.dumps(scope), encoding="utf-8"
    )
    included, loaded_scope = resolve_dream4_networks(
        tmp_path, [1, 2, 3, 4, 5]
    )
    assert included == [1, 2, 3]
    assert loaded_scope == scope

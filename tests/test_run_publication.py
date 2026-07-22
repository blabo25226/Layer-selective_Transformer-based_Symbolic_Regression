"""Tests for completed-run validation and lightweight Git publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_run_summary import main as export_main  # noqa: E402
from scripts.validate_gpu_run import main as validate_main  # noqa: E402


def test_validate_and_export_completed_run(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "example"
    required = [
        run / "phase4_multiseed" / "contrib_aggregate.json",
        run / "phase4_multiseed" / "absolute_improvements_aggregate.json",
        run / "phase4_multiseed" / "layer_ranking_scores.json",
        run / "phase4_multiseed" / "layer_ranking_metadata.json",
        run / "phase4_multiseed" / "layer_rankings.json",
        run / "phase4_multiseed" / "layer_importance_evidence.json",
        run / "phase4_multiseed" / "ranking_stability.json",
        run / "phase5_multiseed" / "summary.json",
        run / "phase6_noise_multiseed" / "summary.json",
        run / "phase8_lodo_multiseed" / "summary.json",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "git": {"commit": "abc", "branch": "test"},
        "checkpoint": {"sha256": "123"},
        "parameters": {"LTSR_DREAM4": "0"},
    }), encoding="utf-8")
    (run / "phase5_multiseed" / "equations.json").write_text(
        json.dumps({"per_problem": [{
            "eq_id": "e1",
            "true_expr": "x_1",
            "pred": "x_1",
            "pred_raw": "x_1",
            "pred_simplified": "x_1",
            "candidate_expressions": ["x_1"],
            "variable_names": ["x_1"],
            "variable_mapping": [{
                "symbol": "x_1", "input_position": 0,
                "source_index": None, "source_name": "x_1",
            }],
            "decoder": "test",
            "failure_reason": None,
            "valid_pred": 1.0,
        }]}),
        encoding="utf-8",
    )
    reports = run / "reports"
    reports.mkdir()
    (reports / "summary.md").write_text("# Summary\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["validate_gpu_run.py", "--run-dir", str(run)])
    assert validate_main() == 0
    out_root = tmp_path / "published"
    monkeypatch.setattr(sys, "argv", [
        "export_run_summary.py", "--run-dir", str(run), "--out-root", str(out_root),
    ])
    assert export_main() == 0
    published = out_root / "example"
    assert (published / "manifest.json").is_file()
    assert (published / "phase5_summary.json").is_file()
    assert (published / "phase4_ranking_scores.json").is_file()
    assert (published / "phase4_rankings.json").is_file()
    assert (published / "phase4_importance_evidence.json").is_file()
    assert (published / "phase4_ranking_stability.json").is_file()
    assert (published / "reports" / "summary.md").is_file()


def test_validator_rejects_incomplete_equation_record(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "invalid"
    required = [
        run / "phase4_multiseed" / "contrib_aggregate.json",
        run / "phase4_multiseed" / "absolute_improvements_aggregate.json",
        run / "phase4_multiseed" / "layer_ranking_scores.json",
        run / "phase4_multiseed" / "layer_ranking_metadata.json",
        run / "phase4_multiseed" / "layer_rankings.json",
        run / "phase4_multiseed" / "layer_importance_evidence.json",
        run / "phase4_multiseed" / "ranking_stability.json",
        run / "phase5_multiseed" / "summary.json",
        run / "phase6_noise_multiseed" / "summary.json",
        run / "phase8_lodo_multiseed" / "summary.json",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "parameters": {"LTSR_DREAM4": "0"},
    }), encoding="utf-8")
    (run / "phase5_multiseed" / "equations.json").write_text(
        json.dumps({"per_problem": [{"eq_id": "e1", "pred": ""}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["validate_gpu_run.py", "--run-dir", str(run)])
    with pytest.raises(SystemExit):
        validate_main()

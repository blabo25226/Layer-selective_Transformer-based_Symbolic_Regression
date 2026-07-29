"""Tests for the dependency-light Colab orchestration layer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colab_runtime import (  # noqa: E402
    config_for,
    copy_tree,
    lock_run_config,
    prepare_continuation_run,
    restore_artifacts,
    run_command,
)


def test_registered_colab_configs_fix_noise_to_point_one():
    smoke = config_for("smoke", "smoke-id")
    paper = config_for("paper", "paper-id", max_parallel_seeds=3)
    assert smoke.noise == "0.1"
    assert smoke.dream4_networks == "1"
    assert paper.noise == "0.1"
    assert paper.dream4_networks == "1 2 3 4 5"
    assert paper.max_parallel_seeds == 3
    env = paper.pipeline_environment(Path("/content/LTSR"), 6, resume=True)
    assert env["NOISE"] == "0.1"
    assert env["START_PHASE"] == env["STOP_AFTER_PHASE"] == "6"
    assert env["STRICT_RESUME"] == "1"
    assert env["RESUME"] == "1"
    assert env["PY"] == "/content/ltsr-py310/bin/python"
    assert env["MPLBACKEND"] == "Agg"
    assert env["LTSR_PHASE7_TARGET_EVAL_BUDGET"] == "60"


def test_reduced_colab_config_applies_the_approved_compute_budget():
    reduced = config_for("reduced", "reduced-id", max_parallel_seeds=2)
    assert reduced.seeds == "0 1 2"
    assert reduced.n_per_skeleton == 12
    assert reduced.eval_limit == 30
    assert reduced.random_layer_seeds == "0 1 2"
    assert reduced.beam == 2
    assert reduced.bfgs_restarts == 2
    assert reduced.bfgs_stop == 0.5
    assert reduced.noise == "0.1"
    assert reduced.dream4_networks == "1 2 3 4 5"
    assert reduced.max_parallel_seeds == 2


def test_run_command_replaces_colab_inline_matplotlib_backend(
    monkeypatch, tmp_path
):
    observed = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(
            command=command,
            cwd=cwd,
            backend=env["MPLBACKEND"],
            unbuffered=env["PYTHONUNBUFFERED"],
            check=check,
        )

    monkeypatch.setattr("colab_runtime.subprocess.run", fake_run)
    run_command(tmp_path, ["python", "-m", "pytest"])
    assert observed == {
        "command": ["python", "-m", "pytest"],
        "cwd": tmp_path,
        "backend": "Agg",
        "unbuffered": "1",
        "check": True,
    }


def test_colab_run_control_refuses_setting_changes_under_same_id(tmp_path):
    drive = tmp_path / "drive"
    config = config_for("smoke", "same-id")
    path = lock_run_config(drive, config)
    assert json.loads(path.read_text(encoding="utf-8"))["noise"] == "0.1"
    assert lock_run_config(drive, config) == path

    # Bounded seed concurrency affects wall-clock only and may be tuned safely.
    parallel_changed = config_for("smoke", "same-id", max_parallel_seeds=2)
    assert lock_run_config(drive, parallel_changed) == path

    changed = config_for("paper", "same-id")
    with pytest.raises(RuntimeError, match="run configuration mismatch"):
        lock_run_config(drive, changed)


def test_copy_tree_ignores_incomplete_drive_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "result.json").write_text('{"ok": true}', encoding="utf-8")
    (source / "result.json.partial").write_text("{", encoding="utf-8")
    assert copy_tree(source, destination) == 1
    assert (destination / "result.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert not (destination / "result.json.partial").exists()


def test_restore_artifacts_does_not_copy_drive_control_file(tmp_path):
    drive = tmp_path / "drive"
    repo = tmp_path / "repo"
    run = drive / "runs" / "smoke"
    run.mkdir(parents=True)
    (run / "colab_control.json").write_text("{}", encoding="utf-8")

    assert restore_artifacts(repo, drive, "smoke") == 0
    assert not (repo / "results" / "runs" / "smoke").exists()


def test_prepare_continuation_run_copies_results_with_lineage(tmp_path):
    drive = tmp_path / "drive"
    source = drive / "runs" / "old"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "git": {"commit": "old-commit", "branch": "old-branch"},
            }
        ),
        encoding="utf-8",
    )
    source_config = config_for("smoke", "old")
    (source / "colab_control.json").write_text(
        json.dumps(source_config.scientific_dict()), encoding="utf-8"
    )
    result = source / "phase7_dream4_size100_seed0_net1" / "size100_results.json"
    result.parent.mkdir()
    result.write_text('{"complete": true}', encoding="utf-8")
    logs = source / "logs"
    logs.mkdir()
    (logs / "pipeline.log").write_text("old log", encoding="utf-8")

    marker = prepare_continuation_run(
        drive,
        source_run_id="old",
        target_config=config_for("smoke", "new"),
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["source_run_id"] == "old"
    assert payload["source_git"]["commit"] == "old-commit"
    assert [row["path"] for row in payload["imported_files"]] == [
        "phase7_dream4_size100_seed0_net1/size100_results.json"
    ]
    target = drive / "runs" / "new"
    assert (
        target / "phase7_dream4_size100_seed0_net1" / "size100_results.json"
    ).is_file()
    assert not (target / "manifest.json").exists()
    assert not (target / "colab_control.json").exists()
    assert not (target / "logs" / "pipeline.log").exists()
    assert (
        prepare_continuation_run(
            drive,
            source_run_id="old",
            target_config=config_for("smoke", "new"),
        )
        == marker
    )


def test_generated_notebooks_have_no_saved_execution_state():
    notebook_dir = ROOT / "notebooks" / "colab"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    assert len(notebooks) == 10
    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert "Python 3.10" in "".join(code_cells[0]["source"])
        assert "/content/ltsr-py310" in "".join(code_cells[0]["source"])
        assert all(cell["execution_count"] is None for cell in code_cells)
        assert all(cell["outputs"] == [] for cell in code_cells)
        if path.name != "00_setup_preflight.ipynb":
            source = "".join(
                "".join(cell["source"]) for cell in code_cells
            )
            assert "dependency_probe" in source
            assert "requirements/gpu.txt" in source


def test_gpu_pipeline_selects_the_noise_specific_phase4_suite():
    script = (ROOT / "scripts" / "run_gpu_pipeline.sh").read_text(
        encoding="utf-8"
    )
    assert 'NOISE_CANON=$("${PY_CMD[@]}"' in script
    assert 'diverse_gpu_n${NOISE_CANON}' in script
    assert "requires exactly one NOISE value" in script
    assert "run_phase4_seed()" in script
    assert "--resume-existing --seed-only" in script
    assert 'if [ "$MAX_PARALLEL_SEEDS" -gt 1 ]' in script

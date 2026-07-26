"""Google Colab orchestration helpers for the LTSR GPU pipeline.

The research logic remains in ``scripts/`` and ``src/``.  This module only
restores persistent artifacts, locks a run configuration, executes one phase,
and mirrors recoverable checkpoints to Google Drive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_URL = (
    "https://github.com/blabo25226/"
    "Layer-selective_Transformer-based_Symbolic_Regression.git"
)
BRANCH = "20260726/gpu-scale-prep-colab"


def require_python_310(executable: str | Path | None = None) -> None:
    """Require the active interpreter or an explicit worker to be Python 3.10."""
    if executable is None:
        version = sys.version.split()[0]
        major_minor = sys.version_info[:2]
    else:
        executable = str(executable)
        try:
            version = subprocess.check_output(
                [
                    executable,
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))",
                ],
                text=True,
            ).strip()
            major_minor = tuple(int(value) for value in version.split(".")[:2])
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise RuntimeError(
                f"cannot execute the requested Python 3.10 worker: {executable}"
            ) from exc
    if major_minor != (3, 10):
        raise RuntimeError(
            "LTSR GPU runs require Python 3.10; selected interpreter is "
            f"{version}. Run the first CondaColab bootstrap cell before setup."
        )


@dataclass(frozen=True)
class ColabRunConfig:
    """Scientific and runtime settings shared by Phase 4--8 notebooks."""

    run_id: str
    kind: str
    seeds: str
    n_per_skeleton: int
    epochs: int
    eval_limit: int
    lr_grid: str
    epoch_grid: str
    patience: int
    random_layer_seeds: str
    nmse_equiv_margin: float
    beam: int
    bfgs_restarts: int
    bfgs_stop: float
    pysr: bool
    dream4: bool
    dream4_networks: str
    dream4_size10_target_limit: int
    dream4_size100_sr_targets: int
    max_parallel_seeds: int = 1
    noise: str = "0.1"
    decode_timeout_sec: int = 240

    def scientific_dict(self) -> dict[str, Any]:
        """Return the immutable configuration persisted beside the Drive run."""
        values = asdict(self)
        # Bounded seed concurrency changes wall-clock only; the pipeline records
        # its current value in every manifest/resume breadcrumb.
        values.pop("max_parallel_seeds")
        return values

    def pipeline_environment(
        self,
        repo_root: Path,
        phase: int,
        resume: bool,
        python_executable: str | Path = "/usr/local/bin/python",
    ) -> dict[str, str]:
        if phase not in range(4, 9):
            raise ValueError(f"pipeline phase must be 4..8, got {phase}")
        return {
            "RUN_ID": self.run_id,
            "SEEDS": self.seeds,
            "NPS": str(self.n_per_skeleton),
            "EPOCHS": str(self.epochs),
            "EVAL_LIMIT": str(self.eval_limit),
            "LR_GRID": self.lr_grid,
            "EPOCH_GRID": self.epoch_grid,
            "PATIENCE": str(self.patience),
            "RANDOM_LAYER_SEEDS": self.random_layer_seeds,
            "NMSE_EQUIV_MARGIN": str(self.nmse_equiv_margin),
            "BEAM": str(self.beam),
            "BFGS_RESTARTS": str(self.bfgs_restarts),
            "BFGS_STOP": str(self.bfgs_stop),
            "NOISE": self.noise,
            "PYSR": "1" if self.pysr else "0",
            "DREAM4": "1" if self.dream4 else "0",
            "DREAM4_NETWORKS": self.dream4_networks,
            "DREAM4_SIZE10_TARGET_LIMIT": str(
                self.dream4_size10_target_limit
            ),
            "DREAM4_SR_TARGETS": str(self.dream4_size100_sr_targets),
            "DREAM4_SHARD_NETWORKS": "1",
            "MAX_PARALLEL_SEEDS": str(self.max_parallel_seeds),
            "RESUME": "1" if resume else "0",
            "STRICT_RESUME": "1",
            "START_PHASE": str(phase),
            "STOP_AFTER_PHASE": str(phase),
            "LTSR_DECODE_TIMEOUT_SEC": str(self.decode_timeout_sec),
            "LTSR_WEIGHTS": str(repo_root / "NSRS" / "weights" / "100M.ckpt"),
            "LTSR_CONFIG": str(
                repo_root / "NSRS" / "jupyter" / "100M" / "config.yaml"
            ),
            "LTSR_EQ_SETTING": str(
                repo_root / "NSRS" / "jupyter" / "100M" / "eq_setting.json"
            ),
            "DREAM4_ROOT": str(repo_root / "data" / "dream4"),
            "PYTHONUNBUFFERED": "1",
            "PY": str(python_executable),
        }


def config_for(kind: str, run_id: str, *, max_parallel_seeds: int = 1) -> ColabRunConfig:
    """Build the pre-registered smoke, pilot, or paper configuration."""
    common = {
        "run_id": run_id,
        "kind": kind,
        "noise": "0.1",
        "decode_timeout_sec": 240,
        "max_parallel_seeds": max_parallel_seeds,
        "nmse_equiv_margin": 0.05,
    }
    if kind == "smoke":
        return ColabRunConfig(
            **common,
            seeds="0 1",
            n_per_skeleton=2,
            epochs=1,
            eval_limit=2,
            lr_grid="1e-4",
            epoch_grid="1",
            patience=0,
            random_layer_seeds="0",
            beam=1,
            bfgs_restarts=1,
            bfgs_stop=0.2,
            pysr=False,
            dream4=True,
            dream4_networks="1",
            dream4_size10_target_limit=2,
            dream4_size100_sr_targets=2,
        )
    if kind == "pilot":
        return ColabRunConfig(
            **common,
            seeds="0 1",
            n_per_skeleton=24,
            epochs=8,
            eval_limit=30,
            lr_grid="1e-5 3e-5 1e-4",
            epoch_grid="4 8",
            patience=2,
            random_layer_seeds="0 1 2 3 4",
            beam=5,
            bfgs_restarts=5,
            bfgs_stop=2.0,
            pysr=True,
            dream4=False,
            dream4_networks="",
            dream4_size10_target_limit=0,
            dream4_size100_sr_targets=0,
        )
    if kind == "paper":
        return ColabRunConfig(
            **common,
            seeds="0 1 2 3 4",
            n_per_skeleton=24,
            epochs=8,
            eval_limit=0,
            lr_grid="1e-5 3e-5 1e-4",
            epoch_grid="4 8",
            patience=2,
            random_layer_seeds="0 1 2 3 4",
            beam=5,
            bfgs_restarts=5,
            bfgs_stop=2.0,
            pysr=True,
            dream4=True,
            dream4_networks="1 2 3 4 5",
            dream4_size10_target_limit=0,
            dream4_size100_sr_targets=0,
        )
    raise ValueError(f"unknown run kind: {kind!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True
    ).strip()


def assert_locked_source(repo_root: Path, drive_root: Path) -> str:
    """Require a clean worktree at the commit recorded by setup notebook."""
    lock_path = drive_root / "source_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing source lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    current = git_output(repo_root, "rev-parse", "HEAD")
    if current != lock.get("commit"):
        raise RuntimeError(
            f"source commit mismatch: lock={lock.get('commit')} current={current}"
        )
    dirty = git_output(repo_root, "status", "--porcelain")
    if dirty:
        raise RuntimeError(f"Colab source worktree is dirty:\n{dirty}")
    return current


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copy2(source, partial)
    os.replace(partial, destination)


def copy_tree(source: Path, destination: Path) -> int:
    """Copy a tree file-by-file with atomic destination replacement."""
    if not source.is_dir():
        return 0
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        relative = path.relative_to(source)
        _atomic_copy(path, destination / relative)
        copied += 1
    return copied


def sync_artifacts(repo_root: Path, drive_root: Path, run_id: str) -> dict[str, int]:
    """Mirror the run and graphs to Drive without deleting prior checkpoints."""
    return {
        "run_files": copy_tree(
            repo_root / "results" / "runs" / run_id,
            drive_root / "runs" / run_id,
        ),
        "graph_files": copy_tree(
            repo_root / "graphs" / run_id,
            drive_root / "graphs" / run_id,
        ),
    }


def restore_artifacts(repo_root: Path, drive_root: Path, run_id: str) -> int:
    """Restore the last Drive checkpoint into the fresh Colab VM."""
    return copy_tree(
        drive_root / "runs" / run_id,
        repo_root / "results" / "runs" / run_id,
    )


def restore_tar(archive: Path, destination: Path) -> None:
    """Safely extract an archive created by the setup notebook."""
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def restore_static_assets(repo_root: Path, drive_root: Path) -> None:
    """Restore checkpoint and external data cached by Phase 0."""
    checkpoint = drive_root / "checkpoints" / "100M.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint cache: {checkpoint}")
    weights = repo_root / "NSRS" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    for name in ("100M.ckpt", "10M.ckpt"):
        target = weights / name
        if not target.is_file() or target.stat().st_size != checkpoint.stat().st_size:
            _atomic_copy(checkpoint, target)

    dream4_archive = drive_root / "data" / "dream4.tar.gz"
    dream4_marker = (
        repo_root
        / "data"
        / "dream4"
        / "Size 10"
        / "DREAM4 training data"
        / "insilico_size10_1"
        / "insilico_size10_1_timeseries.tsv"
    )
    if not dream4_marker.is_file():
        restore_tar(dream4_archive, repo_root / "data")

    human_archive = drive_root / "data" / "gse112372_lps.tar.gz"
    human_dir = repo_root / "data" / "human" / "gse112372_lps"
    if not human_dir.is_dir() or not any(human_dir.rglob("*")):
        restore_tar(human_archive, repo_root / "data" / "human")


def lock_run_config(drive_root: Path, config: ColabRunConfig) -> Path:
    """Create or verify the immutable per-run Colab control file."""
    path = drive_root / "runs" / config.run_id / "colab_control.json"
    expected = config.scientific_dict()
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                "run configuration mismatch; choose a new run_id instead of "
                f"mixing results:\nexisting={existing}\nrequested={expected}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".json.partial")
        partial.write_text(
            json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(partial, path)
    return path


def expected_phase_output(repo_root: Path, config: ColabRunConfig, phase: int) -> Path:
    run = repo_root / "results" / "runs" / config.run_id
    outputs = {
        4: run / "phase4_multiseed" / "layer_ranking_scores.json",
        5: run / "phase5_multiseed" / "summary.json",
        6: run / "phase6_noise_multiseed" / "summary.json",
        7: run / "phase7_multiseed" / "summary.json",
        8: run / "phase8_lodo_multiseed" / "summary.json",
    }
    return outputs[phase]


def run_phase(
    repo_root: Path,
    drive_root: Path,
    config: ColabRunConfig,
    phase: int,
    *,
    sync_interval_sec: int = 180,
    python_executable: str | Path = "/usr/local/bin/python",
) -> Path:
    """Execute one pipeline phase and checkpoint it to Drive periodically."""
    repo_root = repo_root.resolve()
    drive_root = drive_root.resolve()
    if phase == 7 and not config.dream4:
        raise RuntimeError("Phase 7 is disabled for this run configuration")
    require_python_310(python_executable)
    assert_locked_source(repo_root, drive_root)
    lock_run_config(drive_root, config)
    restore_static_assets(repo_root, drive_root)
    restore_artifacts(repo_root, drive_root, config.run_id)

    manifest = repo_root / "results" / "runs" / config.run_id / "manifest.json"
    resume = manifest.is_file()
    env = os.environ.copy()
    env.update(
        config.pipeline_environment(
            repo_root, phase, resume, python_executable=python_executable
        )
    )
    command = ["bash", "scripts/run_gpu_pipeline.sh"]
    print(
        f"Starting Phase {phase}: run={config.run_id} kind={config.kind} "
        f"resume={resume}",
        flush=True,
    )
    process = subprocess.Popen(command, cwd=repo_root, env=env)
    last_sync = 0.0
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now - last_sync >= sync_interval_sec:
                counts = sync_artifacts(repo_root, drive_root, config.run_id)
                print(f"[Drive checkpoint] {counts}", flush=True)
                last_sync = now
            time.sleep(5)
    except BaseException:
        sync_artifacts(repo_root, drive_root, config.run_id)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
    counts = sync_artifacts(repo_root, drive_root, config.run_id)
    print(f"[Drive final sync] {counts}", flush=True)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    expected = expected_phase_output(repo_root, config, phase)
    if not expected.is_file():
        raise RuntimeError(f"Phase {phase} returned success but output is missing: {expected}")
    return expected


def run_command(
    repo_root: Path,
    command: Sequence[str],
    *,
    extra_env: Mapping[str, str] | None = None,
) -> None:
    """Run a diagnostic Phase 0--3 command from the fixed repo root."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    subprocess.run(list(command), cwd=repo_root, env=env, check=True)

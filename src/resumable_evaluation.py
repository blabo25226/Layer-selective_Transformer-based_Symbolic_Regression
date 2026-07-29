"""Atomic target-level checkpoints for long symbolic-regression evaluations."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CHECKPOINT_VERSION = 1


class TargetEvaluationBudgetReached(Exception):
    """Signal a clean process recycle after a bounded number of targets."""


class TargetEvaluationBudget:
    """Count newly checkpointed targets and request a process recycle."""

    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError(f"target evaluation budget must be >= 0, got {limit}")
        self.limit = limit
        self.completed = 0

    def record_checkpoint(self) -> None:
        """Record one durable target and raise when the process budget is spent."""
        self.completed += 1
        if self.limit > 0 and self.completed >= self.limit:
            raise TargetEvaluationBudgetReached(
                f"checkpointed {self.completed} new targets; recycle process"
            )


def capture_rng_state() -> dict[str, Any]:
    """Capture the RNG state needed to continue a target sequence exactly."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_target_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    progress: Mapping[str, Any],
) -> None:
    """Atomically persist progress and the RNG state after a completed target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "identity": dict(identity),
            "progress": dict(progress),
            "rng_state": capture_rng_state(),
        },
        partial,
    )
    os.replace(partial, path)


def load_target_checkpoint(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load a compatible checkpoint, refusing silent configuration mixing."""
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid target checkpoint payload: {path}")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"target checkpoint version mismatch: {payload.get('version')} "
            f"!= {CHECKPOINT_VERSION}"
        )
    identity = payload.get("identity")
    if identity != dict(expected_identity):
        raise RuntimeError(
            "target checkpoint identity mismatch; refuse to mix evaluations:\n"
            f"checkpoint={identity}\nexpected={dict(expected_identity)}"
        )
    if not isinstance(payload.get("progress"), dict):
        raise RuntimeError(f"target checkpoint has no progress mapping: {path}")
    if not isinstance(payload.get("rng_state"), dict):
        raise RuntimeError(f"target checkpoint has no RNG state: {path}")
    return payload


def completed_prefix(
    rows: Sequence[Mapping[str, Any]],
    expected_eq_ids: Sequence[str],
) -> int:
    """Validate that saved rows are an exact prefix of the current problems."""
    if len(rows) > len(expected_eq_ids):
        raise RuntimeError(
            f"checkpoint has {len(rows)} rows for {len(expected_eq_ids)} problems"
        )
    observed = [str(row.get("eq_id")) for row in rows]
    expected = list(expected_eq_ids[: len(rows)])
    if observed != expected:
        raise RuntimeError(
            "target checkpoint problem order mismatch:\n"
            f"checkpoint={observed}\nexpected={expected}"
        )
    return len(rows)

"""Tests for target-level symbolic-regression checkpoints."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.resumable_evaluation import (
    TargetEvaluationBudget,
    TargetEvaluationBudgetReached,
    completed_prefix,
    load_target_checkpoint,
    restore_rng_state,
    save_target_checkpoint,
)


def test_target_evaluation_budget_requests_recycle_after_durable_count():
    budget = TargetEvaluationBudget(2)
    budget.record_checkpoint()
    with pytest.raises(TargetEvaluationBudgetReached, match="2 new targets"):
        budget.record_checkpoint()

    unlimited = TargetEvaluationBudget(0)
    for _ in range(100):
        unlimited.record_checkpoint()

    with pytest.raises(ValueError, match="must be >= 0"):
        TargetEvaluationBudget(-1)


def test_target_checkpoint_restores_rng_and_progress(tmp_path):
    path = tmp_path / "target_checkpoint.pt"
    identity = {"seed": 3, "network": 2}
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    save_target_checkpoint(
        path,
        identity=identity,
        progress={"sr": {"net2": {"condition": {"per_problem": [{"eq_id": "a"}]}}}},
    )
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    payload = load_target_checkpoint(path, expected_identity=identity)
    assert payload is not None
    restore_rng_state(payload["rng_state"])
    observed = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert observed == pytest.approx(expected)
    assert payload["progress"]["sr"]["net2"]["condition"]["per_problem"] == [
        {"eq_id": "a"}
    ]
    assert not path.with_name(path.name + ".partial").exists()


def test_target_checkpoint_refuses_identity_or_problem_order_change(tmp_path):
    path = tmp_path / "target_checkpoint.pt"
    save_target_checkpoint(path, identity={"seed": 0}, progress={"sr": {}})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_target_checkpoint(path, expected_identity={"seed": 1})

    assert completed_prefix([{"eq_id": "a"}], ["a", "b"]) == 1
    with pytest.raises(RuntimeError, match="problem order mismatch"):
        completed_prefix([{"eq_id": "b"}], ["a", "b"])

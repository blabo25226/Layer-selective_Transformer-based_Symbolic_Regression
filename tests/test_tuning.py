"""Tests for validation-only fine-tuning selection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training.single_layer import train_selective  # noqa: E402
from training.tuning import build_config_grid, tune_selective  # noqa: E402


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        nums, tokens = batch
        return self.weight * nums, tokens

    @staticmethod
    def compute_loss(output, target):
        return torch.mean((output - target) ** 2)


def test_build_config_grid_is_cartesian_and_validated():
    grid = build_config_grid([1e-4, 1e-5], [2, 4], patience=1)
    assert len(grid) == 4
    assert {(c.lr, c.epochs) for c in grid} == {
        (1e-4, 2), (1e-4, 4), (1e-5, 2), (1e-5, 4)
    }


def test_tuning_selects_by_validation_only(monkeypatch):
    base = torch.nn.Linear(1, 1, bias=False)
    configs = build_config_grid([1e-3, 1e-4], [2], patience=1)

    def fake_train(model, loader, layers, *, lr, epochs, **kwargs):
        with torch.no_grad():
            model.weight.fill_(lr)
        return {
            "best_val_ce": 0.1 if lr == 1e-4 else 0.5,
            "trainable": 1.0,
            "total": 1.0,
            "epochs": float(epochs),
        }

    monkeypatch.setattr("training.tuning.train_selective", fake_train)
    model, selection = tune_selective(
        base,
        lambda: object(),
        object(),
        None,
        configs,
        device=torch.device("cpu"),
        seed=7,
    )

    assert selection["criterion"] == "validation_ce"
    assert selection["candidate_count"] == 2
    assert selection["selected"]["lr"] == 1e-4
    assert abs(float(model.weight.detach()) - 1e-4) < 1e-10


def test_train_selective_restores_best_validation_epoch():
    train = DataLoader(
        TensorDataset(torch.ones(1), torch.ones(1)), batch_size=1, shuffle=False
    )
    validation = DataLoader(
        TensorDataset(torch.ones(1), torch.zeros(1)), batch_size=1, shuffle=False
    )
    one_epoch = _ToyModel()
    three_epochs = _ToyModel()

    train_selective(
        one_epoch, train, None, epochs=1, lr=0.1,
        device=torch.device("cpu"), val_loader=validation, patience=0,
    )
    info = train_selective(
        three_epochs, train, None, epochs=3, lr=0.1,
        device=torch.device("cpu"), val_loader=validation, patience=0,
    )

    assert info["best_epoch"] == 1.0
    assert info["stopped_epoch"] == 3.0
    assert torch.allclose(one_epoch.weight, three_epochs.weight)

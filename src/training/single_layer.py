"""Single-layer / selective-layer fine-tuning for NeSymReS (Phase 3)."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from models.layer_selector import (
    count_trainable_parameters,
    set_trainable_layers,
    unfreeze_all,
)


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)


def train_selective(
    model: torch.nn.Module,
    train_loader: DataLoader,
    layer_names: Optional[Sequence[str]],
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Fine-tune `model` with only `layer_names` trainable.
    If layer_names is None, train all parameters.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    if layer_names is None:
        unfreeze_all(model)
    else:
        set_trainable_layers(model, list(layer_names), freeze_others=True)

    trainable, total = count_trainable_parameters(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {
            "final_loss": float("nan"),
            "trainable": 0,
            "total": total,
            "epochs": 0,
        }

    optim = torch.optim.Adam(params, lr=lr)
    last_loss = float("nan")

    for _ in range(epochs):
        epoch_losses: List[float] = []
        for batch in train_loader:
            nums, tokens = batch[0].to(device), batch[1].to(device)
            optim.zero_grad(set_to_none=True)
            output, trg = model.forward([nums, tokens])
            loss = model.compute_loss(output, trg)
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
        last_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))

    return {
        "final_loss": last_loss,
        "trainable": float(trainable),
        "total": float(total),
        "epochs": float(epochs),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }

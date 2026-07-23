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
    val_loader: Optional[DataLoader] = None,
    patience: int = 0,
    min_delta: float = 1e-4,
) -> Dict[str, float]:
    """
    Fine-tune `model` with only `layer_names` trainable.
    If layer_names is None, train all parameters.

    If ``val_loader`` is supplied, restore the weights with the best validation
    CE. A positive ``patience`` additionally enables early stopping; zero runs
    every requested epoch but still restores the best checkpoint.
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
            "best_val_ce": float("nan"),
            "best_epoch": 0.0,
            "stopped_epoch": 0.0,
        }

    optim = torch.optim.Adam(params, lr=lr)
    last_loss = float("nan")
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0
    stopped_epoch = 0

    def _ce(loader: DataLoader) -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                nums, tokens = batch[0].to(device), batch[1].to(device)
                output, trg = model.forward([nums, tokens])
                losses.append(float(model.compute_loss(output, trg).cpu()))
        model.train()
        return float(sum(losses) / max(len(losses), 1)) if losses else float("nan")

    for ep in range(epochs):
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
        stopped_epoch = ep + 1

        if val_loader is not None:
            val_ce = _ce(val_loader)
            if val_ce < best_val - min_delta:
                best_val = val_ce
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep + 1
                bad = 0
            else:
                bad += 1
                if patience > 0 and bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "final_loss": last_loss,
        "trainable": float(trainable),
        "total": float(total),
        "epochs": float(stopped_epoch),
        "trainable_fraction": float(trainable / total) if total else 0.0,
        "best_val_ce": float(best_val) if best_state is not None else float("nan"),
        "best_epoch": float(best_epoch),
        "stopped_epoch": float(stopped_epoch),
    }

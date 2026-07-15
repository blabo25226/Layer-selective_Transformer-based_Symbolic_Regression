"""Validation-only hyperparameter selection for selective/full fine-tuning."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .single_layer import clone_model, train_selective


@dataclass(frozen=True)
class FineTuneConfig:
    """One candidate configuration evaluated with validation CE only."""

    lr: float
    epochs: int
    patience: int = 2
    min_delta: float = 1e-4


def build_config_grid(
    lrs: Iterable[float],
    epochs: Iterable[int],
    *,
    patience: int = 2,
    min_delta: float = 1e-4,
) -> list[FineTuneConfig]:
    """Return a deterministic Cartesian grid and reject invalid candidates."""
    configs = [
        FineTuneConfig(float(lr), int(n_epochs), int(patience), float(min_delta))
        for lr in lrs
        for n_epochs in epochs
    ]
    if not configs:
        raise ValueError("Fine-tuning grid must contain at least one candidate")
    for config in configs:
        if config.lr <= 0 or config.epochs <= 0:
            raise ValueError(f"Invalid fine-tuning candidate: {config}")
        if config.patience < 0 or config.min_delta < 0:
            raise ValueError(f"Invalid early-stopping candidate: {config}")
    return configs


def seed_everything(seed: int) -> None:
    """Reset stochastic state so every candidate gets a paired comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tune_selective(
    base_model: torch.nn.Module,
    train_loader_factory: Callable[[], DataLoader],
    val_loader: DataLoader,
    layer_names: Optional[Sequence[str]],
    configs: Sequence[FineTuneConfig],
    *,
    device: torch.device,
    seed: int,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    """Select one model by validation CE without evaluating the test split."""
    if not configs:
        raise ValueError("At least one fine-tuning candidate is required")

    best_model: Optional[torch.nn.Module] = None
    best_score = float("inf")
    best_info: Optional[Dict[str, float]] = None
    best_config: Optional[FineTuneConfig] = None
    trials: list[Dict[str, object]] = []

    for config in configs:
        seed_everything(seed)
        model = clone_model(base_model)
        info = train_selective(
            model,
            train_loader_factory(),
            layer_names,
            epochs=config.epochs,
            lr=config.lr,
            device=device,
            val_loader=val_loader,
            patience=config.patience,
            min_delta=config.min_delta,
        )
        score = float(info.get("best_val_ce", float("nan")))
        trials.append({"config": asdict(config), "val_ce": score, "train": info})
        if math.isfinite(score) and score < best_score:
            if best_model is not None:
                del best_model
            best_model = model
            best_score = score
            best_info = info
            best_config = config
        else:
            del model

    if best_model is None or best_info is None or best_config is None:
        raise RuntimeError("Every fine-tuning candidate produced a non-finite validation CE")

    selection: Dict[str, object] = {
        "criterion": "validation_ce",
        "candidate_count": len(configs),
        "selected": asdict(best_config),
        "best_val_ce": best_score,
        "trials": trials,
        "train": best_info,
    }
    return best_model, selection

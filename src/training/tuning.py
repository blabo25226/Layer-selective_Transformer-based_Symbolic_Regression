"""Validation-only hyperparameter selection for selective/full fine-tuning."""

from __future__ import annotations

import copy
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

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = float("inf")
    best_info: Optional[Dict[str, float]] = None
    best_config: Optional[FineTuneConfig] = None
    trials: list[Dict[str, object]] = []

    grouped: Dict[tuple[float, int, float], list[FineTuneConfig]] = {}
    for config in configs:
        grouped.setdefault(
            (config.lr, config.patience, config.min_delta), []
        ).append(config)

    trial_by_config: Dict[FineTuneConfig, Dict[str, object]] = {}
    for (lr, patience, min_delta), group in grouped.items():
        epoch_caps = sorted({config.epochs for config in group})
        seed_everything(seed)
        model = clone_model(base_model)
        info = train_selective(
            model,
            train_loader_factory(),
            layer_names,
            epochs=max(epoch_caps),
            lr=lr,
            device=device,
            val_loader=val_loader,
            patience=patience,
            min_delta=min_delta,
            checkpoint_epochs=epoch_caps,
        )
        candidates = info.pop("_epoch_candidates")
        for config in group:
            candidate = candidates[config.epochs]
            candidate_info = candidate["train"]
            score = float(candidate_info.get("best_val_ce", float("nan")))
            trial_by_config[config] = {
                "config": asdict(config),
                "val_ce": score,
                "train": candidate_info,
            }
            if math.isfinite(score) and score < best_score:
                best_state = copy.deepcopy(candidate["state_dict"])
                best_score = score
                best_info = candidate_info
                best_config = config
        del model
        del candidates

    trials = [trial_by_config[config] for config in configs]

    if best_state is None or best_info is None or best_config is None:
        raise RuntimeError("Every fine-tuning candidate produced a non-finite validation CE")
    best_model = clone_model(base_model)
    best_model.load_state_dict(best_state)
    best_model.to(device)

    selection: Dict[str, object] = {
        "criterion": "validation_ce",
        "candidate_count": len(configs),
        "selected": asdict(best_config),
        "best_val_ce": best_score,
        "trials": trials,
        "train": best_info,
    }
    return best_model, selection

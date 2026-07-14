"""Thin adapter around pretrained NeSymReS inference."""

from __future__ import annotations

import contextlib
import io
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import omegaconf
import torch

from nesymres.architectures.model import Model
from nesymres.dclasses import BFGSParams, FitParams


def load_nesymres(
    weights: Path,
    config_yaml: Path,
    eq_setting_json: Path,
    beam_size: int = 2,
) -> Tuple[Model, FitParams]:
    with eq_setting_json.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    cfg = omegaconf.OmegaConf.load(config_yaml)

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(
        word2id=eq_setting["word2id"],
        id2word={int(k): v for k, v in eq_setting["id2word"].items()},
        una_ops=eq_setting["una_ops"],
        bin_ops=eq_setting["bin_ops"],
        total_variables=list(eq_setting["total_variables"]),
        total_coefficients=list(eq_setting["total_coefficients"]),
        rewrite_functions=list(eq_setting["rewrite_functions"]),
        bfgs=bfgs,
        beam_size=beam_size,
    )
    model = Model.load_from_checkpoint(str(weights), cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model, params_fit


def pad_features_to_three(X: np.ndarray) -> np.ndarray:
    """NeSymReS expects up to 3 variables (x_1..x_3)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if X.shape[1] >= 3:
        return X[:, :3]
    pad = np.zeros((X.shape[0], 3 - X.shape[1]), dtype=np.float32)
    return np.concatenate([X, pad], axis=1)


def predict_equation(
    model: Model,
    params_fit: FitParams,
    X: np.ndarray,
    y: np.ndarray,
    *,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run NeSymReS fitfunc; return best expression and raw output."""
    Xp = pad_features_to_three(X)
    y = np.asarray(y, dtype=np.float32).ravel()
    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        with torch.no_grad():
            output = fitfunc(Xp, y)

    preds = output.get("best_bfgs_preds") or output.get("all_bfgs_preds") or []
    losses = output.get("best_bfgs_loss") or output.get("all_bfgs_loss") or []
    best_expr = preds[0] if preds else ""
    best_loss = float(losses[0]) if losses else float("inf")
    return {
        "equation": str(best_expr),
        "bfgs_loss": best_loss,
        "all_preds": [str(p) for p in preds],
        "raw": {k: v for k, v in output.items() if k != "raw"},
    }

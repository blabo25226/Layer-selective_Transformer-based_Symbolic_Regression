"""Phase 0 smoke test: NeSymReS inference on a simple synthetic equation."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import omegaconf
import torch
from sympy import lambdify

from nesymres.architectures.model import Model
from nesymres.dclasses import BFGSParams, FitParams

ROOT = Path(__file__).resolve().parents[1]
NSRS = ROOT / "NSRS"
WEIGHTS = NSRS / "weights" / "10M.ckpt"
EQ_SETTING = NSRS / "jupyter" / "100M" / "eq_setting.json"
CONFIG = NSRS / "jupyter" / "100M" / "config.yaml"
TARGET_EQ = "x_1*sin(x_1)"


def main() -> int:
    missing = [p for p in (WEIGHTS, EQ_SETTING, CONFIG) if not p.exists()]
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    with EQ_SETTING.open(encoding="utf-8") as f:
        eq_setting = json.load(f)
    cfg = omegaconf.OmegaConf.load(CONFIG)

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    beam_size = omegaconf.OmegaConf.select(cfg, "inference.beam_size", default=2)
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

    print(f"Loading checkpoint: {WEIGHTS}")
    model = Model.load_from_checkpoint(str(WEIGHTS), cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("Device: cuda")
    else:
        print("Device: cpu")

    fitfunc = partial(model.fitfunc, cfg_params=params_fit)

    number_of_points = 200
    n_variables = 1
    max_supp = cfg.dataset_train.fun_support["max"]
    min_supp = cfg.dataset_train.fun_support["min"]
    x = torch.rand(number_of_points, len(eq_setting["total_variables"])) * (
        max_supp - min_supp
    ) + min_supp
    x[:, n_variables:] = 0
    x_dict = {name: x[:, idx].cpu() for idx, name in enumerate(eq_setting["total_variables"])}
    y = lambdify(",".join(eq_setting["total_variables"]), TARGET_EQ)(**x_dict)

    print(f"Target equation: {TARGET_EQ}")
    output = fitfunc(x, y)
    print("Best predictions:", output["best_bfgs_preds"])
    print("Best losses:", output["best_bfgs_loss"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

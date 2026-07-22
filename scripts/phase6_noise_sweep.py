"""H3 noise-robustness sweep: 2x2 (FT x decode) across noise levels (A-4).

Tests hypothesis H3 — that NeSymReS + TPSR keeps a better accuracy/complexity
tradeoff than plain beam as observation noise grows — by running the Phase 6 2x2
on several noisy versions of the diverse suite and tabulating how NMSE/R2 degrade.

First build the noisy suites, then run this in the NeSymReS env:

    python scripts/generate_diverse_suite.py --noise 0.0 0.05 0.1 0.2 --n-per-skeleton 8
    python scripts/phase6_noise_sweep.py --noise 0.0 0.05 0.1 0.2 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from data.finetune_dataset import (  # noqa: E402
    GRNFinetuneDataset,
    collate_finetune,
    load_split_problems,
)
from models.nesymres_adapter import load_nesymres  # noqa: E402
from training.selective_layers import resolve_selected_layers  # noqa: E402
from training.single_layer import clone_model, train_selective  # noqa: E402

# Reuse Phase 6 building blocks (identical decode / scoring).
from phase6_tpsr_2x2 import (  # noqa: E402
    WEIGHTS,
    CONFIG,
    EQ_SETTING,
    PHASE4_CONTRIB,
    eval_one,
    make_beam_fit_params,
    fmt,
    sanitize,
)
from experiment_runtime import phase_output_paths  # noqa: E402

OUT_DIR, REPORT = phase_output_paths(ROOT, "phase6_noise", "phase6_noise_report.md")


def log(msg: str) -> None:
    print(msg, flush=True)


def suite_dir(root: Path, tag: str, noise: float) -> Path:
    return root / (tag if noise == 0.0 else f"{tag}_n{noise}")


def run_level(
    base_model, params_fit, beam_params, tpsr_kwargs, word2id, data_dir, layers, args, device
) -> Dict[str, Dict[str, Any]]:
    train_problems = load_split_problems(data_dir, "train")
    test_problems = load_split_problems(data_dir, "test")
    if args.eval_limit > 0:
        test_problems = test_problems[: args.eval_limit]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_ds = GRNFinetuneDataset(
        train_problems, word2id, max_points=args.max_points, seed=args.seed
    )
    loader = DataLoader(
        train_ds,
        batch_size=min(args.batch_size, max(len(train_ds), 1)),
        shuffle=True,
        collate_fn=collate_finetune,
        generator=torch.Generator().manual_seed(args.seed),
    )
    ft_model = clone_model(base_model)
    train_selective(ft_model, loader, layers, epochs=args.epochs, lr=args.lr, device=device)

    cells = [
        ("pretrained_beam", clone_model(base_model), "beam"),
        ("pretrained_tpsr", clone_model(base_model), "tpsr"),
        ("selective_beam", ft_model, "beam"),
        ("selective_tpsr", clone_model(ft_model), "tpsr"),
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for name, model, decode in cells:
        torch.manual_seed(args.seed + 10_000)
        np.random.seed(args.seed + 10_000)
        random.seed(args.seed + 10_000)
        model.eval()
        started = time.time()
        ev = eval_one(
            model,
            beam_params if decode == "beam" else params_fit,
            test_problems,
            decode=decode,
            tpsr_kwargs=tpsr_kwargs if decode == "tpsr" else None,
        )
        out[name] = {
            **ev["aggregate"],
            "elapsed_sec": time.time() - started,
            "per_problem": ev["per_problem"],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--data-root", default=str(ROOT / "results" / "synthetic"))
    parser.add_argument("--tag", default="diverse_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--bfgs-restarts", type=int, default=1)
    parser.add_argument("--bfgs-stop-time", type=float, default=0.5)
    parser.add_argument("--rollout", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--layers", default="")
    parser.add_argument("--layer-rule", choices=["top", "middle", "bottom"], default="top")
    parser.add_argument("--layer-mode", choices=["accuracy", "ce"], default="accuracy")
    parser.add_argument("--layer-k", type=int, default=3)
    parser.add_argument("--contributions", default=str(PHASE4_CONTRIB))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explicit = [x.strip() for x in args.layers.split(",") if x.strip()] or None
    layers, layer_src, layer_rule = resolve_selected_layers(
        Path(args.contributions), mode=args.layer_mode, rule=args.layer_rule,
        k=args.layer_k, explicit=explicit,
    )
    log(f"Device: {device} | layers ({layer_rule}, {layer_src}): {layers}")

    base_model, params_fit = load_nesymres(WEIGHTS, CONFIG, EQ_SETTING, beam_size=args.beam_size)
    beam_params = make_beam_fit_params(params_fit, args.beam_size, args.bfgs_restarts, args.bfgs_stop_time)
    tpsr_kwargs = {
        "rollout": args.rollout,
        "horizon": args.horizon,
        "width": args.width,
        "num_beams": args.num_beams,
        "bfgs_restarts": args.bfgs_restarts,
        "bfgs_stop_time": args.bfgs_stop_time,
    }
    word2id = json.loads(EQ_SETTING.read_text(encoding="utf-8"))["word2id"]

    root = Path(args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_noise: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for noise in args.noise:
        d = suite_dir(root, args.tag, noise)
        if not (d / "index.json").exists():
            log(f"SKIP noise={noise}: missing suite {d} (run generate_diverse_suite.py --noise {noise})")
            continue
        t0 = time.time()
        log(f"\n=== noise={noise} | {d.name} ===")
        by_noise[str(noise)] = run_level(
            base_model, params_fit, beam_params, tpsr_kwargs, word2id, d, layers, args, device
        )
        log(f"  done ({time.time()-t0:.1f}s)")

    (OUT_DIR / "noise_sweep.json").write_text(
        json.dumps(sanitize(by_noise), indent=2), encoding="utf-8"
    )

    cells = ["pretrained_beam", "pretrained_tpsr", "selective_beam", "selective_tpsr"]
    noises = sorted(by_noise, key=float)

    def cell(noise: str, name: str, metric: str) -> float:
        return float(by_noise.get(noise, {}).get(name, {}).get(metric, float("nan")))

    lines = [
        "# Phase 6 noise sweep — H3 robustness (A-4)",
        "",
        f"- Layers ({layer_rule}, source=`{layer_src}`): `{', '.join(layers)}`",
        f"- Noise levels: {noises}  |  epochs: {args.epochs}, TPSR rollout={args.rollout}",
        f"- Device: `{device}`",
        "",
        "## Failure-penalized NMSE (median, lower better) vs noise",
        "",
        "| noise | " + " | ".join(cells) + " |",
        "|-------|" + "|".join(["------"] * len(cells)) + "|",
    ]
    for nz in noises:
        lines.append(
            f"| {nz} | " + " | ".join(fmt(cell(nz, c, "penalized_nmse")) for c in cells) + " |"
        )
    lines += ["", "## R² (median, higher better) vs noise", "",
              "| noise | " + " | ".join(cells) + " |",
              "|-------|" + "|".join(["------"] * len(cells)) + "|"]
    for nz in noises:
        lines.append(
            f"| {nz} | " + " | ".join(fmt(cell(nz, c, "r2")) for c in cells) + " |"
        )

    # H3 verdict: does selective_tpsr degrade more slowly than pretrained_beam?
    lines += ["", "## H3 check: robustness slope", ""]
    if len(noises) >= 2:
        lo, hi = noises[0], noises[-1]
        deg_stpsr = cell(hi, "selective_tpsr", "penalized_nmse") - cell(lo, "selective_tpsr", "penalized_nmse")
        deg_sbeam = cell(hi, "selective_beam", "penalized_nmse") - cell(lo, "selective_beam", "penalized_nmse")
        deg_pbeam = cell(hi, "pretrained_beam", "penalized_nmse") - cell(lo, "pretrained_beam", "penalized_nmse")
        lines += [
            f"- ΔNMSE({lo}→{hi}) selective_tpsr = {fmt(deg_stpsr)}",
            f"- ΔNMSE({lo}→{hi}) selective_beam = {fmt(deg_sbeam)}",
            f"- ΔNMSE({lo}→{hi}) pretrained_beam = {fmt(deg_pbeam)}",
            "",
            (
                f"**Verdict:** selective+TPSR degrades "
                f"{'more slowly (supports H3)' if (np.isfinite(deg_stpsr) and np.isfinite(deg_sbeam) and deg_stpsr < deg_sbeam) else 'NOT more slowly (H3 unsupported)'}"
                " than selective+beam under increasing noise."
            ),
        ]
    lines += [
        "",
        "> ⚠️ Report accuracy **and** complexity together (plan H3 is about the "
        "tradeoff). Overlapping performance across cells at a given noise level is "
        "not evidence for TPSR; check per-seed spread as in phase4_multiseed.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {OUT_DIR / 'noise_sweep.json'}")
    log(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

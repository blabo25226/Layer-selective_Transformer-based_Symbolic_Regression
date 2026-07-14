"""Issues 4–5: freeze selected layers and verify trainable parameter counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import omegaconf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from nesymres.architectures.model import Model  # noqa: E402
from models.layer_selector import (  # noqa: E402
    assert_only_layers_trainable,
    count_trainable_parameters,
    freeze_all,
    get_layer_registry,
    set_trainable_layers,
    unfreeze_all,
)

WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
OUT = ROOT / "results" / "phase_results" / "issue4_5_freeze_check.json"
REPORT = ROOT / "results" / "phase_results" / "issue4_5_report.md"


def _dummy_forward_backward(model: Model) -> None:
    """Ensure gradients flow with only selected layers trainable."""
    model.train()
    batch_size, n_pts, dim_in = 2, 32, 4
    # encoder expects (B, N, dim_input) via forward's permute path:
    # batch[0] shape used in model.forward as (B, dim, N) then permuted.
    # Dim must be architecture.dim_input (=4 for point features + y).
    x = torch.randn(batch_size, dim_in, n_pts)
    # tokens: pad=0; length must fit output_dim / length_eq. Simple random ids.
    trg_len = 8
    trg = torch.randint(1, min(10, model.cfg.output_dim), (batch_size, trg_len))
    trg[:, 0] = 1  # start-ish
    batch = [x, trg]
    output, trg_out = model.forward(batch)
    loss = model.compute_loss(output, trg_out)
    loss.backward()


def main() -> int:
    if not WEIGHTS.exists() or not CONFIG.exists():
        print(f"Missing checkpoint or config:\n  {WEIGHTS}\n  {CONFIG}")
        return 1

    cfg = omegaconf.OmegaConf.load(CONFIG)
    model = Model.load_from_checkpoint(str(WEIGHTS), cfg=cfg.architecture)
    registry = get_layer_registry(model)

    unfreeze_all(model)
    _, total = count_trainable_parameters(model)

    cases = {
        "all_frozen": [],
        "output_head": ["output_head"],
        "encoder_0": ["encoder_0"],
        "decoder_0": ["decoder_0"],
        "encoder_0+decoder_0": ["encoder_0", "decoder_0"],
        "all_encoder_blocks": [n for n in registry if n.startswith("encoder_") and n != "encoder_pma"],
        "all_decoder_blocks": [n for n in registry if n.startswith("decoder_")],
    }

    results = {"total_parameters": total, "cases": {}}
    for case_name, layers in cases.items():
        if case_name == "all_frozen":
            freeze_all(model)
        else:
            set_trainable_layers(model, layers, freeze_others=True)
            assert_only_layers_trainable(model, layers)

        trainable, _ = count_trainable_parameters(model)
        entry = {
            "layers": layers,
            "trainable": trainable,
            "fraction": trainable / total if total else 0.0,
        }

        # Gradient smoke check for non-empty selections
        if layers:
            model.zero_grad(set_to_none=True)
            _dummy_forward_backward(model)
            grads = []
            for name in layers:
                mod = registry[name].resolve(model)
                grads.append(
                    any(
                        p.grad is not None and torch.any(p.grad != 0)
                        for p in mod.parameters()
                        if p.requires_grad
                    )
                )
            entry["grad_nonzero_on_selected"] = all(grads)
            if not entry["grad_nonzero_on_selected"]:
                print(f"WARNING: no nonzero grad for case {case_name}")
        results["cases"][case_name] = entry
        print(
            f"{case_name:24s} trainable={trainable:>10,}  "
            f"({100 * entry['fraction']:.2f}%)  layers={layers}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Issues 4–5: freeze and trainable checks",
        "",
        f"- Total parameters: {total:,}",
        "",
        "| case | layers | trainable | % |",
        "|------|--------|-----------|---|",
    ]
    for name, entry in results["cases"].items():
        layer_str = ", ".join(f"`{x}`" for x in entry["layers"]) or "(none)"
        lines.append(
            f"| `{name}` | {layer_str} | {entry['trainable']:,} | {100 * entry['fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "- `requires_grad=False` on non-selected layers (Issue 4)",
            "- Trainable count matches selected modules only (Issue 5)",
            "- Dummy forward/backward produces grads on selected layers",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

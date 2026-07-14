"""Issue 3: list NeSymReS encoder/decoder layers and write a report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import omegaconf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "NSRS" / "src"))

from nesymres.architectures.model import Model  # noqa: E402
from models.layer_selector import (  # noqa: E402
    count_trainable_parameters,
    list_layers,
)

WEIGHTS = ROOT / "NSRS" / "weights" / "10M.ckpt"
CONFIG = ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"
OUT = ROOT / "results" / "phase_results" / "issue3_layer_list.json"
REPORT = ROOT / "results" / "phase_results" / "issue3_report.md"


def main() -> int:
    if not WEIGHTS.exists() or not CONFIG.exists():
        print(f"Missing checkpoint or config:\n  {WEIGHTS}\n  {CONFIG}")
        return 1

    cfg = omegaconf.OmegaConf.load(CONFIG)
    print(f"Loading {WEIGHTS}")
    model = Model.load_from_checkpoint(str(WEIGHTS), cfg=cfg.architecture)
    model.eval()

    rows = list_layers(model)
    trainable, total = count_trainable_parameters(model)

    payload = {
        "checkpoint": str(WEIGHTS),
        "config": str(CONFIG),
        "architecture": {
            "n_l_enc": int(cfg.architecture.n_l_enc),
            "dec_layers": int(cfg.architecture.dec_layers),
            "dim_hidden": int(cfg.architecture.dim_hidden),
        },
        "total_parameters": total,
        "trainable_before_freeze": trainable,
        "layers": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    enc = [r for r in rows if r["kind"] == "encoder"]
    dec = [r for r in rows if r["kind"] == "decoder"]
    lines = [
        "# Issue 3: NeSymReS layer list",
        "",
        f"- Checkpoint: `{WEIGHTS.name}`",
        f"- Config: `{CONFIG.as_posix()}` (100M architecture)",
        f"- Total parameters: {total:,}",
        f"- Encoder units: {len(enc)}",
        f"- Decoder units: {len(dec)}",
        "",
        "| name | kind | module_path | n_params |",
        "|------|------|-------------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['kind']} | `{r['module_path']}` | {r['n_params']:,} |"
        )
    lines.extend(
        [
            "",
            "## Layer naming",
            "",
            "- `encoder_0` = first ISAB (`enc.selfatt1`)",
            "- `encoder_1..N` = remaining ISABs (`enc.selfatt.*`)",
            "- `encoder_pma` = PMA pooling (`enc.outatt`)",
            "- `decoder_i` = `nn.TransformerDecoderLayer` i",
            "- `output_head` = `fc_out`",
            "",
            "Next: Issue 4–5 freeze / trainable checks.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT}")
    print(f"Wrote {REPORT}")
    print(f"Layers: {len(rows)} | total params: {total:,}")
    for r in rows:
        print(f"  {r['name']:16s} {r['kind']:10s} {r['n_params']:>10,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

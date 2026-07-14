"""Fail-fast validation for the GPU experiment environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eq-setting", type=Path, required=True)
    args = parser.parse_args()
    for label, path in (("weights", args.weights), ("config", args.config), ("eq-setting", args.eq_setting)):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    with args.eq_setting.open(encoding="utf-8") as handle:
        setting = json.load(handle)
    if "word2id" not in setting:
        parser.error("eq-setting does not contain word2id")
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment")
    print(f"CUDA OK: torch={torch.__version__}, runtime={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {args.weights.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

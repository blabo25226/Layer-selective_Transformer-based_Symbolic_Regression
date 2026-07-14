"""Generate Phase 7 DREAM-like multi-gene GRN dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dreamlike_grn import generate_dreamlike_dataset  # noqa: E402

OUT = ROOT / "results" / "synthetic" / "phase7_dreamlike_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-genes", type=int, default=10)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    meta = generate_dreamlike_dataset(
        args.out,
        n_genes=args.n_genes,
        n_train_points=args.n_train,
        n_test_points=args.n_test,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    print(f"Wrote {args.out / 'network.json'}")
    print(f"Wrote {args.out / 'expression.npz'}")
    print(f"n_genes={meta['n_genes']} n_edges={meta['n_edges']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

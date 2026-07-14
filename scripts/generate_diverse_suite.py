"""Generate the diverse, structure-split synthetic suite (reviewer note A-1).

Unlike Phase 1 (4 skeleton families), this builds many distinct functional forms
with disjoint TRAIN/TEST skeletons and multiple random parameterizations each, so
per-layer contribution can be estimated with a meaningful sample. Optionally emit
several noise levels for the H3 / noise-robustness experiments (reviewer A-4).

Examples
--------
    python scripts/generate_diverse_suite.py --n-per-skeleton 8
    python scripts/generate_diverse_suite.py --noise 0.0 0.05 0.1 --out-root results/synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic_grn import build_diverse_suite, save_suite  # noqa: E402


def _write_report(out_dir: Path, index: list, noise_std: float, seed: int) -> None:
    by_split = Counter(item["split"] for item in index)
    train_sk = sorted({i["eq_id"].rsplit("_train_", 1)[0] for i in index if i["split"] == "train"})
    test_sk = sorted({i["eq_id"].rsplit("_test_", 1)[0] for i in index if i["split"] == "test"})
    report = out_dir / "suite_report.md"
    report.write_text(
        "\n".join(
            [
                "# Diverse synthetic suite (A-1)",
                "",
                f"- Output: `{out_dir.as_posix()}`",
                f"- Problems: {len(index)}  |  splits: {dict(by_split)}",
                f"- noise_std: {noise_std}  |  seed: {seed}",
                "",
                "## Structure split (disjoint skeletons)",
                "",
                f"- **train** ({len(train_sk)}): {', '.join(train_sk)}",
                f"- **test** ({len(test_sk)}): {', '.join(test_sk)}",
                "",
                "> TEST skeletons are functional forms absent from TRAIN, so decode "
                "metrics measure generalization to unseen structure, not memorization.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-skeleton", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=200)
    parser.add_argument("--support", type=float, nargs=2, default=(0.1, 3.0))
    parser.add_argument(
        "--noise",
        type=float,
        nargs="+",
        default=[0.0],
        help="One or more relative noise levels; each gets its own suite dir.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", default=str(ROOT / "results" / "synthetic"))
    parser.add_argument(
        "--tag",
        default="diverse_v1",
        help="Base dir name; noise>0 appends _n{level}.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    for noise in args.noise:
        name = args.tag if noise == 0.0 else f"{args.tag}_n{noise}"
        out_dir = out_root / name
        datasets = build_diverse_suite(
            n_per_skeleton=args.n_per_skeleton,
            n_points=args.n_points,
            support=tuple(args.support),
            noise_std=noise,
            seed=args.seed,
        )
        index_path = save_suite(datasets, out_dir)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        _write_report(out_dir, index, noise, args.seed)
        print(f"noise={noise}: wrote {len(index)} problems -> {out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

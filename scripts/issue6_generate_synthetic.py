"""Generate Phase 1 synthetic GRN / Hill equation suite (Issue 6)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic_grn import build_phase1_suite, save_suite  # noqa: E402

OUT_DIR = ROOT / "results" / "synthetic" / "phase1_v1"
REPORT = ROOT / "results" / "phase_results" / "phase1_report.md"


def main() -> int:
    datasets = build_phase1_suite(
        n_points=200,
        support=(0.0, 3.0),
        noise_std=0.0,
        seed=0,
    )
    index_path = save_suite(datasets, OUT_DIR)
    index = json.loads(index_path.read_text(encoding="utf-8"))

    by_split = Counter(item["split"] for item in index)
    by_family = Counter(item["family"] for item in index)

    lines = [
        "# Phase 1 / Issue 6: synthetic GRN data",
        "",
        f"- Output: `{OUT_DIR.as_posix()}`",
        f"- Problems: {len(index)}",
        f"- Splits: {dict(by_split)}",
        f"- Families: {dict(by_family)}",
        "",
        "## Design",
        "",
        "- Target: learn ODE right-hand side `y = f(X)` (not full time trajectories yet).",
        "- Families: activation Hill, repression Hill, toggle switch, repressilator.",
        "- Split: parameter-range split within family (train/test use different coeffs).",
        "- Support: `x_i ~ Uniform(0, 3)`, 200 points/problem, no noise in v1.",
        "",
        "## Index (first 12)",
        "",
        "| eq_id | family | split | n_vars | target_expr |",
        "|-------|--------|-------|--------|-------------|",
    ]
    for item in index[:12]:
        lines.append(
            f"| `{item['eq_id']}` | {item['family']} | {item['split']} | "
            f"{item['n_vars']} | `{item['target_expr']}` |"
        )
    lines.extend(
        [
            "",
            f"Full index: `{index_path.as_posix()}`",
            "",
            "## Next",
            "",
            "- Issue 7: PySR baseline on train problems",
            "- Phase 2: NeSymReS / TPSR baselines on the same suite",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(index)} problems to {OUT_DIR}")
    print(f"Wrote {REPORT}")
    print("splits:", dict(by_split))
    print("families:", dict(by_family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

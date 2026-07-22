"""Export lightweight, Git-trackable evidence from an ignored raw run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_manifest import record_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/published"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run_id = run.name
    destination = args.out_root.resolve() / run_id
    if destination.exists():
        parser.error(f"publication directory already exists: {destination}")
    validation = run / "validation.json"
    if not validation.is_file():
        parser.error("validate the run with scripts/validate_gpu_run.py first")
    validation_status = json.loads(validation.read_text(encoding="utf-8")).get("status")
    if validation_status != "validated":
        record_stage(run / "manifest.json", "publication", "failed")
        parser.error(f"run did not pass validation: status={validation_status}")
    destination.mkdir(parents=True)
    for name in ("manifest.json", "validation.json"):
        shutil.copy2(run / name, destination / name)
    reports_out = destination / "reports"
    reports_out.mkdir()
    for report in sorted((run / "reports").glob("*.md")):
        shutil.copy2(report, reports_out / report.name)
    summaries = {
        "phase4_contributions.json": run / "phase4_multiseed" / "contrib_aggregate.json",
        "phase4_absolute_improvements.json": run / "phase4_multiseed" / "absolute_improvements_aggregate.json",
        "phase4_ranking_scores.json": run / "phase4_multiseed" / "layer_ranking_scores.json",
        "phase4_ranking_metadata.json": run / "phase4_multiseed" / "layer_ranking_metadata.json",
        "phase4_rankings.json": run / "phase4_multiseed" / "layer_rankings.json",
        "phase4_importance_evidence.json": run / "phase4_multiseed" / "layer_importance_evidence.json",
        "phase4_ranking_stability.json": run / "phase4_multiseed" / "ranking_stability.json",
        "phase4_status.json": run / "phase4_multiseed" / "contribution_status_aggregate.json",
        "phase5_summary.json": run / "phase5_multiseed" / "summary.json",
        "phase6_summary.json": run / "phase6_noise_multiseed" / "summary.json",
        "phase7_summary.json": run / "phase7_multiseed" / "summary.json",
        "phase8_summary.json": run / "phase8_lodo_multiseed" / "summary.json",
    }
    copied = []
    for name, source in summaries.items():
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    archive_lines = []
    if args.archive:
        archive = args.archive.resolve()
        checksum_path = Path(str(archive) + ".sha256")
        checksum = (
            checksum_path.read_text(encoding="utf-8").split()[0]
            if checksum_path.is_file() else "not recorded"
        )
        archive_lines = [
            f"- Raw archive filename: `{archive.name}`",
            f"- Raw archive SHA256: `{checksum}`",
        ]
    lines = [
        f"# Published GPU run: {run_id}", "",
        f"- Status: `{manifest.get('status')}`",
        f"- Validation: `{validation_status}`",
        f"- Git commit: `{manifest.get('git', {}).get('commit')}`",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- Checkpoint SHA256: `{manifest.get('checkpoint', {}).get('sha256')}`",
        *archive_lines,
        "",
        "This directory contains lightweight reports and aggregate evidence suitable for Git.",
        "Raw logs, generated datasets, and complete per-problem records remain in the archived raw run.",
        "",
        "## Files", "",
    ]
    lines.extend(f"- `{name}`" for name in ["manifest.json", "validation.json", *copied])
    lines.append("- `reports/`")
    (destination / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_stage(run / "manifest.json", "publication", "complete")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

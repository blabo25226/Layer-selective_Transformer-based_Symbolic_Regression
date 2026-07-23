"""Fail-fast validation of a completed GPU run before publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_manifest import record_stage  # noqa: E402

# The pipeline sets status=complete before the run is checked, so a validation
# failure here is the only signal that the run must not be published.
RESUMABLE_STATUSES = {"complete", "validation_failed", "publication_failed"}

REQUIRED_EQUATION_FIELDS = {
    "eq_id",
    "true_expr",
    "pred",
    "pred_raw",
    "pred_simplified",
    "candidate_expressions",
    "variable_names",
    "variable_mapping",
    "decoder",
    "failure_reason",
    "valid_pred",
}


def validate_equation_row(row: object, location: str) -> list[str]:
    if not isinstance(row, dict):
        return [f"{location}: equation row is not an object"]
    issues = []
    missing = sorted(REQUIRED_EQUATION_FIELDS - set(row))
    if missing:
        issues.append(f"{location}: missing fields {missing}")
        return issues
    if not isinstance(row["eq_id"], str) or not row["eq_id"]:
        issues.append(f"{location}: invalid eq_id")
    for key in ("pred", "pred_raw", "pred_simplified"):
        if not isinstance(row[key], str):
            issues.append(f"{location}: {key} is not a string")
    if not isinstance(row["candidate_expressions"], list):
        issues.append(f"{location}: candidate_expressions is not a list")
    names = row["variable_names"]
    mapping = row["variable_mapping"]
    if not isinstance(names, list) or not names:
        issues.append(f"{location}: variable_names is empty or invalid")
    if not isinstance(mapping, list) or not isinstance(names, list) or len(mapping) != len(names):
        issues.append(f"{location}: variable_mapping does not match variable_names")
    elif any(
        not isinstance(item, dict)
        or not {"symbol", "input_position", "source_index", "source_name"}.issubset(item)
        for item in mapping
    ):
        issues.append(f"{location}: variable_mapping contains an invalid entry")
    if not isinstance(row["decoder"], str) or not row["decoder"]:
        issues.append(f"{location}: decoder is empty or invalid")
    valid = bool(row["valid_pred"])
    if valid and (not row["pred_raw"] or not row["pred_simplified"]):
        issues.append(f"{location}: valid prediction has no saved expression")
    if not valid and not row["failure_reason"]:
        issues.append(f"{location}: failed prediction has no failure_reason")
    return issues


def equation_lists(obj: object, prefix: str = "") -> list[tuple[str, list]]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            location = f"{prefix}.{key}" if prefix else key
            if key in {"per_problem", "in_per_problem", "hold_per_problem"} and isinstance(value, list):
                found.append((location, value))
            else:
                found.extend(equation_lists(value, location))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(equation_lists(value, f"{prefix}[{index}]"))
    return found


def count_per_problem(obj) -> tuple[int, int]:
    groups = rows = 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"per_problem", "in_per_problem", "hold_per_problem"} and isinstance(value, list):
                groups += 1
                rows += len(value)
            else:
                g, r = count_per_problem(value)
                groups += g
                rows += r
    elif isinstance(obj, list):
        for value in obj:
            g, r = count_per_problem(value)
            groups += g
            rows += r
    return groups, rows


def fail(run: Path, errors: list[str]) -> int:
    """Persist the failure so neither the manifest nor validation.json look healthy."""
    (run / "validation.json").write_text(
        json.dumps({"status": "failed", "run_dir": str(run), "errors": errors}, indent=2),
        encoding="utf-8",
    )
    record_stage(run / "manifest.json", "validation", "failed")
    print("VALIDATION FAILED:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in RESUMABLE_STATUSES:
        parser.error(f"pipeline did not finish: status={manifest.get('status')}")
    dream4 = manifest.get("parameters", {}).get("LTSR_DREAM4") == "1"
    required = [
        run / "phase4_multiseed" / "contrib_aggregate.json",
        run / "phase4_multiseed" / "absolute_improvements_aggregate.json",
        run / "phase4_multiseed" / "contribution_status_aggregate.json",
        run / "phase4_multiseed" / "layer_ranking_scores.json",
        run / "phase4_multiseed" / "layer_ranking_metadata.json",
        run / "phase4_multiseed" / "layer_rankings.json",
        run / "phase4_multiseed" / "layer_importance_evidence.json",
        run / "phase4_multiseed" / "ranking_stability.json",
        run / "phase5_multiseed" / "summary.json",
        run / "phase6_noise_multiseed" / "summary.json",
        run / "phase8_lodo_multiseed" / "summary.json",
    ]
    if dream4:
        required.append(run / "phase7_multiseed" / "summary.json")
    seeds = manifest.get("parameters", {}).get("LTSR_SEEDS", "").split()
    for seed in seeds:
        required.extend([
            run / "phase4_multiseed" / f"equations_seed{seed}.json",
            run / "phase4_multiseed" / f"raw_scores_seed{seed}.json",
            run / "phase4_multiseed" / f"absolute_improvements_seed{seed}.json",
            run / "phase4_multiseed" / f"contribution_status_seed{seed}.json",
            run / "phase4_multiseed" / f"tuning_seed{seed}.json",
            run / f"phase5_seed{seed}" / "selective_results.json",
            run / f"phase6_noise_seed{seed}" / "noise_sweep.json",
            run / f"phase8_lodo_seed{seed}" / "lodo_results.json",
        ])
        if dream4:
            required.extend([
                run / f"phase7_dream4_size10_seed{seed}" / "size10_results.json",
                run / f"phase7_dream4_size100_seed{seed}" / "size100_results.json",
            ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return fail(run, ["missing required output: " + path for path in missing])
    json_files = [p for p in run.rglob("*.json") if p.name not in {"manifest.json", "validation.json"}]
    groups = rows = 0
    invalid_json = []
    equation_issues = []
    for path in json_files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json.append(str(path))
            continue
        g, r = count_per_problem(obj)
        groups += g
        rows += r
        for location, records in equation_lists(obj):
            for index, record in enumerate(records):
                equation_issues.extend(
                    validate_equation_row(record, f"{path.relative_to(run)}:{location}[{index}]")
                )
        if path.name.startswith("equations_seed") and isinstance(obj, dict):
            for condition, records in obj.items():
                if not isinstance(records, list):
                    equation_issues.append(
                        f"{path.relative_to(run)}:{condition} is not a list"
                    )
                    continue
                for index, record in enumerate(records):
                    equation_issues.extend(
                        validate_equation_row(
                            record, f"{path.relative_to(run)}:{condition}[{index}]"
                        )
                    )
    errors = ["invalid JSON output: " + path for path in invalid_json]
    if groups == 0 or rows == 0:
        errors.append("no per-problem equation records found")
    errors.extend("invalid equation record: " + issue for issue in equation_issues[:20])
    if errors:
        return fail(run, errors)
    result = {
        "status": "validated",
        "run_dir": str(run),
        "required_outputs": [str(p.relative_to(run)) for p in required],
        "json_files": len(json_files),
        "per_problem_groups": groups,
        "per_problem_rows": rows,
        "equation_schema": "v1",
    }
    (run / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    record_stage(run / "manifest.json", "validation", "complete")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

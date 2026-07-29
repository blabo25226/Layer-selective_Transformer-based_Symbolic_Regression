"""Run the Phase 8 PySR LODO baseline locally without importing NeSymReS.

This entry point is intentionally Python-3.12 compatible: the repository's
Hydra-pinned NeSymReS environment remains Python 3.10-only, while PySR and the
human-data/evaluation helpers do not require Hydra.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dreamlike_grn import build_local_problem  # noqa: E402
from data.human import (  # noqa: E402
    build_human_local_problems,
    estimate_derivatives,
    prepare_gse112372,
)
from evaluation.aggregation import aggregate_prediction_scores  # noqa: E402
from evaluation.equation_metrics import score_prediction  # noqa: E402
from evaluation.equation_records import (  # noqa: E402
    dataset_variable_mapping,
    make_equation_record,
)
from evaluation.generalization import aggregate_lodo  # noqa: E402


def stack_donors(panel, donors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for donor in donors:
        X, Y = estimate_derivatives(
            panel.times, panel.X_donors[donor], method="smooth_fd"
        )
        Xs.append(X)
        Ys.append(Y)
    return np.vstack(Xs), np.vstack(Ys)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, default=lambda value: None),
        encoding="utf-8",
    )
    os.replace(partial, path)


class TargetWorker:
    """Keep Julia warm while enforcing a parent-side per-target deadline."""

    def __init__(
        self,
        work_dir: Path,
        *,
        target_timeout_sec: float,
        startup_timeout_sec: float,
    ) -> None:
        self.work_dir = work_dir
        self.target_timeout_sec = target_timeout_sec
        self.startup_timeout_sec = startup_timeout_sec
        self.process: subprocess.Popen[str] | None = None
        self.fresh = True
        self.sequence = 0

    def _start(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        )
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "phase8_pysr_target_worker.py")],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        self.fresh = True

    def _stop(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process = process
            self._stop()

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            self._start()
        assert self.process is not None
        self.sequence += 1
        request_path = self.work_dir / f"request_{self.sequence:04d}.json"
        response_path = self.work_dir / f"response_{self.sequence:04d}.json"
        atomic_json(request_path, payload)
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(
                json.dumps(
                    {
                        "request_path": str(request_path),
                        "response_path": str(response_path),
                    }
                )
                + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._stop()
            request_path.unlink(missing_ok=True)
            return {
                "ok": False,
                "error": f"target worker communication failed: {exc}",
            }
        timeout = (
            self.startup_timeout_sec if self.fresh else self.target_timeout_sec
        )
        self.fresh = False
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if response_path.is_file():
                    return json.loads(response_path.read_text(encoding="utf-8"))
                code = self.process.poll()
                if code is not None:
                    return {
                        "ok": False,
                        "error": f"target worker exited with status {code}",
                    }
                time.sleep(0.2)
            self._stop()
            return {
                "ok": False,
                "timeout": True,
                "error": f"target_process_timeout_after_{timeout:g}s",
            }
        finally:
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)


def empty_failure_record(
    ds,
    panel,
    *,
    failure_reason: str,
    niterations: int,
    random_state: int,
    search_timeout_sec: float,
    target_name: str | None = None,
) -> dict[str, Any]:
    scores = score_prediction(
        ds.y,
        None,
        "",
        ds.spec.variable_names,
        true_expr="",
    )
    extra = {}
    if target_name is not None:
        extra = {"target": target_name, "reused_training_decode": True}
    return make_equation_record(
        eq_id=ds.spec.eq_id,
        predicted_expr="",
        variable_names=ds.spec.variable_names,
        mapping=dataset_variable_mapping(ds, panel.gene_names),
        scores=scores,
        decoder="pysr",
        failure_reason=failure_reason,
        decoder_metadata={
            "niterations": niterations,
            "random_state": random_state,
            "execution": "local_cpu",
            "search_timeout_sec": search_timeout_sec,
            "target_process_failed": True,
        },
        **extra,
    )


def run_fold(
    panel,
    *,
    holdout: str,
    seed: int,
    fold_index: int,
    niterations: int,
    search_timeout_sec: float,
    worker: TargetWorker,
    target_checkpoint: Path,
    target_identity: dict[str, Any],
) -> dict[str, Any]:
    train_donors = [donor for donor in sorted(panel.X_donors) if donor != holdout]
    X_tr, Y_tr = stack_donors(panel, train_donors)
    X_te, Y_te = stack_donors(panel, [holdout])
    problems, selections, _ = build_human_local_problems(
        panel,
        X_tr,
        Y_tr,
        method="prior",
        k=2,
        max_vars=3,
        split="train",
    )
    network = panel.as_grn_like()
    if target_checkpoint.is_file():
        target_state = json.loads(target_checkpoint.read_text(encoding="utf-8"))
        if target_state.get("identity") != target_identity:
            raise RuntimeError(
                f"target checkpoint configuration mismatch: {target_checkpoint}"
            )
    else:
        target_state = {"identity": target_identity, "folds": {}}
    fold_state = target_state["folds"].setdefault(
        str(holdout), {"in_rows": [], "hold_rows": []}
    )
    in_rows = list(fold_state["in_rows"])
    hold_rows = list(fold_state["hold_rows"])
    if len(in_rows) != len(hold_rows) or len(in_rows) > len(problems):
        raise RuntimeError(f"invalid target checkpoint: {target_checkpoint}")

    for target_index, ds in enumerate(problems):
        if target_index < len(in_rows):
            if in_rows[target_index].get("eq_id") != ds.spec.eq_id:
                raise RuntimeError(
                    f"target checkpoint order mismatch: {target_checkpoint}"
                )
            print(
                f"seed={seed} holdout={holdout} "
                f"target={target_index + 1}/{len(problems)}: checkpoint complete",
                flush=True,
            )
            continue
        random_state = seed * 10000 + fold_index * 100 + target_index
        motif = ds.spec.motif or ""
        target_name = (
            motif.split("target=")[-1].split(";")[0]
            if "target=" in motif
            else ""
        )
        target = panel.gene_index(target_name)
        heldout = build_local_problem(
            network,
            X_te,
            Y_te[:, target],
            target,
            selections.get(target, []),
            eq_id=f"{ds.spec.eq_id}_holdout",
            split="holdout",
            include_target=True,
            max_vars=3,
            selection_method="prior",
        )
        payload = {
            "niterations": niterations,
            "random_state": random_state,
            "search_timeout_sec": search_timeout_sec,
            "target_timeout_sec": worker.target_timeout_sec,
            "worker_startup_timeout_sec": worker.startup_timeout_sec,
            "target_name": target_name,
            "train": {
                "eq_id": ds.spec.eq_id,
                "X": ds.X.tolist(),
                "y": ds.y.tolist(),
                "variable_names": list(ds.spec.variable_names),
                "mapping": dataset_variable_mapping(ds, panel.gene_names),
            },
            "holdout": {
                "eq_id": heldout.spec.eq_id,
                "X": heldout.X.tolist(),
                "y": heldout.y.tolist(),
                "variable_names": list(heldout.spec.variable_names),
                "mapping": dataset_variable_mapping(heldout, panel.gene_names),
            },
        }
        started = time.monotonic()
        response = worker.run(payload)
        elapsed = time.monotonic() - started
        if response.get("ok"):
            in_record = response["in_record"]
            hold_record = response["hold_record"]
            status = (
                "valid"
                if hold_record.get("valid_pred", 0.0)
                else "evaluation_failed"
            )
        else:
            reason = str(response.get("error", "target_worker_failed"))
            in_record = empty_failure_record(
                ds,
                panel,
                failure_reason=reason,
                niterations=niterations,
                random_state=random_state,
                search_timeout_sec=search_timeout_sec,
            )
            hold_record = empty_failure_record(
                heldout,
                panel,
                failure_reason=reason,
                niterations=niterations,
                random_state=random_state,
                search_timeout_sec=search_timeout_sec,
                target_name=target_name,
            )
            status = "timeout" if response.get("timeout") else "worker_failed"
        in_rows.append(in_record)
        hold_rows.append(hold_record)
        fold_state["in_rows"] = in_rows
        fold_state["hold_rows"] = hold_rows
        atomic_json(target_checkpoint, target_state)
        print(
            f"seed={seed} holdout={holdout} "
            f"target={target_index + 1}/{len(problems)} "
            f"status={status} elapsed={elapsed:.1f}s",
            flush=True,
        )

    in_aggregate = aggregate_prediction_scores(in_rows)
    hold_aggregate = aggregate_prediction_scores(hold_rows)
    return {
        "in": in_aggregate["penalized_nmse"],
        "hold": hold_aggregate["penalized_nmse"],
        "in_valid_rate": in_aggregate["valid_rate"],
        "hold_valid_rate": hold_aggregate["valid_rate"],
        "hold_near_singularity": hold_aggregate.get(
            "near_singularity_mean", float("nan")
        ),
        "hold_extrapolation_valid": hold_aggregate.get(
            "extrapolation_valid_mean", float("nan")
        ),
        "in_aggregate": in_aggregate,
        "hold_aggregate": hold_aggregate,
        "in_per_problem": in_rows,
        "hold_per_problem": hold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "human" / "gse112372_lps",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--pysr-iters", type=int, default=12)
    parser.add_argument("--search-timeout-sec", type=float, default=15.0)
    parser.add_argument("--target-timeout-sec", type=float, default=30.0)
    parser.add_argument("--worker-startup-timeout-sec", type=float, default=90.0)
    args = parser.parse_args()

    panel = prepare_gse112372(args.data_dir)
    donors = sorted(panel.X_donors)
    worker_dir = args.run_dir / f".phase8_pysr_worker_{os.getpid()}"
    worker = TargetWorker(
        worker_dir,
        target_timeout_sec=args.target_timeout_sec,
        startup_timeout_sec=args.worker_startup_timeout_sec,
    )
    try:
        for seed in args.seeds:
            out_dir = args.run_dir / f"phase8_pysr_seed{seed}"
            checkpoint = out_dir / "fold_checkpoint.json"
            target_checkpoint = out_dir / "target_checkpoint.json"
            identity = {
                "schema_version": 1,
                "seed": seed,
                "donors": donors,
                "pysr_iterations": args.pysr_iters,
                "execution": "local_cpu",
            }
            target_identity = {
                **identity,
                "schema_version": 2,
                "search_timeout_sec": args.search_timeout_sec,
                "target_timeout_sec": args.target_timeout_sec,
                "worker_startup_timeout_sec": args.worker_startup_timeout_sec,
            }
            if checkpoint.is_file():
                state = json.loads(checkpoint.read_text(encoding="utf-8"))
                if state.get("identity") != identity:
                    raise RuntimeError(
                        f"checkpoint configuration mismatch: {checkpoint}"
                    )
                details = state.get("per_fold", [])
            else:
                details = []
            completed = {str(row["holdout"]) for row in details}
            for fold_index, holdout in enumerate(donors):
                if holdout in completed:
                    print(f"seed={seed} holdout={holdout}: checkpoint complete")
                    continue
                print(f"seed={seed} holdout={holdout}: starting", flush=True)
                result = run_fold(
                    panel,
                    holdout=holdout,
                    seed=seed,
                    fold_index=fold_index,
                    niterations=args.pysr_iters,
                    search_timeout_sec=args.search_timeout_sec,
                    worker=worker,
                    target_checkpoint=target_checkpoint,
                    target_identity=target_identity,
                )
                details.append(
                    {
                        "holdout": holdout,
                        "n_targets": len(result["in_per_problem"]),
                        "search_timeout_sec": args.search_timeout_sec,
                        "target_timeout_sec": args.target_timeout_sec,
                        "worker_startup_timeout_sec": args.worker_startup_timeout_sec,
                        "pysr": result,
                    }
                )
                atomic_json(
                    checkpoint, {"identity": identity, "per_fold": details}
                )

            folds = [{"pysr": row["pysr"]} for row in details]
            output = {
                "seed": seed,
                "execution": "local_cpu",
                "pysr_iterations": args.pysr_iters,
                "search_timeout_sec": args.search_timeout_sec,
                "target_timeout_sec": args.target_timeout_sec,
                "worker_startup_timeout_sec": args.worker_startup_timeout_sec,
                "legacy_completed_folds_without_timeout": [
                    str(row["holdout"])
                    for row in details
                    if "search_timeout_sec" not in row
                ],
                "per_fold": details,
                "aggregate": aggregate_lodo(
                    folds, metric="nmse", lower_better=True
                ),
            }
            atomic_json(out_dir / "pysr_results.json", output)
            print(f"seed={seed}: wrote {out_dir / 'pysr_results.json'}")
    finally:
        worker.close()
        shutil.rmtree(worker_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

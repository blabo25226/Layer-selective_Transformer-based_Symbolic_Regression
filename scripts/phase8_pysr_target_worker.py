"""Persistent worker for one bounded Phase 8 PySR target at a time.

The parent process communicates through JSON request/response files so PySR
and Julia may write freely to stdout. Keeping this worker alive amortizes Julia
startup; the parent can still kill the complete process tree if one target
hangs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.pysr_runner import fit_pysr_expression  # noqa: E402
from evaluation.equation_metrics import eval_expression, score_prediction  # noqa: E402
from evaluation.equation_records import make_equation_record  # noqa: E402


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)


def process_target(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    train = payload["train"]
    holdout = payload["holdout"]
    variable_names = train["variable_names"]
    random_state = int(payload["random_state"])
    failure_reason = None
    try:
        expr = fit_pysr_expression(
            np.asarray(train["X"], dtype=float),
            np.asarray(train["y"], dtype=float),
            variable_names,
            niterations=int(payload["niterations"]),
            random_state=random_state,
            timeout_in_seconds=float(payload["search_timeout_sec"]),
        )
    except Exception as exc:
        expr = ""
        failure_reason = f"{type(exc).__name__}: {exc}"

    metadata = {
        "niterations": int(payload["niterations"]),
        "random_state": random_state,
        "execution": "local_cpu",
        "search_timeout_sec": float(payload["search_timeout_sec"]),
        "target_timeout_sec": float(payload["target_timeout_sec"]),
        "worker_startup_timeout_sec": float(payload["worker_startup_timeout_sec"]),
        "parallelism": "multithreading",
    }

    def make_record(part: dict[str, Any], *, reused: bool) -> dict[str, Any]:
        X = np.asarray(part["X"], dtype=float)
        y = np.asarray(part["y"], dtype=float)
        names = part["variable_names"]
        y_hat = eval_expression(expr, X, names) if expr else None
        scores = score_prediction(
            y,
            y_hat,
            expr,
            names,
            true_expr="",
            X=X,
            variable_names=names,
        )
        extra = {}
        if reused:
            extra = {
                "target": payload["target_name"],
                "reused_training_decode": True,
            }
        return make_equation_record(
            eq_id=part["eq_id"],
            predicted_expr=expr,
            variable_names=names,
            mapping=part["mapping"],
            scores=scores,
            decoder="pysr",
            failure_reason=failure_reason,
            decoder_metadata=metadata,
            **extra,
        )

    return {
        "ok": True,
        "in_record": make_record(train, reused=False),
        "hold_record": make_record(holdout, reused=True),
        "elapsed_sec": time.monotonic() - started,
    }


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        command = json.loads(line)
        request_path = Path(command["request_path"])
        response_path = Path(command["response_path"])
        try:
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            response = process_target(payload)
        except BaseException as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_json(response_path, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

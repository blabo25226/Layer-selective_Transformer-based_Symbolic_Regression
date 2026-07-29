from __future__ import annotations

import textwrap

from scripts import phase8_pysr_lodo_local


def _write_fake_worker(root, *, sleep_sec: float = 0.0) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "phase8_pysr_target_worker.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys
            import time
            from pathlib import Path

            for line in sys.stdin:
                command = json.loads(line)
                request = json.loads(
                    Path(command["request_path"]).read_text(encoding="utf-8")
                )
                time.sleep({sleep_sec})
                response = Path(command["response_path"])
                partial = response.with_name(response.name + ".partial")
                partial.write_text(
                    json.dumps({{"ok": True, "pid": os.getpid(), "value": request["value"]}}),
                    encoding="utf-8",
                )
                os.replace(partial, response)
            """
        ),
        encoding="utf-8",
    )


def test_target_worker_reuses_process(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    _write_fake_worker(root)
    monkeypatch.setattr(phase8_pysr_lodo_local, "ROOT", root)
    worker = phase8_pysr_lodo_local.TargetWorker(
        tmp_path / "tasks",
        target_timeout_sec=2.0,
        startup_timeout_sec=2.0,
    )
    try:
        first = worker.run({"value": 1})
        second = worker.run({"value": 2})
    finally:
        worker.close()
    assert first["ok"] is True
    assert second["ok"] is True
    assert first["pid"] == second["pid"]
    assert [first["value"], second["value"]] == [1, 2]


def test_target_worker_kills_timed_out_process(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    _write_fake_worker(root, sleep_sec=5.0)
    monkeypatch.setattr(phase8_pysr_lodo_local, "ROOT", root)
    worker = phase8_pysr_lodo_local.TargetWorker(
        tmp_path / "tasks",
        target_timeout_sec=0.5,
        startup_timeout_sec=0.5,
    )
    response = worker.run({"value": 1})
    assert response["ok"] is False
    assert response["timeout"] is True
    assert worker.process is None

"""Generate the Phase 0--9 Google Colab notebooks without stored outputs."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


PYTHON_310 = code(
    """
    # 必ず最初に実行する。Python 3.10でなければMiniconda導入後にruntimeが再起動する。
    import subprocess
    import sys

    print("active Python:", sys.version)
    if sys.version_info[:2] != (3, 10):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "condacolab==0.1.9"
        ])
        import condacolab
        condacolab.install_miniconda()
        raise SystemExit(
            "Python 3.10 Minicondaを導入しました。runtime再起動後、このNotebookへ再接続し、"
            "このセルをもう一度実行してください。"
        )
    print("Python 3.10: OK")
    """
)

BOOTSTRAP = code(
    """
    # Google認証とDrive mountはユーザー自身が行う。
    from google.colab import drive
    drive.mount("/content/drive")

    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    DRIVE_ROOT = Path("/content/drive/MyDrive/LTSR_colab")
    REPO_ROOT = Path("/content/LTSR")
    BRANCH = "20260726/gpu-scale-prep-colab"
    REPO_URL = (
        "https://github.com/blabo25226/"
        "Layer-selective_Transformer-based_Symbolic_Regression.git"
    )
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = DRIVE_ROOT / "source_lock.json"

    if not (REPO_ROOT / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_ROOT)],
            check=True,
        )

    if lock_path.is_file():
        locked_commit = json.loads(lock_path.read_text(encoding="utf-8"))["commit"]
        subprocess.run(["git", "checkout", "--detach", locked_commit], cwd=REPO_ROOT, check=True)
    else:
        locked_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        partial = lock_path.with_suffix(".json.partial")
        partial.write_text(
            json.dumps({"branch": BRANCH, "commit": locked_commit}, indent=2),
            encoding="utf-8",
        )
        os.replace(partial, lock_path)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from colab_runtime import assert_locked_source, require_python_310

    require_python_310()
    print("locked commit:", assert_locked_source(REPO_ROOT, DRIVE_ROOT))
    print("Drive root:", DRIVE_ROOT)
    """
)

RUN_CONFIG = code(
    """
    # Phase 4--8で同じ3値を使用する。別設定は必ず新しいRUN_IDにする。
    from colab_runtime import config_for

    RUN_KIND = "smoke"  # "smoke" -> "pilot" -> "paper"
    RUN_ID = "colab_smoke_20260726_01"
    MAX_PARALLEL_SEEDS = 1  # smoke実測後にのみ増やす
    CONFIG = config_for(
        RUN_KIND, RUN_ID, max_parallel_seeds=MAX_PARALLEL_SEEDS
    )
    print(json.dumps(CONFIG.scientific_dict(), indent=2, ensure_ascii=False))
    """
)


def common_intro(title: str, purpose: str) -> list[dict]:
    return [
        markdown(
            f"""
            # {title}

            {purpose}

            実行順は、Python 3.10確認 → Drive mount/source固定 → Phaseセルである。
            Python環境導入でruntimeが再起動した場合は、再接続して最初のセルからやり直す。
            """
        ),
        PYTHON_310,
        BOOTSTRAP,
    ]


def setup_notebook() -> dict:
    cells = common_intro(
        "LTSR Colab Phase 0 — setup / preflight",
        "Python 3.10、CUDA、依存関係、checkpoint、DREAM4、GSE112372を準備する。",
    )
    cells += [
        markdown(
            """
            ## 依存関係

            CUDA版PyTorchを先に固定し、その後GPU requirementsとNeSymReSを導入する。
            `requirements/dev.txt`はCPU版torchで上書きするため使用しない。
            """
        ),
        code(
            """
            import subprocess
            import sys

            commands = [
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                [
                    sys.executable, "-m", "pip", "install", "torch==2.5.1",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ],
                [sys.executable, "-m", "pip", "install", "-r", "requirements/gpu.txt"],
                [sys.executable, "-m", "pip", "install", "-e", "NSRS/src"],
                [sys.executable, "-m", "pip", "install", "pytest", "pysr"],
            ]
            for command in commands:
                print("+", " ".join(command), flush=True)
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            """
        ),
        markdown("## checkpointと外部データをDriveへcache"),
        code(
            """
            import shutil
            import tarfile
            import urllib.request

            checkpoint = DRIVE_ROOT / "checkpoints" / "100M.ckpt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not checkpoint.is_file():
                urllib.request.urlretrieve(
                    "https://huggingface.co/TommasoBendinelli/"
                    "NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt",
                    checkpoint,
                )
            print("checkpoint bytes:", checkpoint.stat().st_size)

            dream4_archive = DRIVE_ROOT / "data" / "dream4.tar.gz"
            if not dream4_archive.is_file():
                work = Path("/content/dream4_setup")
                work.mkdir(parents=True, exist_ok=True)
                zip_path = work / "dream4.zip"
                urllib.request.urlretrieve(
                    "https://gnw.sourceforge.net/resources/"
                    "DREAM4%20in%20silico%20challenge.zip",
                    zip_path,
                )
                shutil.unpack_archive(zip_path, work / "extracted")
                size10 = next((work / "extracted").rglob("Size 10"))
                data_root = REPO_ROOT / "data" / "dream4"
                data_root.mkdir(parents=True, exist_ok=True)
                shutil.copytree(size10.parent, data_root, dirs_exist_ok=True)
                dream4_archive.parent.mkdir(parents=True, exist_ok=True)
                with tarfile.open(dream4_archive, "w:gz") as handle:
                    handle.add(data_root, arcname="dream4")

            # Phase 8 download is executed before manifest creation so fingerprints exist.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src")
            subprocess.run(
                [
                    sys.executable, "-c",
                    "from pathlib import Path; "
                    "from data.human import prepare_gse112372; "
                    "p=prepare_gse112372(Path('data/human/gse112372_lps')); "
                    "print(p.source, sorted(p.X_donors))",
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
            human_archive = DRIVE_ROOT / "data" / "gse112372_lps.tar.gz"
            human_archive.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(human_archive, "w:gz") as handle:
                handle.add(
                    REPO_ROOT / "data" / "human" / "gse112372_lps",
                    arcname="gse112372_lps",
                )
            print("cached:", checkpoint, dream4_archive, human_archive)
            """
        ),
        markdown("## GPU preflightとテスト"),
        code(
            """
            from colab_runtime import restore_static_assets, run_command

            restore_static_assets(REPO_ROOT, DRIVE_ROOT)
            run_command(
                REPO_ROOT,
                [
                    sys.executable, "scripts/preflight_gpu.py",
                    "--weights", "NSRS/weights/100M.ckpt",
                    "--config", "NSRS/jupyter/100M/config.yaml",
                    "--eq-setting", "NSRS/jupyter/100M/eq_setting.json",
                ],
            )
            run_command(
                REPO_ROOT,
                [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"],
            )
            run_command(REPO_ROOT, ["bash", "-n", "scripts/run_gpu_pipeline.sh"])
            run_command(REPO_ROOT, [sys.executable, "-m", "pytest", "-q"])
            """
        ),
    ]
    return notebook(cells)


def diagnostic_notebook(phase: int, title: str, command: list[str]) -> dict:
    diagnostic_id = "colab_diagnostic_20260726_01"
    cells = common_intro(title, "Phase 1–3の再現・診断用。GPU本実験の設定選択には使わない。")
    cells += [
        code(
            f"""
            from colab_runtime import restore_artifacts, restore_static_assets, run_command, sync_artifacts

            DIAGNOSTIC_RUN_ID = "{diagnostic_id}"
            restore_static_assets(REPO_ROOT, DRIVE_ROOT)
            restore_artifacts(REPO_ROOT, DRIVE_ROOT, DIAGNOSTIC_RUN_ID)
            diagnostic_dir = REPO_ROOT / "results" / "runs" / DIAGNOSTIC_RUN_ID
            phase1_data = diagnostic_dir / "input_data" / "phase1_v1"
            env = {{
                "LTSR_RUN_DIR": str(diagnostic_dir),
                "LTSR_PHASE1_DATA": str(phase1_data),
                "LTSR_WEIGHTS": str(REPO_ROOT / "NSRS" / "weights" / "100M.ckpt"),
                "LTSR_CONFIG": str(REPO_ROOT / "NSRS" / "jupyter" / "100M" / "config.yaml"),
                "LTSR_EQ_SETTING": str(REPO_ROOT / "NSRS" / "jupyter" / "100M" / "eq_setting.json"),
                "LTSR_PHASE_TAG": "colab",
            }}
            command = {command!r}
            if command[0] == "python":
                command[0] = sys.executable
            run_command(REPO_ROOT, command, extra_env=env)
            print(sync_artifacts(REPO_ROOT, DRIVE_ROOT, DIAGNOSTIC_RUN_ID))
            """
        )
    ]
    return notebook(cells)


def phase_notebook(phase: int, title: str, purpose: str) -> dict:
    cells = common_intro(title, purpose)
    cells += [
        RUN_CONFIG,
        markdown(
            f"""
            ## Phase {phase}を実行

            同じrunの前Phase成果物をDriveから復元し、`START_PHASE={phase}`、
            `STOP_AFTER_PHASE={phase}`、`STRICT_RESUME=1`で実行する。
            3分ごと、および終了・例外時にDriveへ同期する。
            """
        ),
        code(
            f"""
            from colab_runtime import run_phase

            output = run_phase(REPO_ROOT, DRIVE_ROOT, CONFIG, {phase})
            print("Phase {phase} output:", output)
            """
        ),
        code(
            """
            payload = json.loads(output.read_text(encoding="utf-8"))
            print("top-level keys:", sorted(payload) if isinstance(payload, dict) else type(payload))
            print("Drive checkpoint:", DRIVE_ROOT / "runs" / RUN_ID)
            """
        ),
    ]
    return notebook(cells)


def validate_notebook() -> dict:
    cells = common_intro(
        "LTSR Colab Phase 9 — validate / archive",
        "Phase 4–8完了runを検査し、raw runとgraphsのtar.gz＋SHA256をDriveへ保存する。",
    )
    cells += [
        RUN_CONFIG,
        code(
            """
            import subprocess

            from colab_runtime import (
                assert_locked_source,
                restore_artifacts,
                restore_static_assets,
                sha256,
                sync_artifacts,
            )

            assert_locked_source(REPO_ROOT, DRIVE_ROOT)
            restore_static_assets(REPO_ROOT, DRIVE_ROOT)
            restore_artifacts(REPO_ROOT, DRIVE_ROOT, RUN_ID)
            run_dir = REPO_ROOT / "results" / "runs" / RUN_ID
            subprocess.run(
                [sys.executable, "scripts/validate_gpu_run.py", "--run-dir", str(run_dir)],
                cwd=REPO_ROOT,
                check=True,
            )
            sync_artifacts(REPO_ROOT, DRIVE_ROOT, RUN_ID)

            archive_dir = DRIVE_ROOT / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive = archive_dir / f"{RUN_ID}.tar.gz"
            subprocess.run(
                [
                    "tar", "-czf", str(archive),
                    f"results/runs/{RUN_ID}", f"graphs/{RUN_ID}",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            digest = sha256(archive)
            checksum = archive.with_suffix(archive.suffix + ".sha256")
            checksum.write_text(f"{digest}  {archive.name}\\n", encoding="utf-8")
            print("archive:", archive)
            print("sha256:", digest)
            """
        ),
    ]
    return notebook(cells)


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_setup_preflight.ipynb": setup_notebook(),
        "01_phase1_data.ipynb": diagnostic_notebook(
            1,
            "LTSR Colab Phase 1 — synthetic data",
            ["python", "scripts/issue6_generate_synthetic.py"],
        ),
        "02_phase2_baselines.ipynb": diagnostic_notebook(
            2,
            "LTSR Colab Phase 2 — baselines",
            [
                "python", "scripts/phase2_run_baselines.py",
                "--split", "test", "--limit", "2", "--pysr-iters", "2",
            ],
        ),
        "03_phase3_layer_scan.ipynb": diagnostic_notebook(
            3,
            "LTSR Colab Phase 3 — layer scan",
            [
                "python", "scripts/phase3_layer_scan.py",
                "--epochs", "1", "--eval-limit", "2",
                "--conditions", "pretrained,all_params,decoder_4",
            ],
        ),
        "04_phase4_contribution.ipynb": phase_notebook(
            4,
            "LTSR Colab Phase 4 — layer contribution",
            "validationだけでmulti-seed層寄与とlive rankingを生成する。",
        ),
        "05_phase5_selective_ft.ipynb": phase_notebook(
            5,
            "LTSR Colab Phase 5 — selective fine-tuning",
            "top/random/middle/bottom/fullをpaired seedで比較する。",
        ),
        "06_phase6_tpsr.ipynb": phase_notebook(
            6,
            "LTSR Colab Phase 6 — TPSR",
            "noise=0.1だけでFT/TPSR 2×2比較を実行する。noise slopeは評価しない。",
        ),
        "07_phase7_dream4.ipynb": phase_notebook(
            7,
            "LTSR Colab Phase 7 — DREAM4",
            "Size10/100をseed×network shardで実行し、trajectory leakageを防ぐ。",
        ),
        "08_phase8_human_lodo.ipynb": phase_notebook(
            8,
            "LTSR Colab Phase 8 — GSE112372 LODO",
            "4 donorのcross-donor application demoを実行する。",
        ),
        "09_validate_archive.ipynb": validate_notebook(),
    }
    for filename, payload in notebooks.items():
        payload["metadata"]["colab"]["name"] = filename
        (OUT / filename).write_text(
            json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(notebooks)} notebooks to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

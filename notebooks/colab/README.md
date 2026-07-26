# LTSR Google Colab notebooks

`GPU_RUN.md`のPhaseをGoogle Colab Pro上で分割実行するNotebook群である。
番号順に実行する。Phase 1–3は再現・診断用で、GPU本実験はPhase 4–8である。

## 実行順

1. `00_setup_preflight.ipynb`
2. `01_phase1_data.ipynb`（診断）
3. `02_phase2_baselines.ipynb`（診断）
4. `03_phase3_layer_scan.ipynb`（診断）
5. `04_phase4_contribution.ipynb`
6. `05_phase5_selective_ft.ipynb`
7. `06_phase6_tpsr.ipynb`
8. `07_phase7_dream4.ipynb`
9. `08_phase8_human_lodo.ipynb`
10. `09_validate_archive.ipynb`

## ユーザーが行う操作

- Codexアプリ内ブラウザでNotebookを開く。
- Googleログイン、MFA、Drive mountの承認を行う。
- GPU runtimeへ接続する。
- Python 3.10 worker導入でruntimeが再起動した場合、再接続して先頭セルから実行し直す。
- `RUN_KIND`、`RUN_ID`、`MAX_PARALLEL_SEEDS`の確認を行う。

それ以外の通常セル実行、ログ確認、エラー修正、再実行はAIへ委任できる。

## 重要事項

- Colab UI kernelの版にかかわらず、`/content/ltsr-py310/bin/python`のworkerがPython 3.10でなければ後続Phaseを実行しない。
- NotebookをDriveで開いてもrepoファイルは自動的に見えない。
  各Notebookが固定commitを`/content/LTSR`へcloneする。
- Phase 4–8では同じ`RUN_KIND`と`RUN_ID`を使う。
- Phase 6は`noise=0.1`だけを実行し、noise slopeは評価しない。
- Colabの一時VM上の成果物は定期的に
  `MyDrive/LTSR_colab/runs/<run-id>/`へ同期される。
- 本番run開始後にsource commitや科学設定を変更しない。

Notebookは`scripts/build_colab_notebooks.py`から機械生成する。
Notebookを直接編集した場合は、generatorとの不一致を解消してからcommitする。

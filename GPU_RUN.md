# GPU本実験の実行手順

この手順は、CPU上での単体テストとsmoke testが完了した後に使用する。GPU実験はrunごとに
`results/runs/<run-id>/`へ保存され、既存のCPU pilotレポートを上書きしない。

`scripts/run_gpu_pipeline.sh` が一度に回すのは **Phase 4 → 5 → 6 → 8** である。
DREAM4などの **Phase 7 は含まれない**。Phase 7は別スクリプトで実行する。

このスクリプトはbash前提である（Linux / WSL / Git Bashなど）。

## 1. 環境

Python 3.10を推奨する。NeSymReSが使用するHydra 1.0はPython 3.12と互換性がない。

```bash
conda create -n ltsr-gpu python=3.10 -y
conda activate ltsr-gpu
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/gpu.txt
pip install -e NSRS/src
pip install pytest pysr
```

CUDA版PyTorchが維持されていることを確認する。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. checkpointと設定

checkpoint、config、eq_settingは同じモデル構成の組を指定する。ファイル名だけから10M/100Mの
構造互換性を仮定しない。実行前検査はCUDAとファイルの存在、eq_settingの最低限の内容を確認し、
実際のモデルロードはPhase 4開始時にも検証される。

100M重みが未取得なら、例えば次で置く。

```bash
mkdir -p NSRS/weights
huggingface-cli download TommasoBendinelli/NeuralSymbolicRegressionThatScales \
  100M.ckpt --local-dir NSRS/weights
# または:
# wget -O NSRS/weights/100M.ckpt \
#   https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt
```

```bash
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
python scripts/preflight_gpu.py \
  --weights "$LTSR_WEIGHTS" --config "$LTSR_CONFIG" --eq-setting "$LTSR_EQ_SETTING"
```

`run_gpu_pipeline.sh` は `LTSR_WEIGHTS` 未設定だと即失敗する。以降のコマンドでも
上記のexportが同じシェルに残っている必要がある。

## 3. CPUで先に確認する

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

外部モデルを使うテストを含むため、必ずPython 3.10/3.11環境で実行する。

## 4. 小規模GPU smoke test

本実験の前に、別run IDで小規模実行する。

```bash
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
RUN_ID=gpu_smoke NPS=2 SEEDS="0 1" EPOCHS=1 EVAL_LIMIT=2 \
LR_GRID="1e-4" EPOCH_GRID="1" PATIENCE=0 \
BEAM=1 BFGS_RESTARTS=1 BFGS_STOP=0.2 NOISE="0.0" PYSR=0 \
bash scripts/run_gpu_pipeline.sh
```

`results/runs/gpu_smoke/manifest.json`のstatusが`complete`で、各PhaseのJSONとreportが存在することを確認する。
あわせて、いずれかの`per_problem`に推定式（`pred`）と真の式（`true`、存在する場合）が残っていることも見る。

## 5. 本実験

```bash
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
RUN_ID=paper_gpu_01 SEEDS="0 1 2 3 4" NPS=24 EPOCHS=8 \
LR_GRID="1e-5 3e-5 1e-4" EPOCH_GRID="4 8" PATIENCE=2 \
BEAM=5 BFGS_RESTARTS=5 BFGS_STOP=2.0 PYSR=1 \
bash scripts/run_gpu_pipeline.sh
```

`LR_GRID`と`EPOCH_GRID`は、各trainable条件へ同じ候補数を与えるvalidation探索である。
各候補は同じseedとデータ順で学習し、validation CEが最良の重みだけを独立testで一度評価する。
`PATIENCE`はearly stoppingの待機epoch数であり、0でも全epoch中の最良validation重みを復元する。

主な調整項目は`SEEDS`、`NPS`、`EPOCHS`、`LR_GRID`、`EPOCH_GRID`、`PATIENCE`、
`BEAM`、`BFGS_RESTARTS`、`BFGS_STOP`、`NOISE`、`PYSR`である。
Phase 6のTPSR予算はpipeline内で`--rollout 8 --horizon 30 --width 3`に固定されている。
変える場合は`scripts/run_gpu_pipeline.sh`を編集するか、Phase 6を個別実行する。

BFGSは主にCPUを使うため、最初から最大設定にせずsmoke testの時間から全体時間を見積もる。
decode（BFGS）と`--eval-limit`が総時間を支配しやすい。全件・beam5・restart5・stop2.0・5 seedだと
Phase 4 multi-seedだけでも数時間規模になり得る。まず`EVAL_LIMIT=30`程度で回し、問題なければ全件
（`EVAL_LIMIT=0`）へ上げる。

## 6. 出力

seedごとのPhase 5/6は`LTSR_PHASE_TAG=seedN`付きディレクトリへ書き、集約スクリプトが
`*_multiseed/`と`reports/`を作る。

```text
results/runs/<run-id>/
  manifest.json
  logs/
  phase4_multiseed/
    contrib_seed*.json
    contrib_aggregate.json
    raw_scores_seed*.json
    absolute_improvements_seed*.json
    contribution_status_seed*.json
    contribution_status_aggregate.json
    tuning_seed*.json
  phase5_seed*/
  phase5_multiseed/
  phase6_noise_seed*/
  phase6_noise_multiseed/
  phase8_lodo/
  reports/
    phase5_multiseed_report.md
    phase6_noise_multiseed_report.md
    ...

graphs/<run-id>/
  figures/
  tables/
```

manifestにはgit branch/commit、Python、PyTorch、CUDA、GPU、checkpoint SHA256、主要な環境変数、
開始・終了時刻、成否が保存される。途中でコマンドが失敗するとpipelineは停止し、statusは`failed`になる。

`graphs/<run-id>/`はpipeline開始時に空ディレクトリとして作られる。独立した図・表の自動生成は
行わない。可視化するときは[`graphs/README.md`](graphs/README.md)に従い、同じrun ID配下へ置く。

## 7. 結果を採用する条件

- Phase 4はvalidationのみで層を選択し、testを使用していない。
- 各trainable条件は同じLR×epoch候補数で探索され、選択基準はvalidation CEだけである。
- `phase4_multiseed/contribution_status_aggregate.json`で、full FTがpretrainedを改善したseed数を確認する。
- full FTが全seedで改善しない指標は正規化寄与度へ使わず、`absolute_improvements_seed*.json`を参照する。
- 有効なlive Phase 4順位が作れない場合、Phase 5は古いCPU順位へfallbackせず停止する。
- Phase 5は`phase4_multiseed/contrib_aggregate.json`から層を選ぶ。
- Phase 5のtest結果を見てLR、epoch、top-kを選び直さない。
- 問題単位JSONの`per_problem`に推定式・真の式・valid判定・複雑度が残っている。
- valid prediction rateとfailure-penalized NMSEを方法間で比較する。
- symbolic recovery、複雑度、実行時間もNMSEと併記する。
- 少数seed/donorの95% CIはStudentのt区間として解釈する。
- Phase 7（DREAM4など）を別途実行する場合は、有限差分前にtrajectory単位で分割し、
  そのrunも`results/runs/<run-id>/`へ分離して保存する。

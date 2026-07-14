# GPU 実行ガイド（スケールアップ版）

CPU パイロット（`--eval-limit 8`, 3 seed, 10M 重み, 軽量デコード）でパイプラインと
予備シグナルは検証済み。本ドキュメントは、その**論文規模版**を GPU マシンで回す手順。

パイロットとの違い（＝GPU で効くところ）:

| 項目 | パイロット(CPU) | GPU 本番 |
|---|---|---|
| 事前学習重み | 10M.ckpt | **100M.ckpt**（同一アーキ・より良い prior） |
| スイート | n=6, 骨格15 | **n=24〜**（大規模化） |
| seed | 3 | **5〜10**（CI を狭める / A-1） |
| デコード | beam1, restart1, stop0.3 | **beam5, restart5, stop2.0** |
| TPSR | rollout2/width2 | **rollout8/horizon30/width3** |
| eval | test 8件 | **全件** |

> **注意**: NeSymReS の "10M/100M" は**学習式数**であり、モデルは同じ約26M・12層。
> なので 100M へは「重みファイルの差し替え」だけで、config はそのまま使える。

---

## 1. 環境構築（Linux + NVIDIA GPU）

Python **3.10** を推奨（`hydra-core==1.0.0` がそのまま動く）。3.11/3.12 の場合は
`hydra-core==1.3.2 omegaconf==2.3.0` に上げれば OK（nesymres は `import hydra` と
`hydra.utils.to_absolute_path` しか使わない）。

```bash
conda create -n ltsr-gpu python=3.10 -y && conda activate ltsr-gpu
# GPU 版 torch（CUDA は環境に合わせる。例: cu124）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/base.txt          # torch は上で入れたのでスキップされる
pip install -e NSRS/src
pip install pysr                               # Phase 8 の PySR baseline 用（Julia 自動導入）
python -c "import torch; print('CUDA', torch.cuda.is_available())"
```

## 2. 100M 重みの取得

```bash
huggingface-cli download TommasoBendinelli/NeuralSymbolicRegressionThatScales \
    100M.ckpt --local-dir NSRS/weights
# もしくは wget:
# wget -O NSRS/weights/100M.ckpt \
#   https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
```

`LTSR_WEIGHTS` / `LTSR_CONFIG` / `LTSR_EQ_SETTING` を環境変数で指定すると、全スクリプトが
その checkpoint/config を使う（未設定なら同梱の 10M.ckpt）。

## 3. 実行

```bash
# 一括（推奨）
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
bash scripts/run_gpu_pipeline.sh

# 調整したい場合は env で上書き
SEEDS="0 1 2 3 4 5 6 7" NPS=40 BEAM=10 PYSR=1 bash scripts/run_gpu_pipeline.sh
```

個別に回す場合の例:

```bash
python scripts/phase4_multiseed.py --data-dir results/synthetic/diverse_gpu \
    --seeds 0 1 2 3 4 --epochs 8 --beam-size 5 --bfgs-restarts 5 --bfgs-stop-time 2.0
```

## 4. 出力

`results/phase_results/` に各レポートが上書き生成される:

- `phase4_multiseed_report.md` — 層寄与 mean±95%CI・top-3 安定度（A-1）
- `phase5_report.md` — 選択層 vs random/bottom/full（H2, A-3）
- `phase6_noise_report.md` — H3 ノイズ頑健性
- `phase8_lodo_report.md` — donor 交差検証（PySR vs selective）

## 5. 見どころ（パイロットで出た仮説の確定ポイント）

- **層の役割分担**：CE→decoder、予測→encoder が seed 安定で残るか（成功ケースB）。
- **H2**：top-k が random/bottom を CI で明確に上回り、少数層が全層 FT 以上か。
- **H3**：現実的 TPSR 予算でノイズ耐性が改善するか（パイロットでは小予算で不支持）。
- **汎化**：LODO で selective が PySR より holdout NMSE で勝つか（CI 付き）。

## 6. コスト目安

decode(BFGS) は CPU 律速なので、GPU でも `--eval-limit` と `--bfgs-*` が総時間を支配する。
全件・beam5・restart5・stop2.0・5 seed だと Phase 4 multi-seed だけで数時間規模。
まず `--eval-limit 30` 程度で回し、良ければ全件へ。BFGS を並列化したい場合は
decode ループのプロセス並列化が次の最適化候補。

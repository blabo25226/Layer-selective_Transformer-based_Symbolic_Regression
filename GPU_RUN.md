# GPU本実験の実行手順（リモートデスクトップ接続）

この手順は、WindowsのGPU PCへリモートデスクトップ接続し、VS CodeとAI支援を使って本実験を行う場合を想定する。
計算自体は **WSL2上のUbuntu + bash** で実行する。PowerShellやWindows版Pythonから
`scripts/run_gpu_pipeline.sh`を直接実行しない。

`scripts/run_gpu_pipeline.sh`は **Phase 4 → 5 → 6 → 7 → 8** を実行できる。
Phase 7は`DREAM4=1`のときだけ実行し、Size10/100の全networkを複数seedで評価する。

GPU本実験の出力は`results/runs/<run-id>/`へ保存し、既存のCPU pilotを上書きしない。
各runでは合成入力データも`results/runs/<run-id>/input_data/`へ保存する。

## 1. リモートGPU PCの準備

GPU PC側で次を準備する。

- NVIDIAドライバと、WSL2からGPUを利用できるWindows環境
- WSL2のUbuntu 22.04または同等環境
- Windows版VS Codeと`WSL`拡張
- WSL内のGit、Miniconda、`tmux`、`wget`、`unzip`
- 内蔵SSD上の十分な空き容量

Windowsの「電源とバッテリー」で、AC接続中にスリープへ入らない設定にする。長時間run中は、
Windows Updateによる自動再起動の時間帯も確認する。リモートデスクトップを閉じるときは **切断** を選び、
サインアウト、再起動、シャットダウンは行わない。切断後も`tmux`内のプロセスは継続するが、PCのスリープや再起動では停止する。

WSLターミナルで最初に確認する。

```bash
nvidia-smi
df -h .
```

VS CodeはWSL側のリポジトリから開く。

```bash
code .
```

VS Code左下が`WSL: Ubuntu`等になっており、統合ターミナルの`uname -a`がLinuxを示すことを確認する。

## 2. cloneと実行対象commitの固定

現在、GPU本実験用の修正は`gpu-scale-prep`ブランチにある。`main`が同じ内容だと仮定せず、
実行時点でユーザーが指定したブランチまたはcommitを明示的にcheckoutする。

```bash
git clone --branch gpu-scale-prep --single-branch \
  https://github.com/blabo25226/Layer-selective_Transformer-based_Symbolic_Regression.git LTSR
cd LTSR
git status --short
git branch --show-current
git log -1 --oneline
```

`git status --short`が空であることを確認する。実験開始後に`git pull`してコードを入れ替えない。
別PCで作った未pushの変更やcheckpoint、`data/`、`results/runs/`はcloneでは復元されない。

## 3. Python環境

Python 3.10を使う。NeSymReSが使用するHydra 1.0はPython 3.12と互換性がない。

```bash
conda create -n ltsr-gpu python=3.10 -y
conda activate ltsr-gpu
python -m pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/gpu.txt
pip install -e NSRS/src
pip install pytest pysr
```

`requirements/dev.txt`はここでは使わない。`dev.txt`は`cpu.txt`経由で`torch==2.5.1`のCPU wheelを要求し、
直前に入れたCUDAビルドを上書きするためである。テスト実行に必要なのは`pytest`だけなので個別に入れる。

PyTorchを入れた後に次を確認する。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()`が`False`なら本実験を始めない。`nvidia-smi`は動くがPyTorchから見えない場合は、
WSL対応NVIDIAドライバ、インストールしたPyTorchのCUDA build、WSL再起動の要否を確認する。

## 4. cloneに含まれないファイルを復元する

`.gitignore`により、次はclone先に存在しない。

| 対象 | 必要なPhase | 復元方法 |
|---|---|---|
| `NSRS/weights/*.ckpt` | Phase 4–8 | Hugging Faceから取得 |
| `data/dream4/` | Phase 7のみ | GNW公式archiveを取得・展開 |
| `data/human/gse112372_lps/` | Phase 8 | 実装がNCBI GEOから自動取得 |
| `results/runs/` | 新規run | 実行時に自動生成 |
| ローカルだけの外部repo群 | 今回のpipelineでは不要 | cloneしない |

`NSRS/jupyter/100M/config.yaml`と`eq_setting.json`、TPSRのMCTSコードはGit管理されている。
Phase 6はNeSymReS backbone上でTPSR探索を行うため、旧Phase 0で使ったTPSR E2E checkpointは不要である。

### 4.1 NeSymReS checkpoint（必須）

```bash
mkdir -p NSRS/weights
wget -O NSRS/weights/100M.ckpt \
  https://huggingface.co/TommasoBendinelli/NeuralSymbolicRegressionThatScales/resolve/main/100M.ckpt
sha256sum NSRS/weights/100M.ckpt
ls -lh NSRS/weights/100M.ckpt
```

ダウンロードが途中で切れた場合は、サイズだけで成功と判断せず再取得する。pipelineのmanifestにはcheckpointの
SHA256が記録される。checkpoint、config、eq_settingは同じモデル構成の組を指定し、ファイル名だけから互換性を仮定しない。

```bash
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
python scripts/preflight_gpu.py \
  --weights "$LTSR_WEIGHTS" --config "$LTSR_CONFIG" --eq-setting "$LTSR_EQ_SETTING"
```

これらの`export`はrunを起動する`tmux`内でも行う。

### 4.2 GSE112372（Phase 8、pipeline内で使用）

Phase 8は、ファイルがなければNCBI GEOからTPM表とmetadataを自動取得する。ただし本実験では **run開始前に必ず取得しておく**。
manifestの`data_fingerprints`はrun開始時点で計算されるため、Phase 8の自動取得に任せるとそのrunのヒトデータのSHA256が
`exists: false`として記録され、再現性証跡が残らない。長時間runの途中で通信失敗しない利点もある。

```bash
PYTHONPATH=src python -c "from pathlib import Path; from data.human import prepare_gse112372; p=prepare_gse112372(Path('data/human/gse112372_lps')); print(p.source, sorted(p.X_donors))"
find data/human/gse112372_lps -maxdepth 2 -type f -print
```

取得元はNCBI GEO accession `GSE112372`である。再取得が必要な場合だけPhase 8へ`--force-download`を渡す。
pipeline標準実行では既存ファイルを再利用する。

### 4.3 DREAM4（Phase 7を行う場合のみ）

GNW公式ページの`DREAM4 in silico challenge.zip`には、Size 10/100のtraining data、gold standard、
追加情報が含まれる。作業用一時ディレクトリへ展開し、`data/dream4/Size 10`と`Size 100`になるよう配置する。

```bash
DREAM4_TMP=$(mktemp -d)
wget -O "$DREAM4_TMP/dream4.zip" \
  "https://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.zip"
sha256sum "$DREAM4_TMP/dream4.zip"
unzip -q "$DREAM4_TMP/dream4.zip" -d "$DREAM4_TMP/extracted"
DREAM4_SIZE10=$(find "$DREAM4_TMP/extracted" -type d -name "Size 10" -print -quit)
test -n "$DREAM4_SIZE10"
mkdir -p data/dream4
cp -a "$(dirname "$DREAM4_SIZE10")/." data/dream4/
test -f "data/dream4/Size 10/DREAM4 training data/insilico_size10_1/insilico_size10_1_timeseries.tsv"
test -f "data/dream4/Size 100/DREAM4 training data/insilico_size100_1/insilico_size100_1_timeseries.tsv"
test -f "data/dream4/Size 10/DREAM4 gold standards/insilico_size10_1_goldstandard.tsv"
test -f "data/dream4/Size 100/DREAM4 gold standards/insilico_size100_1_goldstandard.tsv"
```

gold standardはregulator selectionのedge F1に必須であり、training dataだけでは
Phase 7が動かないため、上の4つすべてを確認する。`_goldstandard_signed.tsv`は任意で、
存在すれば制御の符号が利用される。

`data/`は`.gitignore`済みなので、`data/dream4/`が誤ってGitへ入ることはない。
archiveのSHA256、取得日、取得元URLをPhase 7のrunメモへ残す。上記一時ディレクトリは確認後に削除してよい。

## 5. CPU側の事前検証

```bash
python -m compileall -q src scripts tests
python -m pytest -q
bash -n scripts/run_gpu_pipeline.sh
```

すべて成功してからGPU smoke testへ進む。テスト数は今後変わり得るため、README記載の過去の件数との一致ではなく、
実行したcommitでfailureが0件であることを確認する。
DREAM4 archiveが未取得なら実データloaderテストはskipされ、4.3の配置後は実データを使って実行される。

## 6. 最初の確認後にAIへ引き渡す

初期設定と1–2時間の動作確認を人が行った後は、`run_gpu_campaign.sh`へ引き渡せる。
このスクリプトは非対話で、smoke test、本実験、Phase 4–8の集約、成果物検査、raw archive、
Git用の軽量成果物作成まで進める。
`PUBLISH_GIT=1`なら、検査済みの軽量成果物だけをcommitして現在のbranchへpushする。

campaignのrun IDは`<campaign-id>_smoke`と`<campaign-id>_full`である。以降の節および§9で
`<run-id>`と書いている箇所は、campaign経由なら`<campaign-id>_full`を指す。§8の手動runと
同じ文字列を`CAMPAIGN_ID`に使うと`results/runs/`に紛らわしい2つのディレクトリが並ぶため、
手動runとcampaignではIDを変える。

**campaignは全問題評価（`EVAL_LIMIT=0`）が既定であり、所要時間を見積もる中間段階を持たない。**
smokeの2問題からいきなり全件本番へ進むため、総時間とVRAMの見積りは§8の中規模runで
先に済ませておく。時間計測のためにcampaign自体を短くしたい場合だけ`EVAL_LIMIT`を明示的に
渡せるが、その実行はpilot扱いであり、最終test結果やhyperparameter選択には使えない。

引き渡し前に次が成立していることを確認する。

- conda環境がactivate済み
- `LTSR_WEIGHTS`、`LTSR_CONFIG`、`LTSR_EQ_SETTING`がexport済み
- 4.2のGSE112372と4.3のDREAM4が取得済み
- `git status --porcelain`が **完全に空**（untrackedを1つでも含むとcampaignは即座にexit 2で停止する）
- `PUBLISH_GIT=1`なら`git config user.name`、`git config user.email`、`origin` remoteが設定済みで、
  `git push origin HEAD`の認証も済んでいる
- AC接続中のスリープと予定外再起動を抑止済み

作業ツリーの検査は厳密なので、前のcampaignや手動runの残骸が起動を妨げる。特に、
commitしていない`results/published/<run-id>/`が残っていると起動できない。commitするか削除してから始める。
`graphs/`のうちsmoke由来のディレクトリ（`graphs/*_smoke/`と`graphs/gpu_smoke_*/`）と
`results/published/*_smoke/`は使い捨てのため`.gitignore`済みであり、起動を妨げない。

すでに人がsmoke testを完了していれば`RUN_SMOKE=0`にする。まだなら`RUN_SMOKE=1`のままAIへ渡す。

```bash
CAMPAIGN_ID=paper_gpu_YYYYMMDD_01
mkdir -p results/runs
tmux new-session -d -s ltsr-auto \
  "cd '$PWD' && CAMPAIGN_ID='$CAMPAIGN_ID' RUN_SMOKE=0 PUBLISH_GIT=1 \
  LTSR_WEIGHTS='$LTSR_WEIGHTS' LTSR_CONFIG='$LTSR_CONFIG' \
  LTSR_EQ_SETTING='$LTSR_EQ_SETTING' bash scripts/run_gpu_campaign.sh \
  > 'results/runs/${CAMPAIGN_ID}_campaign.log' 2>&1"
```

進捗確認は次だけでよい。

```bash
tmux attach -t ltsr-auto
tail -f "results/runs/${CAMPAIGN_ID}_campaign.log"
```

campaignは次の条件で停止する。停止段階によってmanifestの状態と復旧方法が異なるため、
**すべての失敗を「新しいcampaign IDで再実行」で片付けてはならない。**

| 停止条件 | manifestの状態 | 復旧方法 |
|---|---|---|
| 作業ツリーがdirty、git設定不足（起動前検査） | 作成されない | 原因を直して同じIDで起動し直す |
| CUDA/checkpoint/config検査の失敗（preflight） | 作成されない（runディレクトリも作られない） | 環境を直して同じIDで起動し直す |
| smokeまたは本runのいずれかのPhase失敗 | `failed` | 原因を直し、**新しいcampaign ID**で再実行する |
| 必須集約JSON・seed別ファイルの欠落 | `validation_failed` | 原因を切り分け、**新しいcampaign ID**で再実行する |
| JSON破損、式レコードの必須欄欠落、式記録が0件 | `validation_failed` | 同上 |
| archive、`export_run_summary.py`の失敗 | `publication_failed` | **再計算不要。** 失敗したステップだけをやり直す |
| Git commit/push失敗（`PUBLISH_GIT=1`の場合） | `publication_failed` | **再計算不要。** 認証やremoteを直して手動でpushする |

pipelineは検査より前に完走してmanifestを`complete`にするため、検査以降の失敗は
`status`に`validation_failed`／`publication_failed`として上書きされ、`stages`に段階別の成否が残る。
`validation.json`の`status`も`validated`か`failed`のいずれかになり、`failed`のrunは
`export_run_summary.py`が公開を拒否する。

AIはログ、manifestの`status`と`stages`、`validation.json`を調べ、コード・データ・資源不足のどれかを
切り分ける。**再計算が必要なのは上表で「新しいcampaign ID」と書いた行だけ**であり、
公開段階の失敗で数時間〜数十時間の再実行をしてはならない。

## 7. 手動で小規模GPU smoke testを行う場合

VS Codeのターミナルを閉じたりリモートデスクトップを切断したりしても計算が継続するよう、`tmux`内で起動する。

```bash
tmux new -s ltsr-smoke
conda activate ltsr-gpu
cd /path/to/LTSR
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
RUN_ID=gpu_smoke_YYYYMMDD NPS=2 SEEDS="0 1" EPOCHS=1 EVAL_LIMIT=2 \
LR_GRID="1e-4" EPOCH_GRID="1" PATIENCE=0 \
BEAM=1 BFGS_RESTARTS=1 BFGS_STOP=0.2 NOISE="0.0" PYSR=0 DREAM4=0 \
RANDOM_LAYER_SEEDS="0" NMSE_EQUIV_MARGIN=0.05 \
bash scripts/run_gpu_pipeline.sh
```

`RANDOM_LAYER_SEEDS="0"`は必ず付ける。既定は`"0 1 2 3 4"`であり、省略するとPhase 5が
random層集合を5通り学習してsmoke testが数倍遅くなる。campaign内蔵のsmokeもこの設定を使う。

`RUN_ID`は既存ディレクトリと重複させない。pipelineは既存runを検出すると、結果混在を防ぐため停止する。

`tmux`から離れるには`Ctrl-b`に続けて`d`を押す。再接続後は次で戻る。

```bash
tmux list-sessions
tmux attach -t ltsr-smoke
```

別ターミナルから進捗だけを見る場合は次を使う。

```bash
tail -f results/runs/gpu_smoke_YYYYMMDD/logs/pipeline.log
nvidia-smi
```

smoke test後は次を確認する。手動smokeでは検査スクリプトが自動では走らないので、
`python scripts/validate_gpu_run.py --run-dir results/runs/gpu_smoke_YYYYMMDD`を明示的に実行する。

- `manifest.json`の`status`が`complete`で、`stages.validation.status`が`complete`
- `validation.json`の`status`が`validated`
- Phase 4、5、6、8のJSONとreportがrun配下に存在する
- `phase8_lodo_seed*/`がrun配下にあり、`results/phase_results/phase8/`を更新していない
- Phase 8 LODO reportに同じrunのPhase 4から選んだ層とranking sourceが記録される
- `per_problem`に生の最良式、簡約式、候補式一覧、`true_expr`（存在する場合）、変数対応、valid判定、失敗理由、複雑度が残る
- `phase4_multiseed/equations_seed*.json`にPhase 4の条件別・問題別の数式が残る
- `git status --short`に既存の追跡ファイルの変更がない。`RUN_ID`を`gpu_smoke_*`または`*_smoke`の形に
  しておけば`graphs/`側もgitignoreされ、untrackedとして残って次のcampaign起動を妨げることがない

## 8. 手動でGPU本実験を行う場合

smoke testの所要時間とVRAM使用量を記録し、設定を確定してから新しい`tmux` sessionで実行する。

### 8.1 まず中規模runで時間を見積もる

BFGSは主にCPUを使い、decodeと`EVAL_LIMIT`が総時間を支配しやすい。**本番設定をそのまま流す前に、
`EVAL_LIMIT=30`程度の中規模runで所要時間とVRAMを測る。** この中規模runはpilotであり、
これを見てhyperparameterを変更した場合、そのrunは最終test結果の選択には使わない。

```bash
RUN_ID=pilot_gpu_YYYYMMDD_01 SEEDS="0 1" NPS=24 EPOCHS=8 EVAL_LIMIT=30 DREAM4=0 \
LR_GRID="1e-5 3e-5 1e-4" EPOCH_GRID="4 8" PATIENCE=2 \
RANDOM_LAYER_SEEDS="0 1 2 3 4" NMSE_EQUIV_MARGIN=0.05 \
BEAM=5 BFGS_RESTARTS=5 BFGS_STOP=2.0 NOISE="0.0 0.05 0.1 0.2" PYSR=1 \
bash scripts/run_gpu_pipeline.sh
```

Phase 7は`EVAL_LIMIT`の影響を受けない。Size10とSize100の全network（各5個）を全標的について
SRするため、`DREAM4=1`のときはこれが総時間の支配項になり得る。Phase 7の時間は
`DREAM4=1 SEEDS="0"`で1 seedだけ流して別途測る。

### 8.2 本実験

```bash
tmux new -s ltsr-paper
conda activate ltsr-gpu
cd /path/to/LTSR
export LTSR_WEIGHTS="$PWD/NSRS/weights/100M.ckpt"
export LTSR_CONFIG="$PWD/NSRS/jupyter/100M/config.yaml"
export LTSR_EQ_SETTING="$PWD/NSRS/jupyter/100M/eq_setting.json"
RUN_ID=paper_gpu_YYYYMMDD_01 SEEDS="0 1 2 3 4" NPS=24 EPOCHS=8 EVAL_LIMIT=0 DREAM4=1 \
LR_GRID="1e-5 3e-5 1e-4" EPOCH_GRID="4 8" PATIENCE=2 \
RANDOM_LAYER_SEEDS="0 1 2 3 4" NMSE_EQUIV_MARGIN=0.05 \
BEAM=5 BFGS_RESTARTS=5 BFGS_STOP=2.0 NOISE="0.0 0.05 0.1 0.2" PYSR=1 \
bash scripts/run_gpu_pipeline.sh
```

`NOISE`と`EVAL_LIMIT`は既定値と同じだが、Phase 6のコストがノイズ水準の数に比例するため明示する。

`LR_GRID`と`EPOCH_GRID`は、各trainable条件へ同じ候補数を与えるvalidation探索である。
各候補は同じseedとデータ順で学習し、validation CEが最良の重みだけを独立testで一度評価する。
`PATIENCE=0`でも全epoch中の最良validation重みを復元する。

**この探索が適用されるのはPhase 4とPhase 5だけである。** Phase 6、Phase 7、Phase 8は
固定学習率（`--lr`既定 1e-4）と固定epoch数で学習し、validationによる候補選択もbest-weight復元も行わない。
これらのPhaseの結果を「条件間で探索予算をそろえた比較」として提示してはならない。

Phase 6のTPSR予算はpipeline内で`--rollout 8 --horizon 30 --width 3`に固定されている。
変更する場合は、本実験前に設定をコードへ明示し、commitを分ける。方法間ではwall-clock時間または候補評価回数も保存・比較する。

## 9. 出力、Git履歴、回収

```text
results/runs/<run-id>/
  manifest.json
  validation.json                            validate_gpu_run.py の検査結果
  logs/pipeline.log
  input_data/
    diverse_gpu/
    phase7_dreamlike_v1/
  phase4_multiseed/
    equations_seed*.json
    raw_scores_seed*.json
    absolute_improvements_seed*.json
    contribution_status_seed*.json
    tuning_seed*.json
    contrib_seed*.json
    contrib_aggregate.json
    contribution_status_aggregate.json
    absolute_improvements_aggregate.json
    layer_ranking_scores.json
    layer_ranking_metadata.json
    layer_rankings.json
    layer_importance_evidence.json
    ranking_stability.json
  phase5_seed*/
  phase5_multiseed/
  phase6_noise_seed*/
  phase6_noise_multiseed/
  phase7_dream4_size10_seed*/
  phase7_dream4_size100_seed*/
  phase7_multiseed/
  phase8_lodo_seed*/
  phase8_lodo_multiseed/
  reports/

graphs/<run-id>/
  figures/
  tables/
```

manifestにはgit branch/commitとdirty状態、Python、`pip freeze`、PyTorch、CUDA、NVIDIA driver、GPU、
checkpoint SHA256、GSE112372/DREAM4のtree SHA256、主要設定、開始・終了時刻、成否が保存される。
`data_fingerprints`は **run開始時点** の内容から計算されるため、GSE112372とDREAM4は§4.2・§4.3で
先に取得しておく必要がある。

途中で失敗するとpipelineは停止し、`status`は`failed`になる。同じrun IDへ再実行して結果を混ぜず、原因を直して新しいrun IDを使う。
pipeline完走後の検査・公開で失敗した場合は`status`が`validation_failed`または`publication_failed`へ
上書きされ、`stages`に段階ごとの成否と時刻が残る。したがってrunの採否は`status`、`stages`、
`validation.json`の3つで判断する。`status`が`complete`でも`validation.json`が無いrunは未検査であり、公開してはならない。

各`per_problem`行とPhase 4の`equations_seed*.json`には最低限、次を保存する。

- `eq_id`、`true_expr`
- `pred_raw`（decoder/BFGSが返した最良式）と後方互換の`pred`
- `pred_simplified`（SymPyによる簡約式）と`simplification_error`
- `candidate_expressions`（NeSymReSでは返された候補式一覧）
- `variable_names`と`variable_mapping`（局所変数、元列、実遺伝子名）
- `decoder`、`decoder_metadata`、`failure_reason`
- NMSE、R2、variable F1、complexity、valid判定、安全性指標

`validate_gpu_run.py`はこれらの必須フィールド、変数対応の件数、失敗時の理由に加え、
Phase 4のseed別ファイル（`equations_seed*`、`raw_scores_seed*`、`absolute_improvements_seed*`、
`contribution_status_seed*`、`tuning_seed*`）と`contribution_status_aggregate.json`の存在を検査する。
これらは§10の採用条件が根拠として参照するファイルなので、欠けているrunはarchive・Git公開へ進めない。
検査に落ちたrunは`validation.json`が`status: failed`となり、`export_run_summary.py`が公開を拒否する。

### なぜ`results/runs/`をgitignoreするのか

`results/runs/`には、全問題の予測、ログ、生成入力データなど、再生成可能だが大きくなりやすいraw成果物が入る。
これを通常のGit履歴へ入れると、削除後もrepository履歴へ残ってcloneが重くなり、checkpointや外部データを誤って含める危険もある。
そのためraw runはgitignoreし、研究用ストレージのarchiveを正本とする。
campaignのarchive先は既定で`results/archives/`であり、ここもGit管理外である。別ディスクや研究用ストレージへ
直接保存する場合は、開始時に`ARCHIVE_DIR=/mnt/research-storage/ltsr`のように指定する。

一方、GitHubから実験の存在と主要結果を確認できるよう、検査済みrunから次を`results/published/<run-id>/`へ自動抽出する。

- manifestとvalidation結果
- Phase 4–8の集約JSON
- Phase別Markdown report
- checkpoint SHA256、実行commit、branch

`results/published/`と`graphs/`はgitignoreされていない。campaignを`PUBLISH_GIT=1`で起動した場合は、
この軽量成果物と同じrunの図表だけをcommit・pushする。したがって、push後はGitHub上で集約結果と再現情報を確認できる。rawの全式・ログ・入力データは
archive側に残し、公開用READMEにraw archiveの保管場所を追記する。

`results/runs/`と取得データはGit管理外なので、GPU PCだけに置いたままにしない。完了後はrunと対応する図表をarchiveし、
研究用ストレージへコピーする。この研究では、リモートデスクトップ接続時に手元PCのドライブを共有し、
完成したarchiveとSHA256ファイルを共有ドライブ経由で手元PCへ回収する。archive作成例は次のとおりである。

```bash
RUN_ID=paper_gpu_YYYYMMDD_01
tar -czf "${RUN_ID}.tar.gz" "results/runs/${RUN_ID}" "graphs/${RUN_ID}"
sha256sum "${RUN_ID}.tar.gz" > "${RUN_ID}.tar.gz.sha256"
```

campaignを使った場合は、同じ2ファイルが自動的に次へ作成される。archiveには本runに加えて、
smoke run（`results/runs/<campaign-id>_smoke`と`graphs/<campaign-id>_smoke`）と
campaignログ（`results/runs/<campaign-id>_campaign.log`）も含まれる。campaignログはarchive作成時点で
まだ書き込み中なので、archive内のコピーには末尾数行が入らない。最終的な終了行はGPU PC側のログで確認する。

```text
results/archives/<campaign-id>_full.tar.gz
results/archives/<campaign-id>_full.tar.gz.sha256
```

リモートデスクトップ接続前に、接続設定の「ローカル リソース」から回収先ドライブを共有する。
GPU PCのWindows Explorerでは通常、共有したドライブが「リダイレクトされたドライブ」または
`\\tsclient\<drive-letter>`として見える。WSL2で実験した場合は、Windows Explorerから
`\\wsl.localhost\<distribution>\home\<user>\...\LTSR\results\archives`を開き、上の2ファイルを
共有ドライブへコピーする。distribution名はGPU PC側のPowerShellで`wsl -l -q`を実行して確認できる。

コピー後は手元PCのPowerShellでSHA256を再計算し、`.sha256`に記録された値と一致することを確認する。

```powershell
Get-FileHash -Algorithm SHA256 .\<campaign-id>_full.tar.gz
Get-Content .\<campaign-id>_full.tar.gz.sha256
```

一致とarchiveの展開確認が終わるまでは、GPU PC側の`results/runs/`と`results/archives/`を削除しない。

巨大なrun、checkpoint、DREAM4/GEOデータをGit commitしない。コードや文書を変更した場合だけ、差分とテスト結果を確認して
別途commit・pushする。run archiveのコピー後に元データを削除するかは、archiveの展開確認とバックアップ確認を終えてから判断する。

## 10. 結果を採用する条件

- runのmanifestが`status: complete`で、`validation.json`が`status: validated`である。
- Phase 4はvalidationのみで層を選択し、testを使用していない。
- Phase 8は同じGPU runの`phase4_multiseed/layer_ranking_scores.json`を使い、旧CPU順位へfallbackしていない。
- **Phase 4とPhase 5では** 各trainable条件が同じLR×epoch候補数で探索され、選択基準はvalidation CEだけである。
- Phase 6・7・8は固定学習率と固定epochで学習しており、探索予算をそろえた比較としては提示していない。
- `contribution_status_aggregate.json`でfull FTがpretrainedを改善したseed数を確認する。
- full FTが全seedで改善した指標だけ正規化寄与度を使い、それ以外はvalidation上のpretrainedからの絶対改善量へ自動的に切り替えている。
- `layer_ranking_metadata.json`に指標ごとの順位根拠、`layer_rankings.json`に統合順位、`ranking_stability.json`にseed間Spearman/Kendall順位相関が残る。
- `layer_importance_evidence.json`で改善スコアの95%区間が0を超える層を確認し、単に「相対的に最上位」の層を重要層と断定していない。
- 正規化寄与度と絶対改善量のどちらからも有効なlive Phase 4順位を作れない場合だけ、後続Phaseは停止している。
- Phase 5のtest結果を見てLR、epoch、top-kを選び直していない。
- Phase 5は5個のrandom層集合を各training seed内で平均し、top 1/2/3とpretrained、middle、bottom、fullについてNMSE、R2、valid率、式複雑度をpaired比較している。
- top対fullはfailure-penalized NMSE差の95% t区間を保存し、事前指定した`NMSE_EQUIV_MARGIN`内への包含で同等性・非劣性を判定している。
- valid prediction rateとfailure-penalized NMSEを主結果に含める。
- symbolic recovery、variable F1、複雑度、実行時間をNMSEと併記する。
- seed、問題、networkを区別し、paired比較とStudentのt区間を適切な独立単位で計算する。
- 単一seed、単一network、4 donorsの結果を一般的結論にしていない。
- DREAM4ではtrajectory分割後に有限差分を計算している。
- DREAM4のcorr/MI/LASSO候補はtrain trajectoryだけで選び、test trajectoryでは固定している。
- Phase 6はFT主効果、TPSR主効果、交互作用をseed内paired差として集約している。
- Phase 7はSize10/100の全networkを複数seedで評価し、network内平均とseed間t区間を区別している。
- Phase 8は複数training/decode seedでLODOを行い、全式、valid率、特異点、外挿有限性を保存している。
- GSE112372の導関数は真のODE微分ではなくproxyであり、ヒトの真の因果機構を回復したとは表現していない。
- `tan`、危険な除算、特異点、外挿不安定性を、NMSEが良いという理由だけで妥当な式としていない。

Phase 8は4 donorsしかないapplication demoであり、seedを増やしても生物学的独立標本数が増えるわけではない。
seed間CIとdonor間変動を混同せず、真のヒト制御ODEや因果機構を回復したとは主張しない。

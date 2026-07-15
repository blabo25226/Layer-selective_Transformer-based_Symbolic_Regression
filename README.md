# LTSR研究：CPU実験の現状まとめ

## 1. 研究概要

本研究は、事前学習済みTransformer型記号回帰モデル **NeSymReS** を遺伝子制御ネットワーク
（Gene Regulatory Network; GRN）の制御方程式推定へ適応するとき、全パラメータを更新するのではなく、
寄与の大きい少数のTransformer層だけを微調整する方法を検討するものである。

研究題目は次のとおりである。

> **Layer-Selective Transformer-Based Symbolic Regression for Gene Regulatory Equation Discovery**

対象とする問題は、遺伝子発現状態 \(x\) から各遺伝子の変化率を表す関数
\(dx_i/dt=f_i(x)\) を、解釈可能な数式として推定することである。基本モデルにNeSymReS、
数式探索の改善にTPSR、非ニューラル記号回帰の比較対象にPySRを使用した。

研究計画の詳細は [`plan/20260714_firstplan.md`](plan/20260714_firstplan.md)、GPU本実験の手順は
[`GPU_RUN.md`](GPU_RUN.md) に記載している。

## 2. 研究上の問い

現段階では、次の問いを中心に検証している。

1. GRN方程式への適応効果は、NeSymReSの特定層に集中するか。
2. 高寄与層だけの微調整は、全層・ランダム層・低寄与層の微調整より効率的か。
3. 選択的微調整とTPSRを組み合わせることで、精度・式複雑度・ノイズ耐性を改善できるか。
4. 合成GRNで得られた適応方法は、DREAM4やヒト時系列発現データへ転移できるか。

## 3. CPU環境と実装状況

CPU実験はWindows上のPython 3.10環境で実施した。CUDAを利用できないため、大規模なseed反復、
高beam幅、十分なMCTS rolloutを必要とする計算は実施していない。

| 項目 | 状況 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cpu |
| NeSymReS | checkpointロード・推論・選択的微調整に成功 |
| PySR | Juliaバックエンドを含め動作確認済み |
| TPSR | E2EおよびNeSymReSバックボーン用コードを統合 |
| テスト | Python 3.10で **31 passed** |
| GPU計算 | 未実施 |

ローカルの `10M.ckpt` はファイル名と異なり、state dict上はencoder/decoder各5層の
100M設定側アーキテクチャである。そのため `NSRS/jupyter/100M/config.yaml` と組み合わせている。

## 4. 実施した研究Phase

| Phase | 内容 | CPUでの到達点 |
|---|---|---|
| 0 | 環境・checkpoint・ベースラインの動作確認 | 完了 |
| 1 | 合成Hill式・toggle・repressilator・多様な式構造の生成 | 完了 |
| 2 | NeSymReSとPySRの基礎比較 | 完了 |
| 3 | encoder/decoder各層の選択的微調整スキャン | 完了（探索的） |
| 4 | 層寄与度とseed安定性の測定 | CPU小規模実験まで完了 |
| 5 | top/middle/random/bottom/full微調整の比較 | CPU単一seed実験まで完了 |
| 6 | 選択的FT × TPSRの2×2比較とノイズ試験 | smoke testまで完了 |
| 7 | DREAM4 Size10/Size100、SBML teacherによる転移 | CPU小規模実験まで完了 |
| 8 | ヒトLPS刺激時系列、holdout donor、LODO評価 | CPU実験完了 |

## 5. CPU実験結果

### 5.1 Phase 0：実行基盤

次の基本動作を確認した。

- NeSymReSは `x_1*sin(x_1)` を同値な式として復元した。
- PySRは `x*sin(x)` を復元した。
- TPSR E2EモデルをWindows/CPU上でロードし、軽量MCTSを完走した。
- Linuxで保存されたcheckpoint、NumPy 2.0、Python dataclassに対する互換修正を適用した。

詳細：[`results/phase_results/phase0_report.md`](results/phase_results/phase0_report.md)

### 5.2 Phase 1–2：合成GRNとベースライン

最初の合成データは26問題で、activation、repression、toggle、repressilatorを含む。
各問題は200点、変数範囲は概ね \([0,3]\)、初期版はノイズなしである。

9個のtest問題に対する初期比較は次のとおりであった。

| 方法 | median NMSE（ID） | median NMSE（OOD） | median R²（ID） |
|---|---:|---:|---:|
| NeSymReS beam=2 | 0.376 | 0.901 | 0.602 |
| NeSymReS beam=5 | 0.120 | **0.187** | 0.806 |
| PySR | **0.00072** | 0.327 | **0.999** |

小規模な式ではPySRがin-domain適合で大幅に優れていた。一方、この9問題に限れば、
NeSymReS beam=5はOOD中央値でPySRを上回った。ただし問題数が少なく、これだけでは
NeSymReSの一般化優位性を主張できない。

詳細：[`phase1_report.md`](results/phase_results/phase1_report.md)、
[`phase2_report.md`](results/phase_results/phase2_report.md)

### 5.3 Phase 3–4：層ごとの役割

Phase 3の探索では、teacher-forcing cross-entropy（CE）に対してdecoder後段の寄与が大きかった。
Phase 4の3 seeds実験でも、CE寄与の上位は安定して次の3層となった。

| CE順位 | 層 | 平均寄与度 | top-3出現率 |
|---|---|---:|---:|
| 1 | `decoder_4` | 0.811 | 100% |
| 2 | `decoder_3` | 0.800 | 100% |
| 3 | `decoder_2` | 0.626 | 100% |

一方、予測NMSE/R²では `encoder_1`、`encoder_3`、`encoder_5` などencoder側が上位となった。
この不一致から、少なくとも次の可能性が示唆される。

- decoder後段は正解token列を生成するCE最適化に強く関与する。
- encoder側は入力された数値点群の表現と、最終的な数値予測性能に関与する。
- 「高寄与層」は評価指標によって変わるため、単一のランキングだけでは不十分である。

ただしsymbolic recoveryは全条件でほぼ0、variable F1の寄与度も識別不能であった。
したがって、CPU結果が示すのは層ごとのCE・予測性能の差であり、正しい式構造の回復を示したものではない。

詳細：[`phase3_report.md`](results/phase_results/phase3_report.md)、
[`phase4_multiseed_report.md`](results/phase_results/phase4_multiseed_report.md)

### 5.4 Phase 5：選択的微調整

60個の学習問題と8個の評価問題を用いた単一seedのCPU実験では、少数層の微調整が良好だった。

| 条件 | 学習パラメータ比 | NMSE | R² |
|---|---:|---:|---:|
| pretrained | 0% | 0.1946 | 0.6878 |
| top 1 | 9.96% | 0.0309 | 0.9316 |
| top 2 | 19.92% | **0.0167** | **0.9755** |
| top 3 | 29.89% | 0.0272 | 0.9427 |
| random 3 | 13.66% | 0.0852 | 0.8123 |
| bottom 3 | 13.66% | 0.2502 | 0.4542 |
| all parameters | 100% | 0.2818 | 0.4783 |

この実験ではtop層がrandom/bottom/fullを上回った。また、全層微調整はCEを改善しても予測NMSEを悪化させ、
過適合の可能性を示した。少数層だけを更新する研究方向には価値があると考えられる。

ただし、この数値は単一seedであり、後述する旧評価設計の問題も含むため、H2の確証ではなく予備結果である。

詳細：[`phase5_report.md`](results/phase_results/phase5_report.md)

### 5.5 Phase 6：TPSRとの組み合わせ

2問題だけを使った2×2 smoke testでは、選択的FTとTPSRの組み合わせが最良だった。

| 微調整 | 探索 | NMSE | R² |
|---|---|---:|---:|
| なし | beam | 0.203 | 0.684 |
| なし | TPSR | 0.529 | 0.356 |
| 選択的FT | beam | 0.193 | 0.700 |
| 選択的FT | TPSR | **0.082** | **0.879** |

軽量設定では、TPSR単独はbeamより悪化したが、選択的FT後には改善した。この結果は、
微調整されたpriorがMCTSを有効な探索領域へ誘導する可能性を示す。ただしn=2であり、方向確認にすぎない。

ノイズ0.0と0.1の比較では、`selective + TPSR` のNMSE劣化量は `selective + beam` より大きかった。
したがって、現時点ではH3のノイズ耐性仮説は支持されていない。また、旧CPUレポートは
精度–複雑度トレードオフを十分に集計していない。

詳細：[`phase6_report.md`](results/phase_results/phase6_report.md)、
[`phase6_noise_report.md`](results/phase_results/phase6_noise_report.md)

### 5.6 Phase 7：DREAM4への転移

#### Regulator selection

Size10の5 networksを集約したPackage Aでは、候補制御因子のedge F1は次の結果だった。

| 方法 | mean edge F1 |
|---|---:|
| oracle | **0.883** |
| correlation | 0.264 |
| mutual information | 0.279 |
| LASSO | 0.266 |

Size100 net1では、oracle 0.883に対し、correlation/MI/LASSOは約0.06–0.10だった。
DREAM4では、記号回帰以前にregulator preselectionが大きなボトルネックになっている。

#### Local symbolic regression

- Size10 net1では、oracle候補を与えてもNMSEは約0.84だった。
- Size100 net1では、oracle候補でもNMSEは約0.98だった。
- 合成dreamlikeデータによる選択的FTは、DREAM4有限差分ターゲットでは改善しなかった。
- 相関選択の誤りが加わると、Size10のNMSEは約0.98まで悪化した。

SBML由来teacherを使った微調整では、clean SBML holdout NMSEが0.311から0.0038へ改善したが、
DREAM有限差分への転移改善は0.890から0.725に留まった。この差は、teacher domainへの適合が
実データ転移よりはるかに容易であることを示す。

現段階では、DREAM4への転移成功は示されていない。むしろ、有限差分ノイズ、候補選択、
合成–DREAM間のdomain shiftが主要課題であることが明らかになった。

詳細：[`phase7_package_a_report.md`](results/phase_results/phase7_package_a_report.md)、
[`phase7_dream4_report.md`](results/phase_results/phase7_dream4_report.md)、
[`phase7_dream4_size100_report.md`](results/phase_results/phase7_dream4_size100_report.md)

### 5.7 Phase 8：ヒトLPS刺激時系列

GSE112372の20遺伝子、4 donors、5時点を使用した。5時点から推定した導関数は真のODE微分ではなく、
`smooth_fd` によるproxyである。

単一donor holdout実験では、選択的FTが良い結果を示した。

| 方法 | in-donor NMSE | holdout NMSE |
|---|---:|---:|
| pretrained beam | 0.405 | 0.596 |
| selective beam | 0.166 | **0.178** |
| PySR | **0.0054** | 0.502 |

しかし、4 donorsを順番にholdoutするLODOでは結果が逆転した。

| 方法 | mean in-donor NMSE | mean holdout NMSE | mean gap |
|---|---:|---:|---:|
| PySR | **0.0082** | **0.203** | **0.195** |
| selective beam | 0.208 | 0.487 | 0.279 |
| pretrained beam | 0.574 | 1.458 | 0.885 |

したがって、「選択的FTがPySRよりdonor間で一般化する」という主張は支持されない。
単一holdoutの良い結果はdonor選択に依存していた可能性が高い。一方、選択的FTは
pretrained NeSymReSよりは改善しており、domain adaptation自体の効果は示唆される。

生成式には `cos`、`tan`、特異点を持つ比など、生物学的解釈が難しい形も含まれた。
よって、これは真のヒト制御ODEの回復ではなく、候補priorを用いたcross-donor予測のapplication demoと位置づける。

詳細：[`phase8_report.md`](results/phase_results/phase8_report.md)、
[`phase8_lodo_report.md`](results/phase_results/phase8_lodo_report.md)

## 6. CPU研究から得られた結論

現時点で比較的強く言えることは次のとおりである。

1. **NeSymReSの適応効果は層によって異なる。** CEではdecoder後段、数値予測ではencoder側の寄与が大きい。
2. **全層微調整が常に最良ではない。** CPU pilotでは少数層の更新が全層更新より良い場合があった。
3. **選択的FTはNeSymReSのdomain adaptationを改善する可能性がある。** ただしPySRに対する一般的優位性は示されていない。
4. **TPSRとの相互作用は候補として残る。** 小規模実験では選択的FT後のみTPSRが改善したが、統計的検証は未実施である。
5. **DREAM4では候補制御因子の選択が支配的な課題である。** naiveなcorr/MI/LASSOはoracleから大きく劣る。
6. **実データ上の良い単一split結果は信用しすぎてはいけない。** ヒトLODOでは単一holdoutの結論が逆転した。
7. **symbolic recoveryは未達である。** 良いNMSEが得られても、正しい式構造や生物学的機構を回復したとは限らない。

## 7. 現在のCPUレポートを解釈する際の重要な注意

上記の数値は、最新のレビュー修正より前に生成されたCPU pilotを含む。コード側では次を修正済みだが、
CPUのPhase 4以降は計算時間の都合でまだ再実行していない。

- 旧Phase 4はtestを層選択にも使っていた。最新コードはtrainからmotif単位でvalidationを分け、testを選択に使わない。
- 旧Phase 5はPhase 4と同じtest上で比較していた。最新コードは独立testだけを最終評価に使う。
- 旧multi-seedはconditionごとにバッチ順が異なった。最新コードはseed内で乱数状態とバッチ順を固定する。
- 旧95% CIは少数標本でも `1.96 × SEM` だった。最新コードはStudentのt区間を使う。
- 旧DREAM4は有限差分後の行単位分割だった。最新コードは有限差分前にtrajectory単位で分割する。
- 旧集計はdecode失敗を中央値から除外した。最新コードはvalid率とfailure-penalized NMSEを主指標にする。
- 旧Phase 6は複雑度を十分に集約しなかった。最新コードは精度、valid率、式複雑度を同時に保存する。

このため、CPU結果から論文レベルでH2/H3を確定してはならない。現在の適切な表現は次のとおりである。

> 層選択的微調整、TPSR、DREAM4、ヒトLODOを含む研究パイプラインを構築し、CPU小規模実験で
> 動作と予備的傾向、ならびに主要な失敗要因を確認した。

## 8. 最新コードで追加した研究上の改善

- motif単位のtrain/validation分離
- 独立test評価
- seed内paired comparison
- multi-seed集約ランキングによる層選択
- Phase 5/6の全seed反復とt区間
- failure-awareなNMSE/R²、valid率、複雑度集計
- 正解式に実際に現れる変数だけを使うvariable recovery
- DREAM4のtrajectory単位分割
- ヒトprior候補集合内の正しいランキング
- runごとのmanifest、checkpoint SHA256、ログ、出力分離
- CPU/GPU別の依存関係とfail-fast GPU runner

これらの修正は実装・単体テスト済みであるが、新方式による大規模な数値結果はまだ存在しない。

## 9. 次に行うGPU実験

GPU実験では、次を優先する。

1. Phase 4を5–10 seedsで再実行し、validation上の層寄与とtop-3安定性を測定する。
2. Phase 5を同じseed集合で反復し、top層とrandom/bottom/fullのpaired差を求める。
3. Phase 6を複数noise・複数seedで実行し、精度–複雑度ParetoとTPSR interactionを評価する。
4. DREAM4 Size10/100を全network、trajectory分割で再評価する。
5. ヒトLODOではvalid率、式の外挿安定性、非負性、特異点の有無も評価する。
6. exact/skeleton recoveryが0の原因を調べ、式同値判定と探索設定を改善する。

本実験の前に、必ず [`GPU_RUN.md`](GPU_RUN.md) の小規模GPU smoke testを実行する。

## 10. 再現方法

### 環境

```bash
conda create -n ltsr python=3.10 -y
conda activate ltsr
pip install -r requirements/cpu.txt
pip install -e NSRS/src
pip install -r requirements/dev.txt
```

Hydra 1.0との互換性により、Python 3.12は本実験環境としてサポートしない。

### テスト

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

### 主要スクリプト

```text
scripts/generate_diverse_suite.py     構造分離した合成GRNの生成
scripts/phase4_multiseed.py           validation上の層寄与測定
scripts/phase5_selective_train.py     独立testでの選択的FT比較
scripts/phase6_noise_sweep.py         TPSR 2×2・ノイズ試験
scripts/phase7_dream4_size10.py       DREAM4 Size10
scripts/phase7_dream4_size100.py      DREAM4 Size100
scripts/phase8_lodo.py                ヒトleave-one-donor-out
scripts/run_gpu_pipeline.sh           GPU一括実行
```

## 11. リポジトリ構成

```text
plan/          研究計画
src/           データ処理、モデル、学習、評価の共通コード
scripts/       Phase別の実験エントリポイント
tests/         単体テスト
requirements/  CPU/GPU/dev別の依存関係
results/       CPU pilotの結果とrun出力
NSRS/          NeSymReS参照実装
TPSR/          TPSR参照実装
```

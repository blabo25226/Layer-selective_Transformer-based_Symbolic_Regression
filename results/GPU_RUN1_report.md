# GPU_RUN1 実行・結果・考察報告

作成日: 2026-07-30

対象run: `colab_reduced_20260729_03`

実行期間: 2026-07-26〜2026-07-29

作業ブランチ: `20260726/gpu-scale-prep-colab`

## 0. 要約

GPU_RUN1では、事前学習済みTransformer型シンボリック回帰モデルNeSymReSについて、合成遺伝子制御方程式での層寄与度、少数層fine-tuning、TPSRとの組合せ、DREAM4への転移、ヒト時系列データへの適用を、Google Colab ProのNVIDIA L4を中心に評価した。計算資源とColabの連続実行時間に制約があったため、本実行は当初計画を縮小した`reduced run`である。

主要な観測結果は次のとおりである。

1. 合成データのvalidationでは、fine-tuningの寄与が主にdecoderの中後段層へ集中した。特に`decoder_3`は、全層fine-tuningが得た数値誤差改善のほぼ全てを単独で再現した。
2. 合成データのtestでは、上位1〜3層だけのfine-tuningが全層fine-tuningと事前定義したNMSE同等性マージン内で同等だった。上位3層は事前学習モデルを明確に改善し、下位3層より良かった。
3. 一方、上位3層とランダム3層または中央3層の差の95%信頼区間は0を含んだ。「少数層で十分」という結果は支持されたが、「寄与度上位層でなければならない」という結果はまだ強くない。
4. TPSRはNMSEをさらに下げたが、selective fine-tuning後の追加改善は小さく不確実で、計算時間、式複雑度、valid rateの面で大きな代償があった。
5. DREAM4ではselective fine-tuningによりNMSEは改善したが、100変数条件の変数選択F1は約0.05まで低下した。oracle変数選択との差が大きく、高次元化ではTransformer本体より前段の候補制御因子選択が主要な律速となった。
6. ヒトLODO評価では、selective NeSymReSは事前学習モデルを改善したが、ローカルCPUで実行したPySRの方が低いholdout NMSEを示した。ただし4 donors・5時点の小規模データであり、方法間の演算子集合と探索budgetも完全には一致していない。
7. 低いNMSEを得た式にも`tan`、多数の除算、特異点に近い形、真の構造と異なる式が多かった。GPU_RUN1は数値近似能力を示したが、生物学的機構や真の制御ODEの回復を示したものではない。

したがってGPU_RUN1の中心的な学術的結論は、**NeSymReSのGRN適応では少数decoder層の更新で全層更新に近い数値性能を得られる可能性が高いが、高次元変数選択、構造回復、安全な演算子制約、実データでの外的妥当性が未解決である**、というものである。

## 1. この報告書の位置付け

本報告書は、次の保存済み資料と成果物に基づく。

- 当初の研究計画: [`plan/20260714_firstplan.md`](../plan/20260714_firstplan.md)
- GPU実行手順と実施記録: [`GPU_RUN1.md`](../GPU_RUN1.md)
- GPU_RUN2計画案: [`plan/20260729_GPU_RUN2.md`](../plan/20260729_GPU_RUN2.md)
- 最終run: `results/runs/colab_reduced_20260729_03/`
- 各Phaseの`summary.json`、問題単位JSON、生成レポート
- 最終manifest: `results/runs/colab_reduced_20260729_03/manifest.json`
- 検証結果: `results/runs/colab_reduced_20260729_03/validation.json`

`results/runs/`は大容量生成物としてGit管理外である。再検証には、後述するarchiveまたはローカルに保存したrun directoryが必要となる。

## 2. 実際の実行条件

### 2.1 計算環境

| 項目 | 実際の条件 |
|---|---|
| 主な実行基盤 | Google Colab Pro |
| GPU | NVIDIA L4 |
| 研究コード用Python | 3.10.13 |
| OS | Linux |
| PyTorch | 2.5.1+cu124 |
| CUDA | 12.4 |
| Drive保存先 | `/content/drive/MyDrive/LTSR_colab` |
| checkpoint SHA256 | `62aedc41fdb67ecbe3679f5ef030e7ef2bf0f4471c461b68d8814358968b324f` |
| seeds | 0, 1, 2 |
| noise | 0.1のみ |
| 主なbeam size | 2 |
| Phase 8 beam size | 1 |
| Phase 7 target timeout | 240秒 |
| Phase 8 target timeout | 30秒 |
| PySR | ローカルCPU、12 iterations |

ColabのUI自体はPython 3.12環境だったため、研究コードは別のPython 3.10 workerで動かした。これはHydraを含む既存依存関係との互換性を維持するためである。

### 2.2 当初計画からの主な変更

| 項目 | 当初の考え | GPU_RUN1で実際に行ったこと |
|---|---|---|
| ノイズ条件 | 複数水準 | `noise=0.1`だけに限定 |
| smoke実行 | 各段階で確認 | import、接続、出力、致命的エラーの確認だけに縮小 |
| decode budget | より大きいbeam | 主に`beam=2`、Phase 8は`beam=1` |
| Phase 4候補 | 広い層・評価budget | 候補数、beam、反復、評価件数を削減 |
| Phase 7範囲 | 3 seeds × 5 networks | 主集計は3 seeds × networks 1–3 |
| Phase 7保存 | network完了単位中心 | target単位checkpoint、shard、worker recycleを追加 |
| Phase 7 timeout | 長い処理を許容 | 240秒hard wall-clock timeout |
| Phase 8 decode | in-donorとholdoutで別decode | 学習データから1回生成し、同一式を両方で評価 |
| Phase 8 PySR | Colab内で実行 | ローカルCPUで実行し、後から統合 |
| 長時間CPU処理 | Colab内 | GPU_RUN2では原則ローカルへ分離する方針へ変更 |
| 実行中の修正 | 同一runのresume | provenanceを記録した新runへ成果物を継承 |

これらは計算時間を短くするためだけではなく、Colab切断後の再計算を減らし、方法間で何を比較したかを追跡可能にするための変更でもある。

### 2.3 継続runの系譜

長時間実行中にcheckpoint、resume、timeout、memory対策を追加したため、GPU_RUN1は次のrunを順に継承した。

| run ID | 主な役割 |
|---|---|
| `colab_reduced_20260726_01` | Phase 0から開始、Phase 4–6の主要成果物 |
| `colab_reduced_20260728_01` | Phase 7 target checkpointとresume修正 |
| `colab_reduced_20260729_01` | Phase 7 hard timeoutと再開処理 |
| `colab_reduced_20260729_02` | worker recycling、Phase 7縮小集計 |
| `colab_reduced_20260729_03` | 最適化したPhase 8、PySR統合、Phase 9 |

継承元、Git commit、コピー対象とchecksumは`continuation.json`へ保存した。最終manifestが記録するPhase 8時点のGit commitは`9db00f4d567de5c55846587dbda430d1ab6cca0b`であり、Phase 7継承元には別commitも含まれる。このためGPU_RUN1を「単一commitを最初から最後まで固定したrun」と表現してはならない。

## 3. 評価値の読み方

### 3.1 主な指標

- penalized NMSE: 失敗を無視せず罰する数値誤差指標。小さいほど良い。
- R2: 数値予測の当てはまり。大きいほど良い。
- valid rate: 生成式を評価できた割合。
- symbolic recovery / equivalence: 真の式構造を回復した割合。
- variable F1: 真の入力変数・制御因子を選べた程度。
- complexity: 生成式の複雑さ。
- near-singularity rate / extrapolation valid rate: 特異点や外挿不安定性の診断。

低いNMSEは真の式の回復を意味しない。特に有限区間上では、構造が異なる式、不要な除算を含む式、危険な三角関数を含む式でも低いNMSEを得られる。

### 3.2 統計集計

原則として、各seed内で問題またはnetwork/foldを集約し、そのseed-level値を3 seedsで平均した。表中の`±`は、特に断らない限りStudentのt分布による95%信頼区間の半幅である。

ただしseed数は3しかない。95%信頼区間は推定の不確実性を示すが、検出力は低く、分布仮定の確認も難しい。信頼区間が0を含まないことを、そのまま一般的な再現性や生物学的有意性へ拡張してはならない。

## 4. Phase別の実施内容と結果

### 4.1 Phase 0–3: 環境、データ、baseline、層scan

Phase 0ではDrive mount、Python 3.10 worker、checkpoint、DREAM4、GSE112372を準備した。GitHub上の存在しないcommit refによる404、依存関係、checkpoint互換性の問題を修正し、source commitとDrive上の`source_lock.json`を一致させた。

Phase 1では、合成データを`noise=0.1`だけで生成した。他のnoise水準を省略したため、本報告の合成データ結果を異なるnoise強度へ一般化することはできない。

Phase 2はreduced設定のbaselineとして完了した。Phase 3のlayer scanはエラーなく完了し、実測約10分だった。これらはPhase 4以降の層候補評価と比較条件の準備となった。

### 4.2 Phase 4: 層寄与度

Phase 4では、testを層選択に使用せず、motif単位で分離したvalidationだけで層寄与度を評価した。全3 seedsで全層fine-tuningが事前学習モデルを改善したため、baseからfullまでの改善を基準にした正規化寄与度を計算できた。

代表的なvalidation結果は次のとおりである。

| 指標 | Pretrained | `decoder_3`のみ | 全層fine-tuning |
|---|---:|---:|---:|
| Cross entropy | 1.201501 | 0.081479 | 0.054532 |
| Penalized NMSE | 0.058103 | 0.008091 | 0.008014 |
| Penalized R2 | 0.930491 | 0.989994 | 0.990116 |
| Variable F1 | 0.967593 | 0.972222 | 1.000000 |
| Symbolic recovery rate | 0.000000 | 0.527778 | 0.652778 |

`decoder_3`の正規化寄与度は、cross entropyで`0.976506 ± 0.009369`、NMSEで`0.998357 ± 0.006752`、R2で`0.997942 ± 0.005892`だった。数値誤差に関しては、`decoder_3`単独が全層fine-tuningによる改善のほぼ全てを回復した。

seed間のranking安定性も比較的高かった。

| 指標 | Spearman相関 | Kendall相関 |
|---|---:|---:|
| Cross entropy | 0.9907 | 0.9596 |
| NMSE | 0.8834 | 0.7576 |
| R2 | 0.9161 | 0.8182 |
| Variable F1 | 0.8368 | 0.8148 |
| Symbolic recovery | 0.8811 | 0.8495 |

cross entropy、NMSE、R2、symbolic recoveryでは、`decoder_2`、`decoder_3`、`decoder_4`が全seedでtop 3に含まれた。一方、variable F1はpretrainedの時点で高く、改善余地が小さいため、層間差を識別する指標として不安定だった。

**解釈:** fine-tuning効果が全層へ均等に分散しているというより、decoderの中後段へ集中しているという仮説H1を支持する。ただし、これは1種類の合成データ設定と3 seedsに基づく。層が遺伝子制御の「意味」を局在的に保持しているとまでは断定できず、出力token分布を調整しやすいdecoder層だった可能性もある。

### 4.3 Phase 5: selective fine-tuning

Phase 4のvalidation rankingを固定し、testでpretrained、全層、top 1〜3層、middle、bottom、random層を比較した。

| 条件 | Penalized NMSE | Valid rate | Complexity | 記録elapsed秒 |
|---|---:|---:|---:|---:|
| Pretrained | 0.093523 ± 0.015017 | 0.9889 | 18.346 | 158.93 |
| 全層 | 0.014194 ± 0.003711 | 1.0000 | 20.156 | 166.36 |
| Top 1 | 0.014611 ± 0.000930 | 1.0000 | 19.622 | 146.06 |
| Top 2 | 0.015632 ± 0.008361 | 1.0000 | 20.111 | 147.83 |
| Top 3 | 0.015045 ± 0.006008 | 1.0000 | 19.722 | 149.41 |
| Middle 3 | 0.016085 ± 0.007887 | 0.9889 | － | － |
| Bottom 3 | 0.024642 ± 0.008430 | 0.9667 | － | － |
| Random層の平均 | 約0.0174 | 1.0000 | － | － |

Top 3のpaired NMSE差は次のとおりである。負の値はTop 3の方が低NMSEであることを表す。

| 比較 | 平均差 | 95%信頼区間 | 解釈 |
|---|---:|---:|---|
| Top 3 − Pretrained | -0.078479 | [-0.088042, -0.068915] | 明確な改善 |
| Top 3 − 全層 | +0.000850 | [-0.005465, +0.007166] | 実質同等 |
| Top 3 − Bottom 3 | -0.009597 | [-0.014362, -0.004833] | Top 3が良い |
| Top 3 − Random平均 | -0.002401 | [-0.006145, +0.001344] | 差は不明確 |
| Top 3 − Middle 3 | -0.001040 | [-0.006698, +0.004618] | 差は不明確 |

Top 1、Top 2、Top 3は全て、事前定義したNMSE同等性マージン`±0.05`の範囲で全層fine-tuningと同等だった。記録されたpeak memoryはTop 1が約257.8 MB、全層が約620.4 MBで、Top 1は約58%少なかった。一方、elapsedは約146秒対166秒であり、wall-clock短縮は約12%にとどまった。これは学習するparameter以外にdecodeや評価が時間を占めたためと考えられる。

問題単位出力ではTop条件のvalid rateは高かったが、Phase 5 testのsymbolic recoveryは全条件で0だった。したがって、「数値予測が全層と同等」という結論と、「真の式構造を回復した」という結論は分ける必要がある。

**解釈:** 仮説H2のうち、「少数層の更新で全層更新と同程度の数値性能を、少ない更新対象で得られる」は支持された。しかし、Top 3とRandom 3の差は3 seedsでは明確でなく、層rankingそのものの必要性は追加検証が必要である。

### 4.4 Phase 6: TPSR

`noise=0.1`に限定し、pretrainedおよびselective NeSymReSのbeam decodeとTPSRを比較した。

| 方法 | Penalized NMSE | Valid rate | Complexity | 記録elapsed秒 |
|---|---:|---:|---:|---:|
| Pretrained beam | 0.081149 ± 0.020533 | 1.0000 | 18.200 | 157.83 |
| Pretrained + TPSR | 0.057495 ± 0.004922 | 0.9778 | 21.060 | 7063.23 |
| Selective beam | 0.013051 ± 0.003906 | 1.0000 | 19.622 | 127.41 |
| Selective + TPSR | 0.009585 ± 0.001012 | 0.9222 | 25.963 | 6989.26 |

TPSRによるpaired NMSE改善は、pretrainedで`0.023654 ± 0.021241`、selectiveで`0.003465 ± 0.003849`だった。selective条件では信頼区間が0をまたぐため、追加改善は不確実である。さらにselective + TPSRではvalid rateが`0.9222`へ低下し、complexityが増え、near-singularityと外挿失敗も増えた。

summaryへ記録されたelapsedは、selective条件でTPSRがbeamの約55倍だった。MCTS、BFGS、式簡約などCPU側処理が支配的で、L4を使っても比例して短縮されなかった。

**解釈:** TPSRは特に未適応のpretrainedモデルを補う可能性があるが、selective fine-tuning後には限界効用が小さい。GPU_RUN1のbudgetでは、仮説H3の「精度と複雑度のトレードオフに優れる」は支持されなかった。

### 4.5 Phase 7: DREAM4転移

当初は3 seeds、Size10/100、networks 1–5を評価する予定だった。しかしSize100だけで約2,820 target evaluationsとなり、BFGS、式簡約、病的に遅い候補、240秒timeout、Colab切断が総時間を支配した。

実行中にはexit status 137、resume不整合によるstatus 2、runtime切断が起きた。そのためtarget checkpoint、`seed / size / network` shard、60 targetsごとのworker recycling、3分ごとのDrive同期を追加した。

主集計には、結果値を見ずに決めた「全3 seedsで全条件が揃う最大のnetwork prefix」という規則を使い、networks 1–3だけを含めた。networks 4–5の完了済み成果物は保存したが、主集計には混ぜていない。

#### Size10

| 条件 | Penalized NMSE | Valid rate | Complexity |
|---|---:|---:|---:|
| Pretrained + oracle変数 | 0.920720 ± 0.094728 | 0.9778 | 19.504 |
| Selective + correlation変数選択 | 0.890293 ± 0.039117 | 0.7889 | 21.377 |
| Selective + oracle変数 | 0.855371 ± 0.100384 | 0.9333 | 22.093 |

変数選択F1はcorrelationが`0.326808 ± 0.026594`、Lassoが`0.251146 ± 0.071143`、mutual informationが`0.231922 ± 0.071712`、oracle参照が`0.847569`だった。

#### Size100

| 条件 | Penalized NMSE | Valid rate | Complexity |
|---|---:|---:|---:|
| Pretrained + oracle変数 | 0.942570 ± 0.015214 | 0.9988 | 19.316 |
| Selective + correlation変数選択 | 0.900503 ± 0.008218 | 0.8731 | 22.526 |
| Selective + oracle変数 | 0.879365 ± 0.035789 | 0.9077 | 22.433 |

Size100の変数選択F1はcorrelationが`0.046582 ± 0.015382`、Lassoが`0.048448 ± 0.004727`、mutual informationが`0.048263 ± 0.014799`だった。一方、oracle参照は`0.847538`である。

**解釈:** selective fine-tuningはDREAM4でもpretrainedより低いNMSEを示したため、合成データで得た適応の一部は転移した可能性がある。しかしvalid rateは低下し、式は複雑になった。Size100では経験的な変数選択F1が約0.05まで崩壊し、oracleとの差が大きい。高次元GRNでは、数式生成モデルの改善だけでなく候補制御因子選択が主要課題である。

この結果はnetworks 1–3に限られる。DREAM4全5 networksの結果、またはネットワーク一般に対する優位性として報告してはならない。

### 4.6 Phase 8: ヒト時系列LODO

4 donorsを用い、3 donorsで式を学習して残り1 donorで評価するleave-one-donor-outを、3 seeds、10 target genesで実行した。NeSymReSはColab L4、PySRはローカルCPUで実行した。

Phase 8では同じ学習データからin-donor用とholdout用に別々の式を生成していた処理を修正し、式を1回だけ生成して両方で評価した。この変更により計算量をほぼ半減し、比較対象の式も一致した。NeSymReS部分は約3分で完了した。

| 方法 | In-donor NMSE | Holdout NMSE | Generalization gap | Holdout valid rate |
|---|---:|---:|---:|---:|
| Pretrained beam | 0.624095 ± 0.141108 | 0.886691 ± 0.163674 | 0.262596 ± 0.037454 | 0.9917 |
| Selective beam | 0.237847 ± 0.113438 | 0.514499 ± 0.192718 | 0.276652 ± 0.296772 | 1.0000 |
| PySR | 0.010994 ± 0.003032 | 0.241674 ± 0.191592 | 0.230680 ± 0.188562 | 0.9583 |

全120 holdout recordsで確認すると、pretrainedは119件valid、`tan`を含む式が63件、除算を含む式が66件だった。selectiveは120件validだが、`tan`が51件、除算が97件あり、CXCL8ではNMSEが約959となる破局的な式もあった。PySRは115件valid、`tan`は0件、除算は93件で、5件は30秒process timeoutだった。

PySRのholdout NMSEが最も低かったが、その95%信頼区間はselective条件と一部重なる。3 seedsだけから形式的な優越性を断定することはできない。またPySRではseed 0・holdout 1だけが15秒search timeout導入前に完了しており、残りfoldとbudgetが厳密には同一でない。PySRの`random_state`もmultithread実行では完全な決定性を保証しないという警告が出ていた。

さらに、PySRは`+`、`-`、`*`、`/`、`square`の制限された演算子集合を使った一方、NeSymReSは三角関数を含む語彙を生成できた。探索空間、iteration、wall-clock、候補評価数が方法間で一致していないため、これは同一budgetでの厳密なアルゴリズム比較ではない。

**解釈:** selective fine-tuningはヒトdonor間転移でもpretrainedを改善した。一方、この小規模条件ではPySRがより低いNMSEを示し、仮説H4と整合する。ただしGSE112372は5時点・4 donorsであり、有限差分targetは真のODE微分ではない。得られた式を真の制御ODE、因果関係、生物学的機構と呼ぶことはできない。

### 4.7 Phase 9: 検証とarchive

最終runに対してvalidationを実行し、次を確認した。

- `manifest.status`: `complete`
- `validation.status`: `validated`
- JSON files: 82
- 問題単位group: 312
- 問題単位record: 6,831
- equation schema: `v1`
- Phase 8 aggregation: PySRを統合済み

archiveは次の場所へ作成した。

```text
/content/drive/MyDrive/LTSR_colab/archives/colab_reduced_20260729_03.tar.gz
```

SHA256:

```text
e0414a549f87bf2fadb19abbc77bd6b294362c75e94d79fddcc50dcd1646c5eb
```

ローカルではarchive、SHA256ファイル、展開済みコピーを`results/GPU_RUN1_drive/`へ保存した。このdirectoryは大容量成果物を含むため、`.gitkeep`以外をGit管理外としている。

## 5. 研究質問と仮説への回答

### 5.1 研究質問

| 研究質問 | GPU_RUN1からの回答 | 判定 |
|---|---|---|
| RQ1: Pretrained NeSymReSはfine-tuningなしでどの程度復元できるか | 数値近似はある程度可能だが、fine-tuning後より大幅に悪く、Phase 5の構造回復は0だった | 限定的能力 |
| RQ2: fine-tuning効果は特定層へ集中するか | 合成validationでは`decoder_2`〜`decoder_4`へ強く集中 | 支持 |
| RQ3: 少数層は全層fine-tuningに匹敵するか | Top 1〜3がNMSE同等性マージン内で全層と同等 | 数値性能では支持 |
| RQ4: TPSRで精度、単純性、外挿は改善するか | NMSEは改善したが、selective後の差は小さく、複雑度・validity・時間が悪化 | 総合的には未支持 |
| RQ5: NeSymReS/TPSRとPySRの相対性能はどうか | ヒトLODOではPySRが低NMSE。ただしbudgetと演算子が不一致 | 暫定的にPySR優位 |
| RQ6: 層rankingは別GRN・実データへ転移するか | selective FTの数値改善はDREAM4とヒトLODOでも観測。ただし高次元選択と構造回復は弱い | 部分的支持 |

### 5.2 仮説

| 仮説 | 判定 | 根拠 |
|---|---|---|
| H1: fine-tuning効果は一部層へ集中する | 支持 | `decoder_3`が全層のNMSE改善をほぼ再現し、decoder中後段rankingがseed間で安定 |
| H2: 上位少数層で全層と同等の性能を少ない更新parameterで得る | 数値性能では支持 | Top 1〜3が全層と同等、Top 1のpeak memoryは全層より約58%少ない |
| H3: NeSymReS + TPSRが精度と複雑度のトレードオフに優れる | 未支持 | NMSE改善に対し約55倍のelapsed、複雑度増、valid rate低下 |
| H4: 少変数ではNeSymReS/TPSR、高次元ではPySRまたは変数選択 + PySRが優位 | 未確定 | Size100で変数選択F1は崩壊したが、高次元条件でPySRを直接比較していない。ヒト小規模LODOではPySRが低NMSEだったものの、仮説の変数数依存性を直接検証していない |

## 6. 学術的考察

### 6.1 層寄与の局在は何を意味するか

Phase 4と5は、100M規模モデル全体を更新しなくても、少数decoder層の更新でGRN式への数値適応が可能であることを示した。これはparameter-efficient fine-tuningの研究上重要な結果である。特に、Top 1でも全層と同程度だったことは、更新parameter、optimizer state、peak memoryを減らせる可能性を示す。

一方、Top 3とRandom 3の差が明確でなかったため、現状の結果だけでは「精密な層rankingが不可欠」とはいえない。複数のdecoder層が冗長な適応能力を持ち、どの3層を選んでも一定の改善が得られる可能性がある。GPU_RUN2では候補層を増やすだけでなく、random-set反復数を増やして、rankingの付加価値をより高い検出力で評価する必要がある。

### 6.2 数値性能と構造回復の乖離

Phase 5ではNMSEが約0.014まで下がった一方、testのsymbolic recoveryは0だった。Phase 8でも、低NMSE式に多数の除算や複雑な代替表現が含まれた。この乖離は、有限区間内の近似問題としては成功しても、真の数式探索としては成功していないことを示す。

したがって主結果をNMSEだけで順位付けすると、研究目的である「人間が読める遺伝子制御方程式の発見」を過大評価する危険がある。variable F1、skeleton equivalence、complexity、valid rate、外挿、特異点を同格の主指標として扱う必要がある。

### 6.3 TPSRの役割

TPSRは未適応のpretrainedモデルを探索で補正する効果があったが、selective fine-tuning後の追加改善は小さかった。これは、fine-tuningと探索が同じ誤差成分の一部を減らし、相補性が限定的だった可能性を示す。

またTPSRの計算はGPU推論だけでなくCPU上のMCTS、BFGS、式評価に依存する。GPUを高性能化しても総時間が比例して短くならない。GPU_RUN2ではTPSRを当然に全条件へ適用せず、validation上の小規模profilingで追加利益が計算費用を上回る条件だけに限定する方が妥当である。

### 6.4 高次元GRNの本質的な課題

DREAM4 Size100では、oracle変数を与えたselectiveモデルがcorrelation選択より良かった一方、correlation、Lasso、mutual informationのF1はいずれも約0.05だった。これは数式生成以前に候補制御因子を絞る段階が失敗していることを示す。

高次元化に対して単にTransformerを大きくする、beamを増やす、timeoutを延ばすだけでは、入力変数選択の組合せ爆発を解消できない。事前知識、安定性選択、時系列因果候補、sparse group penalty、graph priorなどを使い、候補集合のrecallを保ったまま縮小する手続きが必要である。

### 6.5 ヒトデータでPySRが良かった理由

ヒトLODOはtarget数と観測点が少なく、許可演算子も限定されていた。この条件では、大規模事前学習モデルよりも、小さい探索空間を直接最適化するPySRが有利だった可能性がある。PySRは`tan`を生成せず、NeSymReSより単純な演算子集合で探索したことも有利に働き得る。

ただしPySRの低いNMSEも、真の制御ODEや因果関係を保証しない。in-donorからholdoutへのgapは残り、除算を含む式も多い。観測時点が少ないため有限差分誤差の影響も大きい。現時点では「小規模なLODO予測でPySRが有望」とだけ結論し、機構発見の優位性は主張しない。

### 6.6 計算量爆発から得た方法論的知見

Phase 7の長時間化は、L4の演算性能不足だけが原因ではない。target数、条件数、BFGS、SymPy簡約、病的候補の長いtail、240秒timeout、network完了まで成果物が確定しない保存粒度が積み重なった結果である。

特にtimeoutは、全targetが上限へ近づくと総時間へ線形に効く。GPU_RUN1のSize100を全て240秒で直列実行する理論上限は188時間だった。GPU_RUN2ではtimeoutを事前validationで選び、主実行開始時から15秒へ固定することが最も大きな短縮要因になる。

## 7. 限界と結果を読む際の注意

### 7.1 実験範囲

- `noise=0.1`だけであり、noise robustnessは評価していない。
- seedsは3つだけで、信頼区間の検出力が低い。
- Phase 7主集計はDREAM4 networks 1–3だけで、1–5の完全実行ではない。
- Phase 8は4 donors・5時点・10 target genesのapplication demoである。

### 7.2 計算budgetと方法比較

- NeSymReS、TPSR、PySRで演算子集合、wall-clock、候補評価数が一致していない。
- Phase 8のPySRはseed 0・holdout 1だけ15秒search timeout導入前の結果である。
- PySRのmultithread探索は完全にdeterministicではない。
- Phase 7は240秒timeout、Phase 8は30秒timeoutで、Phase横断の単純な時間比較はできない。
- 条件別`elapsed_sec`はsummaryが記録した処理時間であり、Colab切断、再起動、再接続を含む研究全体のwall-clockではない。

### 7.3 継続run

GPU_RUN1は複数commitと複数runをprovenance付きで継承した。成果物の由来は記録されているが、単一commit・単一環境で最初から最後まで再実行した理想的なpaper runより再現性が複雑である。

### 7.4 指標と安全性

- symbolic recoveryが弱く、低NMSEを機構回復と解釈できない。
- `tan`や危険な除算を含む式が多い。
- 現在のnear-singularity診断だけでは、`tan`の極や未観測領域の発散を十分に検出できない。
- valid式でもNMSEが極端に大きい例があり、valid rateだけでは安全性を表せない。
- complexityは演算子の危険度を区別しないため、同じnode数でも生物学的妥当性は異なる。

### 7.5 データと外的妥当性

- DREAM4の有限差分targetは真のODE微分そのものではない。
- ヒトデータは観測時点とdonor数が少なく、微分推定誤差が大きい。
- donor、target gene、networkを母集団として一般化するには標本数が不足する。
- test結果を見て層rankingを変更してはいないが、GPU_RUN1全体の試行錯誤を経たため、GPU_RUN2は設定を事前固定した独立な確認runとして扱う必要がある。

## 8. GPU_RUN2への展望

GPU_RUN2は、GPU_RUN1の不足分を単に埋める再実行ではなく、計算budgetと演算子を事前登録した確認実験とする。

### 8.1 最優先の変更

1. **演算子集合を統一する。** NeSymReS、TPSR、PySRで可能な限り`+`、`-`、`*`、safe division、`square`へ揃える。`sin`、`cos`、`tan`、逆三角関数、危険な入れ子除算を主実行から除外する。真の合成式が表現可能であることを事前検査する。
2. **timeoutを事前に固定する。** validation上で5、10、15、30秒を比較し、testを見ずに主実行を15秒へ固定する。timeoutは失敗として保存し、valid rateとpenalized指標へ反映する。
3. **probingでPhase 4候補を減らす。** 各層のhidden state、validation CE/NMSE、gradient normを軽量に調べ、5〜8層へ候補を絞る。testは候補選択に使わない。
4. **全長時間Phaseをtarget/fold checkpoint化する。** Phase 7だけでなくPhase 4–8へ広げ、worker recyclingとDrive同期を最初から有効にする。
5. **CPU-only処理をローカルへ移す。** PySR、集計、図表、archive、CPU parameter fittingをColab L4から分離する。TPSRもprofilingでGPU利用率が低ければローカルまたは別queueへ移す。

### 8.2 公平な比較設計

- 同じproblem、seed、train/validation/test splitを使う。
- 同じ演算子集合または演算子差を明示したablationを使う。
- wall-clockだけでなく、候補評価数、timeout率、GPU時間、CPU時間を保存する。
- PySRはdeterministic serialにするか、multithreadを複数反復するかを事前に決める。
- random layer条件の反復数を増やし、top rankingの付加価値を検出できるようにする。
- 同じ学習データから式を1回だけ生成し、固定した式をvalidation/test/holdoutで評価する。

### 8.3 Phase 7の完遂

Phase 7は3 seeds × DREAM4 networks 1–5を全て同じ15秒timeoutで再実行する。Size100の約2,820 evaluationsに対する15秒の直列上限は約11.75時間であり、実際には早く終わるtargetもある。2 seeds並列とcheckpoint overheadを含め、8〜14時間程度を初期見積りとする。ただし本実行前にtarget時間分布をprobeし、Colabの残compute unitsと照合してGo/No-Goを判断する。

Phase 7では次を主結果にする。

- 全5 networksを含むseed-level集計
- 変数選択precision、recall、F1
- oracleと経験的selectorのgap
- selector failureを含むpenalized NMSE
- target時間のp50、p90、p95、最大、timeout率
- Size10からSize100へのscaling curve

### 8.4 数式回復と安全性の強化

- exact、skeleton、symbolic equivalenceを主表へ含める。
- raw式、簡約式、変数対応、失敗理由を全problemで保存する。
- `tan`の極、分母の0接近、観測範囲外の発散、負値などを評価する。
- complexityに加え、危険演算子数、最大入れ子深さ、分母marginを保存する。
- NMSEだけで設定を選ばず、validation上の複合的な選択規則を事前定義する。

### 8.5 実データの次段階

- 微分推定法と平滑化法の感度分析を行う。
- donorを独立単位として扱い、target geneだけを独立反復とみなさない。
- 可能ならdonor数または外部datasetを増やす。
- bootstrapまたは階層modelでdonor・gene間変動を分離する。
- 実データ結果は予測的applicationとして報告し、因果ODE回復とは区別する。

### 8.6 GPU_RUN2のGo/No-Go基準

本実行前に最低限、次を満たす必要がある。

- source commitと全設定が固定されている。
- Python 3.10、checkpoint、data checksumが一致する。
- 15秒timeoutとfailure保存が1〜2 targetsで動作する。
- target/fold resumeで完了済み計算が再実行されない。
- operator allowlistがNeSymReS、TPSR、PySRで記録される。
- target実時間probeから総時間とcompute unitsを見積もれる。
- Phase 7全5 networksを完遂できる資源見込みがある。

## 9. 再現性と成果物

### 9.1 最終成果物

```text
results/runs/colab_reduced_20260729_03/
├─ manifest.json
├─ validation.json
├─ continuation.json
├─ phase4_multiseed/
├─ phase5_multiseed/
├─ phase6_noise_multiseed/
├─ phase7_multiseed/
├─ phase8_lodo_multiseed/
├─ phase8_pysr_seed0/
├─ phase8_pysr_seed1/
├─ phase8_pysr_seed2/
└─ reports/
```

問題単位の式と指標はrun archiveに保存されている。集約表だけから再度結論を作るのではなく、代表的な成功例と失敗例、timeout、invalid式を問題単位recordから確認すること。

### 9.2 archiveの検証

ローカルで展開する前後に、次でchecksumを確認できる。

```powershell
Get-FileHash `
  results/GPU_RUN1_drive/colab_reduced_20260729_03.tar.gz `
  -Algorithm SHA256
```

期待値:

```text
E0414A549F87BF2FADB19ABBC77BD6B294362C75E94D79FDDCC50DCD1646C5EB
```

archiveと展開した大容量runはGitへcommitしない。

## 10. 結論

GPU_RUN1は、少数のdecoder層だけを更新するNeSymReS適応が、合成GRNの数値性能では全層fine-tuningに匹敵し、memory使用量を大きく減らせることを示した。この点は本研究の中心仮説を支持する。

しかし、層ranking上位がrandom層より明確に良いとはまだ示せず、真の式構造の回復も達成できていない。TPSRは費用に対する追加利益が小さく、DREAM4 Size100では変数選択が崩壊し、ヒトLODOではPySRがより低いNMSEを示した。さらに、三角関数、除算、特異点、timeout、実行budget不一致が結果解釈を難しくした。

よってGPU_RUN1は最終的な確証実験ではなく、中心仮説の一部を支持し、次のボトルネックを特定した探索的GPU runと位置付けるのが妥当である。GPU_RUN2では、演算子集合、15秒timeout、probing、CPU/GPU分離、全5 DREAM4 networks、target-level時間分布を事前固定し、単一commitから再実行する。その結果によって初めて、少数層fine-tuningの再現性、層rankingの必要性、PySR/TPSRとの公平な相対性能をより強く評価できる。

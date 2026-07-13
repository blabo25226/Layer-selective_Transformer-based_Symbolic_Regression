# 研究計画書案

## 研究題目（仮）

**層選択型Transformerシンボリック回帰による遺伝子制御方程式の推定**

英語題目案：

**Layer-Selective Transformer-Based Symbolic Regression for Gene Regulatory Equation Discovery**

---

# 1. 研究概要

本研究では、遺伝子発現データから遺伝子制御関係を表す数式を推定する。

中心となるモデルには、事前学習済みTransformer型シンボリック回帰モデルである**NeSymReS**を用いる。必要に応じて、Monte Carlo Tree Searchによって数式生成を改善する**TPSR**を組み合わせる。

さらに、論文 *Is One Layer Enough? Training a Single Transformer Layer Can Match Full-Parameter RL Training* が示した「学習効果が一部のTransformer層に集中する可能性」をNeSymReSへ応用する。同論文では、層を一つずつ更新して得られる改善量を測定するlayer contributionが導入され、Qwen系モデルのRL後学習において、中間層を中心とする少数の層だけの更新が全パラメータ更新に匹敵、または上回る場合が報告されている。

本研究では、この現象が、

* LLM以外のTransformer
* 強化学習以外のfine-tuning
* シンボリック回帰
* 遺伝子制御方程式の推定

でも成立するかを検証する。

---

# 2. 研究背景

## 2.1 GRN推定

Gene Regulatory Network、GRNは、転写因子や遺伝子が他の遺伝子の発現を活性化・抑制する関係を表すネットワークである。

従来のGRN推定では、Random Forest、線形モデル、相互情報量、ベイズネットワーク、ニューラルネットワークなどが用いられることが多い。しかし、これらの方法は主に、

[
A\rightarrow B
]

のような制御辺や重要度を推定するものであり、制御関数の形まで明示的に得られない場合が多い。

本研究では、標的遺伝子 (x_i) の変化を、

[
\frac{dx_i}{dt}
===============

f_i(x_{j_1},x_{j_2},\ldots,x_{j_m})
]

と表し、未知関数 (f_i) をシンボリック回帰によって推定する。

例えば、

[
\frac{dx_i}{dt}
===============

\frac{\alpha x_j^n}{K^n+x_j^n}
-\beta x_i
]

のような式が得られれば、

* (x_j) が (x_i) を活性化する
* 制御が飽和型である
* 協同性が存在する
* (x_i) に分解項がある

といった生物学的解釈が可能になる。

---

## 2.2 シンボリック回帰

シンボリック回帰は、データから人間が読める数式を探索する方法である。

通常のニューラルネットワークが、

[
\hat y=\mathrm{NN}(x)
]

というブラックボックスを学習するのに対し、シンボリック回帰は、

[
\hat y=2.1x^2-\frac{0.8z}{1+z}
]

のような明示的な式を生成する。

PySRは進化的探索を利用する実用的なシンボリック回帰ライブラリであり、複数集団による進化、式の簡約、定数最適化などを組み合わせている。Pythonから利用でき、Julia製の高性能バックエンドを持つ。([arXiv][1])

---

## 2.3 NeSymReS

NeSymReSは、人工的に生成した大量の数式と数値点集合を使ってTransformerを事前学習し、新しい数値データから数式を生成するモデルである。従来の探索を毎回最初から行う方法とは異なり、過去の数式学習経験を新しい問題に利用する。([Proceedings of Machine Learning Research][2])

公式実装と事前学習済みモデルは公開されているため、ゼロからモデルを事前学習する必要はない。([GitHub][3])

ただし、生物データはNeSymReSの事前学習に用いられた人工数式データと性質が異なる。そのため、遺伝子制御方程式へのfine-tuningが必要になる可能性がある。

---

## 2.4 TPSR

TPSRは、事前学習済みTransformerによる数式生成にMonte Carlo Tree Searchを組み込む手法である。

通常のbeam searchでは、Transformerが高い確率を付けた数式候補を中心に生成する。TPSRは先読み探索を行い、

* データへの適合度
* 数式の複雑さ
* 外挿性能
* 非微分可能な評価指標

を数式探索へ反映できる。([arXiv][4])

公式コードも公開されている。([GitHub][5])

本研究では、TPSRをNeSymReSの代替モデルではなく、**NeSymReSが生成する数式を改善する推論時探索法**として扱う。

---

# 3. 先行研究と本研究の位置づけ

## 3.1 Transformer型シンボリック回帰

NeSymReSは、大規模な人工数式による事前学習を用いたTransformer型シンボリック回帰である。([Proceedings of Machine Learning Research][2])

TPSRは、このような事前学習済み数式生成モデルにMCTSを組み込み、精度と複雑性のトレードオフ、外挿、ノイズ耐性を改善する方法である。([OpenReview][6])

---

## 3.2 GRNとシンボリック回帰

GRNにシンボリック回帰を使う研究は、少数ながら存在する。

### LogicGep

LogicGepは、時系列転写データからBoolean networkを推定する方法である。各遺伝子の更新規則をシンボリック回帰問題として扱い、gene expression programmingと多目的最適化を用いて論理式を推定する。([OUP Academic][7])

例えば、

[
x_i(t+1)=x_j(t)\land \neg x_k(t)
]

のような規則を推定する。

ただし、発現量をBoolean値へ変換するため、連続的な濃度依存性を直接表現できない。

### LogicSR

LogicSRは、single-cell遺伝子発現からGRNを推定するため、Boolean論理モデルとシンボリック回帰を統合した手法である。既知の生物学的priorを利用し、MCTSによって予測性能、式の複雑さ、生物学的妥当性を最適化する。([OUP Academic][8])

LogicSRは本研究に非常に近いが、

* LogicSRはBoolean論理式を中心とする
* 本研究は連続値方程式またはODEを対象とする
* LogicSRはNeSymReSを使わない
* Transformerの層寄与は扱わない

という違いがある。

### ScaleSR

ScaleSRは、多変数問題を制御変数によって複数の低次元問題へ分解するニューラルシンボリック回帰であり、合成データ上のgenetic toggle switchとrepressilatorにも適用されている。([arXiv][9])

これは、シンボリック回帰で遺伝子制御方程式を扱うこと自体に実現可能性があることを示している。

---

## 3.3 層選択学習

*Is One Layer Enough?* は、Transformerの層ごとにRL学習を行い、全パラメータ学習による改善の何割を一層だけで回復できるかを調べた。七つのQwen系モデルで、中間層付近に高寄与層が集中し、一層または少数層のみの学習が全層学習に匹敵する場合が報告されている。([arXiv][10])

ただし、この結果はLLMのRL後学習に関するものであり、NeSymReSやシンボリック回帰ではまだ確認されていない。

---

## 3.4 本研究の新規性

既存研究には、

* Transformer型シンボリック回帰
* MCTSによる数式探索
* GRNに対するシンボリック回帰
* Transformerの層選択学習

がそれぞれ存在する。

しかし、

> **Transformer型シンボリック回帰を遺伝子制御方程式へfine-tuningし、層ごとの寄与を測定して、少数層のみで効率的な適応を実現する研究**

は、現時点では明確な先行例がほとんど確認できない。

新規性は、NeSymReSやTPSRそのものを新規に提案することではなく、次の組合せにある。

1. GRN方程式推定へのNeSymReSの適用
2. NeSymReSの層寄与分析
3. 高寄与層だけを用いたfine-tuning
4. TPSRによる推論改善
5. PySRおよび既存GRN推定法との比較

---

# 4. 研究目的

本研究の主目的は、以下の三点である。

## 目的1

NeSymReSが、合成および実データ由来の遺伝子制御関係を表す数式を推定できるか検証する。

## 目的2

NeSymReSの遺伝子制御方程式への適応効果が、特定のTransformer層に集中しているか調べる。

## 目的3

高寄与層のみをfine-tuningすることで、全パラメータfine-tuningより少ない計算量・学習パラメータ数で、同等以上の方程式推定性能を実現できるか検証する。

---

# 5. 研究質問

中心的な研究質問は次の通りである。

### RQ1

事前学習済みNeSymReSは、遺伝子制御方程式をfine-tuningなしでどの程度復元できるか。

### RQ2

遺伝子制御方程式によるfine-tuningは、NeSymReSの特定層に集中するか。

### RQ3

単一層または少数層のみのfine-tuningは、全層fine-tuningに匹敵するか。

### RQ4

NeSymReSにTPSRを組み合わせることで、数式の精度、単純性、外挿性能は向上するか。

### RQ5

NeSymReS／TPSRは、PySRに対してどのような条件で優位または劣位になるか。

### RQ6

合成データで有効だった層ランキングは、別のGRN、ノイズ条件、実データへ転移するか。

---

# 6. 仮説

## 仮説H1

NeSymReSのfine-tuning効果は全層へ均等には分布せず、encoderまたはdecoderの一部の層へ集中する。

## 仮説H2

高寄与層のみをfine-tuningすることで、全層fine-tuningと同程度の予測性能を、より少ない更新パラメータ数で実現できる。

## 仮説H3

NeSymReS単独より、NeSymReS＋TPSRの方が、ノイズ条件下における数式精度と複雑度のトレードオフに優れる。

## 仮説H4

少数の候補制御因子に絞った問題ではNeSymReS／TPSRが有効だが、変数数が増えるとPySRまたは変数選択＋PySRが優位になる。

---

# 7. 研究対象

## 7.1 初期段階：小規模合成GRN

最初からヒト全遺伝子を扱わず、真の方程式が既知の小規模系を使用する。

候補：

* 自己活性化・自己抑制モデル
* 二遺伝子toggle switch
* repressilator
* feed-forward loop
* Hill関数型制御
* S-system
* 2〜5遺伝子の人工ODE

合成データでは、真の式、真の制御辺、真の係数が既知であるため、方法の正しさを厳密に評価できる。

---

## 7.2 中間段階：DREAM／GeneNetWeaver

DREAM4 In Silico Network Challengeは、シミュレーションによる遺伝子ネットワークからsteady-state、time-series、ノックアウト、ノックダウンなどのデータを生成したGRN推定ベンチマークである。gold standardのネットワーク構造も提供されている。([Synapse][11])

GeneNetWeaverは、DREAM3、DREAM4、DREAM5などのネットワーク推定ベンチマーク生成に使われたオープンソース環境である。([OUP Academic][12])

ただし、100遺伝子を同時にNeSymReSへ入れるのではなく、標的遺伝子ごとに候補regulatorを事前選択し、2〜5変数程度の局所問題へ変換する。

---

## 7.3 最終段階：ヒトデータ

最終的には、公開されたヒト遺伝子発現データのうち、次の条件を満たす小規模なデータを選ぶ。

* 時系列またはpseudotimeを持つ
* 細胞分化や刺激応答を含む
* 対象細胞型が限定されている
* 既知の主要転写因子が存在する
* 既存のGRN研究と比較可能
* processed expression matrixが入手できる

ヒトデータでは真の方程式が不明であるため、「真の式を発見した」とは主張せず、

* 未知データへの予測性能
* 複数データセット間の再現性
* 既知TF–target関係との整合性
* 摂動データとの整合性
* 生物学的に不自然な式が含まれないか

によって評価する。

---

# 8. 研究手順

## Phase 0：文献・環境調査

### 目的

既存コードが現在のPython、PyTorch、Colab環境で動作するか確認する。

### 作業

1. GitHubリポジトリを作成
2. NeSymReS公式コードを取得
3. 事前学習済みcheckpointを取得
4. TPSR公式コードを取得
5. PySRを導入
6. Colab用依存関係を固定
7. 公式exampleを再現

### 合格基準

* NeSymReSで既知のサンプル式を出力できる
* TPSRを最低1データセットで実行できる
* PySRで同じデータを解析できる
* 再実行可能なColab Notebookが作成されている

この段階では、*Is One Layer Enough?* のQwen実験を完全再現しない。

---

## Phase 1：合成方程式データ生成

### 目的

GRNに典型的な方程式の学習・評価データを作る。

### 方程式例

活性化：

[
\frac{dx}{dt}
=============

\frac{\alpha y^n}{K^n+y^n}
-\beta x
]

抑制：

[
\frac{dx}{dt}
=============

\frac{\alpha K^n}{K^n+y^n}
-\beta x
]

toggle switch：

[
\frac{dx}{dt}
=============

\frac{\alpha_1}{1+y^{n_1}}-\beta_1x
]

[
\frac{dy}{dt}
=============

\frac{\alpha_2}{1+x^{n_2}}-\beta_2y
]

repressilator：

[
\frac{dx_1}{dt}
===============

\frac{\alpha}{1+x_3^n}-\beta x_1
]

[
\frac{dx_2}{dt}
===============

\frac{\alpha}{1+x_1^n}-\beta x_2
]

[
\frac{dx_3}{dt}
===============

\frac{\alpha}{1+x_2^n}-\beta x_3
]

### 実験条件

* データ点数
* 時間間隔
* 初期値
* パラメータ値
* ガウスノイズ
* 欠測
* 外れ値
* 数値微分誤差

を変化させる。

### データ分割

方程式そのものがtrain/test間で重ならないように分割する。

単に同じ式の異なる係数をtestに置くと、式の暗記を評価してしまうため、

* 式構造による分割
* パラメータ範囲による分割
* ネットワークmotifによる分割

を検討する。

---

## Phase 2：baseline評価

### 比較対象

1. PySR
2. 事前学習済みNeSymReS
3. NeSymReS＋beam search
4. NeSymReS＋TPSR

### 評価項目

* 数式の完全一致
* 数学的等価性
* NMSE
* (R^2)
* 外挿性能
* 式の複雑度
* 正しい変数の回復率
* 正しい制御符号の回復率
* 実行時間
* CPU／GPU使用量

この段階でNeSymReSがPySRより大幅に劣る場合でも、研究は中止しない。次のfine-tuningによって改善可能かを検証する。

---

## Phase 3：NeSymReSのfine-tuning実装

### 比較条件

* 学習なし
* 出力headのみ
* encoder各層を一層ずつ
* decoder各層を一層ずつ
* 全Transformer層
* 全パラメータ
* LoRA
* 中央層のみ
* ランダムに選択した同数層

対象層以外は、

```python
parameter.requires_grad = False
```

として凍結する。

ただし、勾配計算はモデル全体を通す。これは元論文の単一層学習と同様の設定である。

---

## Phase 4：層寄与度の測定

評価性能が大きいほどよい指標の場合、層 (k) の寄与を、

[
C(k)
====

\frac{S_k-S_{\mathrm{base}}}
{S_{\mathrm{full}}-S_{\mathrm{base}}}
]

とする。

* (S_{\mathrm{base}})：fine-tuning前
* (S_k)：第 (k) 層だけfine-tuning
* (S_{\mathrm{full}})：全層fine-tuning

損失が小さいほどよい場合は、

[
C_{\mathrm{loss}}(k)
====================

\frac{L_{\mathrm{base}}-L_k}
{L_{\mathrm{base}}-L_{\mathrm{full}}}
]

を使う。

### 注意

数式回復では、性能を単一指標だけにしない。

例えば、

[
S
=

-\mathrm{NMSE}
-\lambda_1\mathrm{Complexity}
+\lambda_2\mathrm{SymbolicRecovery}
]

のような複合指標を使う方法もあるが、重みづけによる恣意性が生じる。

そのため主分析では、

* 予測精度に基づくlayer contribution
* 数式回復率に基づくlayer contribution
* 変数回復率に基づくlayer contribution

を別々に報告する。

---

## Phase 5：高寄与層だけの学習

層ランキングを得た後、次を比較する。

* 上位1層
* 上位2層
* 上位3層
* 上位 (k) 層
* 中央 (k) 層
* ランダム (k) 層
* 下位 (k) 層
* 全層

主に測るものは、

* 精度
* 数式回復率
* 学習パラメータ数
* GPUメモリ
* 学習時間
* 過学習
* 別GRNへの転移性

である。

---

## Phase 6：TPSRの追加

TPSRは層寄与分析とは分けて評価する。

次の2×2実験を行う。

| Fine-tuning | 推論     |
| ----------- | ------ |
| なし          | 通常デコード |
| なし          | TPSR   |
| 高寄与層のみ      | 通常デコード |
| 高寄与層のみ      | TPSR   |

これにより、

* fine-tuningによる改善
* MCTS探索による改善
* 両者の相互作用

を分離できる。

---

## Phase 7：DREAM／GeneNetWeaverへの適用

各target geneについて、候補regulatorを事前選択する。

候補選択法：

* 真のnetworkの近傍を使うoracle条件
* 相関
* 相互情報量
* LASSO
* Random Forest／GENIE3
* 既知TFリスト

最初はoracle条件を使い、「候補変数が正しいときに式を回復できるか」を評価する。

次に実際の変数選択を導入し、

[
\text{変数選択誤差}
+
\text{数式探索誤差}
]

を含む現実的な設定へ進む。

---

## Phase 8：ヒトデータへの適用

実データでは、一つのtarget geneごとに候補TFを数個へ制限する。

解析例：

[
\frac{dx_{\mathrm{target}}}{dt}
===============================

f(x_{\mathrm{TF1}},x_{\mathrm{TF2}},x_{\mathrm{target}})
]

時間微分は、

* 実時系列
* smoothing後の有限差分
* Gaussian process
* spline
* RNA velocity
* pseudotime上の局所回帰

などから推定する。

ただし、RNA velocityやpseudotimeは真の時間微分ではないため、結果の解釈には制限を付ける。

---

# 9. 評価指標

## 9.1 方程式評価

### 予測精度

[
\mathrm{NMSE}
=============

\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}
]

### 数式回復

* exact match
* SymPyによる簡約後一致
* symbolic equivalence
* tree edit distance

### 外挿性能

学習範囲外の初期値や発現範囲で評価する。

### 数式複雑度

* トークン数
* 演算子数
* 木の深さ
* 非線形演算子数

---

## 9.2 GRN評価

* regulator precision
* regulator recall
* F1 score
* AUROC
* AUPRC
* 活性化／抑制符号の正解率
* network edge recovery

DREAM4ではgold standardネットワークがあるため、ネットワーク構造の定量評価が可能である。([GNW][13])

---

## 9.3 計算効率

* 学習対象パラメータ数
* GPUメモリ
* fine-tuning時間
* 推論時間
* TPSR rollout数
* PySRの探索時間
* Colabセッション内で完了可能か

---

## 9.4 統計的評価

各条件を可能な範囲で複数seed実行する。

最低でも、

* 合成データ生成seed
* モデルfine-tuning seed
* 数式探索seed

を分離する。

結果は平均値だけでなく、標準偏差または信頼区間を報告する。

---

# 10. 比較実験

最低限、次の比較を行う。

| 方法                     | 目的                    |
| ---------------------- | --------------------- |
| PySR                   | 強い実用的baseline         |
| NeSymReS zero-shot     | 事前学習だけの性能             |
| NeSymReS全層fine-tuning  | 標準fine-tuning         |
| NeSymReS単一層fine-tuning | 層集中仮説                 |
| NeSymReS上位層fine-tuning | 提案手法                  |
| NeSymReS中央層fine-tuning | profiling不要のheuristic |
| NeSymReSランダム層          | 層選択の対照                |
| NeSymReS＋TPSR          | 推論探索の効果               |
| 上位層NeSymReS＋TPSR       | 最終候補                  |

可能であれば、

* LASSO
* Random Forest
* GENIE3
* 線形ODE
* 既知Hill関数モデル

も予測またはGRN baselineとして追加する。

---

# 11. 予想される成果

## 成功ケースA

少数層fine-tuningが全層fine-tuningと同等以上になる。

この場合、

* 学習コスト削減
* Colabでも実行可能
* 生物分野への層選択学習の応用
* Transformer SRの内部構造理解

を成果として主張できる。

## 成功ケースB

中間層ではなくencoderまたはdecoder後段が高寄与になる。

これは元論文と異なるが、数値理解と数式生成で層機能が異なることを示す新しい結果になる。

## 成功ケースC

NeSymReSがPySRに勝たないが、特定条件では高速または安定になる。

例えば、

* 多数の類似方程式を連続推定する場合
* fine-tuning後の推論速度
* 特定GRN familyへの特化
* 外挿性能

で優位性が示せれば研究成果になる。

## 否定的結果

どの単一層も全層fine-tuningに近づかない場合でも、

> LLMのRLで観察された層集中は、Transformer型シンボリック回帰へ一般化しない

という検証結果になる。

ただし、その場合は実験条件、モデル規模、データ量、fine-tuning方法の不足と区別する必要がある。

---

# 12. 主なリスクと対策

## リスク1：NeSymReSのコードが動かない

### 対策

* 動作するPython／PyTorchバージョンを固定
* DockerまたはColab環境を記録
* 元コードを直接全面改変せずadapterを作る
* 最小限のinferenceから確認する

---

## リスク2：fine-tuning用APIが整っていない

### 対策

* モデル構造を調査
* encoderとdecoderのblock一覧を抽出
* `requires_grad`を制御するutilityを実装
* 最初は出力headまたは一層だけでgradientが流れるか確認する

---

## リスク3：ColabのGPU時間不足

### 対策

* 事前学習は行わない
* 合成データを小規模化
* 全層スキャンは1回、上位候補のみ複数seed
* checkpointをGoogle Driveへ保存
* mixed precision
* gradient accumulation
* 層ごとの短いpilot run
* encoder／decoderの代表層から先に調べる

---

## リスク4：実データで方程式が回復できない

### 対策

研究の主検証を合成GRNおよびDREAMに置く。ヒトデータは応用例とする。

---

## リスク5：変数数が多すぎる

### 対策

変数選択を前段へ置き、1 target geneあたり2〜5候補regulatorへ限定する。

---

## リスク6：高精度だが生物学的に不自然な式

### 対策

* 式の複雑度制約
* 正負制約
* 既知TF prior
* 分解項
* 非負性
* 外挿範囲での安定性
* 独立データでの検証

を導入する。

---

# 13. GitHubリポジトリ案

```text
layer-aware-symbolic-grn/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   └── colab.txt
├── configs/
│   ├── synthetic/
│   ├── dream/
│   └── human/
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── 01_nesymres_inference.ipynb
│   ├── 02_pysr_baseline.ipynb
│   ├── 03_tpsr_inference.ipynb
│   ├── 04_single_layer_finetuning.ipynb
│   └── 05_dream_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── synthetic_grn.py
│   │   ├── dream.py
│   │   └── human.py
│   ├── models/
│   │   ├── nesymres_adapter.py
│   │   ├── tpsr_adapter.py
│   │   └── layer_selector.py
│   ├── training/
│   │   ├── full_finetune.py
│   │   ├── single_layer.py
│   │   └── selective_layers.py
│   ├── evaluation/
│   │   ├── equation_metrics.py
│   │   ├── grn_metrics.py
│   │   └── layer_contribution.py
│   └── baselines/
│       └── pysr_baseline.py
├── tests/
├── scripts/
└── results/
```

`results/`には巨大なcheckpointを直接置かず、Git LFS、GitHub Releases、Google Driveなどを使う。

---

# 14. 開発の進め方

開発では、一度に大規模な実装を依頼するのではなく、issue単位で進める。

初期issue例：

1. NeSymReS公式exampleの再現
2. Colab用requirementsの作成
3. モデルのencoder／decoder層一覧の取得
4. 対象層以外をfreezeする関数
5. trainable parameter数の検査
6. 合成Hill方程式データ生成
7. PySR baseline
8. 単一層fine-tuning
9. layer contribution計算
10. TPSR adapter

各issueについて、

* 目的
* 入力
* 出力
* acceptance criteria
* test
* 実行例

を明記する。

---

# 15. 暫定スケジュール

## 第1期：1〜2週間

* 文献整理
* GitHub作成
* Colab環境構築
* NeSymReS、TPSR、PySRの動作確認

## 第2期：2〜4週間

* 合成GRNデータ生成
* PySR／NeSymReS baseline
* 評価指標実装

## 第3期：3〜5週間

* 単一層fine-tuning
* layer contribution分析
* 高寄与層選択

## 第4期：2〜4週間

* TPSR統合
* ノイズ・外挿実験
* ablation study

## 第5期：3〜5週間

* DREAM／GeneNetWeaver
* regulator事前選択
* network評価

## 第6期：余力に応じて

* ヒトデータ
* 論文・卒論執筆
* コード整理
* 再現Notebook公開

合計は、最小構成で約3か月、DREAMとヒトデータまで含めると4〜6か月程度を想定する。

---

# 16. 最小達成目標と発展目標

## 最小達成目標

> 合成GRN方程式に対して、NeSymReSの単一層fine-tuningと全層fine-tuningを比較し、層寄与度を測定する。

これだけでも研究の中心仮説を検証できる。

## 標準目標

> 合成GRNおよびDREAMデータで、PySR、NeSymReS、層選択NeSymReS、TPSRを比較する。

## 発展目標

> ヒトsingle-cellまたは時系列発現データから、既知の転写因子priorを用いて候補制御方程式を推定する。

ヒトデータは発展目標であり、研究成立の必須条件にはしない。

---

# 17. 現時点の主要参考文献

## 層選択学習

1. Zhang, Z. et al.
   **Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training.**
   arXiv:2607.01232, 2026.
   単一層学習、layer contribution、高寄与層選択の中心文献。([arXiv][10])

## Transformer型シンボリック回帰

2. Biggio, L. et al.
   **Neural Symbolic Regression that Scales.**
   ICML 2021.
   NeSymReSの提案論文。([Proceedings of Machine Learning Research][2])

3. Shojaee, P. et al.
   **Transformer-based Planning for Symbolic Regression.**
   NeurIPS 2023.
   TransformerとMCTSを組み合わせたTPSRの提案論文。([OpenReview][6])

4. Chu, X. et al.
   **Scalable Neural Symbolic Regression Using Control Variables.**
   2023.
   多変数SRとtoggle switch、repressilatorへの応用。([arXiv][9])

## PySR

5. Cranmer, M.
   **Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.**
   arXiv:2305.01582, 2023.
   PySRのアルゴリズム、設計、科学応用の中心文献。([arXiv][1])

## GRNとシンボリック回帰

6. Zhang, D. et al.
   **LogicGep: Boolean Networks Inference Using Symbolic Regression from Time-Series Transcriptomic Profiling Data.**
   Briefings in Bioinformatics, 2024.
   時系列発現からBoolean規則を推定するSR研究。([OUP Academic][7])

7. Zhang, D. et al.
   **LogicSR: Prior-Guided Symbolic Regression for Gene Regulatory Network Inference from Single-Cell Transcriptomics Data.**
   Briefings in Bioinformatics, 2025.
   single-cell、Boolean SR、MCTS、生物priorを統合する重要な先行研究。([OUP Academic][8])

## GRNベンチマーク

8. Marbach, D. et al.
   **The DREAM4 In Silico Network Challenge.**
   GRN逆推定用の合成ネットワーク、時系列、摂動、gold standardを提供する。([GNW][13])

9. Schaffter, T., Marbach, D., Floreano, D.
   **GeneNetWeaver: In Silico Benchmark Generation and Performance Profiling of Network Inference Methods.**
   Bioinformatics, 2011.
   DREAM系データ生成に使われたGRNシミュレーション環境。([OUP Academic][12])

---

# 18. 計画書としての最終的な研究目的文

> 本研究は、事前学習済みTransformer型シンボリック回帰モデルNeSymReSを遺伝子制御方程式へ適応させ、そのfine-tuning効果が特定のTransformer層へ集中するかを明らかにすることを目的とする。各層を独立にfine-tuningして層寄与度を測定し、高寄与層のみを更新する手法を全層fine-tuning、PySR、TPSRと比較する。まず真の方程式が既知の合成GRNで検証し、次にDREAM／GeneNetWeaverデータ、最終的に小規模なヒト遺伝子発現データへの応用を試みる。これにより、計算資源を抑えながら、精度・解釈可能性・外挿性能を備えた遺伝子制御方程式推定法の構築を目指す。

[1]: https://arxiv.org/abs/2305.01582?utm_source=chatgpt.com "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"
[2]: https://proceedings.mlr.press/v139/biggio21a/biggio21a.pdf?utm_source=chatgpt.com "Neural Symbolic Regression that Scales"
[3]: https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales?utm_source=chatgpt.com "Source code and Dataset creation for the paper \"Neural ..."
[4]: https://arxiv.org/abs/2303.06833?utm_source=chatgpt.com "Transformer-based Planning for Symbolic Regression"
[5]: https://github.com/deep-symbolic-mathematics/tpsr?utm_source=chatgpt.com "deep-symbolic-mathematics/TPSR: [NeurIPS 2023] This is ..."
[6]: https://openreview.net/forum?id=0rVXQEeFEL&utm_source=chatgpt.com "Transformer-based Planning for Symbolic Regression"
[7]: https://academic.oup.com/bib/article/25/4/bbae286/7694187?utm_source=chatgpt.com "LogicGep: Boolean networks inference using symbolic ..."
[8]: https://academic.oup.com/bib/article/26/6/bbaf621/8339795?utm_source=chatgpt.com "LogicSR: prior-guided symbolic regression for gene regulatory ..."
[9]: https://arxiv.org/pdf/2306.04718?utm_source=chatgpt.com "scalable neural symbolic regression using control variables"
[10]: https://arxiv.org/abs/2607.01232?utm_source=chatgpt.com "Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training"
[11]: https://www.synapse.org/Synapse%3Asyn3049712?utm_source=chatgpt.com "DREAM4 - In Silico Network Challenge - syn3049712 - Wiki"
[12]: https://academic.oup.com/bioinformatics/article/27/16/2263/254752?utm_source=chatgpt.com "in silico benchmark generation and performance profiling of ..."
[13]: https://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.pdf?utm_source=chatgpt.com "The DREAM4 In-silico Network Challenge"

# LTSR研究：層選択型シンボリック回帰による遺伝子制御方程式の推定

> **現在地（2026年7月）**：CPU上で研究パイプラインと小規模な予備実験を完了した段階である。
> GPUを用いた多seed・大規模実験は未実施であり、以下の数値は最終的な研究結論ではない。

## 目次

- [1. 研究概要](#1-研究概要)
- [2. 背景](#2-背景)
  - [2.1 GRNと遺伝子制御方程式](#21-grnと遺伝子制御方程式)
  - [2.2 シンボリック回帰](#22-シンボリック回帰)
  - [2.3 TransformerとNeSymReS](#23-transformerとnesymres)
  - [2.4 層選択的fine-tuning](#24-層選択的fine-tuning)
  - [2.5 TPSR](#25-tpsr)
  - [2.6 DREAM4とGeneNetWeaver](#26-dream4とgenenetweaver)
  - [2.7 関連研究と本研究の位置づけ](#27-関連研究と本研究の位置づけ)
- [3. 研究上の問い](#3-研究上の問い)
- [4. 用語解説](#4-用語解説)
- [5. 評価指標](#5-評価指標)
- [6. CPU環境と実装状況](#6-cpu環境と実装状況)
- [7. 実施したPhase](#7-実施したphase)
- [8. CPU実験結果](#8-cpu実験結果)
- [9. CPU研究から得られた結論](#9-cpu研究から得られた結論)
- [10. 結果を読む際の重要な注意](#10-結果を読む際の重要な注意)
- [11. 今後の展望](#11-今後の展望)
- [12. 再現方法](#12-再現方法)
- [13. リポジトリ構成](#13-リポジトリ構成)
- [14. 参考文献](#14-参考文献)

## 1. 研究概要

本研究は、遺伝子発現データから「どの遺伝子が、どの遺伝子を、どのような数式で制御するか」を推定する研究である。
英語題目は次のとおりである。

> **Layer-Selective Transformer-Based Symbolic Regression for Gene Regulatory Equation Discovery**

遺伝子 $i$ の発現量を $x_i(t)$ とすると、その時間変化を

$$
\frac{dx_i}{dt}=f_i(x_1,x_2,\ldots,x_p)
$$

と表せる。本研究の目標は、未知の関数 $f_i$ を、ニューラルネットワーク内部の読めない計算としてではなく、
人間が読める数式として発見することである。

中心モデルには、事前学習済みTransformer型シンボリック回帰モデル **NeSymReS** を使う。
さらに、全パラメータを更新する代わりに、適応への寄与が大きい少数のTransformer層だけをfine-tuningする。
推論時には必要に応じて **TPSR** による木探索を加え、非ニューラル手法 **PySR** と比較する。

研究計画の詳細は [`plan/20260714_firstplan.md`](plan/20260714_firstplan.md)、GPU実験の手順は
[`GPU_RUN.md`](GPU_RUN.md) に記載している。

## 2. 背景

### 2.1 GRNと遺伝子制御方程式

**Gene Regulatory Network（GRN、遺伝子制御ネットワーク）** は、遺伝子や転写因子の制御関係を表すネットワークである。
例えば、遺伝子 $A$ が遺伝子 $B$ の発現を増やすなら $A\rightarrow B$、減らすなら $A\dashv B$ と表す。
GRNを知ることは、細胞が刺激に応答する仕組み、病気で制御が崩れる仕組み、薬の標的候補を理解する助けになる。

多くのGRN推定法は「辺があるか」や「重要度はいくつか」を出力する。しかし、辺だけでは、制御が直線的なのか、
ある濃度で飽和するのか、複数因子が協力するのかまでは分からない。本研究は一歩進んで、制御関数そのものを推定する。

合成データでは、生物学でよく使われる **Hill型制御**を主に扱った。活性化の例は

$$
\frac{dx_i}{dt}
=\frac{\alpha x_j^n}{K^n+x_j^n}-\beta x_i
$$

である。第1項は遺伝子 $j$ による生成、第2項は遺伝子 $i$ の分解を表す。

- $\alpha$：最大生成速度
- $K$：反応が半分程度に達する発現量
- $n$：応答の急さを表すHill係数
- $\beta$：分解速度

抑制の例は

$$
\frac{dx_i}{dt}
=\frac{\alpha K^n}{K^n+x_j^n}-\beta x_i
$$

である。$x_j$ が大きくなるほど第1項が小さくなるため、遺伝子 $j$ が遺伝子 $i$ を抑える。
このように式が得られれば、制御の向きだけでなく、飽和、協同性、分解の強さまで議論できる可能性がある。

実際の時系列データでは $dx/dt$ を直接観測できないため、隣接時点から有限差分で近似する。

$$
\left.\frac{dx}{dt}\right|_{t_k}
\approx \frac{x(t_{k+1})-x(t_k)}{t_{k+1}-t_k}
$$

ただし、測定時点が少ない場合やノイズが大きい場合、この近似自体が大きな誤差源になる。

### 2.2 シンボリック回帰

通常の回帰は、直線や決められた形の式の係数を学習する。例えば線形回帰では

$$
\hat y=w_0+w_1x_1+w_2x_2
$$

という形を人間が先に決め、 $w_0,w_1,w_2$ を求める。一方、**シンボリック回帰（Symbolic Regression; SR）** は、
係数だけでなく、足し算、掛け算、割り算、べき乗、三角関数などの組合せも探索する。

データ集合を $D=\{(\mathbf{x}_i,y_i)\}_{i=1}^{N}$ 、使える数式の集合を $\mathcal{F}$ とすると、概念的には

$$
f^*=\underset{f\in\mathcal{F}}{\mathrm{arg\,min}}\left[\frac{1}{N}\sum_{i=1}^{N}\bigl(y_i-f(\mathbf{x}_i)\bigr)^2+\lambda C(f)\right]
$$

を解く。 $C(f)$ は式の長さや演算子数などの複雑度、
$\lambda$ は「精度」と「単純さ」のどちらを重視するかを決める値である。
単に誤差が小さいだけの巨大な式ではなく、短く説明しやすい式を探す点が重要である。

本研究の比較対象 **PySR** は、複数の式集団を進化させ、式の変形・簡約・定数最適化を繰り返す実用的なSRである [5]。

### 2.3 TransformerとNeSymReS

**Transformer**は、入力中のどの部分に注目するかを計算するattentionを中心としたニューラルネットワークである [1]。
入力行列を $Q,K,V$ に変換するscaled dot-product attentionは、概略

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}\right)V
$$

で表される。Transformerは同じ形の層を複数積み重ねる。**encoder**は入力を内部表現へ変換し、
**decoder**はその表現から出力列を1記号ずつ生成する。

**NeSymReS** は、大量の人工数式と、その数式から作った数値点集合を使ってTransformerを事前学習する手法である [2]。
入力は順序を持たない点集合

$$
\{(\mathbf{x}_1,y_1),\ldots,(\mathbf{x}_N,y_N)\}
$$

で、出力は数式を表すtoken列である。新しい問題を毎回ゼロから探索するのではなく、事前学習で得た
「よく現れる数式の形」をprior（事前知識）として使える点が特徴である。

NeSymReS論文は、大規模な手続き生成データによる事前学習がSRに利用できることを示した。
一方、事前学習分布と生物学的なGRN式の間にはずれがある。本研究は、そのずれを少数層のfine-tuningで埋められるかを調べる。

### 2.4 層選択的fine-tuning

**fine-tuning（微調整）** は、事前学習済みモデルを目的データでもう一度学習し、専門分野へ適応させることである。
全層fine-tuningは柔軟だが、計算・メモリを多く使い、データが少ないと過適合する可能性がある。

本研究の直接の着想は、Zhangらの *Is One Layer Enough?* [4] である。この研究はLLMの強化学習後学習を
層ごとに調べ、改善が少数の中間層へ集中する場合を報告した。層 $l$ だけを学習した損失を $L_l$、
事前学習モデルを $L_{\mathrm{base}}$、全層学習を $L_{\mathrm{full}}$ とすると、損失が小さいほどよい場合の層寄与度を

$$
C_l=
\frac{L_{\mathrm{base}}-L_l}
{L_{\mathrm{base}}-L_{\mathrm{full}}}
$$

のように表せる。 $C_l=1$ なら、その1層だけで全層学習と同程度の改善を回復したことになる。

ただし先行研究の対象はLLM・強化学習であり、NeSymReS・教師あり学習・GRNにはそのまま当てはまらない。
本研究は、この考えを別分野へ移し、どの層が数式生成や数値精度に寄与するかを検証する。

### 2.5 TPSR

**TPSR（Transformer-based Planning for Symbolic Regression）** は、Transformerの出力確率と
**Monte Carlo Tree Search（MCTS）** を組み合わせる方法である [3]。通常のbeam searchは、次のtoken確率が高い候補を
残していく。TPSRは数式を途中まで作った状態から先読みし、完成した式の精度や複雑度を評価して探索方向を修正する。

本研究ではTPSRを別の学習モデルではなく、NeSymReSの数式生成を改善する推論時探索として扱う。
概念的な報酬は、例えば

$$
R(f)=-\mathrm{NMSE}(f)-\lambda C(f)
$$

と書ける。予測誤差が小さく、式も単純なほど報酬が高い。微分できない評価値を探索へ直接組み込めることが利点である。

### 2.6 DREAM4とGeneNetWeaver

**DREAM4 In Silico Network Challenge** は、未知のGRNを遺伝子発現データから復元する国際的な比較課題である [8]。
本研究で使うin silicoデータには、10遺伝子または100遺伝子のネットワーク、時系列、摂動実験、正解の制御辺が含まれる。

データ生成には **GeneNetWeaver（GNW）** が使われた [7]。GNWは実在する大腸菌・酵母ネットワークから部分構造を取り出し、
転写・翻訳、制御、分子ノイズ、実験ノイズを含む動力学モデルを与える。したがって、単純なランダムグラフより
生物学的な構造を持ちつつ、正解ネットワークを知った状態で評価できる。

DREAM4が重要なのは、合成Hill式だけで成功した方法が、より複雑でノイズのある条件でも動くかを調べられるからである。
ただし、本研究で主に利用した公開時系列から得られるのは発現量であり、真の $dx/dt$ ではない。
そのため、有限差分誤差と候補制御因子選択の誤りがSRより前に入る。

### 2.7 関連研究と本研究の位置づけ

- **NeSymReS** はTransformerの大規模事前学習をSRへ導入した [2]。
- **TPSR** はMCTSでTransformerの数式生成を計画問題として改善した [3]。
- **PySR** は進化的探索による強力な非ニューラル比較対象である [5]。
- **ScaleSR** は多変数SRを低次元問題へ分解し、toggle switchやrepressilatorも扱った [9]。
- **LogicGep** と **LogicSR** はGRNをSRとして扱うが、主にBoolean論理規則を出力する [10, 11]。

本研究の新規性は個々の技術そのものではなく、次の組合せにある。

1. Transformer型SRを連続値のGRN方程式へ適応する。
2. NeSymReSのencoder/decoder層ごとの寄与を測る。
3. 高寄与層だけをfine-tuningし、全層・ランダム層と比較する。
4. 層選択的fine-tuningとTPSRの相互作用を検証する。
5. 合成GRN、DREAM4、ヒト時系列へ段階的に転移する。

## 3. 研究上の問い

1. GRN方程式への適応効果は、NeSymReSの特定層に集中するか。
2. 高寄与層だけのfine-tuningは、全層・ランダム層・低寄与層より効率的か。
3. 選択的fine-tuningとTPSRの組合せは、精度・式複雑度・ノイズ耐性を改善するか。
4. 合成GRNで得た方法は、DREAM4やヒト時系列発現データへ転移できるか。
5. 良い数値予測と、正しい式構造の回復は一致するか。

## 4. 用語解説

| 用語 | 高校生向けの説明 |
|---|---|
| **Attention** | 入力の各部分について「今の出力を作るとき、どこをどれだけ重視するか」を計算する仕組み。 |
| **Beam search** | 途中まで作った候補を複数残し、良さそうな候補を枝分かれさせながら完成形を探す方法。 |
| **BFGS / Broyden–Fletcher–Goldfarb–Shanno algorithm** | 数式に含まれる数値定数を、誤差が小さくなるように調整する最適化アルゴリズム。4人の研究者の姓に由来する。 |
| **Best weight / Best checkpoint** | validationで最も良い成績だった時点のモデルの重み。学習の最終時点が最良とは限らないため保存・復元する。 |
| **CE / Cross-Entropy** | 正解tokenに高い確率を付けられたかを測る学習誤差。小さいほどよい。 |
| **Checkpoint** | 学習済みモデルの重みを保存したファイル。ゲームのセーブデータに近い。 |
| **CPU / Central Processing Unit** | コンピュータ全体の汎用的な計算を担当する中央処理装置。 |
| **CUDA / Compute Unified Device Architecture** | NVIDIAが提供する、GPUで汎用計算を行うための並列計算基盤。本研究のGPU実験で使用する。 |
| **Decoder** | Transformerのうち、数式tokenを順番に出力する側。 |
| **Domain shift** | 学習データと本番データの性質が異なること。合成式と実際の生物データの差など。 |
| **DREAM4 / Dialogue for Reverse Engineering Assessments and Methods 4** | 正解ネットワーク付きの人工遺伝子発現データを使うGRN推定ベンチマーク。 |
| **E2E / End-to-End** | 入力から最終出力までを、一つのつながった処理として実行または学習すること。 |
| **Early stopping** | validationの成績が改善しなくなったら学習を早めに止め、過適合を抑える方法。 |
| **Encoder** | Transformerのうち、入力された数値点集合を内部表現へ変換する側。 |
| **Fail-fast** | 必須条件が崩れたとき、古い設定などで処理を続けず、その場で明確なエラーとして停止する設計。 |
| **FD / Finite Difference** | 隣り合う時点の値の差から変化率を近似する方法。有限差分。 |
| **FT / Fine-Tuning** | 既に学習したモデルを、目的に合う少量のデータで追加学習すること。 |
| **Generalization** | 学習に使っていないデータでも正しく働くこと。汎化。 |
| **GEO / Gene Expression Omnibus** | 公開遺伝子発現データベース。本研究ではGSE112372を利用した。 |
| **GNW / GeneNetWeaver** | DREAM系の人工GRNと発現データを生成するソフトウェア。 |
| **GPU / Graphics Processing Unit** | 大量の行列計算を並列処理する画像処理装置。深層学習をCPUより高速に実行できる。 |
| **GRN / Gene Regulatory Network** | 遺伝子同士の活性化・抑制関係を表すネットワーク。 |
| **GSE / GEO Series** | GEOに登録された一つの研究・データシリーズを表すaccessionの接頭辞。`GSE112372`はその登録番号。 |
| **Hill式** | 遺伝子制御の飽和やスイッチらしい応答を表す代表的な数式。 |
| **Hyperparameter** | 学習率やepoch数など、モデルがデータから覚えるのではなく実験者が候補を決める設定。ハイパーパラメータ。 |
| **ID / In-Distribution、OOD / Out-of-Distribution** | IDは学習時に近い分布、OODは学習時の分布から外れた条件での評価。 |
| **LASSO / Least Absolute Shrinkage and Selection Operator** | 不要な変数の係数を0へ近づけ、重要な変数を選びやすくする回帰手法。 |
| **LLM / Large Language Model** | 大量の文章で学習した大規模言語モデル。層選択学習の先行研究で使われた。 |
| **LODO / Leave-One-Donor-Out** | 1人分をテストに回し、残りの人で学習する操作を全員分繰り返す評価。 |
| **LPS / Lipopolysaccharide** | グラム陰性菌の外膜成分。免疫反応を起こす刺激としてヒト時系列実験に使われた。 |
| **LTSR / Layer-Selective Transformer-based Symbolic Regression** | 本研究で検討する、寄与の大きいTransformer層を選択して学習するSRの略称。 |
| **MCTS / Monte Carlo Tree Search** | 試しの先読みを繰り返して有望な枝を探す木探索。 |
| **MI / Mutual Information** | 二つの変数がどの程度情報を共有しているかを測る量。候補制御因子の選択に使う。 |
| **Motif** | GRNに繰り返し現れる小さな接続パターン。toggleやrepressilatorなど。 |
| **NCBI / National Center for Biotechnology Information** | GEOなどの生命科学データベースを運営する米国の国立機関。 |
| **NeSymReS / Neural Symbolic Regression that Scales** | 数値点集合から数式を生成する、事前学習済みTransformer型SR。 |
| **NMSE / Normalized Mean Squared Error** | 平均二乗誤差をデータのばらつきで正規化した指標。0に近いほどよい。 |
| **NSRS / Neural Symbolic Regression that Scales** | 本リポジトリでNeSymReS参照実装を置いているディレクトリ名。 |
| **ODE / Ordinary Differential Equation** | 常微分方程式。時間変化を記述する方程式。 |
| **Oracle** | 本来は未知の正解情報を与えた理想条件。手法の上限やボトルネックを調べるために使う。 |
| **Overfitting** | 学習データには合うが、未知データでは悪くなること。過適合。 |
| **Prior** | 探索前から持つ知識や傾向。既知の制御関係や、事前学習で得た式の出やすさなど。 |
| **PySR / Python用Symbolic Regressionライブラリ** | 進化計算で数式を探索するライブラリ。PySRは製品名であり、本研究ではNeSymReSとの主要比較対象。 |
| **R² / Coefficient of Determination** | 決定係数。予測がデータの変動をどれだけ説明できたかを表し、1に近いほどよい。 |
| **Regulator** | 標的遺伝子の発現に影響を与える遺伝子・転写因子。制御因子。 |
| **Repressilator** | 3遺伝子が環状に抑制し合う人工遺伝子回路。振動を起こし得る。 |
| **RL / Reinforcement Learning** | 試行錯誤と報酬を通じて行動方針を学ぶ強化学習。層選択学習の先行研究で使われた。 |
| **RNA / Ribonucleic Acid** | DNAの情報をもとに作られ、遺伝子発現の測定対象になるリボ核酸。 |
| **Seed** | 乱数の初期値。同じseedなら同じランダム処理を再現しやすい。 |
| **SBML / Systems Biology Markup Language** | 生物学的モデルを機械可読な形で交換するための標準形式。 |
| **SHA-256 / Secure Hash Algorithm 256-bit** | ファイル内容から固定長の識別値を作る方式。checkpointが同一か確認するために使う。 |
| **Symbolic recovery** | 予測値だけでなく、正解と同じ数式構造を回復できたかという評価。 |
| **SR / Symbolic Regression** | データを説明する、人間が読める数式そのものを探索する回帰。 |
| **TF / Transcription Factor** | DNAの転写を促進または抑制し、遺伝子発現を制御する転写因子。 |
| **Token** | 数式をモデルが扱える小単位へ分けたもの。変数、演算子、括弧など。 |
| **TP / True Positive、FP / False Positive、FN / False Negative** | TPは正しく検出したもの、FPは誤検出、FNは見逃し。Precision・Recall・F1の計算に使う。 |
| **TPSR / Transformer-based Planning for Symbolic Regression** | Transformerの候補確率を使い、MCTSで数式を先読み探索する手法。 |
| **Transformer** | Attentionを中心に、多層のencoder/decoderで情報を処理するニューラルネットワーク。 |
| **Validation / Test** | validationは方法や設定の選択用、testは最後の性能確認用。testで方法を選ぶと評価が甘くなる。 |
| **Variable F1 / Variable-recovery F1 Score** | 真の式に必要な変数を、予測式がどの程度過不足なく含むかを測るF1指標。 |

## 5. 評価指標

真値を $y_i$ 、予測を $\hat y_i$ 、真値の平均を $\bar y$ とする。

### NMSE

$$
\mathrm{NMSE}=\frac{\sum_{i=1}^{N}(y_i-\hat y_i)^2}{\sum_{i=1}^{N}(y_i-\bar y)^2+\varepsilon}
$$

0が完全一致で、小さいほどよい。1付近は「平均値だけを予測する」のと同程度である。

### 決定係数 $R^2$

$$
R^2=1-\frac{\sum_{i=1}^{N}(y_i-\hat y_i)^2}{\sum_{i=1}^{N}(y_i-\bar y)^2+\varepsilon}
$$

1が完全一致、0は平均値予測と同程度、負値は平均値予測より悪い。

### Precision・Recall・F1

$$
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad\mathrm{Recall}=\frac{TP}{TP+FN}
$$

$$
F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}
$$

制御辺や使用変数の回復を測る。Precisionは余計な予測の少なさ、Recallは見落としの少なさを表す。

### 式回復と複雑度

- **exact recovery**：文字列または簡約後の式が正解と一致するか。
- **skeleton recovery**：数値定数を一般定数に置き換えた式構造が一致するか。
- **complexity**：演算子・変数・定数などのノード数。小さいほど単純である。
- **valid rate**：生成した式のうち、構文解析と数値評価に成功した割合。

最新コードではdecode失敗を除外せず、失敗へ罰則を与えたNMSEを主指標にする。

## 6. CPU環境と実装状況

CPU実験はWindows上のPython 3.10環境で実施した。高beam幅、十分なMCTS rollout、大規模なseed反復は未実施である。

| 項目 | 状況 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.5.1+cpu |
| NeSymReS | checkpointロード・推論・選択的fine-tuningに成功 |
| PySR | Juliaバックエンドを含め動作確認済み |
| TPSR | E2EおよびNeSymReSバックボーン用コードを統合 |
| テスト | 現在のCPU環境で **36 passed、1 skipped** |
| GPU計算 | 未実施 |

1件のskipは、現在のPython 3.12環境ではNeSymReS/Hydra 1.0の互換テストを実行しないという既存条件による。
GPU本実験ではPython 3.10または3.11を使用する。

ローカルの `10M.ckpt` はファイル名と異なり、state dict上はencoder/decoder各5層の100M設定側アーキテクチャである。
そのため `NSRS/jupyter/100M/config.yaml` と組み合わせている。

## 7. 実施したPhase

| Phase | 内容 | CPUでの到達点 |
|---|---|---|
| 0 | 環境・checkpoint・ベースラインの動作確認 | 完了 |
| 1 | 合成Hill式・toggle・repressilator・多様な式構造の生成 | 完了 |
| 2 | NeSymReSとPySRの基礎比較 | 完了 |
| 3 | encoder/decoder各層の選択的fine-tuningスキャン | 完了（探索的） |
| 4 | 層寄与度とseed安定性の測定 | CPU小規模実験まで完了 |
| 5 | top/middle/random/bottom/full fine-tuningの比較 | CPU単一seedまで完了 |
| 6 | 選択的FT × TPSRの2×2比較とノイズ試験 | smoke testまで完了 |
| 7 | DREAM4 Size10/Size100、SBML teacherによる転移 | CPU小規模実験まで完了 |
| 8 | ヒトLPS刺激時系列、holdout donor、LODO評価 | CPU実験完了 |

## 8. CPU実験結果

### 8.1 Phase 0：実行基盤

- NeSymReSは、入力データの真の関係

  $$
  y=x_1\sin(x_1)
  $$

  を同値な式として復元した。
- PySRの出力も

  $$
  \hat y=x\sin(x)
  $$

  となった。
- TPSR E2EモデルをWindows/CPU上でロードし、軽量MCTSを完走した。
- Linux保存checkpoint、NumPy 2.0、Python dataclassへの互換修正を適用した。

詳細：[`phase0_report.md`](results/phase_results/phase0_report.md)

### 8.2 Phase 1–2：合成GRNとベースライン

最初の合成データは26問題で、activation、repression、toggle、repressilatorを含む。
各問題は200点、変数範囲は概ね $[0,3]$、初期版はノイズなしである。

9個のtest問題に対する初期比較は次のとおりであった。

| 方法 | median NMSE（ID） | median NMSE（OOD） | median $R^2$（ID） |
|---|---:|---:|---:|
| NeSymReS beam=2 | 0.376 | 0.901 | 0.602 |
| NeSymReS beam=5 | 0.120 | **0.187** | 0.806 |
| PySR | **0.00072** | 0.327 | **0.999** |

例えば `rpl_x2_test_25` の真の式は

$$
\frac{dx_2}{dt}=\frac{2.5}{1+x_1^4}-0.6x_2
$$

であり、PySRの保存された出力は

$$
\frac{dx_2}{dt}=-0.6x_2+\frac{2.5}{x_1^4+x_2/x_2}=-0.6x_2+\frac{2.5}{x_1^4+1}
$$

だった。この例では $x_2/x_2=1$ なので真の式と同値で、ID/OODともNMSEはほぼ0だった。
一方、別の抑制問題 `rep_test_20` に対するNeSymReS beam=5の出力例には

$$
\hat f(x_1,x_2)=-x_1+\frac{x_1}{x_2+0.7611}
$$

のように数値的にはある範囲へ適合しても、真のHill型抑制とは構造が異なる式もあった。

小規模な式ではPySRがID適合で大幅に優れていた。この9問題に限ればNeSymReS beam=5はOOD中央値でPySRを上回ったが、
問題数が少なく、一般化優位性は主張できない。

詳細：[`phase1_report.md`](results/phase_results/phase1_report.md)、
[`phase2_report.md`](results/phase_results/phase2_report.md)

### 8.3 Phase 3–4：層ごとの役割

Phase 3の探索では、teacher-forcing CEに対してdecoder後段の寄与が大きかった。
Phase 4の3 seeds実験でも、CE寄与の上位は安定していた。

| CE順位 | 層 | 平均寄与度 | top-3出現率 |
|---|---|---:|---:|
| 1 | `decoder_4` | 0.811 | 100% |
| 2 | `decoder_3` | 0.800 | 100% |
| 3 | `decoder_2` | 0.626 | 100% |

予測NMSE・$R^2$では `encoder_1`、`encoder_3`、`encoder_5` などencoder側が上位だった。
decoder後段は正解token列の生成、encoder側は数値点集合の表現に強く関与する可能性がある。

このPhaseでも各条件でSRを行ったが、symbolic recoveryはほぼ0だった。旧Phase 3/4集約ファイルには
各問題の推定式文字列が保存されておらず、ここへ正確に転載できる出力式はない。この保存欠損は最新コードで改善対象として扱う。

詳細：[`phase3_report.md`](results/phase_results/phase3_report.md)、
[`phase4_multiseed_report.md`](results/phase_results/phase4_multiseed_report.md)

### 8.4 Phase 5：選択的fine-tuning

60個の学習問題と8個の評価問題を用いた単一seedのCPU実験では、少数層のfine-tuningが良好だった。

| 条件 | 学習パラメータ比 | NMSE | $R^2$ |
|---|---:|---:|---:|
| pretrained | 0% | 0.1946 | 0.6878 |
| top 1 | 9.96% | 0.0309 | 0.9316 |
| top 2 | 19.92% | **0.0167** | **0.9755** |
| top 3 | 29.89% | 0.0272 | 0.9427 |
| random 3 | 13.66% | 0.0852 | 0.8123 |
| bottom 3 | 13.66% | 0.2502 | 0.4542 |
| all parameters | 100% | 0.2818 | 0.4783 |

top層はrandom/bottom/fullを上回った。全層fine-tuningはCEを改善しても予測NMSEを悪化させ、過適合の可能性を示した。
ただし単一seedかつ旧評価設計なので、中心仮説の確証ではない。

この旧runの `selective_results.json` は条件別の集約値のみを保存しており、推定数式を保存していない。
したがって捏造を避けるため、このPhaseの出力式は「記録なし」とする。GPU再実験では全問題の式文字列を必須成果物にする。

詳細：[`phase5_report.md`](results/phase_results/phase5_report.md)

### 8.5 Phase 6：TPSRとの組合せ

2問題だけを使った2×2 smoke testでは、選択的FTとTPSRの組合せが最良だった。

| Fine-tuning | 探索 | NMSE | $R^2$ |
|---|---|---:|---:|
| なし | beam | 0.203 | 0.684 |
| なし | TPSR | 0.529 | 0.356 |
| 選択的FT | beam | 0.193 | 0.700 |
| 選択的FT | TPSR | **0.082** | **0.879** |

TPSR単独はbeamより悪化したが、選択的FT後には改善した。微調整されたpriorがMCTSを有効な探索領域へ誘導した可能性がある。
ただし $n=2$ である。ノイズ0.0と0.1の比較では `selective + TPSR` のNMSE劣化量が
`selective + beam` より大きく、現時点でノイズ耐性仮説は支持されない。

この旧runも推定式文字列を保存していないため、出力式は「記録なし」である。GPU再実験ではbeam/TPSR双方の式と複雑度を保存する。

詳細：[`phase6_report.md`](results/phase_results/phase6_report.md)、
[`phase6_noise_report.md`](results/phase_results/phase6_noise_report.md)

### 8.6 Phase 7：DREAM4への転移

#### Regulator selection

Size10の5 networksを集約した結果では、候補制御因子のedge F1は次のとおりだった。

| 方法 | mean edge F1 |
|---|---:|
| oracle | **0.883** |
| correlation | 0.264 |
| mutual information | 0.279 |
| LASSO | 0.266 |

Size100 net1ではoracle 0.883に対し、correlation/MI/LASSOは約0.06–0.10だった。
SR以前のregulator preselectionが大きなボトルネックである。

#### Local symbolic regression

- Size10 net1では、oracle候補でもNMSEは約0.84だった。
- Size100 net1では、oracle候補でもNMSEは約0.98だった。
- 合成dreamlikeデータによる選択的FTはDREAM4有限差分ターゲットを改善しなかった。
- 相関選択の誤りが加わると、Size10のNMSEは約0.98まで悪化した。

Size10 net1の標的G1に対する保存済みのNeSymReS出力例は

$$
\widehat{\frac{dG_1}{dt}}=-0.006125\tan(0.896395G_1-0.563921)
$$

で、NMSEは0.824だった。Size10の公開評価では真のODE式を直接比較できず、この式を真の機構とは解釈できない。
むしろ、Hill型GRNとして不自然な $\tan$ が生成され、予測性能も低いという失敗例である。

SBML由来teacherによるfine-tuningでは、clean SBML holdout NMSEが0.311から0.0038へ改善したが、
DREAM有限差分への改善は0.890から0.725に留まった。teacher domainへの適合が実データ転移より容易であることを示す。

現段階ではDREAM4への転移成功は示されていない。有限差分ノイズ、候補選択、合成–DREAM間のdomain shiftが主要課題である。

詳細：[`phase7_package_a_report.md`](results/phase_results/phase7_package_a_report.md)、
[`phase7_dream4_report.md`](results/phase_results/phase7_dream4_report.md)、
[`phase7_dream4_size100_report.md`](results/phase_results/phase7_dream4_size100_report.md)

### 8.7 Phase 8：ヒトLPS刺激時系列

NCBI GEOの **GSE112372** [12] から、20遺伝子、4 donors、5時点を使用した。
5時点から得た導関数は真のODE微分ではなく、平滑化有限差分によるproxyである。

単一donor holdoutでは選択的FTが良かった。

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

保存された選択的FTの出力例は次のとおりである。ここで $x_1,x_2,x_3$ は標的ごとに選んだ候補制御因子を表す。

$$
\widehat{\frac{d\,\mathrm{CCL5}}{dt}}=0.0002435\,\frac{x_1+x_2+x_3+13251.04}{x_1-2.0272}
$$

$$
\widehat{\frac{d\,\mathrm{CD40}}{dt}}=\left[\cos(x_1+x_2-x_3)-0.06879\right]^2
$$

$$
\widehat{\frac{d\,\mathrm{IFNB1}}{dt}}=\frac{0.62524x_1}{(0.40253x_2-x_3)^2}
$$

これらには分母が0へ近づく特異点や、Hill型制御として解釈しにくい三角関数が含まれる。
したがって、良いholdout NMSEだけを根拠に「真のヒト制御ODEを発見した」とは言えない。

LODOでは「選択的FTがPySRよりdonor間で一般化する」という主張は支持されなかった。
一方、選択的FTはpretrained NeSymReSより改善しており、domain adaptationの効果は示唆される。

詳細：[`phase8_report.md`](results/phase_results/phase8_report.md)、
[`phase8_lodo_report.md`](results/phase_results/phase8_lodo_report.md)

## 9. CPU研究から得られた結論

1. **適応効果は層によって異なる。** CEではdecoder後段、数値予測ではencoder側の寄与が大きかった。
2. **全層fine-tuningが常に最良ではない。** CPU pilotでは少数層の更新が全層更新より良い場合があった。
3. **選択的FTはNeSymReSのdomain adaptationを改善する可能性がある。** ただしPySRへの一般的優位性はない。
4. **TPSRとの相互作用は候補として残る。** 小規模実験では選択的FT後のみ改善したが、統計的検証は未実施である。
5. **DREAM4では候補制御因子選択が支配的な課題である。** 単純なcorr/MI/LASSOはoracleから大きく劣る。
6. **単一splitの良い結果を信用しすぎてはいけない。** ヒトLODOでは単一holdoutの結論が逆転した。
7. **symbolic recoveryは未達である。** 良いNMSEは、正しい式構造や生物学的機構の回復を意味しない。

## 10. 結果を読む際の重要な注意

上記の数値は、最新のレビュー修正より前に生成されたCPU pilotを含む。コード側では次を修正済みだが、
Phase 4以降は計算時間の都合で再実行していない。

- testを層選択へ使わず、trainからmotif単位でvalidationを分離する。
- Phase 5の最終比較に独立testだけを使う。
- seed内で乱数状態とバッチ順を固定し、条件間をpaired comparisonにする。
- 少数標本の95%信頼区間にStudentのt分布を使う。
- DREAM4を有限差分後の行ではなく、有限差分前のtrajectory単位で分割する。
- decode失敗を除外せず、valid率とfailure-penalized NMSEを主指標にする。
- Phase 6で精度・valid率・式複雑度を同時保存する。
- runごとのmanifest、checkpoint SHA256、ログ、出力を分離する。

### 10.1 GPU実験前に追加した公正化・安全対策

Claudeの研究レビューで、CPU pilotのfull FTがpretrainedより悪化しているため、同じ学習率とepochを全条件へ
適用した結果だけでは「少数層で十分」と判断できないことが指摘された。これを受け、GPU用の最新コードには次を実装した。

- **同数候補によるvalidation探索**：top、random、bottom、fullなど、すべてのtrainable条件へ同じ
  学習率×epoch候補数を与える。各候補は同じseed、初期checkpoint、データ順で比較する。
- **early stoppingとbest-weight復元**：validation CEが改善しなくなったら学習を停止できる。
  停止しない設定でも、指定epochの最後ではなくvalidation CEが最良だったepochの重みを復元する。
- **testの隔離**：学習率、epoch、停止時点はvalidationだけで選び、選択済みモデルだけを独立testで一度評価する。
- **探索履歴の保存**：候補数、各候補の設定とvalidation CE、選ばれた学習率・epoch、best epoch、stop epochをJSONへ保存する。
- **full FT基準の検査**：full FTがpretrainedを改善したseedでのみ、full FTを分母とする正規化寄与度を計算する。
  一つでも基準が成立しないseedがある指標は、GPU実験の層順位には使用しない。
- **絶対改善量の保存**：正規化寄与度が使えない場合も研究結果を捨てず、pretrainedからの正規化前の改善量を保存する。
- **指標の混入防止**：未定義のNMSEや$R^2$寄与度を順位平均へ混ぜない。Phase 5の結果には、CE、NMSE、$R^2$のうち
  実際に層順位へ使えた指標名も記録する。
- **fail-fast**：有効なPhase 4順位を作れなかった場合、古いCPU pilotの固定順位へ自動的に戻らず、Phase 5開始前に停止する。

Phase 4のGPU runでは、通常の寄与度に加えて次を保存する。

```text
phase4_multiseed/
  raw_scores_seed*.json                    正規化前のbase/full/各層スコア
  absolute_improvements_seed*.json         pretrainedからの絶対改善量
  contribution_status_seed*.json           seedごとのfull FT基準の成否
  contribution_status_aggregate.json       全seedをまとめた基準成立数
  tuning_seed*.json                        候補設定とvalidation選択履歴
  contrib_aggregate.json                    条件を満たした正規化寄与度
```

これらはGPU実験の設計を修正したものであり、前節までに掲載したCPU pilotを再計算した結果ではない。
したがって、CPU pilotの数値を新しい設計で得た結果として解釈してはならない。

また、旧Phase 3–6では集約ファイルに推定式文字列を残していなかった。数値指標だけでは式の妥当性を監査できないため、
GPU実験では各問題について、真の式、推定式、簡約式、変数対応、複雑度、valid判定を保存する必要がある。

したがって、CPU結果から論文レベルで仮説を確定してはならない。現時点の適切な表現は次のとおりである。

> 層選択的fine-tuning、TPSR、DREAM4、ヒトLODOを含む研究パイプラインを構築し、
> CPU小規模実験で動作、予備的傾向、主要な失敗要因を確認した。

## 11. 今後の展望

### 11.1 次に行うGPU実験

1. **事前確認**：[`GPU_RUN.md`](GPU_RUN.md) に従い、CUDA、checkpoint、設定、依存関係をpreflightで確認する。
2. **Phase 4再実行**：5–10 seedsで層寄与を測り、validation上の順位、top-3出現率、順位相関を求める。
3. **Phase 5本比較**：各条件へ同数の学習率・epoch候補を与え、validation CEによるearly stoppingとbest-weight復元を行う。その後、同じseed集合でtop/random/bottom/fullを独立test上で一度だけ評価し、paired差とt信頼区間を求める。
4. **Phase 6本比較**：複数noise・複数seed・十分なMCTS budgetで2×2比較し、FT効果、TPSR効果、相互作用を分離する。
5. **精度–複雑度評価**：NMSEだけでなく、valid率、式長、演算子数、Pareto frontierを比較する。
6. **DREAM4再評価**：Size10/100の全networkをtrajectory分割で評価し、regulator selectionとSRの誤差を分解する。
7. **ヒトLODO再評価**：valid率、donor間性能、外挿安定性、非負性、特異点の有無を確認する。
8. **全式保存**：各runの全推定式と簡約式を保存し、代表例だけでなく失敗例も追跡可能にする。

### 11.2 研究上の課題

- **正しい式構造の回復**：現在はNMSEが良くてもexact/skeleton recoveryがほぼ0である。
- **全層FT基準の成立確認**：full FTがpretrainedを改善しない指標では正規化寄与度を使わず、正規化前の絶対改善量を報告する。
- **探索空間の生物学的制約**：`tan` や危険な除算を無制限に許すと、局所的に合うが不自然な式が生まれる。
- **特異点対策**：評価範囲内外で分母が0へ近づく式へ罰則を与える必要がある。
- **regulator preselection**：DREAM4 Size100ではこの段階の誤りがSR性能を支配している。
- **導関数推定**：少数時点の有限差分は不安定であり、smoothing、Gaussian process、integral matchingなどとの比較が必要である。
- **domain shift**：合成Hill式、SBML teacher、DREAM4 FD、ヒトRNA-seqの分布差を定量化する必要がある。
- **比較の公平性**：PySR、beam、TPSRで計算時間または候補評価回数をそろえた比較が必要である。
- **統計的検出力**：問題数、network数、seed数、donor数を増やし、単一split依存を避ける必要がある。

### 11.3 発展方向

将来的には、Hill型演算子や非負性などの生物学的priorをsoft constraintとして探索へ入れ、
「予測が合う式」から「機構として反証可能な式」へ近づけたい。また、層寄与ランキングがGRN motif、ノイズ、
データ領域を越えて安定するかを調べることで、層選択が単なる計算節約ではなく、過適合を抑える正則化として働くかを検証する。

GPU実験で中心仮説が支持されなかった場合でも、どの指標でencoder/decoderの役割が分かれるか、
なぜ予測精度とsymbolic recoveryが一致しないか、DREAM4でどの前処理が支配的かは独立した研究成果になり得る。

## 12. 再現方法

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

## 13. リポジトリ構成

```text
plan/          研究計画
src/           データ処理、モデル、学習、評価の共通コード
scripts/       Phase別の実験エントリポイント
tests/         単体テスト
requirements/  CPU/GPU/dev別の依存関係
results/       CPU pilotの結果とrun出力
graphs/        run別の独立した図・表
NSRS/          NeSymReS参照実装
TPSR/          TPSR参照実装
```

新しく作る独立した図・表は [`graphs/README.md`](graphs/README.md) の規約に従い、
`graphs/<run-id>/figures/` または `graphs/<run-id>/tables/` に保存する。

## 14. 参考文献

1. Vaswani, A. et al. (2017). **Attention Is All You Need.** NeurIPS 2017.
   <https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html>
2. Biggio, L. et al. (2021). **Neural Symbolic Regression that Scales.** ICML 2021, PMLR 139:936–945.
   <https://proceedings.mlr.press/v139/biggio21a.html>
   公式実装：<https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales>
3. Shojaee, P. et al. (2023). **Transformer-based Planning for Symbolic Regression.** NeurIPS 2023.
   <https://openreview.net/forum?id=0rVXQEeFEL>
   公式実装：<https://github.com/deep-symbolic-mathematics/tpsr>
4. Zhang, Z. et al. (2026). **Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training.** arXiv:2607.01232.
   <https://arxiv.org/abs/2607.01232>
5. Cranmer, M. (2023). **Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl.** arXiv:2305.01582.
   <https://arxiv.org/abs/2305.01582>
6. PySR documentation and source code.
   <https://github.com/MilesCranmer/PySR>
7. Schaffter, T., Marbach, D., & Floreano, D. (2011). **GeneNetWeaver: In Silico Benchmark Generation and Performance Profiling of Network Inference Methods.** *Bioinformatics*, 27(16), 2263–2270.
   <https://doi.org/10.1093/bioinformatics/btr373>
8. DREAM / GeneNetWeaver. **DREAM4 In Silico Network Challenge.**
   <https://gnw.sourceforge.net/dreamchallenge.html>
9. Chu, X. et al. (2023). **Scalable Neural Symbolic Regression using Control Variables.** arXiv:2306.04718.
   <https://arxiv.org/abs/2306.04718>
10. Zhang, D. et al. (2024). **LogicGep: Boolean Networks Inference Using Symbolic Regression from Time-Series Transcriptomic Profiling Data.** *Briefings in Bioinformatics*, 25(4), bbae286.
    <https://doi.org/10.1093/bib/bbae286>
11. Zhang, D., Liu, Z.-P., & Gao, R. (2025). **LogicSR: Prior-Guided Symbolic Regression for Gene Regulatory Network Inference from Single-Cell Transcriptomics Data.** *Briefings in Bioinformatics*, 26(6), bbaf621.
    <https://doi.org/10.1093/bib/bbaf621>
12. NCBI Gene Expression Omnibus. **GSE112372: Time-series of human LPS stimulated-monocyte derived macrophages.**
    <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE112372>

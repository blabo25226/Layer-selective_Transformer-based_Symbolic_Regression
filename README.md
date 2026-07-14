# LTSR: Layer-selective transfer for symbolic GRN reconstruction

NeSymReSの層別寄与を測定し、高寄与層だけを微調整した記号回帰を合成GRN、DREAM4、
ヒト時系列データで評価する研究コードである。研究計画は
[`plan/20260714_firstplan.md`](plan/20260714_firstplan.md)、GPU本実験の手順は
[`GPU_RUN.md`](GPU_RUN.md)を参照する。

## 対応環境

- Python 3.10または3.11
- CPU: `pip install -r requirements/cpu.txt`
- GPU: CUDA版PyTorchを先に導入後、`pip install -r requirements/gpu.txt`
- NeSymReS: `pip install -e NSRS/src`
- 開発・テスト: `pip install -r requirements/dev.txt`

Hydra 1.0との互換性によりPython 3.12は本実験環境としてサポートしない。

## 検証

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

## 実験の流れ

1. `scripts/generate_diverse_suite.py`: 構造分離した合成GRNを生成
2. `scripts/phase4_multiseed.py`: motif単位のvalidationで層寄与を測定
3. `scripts/phase5_selective_train.py`: 独立testで選択的微調整を比較
4. `scripts/phase6_noise_sweep.py`: TPSRとの2×2比較とノイズ耐性
5. `scripts/phase7_dream4_size10.py` / `phase7_dream4_size100.py`: DREAM4評価
6. `scripts/phase8_lodo.py`: ヒトデータのleave-one-donor-out評価

GPU一括実行ではPhase 4〜6を同じseed集合で反復し、run単位のmanifestとレポートを
`results/runs/<run-id>/`に保存する。

## ディレクトリ

```text
plan/          研究計画
src/           データ処理、モデル、学習、評価の再利用コード
scripts/       Phase別の実験エントリポイント
tests/         CPUで実行できる単体テスト
requirements/  CPU/GPU/dev別の依存関係
results/       コミット済みのパイロット結果とローカルrun出力
NSRS/          NeSymReS参照実装
TPSR/          TPSR参照実装
```

## 評価上の原則

- 層選択はvalidationだけで行い、testを選択に使わない。
- seed内では全conditionに同じ乱数状態とバッチ順を使う。
- decode失敗を除外せず、valid率とfailure-penalized NMSEを報告する。
- DREAM4は有限差分前にtrajectory単位で分割する。
- 少数seed/donorの区間推定にはStudentのt区間を使う。

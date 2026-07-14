# Phase 0 Report

Date: 2026-07-14

## Checkpoint status

| Artifact | Path | Status |
|----------|------|--------|
| NeSymReS weights | `NSRS/weights/10M.ckpt` (302 MB) | OK |
| TPSR E2E backbone | `TPSR/symbolicregression/weights/model1.pt` (357 MB) | OK |
| TPSR NeSymReS backbone | `TPSR/nesymres/weights/10M.ckpt` (hardlink) | OK (see below) |

### Dead Google Drive link workaround

TPSR README の NeSymReS checkpoint (Google Drive) はリンク切れ。
代替として **HuggingFace の `NSRS/weights/10M.ckpt` を TPSR 側へハードリンク** する。

```powershell
powershell -File scripts/setup_phase0_links.ps1
```

### Important: checkpoint / config mismatch

`10M.ckpt` というファイル名だが、state_dict は **100M アーキテクチャ**（encoder/decoder 各5層）。
推論時は `NSRS/jupyter/100M/config.yaml` を使うこと。`10MPaper/config.yaml` ではロードエラーになる。

## Environment

推奨: **Python 3.10** の conda 環境 `ltsr-phase0`

| Package | Version |
|---------|---------|
| Python | 3.10.20 |
| torch | 2.5.1+cpu |
| pytorch-lightning | 1.9.5 |
| hydra-core | 1.0.0 |
| omegaconf | 2.1.2 |
| setuptools | <81 (pkg_resources 必須) |

Python 3.12 の base anaconda では NeSymReS / hydra が動かない。Colab も 3.10 系を推奨。

## Smoke test results

### NeSymReS — PASS

```bash
conda activate ltsr-phase0
python scripts/phase0_nesymres_smoke.py
```

- Target: `x_1*sin(x_1)`
- Result: `((x_1)*(sin(x_1)))`, loss `0.0`
- Device: CPU（ローカルに CUDA なし）

### PySR — PASS

```bash
python scripts/phase0_pysr_smoke.py
```

- Target: `x*sin(x)`
- Result: `sin(x0) * x0`
- 初回実行時に Julia パッケージのダウンロードあり（数分）

### TPSR E2E — PASS (Windows, CPU)

```bash
python scripts/phase0_tpsr_smoke.py
```

- `pathlib.PosixPath = pathlib.WindowsPath` で Linux 訓練済み `model.pt` を Windows でロード可能
- E2E baseline sequence length: 45
- TPSR+MCTS が完了（horizon=20, rollout=1）

### TPSR + NeSymReS backbone

- checkpoint は `TPSR/nesymres/weights/10M.ckpt` で代替可能。
- `tpsr_demo.py --backbone_model nesymres` は Colab/Linux で確認予定。

## Patches applied (vendor code)

| File | Change | Reason |
|------|--------|--------|
| `NSRS/src/nesymres/dclasses.py` | `field(default_factory=BFGSParams)` | Python 3.12 dataclass 互換 |
| `TPSR/nesymres/src/nesymres/dclasses.py` | 同上 | 同上 |
| `TPSR/symbolicregression/trainer.py` | `np.infty` → `np.inf` | NumPy 2.0 互換 |
| `TPSR/symbolicregression/e2e_model.py` | `map_location` 追加 | CPU ロード |

## Phase 0 checklist

- [x] NeSymReS コード取得
- [x] NeSymReS checkpoint 取得・推論成功
- [x] TPSR コード取得
- [x] TPSR E2E checkpoint 取得
- [x] TPSR NeSymReS checkpoint（HuggingFace 経由で代替）
- [x] PySR 導入・同データで解析成功
- [x] 依存関係ファイル (`requirements/base.txt`, `colab.txt`)
- [x] スモークテストスクリプト (`scripts/phase0_*.py`)
- [x] TPSR デモ実行（`scripts/phase0_tpsr_smoke.py`）
- [ ] 再現用 Notebook（`.py` で代替中）

## Follow-up: Issues 3–5 (done)

- Issue 3: `scripts/issue3_list_layers.py` → `issue3_report.md`
- Issue 4–5: `scripts/issue4_5_freeze_check.py` → `issue4_5_report.md`
- Core API: `src/models/layer_selector.py`

## Next steps

1. Issue 6 / Phase 1: 合成 Hill / GRN 方程式データ生成
2. Issue 7: PySR baseline on that data

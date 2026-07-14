# Phase 5: high-contribution selective fine-tuning

- Ranking source: Phase 4 `accuracy`
- Order: `encoder_2, encoder_1, encoder_3, decoder_1, decoder_0, decoder_4, encoder_0, decoder_2, decoder_3, encoder_5, encoder_4, output_head`
- k=3 for middle / random / bottom
- Train FT: 17; test eval: 4
- Epochs: 5, lr: 0.0001
- Decode: beam=1, BFGS restarts=1, stop_time=0.5s
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase5/selective_results.json`

## Conditions

| condition | layers |
|-----------|--------|
| `pretrained` | — |
| `top_1` | encoder_2 |
| `top_2` | encoder_2, encoder_1 |
| `top_3` | encoder_2, encoder_1, encoder_3 |
| `middle_3` | decoder_0, decoder_4, encoder_0 |
| `random_3` | encoder_0, output_head, encoder_2 |
| `bottom_3` | encoder_5, encoder_4, output_head |
| `all_params` | ALL |

## Scores

| condition | trainable | frac | train CE | val CE | gap | NMSE | R² | sym | time (s) | mem (MB) |
|-----------|-----------|------|----------|--------|-----|------|----|-----|----------|----------|
| `pretrained` | 0 | 0 | nan | 1.945 | nan | 0.2453 | 0.6386 | 0 | 26.2 | nan |
| `top_1` | 2,130,944 | 0.08073 | 1.246 | 1.52 | 0.274 | 0.1094 | 0.8233 | 0 | 24.8 | nan |
| `top_2` | 4,261,888 | 0.1615 | 1.156 | 1.507 | 0.3504 | 0.223 | 0.6329 | 0 | 42.4 | nan |
| `top_3` | 6,392,832 | 0.2422 | 1.018 | 1.331 | 0.3126 | 0.1718 | 0.7116 | 0 | 51.2 | nan |
| `middle_3` | 6,702,080 | 0.2539 | 0.1966 | 0.416 | 0.2194 | 0.07605 | 0.8841 | 0.25 | 45.9 | nan |
| `random_3` | 3,604,540 | 0.1366 | 1.107 | 1.331 | 0.224 | 0.07623 | 0.8866 | 0 | 53.0 | nan |
| `bottom_3` | 4,292,668 | 0.1626 | 1.125 | 1.377 | 0.252 | 0.1839 | 0.7076 | 0 | 28.6 | nan |
| `all_params` | 26,395,708 | 1 | 0.01937 | 0.04055 | 0.02117 | 0.5167 | 0.3117 | 1 | 32.1 | nan |

## Recovery vs full FT (efficiency)

- L_base / L_full val CE = 1.945 / 0.04055
- NMSE_base / NMSE_full = 0.2453 / 0.5167

| condition | C_CE | C_NMSE | trainable % |
|-----------|------|--------|-------------|
| `top_1` | 0.2232 | -0.5009 | 8.073% |
| `top_2` | 0.2299 | -0.08237 | 16.15% |
| `top_3` | 0.3223 | -0.271 | 24.22% |
| `middle_3` | 0.8028 | -0.6238 | 25.39% |
| `random_3` | 0.3221 | -0.6231 | 13.66% |
| `bottom_3` | 0.2984 | -0.2265 | 16.26% |

## Ranking (val CE, then NMSE)

| rank | condition | val CE | NMSE | trainable |
|------|-----------|--------|------|-----------|
| 1 | `all_params` | 0.04055 | 0.5167 | 26,395,708 |
| 2 | `middle_3` | 0.416 | 0.07605 | 6,702,080 |
| 3 | `top_3` | 1.331 | 0.1718 | 6,392,832 |
| 4 | `random_3` | 1.331 | 0.07623 | 3,604,540 |
| 5 | `bottom_3` | 1.377 | 0.1839 | 4,292,668 |
| 6 | `top_2` | 1.507 | 0.223 | 4,261,888 |
| 7 | `top_1` | 1.52 | 0.1094 | 2,130,944 |

## Findings

1. **Best selective set by val CE:** `middle_3` (`decoder_0`, `decoder_4`, `encoder_0`) reaches **C_CE ≈ 0.80** with ~25% params — far above accuracy-rank `top_3` (C_CE ≈ 0.32). This set includes late decoder, consistent with Phase 3/4 CE results.
2. **Accuracy-rank top-k is encoder-heavy** and underperforms CE-wise; Phase 4’s metric split shows up again at multi-layer selection.
3. **`all_params` has lowest CE** but **worse median NMSE** than pretrained under light BFGS decode → `C_NMSE` is not meaningful when `NMSE_full > NMSE_base` (negative denom direction). Prefer CE / R² or fuller decode for NMSE claims.
4. Overfit gaps are moderate (~0.22–0.35) for selective runs; full FT gap is small (0.02) on this tiny train set (token CE may memorize).

## Notes

- Transfer proxy: Phase 1 train/test split uses OOD parameter ranges (same equation families).
- Overfit gap = val_CE − train_CE (larger → more overfit).
- `random_k` seed=0 for reproducibility.
- Light BFGS decode; raise flags for paper-quality decode metrics.
- Optional follow-up: re-run with `--ranking ce` (decoder-first top-k).

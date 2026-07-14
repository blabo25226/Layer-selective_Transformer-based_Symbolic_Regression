# Phase 5: high-contribution selective fine-tuning

- Ranking mode: `accuracy` (Phase 4 contributions.json)
- Order: `decoder_3, decoder_4, decoder_1, encoder_2, decoder_2, encoder_3, encoder_1, encoder_4, decoder_0, encoder_0, encoder_5, output_head`
- k=3 for middle / random / bottom
- Train FT: 60; test eval: 8
- Epochs: 5, lr: 0.0001
- Decode: beam=1, BFGS restarts=1, stop_time=0.3s
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase5/selective_results.json`

## Conditions

| condition | layers |
|-----------|--------|
| `pretrained` | — |
| `top_1` | decoder_3 |
| `top_2` | decoder_3, decoder_4 |
| `top_3` | decoder_3, decoder_4, decoder_1 |
| `middle_3` | decoder_2, encoder_3, encoder_1 |
| `random_3` | encoder_0, output_head, encoder_2 |
| `bottom_3` | encoder_0, encoder_5, output_head |
| `all_params` | ALL |

## Scores

| condition | trainable | frac | train CE | val CE | gap | NMSE | R² | sym | time (s) | mem (MB) |
|-----------|-----------|------|----------|--------|-----|------|----|-----|----------|----------|
| `pretrained` | 0 | 0 | nan | 1.331 | nan | 0.1946 | 0.6878 | 0 | 20.9 | nan |
| `top_1` | 2,629,632 | 0.09962 | 0.1066 | 0.367 | 0.2604 | 0.03086 | 0.9316 | 0 | 30.6 | nan |
| `top_2` | 5,259,264 | 0.1992 | 0.05694 | 0.3934 | 0.3365 | 0.01669 | 0.9755 | 0 | 30.4 | nan |
| `top_3` | 7,888,896 | 0.2989 | 0.0427 | 0.4478 | 0.4051 | 0.02722 | 0.9427 | 0 | 35.5 | nan |
| `middle_3` | 6,891,520 | 0.2611 | 0.08414 | 0.4658 | 0.3817 | 0.05127 | 0.9247 | 0 | 35.9 | nan |
| `random_3` | 3,604,540 | 0.1366 | 0.5619 | 0.6791 | 0.1172 | 0.08524 | 0.8123 | 0 | 40.3 | nan |
| `bottom_3` | 3,604,540 | 0.1366 | 0.5579 | 0.7118 | 0.1539 | 0.2502 | 0.4542 | 0 | 41.4 | nan |
| `all_params` | 26,395,708 | 1 | 0.0377 | 0.5994 | 0.5617 | 0.2818 | 0.4783 | 0 | 49.9 | nan |

## Recovery vs full FT (efficiency)

- L_base / L_full val CE = 1.331 / 0.5994
- NMSE_base / NMSE_full = 0.1946 / 0.2818

| condition | C_CE | C_NMSE | trainable % |
|-----------|------|--------|-------------|
| `top_1` | 1.318 | -1.878 | 9.962% |
| `top_2` | 1.282 | -2.04 | 19.92% |
| `top_3` | 1.207 | -1.92 | 29.89% |
| `middle_3` | 1.183 | -1.644 | 26.11% |
| `random_3` | 0.8912 | -1.254 | 13.66% |
| `bottom_3` | 0.8464 | 0.6377 | 13.66% |

## Ranking (val CE, then NMSE)

| rank | condition | val CE | NMSE | trainable |
|------|-----------|--------|------|-----------|
| 1 | `top_1` | 0.367 | 0.03086 | 2,629,632 |
| 2 | `top_2` | 0.3934 | 0.01669 | 5,259,264 |
| 3 | `top_3` | 0.4478 | 0.02722 | 7,888,896 |
| 4 | `middle_3` | 0.4658 | 0.05127 | 6,891,520 |
| 5 | `all_params` | 0.5994 | 0.2818 | 26,395,708 |
| 6 | `random_3` | 0.6791 | 0.08524 | 3,604,540 |
| 7 | `bottom_3` | 0.7118 | 0.2502 | 3,604,540 |

## H2 check: selected layers vs random control

- `top_3` NMSE=0.02722, val CE=0.4478
- `random_3` NMSE=0.08524, val CE=0.6791 (random now excludes the top-3 layers — A-3 fix)
- `middle_3` NMSE=0.05127, val CE=0.4658

**Verdict:** top-k beats the random control on prediction NMSE (Δ=0.05802).

> ⚠️ **Statistical caveat (A-1):** this run uses a small train/eval set (60 train / 8 eval equations, single seed). A top≈random gap of this size is NOT evidence for H2. Re-run `scripts/phase4_multiseed.py` + this script across ≥3 seeds and compare distributions (mean ± CI) before claiming layer selectivity.

## Notes

- Transfer proxy: Phase 1 train/test split uses OOD parameter ranges (same equation families).
- Overfit gap = val_CE − train_CE (larger → more overfit).
- Layer order derived from Phase 4 `contributions.json` (mode `accuracy`); `random_k` seed=0 excludes top-3 for a fair control.
- Light BFGS decode; raise flags for paper-quality decode metrics.

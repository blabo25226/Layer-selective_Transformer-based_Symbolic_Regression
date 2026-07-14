# Phase 4 (multi-seed): layer contribution with CI

- Data: `results/synthetic/diverse_v1` (train 60 / test 8)
- Seeds: [0, 1, 2]  |  epochs: 3, lr: 0.0001
- Device: `cpu`

Contribution `C=1` means the single layer recovers the full-FT gain; `C=0` means no better than pretrained. **A layer is only 'high-contribution' if its CI stays clearly above the random/other layers across seeds.**

## val_ce

| rank | layer | mean C | 95% CI | std | top-3 stability |
|------|-------|--------|--------|-----|-----------------|
| 1 | `decoder_4` | 0.8111 | ±0.04637 | 0.04098 | 100% |
| 2 | `decoder_3` | 0.7999 | ±0.0607 | 0.05364 | 100% |
| 3 | `decoder_2` | 0.6263 | ±0.03043 | 0.02689 | 100% |
| 4 | `decoder_1` | 0.4283 | ±0.02857 | 0.02525 | 0% |
| 5 | `encoder_2` | 0.3137 | ±0.00194 | 0.001714 | 0% |
| 6 | `encoder_3` | 0.3083 | ±0.00202 | 0.001785 | 0% |
| 7 | `encoder_1` | 0.254 | ±0.02563 | 0.02265 | 0% |
| 8 | `encoder_5` | 0.212 | ±0.003854 | 0.003406 | 0% |
| 9 | `encoder_4` | 0.2084 | ±0.004148 | 0.003666 | 0% |
| 10 | `decoder_0` | 0.1821 | ±0.005639 | 0.004983 | 0% |
| 11 | `encoder_0` | 0.1009 | ±0.0007362 | 0.0006506 | 0% |
| 12 | `output_head` | 0.09437 | ±0.006258 | 0.00553 | 0% |

## nmse

| rank | layer | mean C | 95% CI | std | top-3 stability |
|------|-------|--------|--------|-----|-----------------|
| 1 | `encoder_1` | 0.6798 | ±0.03021 | 0.02669 | 100% |
| 2 | `encoder_3` | 0.6597 | ±0.00585 | 0.00517 | 100% |
| 3 | `decoder_3` | 0.6355 | ±0.0242 | 0.02139 | 33% |
| 4 | `encoder_5` | 0.6262 | ±0.005443 | 0.00481 | 33% |
| 5 | `encoder_4` | 0.6262 | ±0.005443 | 0.00481 | 0% |
| 6 | `decoder_2` | 0.5901 | ±0.0994 | 0.08784 | 33% |
| 7 | `encoder_2` | 0.5497 | ±0.05719 | 0.05054 | 0% |
| 8 | `decoder_4` | 0.413 | ±0.2309 | 0.204 | 0% |
| 9 | `encoder_0` | 0.2048 | ±0.2333 | 0.2062 | 0% |
| 10 | `decoder_1` | 0.1154 | ±0.2658 | 0.2349 | 0% |
| 11 | `output_head` | -0.01952 | ±0.1436 | 0.1269 | 0% |
| 12 | `decoder_0` | -0.404 | ±0.02751 | 0.02431 | 0% |

## r2

| rank | layer | mean C | 95% CI | std | top-3 stability |
|------|-------|--------|--------|-----|-----------------|
| 1 | `encoder_3` | 0.7197 | ±0.0124 | 0.01095 | 100% |
| 2 | `encoder_5` | 0.6988 | ±0.01215 | 0.01074 | 67% |
| 3 | `encoder_4` | 0.6988 | ±0.01215 | 0.01074 | 33% |
| 4 | `decoder_3` | 0.6573 | ±0.06181 | 0.05462 | 33% |
| 5 | `decoder_2` | 0.6317 | ±0.08337 | 0.07368 | 33% |
| 6 | `encoder_1` | 0.6101 | ±0.1001 | 0.0885 | 0% |
| 7 | `encoder_2` | 0.5671 | ±0.1127 | 0.09957 | 0% |
| 8 | `decoder_4` | 0.5593 | ±0.157 | 0.1387 | 33% |
| 9 | `decoder_1` | 0.09362 | ±0.2118 | 0.1872 | 0% |
| 10 | `encoder_0` | 0.07333 | ±0.05834 | 0.05156 | 0% |
| 11 | `output_head` | 0.02836 | ±0.4146 | 0.3664 | 0% |
| 12 | `decoder_0` | -0.4444 | ±0.223 | 0.1971 | 0% |

## var_f1

| rank | layer | mean C | 95% CI | std | top-3 stability |
|------|-------|--------|--------|-----|-----------------|
| 1 | `decoder_0` | nan | ±nan | nan | 0% |
| 2 | `decoder_1` | nan | ±nan | nan | 0% |
| 3 | `decoder_2` | nan | ±nan | nan | 0% |
| 4 | `decoder_3` | nan | ±nan | nan | 0% |
| 5 | `decoder_4` | nan | ±nan | nan | 0% |
| 6 | `encoder_0` | nan | ±nan | nan | 0% |
| 7 | `encoder_1` | nan | ±nan | nan | 0% |
| 8 | `encoder_2` | nan | ±nan | nan | 0% |
| 9 | `encoder_3` | nan | ±nan | nan | 0% |
| 10 | `encoder_4` | nan | ±nan | nan | 0% |
| 11 | `encoder_5` | nan | ±nan | nan | 0% |
| 12 | `output_head` | nan | ±nan | nan | 0% |

## sym_rate

| rank | layer | mean C | 95% CI | std | top-3 stability |
|------|-------|--------|--------|-----|-----------------|
| 1 | `decoder_0` | nan | ±nan | nan | 0% |
| 2 | `decoder_1` | nan | ±nan | nan | 0% |
| 3 | `decoder_2` | nan | ±nan | nan | 0% |
| 4 | `decoder_3` | nan | ±nan | nan | 0% |
| 5 | `decoder_4` | nan | ±nan | nan | 0% |
| 6 | `encoder_0` | nan | ±nan | nan | 0% |
| 7 | `encoder_1` | nan | ±nan | nan | 0% |
| 8 | `encoder_2` | nan | ±nan | nan | 0% |
| 9 | `encoder_3` | nan | ±nan | nan | 0% |
| 10 | `encoder_4` | nan | ±nan | nan | 0% |
| 11 | `encoder_5` | nan | ±nan | nan | 0% |
| 12 | `output_head` | nan | ±nan | nan | 0% |

## How to read this

- **Overlapping CIs** between the top layer and mid/random layers ⇒ H2 (layer selectivity) is **not** supported for that metric.
- **top-3 stability < ~66%** ⇒ the 'high-contribution' layer is seed-dependent, not a property of the network.
- Compare the CE ranking (decoder-heavy) vs prediction rankings (encoder-heavy): if they disagree, report them separately (plan §Phase 4).

# Phase 6 noise sweep — H3 robustness (A-4)

- Layers (top_3 of accuracy ranking, source=`phase4`): `decoder_3, decoder_4, decoder_1`
- Noise levels: ['0.0', '0.1']  |  epochs: 5, TPSR rollout=2
- Device: `cpu`

## NMSE (median, lower better) vs noise

| noise | pretrained_beam | pretrained_tpsr | selective_beam | selective_tpsr |
|-------|------|------|------|------|
| 0.0 | 0.2047 | 0.07687 | 0.03068 | 0.01745 |
| 0.1 | 0.225 | 0.2494 | 0.02221 | 0.049 |

## R² (median, higher better) vs noise

| noise | pretrained_beam | pretrained_tpsr | selective_beam | selective_tpsr |
|-------|------|------|------|------|
| 0.0 | 0.6879 | 0.8967 | 0.9427 | 0.9402 |
| 0.1 | 0.6041 | 0.65 | 0.9683 | 0.9205 |

## H3 check: robustness slope

- ΔNMSE(0.0→0.1) selective_tpsr = 0.03155
- ΔNMSE(0.0→0.1) selective_beam = -0.008474
- ΔNMSE(0.0→0.1) pretrained_beam = 0.02029

**Verdict:** selective+TPSR degrades NOT more slowly (H3 unsupported) than selective+beam under increasing noise.

> ⚠️ Report accuracy **and** complexity together (plan H3 is about the tradeoff). Overlapping performance across cells at a given noise level is not evidence for TPSR; check per-seed spread as in phase4_multiseed.

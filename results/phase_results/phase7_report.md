# Phase 7: DREAM-like GRN (oracle + variable selection)

- Dataset: `C:/Document/researches/LTSR/results/synthetic/phase7_dreamlike_v1` (GNW-style synthetic; no Synapse download)
- Genes: 10, edges: 17
- Targets evaluated: 6, k=2 regulators, max_vars=3
- Selective FT layers: `decoder_0, decoder_4, encoder_0`
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase7/dreamlike_results.json`

## Regulator selection (train expression)

| method | mean target F1 | edge P | edge R | edge F1 |
|--------|----------------|--------|--------|---------|
| `oracle` | 1 | 1 | 1 | 1 |
| `corr` | 0.9444 | 0.9167 | 1 | 0.9565 |
| `mi` | 0.9444 | 0.9167 | 1 | 0.9565 |
| `lasso` | 0.9444 | 0.9167 | 1 | 0.9565 |

## Local symbolic regression (test RHS)

| condition | selection | NMSE | R² | var F1 | sym | time (s) |
|-----------|-----------|------|----|--------|-----|----------|
| `pretrained_oracle` | oracle / pretrained | 1.537 | -2.759 | 0.9333 | 0 | 33.9 |
| `selective_oracle` | oracle / selective | 0.06765 | 0.7773 | 1 | 0 | 36.4 |
| `selective_corr` | corr / selective | 0.1639 | 0.4342 | 0.9667 | 0 | 65.9 |

## Findings

1. **Oracle selection** edge F1 = 1 (expected high / perfect on true parents).
2. **Practical selectors** (corr / mi / lasso) edge F1: 0.9565 / 0.9565 / 0.9565.
3. **Selective FT vs pretrained** (oracle locals): ΔNMSE = -1.47.
4. **Selection error compounding**: selective+corr NMSE = 0.1639 vs selective+oracle 0.06765 (Δ = 0.09624).

## Notes

- This is a **DREAM4/GNW-style** synthetic substitute until official Synapse dumps are wired in.
- Per-target problems keep ≤3 variables for NeSymReS.
- Next: import real DREAM4 gold standard + steady-state / time series when available.

# Phase 7c: official DREAM4 Size100

- Data root: `C:/Document/researches/LTSR/data/dream4`
- Networks: [1]
- Supervision: timeseries finite-difference `dx/dt` (70/30; ~200 rows/net — no multifactorial in Size100 training set)
- Transfer FT: selective `decoder_0, decoder_4, encoder_0` on synthetic dreamlike
- Selection on genes with parents; SR on up to 20 parent-genes; k=2
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase7_dream4/size100_results.json`

## Regulator selection

### Network 1

| method | mean target F1 | edge P | edge R | edge F1 |
|--------|----------------|--------|--------|---------|
| `oracle` | 0.9278 | 1 | 0.7898 | 0.8825 |
| `corr` | 0.07529 | 0.07558 | 0.07386 | 0.07471 |
| `mi` | 0.06921 | 0.06395 | 0.0625 | 0.06322 |
| `lasso` | 0.1045 | 0.09884 | 0.09659 | 0.0977 |

## Local SR (FD targets)

### Network 1

| condition | NMSE | R² | var F1 | time (s) |
|-----------|------|----|--------|----------|
| `pretrained_oracle` | 0.9792 | 0.01448 | 0.5767 | 29.5 |
| `selective_oracle` | 0.9882 | -0.0003389 | 0.7 | 31.8 |
| `selective_corr` | 0.9922 | 0.0006529 | 0.69 | 30.7 |

## Aggregate (mean over networks)

| condition | mean NMSE | mean R² |
|-----------|-----------|---------|
| `pretrained_oracle` | 0.9792 | 0.01448 |
| `selective_oracle` | 0.9882 | -0.0003389 |
| `selective_corr` | 0.9922 | 0.0006529 |

## Findings

1. Size100 loader works: 100 genes, 176 edges, FD matrix (200×100); no multifactorial in training set.
2. **Selection gap widens vs Size10**: oracle edge F1≈0.88; corr/mi/lasso ≈0.06–0.10 on FD `dx/dt`.
3. **Local SR is hard**: NMSE≈0.98 even with oracle candidates (noisy FD + domain shift; transfer FT unused).
4. Message for the paper pipeline: on Size100, **regulator preselection quality dominates**; naive corr is near-chance.

## Notes

- Size100 has **no multifactorial** file in the main training folder (unlike Size10); evaluation uses timeseries FD only.
- Gold TSV lists all gene pairs with 0/1; we keep edges with flag=1.
- Oracle edge recall can be <1 when true degree > k.
- Re-run all nets: `python scripts/phase7_dream4_size100.py --all-nets`.

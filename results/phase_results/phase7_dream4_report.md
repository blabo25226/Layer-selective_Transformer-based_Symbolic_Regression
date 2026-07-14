# Phase 7b: official DREAM4 Size10

- Data root: `C:/Document/researches/LTSR/data/dream4`
- Networks: [1]
- Supervision: timeseries finite-difference `dx/dt` (70/30 split)
- Transfer FT: selective `decoder_0, decoder_4, encoder_0` on synthetic dreamlike oracle
- k=2, max_vars=3, target_limit=10
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase7_dream4/size10_results.json`

## Regulator selection

### Network 1

| method | mean target F1 | edge P | edge R | edge F1 |
|--------|----------------|--------|--------|---------|
| `oracle` | 0.9133 | 1 | 0.6667 | 0.8 |
| `corr` | 0.2667 | 0.25 | 0.3333 | 0.2857 |
| `mi` | 0.03333 | 0.05 | 0.06667 | 0.05714 |
| `lasso` | 0.2667 | 0.25 | 0.3333 | 0.2857 |

## Local SR (no true ODE skeleton in public DREAM files)

Symbolic exact-match is N/A; report predictive NMSE / R² on held-out FD samples.

### Network 1

| condition | NMSE | R² | var F1 | time (s) |
|-----------|------|----|--------|----------|
| `pretrained_oracle` | 0.8408 | 0.1555 | 0.7167 | 13.1 |
| `selective_oracle` | 0.8686 | 0.1217 | 0.7167 | 12.6 |
| `selective_corr` | 0.9816 | 0.0155 | 0.59 | 14.5 |

## Aggregate (mean over networks)

| condition | mean NMSE | mean R² |
|-----------|-----------|---------|
| `pretrained_oracle` | 0.8408 | 0.1555 |
| `selective_oracle` | 0.8686 | 0.1217 |
| `selective_corr` | 0.9816 | 0.0155 |

## Findings

1. **Loader OK**: official Size10 net1 gold standard + timeseries FD wired into Phase-7 pipeline.
2. **Selection**: oracle ≫ corr/mi/lasso on FD-estimated `dx/dt` (edge F1 0.80 vs ~0.29). Oracle edge R=0.67 because `k=2` truncates high-degree parents (NeSymReS ≤3-var budget).
3. **SR**: predictive accuracy is modest (NMSE≈0.84); transfer selective FT from synthetic dreamlike **does not help** yet (domain shift / no DREAM teacher equations).
4. **Compounding**: corr selection further hurts SR (NMSE 0.98).

## Notes

- Official DREAM4 from GNW (`data/dream4`); gitignored.
- Gold standard edges used for selection; ODE XML not parsed for teacher equations.
- Next options: Size100, multifactorial-only features, SBML ODE teacher labels, `--all-nets`.

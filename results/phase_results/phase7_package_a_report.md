# Package A: transfer hardening + Size10 multi-net + comparison

## Overfit-aware SBML-FT

- Train nets: [1, 2, 3, 4, 5]
- Label noise std (rel): 0.08
- Mix: nonoise protein-TF trajectory + random SBML RHS
- Early stop patience=3, lr=5e-05
- train_CE=0.06009, val_CE=0.06757, gap=0.007484
- stopped_epoch=12.0

| eval | pretrained NMSE | SBML-FT NMSE |
|------|-----------------|--------------|
| SBML holdout (clean teacher) | 0.3112 | 0.003784 |
| DREAM FD transfer (net1, oracle locals) | 0.8901 | 0.725 |

## Size10 regulator selection (mean edge F1 over nets)

| method | mean edge F1 | std |
|--------|--------------|-----|
| `oracle` | 0.8832 | 0.05472 |
| `corr` | 0.2644 | 0.13 |
| `mi` | 0.2788 | 0.09457 |
| `lasso` | 0.2659 | 0.09416 |

## Method comparison (Size10 net1, oracle locals, noisy FD)

| method | NMSE | R2 | time (s) |
|--------|------|----|----------|
| `pysr` | 0.6527 | 0.3405 | 69.6 |
| `pretrained_beam` | 0.8901 | 0.09427 | 12.6 |
| `selective_dreamlike_beam` | 0.9766 | 0.01172 | 9.5 |
| `sbml_ft_beam` | 0.725 | 0.2671 | 9.5 |
| `sbml_ft_tpsr` | 0.7122 | 0.2753 | 8.2 |

## Reading the overfit risk

- Holdout gain (clean SBML): 0.3075 NMSE drop
- Transfer gain (noisy DREAM FD): 0.1651 NMSE drop
- If holdout >> transfer, SBML-FT still mostly memorizes teacher domain.
- CE gap (val-train) should stay modest; large gap => reduce epochs / raise noise.

- Results JSON: `C:/Document/researches/LTSR/results/phase_results/phase7_package_a/package_a_results.json`

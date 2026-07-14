# Phase 8 LODO — cross-donor generalization

- Donors (folds): ['1', '10', '11', '2']
- Derivative: `smooth_fd`  |  selective layers: `decoder_3, decoder_4, decoder_1`
- PySR included: True

Generalization gap = median holdout NMSE − median in-donor NMSE, averaged over LODO folds (positive = overfits to training donors).

| method | mean in-NMSE | mean hold-NMSE | hold 95% CI | gap | gap 95% CI | folds |
|--------|--------------|----------------|-------------|-----|------------|-------|
| `pysr` | 0.008229 | 0.2027 | ±0.2572 | 0.1945 | ±0.26 | 4 |
| `selective_beam` | 0.2076 | 0.4868 | ±0.3447 | 0.2792 | ±0.3356 | 4 |
| `pretrained_beam` | 0.5737 | 1.458 | ±1.185 | 0.8846 | ±1.309 | 4 |

## Key claim: does selective-FT generalize better than PySR?

- PySR: in=0.008229 → hold=0.2027 (gap 0.1945)
- selective_beam: in=0.2076 → hold=0.4868 (gap 0.2792)

**Verdict:** selective-FT NeSymReS does NOT beat PySR on holdout NMSE across LODO folds. Check whether the holdout-NMSE CIs overlap before claiming significance.

> ⚠️ Derivatives are proxies (`smooth_fd`), not true time derivatives, so this measures cross-donor consistency of the fitted RHS, not recovery of a true ODE. Grow the donor/gene panel to tighten the CIs.

# Phase 8 LODO — cross-donor generalization

- Donors (folds): ['1', '10', '11', '2']
- Derivative: `smooth_fd`  |  selective layers: `decoder_3, decoder_4, decoder_1`
- PySR included: False

Generalization gap = median holdout NMSE − median in-donor NMSE, averaged over LODO folds (positive = overfits to training donors).

| method | mean in-NMSE | mean hold-NMSE | hold 95% CI | gap | gap 95% CI | folds |
|--------|--------------|----------------|-------------|-----|------------|-------|
| `selective_beam` | 0.1838 | 0.4688 | ±0.3158 | 0.285 | ±0.363 | 4 |
| `pretrained_beam` | 0.5538 | 0.8554 | ±0.3127 | 0.3017 | ±0.463 | 4 |

> ⚠️ Derivatives are proxies (`smooth_fd`), not true time derivatives, so this measures cross-donor consistency of the fitted RHS, not recovery of a true ODE. Grow the donor/gene panel to tighten the CIs.

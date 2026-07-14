# Phase 8: human LPS macrophage application (GSE112372)

## Setup

- Source: `GSE112372` TPM subset
- Genes: 20 (RELA, NFKB1, NFKBIA, IRF3, IRF7, STAT1, STAT2, JUN, FOS, SPI1, CEBPB, TNF, IL6, IL1B, CXCL8, CXCL10, CCL5, IFNB1, CD40, PTGS2)
- Times (h): [0.0, 0.5, 3.0, 8.0, 16.0]
- Train donors: ['1', '10', '2']; holdout donor: `11`
- Derivative: `smooth_fd` (proxy, not true time derivative)
- Candidates per target: k=2, max_vars=3
- Selective FT layers: `decoder_0, decoder_4, encoder_0`
- Results JSON: `C:/Document/researches/LTSR/results/phase_results/phase8/phase8_results.json`

## Regulator selection vs curated prior

| method | edge F1 | precision | recall | mean prior recall |
|--------|---------|-----------|--------|-------------------|
| `prior` | 0.8444 | 1 | 0.7308 | 0.84 |
| `prior_corr` | 0.8444 | 1 | 0.7308 | 0.84 |
| `corr` | 0 | 0 | 0 | 0 |
| `mi` | 0.04348 | 0.05 | 0.03846 | 0.02 |

## Local SR (prior candidates)

### In-donor (train donors)

| method | NMSE | R2 | time (s) |
|--------|------|----|----------|
| `pretrained_beam` | 0.4053 | 0.3669 | 26 |
| `selective_dreamlike_beam` | 0.1655 | 0.765 | 24.7 |
| `pysr` | 0.005431 | 0.9922 | 49.8 |

### Holdout donor

| method | NMSE | R2 |
|--------|------|----|
| `pretrained_beam` | 0.5959 | 0.1189 |
| `selective_dreamlike_beam` | 0.1783 | 0.715 |
| `pysr` | 0.5016 | 0.2313 |

## Example predicted equations (prior targets)

- `CCL5`: `0.00024347382513762*(x_1 + x_2 + x_3 + 13251.0397959273)/(x_1 - 2.0271809532178)` (holdout NMSE=0.1229)
- `CD40`: `(cos(x_1 + x_2 - x_3) - 0.0687897586697591)**2` (holdout NMSE=0.6053)
- `CXCL10`: `0.000422041109978109*(x_1 + x_2 + 8299.36863354038)/(x_1 - 1.82201911054012)` (holdout NMSE=0.2156)
- `CXCL8`: `(x_1 - 8.51000835597328)*tan(x_1 + 2.49126714947034)/(x_2 - 0.726113773611434)` (holdout NMSE=0.1116)
- `IFNB1`: `0.625238220523951*x_1/(0.402529040364089*x_2 - x_3)**2` (holdout NMSE=0.1606)

## Interpretation limits

- Do **not** claim recovery of true human regulatory ODEs.
- Evaluate by **held-out donor prediction** and **prior TF consistency** only.
- RNA-seq TPM + FD/spline yields a coarse proxy for dx/dt on 5 time points.
- Curated prior is soft gold for this inflammatory panel.

## Findings

1. Prior-constrained selection edge F1=0.8444 vs data-only corr F1=0.
2. Best holdout donor NMSE: `selective_dreamlike_beam` = 0.1783.
3. Phase 8 is an application demo; main LTSR claims remain on synthetic + DREAM4.

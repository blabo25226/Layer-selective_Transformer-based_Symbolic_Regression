# Phase 1 / Issue 6: synthetic GRN data

- Output: `C:/Document/researches/LTSR/results/synthetic/phase1_v1`
- Problems: 26
- Splits: {'train': 17, 'test': 9}
- Families: {'activation': 6, 'repression': 5, 'toggle': 6, 'repressilator': 9}

## Design

- Target: learn ODE right-hand side `y = f(X)` (not full time trajectories yet).
- Families: activation Hill, repression Hill, toggle switch, repressilator.
- Split: parameter-range split within family (train/test use different coeffs).
- Support: `x_i ~ Uniform(0, 3)`, 200 points/problem, no noise in v1.

## Index (first 12)

| eq_id | family | split | n_vars | target_expr |
|-------|--------|-------|--------|-------------|
| `act_train_1` | activation | train | 2 | `alpha*x_2**n/(K**n+x_2**n)-beta*x_1` |
| `act_train_2` | activation | train | 2 | `alpha*x_2**n/(K**n+x_2**n)-beta*x_1` |
| `act_train_3` | activation | train | 2 | `alpha*x_2**n/(K**n+x_2**n)-beta*x_1` |
| `act_train_4` | activation | train | 2 | `alpha*x_2**n/(K**n+x_2**n)-beta*x_1` |
| `rep_train_5` | repression | train | 2 | `alpha*K**n/(K**n+x_2**n)-beta*x_1` |
| `rep_train_6` | repression | train | 2 | `alpha*K**n/(K**n+x_2**n)-beta*x_1` |
| `rep_train_7` | repression | train | 2 | `alpha*K**n/(K**n+x_2**n)-beta*x_1` |
| `tog_dx_train_8` | toggle | train | 2 | `alpha1/(1+x_2**n1)-beta1*x_1` |
| `tog_dy_train_9` | toggle | train | 2 | `alpha2/(1+x_1**n2)-beta2*x_2` |
| `tog_dx_train_10` | toggle | train | 2 | `alpha1/(1+x_2**n1)-beta1*x_1` |
| `tog_dy_train_11` | toggle | train | 2 | `alpha2/(1+x_1**n2)-beta2*x_2` |
| `rpl_x1_train_12` | repressilator | train | 3 | `alpha/(1+x_3**n)-beta*x_1` |

Full index: `C:/Document/researches/LTSR/results/synthetic/phase1_v1/index.json`

## Next

- Issue 7: PySR baseline on train problems — **smoke PASS** (see below)
- Phase 2: NeSymReS / TPSR baselines on the same suite

## Issue 7 smoke (PySR on `act_train_1`)

True template: `alpha*x_2**n/(K**n+x_2**n)-beta*x_1` with `alpha=1,K=1,n=2,beta=0.5`

PySR found (algebraically equivalent):

`(-1.0 / (square(x_2) + 1.0)) + (1.0 - (x_1 * 0.5))`

NMSE ≈ 0. Details: `results/phase_results/issue7_pysr_smoke.json`

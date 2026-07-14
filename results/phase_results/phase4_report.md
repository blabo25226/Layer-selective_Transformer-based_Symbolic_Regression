# Phase 4: layer contribution

- Train FT examples: 60 (Phase 1 train)
- Eval problems: 8 test
- Epochs: 3, lr: 0.0001
- Decode: beam=1, BFGS restarts=1, stop_time=0.3s
- Device: `cpu`
- Raw results: `C:/Document/researches/LTSR/results/phase_results/phase4/layer_contribution.json`
- Contributions: `C:/Document/researches/LTSR/results/phase_results/phase4/contributions.json`

## Raw scores

| condition | trainable | val CE | NMSE med | R² med | var F1 | sym rate | time (s) |
|-----------|-----------|--------|----------|--------|--------|----------|----------|
| `pretrained` | 0 | 1.331 | 0.2729 | 0.6873 | 0.875 | 0 | 21.6 |
| `output_head` | 30,780 | 1.199 | 0.3201 | 0.4926 | 0.875 | 0 | 28.2 |
| `encoder_0` | 1,442,816 | 1.159 | 0.157 | 0.7387 | 0.75 | 0 | 35.9 |
| `encoder_1` | 2,130,944 | 1.035 | 0.06372 | 0.8739 | 0.75 | 0 | 33.6 |
| `encoder_2` | 2,130,944 | 1 | 0.07169 | 0.8199 | 1 | 0 | 33.8 |
| `encoder_3` | 2,130,944 | 0.9987 | 0.06843 | 0.8905 | 1 | 0 | 28.6 |
| `encoder_4` | 2,130,944 | 1.03 | 0.08069 | 0.8657 | 1 | 0 | 26.4 |
| `encoder_5` | 2,130,944 | 1.036 | 0.07739 | 0.8941 | 1 | 0 | 25.8 |
| `decoder_0` | 2,629,632 | 1.11 | 0.2658 | 0.5915 | 1 | 0 | 27.6 |
| `decoder_1` | 2,629,632 | 0.7751 | 0.1458 | 0.715 | 1 | 0 | 24.3 |
| `decoder_2` | 2,629,632 | 0.5892 | 0.08079 | 0.866 | 1 | 0 | 26.3 |
| `decoder_3` | 2,629,632 | 0.4342 | 0.06111 | 0.8966 | 1 | 0 | 24.1 |
| `decoder_4` | 2,629,632 | 0.4716 | 0.03068 | 0.9445 | 1 | 0 | 25.7 |
| `all_params` | 26,395,708 | 0.4955 | 0.2694 | 0.4003 | 1 | 0 | 38.4 |

## Layer contribution (separate metrics; plan §Phase 4)

Formulas:

- Higher-better: `C = (S_k - S_base) / (S_full - S_base)`
- Lower-better: `C = (L_base - L_k) / (L_base - L_full)`

`S_base` / `L_base` = `pretrained`, `S_full` / `L_full` = `all_params`.

### val_ce — Cross-entropy (token teacher-forcing)

(lower better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_3` | 1.073 |
| 2 | `decoder_4` | 1.029 |
| 3 | `decoder_2` | 0.888 |
| 4 | `decoder_1` | 0.6654 |
| 5 | `encoder_3` | 0.3977 |
| 6 | `encoder_2` | 0.396 |
| 7 | `encoder_4` | 0.3605 |
| 8 | `encoder_1` | 0.3544 |
| 9 | `encoder_5` | 0.3535 |
| 10 | `decoder_0` | 0.2651 |
| 11 | `encoder_0` | 0.2063 |
| 12 | `output_head` | 0.1575 |

### nmse — Prediction NMSE (median, lower better)

(lower better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_4` | 68.82 |
| 2 | `decoder_3` | 60.17 |
| 3 | `encoder_1` | 59.43 |
| 4 | `encoder_3` | 58.09 |
| 5 | `encoder_2` | 57.17 |
| 6 | `encoder_5` | 55.55 |
| 7 | `encoder_4` | 54.61 |
| 8 | `decoder_2` | 54.58 |
| 9 | `decoder_1` | 36.12 |
| 10 | `encoder_0` | 32.94 |
| 11 | `decoder_0` | 2.023 |
| 12 | `output_head` | -13.4 |

### r2 — Prediction R² (median, higher better)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `output_head` | 0.6786 |
| 2 | `decoder_0` | 0.334 |
| 3 | `decoder_1` | -0.09634 |
| 4 | `encoder_0` | -0.1792 |
| 5 | `encoder_2` | -0.4621 |
| 6 | `encoder_4` | -0.6216 |
| 7 | `decoder_2` | -0.6226 |
| 8 | `encoder_1` | -0.65 |
| 9 | `encoder_3` | -0.7079 |
| 10 | `encoder_5` | -0.7206 |
| 11 | `decoder_3` | -0.7293 |
| 12 | `decoder_4` | -0.8962 |

### var_f1 — Variable recovery F1 (mean)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_0` | 1 |
| 2 | `decoder_1` | 1 |
| 3 | `decoder_2` | 1 |
| 4 | `decoder_3` | 1 |
| 5 | `decoder_4` | 1 |
| 6 | `encoder_2` | 1 |
| 7 | `encoder_3` | 1 |
| 8 | `encoder_4` | 1 |
| 9 | `encoder_5` | 1 |
| 10 | `output_head` | 0 |
| 11 | `encoder_0` | -1 |
| 12 | `encoder_1` | -1 |

### sym_rate — Symbolic recovery rate (mean)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_0` | nan |
| 2 | `decoder_1` | nan |
| 3 | `decoder_2` | nan |
| 4 | `decoder_3` | nan |
| 5 | `decoder_4` | nan |
| 6 | `encoder_0` | nan |
| 7 | `encoder_1` | nan |
| 8 | `encoder_2` | nan |
| 9 | `encoder_3` | nan |
| 10 | `encoder_4` | nan |
| 11 | `encoder_5` | nan |
| 12 | `output_head` | nan |

## Consensus ranking (mean rank across metrics)

| rank | condition | mean metric-rank |
|------|-----------|------------------|
| 1 | `decoder_1` | 4.00 |
| 2 | `decoder_3` | 4.40 |
| 3 | `decoder_2` | 4.80 |
| 4 | `decoder_0` | 5.00 |
| 5 | `decoder_4` | 5.00 |
| 6 | `encoder_2` | 6.00 |
| 7 | `encoder_3` | 6.80 |
| 8 | `encoder_1` | 7.60 |
| 9 | `encoder_4` | 7.60 |
| 10 | `encoder_0` | 8.40 |
| 11 | `encoder_5` | 9.00 |
| 12 | `output_head` | 9.40 |

## Notes

- Primary Phase-4 claim uses **separate** contribution tables (accuracy / symbolic / variable), not a single weighted composite.
- Symbolic recovery = max(exact, skeleton-equivalent after constant→c, sympy equiv).
- Variable recovery = presence of true `x_i` names in predicted string (F1).
- Decode BFGS defaults are light (`beam=1`, `restarts=1`, `stop_time=0.5`) for scan throughput; raise `--bfgs-restarts` / `--bfgs-stop-time` for paper numbers.

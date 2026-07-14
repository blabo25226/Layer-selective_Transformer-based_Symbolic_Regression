# Phase 4: layer contribution

- Train FT examples: 17 (Phase 1 train)
- Eval problems: 4 test
- Epochs: 3, lr: 0.0001
- Decode: beam=1, BFGS restarts=1, stop_time=0.5s
- Device: `cpu`
- Raw results: `C:/Document/researches/LTSR/results/phase_results/phase4/layer_contribution.json`
- Contributions: `C:/Document/researches/LTSR/results/phase_results/phase4/contributions.json`

## Raw scores

| condition | trainable | val CE | NMSE med | R² med | var F1 | sym rate | time (s) |
|-----------|-----------|--------|----------|--------|--------|----------|----------|
| `pretrained` | 0 | 1.945 | 0.7857 | 0.185 | 1 | 0 | 34.7 |
| `output_head` | 30,780 | 1.856 | 0.6219 | 0.2397 | 1 | 0 | 26.5 |
| `encoder_0` | 1,442,816 | 1.802 | 0.3052 | 0.4905 | 1 | 0 | 20.8 |
| `encoder_1` | 2,130,944 | 1.548 | 0.2067 | 0.6521 | 1 | 0 | 19.8 |
| `encoder_2` | 2,130,944 | 1.578 | 0.0935 | 0.8684 | 1 | 0 | 30.4 |
| `encoder_3` | 2,130,944 | 1.659 | 0.2067 | 0.6521 | 1 | 0 | 27.2 |
| `encoder_4` | 2,130,944 | 1.811 | 0.5542 | 0.3557 | 1 | 0 | 19.1 |
| `encoder_5` | 2,130,944 | 1.81 | 0.5089 | 0.4099 | 1 | 0 | 20.8 |
| `decoder_0` | 2,629,632 | 1.613 | 0.2453 | 0.6386 | 1 | 0 | 22.2 |
| `decoder_1` | 2,629,632 | 1.323 | 0.2453 | 0.6386 | 1 | 0 | 24.8 |
| `decoder_2` | 2,629,632 | 1.083 | 0.464 | 0.2223 | 1 | 0 | 45.6 |
| `decoder_3` | 2,629,632 | 0.8427 | 0.6219 | 0.2397 | 1 | 0 | 41.1 |
| `decoder_4` | 2,629,632 | 0.7828 | 0.5188 | 0.349 | 1 | 0 | 35.1 |
| `all_params` | 26,395,708 | 0.2141 | 0.05897 | 0.9375 | 0.75 | 0.25 | 42.7 |

## Layer contribution (separate metrics; plan §Phase 4)

Formulas:

- Higher-better: `C = (S_k - S_base) / (S_full - S_base)`
- Lower-better: `C = (L_base - L_k) / (L_base - L_full)`

`S_base` / `L_base` = `pretrained`, `S_full` / `L_full` = `all_params`.

### val_ce — Cross-entropy (token teacher-forcing)

(lower better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_4` | 0.6713 |
| 2 | `decoder_3` | 0.6367 |
| 3 | `decoder_2` | 0.498 |
| 4 | `decoder_1` | 0.3592 |
| 5 | `encoder_1` | 0.229 |
| 6 | `encoder_2` | 0.2117 |
| 7 | `decoder_0` | 0.1918 |
| 8 | `encoder_3` | 0.165 |
| 9 | `encoder_0` | 0.08247 |
| 10 | `encoder_5` | 0.07805 |
| 11 | `encoder_4` | 0.07719 |
| 12 | `output_head` | 0.05145 |

### nmse — Prediction NMSE (median, lower better)

(lower better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `encoder_2` | 0.9525 |
| 2 | `encoder_1` | 0.7967 |
| 3 | `encoder_3` | 0.7967 |
| 4 | `decoder_0` | 0.7435 |
| 5 | `decoder_1` | 0.7435 |
| 6 | `encoder_0` | 0.6611 |
| 7 | `decoder_2` | 0.4426 |
| 8 | `encoder_5` | 0.3808 |
| 9 | `decoder_4` | 0.3673 |
| 10 | `encoder_4` | 0.3186 |
| 11 | `decoder_3` | 0.2254 |
| 12 | `output_head` | 0.2254 |

### r2 — Prediction R² (median, higher better)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `encoder_2` | 0.9081 |
| 2 | `encoder_1` | 0.6207 |
| 3 | `encoder_3` | 0.6207 |
| 4 | `decoder_0` | 0.6028 |
| 5 | `decoder_1` | 0.6028 |
| 6 | `encoder_0` | 0.406 |
| 7 | `encoder_5` | 0.2988 |
| 8 | `encoder_4` | 0.2268 |
| 9 | `decoder_4` | 0.2179 |
| 10 | `decoder_3` | 0.07272 |
| 11 | `output_head` | 0.07272 |
| 12 | `decoder_2` | 0.04956 |

### var_f1 — Variable recovery F1 (mean)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_0` | -0 |
| 2 | `decoder_1` | -0 |
| 3 | `decoder_2` | -0 |
| 4 | `decoder_3` | -0 |
| 5 | `decoder_4` | -0 |
| 6 | `encoder_0` | -0 |
| 7 | `encoder_1` | -0 |
| 8 | `encoder_2` | -0 |
| 9 | `encoder_3` | -0 |
| 10 | `encoder_4` | -0 |
| 11 | `encoder_5` | -0 |
| 12 | `output_head` | -0 |

### sym_rate — Symbolic recovery rate (mean)

(higher better raw → C above)

| rank | condition | C |
|------|-----------|---|
| 1 | `decoder_0` | 0 |
| 2 | `decoder_1` | 0 |
| 3 | `decoder_2` | 0 |
| 4 | `decoder_3` | 0 |
| 5 | `decoder_4` | 0 |
| 6 | `encoder_0` | 0 |
| 7 | `encoder_1` | 0 |
| 8 | `encoder_2` | 0 |
| 9 | `encoder_3` | 0 |
| 10 | `encoder_4` | 0 |
| 11 | `encoder_5` | 0 |
| 12 | `output_head` | 0 |

## Consensus ranking

### Accuracy only (`val_ce` + `nmse` + `r2`) — preferred for Phase 5 selection

`var_f1` / `sym_rate` were near-constant on this decode budget (see below), so they are excluded here.

| rank | condition | mean metric-rank |
|------|-----------|------------------|
| 1 | `encoder_2` | 2.67 |
| 2 | `encoder_1` | 3.00 |
| 3 | `encoder_3` | 4.67 |
| 4 | `decoder_1` | 4.67 |
| 5 | `decoder_0` | 5.00 |
| 6 | `decoder_4` | 6.33 |
| 7 | `encoder_0` | 7.00 |
| 8 | `decoder_2` | 7.33 |
| 9 | `decoder_3` | 7.67 |
| 10 | `encoder_5` | 8.33 |
| 11 | `encoder_4` | 9.67 |
| 12 | `output_head` | 11.67 |

### Metric split (important)

| Signal | High-contribution layers | Note |
|--------|--------------------------|------|
| Token CE | `decoder_4` > `decoder_3` > `decoder_2` | Matches Phase 3 |
| Decode NMSE / R² | `encoder_2` > `encoder_1` ≈ `encoder_3` | Light BFGS decode |
| Variable F1 | flat (all ≈1 except `all_params` 0.75) | Not discriminative at n=4 |
| Symbolic recovery | 0 for single-layer; 0.25 only for `all_params` | Needs fuller decode / more epochs |

## Notes

- Primary Phase-4 claim uses **separate** contribution tables (accuracy / symbolic / variable), not a single weighted composite.
- Symbolic recovery = max(exact, skeleton-equivalent after constant→c, sympy equiv).
- Variable recovery = presence of true `x_i` names in predicted string (F1).
- Decode BFGS defaults are light (`beam=1`, `restarts=1`, `stop_time=0.5`) for scan throughput; raise `--bfgs-restarts` / `--bfgs-stop-time` for paper numbers.
- For Phase 5, start from top accuracy layers: **`encoder_2`, `encoder_1`, `decoder_4`** (CE-dominant late decoder).

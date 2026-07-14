# Phase 6: TPSR 2×2 (fine-tune × decode)

- High-contrib layers (Phase 5 `middle_3`): `decoder_0, decoder_4, encoder_0`
- Train FT examples: 17; eval test: 2
- Epochs: 5, lr: 0.0001
- Beam BFGS: beam=1, restarts=1, stop=0.5s
- TPSR: rollout=1, horizon=25, width=2, num_beams=1
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase6/tpsr_2x2.json`

## 2×2 results

| Fine-tune | Decode | NMSE med | R² med | var F1 | sym | time/eq (s) | total (s) |
|-----------|--------|----------|--------|--------|-----|------------|----------|
| none | beam | 0.2031 | 0.6844 | 1 | 0 | 2.732 | 7.8 |
| none | TPSR | 0.5287 | 0.3557 | 1 | 0 | 11.3 | 27.2 |
| selective | beam | 0.1933 | 0.7002 | 1 | 0 | 4.042 | 11.9 |
| selective | TPSR | 0.08243 | 0.8793 | 1 | 0 | 6.772 | 15.7 |

## Effect decomposition (NMSE ↓ better)

- Δ FT | beam: NMSE(selective_beam) − NMSE(pretrained_beam) = -0.009755
- Δ TPSR | pretrained: NMSE(pretrained_tpsr) − NMSE(pretrained_beam) = 0.3256
- Δ TPSR | selective: NMSE(selective_tpsr) − NMSE(selective_beam) = -0.1109
- Interaction (NMSE): [selective_tpsr − selective_beam] − [pretrained_tpsr − pretrained_beam] = -0.4365

## Effect decomposition (R² ↑ better)

- Δ FT | beam: 0.01576
- Δ TPSR | pretrained: -0.3287
- Δ TPSR | selective: 0.1792

## Findings

1. **Best cell:** selective FT + TPSR (NMSE 0.082, R² 0.88).
2. **FT alone** (beam): small gain vs pretrained beam.
3. **TPSR alone** (pretrained): *worse* than beam under this light MCTS budget — prior without FT is a poor guide for rollouts.
4. **Positive interaction:** TPSR helps only after selective FT (Δ NMSE −0.11 with FT vs +0.33 without). Consistent with plan goal of separating FT / MCTS / interaction.

## Notes

- Plan Phase 6: separate FT gain, MCTS gain, and interaction.
- TPSR uses NeSymReS backbone + UCT (not E2E).
- Light MCTS/BFGS budgets for CPU; raise `--rollout` / `--horizon` for paper runs.
- Eval n=2; treat as directional smoke for Phase 6.

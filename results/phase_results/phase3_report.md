# Phase 3: selective-layer fine-tuning

- Train problems: 17 tokenized (from Phase 1 train)
- Eval problems: 4 test (NMSE); val CE on tokenized test
- Epochs: 3, lr: 0.0001, max_points: 80
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase3/layer_scan.json`

## Ranking by validation CE (lower is better)

| rank | condition | trainable | train CE | val CE | NMSE med | time (s) |
|------|-----------|-----------|----------|--------|----------|----------|
| 1 | `all_params` | 26,395,708 | 0.105 | 0.1933 | 0.07739 | 20.8 |
| 2 | `all_decoder_blocks` | 13,148,160 | 0.1943 | 0.2879 | 0.09884 | 15.9 |
| 3 | `random_one_each` | 4,760,576 | 0.6705 | 0.7239 | 0.1203 | 49.8 |
| 4 | `decoder_4` | 2,629,632 | 0.83 | 0.7591 | 0.2556 | 35.1 |
| 5 | `decoder_3` | 2,629,632 | 0.734 | 0.8133 | 0.3988 | 35.6 |
| 6 | `decoder_2` | 2,629,632 | 0.9116 | 1.02 | 0.9378 | 34.9 |
| 7 | `middle_decoder` | 2,629,632 | 0.9379 | 1.047 | 0.721 | 37.5 |
| 8 | `decoder_1` | 2,629,632 | 1.273 | 1.321 | 0.3649 | 37.7 |
| 9 | `all_encoder_blocks` | 12,097,536 | 1.089 | 1.358 | 0.1537 | 35.1 |
| 10 | `encoder_1` | 2,130,944 | 1.486 | 1.528 | 0.1696 | 37.3 |
| 11 | `encoder_2` | 2,130,944 | 1.607 | 1.564 | 0.3504 | 39.0 |
| 12 | `decoder_0` | 2,629,632 | 1.549 | 1.634 | 0.2453 | 39.7 |
| 13 | `encoder_3` | 2,130,944 | 1.565 | 1.667 | 0.3832 | 38.0 |
| 14 | `middle_encoder` | 2,130,944 | 1.658 | 1.672 | 0.2067 | 38.6 |
| 15 | `encoder_0` | 1,442,816 | 1.603 | 1.792 | 0.4137 | 16.5 |
| 16 | `encoder_5` | 2,130,944 | 1.787 | 1.815 | 0.3988 | 35.5 |
| 17 | `encoder_4` | 2,130,944 | 1.785 | 1.818 | 0.5542 | 37.5 |
| 18 | `output_head` | 30,780 | 1.887 | 1.854 | 0.2453 | 17.0 |
| 19 | `pretrained` | 0 | nan | 1.945 | 0.2453 | 20.5 |

## Layer contribution preview (CE recovery toward all_params)

- L_base (pretrained) val CE = 1.945
- pretrained NMSE med = 0.2453
- L_full (all_params) val CE = 0.1933

| condition | C_CE = (L_base - L_k) / (L_base - L_full) |
|-----------|------------------------------------------|
| `output_head` | 0.05151 |
| `encoder_0` | 0.08728 |
| `encoder_1` | 0.2378 |
| `encoder_2` | 0.2175 |
| `encoder_3` | 0.1583 |
| `encoder_4` | 0.07211 |
| `encoder_5` | 0.0743 |
| `decoder_0` | 0.1775 |
| `decoder_1` | 0.3559 |
| `decoder_2` | 0.5279 |
| `decoder_3` | 0.646 |
| `decoder_4` | 0.6769 |
| `middle_encoder` | 0.1558 |
| `middle_decoder` | 0.5123 |
| `random_one_each` | 0.6971 |
| `all_encoder_blocks` | 0.335 |
| `all_decoder_blocks` | 0.946 |

## Notes

- Fine-tuning uses teacher-forcing CE on GRN equation skeletons (constants -> `c`).
- Gradients flow through the full model; only selected layers have `requires_grad=True`.
- NMSE uses light BFGS (1 restart, 0.5s) for scan speed; Phase 4 should use fuller decoding.
- LoRA not included in v1.

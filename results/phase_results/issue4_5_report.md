# Issues 4–5: freeze and trainable checks

- Total parameters: 26,395,708

| case | layers | trainable | % |
|------|--------|-----------|---|
| `all_frozen` | (none) | 0 | 0.00% |
| `output_head` | `output_head` | 30,780 | 0.12% |
| `encoder_0` | `encoder_0` | 1,442,816 | 5.47% |
| `decoder_0` | `decoder_0` | 2,629,632 | 9.96% |
| `encoder_0+decoder_0` | `encoder_0`, `decoder_0` | 4,072,448 | 15.43% |
| `all_encoder_blocks` | `encoder_0`, `encoder_1`, `encoder_2`, `encoder_3`, `encoder_4`, `encoder_5` | 12,097,536 | 45.83% |
| `all_decoder_blocks` | `decoder_0`, `decoder_1`, `decoder_2`, `decoder_3`, `decoder_4` | 13,148,160 | 49.81% |

## Acceptance

- `requires_grad=False` on non-selected layers (Issue 4)
- Trainable count matches selected modules only (Issue 5)
- Dummy forward/backward produces grads on selected layers

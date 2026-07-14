# Issue 3: NeSymReS layer list

- Checkpoint: `10M.ckpt`
- Config: `C:/Document/researches/LTSR/NSRS/jupyter/100M/config.yaml` (100M architecture)
- Total parameters: 26,395,708
- Encoder units: 7
- Decoder units: 5

| name | kind | module_path | n_params |
|------|------|-------------|----------|
| `encoder_0` | encoder | `enc.selfatt1` | 1,442,816 |
| `encoder_1` | encoder | `enc.selfatt.0` | 2,130,944 |
| `encoder_2` | encoder | `enc.selfatt.1` | 2,130,944 |
| `encoder_3` | encoder | `enc.selfatt.2` | 2,130,944 |
| `encoder_4` | encoder | `enc.selfatt.3` | 2,130,944 |
| `encoder_5` | encoder | `enc.selfatt.4` | 2,130,944 |
| `encoder_pma` | encoder | `enc.outatt` | 1,057,792 |
| `decoder_0` | decoder | `decoder_transfomer.layers.0` | 2,629,632 |
| `decoder_1` | decoder | `decoder_transfomer.layers.1` | 2,629,632 |
| `decoder_2` | decoder | `decoder_transfomer.layers.2` | 2,629,632 |
| `decoder_3` | decoder | `decoder_transfomer.layers.3` | 2,629,632 |
| `decoder_4` | decoder | `decoder_transfomer.layers.4` | 2,629,632 |
| `output_head` | head | `fc_out` | 30,780 |
| `tok_embedding` | embedding | `tok_embedding` | 30,720 |
| `pos_embedding` | embedding | `pos_embedding` | 30,720 |

## Layer naming

- `encoder_0` = first ISAB (`enc.selfatt1`)
- `encoder_1..N` = remaining ISABs (`enc.selfatt.*`)
- `encoder_pma` = PMA pooling (`enc.outatt`)
- `decoder_i` = `nn.TransformerDecoderLayer` i
- `output_head` = `fc_out`

Next: Issue 4–5 freeze / trainable checks.

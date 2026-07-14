# Phase 7d: SBML-supervised selective fine-tuning (DREAM4 Size10)

- SBML: `C:/Document/researches/LTSR/data/dream4/Size 10/Supplementary information/insilico_size10_1/Goldstandard/insilico_size10_1.xml`
- Teacher ODEs reconstructed from GNW parameters (Hill modules; protein≈mRNA quasi-steady proxy)
- FT problems: 7 genes with ≤2 parents (7 tokenized)
- Layers: `decoder_0, decoder_4, encoder_0`, epochs=8
- Device: `cpu`
- Results: `C:/Document/researches/LTSR/results/phase_results/phase7_sbml_ft/size10_net1_sbml_ft.json`

## Results

| condition | NMSE | R² | time (s) |
|-----------|------|----|----------|
| `pretrained_sbml_holdout` | 0.396 | 0.5867 | 13.8 |
| `sbml_ft_holdout` | 0.009407 | 0.9898 | 11.9 |
| `pretrained_dream_fd` | 0.9724 | 0.02746 | 17.6 |
| `sbml_ft_dream_fd` | 0.9227 | 0.0773 | 16.1 |

## Findings

1. **In-distribution (SBML teacher holdout):** ΔNMSE = -0.3865 (FT − pretrained).
2. **Transfer to noisy DREAM FD:** ΔNMSE = -0.04976.
3. SBML files lack MathML; reconstruction is Hill/module based on GNW params.
4. Multi-regulator / constitutive dynamics remain approximate under the proxy.

## Notes

- Prefer genes with ≤2 parents for cleaner teacher strings.
- Extend with `--net-id` 2..5 or Size100 SBML similarly.

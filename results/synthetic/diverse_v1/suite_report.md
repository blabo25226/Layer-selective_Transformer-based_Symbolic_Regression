# Diverse synthetic suite (A-1)

- Output: `C:/Document/researches/LTSR/results/synthetic/diverse_v1`
- Problems: 90  |  splits: {'train': 60, 'test': 30}
- noise_std: 0.0  |  seed: 0

## Structure split (disjoint skeletons)

- **train** (10): additive_act, hill_act_n2, hill_act_n3, hill_rep_n2, linear2, mass_action, michaelis, self_act_n2, sqrt_sat, toggle_n2
- **test** (5): hill_act_n4, hill_rep_n3, product_hill, ratio_xy, sum_linear3

> TEST skeletons are functional forms absent from TRAIN, so decode metrics measure generalization to unseen structure, not memorization.

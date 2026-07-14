import numpy as np

from src.data.human import select_human_regulators


class Panel:
    n_genes = 4

    def prior_parents(self, target):
        return [0, 1]


def test_prior_correlation_ranks_every_candidate_inside_pool():
    y = np.arange(10, dtype=float)
    X = np.column_stack([
        np.zeros(10), y * 0.5, y, y * 2.0,
    ])
    assert select_human_regulators(Panel(), X, y, target=3, k=1, method="prior_corr") == [1]

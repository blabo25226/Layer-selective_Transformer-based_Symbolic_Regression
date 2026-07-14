"""Build NeSymReS fine-tuning batches from Phase 1 synthetic GRN problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from nesymres.architectures.data import constants_to_placeholder, tokenize, tokens_padding
from nesymres.dataset.generator import Generator, UnknownSymPyOperator

from .synthetic_grn import SampledDataset, load_problem


def instantiate_expr(ds: SampledDataset) -> str:
    """Fill template parameters into a concrete numeric expression string."""
    p = ds.spec.parameters
    fam = ds.spec.family
    if fam == "activation":
        return (
            f"({p['alpha']})*x_2**({p['n']})/"
            f"(({p['K']})**({p['n']})+x_2**({p['n']}))-({p['beta']})*x_1"
        )
    if fam == "repression":
        return (
            f"({p['alpha']})*({p['K']})**({p['n']})/"
            f"(({p['K']})**({p['n']})+x_2**({p['n']}))-({p['beta']})*x_1"
        )
    if fam == "toggle":
        if ds.spec.motif.endswith("dx"):
            return f"({p['alpha1']})/(1+x_2**({p['n1']}))-({p['beta1']})*x_1"
        return f"({p['alpha2']})/(1+x_1**({p['n2']}))-({p['beta2']})*x_2"
    if fam == "repressilator":
        target_idx = int(p["target_idx"])
        repressors = {0: 2, 1: 0, 2: 1}
        ti = target_idx + 1
        ri = repressors[target_idx] + 1
        return f"({p['alpha']})/(1+x_{ri}**({p['n']}))-({p['beta']})*x_{ti}"
    raise ValueError(fam)


def expression_to_tokens(expr: str, word2id: Dict[str, int]) -> Optional[List[int]]:
    try:
        skeleton = constants_to_placeholder(expr)
        prefix = Generator.sympy_to_prefix(skeleton)
        return tokenize(prefix, word2id)
    except (UnknownSymPyOperator, KeyError, RecursionError, Exception):
        return None


def points_to_nesymres_tensor(X: np.ndarray, y: np.ndarray, n_vars: int = 3) -> torch.Tensor:
    """
    Convert (N, d), (N,) to NeSymReS batch tensor shape (4, N):
    [x_1, x_2, x_3, y].
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    n = X.shape[0]
    feats = np.zeros((n_vars, n), dtype=np.float32)
    d = min(X.shape[1], n_vars)
    feats[:d, :] = X[:, :d].T
    out = np.concatenate([feats, y[None, :]], axis=0)  # (4, N)
    return torch.from_numpy(out)


class GRNFinetuneDataset(Dataset):
    """Teacher-forcing dataset: numerical points + tokenized equation skeleton."""

    def __init__(
        self,
        problems: Sequence[SampledDataset],
        word2id: Dict[str, int],
        max_points: int = 100,
        seed: int = 0,
    ):
        self.word2id = word2id
        self.max_points = max_points
        self.rng = np.random.default_rng(seed)
        self.items: List[Tuple[torch.Tensor, List[int], str]] = []
        for ds in problems:
            expr = instantiate_expr(ds)
            toks = expression_to_tokens(expr, word2id)
            if toks is None:
                continue
            n = ds.X.shape[0]
            idx = self.rng.choice(n, size=min(max_points, n), replace=False)
            num = points_to_nesymres_tensor(ds.X[idx], ds.y[idx])
            self.items.append((num, toks, ds.spec.eq_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        num, toks, eq_id = self.items[i]
        return num, toks, eq_id


def collate_finetune(batch):
    nums = torch.stack([b[0] for b in batch], dim=0)  # (B, 4, N)
    tokens = tokens_padding([b[1] for b in batch]).long()
    eq_ids = [b[2] for b in batch]
    return nums, tokens, eq_ids


def load_split_problems(data_dir: Path, split: str) -> List[SampledDataset]:
    index = json.loads((data_dir / "index.json").read_text(encoding="utf-8"))
    out = []
    for item in index:
        if item["split"] != split:
            continue
        out.append(load_problem(data_dir / item["file"]))
    return out

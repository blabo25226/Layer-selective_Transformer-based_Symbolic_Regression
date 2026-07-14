"""Load official DREAM4 In Silico Challenge Size10 / Size100 data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dreamlike_grn import GRNNetwork, build_local_problem
from .synthetic_grn import SampledDataset, EquationSpec


DREAM4_ROOT_CANDIDATES = (
    Path("data/dream4"),
)


def find_dream4_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    # relative to repo root (caller usually chdirs or passes absolute)
    for rel in DREAM4_ROOT_CANDIDATES:
        if rel.exists():
            return rel.resolve()
    raise FileNotFoundError("data/dream4 not found")


def size10_dir(root: Path) -> Path:
    return root / "Size 10"


def _gene_index(name: str) -> int:
    m = re.fullmatch(r"G(\d+)", name.strip().strip('"'))
    if not m:
        raise ValueError(f"Unexpected gene name: {name}")
    return int(m.group(1)) - 1


def load_goldstandard_tsv(path: Path) -> List[Tuple[int, int]]:
    """Return undirected? No — directed edges (regulator, target) where flag==1."""
    edges = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('"', "").split("\t")
            if len(parts) < 3:
                continue
            r, t, flag = parts[0], parts[1], parts[2]
            if flag.strip() in ("1", "1.0", "+"):
                edges.append((_gene_index(r), _gene_index(t)))
            elif flag.strip() == "-":
                # signed file variant without numeric flag
                edges.append((_gene_index(r), _gene_index(t)))
    return edges


def load_signed_goldstandard(path: Path) -> List[Tuple[int, int, str]]:
    """(regulator, target, 'act'|'rep') from signed TSV."""
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace('"', "").split("\t")
            if len(parts) < 3:
                continue
            r, t, s = parts[0], parts[1], parts[2].strip()
            sign = "act" if s in ("+", "1", "act") else "rep"
            out.append((_gene_index(r), _gene_index(t), sign))
    return out


def load_expression_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read DREAM TSV with quoted gene header, no index column."""
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().replace('"', "").split("\t")
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split("\t")])
    return header, np.asarray(rows, dtype=float)


def load_timeseries(path: Path) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """
    Parse DREAM4 timeseries TSV (multiple trajectories separated by blank lines).

    Returns gene_names, list of time vectors, list of X arrays (T, n_genes).
    """
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    header = lines[0].strip().replace('"', "").split("\t")
    assert header[0] == "Time", header
    genes = header[1:]
    traj_t: List[List[float]] = []
    traj_x: List[List[List[float]]] = []
    cur_t: List[float] = []
    cur_x: List[List[float]] = []
    for line in lines[1:]:
        if not line.strip():
            if cur_t:
                traj_t.append(cur_t)
                traj_x.append(cur_x)
                cur_t, cur_x = [], []
            continue
        parts = line.strip().split("\t")
        cur_t.append(float(parts[0]))
        cur_x.append([float(v) for v in parts[1:]])
    if cur_t:
        traj_t.append(cur_t)
        traj_x.append(cur_x)
    times = [np.asarray(t, dtype=float) for t in traj_t]
    xs = [np.asarray(x, dtype=float) for x in traj_x]
    return genes, times, xs


def finite_difference_rhs(
    times: Sequence[np.ndarray],
    xs: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X_state, dX/dt) by forward differences within each trajectory.
    X_state uses value at t_i; derivative (x_{i+1}-x_i)/(t_{i+1}-t_i).
    """
    X_rows = []
    Y_rows = []
    for t, x in zip(times, xs):
        if len(t) < 2:
            continue
        dt = np.diff(t)
        dx = np.diff(x, axis=0)
        # avoid zero dt
        dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
        dydt = dx / dt[:, None]
        X_rows.append(x[:-1])
        Y_rows.append(dydt)
    if not X_rows:
        raise ValueError("No usable timeseries segments")
    return np.vstack(X_rows), np.vstack(Y_rows)


def trajectory_train_test_split(
    times: Sequence[np.ndarray],
    xs: Sequence[np.ndarray],
    *,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split complete trajectories before finite differencing."""
    if len(times) != len(xs) or len(times) < 2:
        raise ValueError("at least two paired trajectories are required")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(times))
    n_test = min(len(times) - 1, max(1, round(len(times) * test_fraction)))
    test_idx = set(int(i) for i in order[:n_test])
    tr_t = [t for i, t in enumerate(times) if i not in test_idx]
    tr_x = [x for i, x in enumerate(xs) if i not in test_idx]
    te_t = [t for i, t in enumerate(times) if i in test_idx]
    te_x = [x for i, x in enumerate(xs) if i in test_idx]
    X_tr, Y_tr = finite_difference_rhs(tr_t, tr_x)
    X_te, Y_te = finite_difference_rhs(te_t, te_x)
    return X_tr, Y_tr, X_te, Y_te


@dataclass
class Dream4Size10Network:
    net_id: int
    n_genes: int
    gene_names: List[str]
    edges: List[Tuple[int, int, str]]
    root: Path

    def as_grn_network(self) -> GRNNetwork:
        return GRNNetwork(n_genes=self.n_genes, edges=list(self.edges), parameters={})

    def parents(self, target: int) -> List[Tuple[int, str]]:
        return [(r, s) for r, t, s in self.edges if t == target]


def load_size10_network(root: Path, net_id: int = 1) -> Dream4Size10Network:
    """Load gold standard (+ signed if available) for insilico_size10_{net_id}."""
    s10 = size10_dir(root)
    gold = s10 / "DREAM4 gold standards" / f"insilico_size10_{net_id}_goldstandard.tsv"
    signed = (
        s10
        / "Supplementary information"
        / f"insilico_size10_{net_id}"
        / "Goldstandard"
        / f"insilico_size10_{net_id}_goldstandard_signed.tsv"
    )
    gene_names = [f"G{i}" for i in range(1, 11)]
    if signed.exists():
        edges = load_signed_goldstandard(signed)
    else:
        edges = [(r, t, "act") for r, t in load_goldstandard_tsv(gold)]
    return Dream4Size10Network(
        net_id=net_id,
        n_genes=10,
        gene_names=gene_names,
        edges=edges,
        root=root,
    )


def load_size10_expression_bundle(
    root: Path, net_id: int = 1
) -> Dict[str, np.ndarray]:
    """
    Returns:
      X_multi: multifactorial steady-ish samples (N, 10)
      X_ts, Y_ts: timeseries states and finite-diff derivatives (M, 10)
      X_ko, X_kd: knockout / knockdown matrices if present
    """
    train = (
        size10_dir(root)
        / "DREAM4 training data"
        / f"insilico_size10_{net_id}"
    )
    _, X_multi = load_expression_matrix(train / f"insilico_size10_{net_id}_multifactorial.tsv")
    genes, times, xs = load_timeseries(train / f"insilico_size10_{net_id}_timeseries.tsv")
    X_ts, Y_ts = finite_difference_rhs(times, xs)
    out = {"X_multi": X_multi, "X_ts": X_ts, "Y_ts": Y_ts,
           "gene_names": np.array(genes), "times": times, "trajectories": xs}
    ko = train / f"insilico_size10_{net_id}_knockouts.tsv"
    kd = train / f"insilico_size10_{net_id}_knockdowns.tsv"
    if ko.exists():
        _, out["X_ko"] = load_expression_matrix(ko)
    if kd.exists():
        _, out["X_kd"] = load_expression_matrix(kd)
    return out


def build_dream4_local_problems(
    network: Dream4Size10Network,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    method: str,
    k: int,
    split: str,
    max_vars: int = 3,
    target_limit: int = 0,
    include_target: bool = True,
    target_ids: Optional[Sequence[int]] = None,
    size_tag: Optional[int] = None,
) -> Tuple[List[SampledDataset], Dict[int, List[int]], List[dict]]:
    """Same contract as Phase 7 dreamlike: per-target local SR problems."""
    from .regulator_selection import (
        select_regulators,
        selection_metrics,
        oracle_regulators,
    )

    grn = network.as_grn_network()
    if target_ids is not None:
        targets = [int(t) for t in target_ids]
    else:
        n_targets = (
            network.n_genes if target_limit <= 0 else min(target_limit, network.n_genes)
        )
        targets = list(range(n_targets))
    tag = size_tag if size_tag is not None else network.n_genes
    problems: List[SampledDataset] = []
    selections: Dict[int, List[int]] = {}
    sel_rows = []
    for t in targets:
        y = Y[:, t]
        regs = select_regulators(method, grn, X, y, t, k=k)
        selections[t] = regs
        true = oracle_regulators(grn, t)
        sm = selection_metrics(true, regs)
        sel_rows.append(
            {
                "target": t,
                "target_name": network.gene_names[t],
                "true": true,
                "pred": regs,
                **sm,
            }
        )
        cols: List[int] = []
        if include_target and t not in regs:
            cols.append(t)
        for r in regs:
            if r not in cols:
                cols.append(int(r))
        cols = cols[:max_vars] or [t]
        Xloc = X[:, cols].astype(float)
        var_names = [f"x_{i+1}" for i in range(len(cols))]
        mapping = ",".join(f"x_{i+1}=G{g+1}" for i, g in enumerate(cols))
        true_parents = [f"G{r+1}" for r in true]
        ds = SampledDataset(
            spec=EquationSpec(
                eq_id=f"dream4_s{tag}_{network.net_id}_{split}_{method}_t{t}",
                family="dream4",
                target_expr=f"f({','.join(var_names)})  # parents={true_parents}",
                variable_names=var_names,
                parameters={
                    "target_gene": float(t),
                    "net_id": float(network.net_id),
                    "size": float(tag),
                    **{f"gene_col_{i}": float(g) for i, g in enumerate(cols)},
                },
                split=split,
                motif=mapping,
            ),
            X=Xloc,
            y=np.asarray(y, dtype=float),
            noise_std=0.0,
        )
        problems.append(ds)
    return problems, selections, sel_rows


def list_size10_net_ids(root: Path) -> List[int]:
    gold_dir = size10_dir(root) / "DREAM4 gold standards"
    ids = []
    for p in sorted(gold_dir.glob("insilico_size10_*_goldstandard.tsv")):
        m = re.search(r"insilico_size10_(\d+)_goldstandard", p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


# --- Generalized Size10 / Size100 API ---

Dream4Network = Dream4Size10Network  # alias


def size_dir(root: Path, size: int) -> Path:
    if size not in (10, 100):
        raise ValueError(f"Unsupported DREAM4 size: {size}")
    return root / f"Size {size}"


def list_net_ids(root: Path, size: int = 10) -> List[int]:
    gold_dir = size_dir(root, size) / "DREAM4 gold standards"
    ids = []
    for p in sorted(gold_dir.glob(f"insilico_size{size}_*_goldstandard.tsv")):
        m = re.search(rf"insilico_size{size}_(\d+)_goldstandard", p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def load_dream4_network(root: Path, size: int = 10, net_id: int = 1) -> Dream4Network:
    """Load Size10 or Size100 gold standard (+ signed if available)."""
    sdir = size_dir(root, size)
    gold = sdir / "DREAM4 gold standards" / f"insilico_size{size}_{net_id}_goldstandard.tsv"
    signed = (
        sdir
        / "Supplementary information"
        / f"insilico_size{size}_{net_id}"
        / "Goldstandard"
        / f"insilico_size{size}_{net_id}_goldstandard_signed.tsv"
    )
    gene_names = [f"G{i}" for i in range(1, size + 1)]
    if signed.exists():
        edges = load_signed_goldstandard(signed)
    else:
        edges = [(r, t, "act") for r, t in load_goldstandard_tsv(gold)]
    return Dream4Network(
        net_id=net_id,
        n_genes=size,
        gene_names=gene_names,
        edges=edges,
        root=root,
    )


def load_dream4_expression_bundle(
    root: Path, size: int = 10, net_id: int = 1
) -> Dict[str, np.ndarray]:
    """
    Timeseries finite-diff always; multifactorial when present (Size10).
    Size100 challenge training set has no multifactorial file.
    """
    train = (
        size_dir(root, size)
        / "DREAM4 training data"
        / f"insilico_size{size}_{net_id}"
    )
    genes, times, xs = load_timeseries(
        train / f"insilico_size{size}_{net_id}_timeseries.tsv"
    )
    X_ts, Y_ts = finite_difference_rhs(times, xs)
    out: Dict[str, np.ndarray] = {
        "X_ts": X_ts,
        "Y_ts": Y_ts,
        "gene_names": np.array(genes),
        "times": times,
        "trajectories": xs,
    }
    multi = train / f"insilico_size{size}_{net_id}_multifactorial.tsv"
    if multi.exists():
        _, out["X_multi"] = load_expression_matrix(multi)
    ko = train / f"insilico_size{size}_{net_id}_knockouts.tsv"
    kd = train / f"insilico_size{size}_{net_id}_knockdowns.tsv"
    if ko.exists():
        _, out["X_ko"] = load_expression_matrix(ko)
    if kd.exists():
        _, out["X_kd"] = load_expression_matrix(kd)
    return out


def targets_with_parents(network: Dream4Network) -> List[int]:
    return sorted({t for _, t, _ in network.edges})

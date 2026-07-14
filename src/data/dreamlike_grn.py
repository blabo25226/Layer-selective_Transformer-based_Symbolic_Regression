"""GeneNetWeaver / DREAM4–style synthetic multi-gene GRNs (Phase 7).

Generates a small gold-standard network, samples RHS points y = dx_i/dt,
and builds per-target local problems after regulator preselection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .synthetic_grn import SampledDataset, EquationSpec, add_gaussian_noise


@dataclass
class GRNNetwork:
    n_genes: int
    edges: List[Tuple[int, int, str]]  # (regulator, target, sign) sign in {act,rep}
    parameters: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def parents(self, target: int) -> List[Tuple[int, str]]:
        return [(r, s) for r, t, s in self.edges if t == target]

    def adjacency(self) -> np.ndarray:
        A = np.zeros((self.n_genes, self.n_genes), dtype=int)
        for r, t, _ in self.edges:
            A[r, t] = 1
        return A


def _default_hill_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "alpha": float(rng.uniform(0.8, 2.0)),
        "K": float(rng.uniform(0.5, 1.5)),
        "n": float(rng.choice([1.0, 2.0, 3.0])),
        "beta": float(rng.uniform(0.3, 0.8)),
        "basal": float(rng.uniform(0.0, 0.2)),
    }


def generate_random_grn(
    n_genes: int = 10,
    edges_per_gene: Tuple[int, int] = (1, 2),
    seed: int = 0,
    allow_self: bool = False,
) -> GRNNetwork:
    """Sparse random GRN with 1–2 regulators per target (no cycles required)."""
    rng = np.random.default_rng(seed)
    edges: List[Tuple[int, int, str]] = []
    params: Dict[str, Dict[str, float]] = {}
    for t in range(n_genes):
        k = int(rng.integers(edges_per_gene[0], edges_per_gene[1] + 1))
        pool = [g for g in range(n_genes) if allow_self or g != t]
        if not pool:
            continue
        k = min(k, len(pool))
        regs = rng.choice(pool, size=k, replace=False).tolist()
        for r in regs:
            sign = "act" if rng.random() < 0.5 else "rep"
            edges.append((int(r), int(t), sign))
            params[f"{r}->{t}"] = _default_hill_params(rng)
    return GRNNetwork(n_genes=n_genes, edges=edges, parameters=params)


def _hill_term(x_reg: np.ndarray, sign: str, p: Dict[str, float]) -> np.ndarray:
    xn = np.power(np.maximum(x_reg, 0.0), p["n"])
    kn = p["K"] ** p["n"]
    if sign == "act":
        return p["alpha"] * xn / (kn + xn + 1e-12)
    return p["alpha"] * kn / (kn + xn + 1e-12)


def rhs_for_target(
    network: GRNNetwork,
    X_all: np.ndarray,
    target: int,
) -> np.ndarray:
    """Compute dx_target/dt from full expression matrix X_all (N, n_genes)."""
    parents = network.parents(target)
    y = np.zeros(X_all.shape[0], dtype=float)
    # degradation always uses target itself
    beta = 0.5
    basal = 0.0
    for r, sign in parents:
        p = network.parameters[f"{r}->{target}"]
        y = y + _hill_term(X_all[:, r], sign, p)
        beta = p["beta"]
        basal = p.get("basal", 0.0)
    if not parents:
        # constitutive decay only
        basal = 0.1
        beta = 0.5
    y = y + basal - beta * X_all[:, target]
    return y


def sample_expression(
    network: GRNNetwork,
    n_points: int,
    support: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    lo, hi = support
    return rng.uniform(lo, hi, size=(n_points, network.n_genes))


def true_local_expr(network: GRNNetwork, target: int, *, numeric: bool = False) -> str:
    """Human-readable skeleton (or numeric) with gene indices as x_{i+1}."""
    parts = []
    parents = network.parents(target)
    beta = 0.5
    basal = 0.0
    for r, sign in parents:
        p = network.parameters[f"{r}->{target}"]
        ri = r + 1
        if numeric:
            a, K, n = p["alpha"], p["K"], p["n"]
            if sign == "act":
                parts.append(
                    f"({a:g})*x_{ri}**({n:g})/(({K:g})**({n:g})+x_{ri}**({n:g}))"
                )
            else:
                parts.append(
                    f"({a:g})*({K:g})**({n:g})/(({K:g})**({n:g})+x_{ri}**({n:g}))"
                )
        else:
            if sign == "act":
                parts.append(f"alpha*x_{ri}**n/(K**n+x_{ri}**n)")
            else:
                parts.append(f"alpha*K**n/(K**n+x_{ri}**n)")
        beta = p["beta"]
        basal = p.get("basal", 0.0)
    if basal:
        parts.insert(0, f"{basal:g}")
    body = "+".join(parts) if parts else "0"
    ti = target + 1
    if numeric:
        return f"{body}-({beta:g})*x_{ti}"
    return f"{body}-beta*x_{ti}"


def build_local_problem(
    network: GRNNetwork,
    X_all: np.ndarray,
    y_target: np.ndarray,
    target: int,
    candidate_regs: Sequence[int],
    *,
    eq_id: str,
    split: str,
    include_target: bool = True,
    max_vars: int = 3,
    selection_method: str = "oracle",
) -> SampledDataset:
    """
    Build a NeSymReS-local problem: columns are [target?, candidates...] capped at max_vars.
    Variable names remapped to x_1..x_d in column order.
    """
    cols: List[int] = []
    if include_target and target not in candidate_regs:
        cols.append(target)
    for r in candidate_regs:
        if r not in cols:
            cols.append(int(r))
    cols = cols[:max_vars]
    if not cols:
        cols = [target]

    X = X_all[:, cols].astype(float)
    # remap true expr variable indices to local x_1.. (best-effort skeleton)
    gene_to_local = {g: i + 1 for i, g in enumerate(cols)}
    import re

    def remap(expr: str) -> str:
        def repl(m):
            g = int(m.group(1)) - 1
            if g in gene_to_local:
                return f"x_{gene_to_local[g]}"
            return m.group(0)

        return re.sub(r"x_(\d+)", repl, expr)

    local_expr = remap(true_local_expr(network, target, numeric=False))
    numeric_expr = remap(true_local_expr(network, target, numeric=True))
    var_names = [f"x_{i+1}" for i in range(len(cols))]
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="dreamlike",
            target_expr=local_expr,
            variable_names=var_names,
            parameters={
                "target_gene": float(target),
                "genes": float(len(cols)),
                **{f"gene_col_{i}": float(g) for i, g in enumerate(cols)},
            },
            split=split,
            # motif carries numeric expression for fine-tune tokenization
            motif=numeric_expr,
        ),
        X=X,
        y=np.asarray(y_target, dtype=float),
        noise_std=0.0,
    )


def generate_dreamlike_dataset(
    out_dir: Path,
    *,
    n_genes: int = 10,
    n_train_points: int = 200,
    n_test_points: int = 100,
    support: Tuple[float, float] = (0.1, 2.0),
    noise_std: float = 0.02,
    seed: int = 7,
) -> Dict:
    """Write network gold standard + full expression matrices."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    network = generate_random_grn(n_genes=n_genes, seed=seed)

    X_train = sample_expression(network, n_train_points, support, rng)
    X_test = sample_expression(network, n_test_points, support, rng)

    Y_train = np.stack(
        [rhs_for_target(network, X_train, t) for t in range(n_genes)], axis=1
    )
    Y_test = np.stack(
        [rhs_for_target(network, X_test, t) for t in range(n_genes)], axis=1
    )
    Y_train = add_gaussian_noise(Y_train, noise_std, rng)
    Y_test = add_gaussian_noise(Y_test, noise_std, rng)

    meta = {
        "n_genes": n_genes,
        "n_edges": len(network.edges),
        "edges": [
            {"regulator": r, "target": t, "sign": s} for r, t, s in network.edges
        ],
        "parameters": network.parameters,
        "support": list(support),
        "noise_std": noise_std,
        "seed": seed,
    }
    (out_dir / "network.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "expression.npz",
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
    )
    return meta


def load_network(path: Path) -> GRNNetwork:
    meta = json.loads(path.read_text(encoding="utf-8"))
    edges = [(e["regulator"], e["target"], e["sign"]) for e in meta["edges"]]
    return GRNNetwork(
        n_genes=meta["n_genes"],
        edges=edges,
        parameters=meta.get("parameters", {}),
    )


def load_expression(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}

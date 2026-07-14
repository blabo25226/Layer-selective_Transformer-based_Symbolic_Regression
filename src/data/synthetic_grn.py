"""Synthetic GRN / Hill equation datasets for symbolic regression (Phase 1 / Issue 6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EquationSpec:
    """One symbolic regression problem: y = f(X)."""

    eq_id: str
    family: str  # activation | repression | toggle | repressilator
    target_expr: str  # human-readable sympy-like string
    variable_names: List[str]
    parameters: Dict[str, float]
    split: str = "train"  # train | val | test
    motif: str = ""


@dataclass
class SampledDataset:
    spec: EquationSpec
    X: np.ndarray  # (n_points, n_vars)
    y: np.ndarray  # (n_points,)
    noise_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "X": self.X.tolist(),
            "y": self.y.tolist(),
            "noise_std": self.noise_std,
        }


def hill_activation(x: np.ndarray, y: np.ndarray, alpha: float, k: float, n: float, beta: float) -> np.ndarray:
    """dx/dt = alpha * y^n / (K^n + y^n) - beta * x"""
    yn = np.power(np.maximum(y, 0.0), n)
    kn = k**n
    return alpha * yn / (kn + yn + 1e-12) - beta * x


def hill_repression(x: np.ndarray, y: np.ndarray, alpha: float, k: float, n: float, beta: float) -> np.ndarray:
    """dx/dt = alpha * K^n / (K^n + y^n) - beta * x"""
    yn = np.power(np.maximum(y, 0.0), n)
    kn = k**n
    return alpha * kn / (kn + yn + 1e-12) - beta * x


def toggle_dx(x: np.ndarray, y: np.ndarray, alpha: float, n: float, beta: float) -> np.ndarray:
    """dx/dt = alpha / (1 + y^n) - beta * x"""
    return alpha / (1.0 + np.power(np.maximum(y, 0.0), n) + 1e-12) - beta * x


def repressilator_dxi(
    xi: np.ndarray, x_repressor: np.ndarray, alpha: float, n: float, beta: float
) -> np.ndarray:
    """dxi/dt = alpha / (1 + x_repressor^n) - beta * xi"""
    return alpha / (1.0 + np.power(np.maximum(x_repressor, 0.0), n) + 1e-12) - beta * xi


def _sample_box(
    n_points: int,
    low: Sequence[float],
    high: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    low_a = np.asarray(low, dtype=float)
    high_a = np.asarray(high, dtype=float)
    return rng.uniform(low_a, high_a, size=(n_points, len(low_a)))


def add_gaussian_noise(y: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    if noise_std <= 0:
        return y.copy()
    scale = noise_std * (np.std(y) + 1e-8)
    return y + rng.normal(0.0, scale, size=y.shape)


def make_activation_problem(
    eq_id: str,
    params: Dict[str, float],
    split: str,
    n_points: int,
    support: Tuple[float, float],
    noise_std: float,
    rng: np.random.Generator,
) -> SampledDataset:
    alpha, k, n, beta = params["alpha"], params["K"], params["n"], params["beta"]
    X = _sample_box(n_points, [support[0], support[0]], [support[1], support[1]], rng)
    x, yreg = X[:, 0], X[:, 1]
    y = hill_activation(x, yreg, alpha, k, n, beta)
    y = add_gaussian_noise(y, noise_std, rng)
    expr = f"alpha*x_2**n/(K**n+x_2**n)-beta*x_1"
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="activation",
            target_expr=expr,
            variable_names=["x_1", "x_2"],
            parameters=params,
            split=split,
            motif="single_activation",
        ),
        X=X,
        y=y,
        noise_std=noise_std,
    )


def make_repression_problem(
    eq_id: str,
    params: Dict[str, float],
    split: str,
    n_points: int,
    support: Tuple[float, float],
    noise_std: float,
    rng: np.random.Generator,
) -> SampledDataset:
    alpha, k, n, beta = params["alpha"], params["K"], params["n"], params["beta"]
    X = _sample_box(n_points, [support[0], support[0]], [support[1], support[1]], rng)
    x, yreg = X[:, 0], X[:, 1]
    y = hill_repression(x, yreg, alpha, k, n, beta)
    y = add_gaussian_noise(y, noise_std, rng)
    expr = f"alpha*K**n/(K**n+x_2**n)-beta*x_1"
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="repression",
            target_expr=expr,
            variable_names=["x_1", "x_2"],
            parameters=params,
            split=split,
            motif="single_repression",
        ),
        X=X,
        y=y,
        noise_std=noise_std,
    )


def make_toggle_problem(
    eq_id: str,
    params: Dict[str, float],
    split: str,
    which: str,
    n_points: int,
    support: Tuple[float, float],
    noise_std: float,
    rng: np.random.Generator,
) -> SampledDataset:
    """Toggle switch: learn either dx/dt or dy/dt as a 2-variable problem."""
    X = _sample_box(n_points, [support[0], support[0]], [support[1], support[1]], rng)
    x, yv = X[:, 0], X[:, 1]
    if which == "dx":
        y = toggle_dx(x, yv, params["alpha1"], params["n1"], params["beta1"])
        expr = "alpha1/(1+x_2**n1)-beta1*x_1"
        target = "dx"
    else:
        y = toggle_dx(yv, x, params["alpha2"], params["n2"], params["beta2"])
        expr = "alpha2/(1+x_1**n2)-beta2*x_2"
        target = "dy"
    y = add_gaussian_noise(y, noise_std, rng)
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="toggle",
            target_expr=expr,
            variable_names=["x_1", "x_2"],
            parameters={**params, "target": float(0 if target == "dx" else 1)},
            split=split,
            motif=f"toggle_{target}",
        ),
        X=X,
        y=y,
        noise_std=noise_std,
    )


def make_repressilator_problem(
    eq_id: str,
    params: Dict[str, float],
    split: str,
    target_idx: int,
    n_points: int,
    support: Tuple[float, float],
    noise_std: float,
    rng: np.random.Generator,
) -> SampledDataset:
    """3-gene repressilator; learn d x_i /dt from (x1,x2,x3)."""
    X = _sample_box(
        n_points,
        [support[0]] * 3,
        [support[1]] * 3,
        rng,
    )
    alpha, n, beta = params["alpha"], params["n"], params["beta"]
    # classic cycle: 1<-3, 2<-1, 3<-2
    repressors = {0: 2, 1: 0, 2: 1}
    xi = X[:, target_idx]
    xr = X[:, repressors[target_idx]]
    y = repressilator_dxi(xi, xr, alpha, n, beta)
    y = add_gaussian_noise(y, noise_std, rng)
    repressor_name = f"x_{repressors[target_idx] + 1}"
    target_name = f"x_{target_idx + 1}"
    expr = f"alpha/(1+{repressor_name}**n)-beta*{target_name}"
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="repressilator",
            target_expr=expr,
            variable_names=["x_1", "x_2", "x_3"],
            parameters={**params, "target_idx": float(target_idx)},
            split=split,
            motif=f"repressilator_d{target_name}",
        ),
        X=X,
        y=y,
        noise_std=noise_std,
    )


# Default parameter grids: train / test use different ranges (parameter-range split)
DEFAULT_PARAM_POOLS = {
    "activation": {
        "train": [
            {"alpha": 1.0, "K": 1.0, "n": 2.0, "beta": 0.5},
            {"alpha": 1.5, "K": 0.8, "n": 2.0, "beta": 0.4},
            {"alpha": 2.0, "K": 1.2, "n": 3.0, "beta": 0.6},
            {"alpha": 1.2, "K": 1.5, "n": 2.0, "beta": 0.3},
        ],
        "test": [
            {"alpha": 2.5, "K": 0.6, "n": 4.0, "beta": 0.7},
            {"alpha": 0.8, "K": 2.0, "n": 3.0, "beta": 0.2},
        ],
    },
    "repression": {
        "train": [
            {"alpha": 1.0, "K": 1.0, "n": 2.0, "beta": 0.5},
            {"alpha": 1.5, "K": 0.8, "n": 2.0, "beta": 0.4},
            {"alpha": 2.0, "K": 1.2, "n": 3.0, "beta": 0.6},
        ],
        "test": [
            {"alpha": 2.5, "K": 0.5, "n": 4.0, "beta": 0.8},
            {"alpha": 0.7, "K": 1.8, "n": 3.0, "beta": 0.25},
        ],
    },
    "toggle": {
        "train": [
            {"alpha1": 1.0, "n1": 2.0, "beta1": 0.5, "alpha2": 1.0, "n2": 2.0, "beta2": 0.5},
            {"alpha1": 1.5, "n1": 3.0, "beta1": 0.4, "alpha2": 1.2, "n2": 2.0, "beta2": 0.6},
        ],
        "test": [
            {"alpha1": 2.0, "n1": 4.0, "beta1": 0.3, "alpha2": 1.8, "n2": 3.0, "beta2": 0.7},
        ],
    },
    "repressilator": {
        "train": [
            {"alpha": 1.5, "n": 2.0, "beta": 0.5},
            {"alpha": 2.0, "n": 3.0, "beta": 0.4},
        ],
        "test": [
            {"alpha": 2.5, "n": 4.0, "beta": 0.6},
        ],
    },
}


def build_phase1_suite(
    n_points: int = 200,
    support: Tuple[float, float] = (0.0, 3.0),
    noise_std: float = 0.0,
    seed: int = 0,
) -> List[SampledDataset]:
    """
    Build train/test problems with:
    - structure split (families present in both, but identities differ)
    - parameter-range split within family
    """
    rng = np.random.default_rng(seed)
    datasets: List[SampledDataset] = []
    counter = 0

    for split in ("train", "test"):
        for params in DEFAULT_PARAM_POOLS["activation"][split]:
            counter += 1
            datasets.append(
                make_activation_problem(
                    f"act_{split}_{counter}",
                    params,
                    split,
                    n_points,
                    support,
                    noise_std,
                    rng,
                )
            )
        for params in DEFAULT_PARAM_POOLS["repression"][split]:
            counter += 1
            datasets.append(
                make_repression_problem(
                    f"rep_{split}_{counter}",
                    params,
                    split,
                    n_points,
                    support,
                    noise_std,
                    rng,
                )
            )
        for params in DEFAULT_PARAM_POOLS["toggle"][split]:
            for which in ("dx", "dy"):
                counter += 1
                datasets.append(
                    make_toggle_problem(
                        f"tog_{which}_{split}_{counter}",
                        params,
                        split,
                        which,
                        n_points,
                        support,
                        noise_std,
                        rng,
                    )
                )
        for params in DEFAULT_PARAM_POOLS["repressilator"][split]:
            for target_idx in range(3):
                counter += 1
                datasets.append(
                    make_repressilator_problem(
                        f"rpl_x{target_idx+1}_{split}_{counter}",
                        params,
                        split,
                        target_idx,
                        n_points,
                        support,
                        noise_std,
                        rng,
                    )
                )
    return datasets


def _evaluate_expr_on_X(expr: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a numeric x_1..x_n expression on rows of X via sympy lambdify."""
    from sympy import lambdify, symbols, sympify

    n_vars = X.shape[1]
    syms = symbols(" ".join(f"x_{i+1}" for i in range(max(n_vars, 1))))
    if not isinstance(syms, (list, tuple)):
        syms = (syms,)
    fn = lambdify(syms, sympify(expr), modules=["numpy"])
    cols = [X[:, i] for i in range(n_vars)]
    out = np.asarray(fn(*cols), dtype=float)
    return np.broadcast_to(out, (X.shape[0],)).astype(np.float32)


def make_expr_problem(
    eq_id: str,
    expr: str,
    n_vars: int,
    skeleton: str,
    split: str,
    n_points: int,
    support: Tuple[float, float],
    noise_std: float,
    rng: np.random.Generator,
) -> SampledDataset:
    """Build one SR problem from a concrete numeric x_1..x_n expression string.

    The numeric expression is stored in ``motif`` so the existing dreamlike
    plumbing (``finetune_dataset.instantiate_expr``) returns it as the teacher.
    """
    X = _sample_box(n_points, [support[0]] * n_vars, [support[1]] * n_vars, rng)
    y = _evaluate_expr_on_X(expr, X)
    y = add_gaussian_noise(y, noise_std, rng)
    return SampledDataset(
        spec=EquationSpec(
            eq_id=eq_id,
            family="dreamlike",  # reuse dreamlike teacher path (motif = numeric expr)
            target_expr=expr,
            variable_names=[f"x_{i+1}" for i in range(n_vars)],
            parameters={"skeleton": 0.0},
            split=split,
            motif=expr,
        ),
        X=X,
        y=y,
        noise_std=noise_std,
    )


def _r(rng: np.random.Generator, lo: float, hi: float) -> float:
    return round(float(rng.uniform(lo, hi)), 3)


# Skeleton templates: name -> (n_vars, builder(rng) -> numeric expr string).
# Split is by STRUCTURE — TRAIN and TEST skeletons are disjoint, so the model is
# evaluated on functional forms it never saw during fine-tuning (plan §Phase 1,
# reviewer note A-1: many distinct skeletons, not 4).
def _diverse_skeletons() -> Dict[str, Dict[str, "Tuple[int, Callable]"]]:
    def hill_act(n):
        return lambda g: (
            2,
            f"{_r(g,0.5,2.5)}*x_2**{n}/({_r(g,0.3,1.8)}+x_2**{n})-{_r(g,0.2,0.8)}*x_1",
        )

    def hill_rep(n):
        return lambda g: (
            2,
            f"{_r(g,0.5,2.5)}*{_r(g,0.5,2.0)}/({_r(g,0.3,1.8)}+x_2**{n})-{_r(g,0.2,0.8)}*x_1",
        )

    train = {
        "hill_act_n2": hill_act(2),
        "hill_act_n3": hill_act(3),
        "hill_rep_n2": hill_rep(2),
        "toggle_n2": lambda g: (2, f"{_r(g,0.5,2.5)}/(1+x_2**2)-{_r(g,0.2,0.8)}*x_1"),
        "linear2": lambda g: (2, f"{_r(g,0.3,2.0)}*x_2-{_r(g,0.2,0.9)}*x_1"),
        "mass_action": lambda g: (3, f"{_r(g,0.3,1.5)}*x_2*x_3-{_r(g,0.2,0.8)}*x_1"),
        "michaelis": lambda g: (2, f"{_r(g,0.5,2.5)}*x_2/({_r(g,0.3,1.8)}+x_2)-{_r(g,0.2,0.8)}*x_1"),
        "self_act_n2": lambda g: (
            1,
            f"{_r(g,0.5,2.5)}*x_1**2/({_r(g,0.3,1.8)}+x_1**2)-{_r(g,0.2,0.8)}*x_1",
        ),
        "additive_act": lambda g: (
            3,
            f"{_r(g,0.4,1.6)}*x_2**2/({_r(g,0.3,1.5)}+x_2**2)"
            f"+{_r(g,0.4,1.6)}*x_3**2/({_r(g,0.3,1.5)}+x_3**2)-{_r(g,0.2,0.8)}*x_1",
        ),
        "sqrt_sat": lambda g: (2, f"{_r(g,0.5,2.0)}*sqrt(x_2)-{_r(g,0.2,0.8)}*x_1"),
    }
    test = {
        "hill_act_n4": hill_act(4),
        "hill_rep_n3": hill_rep(3),
        "product_hill": lambda g: (
            3,
            f"{_r(g,0.5,2.0)}*x_2**2*x_3/({_r(g,0.3,1.5)}+x_2**2)-{_r(g,0.2,0.8)}*x_1",
        ),
        "ratio_xy": lambda g: (3, f"{_r(g,0.5,2.0)}*x_2/({_r(g,0.5,2.0)}+x_3)-{_r(g,0.2,0.8)}*x_1"),
        "sum_linear3": lambda g: (
            3,
            f"{_r(g,0.3,1.5)}*x_2+{_r(g,0.3,1.5)}*x_3-{_r(g,0.2,0.9)}*x_1",
        ),
    }
    return {"train": train, "test": test}


def build_diverse_suite(
    n_per_skeleton: int = 8,
    n_points: int = 200,
    support: Tuple[float, float] = (0.1, 3.0),
    noise_std: float = 0.0,
    seed: int = 0,
) -> List[SampledDataset]:
    """Structure-split suite with many distinct skeletons (reviewer note A-1).

    TRAIN and TEST use **disjoint** functional forms; within each skeleton,
    ``n_per_skeleton`` random parameterizations are drawn. With defaults this is
    ~11 train skeletons × 8 = 88 train and 5 test skeletons × 8 = 40 test problems.
    """
    rng = np.random.default_rng(seed)
    skels = _diverse_skeletons()
    datasets: List[SampledDataset] = []
    counter = 0
    for split in ("train", "test"):
        for sk_name, builder in skels[split].items():
            for j in range(n_per_skeleton):
                counter += 1
                n_vars, expr = builder(rng)
                datasets.append(
                    make_expr_problem(
                        f"{sk_name}_{split}_{j}",
                        expr,
                        n_vars,
                        sk_name,
                        split,
                        n_points,
                        support,
                        noise_std,
                        rng,
                    )
                )
    return datasets


def save_suite(datasets: List[SampledDataset], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for ds in datasets:
        path = out_dir / f"{ds.spec.eq_id}.npz"
        np.savez_compressed(
            path,
            X=ds.X,
            y=ds.y,
            noise_std=np.array([ds.noise_std]),
            meta=np.array(json.dumps(asdict(ds.spec))),
        )
        index.append(
            {
                "eq_id": ds.spec.eq_id,
                "family": ds.spec.family,
                "split": ds.spec.split,
                "motif": ds.spec.motif,
                "n_points": int(ds.X.shape[0]),
                "n_vars": int(ds.X.shape[1]),
                "file": path.name,
                "target_expr": ds.spec.target_expr,
                "parameters": ds.spec.parameters,
            }
        )
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def load_problem(path: Path) -> SampledDataset:
    data = np.load(path, allow_pickle=False)
    meta_raw = data["meta"]
    meta_str = meta_raw.item() if getattr(meta_raw, "ndim", 0) == 0 else str(meta_raw)
    if isinstance(meta_str, bytes):
        meta_str = meta_str.decode("utf-8")
    meta = json.loads(meta_str)
    spec = EquationSpec(**meta)
    return SampledDataset(
        spec=spec,
        X=data["X"],
        y=data["y"],
        noise_std=float(np.asarray(data["noise_std"]).reshape(-1)[0]),
    )

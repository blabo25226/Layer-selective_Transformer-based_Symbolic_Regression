"""Human expression time-series helpers for Phase 8.

Primary demo source: GEO GSE112372
  Time-series of human LPS-stimulated monocyte-derived macrophages.
  TPM matrix (~1 MB) + sample metadata.

True ODEs are unknown. Evaluation uses:
  - held-out predictive NMSE / R2 on dx/dt proxies
  - consistency with a curated TF–target prior (literature / DoRothEA-style)
"""

from __future__ import annotations

import gzip
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dreamlike_grn import build_local_problem
from .regulator_selection import correlation_select, lasso_select, mi_select
from .synthetic_grn import SampledDataset

GSE112372_TPM_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112372/suppl/"
    "GSE112372_TPM_genes.txt.gz"
)
GSE112372_META_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE112nnn/GSE112372/suppl/"
    "GSE112372_metadata_macrophageLPS_samples.txt.gz"
)

# Curated macrophage LPS inflammatory panel (human gene symbols).
DEFAULT_PANEL: List[str] = [
    # TFs / signaling hubs
    "RELA",
    "NFKB1",
    "NFKBIA",
    "IRF3",
    "IRF7",
    "STAT1",
    "STAT2",
    "JUN",
    "FOS",
    "SPI1",
    "CEBPB",
    # cytokine / chemokine targets
    "TNF",
    "IL6",
    "IL1B",
    "CXCL8",
    "CXCL10",
    "CCL5",
    "IFNB1",
    "CD40",
    "PTGS2",
]

# GSE112372 TPM rows are Ensembl gene IDs (GRCh38), not symbols.
SYMBOL_TO_ENSEMBL: Dict[str, str] = {
    "RELA": "ENSG00000173039",
    "NFKB1": "ENSG00000109320",
    "NFKBIA": "ENSG00000100906",
    "IRF3": "ENSG00000126456",
    "IRF7": "ENSG00000185507",
    "STAT1": "ENSG00000115415",
    "STAT2": "ENSG00000170581",
    "JUN": "ENSG00000177606",
    "FOS": "ENSG00000170345",
    "SPI1": "ENSG00000066336",
    "CEBPB": "ENSG00000172216",
    "TNF": "ENSG00000232810",
    "IL6": "ENSG00000136244",
    "IL1B": "ENSG00000125538",
    "CXCL8": "ENSG00000169429",
    "CXCL10": "ENSG00000169245",
    "CCL5": "ENSG00000271503",
    "IFNB1": "ENSG00000157601",
    "CD40": "ENSG00000101017",
    "PTGS2": "ENSG00000073756",
}

# Soft gold: literature-style directed TF -> target edges for this panel.
DEFAULT_PRIOR_EDGES: List[Tuple[str, str]] = [
    ("RELA", "TNF"),
    ("RELA", "IL6"),
    ("RELA", "IL1B"),
    ("RELA", "NFKBIA"),
    ("RELA", "CXCL8"),
    ("RELA", "CCL5"),
    ("RELA", "PTGS2"),
    ("NFKB1", "TNF"),
    ("NFKB1", "IL6"),
    ("NFKB1", "NFKBIA"),
    ("NFKB1", "CXCL8"),
    ("IRF3", "IFNB1"),
    ("IRF3", "CXCL10"),
    ("IRF3", "CCL5"),
    ("IRF7", "IFNB1"),
    ("IRF7", "CXCL10"),
    ("STAT1", "CXCL10"),
    ("STAT1", "CD40"),
    ("STAT2", "CXCL10"),
    ("JUN", "IL6"),
    ("JUN", "TNF"),
    ("FOS", "IL6"),
    ("FOS", "TNF"),
    ("CEBPB", "IL6"),
    ("CEBPB", "IL1B"),
    ("SPI1", "CD40"),
]

TIME_PATTERNS = (
    (re.compile(r"^T0(?:_|$)", re.I), 0.0),
    (re.compile(r"T30m", re.I), 0.5),
    (re.compile(r"T3h", re.I), 3.0),
    (re.compile(r"T8h", re.I), 8.0),
    (re.compile(r"T16h", re.I), 16.0),
)


@dataclass
class HumanPanelDataset:
    source: str
    gene_names: List[str]
    times: np.ndarray  # (T,) hours
    X_mean: np.ndarray  # (T, G) mean log1p expression over donors
    X_donors: Dict[str, np.ndarray]  # donor_id -> (T, G)
    prior_edges: List[Tuple[str, str]] = field(default_factory=list)
    sample_meta: Dict[str, str] = field(default_factory=dict)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    def gene_index(self, name: str) -> int:
        return self.gene_names.index(name)

    def prior_parents(self, target: int) -> List[int]:
        tname = self.gene_names[target]
        out = []
        for r, t in self.prior_edges:
            if t == tname and r in self.gene_names:
                out.append(self.gene_index(r))
        return out

    def as_grn_like(self):
        """Minimal network interface for local-problem builders (no true ODE)."""
        edges = []
        params = {}
        for r, t in self.prior_edges:
            if r in self.gene_names and t in self.gene_names:
                ri, ti = self.gene_index(r), self.gene_index(t)
                edges.append((ri, ti, "act"))
                # Placeholder Hill params — true human ODEs are unknown.
                params[f"{ri}->{ti}"] = {
                    "alpha": 1.0,
                    "K": 1.0,
                    "n": 2.0,
                    "beta": 0.5,
                    "basal": 0.0,
                }
        from .dreamlike_grn import GRNNetwork

        return GRNNetwork(n_genes=self.n_genes, edges=edges, parameters=params)


def download_file(url: str, dest: Path, timeout: int = 120) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — curated GEO URL
    tmp.replace(dest)
    return dest


def parse_time_hours(sample_name: str) -> Optional[float]:
    name = sample_name.strip().strip('"')
    for pat, hours in TIME_PATTERNS:
        if pat.search(name):
            return hours
    return None


def parse_donor_id(sample_name: str) -> str:
    """Extract trailing donor token (e.g. T3h_LPS_11 -> 11)."""
    name = sample_name.strip().strip('"')
    m = re.search(r"_(\d+)$", name)
    return m.group(1) if m else "unknown"


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def load_tpm_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load GSE112372-style TPM table.

    Expected: first column gene symbol / id, remaining columns sample TPMs.
    Returns gene_ids, sample_names, matrix (G, S).
    """
    with _open_text(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        # header may start with gene column name
        if len(header) < 2:
            raise ValueError(f"Unexpected TPM header in {path}")
        # Some GEO files include a leading empty or 'Gene' column name
        sample_names = header[1:]
        genes: List[str] = []
        rows: List[List[float]] = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            genes.append(parts[0].strip().strip('"'))
            rows.append([float(x) if x not in ("", "NA", "nan") else 0.0 for x in parts[1:]])
    mat = np.asarray(rows, dtype=float)
    if mat.shape[1] != len(sample_names):
        raise ValueError(
            f"TPM shape mismatch: genes x samples = {mat.shape}, "
            f"n_samples header={len(sample_names)}"
        )
    return genes, sample_names, mat


def load_metadata_sample_names(path: Path) -> List[str]:
    """Read sample identifiers from metadata (optional; TPM header often enough)."""
    names = []
    with _open_text(path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if not parts:
                continue
            if i == 0 and parts[0].lower() in ("sample", "sample_id", "title"):
                continue
            names.append(parts[0].strip().strip('"'))
    return names


def log1p_normalize(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def _resolve_panel_indices(
    genes: Sequence[str],
    panel: Sequence[str],
    symbol_to_ensembl: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], List[int], List[str]]:
    """Map panel symbols to TPM row indices (symbol or Ensembl)."""
    symbol_to_ensembl = symbol_to_ensembl or SYMBOL_TO_ENSEMBL
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    present: List[str] = []
    idxs: List[int] = []
    missing: List[str] = []
    for sym in panel:
        key = sym.upper()
        ens = symbol_to_ensembl.get(sym, symbol_to_ensembl.get(key, "")).upper()
        if key in gene_to_idx:
            present.append(sym)
            idxs.append(gene_to_idx[key])
        elif ens and ens in gene_to_idx:
            present.append(sym)
            idxs.append(gene_to_idx[ens])
        else:
            missing.append(sym)
    return present, idxs, missing


def build_panel_from_tpm(
    genes: Sequence[str],
    samples: Sequence[str],
    tpm: np.ndarray,
    panel: Sequence[str] = DEFAULT_PANEL,
    prior_edges: Sequence[Tuple[str, str]] = DEFAULT_PRIOR_EDGES,
) -> HumanPanelDataset:
    """Subset TPM to panel genes; average donors at shared timepoints."""
    present, gene_idx, missing = _resolve_panel_indices(genes, panel)
    if len(present) < 5:
        raise RuntimeError(
            f"Too few panel genes found in TPM ({len(present)}). Missing e.g. {missing[:8]}"
        )

    # sample -> (time, donor)
    usable = []
    for s_i, sname in enumerate(samples):
        th = parse_time_hours(sname)
        if th is None:
            continue
        usable.append((s_i, th, parse_donor_id(sname), sname))
    if not usable:
        raise RuntimeError("No samples matched known LPS time labels (T0/T30m/T3h/...)")

    times_sorted = sorted({th for _, th, _, _ in usable})
    time_to_row = {t: i for i, t in enumerate(times_sorted)}
    donors = sorted({d for _, _, d, _ in usable})

    G = len(present)
    T = len(times_sorted)
    sub = tpm[gene_idx, :]  # (G, S)

    donor_tensors: Dict[str, np.ndarray] = {
        d: np.full((T, G), np.nan, dtype=float) for d in donors
    }
    for s_i, th, donor, _ in usable:
        r = time_to_row[th]
        donor_tensors[donor][r, :] = log1p_normalize(sub[:, s_i])

    # fill remaining NaN by donor-mean of available; then mean over donors
    stacks = []
    kept_donors = {}
    for d, arr in donor_tensors.items():
        if np.isnan(arr).all():
            continue
        # interpolate NaNs along time per gene if partial
        filled = arr.copy()
        for g in range(G):
            col = filled[:, g]
            if np.isnan(col).all():
                filled[:, g] = 0.0
                continue
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                continue
            if ok.sum() < T:
                filled[~ok, g] = np.interp(
                    np.asarray(times_sorted)[~ok],
                    np.asarray(times_sorted)[ok],
                    col[ok],
                )
        kept_donors[d] = filled
        stacks.append(filled)

    X_mean = np.mean(np.stack(stacks, axis=0), axis=0)
    prior = [(r, t) for r, t in prior_edges if r in present and t in present]
    return HumanPanelDataset(
        source="GSE112372",
        gene_names=list(present),
        times=np.asarray(times_sorted, dtype=float),
        X_mean=X_mean,
        X_donors=kept_donors,
        prior_edges=prior,
        sample_meta={"missing_panel_genes": ",".join(missing)},
    )


def smoothed_finite_difference(
    times: np.ndarray,
    X: np.ndarray,
    *,
    smooth_window: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate dx/dt via central/forward differences.

    If smooth_window >= 3 (odd preferred), apply moving-average on X first.
    Returns X_state (n, G), Y (n, G) aligned to interior or left points.
    """
    times = np.asarray(times, dtype=float)
    X = np.asarray(X, dtype=float)
    if smooth_window and smooth_window >= 3:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        kernel = np.ones(w) / w
        Xs = np.zeros_like(X)
        for g in range(X.shape[1]):
            padded = np.pad(X[:, g], (pad, pad), mode="edge")
            Xs[:, g] = np.convolve(padded, kernel, mode="valid")
        X = Xs

    if len(times) < 2:
        raise ValueError("Need >=2 time points")
    # forward difference on consecutive points
    dt = np.diff(times)
    dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    Y = np.diff(X, axis=0) / dt[:, None]
    return X[:-1].copy(), Y


def spline_derivative(
    times: np.ndarray,
    X: np.ndarray,
    *,
    k: int = 3,
    n_eval: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cubic (or lower) spline derivative; falls back to FD if scipy missing / too few pts."""
    times = np.asarray(times, dtype=float)
    X = np.asarray(X, dtype=float)
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        return smoothed_finite_difference(times, X)

    T, G = X.shape
    kk = min(k, max(T - 1, 1))
    if T < 4:
        return smoothed_finite_difference(times, X)

    if n_eval is None:
        t_eval = times
    else:
        t_eval = np.linspace(times.min(), times.max(), int(n_eval))

    X_out = np.zeros((len(t_eval), G))
    Y_out = np.zeros((len(t_eval), G))
    for g in range(G):
        # s slightly >0 for mild smoothing on noisy expression
        spl = UnivariateSpline(times, X[:, g], k=kk, s=0.5 * T)
        X_out[:, g] = spl(t_eval)
        Y_out[:, g] = spl.derivative(1)(t_eval)
    return X_out, Y_out


def estimate_derivatives(
    times: np.ndarray,
    X: np.ndarray,
    method: str = "smooth_fd",
) -> Tuple[np.ndarray, np.ndarray]:
    method = method.lower()
    if method in ("fd", "finite_difference", "smooth_fd"):
        win = 3 if method == "smooth_fd" and len(times) >= 5 else 0
        return smoothed_finite_difference(times, X, smooth_window=win)
    if method in ("spline", "gp_proxy"):
        # gp_proxy: spline as stand-in without extra deps
        return spline_derivative(times, X)
    raise ValueError(f"Unknown derivative method: {method}")


def select_human_regulators(
    panel: HumanPanelDataset,
    X: np.ndarray,
    y: np.ndarray,
    target: int,
    k: int,
    method: str = "prior",
) -> List[int]:
    """
    Restrict candidates using prior and/or data-driven ranking.

    prior: prior parents (up to k)
    prior_corr / prior_mi / prior_lasso: rank within prior pool (fallback to all genes)
    corr / mi / lasso: rank among all genes excluding target
    """
    method = method.lower()
    prior = panel.prior_parents(target)
    if method == "prior":
        return prior[:k]
    if method.startswith("prior_"):
        base = method.split("prior_", 1)[1]
        pool = prior if prior else [g for g in range(panel.n_genes) if g != target]
        X_pool = X[:, pool]
        if base == "corr":
            local = correlation_select(X_pool, y, -1, k=len(pool), exclude_target=False)
        elif base == "mi":
            local = mi_select(X_pool, y, -1, k=len(pool), exclude_target=False)
        elif base == "lasso":
            local = lasso_select(X_pool, y, -1, k=len(pool), exclude_target=False)
        else:
            raise ValueError(method)
        ranked = [pool[i] for i in local]
        return ranked[:k]
    if method == "corr":
        return correlation_select(X, y, target, k=k)
    if method == "mi":
        return mi_select(X, y, target, k=k)
    if method == "lasso":
        return lasso_select(X, y, target, k=k)
    raise ValueError(f"Unknown selection method: {method}")


def build_human_local_problems(
    panel: HumanPanelDataset,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    method: str = "prior",
    k: int = 2,
    max_vars: int = 3,
    include_target: bool = True,
    target_genes: Optional[Sequence[str]] = None,
    split: str = "all",
) -> Tuple[List[SampledDataset], Dict[int, List[int]], List[dict]]:
    """Build per-target local SR problems; skip targets with empty regulator set."""
    network = panel.as_grn_like()
    if target_genes is None:
        # prefer genes that appear as prior targets
        tnames = sorted({t for _, t in panel.prior_edges})
        targets = [panel.gene_index(n) for n in tnames if n in panel.gene_names]
    else:
        targets = [panel.gene_index(n) for n in target_genes if n in panel.gene_names]

    problems: List[SampledDataset] = []
    selections: Dict[int, List[int]] = {}
    rows: List[dict] = []
    for t in targets:
        regs = select_human_regulators(panel, X, Y[:, t], t, k=k, method=method)
        if not regs and method == "prior":
            continue
        if not regs:
            regs = correlation_select(X, Y[:, t], t, k=k)
        selections[t] = regs
        eq_id = f"human_{panel.source}_{panel.gene_names[t]}_{method}"
        ds = build_local_problem(
            network,
            X,
            Y[:, t],
            t,
            regs,
            eq_id=eq_id,
            split=split,
            include_target=include_target,
            max_vars=max_vars,
            selection_method=method,
        )
        # replace opaque motif with gene-level annotation (no true expr)
        ds.spec.target_expr = "unknown"
        ds.spec.motif = (
            f"target={panel.gene_names[t]};regs="
            + ",".join(panel.gene_names[r] for r in regs)
        )
        problems.append(ds)
        prior = panel.prior_parents(t)
        hit = len(set(regs) & set(prior))
        rows.append(
            {
                "target": panel.gene_names[t],
                "regs": [panel.gene_names[r] for r in regs],
                "prior": [panel.gene_names[r] for r in prior],
                "prior_hit": hit,
                "prior_recall": float(hit / max(len(prior), 1)),
                "n_points": int(len(ds.y)),
            }
        )
    return problems, selections, rows


def prior_edge_recovery(
    panel: HumanPanelDataset,
    selections: Dict[int, List[int]],
) -> Dict[str, float]:
    true_edges = [
        (panel.gene_index(r), panel.gene_index(t))
        for r, t in panel.prior_edges
        if r in panel.gene_names and t in panel.gene_names
    ]
    pred = []
    for t, regs in selections.items():
        for r in regs:
            pred.append((r, t))
    try:
        from evaluation.grn_metrics import edge_recovery
    except ImportError:
        from ..evaluation.grn_metrics import edge_recovery

    return edge_recovery(true_edges, pred)


def prepare_gse112372(
    out_dir: Path,
    *,
    panel: Sequence[str] = DEFAULT_PANEL,
    force_download: bool = False,
) -> HumanPanelDataset:
    """Download (if needed), cache panel JSON/NPZ under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tpm_path = raw_dir / "GSE112372_TPM_genes.txt.gz"
    meta_path = raw_dir / "GSE112372_metadata_macrophageLPS_samples.txt.gz"
    cache_npz = out_dir / "panel.npz"
    cache_meta = out_dir / "panel_meta.json"

    if force_download:
        for p in (tpm_path, meta_path, cache_npz, cache_meta):
            if p.exists():
                p.unlink()

    if cache_npz.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        z = np.load(cache_npz, allow_pickle=True)
        donors = {k: z[f"donor_{k}"] for k in meta["donors"]}
        return HumanPanelDataset(
            source=meta["source"],
            gene_names=list(meta["gene_names"]),
            times=z["times"],
            X_mean=z["X_mean"],
            X_donors=donors,
            prior_edges=[tuple(e) for e in meta["prior_edges"]],
            sample_meta=meta.get("sample_meta", {}),
        )

    download_file(GSE112372_TPM_URL, tpm_path)
    try:
        download_file(GSE112372_META_URL, meta_path)
    except Exception:
        pass

    genes, samples, mat = load_tpm_matrix(tpm_path)
    ds = build_panel_from_tpm(genes, samples, mat, panel=panel, prior_edges=DEFAULT_PRIOR_EDGES)

    payload = {"times": ds.times, "X_mean": ds.X_mean}
    for d, arr in ds.X_donors.items():
        payload[f"donor_{d}"] = arr
    np.savez_compressed(cache_npz, **payload)
    cache_meta.write_text(
        json.dumps(
            {
                "source": ds.source,
                "gene_names": ds.gene_names,
                "donors": list(ds.X_donors.keys()),
                "prior_edges": [list(e) for e in ds.prior_edges],
                "sample_meta": ds.sample_meta,
                "n_genes_full_tpm": len(genes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ds


def save_panel_summary(ds: HumanPanelDataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source": ds.source,
                "gene_names": ds.gene_names,
                "times_h": ds.times.tolist(),
                "n_donors": len(ds.X_donors),
                "prior_edges": [list(e) for e in ds.prior_edges],
                "X_mean_shape": list(ds.X_mean.shape),
                "sample_meta": ds.sample_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

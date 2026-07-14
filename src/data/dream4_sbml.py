"""Parse DREAM4 GNW SBML and reconstruct teacher ODEs for supervised FT.

DREAM4 SBML kineticLaws store parameters but not MathML. We reconstruct
GeneNetWeaver-style activation from reaction names + Hill parameters:

  dx_i/dt = max_i * f_i(regulators) - delta_i * x_i

where f_i is assembled from modules in the synthesis reaction name
(e.g. ``~(1) + (2*3)``) using occupancy hills with coefficients a_*.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

NS = {"s": "http://www.sbml.org/sbml/level2"}


@dataclass
class ModuleSpec:
    repressing: bool
    regulator_indices: List[int]  # 1-based indices into modifiers list
    binds_as_complex: bool = True


@dataclass
class GeneODE:
    gene: str
    gene_idx: int  # 0-based
    modifiers: List[str]
    modules: List[ModuleSpec]
    params: Dict[str, float]
    max_rate: float
    delta: float
    constitutive: bool = False

    def parent_indices(self, gene_to_idx: Dict[str, int]) -> List[int]:
        return [gene_to_idx[m] for m in self.modifiers if m in gene_to_idx]


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_modules(reaction_name: str) -> List[ModuleSpec]:
    """Parse ``G2_synthesis: ~(1) + ~(2*3)`` module list."""
    if ":" not in reaction_name:
        return []
    payload = reaction_name.split(":", 1)[1].strip()
    if payload.lower().startswith("no inputs"):
        return []
    modules: List[ModuleSpec] = []
    # split on + at top level
    parts = re.split(r"\s*\+\s*", payload)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        repressing = part.startswith("~")
        if repressing:
            part = part[1:].strip()
        m = re.fullmatch(r"\(([^)]*)\)", part)
        inner = m.group(1) if m else part
        idxs = [int(x) for x in inner.split("*") if x.strip().isdigit()]
        modules.append(
            ModuleSpec(repressing=repressing, regulator_indices=idxs, binds_as_complex=True)
        )
    return modules


def occupancy(x: np.ndarray, k: float, n: float) -> np.ndarray:
    xn = np.power(np.maximum(x, 0.0), n)
    kn = max(k, 1e-12) ** n
    return xn / (kn + xn + 1e-12)


def parse_sbml_gene_odes(xml_path: Path) -> Dict[str, GeneODE]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    species = []
    for sp in root.findall(".//s:listOfSpecies/s:species", NS):
        sid = sp.attrib["id"]
        if sid != "_void_":
            species.append(sid)
    gene_to_idx = {g: i for i, g in enumerate(species)}

    synth: Dict[str, ET.Element] = {}
    deg: Dict[str, ET.Element] = {}
    for rx in root.findall(".//s:listOfReactions/s:reaction", NS):
        rid = rx.attrib["id"]
        if rid.endswith("_synthesis"):
            gene = rid[: -len("_synthesis")]
            synth[gene] = rx
        elif rid.endswith("_degradation"):
            gene = rid[: -len("_degradation")]
            deg[gene] = rx

    out: Dict[str, GeneODE] = {}
    for gene in species:
        rx = synth[gene]
        name = rx.attrib.get("name", "")
        mods = [
            m.attrib["species"]
            for m in rx.findall(".//s:listOfModifiers/s:modifierSpeciesReference", NS)
        ]
        params = {
            p.attrib["id"]: float(p.attrib["value"])
            for p in rx.findall(".//s:kineticLaw//s:parameter", NS)
        }
        modules = parse_modules(name)
        # fill bindsAsComplex from params when present
        for mi, mod in enumerate(modules, start=1):
            key = f"bindsAsComplex_{mi}"
            if key in params:
                mod.binds_as_complex = bool(params[key] >= 0.5)

        delta = 0.1
        if gene in deg:
            dparams = {
                p.attrib["id"]: float(p.attrib["value"])
                for p in deg[gene].findall(".//s:kineticLaw//s:parameter", NS)
            }
            delta = float(dparams.get("delta", delta))
        max_rate = float(params.get("max", 0.0))
        out[gene] = GeneODE(
            gene=gene,
            gene_idx=gene_to_idx[gene],
            modifiers=mods,
            modules=modules,
            params=params,
            max_rate=max_rate,
            delta=delta,
            constitutive=len(modules) == 0,
        )
    return out


def _module_activity(
    ode: GeneODE,
    module_idx: int,
    mod: ModuleSpec,
    X_full: np.ndarray,
    gene_to_idx: Dict[str, int],
) -> np.ndarray:
    """Compute fractional occupancy contribution in [approx 0,1] for one module."""
    p = ode.params
    regs = []
    for j in mod.regulator_indices:
        # j is 1-based into modifiers
        if j < 1 or j > len(ode.modifiers):
            continue
        gname = ode.modifiers[j - 1]
        regs.append(X_full[:, gene_to_idx[gname]])
    if not regs:
        return np.ones(X_full.shape[0], dtype=float)

    # per-regulator occupancy using k_j, n_j (index = regulator position in module list order)
    # GNW assigns k_i sequentially over all regulators of the gene
    # Map: global TF order in modifiers → use k_{modifier_pos}
    occs = []
    for j in mod.regulator_indices:
        gname = ode.modifiers[j - 1]
        x = X_full[:, gene_to_idx[gname]]
        k = float(p.get(f"k_{j}", 0.5))
        n = float(p.get(f"n_{j}", 2.0))
        occs.append(occupancy(x, k, n))

    if mod.binds_as_complex and len(occs) > 1:
        b = occs[0]
        for o in occs[1:]:
            b = b * o
    else:
        # independent: average occupancy
        b = sum(occs) / len(occs)

    # config coefficients: for single-module genes use a0,a1
    # multi-module: approximate with module-wise a pairs when available
    if len(ode.modules) == 1:
        a0 = float(p.get("a_0", 0.0))
        a1 = float(p.get("a_1", 1.0))
    else:
        # fall back: repressing modules prefer high a0; activating prefer high a1
        a0 = float(p.get("a_0", 0.0 if not mod.repressing else 1.0))
        a1 = float(p.get(f"a_{module_idx}", 1.0 if not mod.repressing else 0.0))
        if mod.repressing and a1 > a0:
            a0, a1 = max(a0, 1.0), min(a1, 0.2)
        if (not mod.repressing) and a0 > a1:
            a0, a1 = min(a0, 0.2), max(a1, 1.0)

    return a0 * (1.0 - b) + a1 * b


def relative_activation(
    ode: GeneODE, X_full: np.ndarray, gene_to_idx: Dict[str, int]
) -> np.ndarray:
    if ode.constitutive or not ode.modules:
        return np.full(X_full.shape[0], float(ode.params.get("a_0", 1.0)), dtype=float)
    acts = [
        _module_activity(ode, i, mod, X_full, gene_to_idx)
        for i, mod in enumerate(ode.modules, start=1)
    ]
    # additive modules (OR of regulatory programs), clipped
    f = acts[0]
    for a in acts[1:]:
        f = f + a
    # soft normalize into ~[0,1]
    f = f / max(len(acts), 1)
    return np.clip(f, 0.0, 1.5)


def rhs_from_ode(
    ode: GeneODE, X_full: np.ndarray, gene_to_idx: Dict[str, int]
) -> np.ndarray:
    f = relative_activation(ode, X_full, gene_to_idx)
    x_i = X_full[:, ode.gene_idx]
    return ode.max_rate * f - ode.delta * x_i


def expression_string(
    ode: GeneODE,
    *,
    local_map: Dict[str, str],
    numeric: bool = True,
) -> str:
    """
    Build a sympy-friendly teacher string in local variables x_*.
    Uses the same Hill skeleton as the numeric evaluator (single-module exact;
    multi-module as sum of module terms).
    """
    target = local_map[ode.gene]
    if ode.constitutive or not ode.modules:
        a0 = ode.params.get("a_0", 1.0)
        if numeric:
            return f"({ode.max_rate * a0:g})-({ode.delta:g})*{target}"
        return f"c-({ode.delta:g})*{target}"

    terms = []
    for mi, mod in enumerate(ode.modules, start=1):
        reg_locals = []
        for j in mod.regulator_indices:
            gname = ode.modifiers[j - 1]
            if gname not in local_map:
                continue
            reg_locals.append((j, local_map[gname]))
        if not reg_locals:
            continue
        # occupancy product or average as string
        occ_bits = []
        for j, loc in reg_locals:
            k = ode.params.get(f"k_{j}", 0.5)
            n = ode.params.get(f"n_{j}", 2.0)
            if numeric:
                occ_bits.append(
                    f"(({loc})**({n:g}))/(({k:g})**({n:g})+({loc})**({n:g}))"
                )
            else:
                occ_bits.append(f"(({loc})**n)/(K**n+({loc})**n)")
        if mod.binds_as_complex and len(occ_bits) > 1:
            b = "*".join(f"({o})" for o in occ_bits)
        else:
            b = "(" + "+".join(occ_bits) + f")/{len(occ_bits)}"
        if len(ode.modules) == 1:
            a0 = ode.params.get("a_0", 0.0)
            a1 = ode.params.get("a_1", 1.0)
        else:
            a0 = ode.params.get("a_0", 1.0 if mod.repressing else 0.0)
            a1 = ode.params.get(f"a_{mi}", 0.0 if mod.repressing else 1.0)
        if numeric:
            terms.append(f"(({a0:g})*(1-({b}))+({a1:g})*({b}))")
        else:
            terms.append(f"(a0*(1-({b}))+a1*({b}))")
    if not terms:
        body = "0"
    elif len(terms) == 1:
        body = terms[0]
    else:
        body = "(" + "+".join(terms) + f")/{len(terms)}"
    if numeric:
        return f"({ode.max_rate:g})*({body})-({ode.delta:g})*{target}"
    return f"m*({body})-d*{target}"


def sbml_path_for(root: Path, size: int, net_id: int) -> Path:
    return (
        root
        / f"Size {size}"
        / "Supplementary information"
        / f"insilico_size{size}_{net_id}"
        / "Goldstandard"
        / f"insilico_size{size}_{net_id}.xml"
    )


def rhs_mrna(
    ode: GeneODE,
    X_mrna: np.ndarray,
    X_tf: np.ndarray,
    gene_to_idx: Dict[str, int],
) -> np.ndarray:
    """dx/dt with TF inputs from X_tf and self-degradation on mRNA."""
    f = relative_activation(ode, X_tf, gene_to_idx)
    return ode.max_rate * f - ode.delta * X_mrna[:, ode.gene_idx]


def sample_supervised_points(
    odes: Dict[str, GeneODE],
    gene_names: Sequence[str],
    *,
    n_points: int = 200,
    support: Tuple[float, float] = (0.05, 1.0),
    seed: int = 0,
    use_protein_proxy: bool = True,
    label_noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw random expression states and evaluate reconstructed RHS.

    When ``use_protein_proxy``, TF inputs equal mRNA (quasi-steady protein≈mRNA).
    ``label_noise_std`` adds Gaussian noise relative to std(y) (overfit guard).
    """
    rng = np.random.default_rng(seed)
    lo, hi = support
    n_genes = len(gene_names)
    X = rng.uniform(lo, hi, size=(n_points, n_genes))
    X_tf = X.copy() if use_protein_proxy else X.copy()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    Y = np.zeros_like(X)
    for g in gene_names:
        Y[:, gene_to_idx[g]] = rhs_mrna(odes[g], X, X_tf, gene_to_idx)
    if label_noise_std > 0:
        for j in range(Y.shape[1]):
            scale = label_noise_std * (float(np.std(Y[:, j])) + 1e-8)
            Y[:, j] = Y[:, j] + rng.normal(0.0, scale, size=Y.shape[0])
    return X, Y


def load_nonoise_aligned_fd(
    root: Path, size: int, net_id: int
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return gene_names, X_mrna_state, X_protein_state, Y_mrna_fd from
    supplementary noiseless ODE timeseries (aligned by trajectory cut).
    """
    from .dream4 import finite_difference_rhs, load_timeseries

    base = (
        root
        / f"Size {size}"
        / "Supplementary information"
        / f"insilico_size{size}_{net_id}"
        / "ODEs without experimental noise"
    )
    mrna_path = base / f"insilico_size{size}_{net_id}_nonoise_timeseries.tsv"
    prot_path = base / f"insilico_size{size}_{net_id}_nonoise_proteins_timeseries.tsv"
    genes, t_m, x_m = load_timeseries(mrna_path)
    _, t_p, x_p = load_timeseries(prot_path)
    Xm, Ym = finite_difference_rhs(t_m, x_m)
    Xp_rows = []
    for xp in x_p:
        Xp_rows.append(xp[:-1])
    Xp = np.vstack(Xp_rows)
    if Xp.shape != Xm.shape:
        n = min(Xp.shape[0], Xm.shape[0])
        Xm, Ym, Xp = Xm[:n], Ym[:n], Xp[:n]
    return genes, Xm, Xp, Ym


def mix_supervised_and_trajectory(
    odes: Dict[str, GeneODE],
    gene_names: Sequence[str],
    Xm: np.ndarray,
    Xp: np.ndarray,
    *,
    n_random: int = 200,
    seed: int = 0,
    label_noise_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine random SBML-RHS samples with trajectory mRNA states using protein TFs.
    Y always from reconstructed ODE (teacher consistency for tokenization).
    """
    rng = np.random.default_rng(seed)
    X_rand, Y_rand = sample_supervised_points(
        odes,
        gene_names,
        n_points=n_random,
        seed=seed,
        label_noise_std=label_noise_std,
    )
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    Y_traj = np.zeros_like(Xm)
    for g in gene_names:
        Y_traj[:, gene_to_idx[g]] = rhs_mrna(odes[g], Xm, Xp, gene_to_idx)
    if label_noise_std > 0:
        for j in range(Y_traj.shape[1]):
            scale = label_noise_std * (float(np.std(Y_traj[:, j])) + 1e-8)
            Y_traj[:, j] = Y_traj[:, j] + rng.normal(0.0, scale, size=Y_traj.shape[0])
    # Features for SR/FT use mRNA columns (observed); TF dynamics baked into Y via Xp
    X = np.vstack([X_rand, Xm])
    Y = np.vstack([Y_rand, Y_traj])
    return X, Y


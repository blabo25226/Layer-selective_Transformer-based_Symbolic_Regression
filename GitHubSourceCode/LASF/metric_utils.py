"""
Complete Metric Computation Utilities (GPU Optimized & Vectorized)
"""
import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Registry & Helper
# ============================================================================
ALL_METRICS = [
    # Information-Theoretic
    'prompt_entropy', 'dataset_entropy', 'infonce', 'lidar', 'dime',
    # Geometric
    'curvature', 'intrinsic_dimension',
    # Magnitude & Spectral
    'effective_rank', 'l2_norm', 'spectral_metrics', 'sparsity', 'gini_coefficient',
    # Alignment (Requires Base + SFT)
    'cka', 'cosine_similarity', 'mean_shift', 'change_intensity'
]

# Metrics that require augmented views (N, K, D)
CONTRASTIVE_METRICS = ['infonce', 'lidar', 'dime']
# Metrics that require pairs (Base + SFT)
ALIGNMENT_METRICS = ['cka', 'cosine_similarity', 'mean_shift', 'change_intensity']
# Metrics that require sequence dimension (N, T, D)
SEQUENCE_METRICS = ['prompt_entropy', 'curvature', 'sparsity']

# [prompt entropy, dataset entropy, effective rank, curvature, cka, cosine similarity, mean shift, sparsity, L2 norm, condition number, rank deficiency, spectral norm]

# ============================================================================
# Core Matrix Entropy
# ============================================================================

def matrix_alpha_entropy(eigenvalues: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Compute matrix-based alpha-entropy from eigenvalues."""
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)
    trace = torch.sum(eigenvalues, dim=-1, keepdim=True)
    p = eigenvalues / (trace + 1e-10)

    if abs(alpha - 1.0) < 1e-6:
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    else:
        entropy = (1.0 / (1.0 - alpha)) * torch.log(torch.sum(p ** alpha, dim=-1) + 1e-10)

    mask = (trace.squeeze(-1) < 1e-10)
    entropy = torch.where(mask, torch.tensor(0.0, device=entropy.device), entropy)
    return entropy


def normalize_entropy(
    entropy: Union[torch.Tensor, float],
    normalization: str,
    N: int,
    D: int
) -> float:
    """Normalize entropy by various schemes."""
    # Convert tensor to float if needed
    if isinstance(entropy, torch.Tensor):
        entropy = entropy.item()

    if normalization == 'maxEntropy':
        denom = math.log(min(N, D))
    elif normalization == 'logN':
        denom = math.log(N)
    elif normalization == 'logD':
        denom = math.log(D)
    elif normalization == 'logNlogD':
        denom = math.log(N) * math.log(D)
    elif normalization == 'raw':
        return entropy
    else:
        # Default fallback
        denom = math.log(min(N, D))

    if denom <= 1e-9:
        return 0.0
    return entropy / denom


# ============================================================================
# Information-Theoretic Metrics
# ============================================================================

def compute_prompt_entropy(
    hidden_states: Dict[int, torch.Tensor],
    alpha: float = 1.0,
    normalizations: List[str] = ['maxEntropy']
) -> Dict[str, List[float]]:
    """Prompt entropy (requires hidden_states N, T, D)."""
    results = {norm: [] for norm in normalizations}

    for layer_idx in sorted(hidden_states.keys()):
        hidden = hidden_states[layer_idx]
        if hidden.dim() != 3: # Check dimensions
            for norm in normalizations: results[norm].append(0.0)
            continue

        N, T, D = hidden.shape

        # ===== FIX: Convert to float32 BEFORE computing gram matrix =====
        hidden = hidden.float()

        # ===== FIX: Normalize to prevent numerical overflow =====
        # Scale hidden states to unit norm per sequence
        hidden = hidden / (torch.norm(hidden, dim=2, keepdim=True) + 1e-10)

        # Gram matrix (N, T, T) - now properly scaled
        gram = torch.bmm(hidden, hidden.transpose(1, 2)) / D

        # ===== FIX: Add regularization for numerical stability =====
        reg_strength = 1e-6
        identity = torch.eye(T, device=gram.device, dtype=gram.dtype).unsqueeze(0).expand(N, -1, -1)
        gram = gram + reg_strength * identity

        try:
            # Use SVD instead of eigenvalue decomposition (more stable)
            # For symmetric matrices: eigenvalues ≈ singular values
            U, S, Vh = torch.linalg.svd(gram)
            eigenvalues = S  # Shape: (N, T)
            entropies = matrix_alpha_entropy(eigenvalues, alpha)
            avg_entropy = torch.mean(entropies).item()
        except Exception as e:
            logger.warning(f"Layer {layer_idx} entropy computation failed: {e}, trying eigenvalue fallback")
            try:
                eigenvalues = torch.linalg.eigvalsh(gram)
                entropies = matrix_alpha_entropy(eigenvalues, alpha)
                avg_entropy = torch.mean(entropies).item()
            except Exception as e2:
                logger.error(f"Layer {layer_idx} both SVD and eigh failed: {e2}")
                avg_entropy = 0.0

        for norm in normalizations:
            results[norm].append(normalize_entropy(avg_entropy, norm, T, D))

    return results


def compute_dataset_entropy(
    pooled_states: Dict[int, torch.Tensor],
    alpha: float = 1.0,
    normalizations: List[str] = ['maxEntropy']
) -> Dict[str, List[float]]:
    """Dataset entropy (requires pooled_states N, D)."""
    results = {norm: [] for norm in normalizations}

    for layer_idx in sorted(pooled_states.keys()):
        hidden = pooled_states[layer_idx]
        N, D = hidden.shape

        # ===== FIX: Convert to float32 first =====
        hidden = hidden.float()

        # ===== FIX: Normalize data for numerical stability =====
        hidden = hidden / (torch.norm(hidden, dim=1, keepdim=True) + 1e-10)

        try:
            if N > D:
                # Use SVD for tall matrices (more stable)
                s_vals = torch.linalg.svdvals(hidden)
                eigenvalues = s_vals ** 2
            else:
                # For wide matrices, compute gram matrix
                gram = hidden @ hidden.T / D

                # ===== FIX: Add regularization =====
                reg_strength = 1e-6
                gram = gram + reg_strength * torch.eye(N, device=gram.device, dtype=gram.dtype)

                # Try SVD first (more stable)
                try:
                    U, S, Vh = torch.linalg.svd(gram)
                    eigenvalues = S
                except Exception:
                    # Fallback to eigenvalue decomposition
                    eigenvalues = torch.linalg.eigvalsh(gram)

            entropy = matrix_alpha_entropy(eigenvalues, alpha).item()
        except Exception as e:
            logger.warning(f"Layer {layer_idx} dataset entropy failed: {e}")
            entropy = 0.0

        for norm in normalizations:
            results[norm].append(normalize_entropy(entropy, norm, N, D))

    return results


def compute_infonce(
    augmented_states: Dict[int, torch.Tensor],
    temperature: float = 0.1
) -> Dict[str, List[float]]:
    """InfoNCE (requires augmented_states N, K, D where K>=2)."""
    results = {'raw': [], 'mi_lower_bound': []}

    for layer_idx in sorted(augmented_states.keys()):
        hidden = augmented_states[layer_idx]
        if hidden.dim() < 3 or hidden.shape[1] < 2:
            results['raw'].append(0.0)
            results['mi_lower_bound'].append(0.0)
            continue

        N = hidden.shape[0]
        view_a = torch.nn.functional.normalize(hidden[:, 0, :], dim=1)
        view_b = torch.nn.functional.normalize(hidden[:, 1, :], dim=1)

        logits = view_a @ view_b.T / temperature
        labels = torch.arange(N, device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels).item()

        results['raw'].append(loss)
        results['mi_lower_bound'].append(1.0 - loss / (math.log(N) + 1e-9))
    return results


def compute_lidar(
    augmented_states: Dict[int, torch.Tensor],
    alpha: float = 1.0,
    normalizations: List[str] = ['maxEntropy'],
    delta: float = 1e-4
) -> Dict[str, List[float]]:
    """LiDAR (requires augmented_states N, K, D)."""
    results = {norm: [] for norm in normalizations}

    for layer_idx in sorted(augmented_states.keys()):
        hidden = augmented_states[layer_idx]
        if hidden.dim() < 3:
            for norm in normalizations: results[norm].append(0.0)
            continue

        N, K, D = hidden.shape
        try:
            global_mean = torch.mean(hidden, dim=(0, 1))
            class_means = torch.mean(hidden, dim=1)
            diff = class_means - global_mean
            S_b = (diff.T @ diff) / N

            # Simplified S_w calculation for memory
            S_w = torch.zeros((D, D), device=hidden.device)
            # Batch calculation if needed, here simplified
            centered = hidden - class_means.unsqueeze(1)
            centered = centered.reshape(-1, D)
            S_w = (centered.T @ centered) / (N * K) + delta * torch.eye(D, device=hidden.device)

            eig_vals, eig_vecs = torch.linalg.eigh(S_w)
            inv_sqrt = eig_vecs @ torch.diag(1.0 / torch.sqrt(torch.clamp(eig_vals, min=1e-9))) @ eig_vecs.T
            lda_matrix = inv_sqrt @ S_b @ inv_sqrt

            lda_eigs = torch.linalg.eigvalsh(lda_matrix)
            lidar = matrix_alpha_entropy(lda_eigs, alpha).item()
        except Exception:
            lidar = 0.0

        for norm in normalizations:
            results[norm].append(normalize_entropy(lidar, norm, N, D))
    return results


def compute_dime(
    augmented_states: Dict[int, torch.Tensor],
    alpha: float = 1.0,
    normalizations: List[str] = ['maxEntropy']
) -> Dict[str, List[float]]:
    """DiME (requires augmented_states N, K>=2, D)."""
    results = {norm: [] for norm in normalizations}

    for layer_idx in sorted(augmented_states.keys()):
        hidden = augmented_states[layer_idx]
        if hidden.dim() < 3 or hidden.shape[1] < 2:
            for norm in normalizations: results[norm].append(0.0)
            continue

        N, _, D = hidden.shape
        view_a = hidden[:, 0, :]
        view_b = hidden[:, 1, :]

        try:
            # Use SVD for stability if N > D
            if N > D:
                cov_a = view_a.T @ view_a
                cov_b = view_b.T @ view_b
            else:
                cov_a = view_a @ view_a.T
                cov_b = view_b @ view_b.T

            eig_a = torch.linalg.eigvalsh(cov_a.float())
            eig_b = torch.linalg.eigvalsh(cov_b.float())

            ent_a = matrix_alpha_entropy(eig_a, alpha)
            ent_b = matrix_alpha_entropy(eig_b, alpha)
            dime = torch.abs(ent_a - ent_b).item()
        except Exception:
            dime = 0.0

        for norm in normalizations:
            results[norm].append(normalize_entropy(dime, norm, N, D))
    return results


# ============================================================================
# Geometric & Magnitude
# ============================================================================

def compute_intrinsic_dimension(pooled_states: Dict[int, torch.Tensor], k: int = 2) -> Dict[str, List[float]]:
    # results = {'raw': [], 'normalized': []}
    results = {'normalized': []}
    for layer_idx in sorted(pooled_states.keys()):
        hidden = pooled_states[layer_idx]
        N, D = hidden.shape
        # Simplified placeholder if N too small
        if N <= k + 1:
            # results['raw'].append(float(D))
            results['normalized'].append(1.0)
            continue

        # Sampling for speed
        sample_size = min(N, 1000)
        idx = torch.randperm(N)[:sample_size]
        subset = hidden[idx]

        try:
            dists = torch.cdist(subset.float(), subset.float())
            topk = torch.topk(dists, k=k+1, dim=1, largest=False).values
            r_k = topk[:, -1]
            r_k1 = topk[:, -2]

            # ID estimator
            mu = r_k / (r_k1 + 1e-9)
            id_val = 1.0 / torch.mean(torch.log(mu + 1e-9)).item()
            id_val = min(max(id_val, 0.0), D)
        except Exception:
            id_val = 0.0

        # results['raw'].append(id_val)
        results['normalized'].append(id_val / (math.log(N) if N > 1 else 1))
    return results

def compute_curvature(hidden_states: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    # results = {'raw': [], 'normalized': []}
    results = {'normalized': []}
    for layer_idx in sorted(hidden_states.keys()):
        hidden = hidden_states[layer_idx] # N, T, D
        if hidden.dim() != 3 or hidden.shape[1] < 3:
            # results['raw'].append(0.0);
            results['normalized'].append(0.0)
            continue

        delta = hidden[:, 1:, :] - hidden[:, :-1, :]
        v1 = delta[:, :-1, :]
        v2 = delta[:, 1:, :]
        sim = torch.nn.functional.cosine_similarity(v1, v2, dim=2)
        angle = torch.acos(torch.clamp(sim, -0.999, 0.999))
        curv = torch.mean(angle).item()

        # results['raw'].append(curv)
        results['normalized'].append(curv / math.pi)
    return results

def compute_effective_rank(pooled_states: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    # results = {'raw': [], 'normalized': []}
    results = {'normalized': []}
    for layer_idx in sorted(pooled_states.keys()):
        hidden = pooled_states[layer_idx]
        N, D = hidden.shape
        try:
            s_vals = torch.linalg.svdvals(hidden.float())
            p = (s_vals ** 2) / torch.sum(s_vals ** 2 + 1e-9)
            ent = -torch.sum(p * torch.log(p + 1e-9)).item()
            erank = math.exp(ent)
        except Exception:
            erank = 1.0

        # results['raw'].append(erank)
        results['normalized'].append(erank / min(N, D))
    return results

def compute_l2_norm(pooled_states: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    results = {'mean': [], 'std': []} 
    for layer_idx in sorted(pooled_states.keys()):
        norms = torch.norm(pooled_states[layer_idx].float(), dim=1)
        results['mean'].append(torch.mean(norms).item())
        results['std'].append(torch.std(norms).item()) 
    return results


def compute_spectral_metrics(pooled_states: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
    results = {'spectral_norm': [], 'condition_number': [], 'rank_deficiency': []}

    for layer_idx in sorted(pooled_states.keys()):
        hidden = pooled_states[layer_idx]
        N, D = hidden.shape
        try:
            s_vals = torch.linalg.svdvals(hidden.float())
            s_max = s_vals[0].item()
            s_min = s_vals[-1].item()
            cond = s_max / (s_min + 1e-9)

            thresh = 0.01 * s_max
            effective_rank = torch.sum(s_vals > thresh).item()
            rank_def = 1.0 - (effective_rank / min(N, D))

        except Exception:
            s_max, cond, rank_def = 0.0, 0.0, 0.0

        results['spectral_norm'].append(s_max)
        results['condition_number'].append(cond)
        results['rank_deficiency'].append(rank_def)  

    return results


def compute_sparsity(hidden_states: Dict[int, torch.Tensor], threshold: float = 0.01) -> List[float]:
    res = []
    for layer_idx in sorted(hidden_states.keys()):
        hidden = hidden_states[layer_idx]
        thresh_val = threshold * torch.max(torch.abs(hidden))
        sparsity = torch.mean((torch.abs(hidden) < thresh_val).float()).item()
        res.append(sparsity)
    return res

def compute_gini_coefficient(pooled_states: Dict[int, torch.Tensor]) -> List[float]:
    res = []
    for layer_idx in sorted(pooled_states.keys()):
        norms = torch.norm(pooled_states[layer_idx].float(), dim=1)
        sorted_norms, _ = torch.sort(norms)
        N = len(sorted_norms)
        idx = torch.arange(1, N + 1, device=norms.device).float()
        gini = (2 * torch.sum(idx * sorted_norms)) / (N * torch.sum(sorted_norms) + 1e-9) - (N + 1)/N
        res.append(gini.item())
    return res


# ============================================================================
# Alignment (Base vs SFT)
# ============================================================================

def compute_cka(base_states: Dict[int, torch.Tensor], sft_states: Dict[int, torch.Tensor]) -> List[float]:
    res = []
    keys = sorted(list(set(base_states.keys()) & set(sft_states.keys())))
    for k in keys:
        X = base_states[k].float(); Y = sft_states[k].float()
        if X.dim() > 2: X = X.flatten(1)
        if Y.dim() > 2: Y = Y.flatten(1)

        # Linear CKA
        gram_x = X @ X.T
        gram_y = Y @ Y.T

        # Centering
        mean_x = torch.mean(gram_x, dim=0, keepdim=True)
        gram_x = gram_x - mean_x - mean_x.T + torch.mean(mean_x)
        mean_y = torch.mean(gram_y, dim=0, keepdim=True)
        gram_y = gram_y - mean_y - mean_y.T + torch.mean(mean_y)

        num = torch.sum(gram_x * gram_y)
        denom = torch.sqrt(torch.sum(gram_x**2) * torch.sum(gram_y**2))
        res.append((num / (denom + 1e-9)).item())
    return res

def compute_cosine_similarity(base_states: Dict, sft_states: Dict) -> List[float]:
    res = []
    keys = sorted(list(set(base_states.keys()) & set(sft_states.keys())))
    for k in keys:
        X = base_states[k].float(); Y = sft_states[k].float()
        if X.dim() > 2: X = X.flatten(1)
        if Y.dim() > 2: Y = Y.flatten(1)
        sim = torch.nn.functional.cosine_similarity(X, Y, dim=1)
        res.append(torch.mean(sim).item())
    return res

def compute_mean_shift(base_states: Dict, sft_states: Dict) -> List[float]:
    res = []
    keys = sorted(list(set(base_states.keys()) & set(sft_states.keys())))
    for k in keys:
        if base_states[k].dim() > 2: continue # Requires pooled
        diff = torch.mean(sft_states[k].float(), dim=0) - torch.mean(base_states[k].float(), dim=0)
        res.append(torch.norm(diff).item())
    return res

def compute_change_intensity(base_states: Dict, sft_states: Dict) -> List[float]:
    res = []
    keys = sorted(list(set(base_states.keys()) & set(sft_states.keys())))
    for k in keys:
        X = base_states[k].float(); Y = sft_states[k].float()
        if X.dim() > 2: X = X.flatten(1)
        if Y.dim() > 2: Y = Y.flatten(1)
        dists = torch.norm(Y - X, dim=1)
        res.append(torch.mean(dists).item())
    return res

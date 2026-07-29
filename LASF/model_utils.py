"""
Model loading and representation extraction utilities
"""
import os
import torch
import logging
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from config import MODEL_CONFIGS, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


def load_model(
        model_family: str,
        scale: str,
        variant: str,
        cache_dir: str = MODEL_CACHE_DIR,
        device_map: str = 'auto',
        dtype: str = 'float32',
        download_only: bool = False
) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load model and tokenizer from HuggingFace

    Args:
        model_family: Model family (e.g., 'gemma3')
        scale: Model scale (e.g., '1b', '27b')
        variant: 'base' or 'instruct'
        cache_dir: Cache directory
        device_map: Device map for model loading
        dtype: Data type ('float32', 'float16', 'bfloat16')
        download_only: Only download, don't load

    Returns:
        (model, tokenizer) or (None, None) if download_only
    """
    # Validate inputs
    if model_family not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_family: {model_family}")
    if scale not in MODEL_CONFIGS[model_family]:
        raise ValueError(f"Unknown scale for {model_family}: {scale}")
    if variant not in MODEL_CONFIGS[model_family][scale]:
        raise ValueError(f"Unknown variant: {variant}")

    model_path = MODEL_CONFIGS[model_family][scale][variant]
    logger.info(f"Target: {model_family}-{scale}-{variant} | Repo: {model_path}")

    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)

    # -----------------------------------------------------------
    # Download only mode - use snapshot_download to avoid OOM
    # -----------------------------------------------------------
    if download_only:
        logger.info(f" ✅ Starting download for {model_path} (Files only, no RAM loading)...")
        try:
            snapshot_download(
                repo_id=model_path,
                cache_dir=cache_dir,
                max_workers=1, 
                # If HF_TOKEN is set in environment, it will be used automatically
            )
            logger.info(f" ✅  Successfully downloaded {model_family}-{scale}-{variant} to {cache_dir}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise e

        return None, None

    # -----------------------------------------------------------
    # Normal loading mode - load from cache or download if needed
    # -----------------------------------------------------------

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,  # Use repo_id, not cache_dir path
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model weights from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,  # ✓ Fixed: Use repo_id, not cache_dir path
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        output_hidden_states=True
    )

    model.eval()

    logger.info(f" ✅  Loaded {model_family}-{scale}-{variant}")
    logger.info(f" ✅  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else device_map}")
    logger.info(f" ✅  Dtype: {torch_dtype}")
    logger.info(f" ✅ Cache dir: {cache_dir}")

    return model, tokenizer


def get_num_layers(model_family: str, scale: str) -> int:
    """Get number of layers for a model"""
    return MODEL_CONFIGS[model_family][scale]['num_layers']


def parse_layer_indices(strategy: str, num_layers: int) -> List[int]:
    """
    Parse layer index strategy

    Args:
        strategy: 'all', 'key', or 'sparse'
        num_layers: Total number of layers

    Returns:
        List of layer indices
    """
    if strategy == 'all':
        return list(range(num_layers))

    elif strategy == 'key':
        # Sample every 8 layers + first and last
        step = max(1, num_layers // 8)
        indices = list(range(0, num_layers, step))
        if (num_layers - 1) not in indices:
            indices.append(num_layers - 1)
        return indices

    elif strategy == 'sparse':
        # Quartiles
        indices = [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1
        ]
        # Remove duplicates and sort
        return sorted(list(set(indices)))

    else:
        raise ValueError(f"Unknown layer_indices strategy: {strategy}")


@torch.no_grad()
def extract_representations(
        model: AutoModelForCausalLM,
        batch: Dict,
        layer_indices: List[int],
        pooling_method: Optional[str] = 'mean',
        dtype: str = 'float32'
) -> Dict:
    """
    Extract representations from model

    Args:
        model: HuggingFace model
        batch: Input batch with 'input_ids' and 'attention_mask'
        layer_indices: Which layers to extract
        pooling_method: 'mean', 'last', or None (no pooling)
        dtype: Output dtype

    Returns:
        {
            'hidden_states': {layer_idx: tensor(B, T, D)} or None,
            'pooled_states': {layer_idx: tensor(B, D)} or None
        }
    """
    # Move batch to model device
    device = next(model.parameters()).device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    # Extract hidden states
    all_hidden_states = outputs.hidden_states  # Tuple of (B, T, D)

    result = {
        'hidden_states': None,
        'pooled_states': None
    }

    # Store hidden states (if no pooling)
    if pooling_method is None:
        result['hidden_states'] = {}
        for layer_idx in layer_indices:
            if layer_idx < len(all_hidden_states):
                hidden = all_hidden_states[layer_idx]
                # Convert dtype
                if dtype == 'float16':
                    hidden = hidden.half()
                elif dtype == 'float32':
                    hidden = hidden.float()
                result['hidden_states'][layer_idx] = hidden.cpu()

    # Store pooled states
    if pooling_method is not None:
        result['pooled_states'] = {}
        for layer_idx in layer_indices:
            if layer_idx < len(all_hidden_states):
                hidden = all_hidden_states[layer_idx]

                # Pool
                if pooling_method == 'mean':
                    # Mean pooling over sequence (considering attention mask)
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    pooled = sum_hidden / sum_mask

                elif pooling_method == 'last':
                    # Take last token (rightmost non-padded)
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = hidden.size(0)
                    pooled = hidden[range(batch_size), sequence_lengths]

                else:
                    raise ValueError(f"Unknown pooling_method: {pooling_method}")

                # Convert dtype
                if dtype == 'float16':
                    pooled = pooled.half()
                elif dtype == 'float32':
                    pooled = pooled.float()

                result['pooled_states'][layer_idx] = pooled.cpu()

    return result

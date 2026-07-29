"""
Save and checkpoint utilities
"""
import os
import torch
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from config import REPRESENTATION_DIR, CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def get_save_path(
        model_family: str,
        scale: str,
        variant: str,
        dataset_name: str,
        sample_id: int,
        output_dir: str = REPRESENTATION_DIR
) -> str:
    """
    Generate save path for a representation file

    Format: outputs/representations/{model_family}/{scale}_{variant}/{dataset}/{sample_id:05d}.pt

    Example: outputs/representations/gemma3/1b_base/mmlu/00042.pt
    """
    model_dir = f"{scale}_{variant}"
    dir_path = os.path.join(output_dir, model_family, model_dir, dataset_name)
    os.makedirs(dir_path, exist_ok=True)

    filename = f"{sample_id:05d}.pt"
    return os.path.join(dir_path, filename)


def check_exists(save_path: str) -> bool:
    """Check if representation file already exists"""
    return os.path.exists(save_path)


def save_representation(save_path: str, sample_data: Dict):
    """
    Save representation to disk atomically

    Args:
        save_path: Path to save file
        sample_data: Dict containing:
            - sample_id: int
            - input_text: str
            - input_ids: tensor
            - hidden_states: dict of {layer_idx: tensor} or None
            - pooled_states: dict of {layer_idx: tensor} or None
            - metadata: dict
    """
    # Atomic save: write to temp file, then rename
    temp_path = save_path + '.tmp'

    try:
        torch.save(sample_data, temp_path)
        logger.info(f" ✅ saved to: {temp_path}")
        os.rename(temp_path, save_path)
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def load_representation(save_path: str) -> Dict:
    """Load saved representation"""
    return torch.load(save_path)


def get_checkpoint_path(
        model_family: str,
        scale: str,
        variant: str,
        dataset_name: str,
        checkpoint_dir: str = CHECKPOINT_DIR
) -> str:
    """Get checkpoint file path"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f"{model_family}_{scale}_{variant}_{dataset_name}.json"
    return os.path.join(checkpoint_dir, filename)


def save_checkpoint(
        model_family: str,
        scale: str,
        variant: str,
        dataset_name: str,
        processed_ids: List[int],
        total_samples: int,
        checkpoint_dir: str = CHECKPOINT_DIR,
        processed_token_ids: Optional[List[int]] = None
):
    """
    Save checkpoint for resuming

    Args:
        model_family, scale, variant, dataset_name: Model and dataset info
        processed_ids: List of processed sample IDs (pooled)
        total_samples: Total number of samples
        checkpoint_dir: Directory to save checkpoint
        processed_token_ids: List of processed token-level sample IDs
    """
    checkpoint_path = get_checkpoint_path(
        model_family, scale, variant, dataset_name, checkpoint_dir
    )

    checkpoint = {
        'model_family': model_family,
        'scale': scale,
        'variant': variant,
        'dataset': dataset_name,
        'processed_sample_ids': sorted(processed_ids),
        'processed_token_sample_ids': sorted(processed_token_ids) if processed_token_ids else [],
        'total_samples': total_samples,
        'num_processed': len(processed_ids),
        'num_token_processed': len(processed_token_ids) if processed_token_ids else 0,
        'timestamp': datetime.now().isoformat()
    }

    # Atomic save
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.rename(temp_path, checkpoint_path)

    logger.info(f"  ✅  Checkpoint saved: {len(processed_ids)}/{total_samples} pooled, " +
                f"{len(processed_token_ids) if processed_token_ids else 0} token-level")


def load_checkpoint(
        model_family: str,
        scale: str,
        variant: str,
        dataset_name: str,
        checkpoint_dir: str = CHECKPOINT_DIR
) -> Dict:
    """
    Load checkpoint

    Returns:
        Checkpoint dict, or empty dict if not exists
    """
    checkpoint_path = get_checkpoint_path(
        model_family, scale, variant, dataset_name, checkpoint_dir
    )

    if not os.path.exists(checkpoint_path):
        return {
            'processed_sample_ids': [],
            'processed_token_sample_ids': [],
            'num_processed': 0,
            'num_token_processed': 0
        }

    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    # Backward compatibility
    if 'processed_token_sample_ids' not in checkpoint:
        checkpoint['processed_token_sample_ids'] = []
        checkpoint['num_token_processed'] = 0

    logger.info(f"  ✅  Loaded checkpoint: {checkpoint['num_processed']}/{checkpoint.get('total_samples', '?')} pooled, " +
                f"{checkpoint['num_token_processed']} token-level")

    return checkpoint
"""
Main script for extracting representations from models

Optimized logic:
- Precise model selection (family/scale pairs)
- Memory efficient: Load model ONCE -> Process all datasets -> Unload
- Dtype control: Separate model computation dtype and output storage dtype
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import logging
import torch
import random
import numpy as np
import gc
from tqdm import tqdm
from datetime import datetime

from config import (
    MODEL_CONFIGS, DATASET_CONFIGS,
    DATASET_CACHE_DIR, MODEL_CACHE_DIR,
    LOG_DIR, REPRESENTATION_DIR,
)
from data_utils import download_dataset, sample_dataset, create_dataloader
from model_utils import load_model, get_num_layers, parse_layer_indices, extract_representations
from save_utils import (
    get_save_path, check_exists, save_representation,
    save_checkpoint, load_checkpoint
)


def setup_logging():
    """Setup logging"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        LOG_DIR,
        f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract representations from language models')

    # Model Selection (Combined family/scale)
    parser.add_argument('--models', type=str, nargs='+',
                        default=['mistral/7b'],
                        choices=['mistral/7b', 'olmo2/1b', 'olmo2/7b', 'olmo2/13b', 'olmo2/32b'],
                        help='Model configs in format "family/scale" (e.g. "llama32/1b" "qwen25/7b")')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen'],
                        choices=['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen'],
                        help='Dataset names (e.g., mmlu gsm8k wikitext)')

    parser.add_argument('--variant', type=str, nargs='+', default=['base', 'sft'],
                        choices=['base', 'sft', 'dpo', 'rlvr', 'instruct'],  # <--- 这里增加了选项
                        help='Model variants to process for each model')
    # Representation extraction arguments
    parser.add_argument('--layer_indices', type=str, default='all',
                        choices=['all', 'key', 'sparse'],
                        help='Layer sampling strategy')

    # Computation arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device_map', type=str, default='auto',
                        help='Device map for model')

    # Dtype Control
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16', 'auto'],
                        help='Dtype for model loading and forward pass computation')
    parser.add_argument('--save_dtype', type=str, default='float16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Dtype for saving representations to disk (affects file size)')

    # System arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--dataset_cache_dir', type=str, default=DATASET_CACHE_DIR,
                        help='Dataset cache directory')
    parser.add_argument('--model_cache_dir', type=str, default=MODEL_CACHE_DIR,
                        help='Model cache directory')
    parser.add_argument('--output_dir', type=str, default=REPRESENTATION_DIR,
                        help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recompute existing files')
    parser.add_argument('--download_only', action='store_true',)

    return parser.parse_args()


def compute_pooled_from_hidden(hidden_states, attention_mask):
    """Compute pooled states from hidden states using mean pooling"""
    pooled_states = {}
    for layer_idx, hidden in hidden_states.items():
        # hidden shape: (B, T, D)
        mask_expanded = attention_mask.unsqueeze(-1).to(hidden.device).float()
        masked_hidden = hidden * mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask
        pooled_states[layer_idx] = pooled.cpu()
    return pooled_states


def process_dataset_with_model(
        model,
        tokenizer,
        model_family: str,
        scale: str,
        variant: str,
        dataset_name: str,
        args,
        logger
):
    """
    Process a single dataset using the ALREADY LOADED model.
    """
    logger.info(f"  > Processing Dataset: {dataset_name}")

    # Get dataset config
    dataset_config = DATASET_CONFIGS[dataset_name]
    is_token_level = dataset_config['type'] == 'token-level'
    n_samples_pooled = dataset_config['n_samples_pooled']
    n_samples_token = dataset_config['n_samples_token'] if is_token_level else 0
    max_length = dataset_config['max_length']

    # Load dataset
    dataset = download_dataset(dataset_name, args.dataset_cache_dir)
    dataset_sampled = sample_dataset(
        dataset,
        n_samples_pooled,
        args.seed,
        strategy='random'
    )

    # Parse layer indices
    num_layers = get_num_layers(model_family, scale)
    layer_indices = parse_layer_indices(args.layer_indices, num_layers)

    # Load checkpoint
    checkpoint = load_checkpoint(
        model_family, scale, variant, dataset_name
    ) if args.resume else {'processed_sample_ids': []}

    processed_ids = set(checkpoint['processed_sample_ids'])

    # Filter unprocessed samples
    all_indices = list(range(len(dataset_sampled)))
    remaining_indices = [i for i in all_indices if i not in processed_ids]

    if len(remaining_indices) == 0:
        logger.info(f"    ✓ Dataset {dataset_name} already fully processed!")
        return

    # Create dataloader
    remaining_dataset = dataset_sampled.select(remaining_indices)
    dataloader = create_dataloader(
        remaining_dataset,
        dataset_name,
        tokenizer,
        args.batch_size,
        max_length
    )

    logger.info(f"  ✅    Starting extraction for {len(remaining_indices)} samples...")

    token_level_count = 0
    pooled_only_count = 0

    # Determine save dtype map
    save_dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    target_save_dtype = save_dtype_map[args.save_dtype]

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"    Extracting [{dataset_name}]")):
        sample_ids = batch['sample_id']
        texts = batch['text']

        # Extract hidden states (Using model_dtype implicitly via model weights)
        # Note: We request output in args.save_dtype to save memory conversion steps later
        representations = extract_representations(
            model,
            batch,
            layer_indices,
            pooling_method=None,
            dtype=args.save_dtype # Pass save_dtype here to get desired output format directly
        )

        # Compute pooled states (will inherit dtype from representations)
        pooled_states = compute_pooled_from_hidden(
            representations['hidden_states'],
            batch['attention_mask']
        )

        # Save each sample
        for i, sample_id in enumerate(sample_ids):
            save_path = get_save_path(
                model_family, scale, variant,
                dataset_name, sample_id, args.output_dir
            )

            if not args.force_recompute and check_exists(save_path):
                processed_ids.add(sample_id)
                continue

            should_save_hidden = is_token_level and sample_id < n_samples_token

            # Prepare sample data
            sample_data = {
                'sample_id': sample_id,
                'input_text': texts[i],
                'input_ids': batch['input_ids'][i].cpu(),
                'hidden_states': None,
                'pooled_states': {},
                'metadata': {
                    'model_family': model_family,
                    'scale': scale,
                    'variant': variant,
                    'dataset': dataset_name,
                    'layer_indices': layer_indices,
                    'model_dtype': args.model_dtype, # Record what model ran in
                    'save_dtype': args.save_dtype,   # Record what we saved
                    'has_token_level': should_save_hidden,
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Add pooled states
            for layer_idx in pooled_states:
                # Ensure tensor is in target save dtype
                sample_data['pooled_states'][layer_idx] = pooled_states[layer_idx][i].to(target_save_dtype)

            # Add hidden states if needed
            if should_save_hidden:
                sample_data['hidden_states'] = {}
                for layer_idx in representations['hidden_states']:
                    sample_data['hidden_states'][layer_idx] = representations['hidden_states'][layer_idx][i].to(target_save_dtype)
                token_level_count += 1
            else:
                pooled_only_count += 1

            save_representation(save_path, sample_data)
            processed_ids.add(sample_id)

    # Save checkpoint after finishing the dataset
    save_checkpoint(
        model_family, scale, variant, dataset_name,
        list(processed_ids), len(dataset_sampled)
    )
    logger.info(f"  ✅✅✅✅   Extracting Representations End~~✅ ✅ ✅ ")


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logger = setup_logging()

    # Validate inputs
    parsed_models = []
    for model_str in args.models:
        try:
            family, scale = model_str.split('/')
            if family not in MODEL_CONFIGS or scale not in MODEL_CONFIGS[family]:
                raise ValueError(f"Invalid model config: {model_str}")
            parsed_models.append((family, scale))
        except ValueError:
            raise ValueError(f"Format error. Use 'family/scale', got '{model_str}'")

    for dataset_name in args.dataset:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info("=" * 20)
    logger.info(f" ✅ Target Models: {parsed_models}")
    logger.info(f" ✅ Target Variants: {args.variant}")
    logger.info(f" ⭕️ Target Datasets: {args.dataset}")
    logger.info(f"Dtypes -> Model: {args.model_dtype} | Save: {args.save_dtype}")
    logger.info("=" * 20)

    # Outer Loop: Model Family/Scale
    # We load the model once, run all variants/datasets, then unload.
    for family, scale in parsed_models:

        # Middle Loop: Variant (Base vs Instruct)
        # Note: Base and Instruct are different weights, so we MUST reload model here.
        for variant in args.variant:
            if variant not in MODEL_CONFIGS[family][scale]:
                logger.warning(f"Variant {variant} not found for {family}/{scale}, skipping...")
                continue

            logger.info(f"{'=' * 40}")
            logger.info(f" ❤️ Loading Model: {family}-{scale}-{variant}")
            logger.info(f"{'=' * 40}")

            try:
                # 1. Load Model
                model, tokenizer = load_model(
                    family,
                    scale,
                    variant,
                    args.model_cache_dir,
                    args.device_map,
                    args.model_dtype, # Use the computation dtype
                    download_only=args.download_only
                )

                # 2. Iterate Datasets with SAME model
                for dataset_name in args.dataset:
                    process_dataset_with_model(
                        model,
                        tokenizer,
                        family,
                        scale,
                        variant,
                        dataset_name,
                        args,
                        logger
                    )

                logger.info(f" ✅ Completed all datasets for {family}-{scale}-{variant}")

            except Exception as e:
                logger.error(f"Error processing {family}-{scale}-{variant}: {e}", exc_info=True)

            finally:
                # 3. Clean up memory before next model/variant
                logger.info("Cleaning up GPU memory...")
                if 'model' in locals():
                    del model
                if 'tokenizer' in locals():
                    del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared.\n")

    logger.info("=" * 20)
    logger.info(" ✅  ALL TASKS COMPLETED")
    logger.info("=" * 20)


if __name__ == '__main__':
    main()
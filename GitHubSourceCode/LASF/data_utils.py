"""
Data loading and preprocessing utilities
"""
import os
import random
import numpy as np
from datasets import load_dataset, Dataset, load_from_disk
from typing import Dict, List, Tuple
import logging

from config import DATASET_CONFIGS, DATASET_CACHE_DIR

logger = logging.getLogger(__name__)


def download_dataset(dataset_name: str, cache_dir: str = DATASET_CACHE_DIR) -> Dataset:
    config = DATASET_CONFIGS[dataset_name]

    local_path = os.path.join(cache_dir,  dataset_name)

    if os.path.exists(local_path):
        logger.info(f" ✅ Loading {dataset_name} from local disk: {local_path}")
        return load_from_disk(local_path)

    logger.info(f" ✅ Downloading {dataset_name} from HuggingFace...")
    if config['subset']:
        dataset = load_dataset(
            config['hf_name'],
            config['subset'],
            split=config['split'],
            cache_dir=cache_dir,
        )
    else:
        dataset = load_dataset(
            config['hf_name'],
            split=config['split'],
            cache_dir=cache_dir,
        )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    dataset.save_to_disk(local_path)
    logger.info(f"Saved {dataset_name} to {local_path}")

    return dataset




def format_sample(dataset_name: str, sample: Dict) -> str:
    """
    Format a single sample into text string
    """
    config = DATASET_CONFIGS[dataset_name]
    format_type = config['format_type']

    try:
        if format_type == 'multiple_choice':
            # MMLU format
            question = sample.get('question', '')
            choices = sample.get('choices', [])
            answer = sample.get('answer', 0)

            choice_labels = ['A', 'B', 'C', 'D']
            formatted_choices = ' '.join([
                f"{choice_labels[i]}) {choice}"
                for i, choice in enumerate(choices)
            ])
            formatted = f"Question: {question}\n{formatted_choices}\nAnswer: {choice_labels[answer]}"

        elif format_type == 'qa':
            # GSM8K format
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            formatted = f"Question: {question}\nAnswer: {answer}"

        elif format_type == 'text':
            # WikiText format
            formatted = sample.get('text', '')
            if len(formatted.strip()) < 10:
                return None

        elif format_type == 'code':
            # HumanEval format
            prompt = sample.get('prompt', '')
            canonical_solution = sample.get('canonical_solution', '')
            formatted = f"{prompt}\n{canonical_solution}"

        elif format_type == 'instruction':
            # IFEval format
            prompt = sample.get('prompt', '')
            formatted = prompt

        elif format_type == 'conversation':
            # MT-Bench format
            # MT-Bench has multi-turn conversations
            turns = sample.get('turns', [])
            if isinstance(turns, list) and len(turns) > 0:
                # Use first turn for representation extraction
                formatted = turns[0]
            else:
                formatted = sample.get('prompt', '')

        elif format_type == 'text_classification':
            # ToxiGen format
            text = sample.get('text', '')
            # Optional: include label for context
            # label = sample.get('toxicity_ai', 0)
            formatted = text
            if len(formatted.strip()) < 10:
                return None

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        return formatted

    except Exception as e:
        logger.warning(f"Failed to format sample: {e}")
        return None


def sample_dataset(
        dataset: Dataset,
        n_samples: int,
        seed: int = 42,
        strategy: str = 'random'
) -> Dataset:
    """
    Sample n_samples from dataset

    Args:
        dataset: HuggingFace Dataset
        n_samples: Number of samples to select
        seed: Random seed
        strategy: 'random' or 'stratified'

    Returns:
        Sampled dataset
    """
    random.seed(seed)
    np.random.seed(seed)

    total_samples = len(dataset)
    n_samples = min(n_samples, total_samples)

    if strategy == 'random':
        indices = random.sample(range(total_samples), n_samples)

    elif strategy == 'stratified':
        # For MMLU: stratify by subject
        if 'subject' in dataset.column_names:
            subjects = dataset['subject']
            unique_subjects = list(set(subjects))
            samples_per_subject = n_samples // len(unique_subjects)

            indices = []
            for subject in unique_subjects:
                subject_indices = [i for i, s in enumerate(subjects) if s == subject]
                if len(subject_indices) <= samples_per_subject:
                    indices.extend(subject_indices)
                else:
                    indices.extend(random.sample(subject_indices, samples_per_subject))

            # Fill remaining slots randomly if needed
            if len(indices) < n_samples:
                remaining = n_samples - len(indices)
                all_indices = set(range(total_samples))
                available = list(all_indices - set(indices))
                indices.extend(random.sample(available, remaining))
        else:
            # Fallback to random
            indices = random.sample(range(total_samples), n_samples)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    sampled = dataset.select(indices)
    logger.info(f"Sampled {len(sampled)} samples using {strategy} strategy")

    return sampled


def create_dataloader(
        dataset: Dataset,
        dataset_name: str,
        tokenizer,
        batch_size: int,
        max_length: int
):
    """
    Create PyTorch DataLoader with tokenization

    Args:
        dataset: HuggingFace Dataset
        dataset_name: Name of dataset
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Max sequence length

    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader

    # Format all samples
    formatted_texts = []
    valid_indices = []

    for idx, sample in enumerate(dataset):
        text = format_sample(dataset_name, sample)
        if text is not None:
            formatted_texts.append(text)
            valid_indices.append(idx)

    logger.info(f"  ✅ Formatted {len(formatted_texts)}/{len(dataset)} samples")

    # Tokenize
    encoded = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Will pad in collate_fn
        return_tensors=None
    )

    # Create dataset with sample_id
    processed_dataset = []
    for i, sample_id in enumerate(valid_indices):
        processed_dataset.append({
            'sample_id': sample_id,
            'input_ids': encoded['input_ids'][i],
            'attention_mask': encoded['attention_mask'][i],
            'text': formatted_texts[i]
        })

    def collate_fn(batch):
        """Custom collate function with dynamic padding"""
        import torch

        sample_ids = [item['sample_id'] for item in batch]
        texts = [item['text'] for item in batch]

        # Pad sequences
        max_len = max(len(item['input_ids']) for item in batch)

        input_ids = []
        attention_masks = []

        for item in batch:
            ids = item['input_ids']
            mask = item['attention_mask']

            # Pad to max_len
            padding_length = max_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * padding_length
            mask = mask + [0] * padding_length

            input_ids.append(ids)
            attention_masks.append(mask)

        return {
            'sample_id': sample_ids,
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_masks),
            'text': texts
        }

    dataloader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Must be 0 for tokenizer safety
        pin_memory=True
    )

    return dataloader

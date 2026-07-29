"""
Configuration file for models and datasets
"""
import os

# ============================================================================
# Directories
# ============================================================================
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/xxx/xxx/base_sft/dataset_cache')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/xxx/xxx/base_sft/model_cache')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/xxx/xxx/base_sft/outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
REPRESENTATION_DIR = os.path.join(OUTPUT_DIR, 'representations')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
METRIC_DIR = os.path.join(OUTPUT_DIR, 'metrics')

# ============================================================================
# Model Configurations
# ============================================================================
# [mistral/7b, olmo2/1b, olmo2/7b, olmo2/13b, olmo2/32b]
MODEL_CONFIGS = {
    'mistral': {
        '7b': {
            'base': 'mistralai/Mistral-7B-v0.1',
            'sft': 'mistralai/Mistral-7B-Instruct-v0.1',
            'num_layers': 32,
        },
    },
    'olmo2': { 
        '1b': {
            'base': 'allenai/OLMo-2-0425-1B',
            'sft': 'allenai/OLMo-2-0425-1B-SFT',
            'dpo': 'allenai/OLMo-2-0425-1B-DPO',
            'rlvr': 'allenai/OLMo-2-0425-1B-RLVR1', 
            'instruct': 'allenai/OLMo-2-0425-1B-Instruct', 
            'num_layers': 16, # Estimated for 1B size
        },
        '7b': {
            'base': 'allenai/OLMo-2-1124-7B',
            'sft': 'allenai/OLMo-2-1124-7B-SFT',
            'dpo': 'allenai/OLMo-2-1124-7B-DPO',
            'rlvr': 'allenai/OLMo-2-1124-7B-Instruct', 
            # 'instruct': 'allenai/OLMo-2-1124-7B-Instruct',
            'num_layers': 32, # Confirmed
        },
        '13b': {
            'base': 'allenai/OLMo-2-1124-13B',
            'sft': 'allenai/OLMo-2-1124-13B-SFT',
            'dpo': 'allenai/OLMo-2-1124-13B-DPO',
            'rlvr': 'allenai/OLMo-2-1124-13B-Instruct',
            # 'instruct': 'allenai/OLMo-2-1124-13B-Instruct',
            'num_layers': 40, # Confirmed
        },
        '32b': {
            'base': 'allenai/OLMo-2-0325-32B',
            'sft': 'allenai/OLMo-2-0325-32B-SFT',
            'dpo': 'allenai/OLMo-2-0325-32B-DPO', 
            'rlvr': 'allenai/OLMo-2-0325-32B-Instruct',
            # 'instruct': 'allenai/OLMo-2-0325-32B-Instruct',
            'num_layers': 64, # Estimated, verify with config.json
        },
    },
}
# ============================================================================
# Dataset Configurations
# ============================================================================
MAX_LENGTH = 128
N_POOLED = 200
N_TOKEN = 50
# ['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen']
DATASET_CONFIGS = {
    # Token-level datasets: save both pooled (1000 samples) and token-level (100 samples)
    'mmlu': {
        'hf_name': 'cais/mmlu',
        'subset': 'all',
        'split_test': 'test',
        'split_train': 'auxiliary_train', # Training split
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,  # Number of samples for pooled representations
        'n_samples_token': N_TOKEN,     # Number of samples for token-level representations
        'max_length': MAX_LENGTH,
        'format_type': 'multiple_choice',
        'metric_mode': 'exact_match',
    },
    'gsm8k': {
        'hf_name': 'openai/gsm8k',
        'subset': 'main',
        'split_test': 'test',
        'split_train': 'train', # Training split
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'qa',
        'metric_mode': 'exact_match',
    },
    'gsm8kgradient': {
            'hf_name': 'openai/gsm8k',
            'subset': 'main',
            'split_test': 'train',
            'split_train': 'train',
            'type': 'token-level',
            'n_samples_pooled': N_POOLED,
            'n_samples_token': N_TOKEN,
            'max_length': MAX_LENGTH,
            'format_type': 'qa',
        },
    'wikitext': {
        'hf_name': 'Salesforce/wikitext',
        'subset': 'wikitext-103-v1',
        'split_test': 'test',
        'split_train': 'train',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'text',
        'metric_mode': 'perplexity',
    },
    'ifeval': {
        'hf_name': 'google/IFEval',
        'subset': None,
        'split_test': 'train',
        'split_train': 'train',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'instruction',
        'metric_mode': 'loss',
    },

    # Pooled-only datasets
    'humaneval': {
        'hf_name': 'openai_humaneval',
        'subset': None,
        'split_test': 'test',
        'split_train': 'train',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'code',
        'metric_mode': 'loss',
    },
    'mt_bench': {
        'hf_name': 'HuggingFaceH4/mt_bench_prompts',
        'subset': None,
        'split_test': 'train',
        'split_train': 'train',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'conversation',
        'metric_mode': 'loss',
    },
    'toxigen': {
        'hf_name': 'toxigen/toxigen-data',
        'subset': 'annotated',
        'split_test': 'test',
        'split_train': 'train',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'text_classification',
        'metric_mode': 'perplexity',
    },
}

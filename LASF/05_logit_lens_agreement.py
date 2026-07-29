
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import logging
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import numpy as np

from config import REPRESENTATION_DIR, MODEL_CONFIGS, METRIC_DIR
from model_utils import load_model

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Plot Style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: Logit Lens Agreement (Base vs SFT)")

    # 1. Models: Combine family and scale (e.g. olmo2/1b)
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/1b'],
                        help='Model configs in format "family/scale" (e.g. olmo2/1b)')

    # 2. Datasets: Support multiple datasets on one plot
    parser.add_argument('--datasets', type=str, nargs='+', default=['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval'],
                        help='List of datasets to analyze (e.g. mmlu gsm8k)')

    # 3. Variants
    parser.add_argument('--sft_variant', type=str, default='sft', 
                        help='Variant to compare with base (e.g. sft, dpo, rlvr)')

    parser.add_argument('--max_samples', type=int, default=100, help='Max samples to analyze')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def get_lightweight_components(model_family, scale, variant, device):
    """
    Load model, extract Head and Norm, then delete model to save VRAM.
    """
    logger.info(f"    🏗️ Loading {variant} components for {model_family}/{scale}...")
    try:
        model, _ = load_model(
            model_family, scale, variant, 
            download_only=False, device_map=device
        )

        # Extract components
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm
        elif hasattr(model, 'norm'):
            norm = model.norm
        else:
            logger.warning("    ⚠️ Could not find LayerNorm. Using Identity.")
            norm = torch.nn.Identity()

        lm_head = model.lm_head

        # Move to device and detach from graph
        norm = norm.to(device)
        lm_head = lm_head.to(device)

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return norm, lm_head
    except Exception as e:
        logger.error(f"    ❌ Error loading {variant}: {e}")
        return None, None

def load_reps(path):
    """Load representation and extract the last token vector"""
    try:
        data = torch.load(path, map_location='cpu')
        # We need token-level hidden states (Last token)
        if 'hidden_states' in data and data['hidden_states'] is not None:
            # Sort layers
            layers = sorted(data['hidden_states'].keys())
            # Extract last token state: (D,)
            # hidden_states is usually (T, D) or (1, T, D)
            reps = {}
            for l in layers:
                tensor = data['hidden_states'][l]
                if tensor.dim() == 3: # (B, T, D)
                    reps[l] = tensor[0, -1, :] 
                elif tensor.dim() == 2: # (T, D)
                    reps[l] = tensor[-1, :]
                else:
                    reps[l] = tensor
            return reps
    except Exception as e:
        return None
    return None

def process_single_model(family, scale, args):
    """Process one model scale
with multiple datasets"""
    logger.info(f"🚀 Processing Model: {family}/{scale}")

    # 1. Load Components (Base & SFT) separately
    # This ensures we use the correct decoder for each representation
    base_norm, base_head = get_lightweight_components(family, scale, 'base', args.device)
    sft_norm, sft_head = get_lightweight_components(family, scale, args.sft_variant, args.device)

    if base_norm is None or sft_norm is None:
        logger.error("Failed to load model components. Skipping.")
        return

    # Store results for plotting: {dataset_name: [agreement_per_layer]}
    dataset_results = {}

    for dataset in args.datasets:
        logger.info(f"  📂 Analyzing Dataset: {dataset}")

        base_dir = os.path.join(REPRESENTATION_DIR, family, f"{scale}_base", dataset)
        sft_dir = os.path.join(REPRESENTATION_DIR, family, f"{scale}_{args.sft_variant}", dataset)

        if not os.path.exists(base_dir) or not os.path.exists(sft_dir):
            logger.warning(f"    ⚠️ Data not found for {dataset}. Skipping.")
            continue

        base_files = sorted(glob(os.path.join(base_dir, "*.pt")))
        sft_files = sorted(glob(os.path.join(sft_dir, "*.pt")))

        # Map by filename to ensure matching
        base_map = {os.path.basename(f): f for f in base_files}
        sft_map = {os.path.basename(f): f for f in sft_files}

        common_ids = sorted(list(set(base_map.keys()) & set(sft_map.keys())))
        if len(common_ids) > args.max_samples:
            common_ids = common_ids[:args.max_samples]

        logger.info(f"    🔍 Found {len(common_ids)} common samples.")

        layer_agreements = defaultdict(list)

        for fname in tqdm(common_ids, desc=f"    Computing ({dataset})", leave=False):
            base_reps = load_reps(base_map[fname])
            sft_reps = load_reps(sft_map[fname])

            if not base_reps or not sft_reps: continue

            common_layers = sorted(list(set(base_reps.keys()) & set(sft_reps.keys())))

            for layer in common_layers:
                # Move to GPU
                h_base = base_reps[layer].to(args.device).float()
                h_sft = sft_reps[layer].to(args.device).float()

                with torch.no_grad():
                    # Base decoding path
                    logits_base = base_head(base_norm(h_base))
                    top1_base = torch.argmax(logits_base).item()

                    # SFT decoding path (CORRECTED: Use SFT Head)
                    logits_sft = sft_head(sft_norm(h_sft))
                    top1_sft = torch.argmax(logits_sft).item()

                    # Check match
                    agree = 1.0 if top1_base == top1_sft else 0.0
                    layer_agreements[layer].append(agree)

        # Aggregate
        if layer_agreements:
            layers = sorted(layer_agreements.keys())
            avg_agreement = [np.mean(layer_agreements[l]) for l in layers]
            dataset_results[dataset] = (layers, avg_agreement)

    # Plotting for this model scale
    if dataset_results:
        plot_results(family, scale, args.sft_variant, dataset_results)

def plot_results(family, scale, sft_variant, dataset_results):
    plt.figure(figsize=(8, 6), dpi=300) # Increased size

    markers = ['o', 's', '^', 'D', 'v', '<', '>']

    for idx, (dataset, (layers, scores)) in enumerate(dataset_results.items()):
        marker = markers[idx % len(markers)]
        plt.plot(
            layers, scores, 
            marker=marker, 
            markersize=8, 
            linewidth=2.5, # Thicker lines
            label=dataset,
            alpha=0.9
        )

    plt.title(f"Logit Lens Top-1 Agreement\n{family}/{scale} (Base vs {sft_variant})", fontsize=18, fontweight='bold')
    plt.xlabel("Layer Depth", fontsize=14)
    plt.ylabel("Agreement Rate", fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_name = f"logit_lens_agreement_{family}_{scale}.png"
    save_path = os.path.join(METRIC_DIR, save_name)
    plt.savefig(save_path)
    logger.info(f"📸 Plot saved to {save_path}")
    plt.show() # Commented out for batch running
    plt.close()

def main():
    args = parse_args()

    # Ensure metric dir exists
    os.makedirs(METRIC_DIR, exist_ok=True)

    # Loop over models
    for model_str in args.models:
        if '/' not in model_str:
            logger.error(f"Invalid model format: {model_str}. Use family/scale")
            continue

        family, scale = model_str.split('/')

        # Check config validity
        if family not in MODEL_CONFIGS or scale not in MODEL_CONFIGS[family]:
             logger.error(f"Model {family}/{scale} not found in config.py")
             continue

        process_single_model(family, scale, args)

if __name__ == "__main__":
    main()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import logging
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm

from config import REPRESENTATION_DIR, MODEL_CONFIGS, METRIC_DIR
from model_utils import load_model

# Setup Logging & Style
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: Logit Lens Visualization (Trace & Confidence)")

    # 1. Models: Combined family/scale
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/1b'],
                        help='Model configs in format "family/scale" (e.g. olmo2/7b)')

    # 2. Datasets
    parser.add_argument('--datasets', type=str, nargs='+', default=['mmlu'],
                        help='List of datasets (e.g. mmlu gsm8k)')

    # 3. Variants
    parser.add_argument('--sft_variant', type=str, default='sft',
                        help='Variant to compare with base (e.g. sft, dpo)')

    # 4. Sampling Control
    parser.add_argument('--sample_ids', type=int, nargs='+', default=[3, 4, 5],
                        help='Specific sample IDs to visualize. If None, uses max_samples.')
    parser.add_argument('--max_samples', type=int, default=1,
                        help='Number of samples to visualize if sample_ids is not provided.')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()


def get_lightweight_components(model_family, scale, variant, device):
    """
    Load model, extract Head and Norm, then delete model to save VRAM.
    """
    logger.info(f"    🏗️ Loading components for {model_family}/{scale} ({variant})...")
    try:
        model, tokenizer = load_model(
            model_family, scale, variant,
            download_only=False, device_map=device
        )

        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm
        elif hasattr(model, 'norm'):
            norm = model.norm
        else:
            norm = torch.nn.Identity()

        lm_head = model.lm_head

        # Move to device
        norm = norm.to(device)
        lm_head = lm_head.to(device)

        # Keep tokenizer
        tokenizer = tokenizer

        # Cleanup model body
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return norm, lm_head, tokenizer
    except Exception as e:
        logger.error(f"    ❌ Error loading {variant}: {e}")
        return None, None, None


def decode_layer(hidden, norm, head, tokenizer):
    """
    Decode a single layer's hidden state.
    Returns: (Top-1 Token String, Top-1 Probability)
    """
    with torch.no_grad():
        # Norm -> Head
        # hidden: (D,)
        h_device = hidden.to(head.weight.device).float()
        normalized = norm(h_device)
        logits = head(normalized)
        probs = torch.softmax(logits, dim=-1)

        # Top-1
        top1_id = torch.argmax(logits).item()
        top1_prob = probs[top1_id].item()

        # Decode
        token_str = tokenizer.decode([top1_id]).strip()
        # Handle special tokens for cleaner display
        token_str = repr(token_str) if not token_str else token_str

        return token_str, top1_prob


def visualize_sample(
        family, scale, dataset, sample_id,
        base_path, sft_path,
        base_components, sft_components,
        args
):
    """Process and visualize a single sample"""
    base_norm, base_head, base_tok = base_components
    sft_norm, sft_head, sft_tok = sft_components

    # Load Data
    base_data = torch.load(base_path, map_location='cpu')
    sft_data = torch.load(sft_path, map_location='cpu')

    layers = sorted(base_data['hidden_states'].keys())

    # Containers for Plotting
    plot_data = {
        'layer': [],
        'base_prob': [],
        'sft_prob': [],
        'base_token': [],
        'sft_token': []
    }

    records = []

    # Decode Layer by Layer
    for layer in layers:
        # Extract last token vector
        # Handle (1, T, D) or (T, D) shape
        h_base = base_data['hidden_states'][layer]
        if h_base.dim() == 3:
            h_base = h_base[0, -1, :]
        elif h_base.dim() == 2:
            h_base = h_base[-1, :]

        h_sft = sft_data['hidden_states'][layer]
        if h_sft.dim() == 3:
            h_sft = h_sft[0, -1, :]
        elif h_sft.dim() == 2:
            h_sft = h_sft[-1, :]

        # Decode Base (using Base components)
        t_base, p_base = decode_layer(h_base, base_norm, base_head, base_tok)

        # Decode SFT (using SFT components)
        t_sft, p_sft = decode_layer(h_sft, sft_norm, sft_head, sft_tok)

        # Store
        plot_data['layer'].append(layer)
        plot_data['base_prob'].append(p_base)
        plot_data['sft_prob'].append(p_sft)
        plot_data['base_token'].append(t_base)
        plot_data['sft_token'].append(t_sft)

        records.append({
            "Layer": layer,
            "Base Token": t_base,
            "Base Conf": f"{p_base:.1%}",
            "SFT Token": t_sft,
            "SFT Conf": f"{p_sft:.1%}",
            "Diff": "Wait" if t_base == t_sft else "Split"
        })

    # --- 1. Print Text Table ---
    print(f"\n{'=' * 20} Sample {sample_id} ({dataset}) {'=' * 20}")
    print(f"Prompt Snippet: {base_data['input_text'][:60]}..., {base_data['input_text'][60:70]}")
    df = pd.DataFrame(records)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(df)

    # --- 2. Plot Confidence Curve ---
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot Base Line
    plt.plot(plot_data['layer'], plot_data['base_prob'],
             label='Base Confidence', color='#7f7f7f',
             linestyle='--', linewidth=2.5, marker='o', markersize=8, alpha=0.8)

    # Plot SFT Line
    plt.plot(plot_data['layer'], plot_data['sft_prob'],
             label=f'{args.sft_variant.upper()} Confidence', color='#d62728',
             linestyle='-', linewidth=3.0, marker='s', markersize=8)

    # Add Text Labels for key changes (optional, avoiding clutter)
    # Just label the final tokens
    plt.annotate(f"Base: {plot_data['base_token'][-1]}",
                 (layers[-1], plot_data['base_prob'][-1]),
                 textcoords="offset points", xytext=(-20, 10), ha='right', fontsize=12)
    plt.annotate(f"SFT: {plot_data['sft_token'][-1]}",
                 (layers[-1], plot_data['sft_prob'][-1]),
                 textcoords="offset points", xytext=(-20, -15), ha='right', fontsize=12, color='#d62728')

    plt.title(f"Logit Lens Top-1 Confidence Evolution\n{family}/{scale} - Sample {sample_id}", fontsize=16,
              fontweight='bold')
    plt.xlabel("Layer Depth", fontsize=14, fontweight='bold')
    plt.ylabel("Top-1 Token Probability", fontsize=14, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=12, loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save
    save_name = f"logit_lens_vis_{family}_{scale}_{dataset}_{sample_id}.png"
    save_path = os.path.join(METRIC_DIR, save_name)
    plt.savefig(save_path)
    plt.show()
    logger.info(f"    📸 Plot saved to {save_path}")

    # plt.show() # Uncomment if running interactively
    plt.close()


def process_single_model(family, scale, args):
    """Load components once, then iterate datasets/samples"""
    logger.info(f"🚀 Processing Model: {family}/{scale}")

    # Load Components Separately!
    base_comps = get_lightweight_components(family, scale, 'base', args.device)
    sft_comps = get_lightweight_components(family, scale, args.sft_variant, args.device)

    if None in base_comps or None in sft_comps:
        logger.error("Failed to load components. Skipping.")
        return

    for dataset in args.datasets:
        logger.info(f"  📂 Dataset: {dataset}")

        base_dir = os.path.join(REPRESENTATION_DIR, family, f"{scale}_base", dataset)
        sft_dir = os.path.join(REPRESENTATION_DIR, family, f"{scale}_{args.sft_variant}", dataset)

        base_files = sorted(glob(os.path.join(base_dir, "*.pt")))
        sft_files = sorted(glob(os.path.join(sft_dir, "*.pt")))

        base_map = {os.path.basename(f): f for f in base_files}
        sft_map = {os.path.basename(f): f for f in sft_files}

        common_ids = sorted(list(set(base_map.keys()) & set(sft_map.keys())))

        # Determine which samples to run
        target_ids = []
        if args.sample_ids:
            # User specified IDs (parse filename to ID if needed, here assuming ID in filename)
            # Filenames are 00042.pt
            for target in args.sample_ids:
                fname = f"{target:05d}.pt"
                if fname in common_ids:
                    target_ids.append(fname)
        else:
            # Take first N
            target_ids = common_ids[:args.max_samples]

        logger.info(f"    🔍 Visualizing {len(target_ids)} samples...")

        for fname in target_ids:
            # Parse ID from filename
            sid = int(fname.split('.')[0])
            visualize_sample(
                family, scale, dataset, sid,
                base_map[fname], sft_map[fname],
                base_comps, sft_comps,
                args
            )


def main():
    args = parse_args()
    os.makedirs(METRIC_DIR, exist_ok=True)

    for model_str in args.models:
        if '/' not in model_str: continue
        family, scale = model_str.split('/')
        if family not in MODEL_CONFIGS or scale not in MODEL_CONFIGS[family]:
            continue

        process_single_model(family, scale, args)


if __name__ == "__main__":
    main()
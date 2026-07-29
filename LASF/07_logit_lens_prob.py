import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import torch
import logging
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from config import MODEL_CONFIGS, METRIC_DIR
from model_utils import load_model
from data_utils import download_dataset, format_sample

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup Plot Style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")
# 论文级绘图参数
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 3: Aggregated Probability Curve (Mean + Std)")

    # 1. Models
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/1b'],
                        choices=['llama32/1b', 'llama32/3b', 'gemma3/1b', 'gemma3/27b', 'mistral/7b',
                                 'qwen25/7b', 'qwen25/14b', 'qwen25/32b', 'qwen25/72b',
                                 'olmo2/1b', 'olmo2/7b', 'olmo2/13b', 'olmo2/32b'],
                        help='Model configs (e.g. olmo2/1b)')

    # 2. Variants
    parser.add_argument('--variants', type=str, nargs='+', default=['base', 'sft'],
                        help='Variants to compare (default: base sft)')

    # 3. Dataset
    parser.add_argument('--dataset', type=str, default='mmlu', choices=['mmlu', 'gsm8k'],
                        help='Dataset name (Recommended: mmlu, gsm8k)')

    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to aggregate (More is better for smooth curves)')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def get_gold_target_token_id(dataset_name, sample, tokenizer):
    """
    解析 Dataset 的 Ground Truth，并返回对应的 Token ID
    """
    target_str = None

    # --- 1. MMLU ---
    if dataset_name == 'mmlu':
        # sample['answer'] is int 0-3
        ans_idx = sample.get('answer')
        mapping = [' A', ' B', ' C', ' D']  # 注意前导空格
        if ans_idx is not None and 0 <= ans_idx < 4:
            target_str = mapping[ans_idx]

    # --- 2. GSM8K ---
    elif dataset_name == 'gsm8k':
        # sample['answer'] format: "Explanation... #### 2280"
        raw_ans = sample.get('answer', '')
        if '####' in raw_ans:
            # 取 #### 后面的数字部分
            final_answer = raw_ans.split('####')[-1].strip()
            # 加上前导空格 (因为前面通常是换行或空格)
            target_str = " " + final_answer
        else:
            # 格式不对，跳过
            return None

    if target_str is None:
        return None

    # Tokenize
    # add_special_tokens=False is CRITICAL
    ids = tokenizer.encode(target_str, add_special_tokens=False)

    if not ids:
        return None

    # 取第一个 token。
    # 例如 " 2280" 可能会被切分成 [" 2", "2", "80"]，我们只看第一个 " 2"
    return ids[0]


def compute_probs_batch(model, tokenizer, batch_samples, device, dataset_name):
    """
    Process a batch of samples and return probability traces.
    Returns: {sample_idx: [prob_layer_0, prob_layer_1, ...]}
    """
    # Components
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'norm'):
        norm = model.norm
    else:
        norm = torch.nn.Identity()
    lm_head = model.lm_head

    results = {}

    for idx, (text, raw_sample) in batch_samples.items():
        # 1. Find Target ID
        target_id = get_gold_target_token_id(dataset_name, raw_sample, tokenizer)
        if target_id is None:
            continue

        # 2. Tokenize Input
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # 3. Forward
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        # 4. Trace Probabilities
        sample_probs = []
        for h in hidden_states:
            # Last token: (D,)
            h_last = h[0, -1, :].float()

            # Logit Lens
            logits = lm_head(norm(h_last))
            probs = torch.softmax(logits, dim=-1)

            p = probs[target_id].item()
            sample_probs.append(p)

        results[idx] = sample_probs

    return results


def process_model_aggregation(family, scale, args, samples_batch):
    logger.info(f"🚀 Processing {family}/{scale} (Aggregating {len(samples_batch)} samples)")

    # Data storage: {variant: [[layer0, layer1...], [layer0...], ...]}
    # Matrix shape: (N_samples, N_layers)
    variant_data = {}

    # 1. Iterate Variants
    for variant in args.variants:
        logger.info(f"  🏗️ Loading {variant}...")
        model, tokenizer = load_model(family, scale, variant, device_map=args.device)

        # Compute probs for all samples
        # dict: {idx: [p0, p1...]}
        raw_results = compute_probs_batch(model, tokenizer, samples_batch, args.device, args.dataset)

        # Convert to matrix for aggregation
        matrix = []
        for idx in sorted(raw_results.keys()):
            matrix.append(raw_results[idx])

        if matrix:
            variant_data[variant] = np.array(matrix)  # (N, Layers)
            logger.info(f"    ✅ Collected data for {len(matrix)} samples")
        else:
            logger.warning(f"    ⚠️ No valid data collected for {variant}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Plotting (Aggregated)
    if not variant_data:
        return

    logger.info("  📊 Generating Aggregated Plot...")
    plt.figure(figsize=(10, 7), dpi=300)

    colors = {'base': '#7f7f7f', 'sft': '#d62728', 'dpo': '#1f77b4', 'rlvr': '#2ca02c'}

    for variant, matrix in variant_data.items():
        # matrix: (N_samples, N_layers)
        # Compute Mean and Std along axis 0 (samples)
        mean_curve = np.mean(matrix, axis=0)
        std_curve = np.std(matrix, axis=0)

        # Semantic Error Band (Mean +/- 0.2 * Std for visualization clarity, or full Std)
        # Usually Standard Error (Std / sqrt(N)) is better for confidence intervals,
        # but here we want to show data variance, so let's use 0.5 * Std to not clutter.
        lower_bound = np.clip(mean_curve - 0.5 * std_curve, 0, 1)
        upper_bound = np.clip(mean_curve + 0.5 * std_curve, 0, 1)

        layers = range(len(mean_curve))
        color = colors.get(variant, 'black')

        # Plot Mean
        plt.plot(layers, mean_curve,
                 label=f"{variant.upper()} (Mean)",
                 color=color, linewidth=3, marker='o', markersize=6)

        # Plot Shadow (Variance)
        plt.fill_between(layers, lower_bound, upper_bound,
                         color=color, alpha=0.15)

    plt.title(f"Target Probability Evolution (Aggregated N={len(matrix)})\n{family}/{scale} on {args.dataset.upper()}",
              fontweight='bold', pad=15)
    plt.xlabel("Layer Depth", fontweight='bold')
    plt.ylabel("Probability of Correct Answer", fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(METRIC_DIR, f"agg_prob_{family}_{scale}_{args.dataset}.png")
    plt.savefig(save_path)
    logger.info(f"    📸 Saved: {save_path}")
    plt.show()


def main():
    args = parse_args()

    # 1. Load Dataset
    if args.dataset not in ['mmlu', 'gsm8k']:
        logger.error("❌ For probability tracing, please use 'mmlu' or 'gsm8k' ONLY.")
        return

    raw_dataset = download_dataset(args.dataset)

    # Random Sample
    import random
    random.seed(args.seed)
    indices = random.sample(range(len(raw_dataset)), min(args.num_samples, len(raw_dataset)))

    # Prepare Batch {idx: (text, raw_sample)}
    # Note: Text should NOT include the answer!
    samples_batch = {}
    for idx in indices:
        raw_sample = raw_dataset[int(idx)]
        text = format_sample(args.dataset, raw_sample)
        if text:
            samples_batch[idx] = (text, raw_sample)

    # 2. Run
    for model_str in args.models:
        if '/' not in model_str: continue
        family, scale = model_str.split('/')
        process_model_aggregation(family, scale, args, samples_batch)


if __name__ == "__main__":
    main()
"""
11_layerwise_probing.py
[归因实验] Zero-shot Layer-wise Probing (Early Exit)
优化版：
1. 同时绘制 Relative Depth 和 Absolute Layer 两种图表
2. 优化绘图样式 (figsize=8x6, font=14, marker=8, legend=upper center/2col)
3. 保持 HF_ENDPOINT 环境变量防止连接超时
"""
import os
import sys

# =========================================================
# 关键设置：HF 镜像
# =========================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F
import argparse
import logging
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# 引入项目路径
sys.path.insert(0, os.getcwd())
from config import MODEL_CACHE_DIR, OUTPUT_DIR
from model_utils import load_model
from data_utils import download_dataset, sample_dataset, create_dataloader

# --- 绘图样式全局配置 (Request 2) ---
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.markersize': 8,
    'lines.linewidth': 2.5
})

# --- 彩色日志辅助 ---
def cprint(text, color='default'):
    """打印彩色 Markdown 风格日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    colors = {
        'green': '🟢', 'red': '🔴', 'yellow': '🟡', 'blue': '🔵', 'default': ''
    }
    icon = colors.get(color, '')
    print(f"[{timestamp}] {icon} {text}")

def setup_logging():
    os.makedirs(os.path.join(OUTPUT_DIR, 'probing'), exist_ok=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description='Layer-wise Probing')
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/13b'], help='Model configs')
    parser.add_argument('--variants', type=str, nargs='+', default=['base', 'sft'], help='Variants')
    parser.add_argument('--datasets', type=str, nargs='+', default=['mmlu'], help='Datasets')
    parser.add_argument('--n_samples', type=int, default=100, help='Samples to probe')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--model_dtype', type=str, default='bfloat16')
    parser.add_argument('--model_cache_dir', type=str, default=MODEL_CACHE_DIR)
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'probing'))

    args = parser.parse_args()
    args.models = sorted(args.models)
    return args

def get_final_norm(model):
    candidates = [
        ['model', 'norm'], ['model', 'final_layernorm'],
        ['base_model', 'norm'], ['norm'], ['model', 'transformer', 'ln_f']
    ]
    for path in candidates:
        module = model
        try:
            for attr in path: module = getattr(module, attr)
            return module
        except AttributeError: continue
    cprint("Could not find Final Norm layer!", 'red')
    return None

def run_early_exit_eval(model, tokenizer, dataset_name, args, device):
    dataset = download_dataset(dataset_name)
    sampled_data = sample_dataset(dataset, args.n_samples, strategy='random')
    dataloader = create_dataloader(sampled_data, dataset_name, tokenizer, args.batch_size, args.max_length)

    final_norm = get_final_norm(model)
    lm_head = model.lm_head
    num_layers = model.config.num_hidden_layers
    if final_norm is None: return None

    layer_losses = defaultdict(float)
    layer_accs = defaultdict(float)
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"    ⚡ Probing {dataset_name}", leave=False):
            input_ids = batch['input_ids'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=batch['attention_mask'].to(device), output_hidden_states=True)
            hidden_states = outputs.hidden_states

            labels = input_ids.clone()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != tokenizer.pad_token_id)
            num_valid_tokens = valid_mask.sum().item()
            if num_valid_tokens == 0: continue

            start_idx = 1 if len(hidden_states) > num_layers else 0
            for i in range(start_idx, len(hidden_states)):
                layer_idx = i - start_idx
                if layer_idx >= num_layers: break

                # Early Exit
                h = hidden_states[i]
                logits = lm_head(final_norm(h))

                # Metrics
                shift_logits = logits[..., :-1, :].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=tokenizer.pad_token_id, reduction='sum')
                preds = torch.argmax(shift_logits, dim=-1)
                acc = ((preds == shift_labels) & valid_mask).sum()

                layer_losses[layer_idx] += loss.item()
                layer_accs[layer_idx] += acc.item()
            total_tokens += num_valid_tokens

    return {
        'loss': {l: v/total_tokens for l,v in layer_losses.items()},
        'acc': {l: v/total_tokens for l,v in layer_accs.items()},
        'perplexity': {l: np.exp(v/total_tokens) for l,v in layer_losses.items()}
    }

def plot_probing_results(all_results, output_dir):
    """
    画图函数优化：
    1. 循环生成 Relative 和 Absolute 两种图
    2. 应用 figsize=(8,6), upper center legend, ncol=2
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_to_plot = ['acc', 'perplexity']

    # 定义两种绘图模式 (Request 1)
    plot_modes = [
        {'type': 'relative', 'xlabel': 'Relative Depth (0=Input, 1=Output)'},
        {'type': 'absolute', 'xlabel': 'Absolute Layer Number'}
    ]

    for metric in metrics_to_plot:
        for mode in plot_modes:
            plt.figure(figsize=(8, 6)) # (Request 2)
            has_data = False

            for key, res in all_results.items():
                if metric not in res: continue

                data_dict = res[metric]
                layers = sorted(data_dict.keys()) # [0, 1, 2, ...]
                values = [data_dict[l] for l in layers]

                # 计算 X 轴数据
                if mode['type'] == 'relative':
                    max_l = max(layers) if layers else 1
                    x_axis = [l / max_l for l in layers]
                else:
                    x_axis = layers # Absolute

                # 样式逻辑：Base虚线，SFT实线
                style = '--' if 'base' in key else '-'
                alpha = 0.7 if 'base' in key else 1.0
                marker_style = 'o' if 'base' in key else 's' # Base用圆，SFT用方块区分

                plt.plot(x_axis, values, label=key, linestyle=style,
                         marker=marker_style, alpha=alpha) # markersize由全局rcParams控制(8)
                has_data = True

            if has_data:
                title_suffix = "(Relative)" if mode['type'] == 'relative' else "(Absolute)"
                plt.title(f"Layer-wise Probing: {metric.upper()} {title_suffix}")
                plt.xlabel(mode['xlabel'])
                plt.ylabel(metric.capitalize())

                # (Request 2) Legend: Upper Center, 2 Columns
                plt.legend(loc='upper center', ncol=2, framealpha=0.9, fancybox=True)

                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()

                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                save_name = f'probing_{metric}_{mode["type"]}_{timestamp}.png'
                save_path = os.path.join(output_dir, save_name)
                plt.savefig(save_path, dpi=300)
                cprint(f"Saved plot: {save_name}", 'green')
                plt.show()
                plt.close()

def main():
    args = parse_args()
    setup_logging()

    cprint("="*40, 'blue')
    cprint(f"🚀 Start Layer-wise Probing", 'blue')
    cprint(f"📦 Models: {args.models}", 'blue')
    cprint(f"🎨 Variants: {args.variants}", 'blue')
    cprint("="*40, 'blue')

    all_results = {}

    for model_str in args.models:
        family, scale = model_str.split('/')
        for variant in args.variants:
            cprint(f"🔄 Loading {family}/{scale} - {variant}...", 'yellow')
            try:
                model, tokenizer = load_model(
                    family, scale, variant,
                    cache_dir=args.model_cache_dir,
                    dtype=args.model_dtype,
                    device_map=args.device_map
                )
                device = next(model.parameters()).device

                for dataset_name in args.datasets:
                    cprint(f"  🧪 Probing on {dataset_name}...", 'blue')
                    results = run_early_exit_eval(model, tokenizer, dataset_name, args, device)
                    if results:
                        key = f"{family}/{scale}-{variant}-{dataset_name}"
                        all_results[key] = results
                        last_layer = max(results['acc'].keys())
                        cprint(f"    ✅ Final Acc: {results['acc'][last_layer]:.4f}", 'green')
            except Exception as e:
                cprint(f"❌ Error: {e}", 'red')
            finally:
                if 'model' in locals(): del model
                if 'tokenizer' in locals(): del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

    json_path = os.path.join(args.output_dir, f'probing_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(json_path, 'w') as f: json.dump(all_results, f, indent=2)
    cprint(f"💾 Saved raw results to {json_path}", 'green')

    plot_probing_results(all_results, args.output_dir)

if __name__ == "__main__":
    main()
"""
12_layer_swapping.py
[因果实验] 模型手术 (Model Swapping) - 动态分段版
功能：
1. 支持多数据集批量运行。
2. 性能优化：使用内存快照回滚替代硬盘重载。
3. 动态层选择：根据模型总层数自动分段 (Segment)，并在段内取指定深度 (Depth)。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import copy
import re
import argparse
import logging
import json
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy.printing.pretty.pretty_symbology import line_width
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# 引入项目路径
sys.path.insert(0, os.getcwd())
# 关键：设置镜像防止超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config import MODEL_CACHE_DIR, OUTPUT_DIR
from model_utils import load_model, get_num_layers
from data_utils import download_dataset, sample_dataset, create_dataloader

# 绘图风格
plt.style.use('seaborn-v0_8-paper')

# --- 彩色日志 ---
def cprint(text, color='default'):
    t = datetime.now().strftime('%H:%M:%S')
    colors = {
        'green': '🟢', 'red': '🔴', 'yellow': '🟡', 'blue': '🔵',
        'purple': '🟣', 'white': '⚪'
    }
    icon = colors.get(color, '')
    print(f"[{t}] {icon} {text}")

def setup_logging():
    os.makedirs(os.path.join(OUTPUT_DIR, 'swapping'), exist_ok=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)

def get_module_by_name(model, module_name):
    parts = module_name.split('.')
    curr = model
    for part in parts:
        curr = getattr(curr, part)
    return curr

def copy_layer_weights(target_layer, source_layer, device):
    """直接复制层参数"""
    source_state = source_layer.state_dict()
    for name, param in source_state.items():
        target_param = dict(target_layer.named_parameters()).get(name)
        if target_param is None:
            target_param = dict(target_layer.named_buffers()).get(name)

        if target_param is not None:
            with torch.no_grad():
                target_param.copy_(param.to(device))
            del param
    del source_state
    torch.cuda.empty_cache()

# --- 核心手术逻辑 ---
def perform_surgery(base_model, sft_model, layers_to_swap, head_mode):
    """
    base_model: 在 GPU
    sft_model: 在 CPU
    """
    # 1. 替换 Transformer Layers
    if layers_to_swap:
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            base_layers = base_model.model.layers
            sft_layers = sft_model.model.layers
        else:
            base_layers = base_model.layers
            sft_layers = sft_model.layers

        device = next(base_model.parameters()).device

        for layer_idx in layers_to_swap:
            copy_layer_weights(base_layers[layer_idx], sft_layers[layer_idx], device)

    # 2. 替换 Head 和 Norm
    if head_mode == 'transferred':
        device = next(base_model.parameters()).device
        # Swap Final Norm
        norm_names = ['model.norm', 'model.final_layernorm', 'base_model.norm']
        for name in norm_names:
            try:
                base_norm = get_module_by_name(base_model, name)
                sft_norm = get_module_by_name(sft_model, name)
                copy_layer_weights(base_norm, sft_norm, device)
                break
            except AttributeError:
                continue

        # Swap LM Head
        copy_layer_weights(base_model.lm_head, sft_model.lm_head, device)
        torch.cuda.empty_cache()

def evaluate_model(model, dataloader, device, tokenizer):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != tokenizer.pad_token_id)
            correct = (preds == shift_labels) & mask

            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            del input_ids, attn_mask, labels, outputs, logits

    avg_loss = total_loss / total_tokens
    ppl = np.exp(avg_loss)
    acc = total_correct / total_tokens
    return ppl, acc

def plot_results(results, base_ref, sft_ref, output_dir, model_name, dataset_name):
    """
    画图增强：加入 Base 和 SFT 的参考线
    """
    os.makedirs(output_dir, exist_ok=True)
    # 这里的 keys 已经是 "Seg 1", "Seg 2" 这样的易读标签了
    labels = list(results.keys())
    if not labels: return

    head_modes = list(results[labels[0]].keys())
    metrics = ['acc', 'ppl']

    for metric in metrics:
        plt.figure(figsize=(6, 4), dpi=300)

        # 1. 画 Base 参考线
        if base_ref:
            val = base_ref[metric]
            plt.axhline(y=val, color='gray', linestyle='--',  label=f'Base: {val:.4f}', alpha=0.7)

        # 2. 画 SFT 参考线
        if sft_ref:
            val = sft_ref[metric]
            plt.axhline(y=val, color='red', linestyle='--', label=f'SFT: {val:.4f}', alpha=0.7)

        # 3. 画实验曲线
        for mode in head_modes:
            y_values = [results[lbl][mode][metric] for lbl in labels]
            marker = 'o' if mode == 'original' else 's'
            plt.plot(labels, y_values, label=f"Swap + {mode} Head", marker=marker)

        plt.title(f"Model Swapping: {metric.upper()} ({model_name} on {dataset_name})")
        plt.xlabel("Model Segments (Depth)")
        plt.ylabel("Perplexity" if metric == 'ppl' else "Accuracy")
        if metric == 'ppl': plt.yscale('log')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        timestamp = datetime.now().strftime('%H%M')
        save_path = os.path.join(output_dir, f'swap_{metric}_{dataset_name}_{timestamp}.pdf')
        plt.savefig(save_path, dpi=300)
        plt.show()
        cprint(f"📸 Saved plot: {save_path}", 'green')
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/7b'], help='Models')
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen'],
                        help='Datasets')

    # --- 修改部分：替换原有的 swap_ranges，改为动态分段参数 ---
    parser.add_argument('--num_segments', type=int, default=5,
                        help='Divide the model into N segments (e.g. 5 means 20% chunks).')
    parser.add_argument('--swap_depth', type=int, default=2,
                        help='Number of layers to swap within each segment (from the end of that segment).')
    # ---------------------------------------------------------

    parser.add_argument('--lm_heads', type=str, nargs='+', default=['original', 'transferred'])
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model_cache_dir', type=str, default=MODEL_CACHE_DIR)
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'swapping'))
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()

    cprint("="*40, 'blue')
    cprint(f"🚀 Start Model Swapping (Dynamic Segments)", 'blue')
    cprint(f"📦 Models: {args.models}", 'blue')
    cprint(f"🔢 Segments: {args.num_segments} | Depth per seg: {args.swap_depth}", 'blue')
    cprint("="*40, 'blue')

    for model_str in args.models:
        family, scale = model_str.split('/')
        num_layers = get_num_layers(family, scale)
        cprint(f"\nModel {model_str} has {num_layers} layers.", 'purple')

        # --- 动态生成层配置 (Dynamic Layer Configuration) ---
        # 逻辑：将模型切分为 num_segments 份，每份取最后 swap_depth 层
        experiment_configs = [] # List of (Label, LayerIndices)

        # 1. 总是包含一个 "No Swap" 基准点
        experiment_configs.append(("No Swap", []))

        segment_size = num_layers / args.num_segments

        for i in range(args.num_segments):
            # 计算当前分段的起始和结束 (Float -> Int)
            seg_start_idx = int(i * segment_size)
            seg_end_idx = int((i + 1) * segment_size)

            # 确保不越界
            seg_end_idx = min(seg_end_idx, num_layers)

            # 在当前分段内，取最后 swap_depth 层
            # 例如分段是 [0, 6)，depth=2，则取 [4, 5]
            swap_start = max(seg_start_idx, seg_end_idx - args.swap_depth)

            # 生成具体的层索引列表
            layers_indices = list(range(swap_start, seg_end_idx))

            # 生成可读标签
            label = f"Seg {i+1} ({len(layers_indices)}L)"
            experiment_configs.append((label, layers_indices))

            # cprint(f"  Plan: {label} -> Layers {layers_indices}", 'white')

        # --- 阶段 1: 加载模型 (只需一次) ---
        cprint(f"\n[Init] Loading SFT Model (Donor) to CPU...", 'yellow')
        sft_model, tokenizer = load_model(family, scale, 'sft', device_map='cpu', dtype='float16')
        sft_model.eval()

        cprint(f"[Init] Loading Base Model (Patient) to GPU...", 'yellow')
        base_model, _ = load_model(family, scale, 'base', device_map='cuda', dtype='float16')
        base_model.eval()
        device = next(base_model.parameters()).device

        # --- 性能优化关键: 内存快照 ---
        cprint(f"💾 Creating Base model snapshot in RAM...", 'white')
        base_snapshot = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        # --- 阶段 2: 遍历数据集 ---
        for dataset_name in args.dataset:
            cprint(f"\n📂 Processing Dataset: {dataset_name}", 'blue')

            # 准备数据
            dataset = download_dataset(dataset_name)
            sampled = sample_dataset(dataset, args.n_samples)
            dataloader = create_dataloader(sampled, dataset_name, tokenizer, args.batch_size, 512)

            # 2.1 评估纯 Base (Reference)
            cprint("  ⚖️  Evaluating Base Reference...", 'white')
            base_model.load_state_dict(base_snapshot, strict=False)
            base_ppl, base_acc = evaluate_model(base_model, dataloader, device, tokenizer)
            base_results = {'ppl': base_ppl, 'acc': base_acc}

            # 2.2 评估纯 SFT (Reference)
            cprint("  ⚖️  Evaluating SFT Reference...", 'white')
            base_model.load_state_dict(sft_model.state_dict(), strict=False)
            sft_ppl, sft_acc = evaluate_model(base_model, dataloader, device, tokenizer)
            sft_results = {'ppl': sft_ppl, 'acc': sft_acc}

            # 恢复 Base 准备开始实验
            base_model.load_state_dict(base_snapshot, strict=False)

            # 结果容器
            model_results = defaultdict(lambda: defaultdict(dict))

            # --- 阶段 3: 遍历手术方案 (Modified Loop) ---
            total_exps = len(experiment_configs) * len(args.lm_heads)
            pbar = tqdm(total=total_exps, desc="Running Swaps")

            # 使用生成的 experiment_configs 替代原来的 range_str
            for label, layers_indices in experiment_configs:
                for head_mode in args.lm_heads:
                    # 1. 快速重置
                    base_model.load_state_dict(base_snapshot, strict=False)

                    # 2. 执行手术
                    perform_surgery(base_model, sft_model, layers_indices, head_mode)

                    # 3. 评估
                    ppl, acc = evaluate_model(base_model, dataloader, device, tokenizer)

                    # 这里使用 Label 作为 Key，绘图时会自动显示为 "Seg 1", "Seg 2" 等
                    model_results[label][head_mode] = {'ppl': ppl, 'acc': acc}
                    pbar.update(1)

            pbar.close()

            # --- 阶段 4: 保存与画图 ---
            json_path = os.path.join(args.output_dir, f'swap_{dataset_name}_{family}.json')
            save_data = {
                'base_ref': base_results,
                'sft_ref': sft_results,
                'experiments': json.loads(json.dumps(model_results))
            }
            with open(json_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            plot_results(model_results, base_results, sft_results, args.output_dir, model_str, dataset_name)

        del sft_model, base_model, base_snapshot
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
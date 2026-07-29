"""
13_gradient_flow_analysis.py
E5: Gradient Flow Analysis - 梯度流与表征变化关联分析

Fixed:
1. Support for float epochs (e.g., 0.5)
2. Enhanced plotting (DPI 300, larger fonts, thicker lines)
3. Argument parsing types
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
import argparse
import logging
import json
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import scipy.stats

sys.path.insert(0, os.getcwd())
from config import MODEL_CACHE_DIR, OUTPUT_DIR, METRIC_DIR
from model_utils import load_model
from data_utils import download_dataset, sample_dataset, create_dataloader

# 设定绘图风格
plt.style.use('seaborn-v0_8-paper')
# 解决中文显示问题（如果环境支持）
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 彩色日志
# ============================================================================
def cprint(text, color='default'):
    t = datetime.now().strftime('%H:%M:%S')
    colors = {'green': '🟢', 'red': '🔴', 'yellow': '🟡', 'blue': '🔵', 'purple': '🟣'}
    icon = colors.get(color, '')
    print(f"[{t}] {icon} {text}")

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"gradient_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.getLogger("transformers").setLevel(logging.ERROR)
    return logging.getLogger(__name__)

# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='E5: Gradient Flow Analysis')

    # 列表参数使用 nargs='+'
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/7b'],
                        help='Model configs in format "family/scale"')
    parser.add_argument('--datasets', type=str, nargs='+', default=['mmlu'],
                        choices=['mmlu', 'gsm8kgradient'],
                        help='Datasets for SFT training')

    # Training config
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')

    # FIX: 修改为 float 以支持 0.5 epoch
    parser.add_argument('--num_epochs', type=float, default=0.01,
                        help='Number of training epochs (can be float, e.g. 0.5)')

    parser.add_argument('--max_length', type=int, default=512,
                        help='Max sequence length')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Warmup steps')

    # System
    parser.add_argument('--model_dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'gradient_flow'))

    # Analysis
    parser.add_argument('--representation_metrics_file', type=str, default=None,
                        help='Path to your representation metrics CSV')
    parser.add_argument('--save_frequency', type=int, default=1000,
                        help='Save gradient statistics every N steps')

    return parser.parse_args()

# ============================================================================
# 梯度记录器 (保持不变)
# ============================================================================
class GradientRecorder:
    def __init__(self, model):
        self.model = model
        self.gradient_norms = defaultdict(list)
        self.param_names = []

        self.initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone().cpu()
                self.param_names.append(name)

    def record_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.gradient_norms[name].append(grad_norm)

    def compute_statistics(self):
        stats = {}
        for name in self.param_names:
            if name in self.gradient_norms and len(self.gradient_norms[name]) > 0:
                norms = self.gradient_norms[name]
                stats[name] = {
                    'mean': np.mean(norms),
                    'std': np.std(norms),
                    'max': np.max(norms),
                    'min': np.min(norms)
                }
        return stats

    def compute_param_changes(self):
        changes = {}
        for name, param in self.model.named_parameters():
            if name in self.initial_params:
                delta = param.data.cpu() - self.initial_params[name]
                changes[name] = delta.norm().item()
        return changes

# ============================================================================
# 训练函数 (已修复 Float Epoch 问题)
# ============================================================================
def train_with_gradient_tracking(model, tokenizer, dataloader, args, logger):
    device = next(model.parameters()).device
    recorder = GradientRecorder(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 计算总步数 (支持 float epoch)
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_steps = int(steps_per_epoch * args.num_epochs)

    # 向上取整计算需要遍历的 epoch 次数
    epochs_to_run = math.ceil(args.num_epochs)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()
    global_step = 0
    stop_training = False

    cprint(f"Starting training: {total_steps} optimization steps ({args.num_epochs} epochs)", 'blue')

    for epoch in range(epochs_to_run):
        if stop_training: break

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs_to_run}")

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                recorder.record_gradients()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                pbar.set_postfix({'loss': loss.item() * args.gradient_accumulation_steps})

                # Check stop condition
                if global_step >= total_steps:
                    stop_training = True
                    break

                if global_step % args.save_frequency == 0:
                    cprint(f"Step {global_step}: Saving checkpoint...", 'yellow')

        avg_loss = epoch_loss / (len(dataloader) if len(dataloader) > 0 else 1)
        cprint(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}", 'green')

    return recorder

# ============================================================================
# 层级聚合
# ============================================================================
def aggregate_to_layers(param_stats, model_family):
    layer_stats = defaultdict(lambda: {'grad_norm': [], 'param_change': []})

    for param_name, stats in param_stats.items():
        if 'layers.' in param_name:
            try:
                # 尝试解析层号，适配不同模型命名 (e.g., model.layers.0)
                parts = param_name.split('.')
                layer_idx = -1
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i+1])
                        break

                if layer_idx != -1:
                    if 'grad_norm' in stats:
                        layer_stats[layer_idx]['grad_norm'].append(stats['grad_norm'])
                    if 'param_change' in stats:
                        layer_stats[layer_idx]['param_change'].append(stats['param_change'])
            except:
                continue

    aggregated = {}
    for layer_num in sorted(layer_stats.keys()):
        aggregated[layer_num] = {
            'mean_grad_norm': np.mean(layer_stats[layer_num]['grad_norm']) if layer_stats[layer_num]['grad_norm'] else 0,
            'mean_param_change': np.mean(layer_stats[layer_num]['param_change']) if layer_stats[layer_num]['param_change'] else 0
        }
    return aggregated

# ============================================================================
# 相关性分析
# ============================================================================
def correlation_analysis(gradient_data, representation_metrics_file, output_dir):

    if representation_metrics_file is None or not os.path.exists(representation_metrics_file):
        return None

    repr_df = pd.read_csv(representation_metrics_file)
    grad_df = pd.DataFrame([
        {'layer': k, 'grad_norm': v['mean_grad_norm'], 'param_change': v['mean_param_change']}
        for k, v in gradient_data.items()
    ])

    merged = pd.merge(grad_df, repr_df, on='layer', how='inner')
    if len(merged) < 3: return None

    correlations = {}
    metrics_to_correlate = [col for col in repr_df.columns if col != 'layer']

    for metric in metrics_to_correlate:
        if metric in merged.columns:
            corr_grad, p_grad = scipy.stats.pearsonr(merged['grad_norm'], merged[metric])
            corr_change, p_change = scipy.stats.pearsonr(merged['param_change'], merged[metric])
            correlations[metric] = {
                'grad_norm_corr': corr_grad,
                'grad_norm_p': p_grad,
                'param_change_corr': corr_change,
                'param_change_p': p_change
            }

    corr_df = pd.DataFrame(correlations).T
    corr_df.to_csv(os.path.join(output_dir, 'gradient_representation_correlation.csv'))

    # 优化相关性热力图
    plt.figure(figsize=(6, 4), dpi=300)
    sns.heatmap(corr_df[['grad_norm_corr', 'param_change_corr']],
                annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot_kws={"size": 14})
    plt.title('Correlation: Gradient vs Representation Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.show()
    plt.close()

    return correlations

# ============================================================================
# 可视化 (已优化：字体、线宽、DPI)
# ============================================================================
def visualize_gradients(layer_data, output_dir, model_name):
    layers = sorted(layer_data.keys())
    if not layers:
        cprint(f"No layer data found for {model_name}, skipping plot.", 'red')
        return

    grad_norms = [layer_data[l]['mean_grad_norm'] for l in layers]
    param_changes = [layer_data[l]['mean_param_change'] for l in layers]

    # 增大画布，提高DPI
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

    # 样式优化：加粗线条，增大点的大小
    line1 = ax1.plot(layers, grad_norms, 'b-o', label='Gradient Norm', linewidth=3, markersize=8, alpha=0.8)
    ax1.set_xlabel('Layer Index', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Mean Gradient Norm', color='b', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b', labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    line2 = ax2.plot(layers, param_changes, 'r-s', label='Param Change', linewidth=3, markersize=8, alpha=0.8)
    ax2.set_ylabel('Parameter Change (L2)', color='r', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r', labelsize=14)

    plt.title(f'Gradient Flow Analysis: {model_name}', fontsize=18, fontweight='bold', pad=20)

    # 合并图例并调整位置和字体
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=14, frameon=True, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'gradient_flow_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # 关闭画布，防止内存泄漏

    cprint(f"Saved optimized gradient flow plot to {save_path}", 'green')

# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    output_dir = args.output_dir
    logger = setup_logging(output_dir)

    cprint("="*60, 'blue')
    cprint("E5: Gradient Flow Analysis (Optimized)", 'blue')
    cprint(f"Models: {args.models}", 'blue')
    cprint(f"Datasets: {args.datasets}", 'blue')
    cprint(f"Epochs: {args.num_epochs}", 'blue')
    cprint("="*60, 'blue')

    all_results = {}

    for model_str in args.models:
        family, scale = model_str.split('/')
        cprint(f"\nProcessing: {family}/{scale}", 'purple')

        try:
            # 1. Base Model
            cprint("Loading Base model ...", 'yellow')
            model, tokenizer = load_model(
                family, scale, 'base',
                cache_dir=MODEL_CACHE_DIR,
                device_map=args.device_map,
                dtype=args.model_dtype
            )

            # 2. Data & Training
            for dataset_name in args.datasets:
                cprint(f"Dataset: {dataset_name}", 'yellow')
                dataset = download_dataset(dataset_name)
                sampled = sample_dataset(dataset, args.n_samples, strategy='random')
                dataloader = create_dataloader(
                    sampled, dataset_name, tokenizer,
                    args.batch_size, args.max_length
                )

                # 3. Train
                recorder = train_with_gradient_tracking(model, tokenizer, dataloader, args, logger)

                # 4. Stats
                grad_stats = recorder.compute_statistics()
                param_changes = recorder.compute_param_changes()

                combined = {}
                for name in recorder.param_names:
                    combined[name] = {
                        'grad_norm': grad_stats.get(name, {}).get('mean', 0),
                        'param_change': param_changes.get(name, 0)
                    }

                layer_data = aggregate_to_layers(combined, family)

                # 5. Save & Plot
                model_id = f"{family}_{scale}_{dataset_name}"
                result_dir = os.path.join(output_dir, model_id)
                os.makedirs(result_dir, exist_ok=True)

                pd.DataFrame([
                    {'layer': k, **v} for k, v in layer_data.items()
                ]).to_csv(os.path.join(result_dir, 'stats.csv'), index=False)

                visualize_gradients(layer_data, result_dir, model_id)

                # 8. 相关性分析
                # 自动推断 metrics 文件路径
                metrics_file = args.representation_metrics_file
                if metrics_file is None:
                    # 默认寻找 Base 模型的静态指标 (格式对齐 01_compute_and_visualize_metrics.py)
                    auto_filename = f"metrics_{family}-{scale}_FULL.csv"
                    auto_path = os.path.join(METRIC_DIR, 'csv_reports', auto_filename)

                    if os.path.exists(auto_path):
                        metrics_file = auto_path
                        cprint(f"Found auto metrics file: {metrics_file}", 'green')
                    else:
                        # 构造提示命令
                        cmd_hint = f"python compute_and_visualize_metrics.py --models {family}/{scale} --dataset {dataset_name}"
                        cprint(f"⚠️  Metrics file not found at {auto_path}", 'yellow')
                        cprint(f"   To generate it, run: {cmd_hint}", 'yellow')

                if metrics_file:
                    correlation_analysis(layer_data, metrics_file, result_dir)

                all_results[model_id] = {'num_layers': len(layer_data)}

        except Exception as e:
            cprint(f"❌ Error: {e}", 'red')
            logger.error(f"Failed: {e}", exc_info=True)
        finally:
            if 'model' in locals(): del model
            gc.collect()
            torch.cuda.empty_cache()

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == '__main__':
    main()
import os
import argparse
import logging
import torch
import hashlib
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import sys
sys.path.insert(0, os.getcwd())

from config import REPRESENTATION_DIR, METRIC_DIR
from metric_utils import (
    compute_prompt_entropy, compute_dataset_entropy,
    compute_curvature, compute_effective_rank, compute_l2_norm,
    compute_spectral_metrics, compute_sparsity,
    compute_cka, compute_cosine_similarity, compute_mean_shift
)

plt.style.use('seaborn-v0_8-paper')
# sns.set_palette("tab10") # 将在画图函数中动态设置

def setup_logging():
    os.makedirs(METRIC_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Compute metrics for Base/SFT/DPO/RLVR/Instruct')

    parser.add_argument('--models', type=str, nargs='+',
                        default=['olmo2/7b', 'olmo2/13b', 'olmo2/32b'],
                        choices=['mistral/7b', 'olmo2/1b', 'olmo2/7b', 'olmo2/13b', 'olmo2/32b'],
                        help='Model configs in format "family/scale"')
    parser.add_argument('--dataset', type=str, default='toxigen',
                        choices=['mmlu', 'gsm8k', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen'],
                        help='Dataset name')

    parser.add_argument('--variants', type=str, nargs='+',
                        default=['sft'],
                        choices=['sft', 'dpo', 'rlvr', 'instruct'],
                        help='Variants to compare against Base')

    ALL_METRICS = ['prompt_entropy', 'dataset_entropy', 'curvature',
                   'effective_rank', 'l2_norm', 'spectral_metrics',
                   'sparsity', 'cka', 'cosine_similarity', 'mean_shift']

    parser.add_argument('--metrics', type=str, nargs='+', default=['all'],
                        choices=['all'] + ALL_METRICS,
                        help='List of metrics to compute')
    parser.add_argument('--normalization', type=str, nargs='+',
                        default=['maxEntropy'],
                        help='Normalization schemes for entropy metrics')

    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of files to load per batch for streaming')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max pooled samples to load for global metrics')
    parser.add_argument('--force_recompute', action='store_true')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    if 'all' in args.metrics:
        args.metrics = sorted(ALL_METRICS)
    else:
        args.metrics = sorted(args.metrics)
    args.models = sorted(args.models)
    return args

# ============================================================================
# NEW: Model-Specific Cache Logic
# ============================================================================
def get_model_specific_cache_path(args, model_str):
    """Generate deterministic cache filename specific to ONE model"""
    config_dict = {
        'dataset': args.dataset,
        'model': model_str, # Use specific model, NOT args.models list
        'variants': args.variants,
        'metrics': args.metrics,
        'normalization': args.normalization,
        'max_samples': args.max_samples
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    # Filename includes model name for clarity
    safe_model_name = model_str.replace('/', '_')
    filename = f"cache_{args.dataset}_{safe_model_name}.pt"
    return os.path.join(METRIC_DIR, filename)

def load_single_file(path):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        logging.error(f"❌ Error loading {path}: {e}")
        return None

# ============================================================================
# Computation Functions (Unchanged logic, just wrappers)
# ============================================================================

def compute_sequence_metrics_streaming(model_family, scale, variant, dataset, metrics_list, normalizations, device, base_dir=REPRESENTATION_DIR, batch_size=10):
    # ... (Same as before, abridged for brevity) ...
    needed_metrics = set(['prompt_entropy', 'curvature', 'sparsity']) & set(metrics_list)
    if not needed_metrics: return {}

    logging.info(f"    🌊 [Stream] Computing sequence metrics ({variant})...")
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))
    if not files: return {}

    results_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    num_batches = (len(files) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    🌊 Processing {variant}", leave=False):
        batch_files = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded_samples = list(executor.map(load_single_file, batch_files))

        batch_hidden = defaultdict(list)
        has_hidden = False
        for s in loaded_samples:
            if s and s.get('hidden_states'):
                has_hidden = True
                for layer, tensor in s['hidden_states'].items():
                    batch_hidden[layer].append(tensor)

        if not has_hidden: continue

        hidden_batch_stacked = {}
        for k, v in batch_hidden.items():
            try:
                t = torch.stack(v).float()
                if device == 'cuda': t = t.to(device)
                hidden_batch_stacked[k] = t
            except: pass

        batch_results = {}
        if 'prompt_entropy' in needed_metrics:
            batch_results['prompt_entropy'] = compute_prompt_entropy(hidden_batch_stacked, normalizations=normalizations)
        if 'curvature' in needed_metrics:
            batch_results['curvature'] = compute_curvature(hidden_batch_stacked)
        if 'sparsity' in needed_metrics:
            batch_results['sparsity'] = compute_sparsity(hidden_batch_stacked)

        for metric, layers_data in batch_results.items():
            if isinstance(layers_data, dict):
                for sub_key, vals in layers_data.items():
                    for layer_i, val in enumerate(vals):
                        results_agg[metric][sub_key][layer_i].append(val)
            elif isinstance(layers_data, list):
                for layer_i, val in enumerate(layers_data):
                    results_agg[metric]['raw'][layer_i].append(val)

        del hidden_batch_stacked, batch_hidden, loaded_samples
        if device == 'cuda': torch.cuda.empty_cache()

    final_results = {}
    for metric, sub_dict in results_agg.items():
        final_results[metric] = {}
        for sub_key, layers_dict in sub_dict.items():
            sorted_layers = sorted(layers_dict.keys())
            if not sorted_layers: continue
            averaged_vals = []
            max_layer = max(sorted_layers)
            for i in range(max_layer + 1):
                vals = layers_dict.get(i, [])
                averaged_vals.append(sum(vals) / len(vals) if vals else 0.0)
            final_results[metric][sub_key] = averaged_vals
    return final_results

def load_pooled_subset(model_family, scale, variant, dataset, max_samples, base_dir=REPRESENTATION_DIR, batch_size=50):
    logging.info(f"    📥 [Load] Loading pooled subset ({variant}, max={max_samples})...")
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))
    if not files: return None

    import random
    random.seed(42)
    random.shuffle(files)

    pooled_agg = defaultdict(list)
    current_count = 0
    for i in range(0, len(files), batch_size):
        if current_count >= max_samples: break
        batch_files = files[i : i+batch_size]
        with ThreadPoolExecutor(max_workers=4) as executor:
            samples = list(executor.map(load_single_file, batch_files))
        for s in samples:
            if not s: continue
            if current_count >= max_samples: break
            for layer, tensor in s.get('pooled_states', {}).items():
                pooled_agg[layer].append(tensor.float().cpu())
            current_count += 1

    final_pooled = {}
    for k, v in pooled_agg.items():
        final_pooled[k] = torch.stack(v)
    return final_pooled

def compute_pooled_metrics_in_memory(pooled_states, metrics_list, normalizations, device):
    results = {}
    needed = set(['dataset_entropy', 'effective_rank', 'l2_norm', 'spectral_metrics']) & set(metrics_list)
    if not pooled_states or not needed: return results

    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if 'dataset_entropy' in metrics_list:
        res = {norm: [] for norm in normalizations}
        for layer in sorted(pooled_states.keys()):
            tensor = pooled_states[layer]
            if use_gpu: tensor = tensor.to(device)
            layer_res = compute_dataset_entropy({layer: tensor}, normalizations=normalizations)
            for norm in normalizations:
                if layer_res[norm]: res[norm].append(layer_res[norm][0])
            if use_gpu: del tensor; torch.cuda.empty_cache()
        results['dataset_entropy'] = res

    for metric, func in [('effective_rank', compute_effective_rank), ('l2_norm', compute_l2_norm), ('spectral_metrics', compute_spectral_metrics)]:
        if metric in metrics_list:
            metrics_out = defaultdict(list)
            for layer in sorted(pooled_states.keys()):
                tensor = pooled_states[layer]
                if use_gpu: tensor = tensor.to(device)
                val = func({layer: tensor})
                if isinstance(val, dict):
                    for k, v in val.items(): metrics_out[k].append(v[0])
                elif isinstance(val, list):
                    metrics_out['raw'].append(val[0])
                if use_gpu: del tensor; torch.cuda.empty_cache()
            if isinstance(val, dict): results[metric] = dict(metrics_out)
            else: results[metric] = metrics_out.get('raw', [])
    return results

def compute_alignment_metrics_safe(base_pooled, target_pooled, metrics_list, device):
    results = {}
    needed = set(['cka', 'cosine_similarity', 'mean_shift']) & set(metrics_list)
    if not base_pooled or not target_pooled or not needed: return results

    common_layers = sorted(list(set(base_pooled.keys()) & set(target_pooled.keys())))
    if not common_layers: return results
    min_len = min(base_pooled[common_layers[0]].shape[0], target_pooled[common_layers[0]].shape[0])
    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if 'cka' in metrics_list:
        res = []
        for l in common_layers:
            b, t = base_pooled[l][:min_len], target_pooled[l][:min_len]
            if use_gpu: b, t = b.to(device), t.to(device)
            res.append(compute_cka({l: b}, {l: t})[0])
            if use_gpu: del b, t; torch.cuda.empty_cache()
        results['cka'] = res

    if 'cosine_similarity' in metrics_list:
        base_aligned = {l: base_pooled[l][:min_len] for l in common_layers}
        target_aligned = {l: target_pooled[l][:min_len] for l in common_layers}
        if use_gpu:
            base_gpu = {k: v.to(device) for k, v in base_aligned.items()}
            target_gpu = {k: v.to(device) for k, v in target_aligned.items()}
            results['cosine_similarity'] = compute_cosine_similarity(base_gpu, target_gpu)
            del base_gpu, target_gpu; torch.cuda.empty_cache()
        else:
            results['cosine_similarity'] = compute_cosine_similarity(base_aligned, target_aligned)

    if 'mean_shift' in metrics_list:
        base_aligned = {l: base_pooled[l][:min_len] for l in common_layers}
        target_aligned = {l: target_pooled[l][:min_len] for l in common_layers}
        results['mean_shift'] = compute_mean_shift(base_aligned, target_aligned)

    return results

# ============================================================================
# NEW: Improved Visualization
# ============================================================================

def visualize_results(results_store, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 预先计算颜色映射，确保同一模型架构(Family-Scale)颜色一致
    all_keys = set()
    for m_data in results_store.values():
        all_keys.update(m_data.keys())

    # 提取唯一前缀 (例如: olmo2-1b, mistral-7b)
    # 假设 key 格式为: family-scale-variant
    unique_prefixes = sorted(list(set(["-".join(k.split('-')[:2]) for k in all_keys])))

    # 使用 tab10 调色板
    palette = sns.color_palette("tab10", n_colors=len(unique_prefixes))
    color_map = {prefix: palette[i] for i, prefix in enumerate(unique_prefixes)}

    def format_legend_label(label):
        """将 olmo2-1b-base 转换为 OLMo2-1B-Base"""
        parts = label.split('-')
        new_parts = []
        for p in parts:
            if p.lower() == 'olmo2': new_parts.append('OLMo2')
            elif p.lower() == 'mistral': new_parts.append('Mistral')
            elif p.lower() == 'sft': new_parts.append('SFT')
            elif p.lower() == 'base': new_parts.append('Base')
            elif p.lower() == 'vs': new_parts.append('vs')
            elif p.endswith('b') and p[:-1].isdigit(): new_parts.append(p.upper()) # 1b -> 1B
            else: new_parts.append(p.capitalize())
        return "-".join(new_parts)

    for metric_name, model_data in results_store.items():
        if not model_data: continue

        # 检查子键 (normalization 等)
        first_val = next(iter(model_data.values()))
        if isinstance(first_val, dict):
            sub_keys = sorted(list(first_val.keys()))
        else:
            sub_keys = [None]

        for sub_key in sub_keys:
            # 修改 1: figsize 调大
            plt.figure(figsize=(8, 6), dpi=800)
            has_data = False

            # 对 keys 排序，确保图例顺序一致
            sorted_keys = sorted(model_data.keys())

            for model_label in sorted_keys:
                values = model_data[model_label]
                y_vals = values[sub_key] if sub_key else values
                if not y_vals: continue

                # 获取颜色 (基于 family-scale)
                parts = model_label.split('-')
                prefix = f"{parts[0]}-{parts[1]}"
                color = color_map.get(prefix, 'black')

                # 修改 2: 样式逻辑 (Base虚线星号, 其他实线圆点)
                if 'base' in model_label and 'vs' not in model_label:
                    style = '--'
                    marker = 'o'
                    # Base 透明度稍高一点点，或者保持不透明
                    alpha = 0.7
                else:
                    style = '-'
                    marker = '*'
                    alpha = 0.9

                # 修改 3: 线宽和点大小
                x_vals = range(1, len(y_vals))
                label_str = format_legend_label(model_label)

                plt.plot(x_vals, y_vals[1:],
                         linestyle=style,
                         linewidth=4,
                         marker=marker,
                         markersize=8,
                         color=color,
                         label=label_str,
                         alpha=alpha)
                has_data = True

            if has_data:
                # 修改 4: 移除 Title, Y轴显示 Metric Name, 字体变大；自定义 Y 轴标签格式化逻辑
                if metric_name.lower() == 'cka':
                    clean_metric_name = "CKA"
                elif metric_name == 'spectral_metrics' and sub_key:
                    # 对于谱分析，直接显示子项名称 (如 Condition Number, Rank Deficiency)
                    clean_metric_name = sub_key.replace("_", " ").title()
                else:
                    # 对于 Entropy, Effective Rank 等，忽略 sub_key，只显示主指标名
                    clean_metric_name = metric_name.replace("_", " ").title()

                plt.xlabel("Layer Depth", fontsize=18)
                plt.ylabel(clean_metric_name, fontsize=18)

                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)

                plt.legend(fontsize=18, loc='best')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()

                clean_name = f"{metric_name}_{sub_key}" if sub_key else metric_name
                clean_name = clean_name.replace("/", "_").replace(" ", "_")

                save_path = os.path.join(output_dir, f"{clean_name}.pdf")
                plt.savefig(save_path, dpi=800)
                logging.info(f"    📸 [Plot] Saved: {clean_name}.pdf")
                plt.show()
                plt.close()

# ============================================================================
# Main Loop (Updated for Split Caching)
# ============================================================================

def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 20)
    logger.info(f"🚀 [Start] Metric Computation")
    logger.info(f"📍 Models: {args.models}")
    logger.info(f"🎨 Variants: {args.variants}")
    logger.info("=" * 20)

    # 这里的 visualization_store 用于收集所有模型的数据，用于最后统一画图
    global_visualization_store = defaultdict(dict)

    for model_str in args.models:
        family, scale = model_str.split('/')
        logger.info(f"{'='*40}📦 [Model] Processing {family}/{scale}{'='*40}")

        # 1. 检查是否存在该模型的独立缓存
        model_cache_path = get_model_specific_cache_path(args, model_str)
        model_results = None

        if os.path.exists(model_cache_path) and not args.force_recompute:
            logger.info(f"✨ [Cache] Loading found cache for {model_str}: {model_cache_path}")
            model_results = torch.load(model_cache_path)

        # 2. 如果没有缓存，开始计算
        if model_results is None:
            model_results = defaultdict(dict) # Store results for THIS model only

            # --- 2.1 BASE Model ---
            logger.info("  🛡️ [1/3] Processing BASE (Reference)...")
            base_seq = compute_sequence_metrics_streaming(
                family, scale, 'base', args.dataset, args.metrics, args.normalization, args.device, batch_size=args.batch_size
            )
            for m, vals in base_seq.items():
                model_results[m][f"{family}-{scale}-base"] = vals

            base_pooled = load_pooled_subset(
                family, scale, 'base', args.dataset, max_samples=args.max_samples, batch_size=args.batch_size
            )

            if base_pooled:
                base_global = compute_pooled_metrics_in_memory(base_pooled, args.metrics, args.normalization, args.device)
                for m, vals in base_global.items():
                    model_results[m][f"{family}-{scale}-base"] = vals
            else:
                logger.warning("    ⚠️ Base model data missing!")

            # --- 2.2 Variants ---
            for var in args.variants:
                logger.info(f"🎨 [2/3] Processing Variant: {var}...")

                var_seq = compute_sequence_metrics_streaming(
                    family, scale, var, args.dataset, args.metrics, args.normalization, args.device, batch_size=args.batch_size
                )
                for m, vals in var_seq.items():
                    model_results[m][f"{family}-{scale}-{var}"] = vals

                var_pooled = load_pooled_subset(
                    family, scale, var, args.dataset, max_samples=args.max_samples, batch_size=args.batch_size
                )

                if var_pooled:
                    var_global = compute_pooled_metrics_in_memory(var_pooled, args.metrics, args.normalization, args.device)
                    for m, vals in var_global.items():
                        model_results[m][f"{family}-{scale}-{var}"] = vals

                    if base_pooled:
                        logger.info(f"    📐 [Align] Computing Alignment: Base vs {var}")
                        align = compute_alignment_metrics_safe(base_pooled, var_pooled, args.metrics, args.device)
                        for m, vals in align.items():
                            model_results[m][f"{family}-{scale}"] = vals

                del var_pooled; gc.collect()
                if args.device == 'cuda': torch.cuda.empty_cache()

            del base_pooled; gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()

            # --- 2.3 保存该模型的缓存 ---
            logger.info(f"💾 [Save] Saving cache for {model_str} to {model_cache_path}")
            os.makedirs(os.path.dirname(model_cache_path), exist_ok=True)  # 强制创建父目录，防止报错
            torch.save(dict(model_results), model_cache_path)

        # 3. 将该模型的结果合并到全局 Store
        for metric, m_dict in model_results.items():
            global_visualization_store[metric].update(m_dict)

    # 4. 可视化 (使用合并后的数据)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # viz_dir = os.path.join(METRIC_DIR, 'plots', args.dataset, timestamp)
    viz_dir = os.path.join(METRIC_DIR, 'plots', args.dataset)
    logger.info(f"📊 [Plot] Generating plots to {viz_dir}...")
    visualize_results(global_visualization_store, viz_dir)

    # 5. CSV Output (Consolidated)
    csv_output_dir = os.path.join(METRIC_DIR, 'csv_reports')
    os.makedirs(csv_output_dir, exist_ok=True)

    # Generate CSV per model scale
    for model_str in args.models:
        family, scale = model_str.split('/')
        prefix = f"{family}-{scale}"
        layer_data = defaultdict(dict)

        for metric_name, models_data in global_visualization_store.items():
            for model_key, values in models_data.items():
                if not model_key.startswith(prefix): continue
                suffix = model_key[len(prefix) + 1:]
                col_name = f"{metric_name}_{suffix}"

                final_values = []
                if isinstance(values, list): final_values = values
                elif isinstance(values, dict):
                    # Prefer maxEntropy or raw
                    for norm_key in ['maxEntropy', 'raw', 'normalized']:
                        if norm_key in values:
                            final_values = values[norm_key]
                            break
                    if not final_values and values:
                        final_values = list(values.values())[0]

                for layer_idx, val in enumerate(final_values):
                    layer_data[layer_idx][col_name] = val

        if layer_data:
            df = pd.DataFrame.from_dict(layer_data, orient='index')
            df.index.name = 'layer'
            df.sort_index(inplace=True)
            df.reset_index(inplace=True)
            save_name = f"metrics_{prefix}_FULL.csv"
            df.to_csv(os.path.join(csv_output_dir, save_name), index=False)
            logger.info(f"💾 [CSV] Saved metrics to: {save_name}")

    logger.info("🎉 [Done] All tasks completed.")

if __name__ == "__main__":
    main()
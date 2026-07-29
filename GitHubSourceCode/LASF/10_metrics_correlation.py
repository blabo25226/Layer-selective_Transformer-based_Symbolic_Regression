"""
00_analysis_correlation_enhanced.py
关联分析：计算 Metrics 之间的相关性 (美化版)
在原版基础上优化了热力图的视觉效果
"""
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import hashlib
import json
import logging
import sys
import scipy

# 引入配置路径
sys.path.insert(0, os.getcwd())
from config import METRIC_DIR

# --- 增强的样式配置 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 10),
    'figure.dpi': 300,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.markersize': 6,
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
})


def cprint(text, color='default'):
    if color == 'green':
        print(f"🟢 {text}")
    elif color == 'red':
        print(f"🔴 {text}")
    elif color == 'yellow':
        print(f"🟡 {text}")
    else:
        print(text)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze metrics correlation')

    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/7b'],
                        help='Model configs in format "family/scale"')

    parser.add_argument('--dataset', type=str, default='mmlu',
                        help='Dataset name')

    parser.add_argument('--variants', type=str, nargs='+',
                        default=['sft', 'dpo', 'rlvr', 'instruct'],
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

    parser.add_argument('--max_samples', type=int, default=1000)

    args = parser.parse_args()

    # Metrics: 如果是 all，需要展开并排序
    if 'all' in args.metrics:
        args.metrics = sorted(ALL_METRICS)
    else:
        args.metrics = sorted(args.metrics)

    # Models: 原文件进行了排序
    args.models = sorted(args.models)

    return args


def get_cache_path(args):
    """复用 Hash 计算逻辑"""
    config_dict = {
        'dataset': args.dataset,
        'models': args.models,
        'variants': args.variants,
        'metrics': args.metrics,
        'normalization': args.normalization,
        'max_samples': args.max_samples
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    filename = f"cache_{args.dataset}_{run_hash}.pt"
    return os.path.join(METRIC_DIR, filename)


def flatten_and_align_data(cache_data):
    """展平并对齐数据"""
    records = []
    temp_data = {}

    for metric_name, model_dict in cache_data.items():
        for model_key, layer_values in model_dict.items():
            parts = model_key.split('-')
            if 'vs' in parts:
                try:
                    vs_index = parts.index('vs')
                    family = parts[0]
                    scale = parts[1]
                    variant = parts[vs_index + 1]
                except:
                    continue
            else:
                family = parts[0]
                scale = parts[1]
                variant = parts[2]

            final_values = []
            if isinstance(layer_values, dict):
                first_key = sorted(list(layer_values.keys()))[0]
                final_values = layer_values[first_key]
            elif isinstance(layer_values, list):
                final_values = layer_values

            for layer_idx, val in enumerate(final_values):
                row_id = (family, scale, variant, layer_idx)
                if row_id not in temp_data:
                    temp_data[row_id] = {
                        'family': family, 'scale': scale, 'variant': variant, 'layer': layer_idx
                    }
                temp_data[row_id][metric_name] = val

    df = pd.DataFrame(list(temp_data.values()))
    return df


def plot_correlation_heatmap(df, output_dir, file_tag):
    """
    美化版的相关性热力图

    改进点：
    1. 更大的画布和更好的分辨率
    2. 优化的配色方案 (RdYlBu_r)
    3. 美化的指标名称显示
    4. 加粗的数值标注
    5. 优化的标签旋转和对齐
    6. 更清晰的网格线
    7. 增强的颜色条
    """
    non_metric_cols = ['family', 'scale', 'variant', 'layer']
    metric_cols = [c for c in df.columns if c not in non_metric_cols]

    if len(metric_cols) < 2:
        cprint("Not enough metrics to plot correlation!", 'red')
        return

    corr_matrix = df[metric_cols].corr(method='pearson')

    # 美化指标名称：下划线替换为空格，首字母大写
    display_names = {col: col.replace('_', ' ').title() for col in metric_cols}
    corr_matrix_display = corr_matrix.rename(columns=display_names, index=display_names)

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))

    # 创建上三角mask
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 绘制热力图
    sns.heatmap(
        corr_matrix_display,
        annot=True,
        fmt=".3f",            # 显示三位小数
        cmap='RdYlBu_r',      # 红-黄-蓝反向配色
        vmin=-1,
        vmax=1,
        center=0,
        mask=mask,
        square=True,
        linewidths=1.5,       # 更宽的网格线
        linecolor='white',    # 白色网格线
        cbar_kws={
            "shrink": 0.8,
            "aspect": 30,
            "pad": 0.02,
            "label": "Pearson Correlation Coefficient"
        },
        annot_kws={"size": 9, "weight": "bold"},  # 加粗数值
        ax=ax
    )

    # 设置标题
    ax.set_title(
        f'Metrics Correlation Matrix (Pearson)\nN = {len(df)} Layers',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    # 旋转x轴标签
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=11,
        rotation_mode='anchor'
    )

    # y轴标签保持水平
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=11
    )

    # 添加边框
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(2)
        spine.set_visible(True)

    plt.tight_layout()

    # 保存高分辨率图片
    save_path = os.path.join(output_dir, f'correlation_matrix_{file_tag}_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    cprint(f"✨ Saved enhanced correlation plot to: {save_path}", 'green')


def plot_correlation_clustermap(df, output_dir, file_tag):
    """
    聚类热力图版本 - 自动将相似的指标聚类在一起
    """
    non_metric_cols = ['family', 'scale', 'variant', 'layer']
    metric_cols = [c for c in df.columns if c not in non_metric_cols]

    if len(metric_cols) < 2:
        cprint("Not enough metrics to plot clustermap!", 'red')
        return

    corr_matrix = df[metric_cols].corr(method='pearson')

    # 美化指标名称
    display_names = {col: col.replace('_', ' ').title() for col in metric_cols}
    corr_matrix_display = corr_matrix.rename(columns=display_names, index=display_names)

    # 创建聚类热力图
    g = sns.clustermap(
        corr_matrix_display,
        annot=True,
        fmt=".3f",
        cmap='RdYlBu_r',
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=1.5,
        linecolor='white',
        figsize=(13, 11),
        cbar_kws={
            "shrink": 0.8,
            "label": "Pearson Correlation"
        },
        annot_kws={"size": 9, "weight": "bold"},
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.83, 0.03, 0.15)
    )

    # 设置标题
    g.fig.suptitle(
        f'Clustered Metrics Correlation Matrix\nN = {len(df)} Layers',
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    # 调整标签
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha='right',
        fontsize=10
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        fontsize=10
    )

    save_path = os.path.join(output_dir, f'correlation_clustermap_{file_tag}_enhanced.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    cprint(f"✨ Saved clustermap to: {save_path}", 'green')


def main():
    args = parse_args()

    cache_path = get_cache_path(args)
    cprint(f"Looking for cache file: {cache_path}", 'yellow')

    if not os.path.exists(cache_path):
        cprint(f"File not found! Expected Hash: {cache_path.split('_')[-1].replace('.pt', '')}", 'red')
        return

    try:
        cache_data = torch.load(cache_path)
        cprint(f"Successfully loaded cache.", 'green')
    except Exception as e:
        cprint(f"Error loading cache: {e}", 'red')
        return

    output_dir = os.path.join(METRIC_DIR, 'plots', 'correlation_analysis')
    os.makedirs(output_dir, exist_ok=True)

    df = flatten_and_align_data(cache_data)

    cprint(f"Data processed. Total observations (layers): {len(df)}", 'green')
    cprint(f"Metrics found: {[c for c in df.columns if c not in ['family', 'scale', 'variant', 'layer']]}", 'yellow')

    # 绘制美化版热力图
    cprint("\n📊 Generating enhanced correlation heatmap...", 'yellow')
    plot_correlation_heatmap(df, output_dir, "global")

    # 绘制聚类版本
    cprint("\n📊 Generating clustered correlation heatmap...", 'yellow')
    plot_correlation_clustermap(df, output_dir, "global")

    # 画 SFT 专属图
    if 'sft' in df['variant'].unique():
        df_sft = df[df['variant'] == 'sft']
        if len(df_sft) > 10:
            cprint("\n📊 Generating SFT-only correlation heatmap...", 'yellow')
            plot_correlation_heatmap(df_sft, output_dir, "sft_only")
            plot_correlation_clustermap(df_sft, output_dir, "sft_only")

    cprint("\n✅ All plots generated successfully!", 'green')


if __name__ == "__main__":
    main()
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, OrderedDict

sys.path.insert(0, os.getcwd())
from config import OUTPUT_DIR, DATASET_CACHE_DIR, MODEL_CACHE_DIR
from model_utils import load_model, extract_representations, parse_layer_indices
from data_utils import download_dataset, sample_dataset, create_dataloader
from metric_utils import compute_cka, compute_cosine_similarity, compute_mean_shift

# === Plotting Style ===
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5,
    'figure.dpi': 300
})

def parse_args():
    parser = argparse.ArgumentParser(
        description="Robustness Check: Ultimate Fix with Fixed Data Seed"
    )

    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/7b'])
    parser.add_argument('--dataset', type=str, nargs='+', default=['wikitext'])
    # 这些 seed 现在只代表"轮次"，不再影响数据采样
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1234, 2024])
    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['cka', 'cosine_similarity', 'mean_shift'],
                        choices=['cka', 'cosine_similarity', 'mean_shift'])

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--layer_indices', type=str, default='all',
                        choices=['all', 'key', 'sparse'])
    parser.add_argument('--max_length', type=int, default=256)

    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--model_dtype', type=str, default='bfloat16')
    parser.add_argument('--model_cache_dir', type=str, default=MODEL_CACHE_DIR)
    parser.add_argument('--dataset_cache_dir', type=str, default=DATASET_CACHE_DIR)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(OUTPUT_DIR, 'robustness_plots'))

    return parser.parse_args()


def extract_with_sample_ids(model, tokenizer, dataset_name, data_seed, layer_indices, args):
    """
    提取表征，返回带有sample_id索引的OrderedDict
    关键修改：使用 data_seed 控制采样，确保跨模型对比时数据一致
    """
    # 采样数据 (使用固定的 data_seed)
    raw_dataset = download_dataset(dataset_name, args.dataset_cache_dir)
    sampled_dataset = sample_dataset(raw_dataset, args.max_samples, seed=data_seed)

    # 创建dataloader
    dataloader = create_dataloader(
        sampled_dataset, dataset_name, tokenizer,
        args.batch_size, max_length=args.max_length
    )

    # 使用OrderedDict确保顺序
    sample_reps = OrderedDict()

    model.eval() # 确保 Dropout 关闭

    # 这里的 desc 显示 data_seed 以便于确认
    for batch in tqdm(dataloader, desc=f"    Extracting (data_seed={data_seed})", leave=False):
        sample_ids = batch['sample_id']

        with torch.no_grad():
            # Teacher Forcing: 直接 forward pass，不生成
            reps = extract_representations(
                model, batch, layer_indices,
                pooling_method='mean', dtype='float32'
            )

        # 为每个样本单独存储
        for i, sample_id in enumerate(sample_ids):
            sample_reps[sample_id] = {}
            for layer_idx, tensor in reps['pooled_states'].items():
                # 存储单个样本的表征
                sample_reps[sample_id][layer_idx] = tensor[i].cpu()

    # print(f"      ✅ Extracted {len(sample_reps)} samples with IDs: {list(sample_reps.keys())[:3]}...")
    return sample_reps


def align_and_stack_representations(base_reps_dict, sft_reps_dict):
    """
    对齐两个模型的表征，确保样本ID完全匹配
    """
    # 找到共同的sample IDs
    common_ids = sorted(list(set(base_reps_dict.keys()) & set(sft_reps_dict.keys())))

    if len(common_ids) == 0:
        print("❌ Critical Error: No common sample IDs found!")
        print(f"Base IDs sample: {list(base_reps_dict.keys())[:5]}")
        print(f"SFT IDs sample: {list(sft_reps_dict.keys())[:5]}")
        raise ValueError("No common sample IDs found! Check data_seed settings.")

    # 获取层索引
    first_sample_base = list(base_reps_dict.values())[0]
    layer_indices = sorted(first_sample_base.keys())

    # 对齐并堆叠
    base_aligned = {}
    sft_aligned = {}

    for layer_idx in layer_indices:
        base_list = []
        sft_list = []

        for sample_id in common_ids:
            base_list.append(base_reps_dict[sample_id][layer_idx])
            sft_list.append(sft_reps_dict[sample_id][layer_idx])

        base_aligned[layer_idx] = torch.stack(base_list)
        sft_aligned[layer_idx] = torch.stack(sft_list)

    return base_aligned, sft_aligned


def compute_metric(reps_a, reps_b, metric_name):
    """计算对齐度量"""
    first_layer = sorted(reps_a.keys())[0]
    assert reps_a[first_layer].shape == reps_b[first_layer].shape

    if metric_name == 'cka':
        return compute_cka(reps_a, reps_b)
    elif metric_name == 'cosine_similarity':
        return compute_cosine_similarity(reps_a, reps_b)
    elif metric_name == 'mean_shift':
        return compute_mean_shift(reps_a, reps_b)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("🔬 Robustness Check: ULTIMATE FIX (Fixed Data Seed)")
    print(f"📊 Models: {args.models}")
    print(f"📚 Datasets: {args.dataset}")
    print(f"🌱 Loop Seeds: {args.seeds}")

    # === 关键修复：定义一个全局固定的 Data Seed ===
    # 确保所有模型、所有轮次都处理完全相同的 100 个样本
    FIXED_DATA_SEED = args.seeds[0]
    print(f"🔒 FIXED DATA SEED: {FIXED_DATA_SEED} (Used for ALL samplings)")
    print("="*70)

    reference_seed = args.seeds[0]

    for model_str in args.models:
        family, scale = model_str.split('/')

        for d_name in args.dataset:
            print(f"\n{'='*70}")
            print(f"🚀 Processing: {family}/{scale} on {d_name}")
            print(f"{'='*70}")

            # 获取层索引 (Dummy load)
            temp_model, _ = load_model(family, scale, 'base',
                                      cache_dir=args.model_cache_dir,
                                      device_map='cpu',
                                      dtype=args.model_dtype)
            num_layers = temp_model.config.num_hidden_layers
            layer_indices = parse_layer_indices(args.layer_indices, num_layers)
            del temp_model
            gc.collect()

            # ==============================================================
            # 步骤 1: 提取 Base（循环不同 loop_seed，但数据固定）
            # ==============================================================
            print(f"\n  ⏳ [1/2] Loading BASE model...")
            base_model, base_tokenizer = load_model(
                family, scale, 'base',
                cache_dir=args.model_cache_dir,
                device_map=args.device_map,
                dtype=args.model_dtype
            )

            base_reps_storage = {}
            for loop_seed in args.seeds:
                print(f"    Processing loop_seed {loop_seed}...")
                # 核心修正：这里传入 FIXED_DATA_SEED
                base_reps_storage[loop_seed] = extract_with_sample_ids(
                    base_model, base_tokenizer, d_name, FIXED_DATA_SEED, layer_indices, args
                )

            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            print("  ✅ Base done\n")

            # ==============================================================
            # 步骤 2: 提取 SFT（循环不同 loop_seed，但数据固定）
            # ==============================================================
            print(f"  ⏳ [2/2] Loading SFT model...")
            sft_model, sft_tokenizer = load_model(
                family, scale, 'sft',
                cache_dir=args.model_cache_dir,
                device_map=args.device_map,
                dtype=args.model_dtype
            )

            sft_reps_storage = {}
            for loop_seed in args.seeds:
                print(f"    Processing loop_seed {loop_seed}...")
                # 核心修正：这里传入 FIXED_DATA_SEED
                sft_reps_storage[loop_seed] = extract_with_sample_ids(
                    sft_model, sft_tokenizer, d_name, FIXED_DATA_SEED, layer_indices, args
                )

            del sft_model
            gc.collect()
            torch.cuda.empty_cache()
            print("  ✅ SFT done\n")

            # ==============================================================
            # 步骤 3: 对齐并计算 metrics
            # ==============================================================
            for metric_name in args.metrics:
                print(f"  📊 Computing {metric_name.upper()}...")

                fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

                # === 线 1: Base vs SFT (Primary Interest) ===
                print(f"    - Aligning Base vs SFT (ref seed)")
                base_aligned, sft_aligned = align_and_stack_representations(
                    base_reps_storage[reference_seed],
                    sft_reps_storage[reference_seed]
                )

                scores_base_vs_sft = compute_metric(base_aligned, sft_aligned, metric_name)
                layers = list(range(len(scores_base_vs_sft)))

                ax.plot(layers, scores_base_vs_sft,
                       label=f'Base vs SFT',
                       color='black', linestyle='--', linewidth=3,
                       marker='s', markersize=7, alpha=0.9, zorder=10)

                # === 线 2-N: SFT vs SFT (Check Stability) ===
                # 预期：如果加载的是同一个checkpoint，这些线应该是1.0
                colors_sft = sns.color_palette("husl", len(args.seeds))

                for idx, seed in enumerate(args.seeds):
                    if seed == reference_seed: continue # Skip self vs self for clarity

                    print(f"    - Aligning SFT(ref) vs SFT({seed})")
                    sft_ref_aligned, sft_other_aligned = align_and_stack_representations(
                        sft_reps_storage[reference_seed],
                        sft_reps_storage[seed]
                    )
                    scores = compute_metric(sft_ref_aligned, sft_other_aligned, metric_name)

                    ax.plot(layers, scores,
                           label=f'SFT vs SFT ({seed})',
                           color=colors_sft[idx], linestyle='-', linewidth=2,
                           alpha=0.7)

                # === 线 N+1: Base vs Base (Check Stability) ===
                # 预期：这些线应该是1.0
                colors_base = sns.color_palette("Set2", len(args.seeds))

                for idx, seed in enumerate(args.seeds):
                    if seed == reference_seed: continue

                    print(f"    - Aligning Base(ref) vs Base({seed})")
                    base_ref_aligned, base_other_aligned = align_and_stack_representations(
                        base_reps_storage[reference_seed],
                        base_reps_storage[seed]
                    )
                    scores = compute_metric(base_ref_aligned, base_other_aligned, metric_name)

                    ax.plot(layers, scores,
                           label=f'Base vs Base ({seed})',
                           color=colors_base[idx], linestyle='-.', linewidth=2,
                           alpha=0.7)

                # === 美化 ===
                metric_display = metric_name.replace('_', ' ').title()
                ax.set_title(f"Representation Stability: {metric_display}\n{family}/{scale} - {d_name}",
                            fontsize=16, fontweight='bold')
                ax.set_xlabel("Layer Depth", fontsize=14)
                ax.set_ylabel(metric_display, fontsize=14)

                if metric_name in ['cka', 'cosine_similarity']:
                    ax.set_ylim(0.0, 1.05) # 稍微留点空间看清楚 1.0 的线
                elif metric_name == 'mean_shift':
                    ax.set_ylim(bottom=0.0)

                # 调整 Legend 位置，避免遮挡黑线
                ax.legend(loc='lower left', frameon=True, fontsize=9, framealpha=0.95)
                ax.grid(True, linestyle='--', alpha=0.3)
                plt.ylim(0.8, 1.1)
                plt.tight_layout()

                save_path = os.path.join(
                    args.output_dir,
                    f"fixed_input_{metric_name}_{family}_{scale}_{d_name}.png"
                )
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"  📸 Saved: {save_path}\n")

                # 如果在远程跑，可以注释掉 show
                # plt.show()
                plt.close()

    print("\n" + "="*70)
    print("🎉 All tasks completed!")
    print("="*70)


if __name__ == "__main__":
    main()
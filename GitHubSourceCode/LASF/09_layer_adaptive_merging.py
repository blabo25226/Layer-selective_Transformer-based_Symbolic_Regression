"""
09_layer_adaptive_merging.py
[实验] 基于层敏感性的自适应模型融合 (Layer-Adaptive Merging) - 深度控制修复版

功能：
1. 实现 Theta_new = alpha * Theta_SFT + (1 - alpha) * Theta_Base
2. 支持 --swap_depth: 每个 Segment 内部只对最后 N 层进行融合，其余层保持 Base。
3. 修正了 SFT 权重加载和融合的数值稳定性问题。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import logging
import json
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# 引入项目路径
sys.path.insert(0, os.getcwd())

from config import MODEL_CACHE_DIR, OUTPUT_DIR
from model_utils import load_model, get_num_layers
from data_utils import download_dataset, sample_dataset, create_dataloader

# 绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")

def cprint(text, color='default'):
    t = datetime.now().strftime('%H:%M:%S')
    colors = {
        'green': '\033[92m', 'red': '\033[91m', 'yellow': '\033[93m', 'blue': '\033[94m',
        'purple': '\033[95m', 'white': '\033[0m'
    }
    end_color = '\033[0m'
    color_code = colors.get(color, '\033[0m')
    print(f"[{t}] {color_code}{text}{end_color}")

def setup_logging():
    os.makedirs(os.path.join(OUTPUT_DIR, 'merging'), exist_ok=True)
    logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================================
# 核心融合逻辑 (Fixed)
# ============================================================================

def generate_layer_alpha_map(num_layers, alpha_config_str, swap_depth):
    """
    生成每一层的 alpha 值。
    逻辑：
    1. 将 layers 分成 len(alphas) 个 segment。
    2. 对于每个 segment，只有最后 swap_depth 层应用该 segment 的 alpha。
    3. 其余层 alpha = 0 (Base)。
    """
    try:
        segment_alphas = [float(x) for x in alpha_config_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid alpha config: {alpha_config_str}")

    num_segments = len(segment_alphas)
    segment_size = num_layers / num_segments
    alpha_map = {} # layer_idx -> alpha_value

    # 初始化所有层为 0
    for i in range(num_layers):
        alpha_map[i] = 0.0

    for i in range(num_segments):
        target_alpha = segment_alphas[i]
        if target_alpha == 0.0:
            continue

        # 计算当前 Segment 的范围
        seg_start_idx = int(i * segment_size)
        seg_end_idx = int((i + 1) * segment_size)
        seg_end_idx = min(seg_end_idx, num_layers)

        # 计算生效的层 (Depth Control)
        # 例如 [0,1,2,3], depth=2 -> 生效的是 2,3
        active_start = max(seg_start_idx, seg_end_idx - swap_depth)

        for layer_idx in range(active_start, seg_end_idx):
            alpha_map[layer_idx] = target_alpha

    return alpha_map

def get_module_by_name(model, module_name):
    """Safe retrieval of module by name"""
    parts = module_name.split('.')
    curr = model
    try:
        for part in parts:
            curr = getattr(curr, part)
        return curr
    except AttributeError:
        return None

def perform_adaptive_merging(base_model, sft_model, alpha_map):
    """
    执行融合：Base = (1-alpha)*Base + alpha*SFT
    使用 torch.lerp 提高数值稳定性。
    """
    device = next(base_model.parameters()).device

    # 1. 获取 Layer 列表
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        base_layers = base_model.model.layers
        sft_layers = sft_model.model.layers
    else:
        base_layers = base_model.layers
        sft_layers = sft_model.layers

    num_layers = len(base_layers)

    # 2. 遍历所有层进行融合
    for i in tqdm(range(num_layers), desc="Merging Layers", leave=False):
        alpha = alpha_map.get(i, 0.0)

        # 优化：Alpha=0 跳过
        if alpha < 1e-6:
            continue

        base_layer = base_layers[i]
        sft_layer = sft_layers[i]
        sft_state = sft_layer.state_dict()

        for name, base_param in base_layer.named_parameters():
            if name in sft_state:
                sft_param = sft_state[name].to(device)

                # 优化：Alpha=1 直接复制 (Pure SFT)
                if abs(alpha - 1.0) < 1e-6:
                    with torch.no_grad():
                        base_param.data.copy_(sft_param.data)
                # 混合：Lerp
                else:
                    with torch.no_grad():
                        # lerp(start, end, weight) = start + weight * (end - start)
                        # = base + alpha * (sft - base) = (1-a)base + a*sft
                        base_param.data.lerp_(sft_param.data, alpha)

                del sft_param

    # 3. 处理 Head 和 Final Norm
    # 策略：使用最后一层的 alpha 决定 Head
    last_alpha = alpha_map.get(num_layers - 1, 0.0)
    if last_alpha > 1e-6:
        # Norm
        norm_names = ['model.norm', 'model.final_layernorm']
        for name in norm_names:
            base_norm = get_module_by_name(base_model, name)
            sft_norm = get_module_by_name(sft_model, name)
            if base_norm and sft_norm:
                sft_state = sft_norm.state_dict()
                for n, p in base_norm.named_parameters():
                    if n in sft_state:
                        s_p = sft_state[n].to(device)
                        with torch.no_grad():
                            if abs(last_alpha - 1.0) < 1e-6:
                                p.data.copy_(s_p.data)
                            else:
                                p.data.lerp_(s_p.data, last_alpha)

        # LM Head
        if hasattr(base_model, 'lm_head') and hasattr(sft_model, 'lm_head'):
            sft_state = sft_model.lm_head.state_dict()
            for n, p in base_model.lm_head.named_parameters():
                if n in sft_state:
                    s_p = sft_state[n].to(device)
                    with torch.no_grad():
                        if abs(last_alpha - 1.0) < 1e-6:
                            p.data.copy_(s_p.data)
                        else:
                            p.data.lerp_(s_p.data, last_alpha)

    torch.cuda.empty_cache()

# ============================================================================
# 评估工具
# ============================================================================

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

    avg_loss = total_loss / (total_tokens + 1e-9)
    ppl = np.exp(avg_loss) if avg_loss < 20 else 1e5 # Cap extreme PPL
    acc = total_correct / (total_tokens + 1e-9)
    return ppl, acc

def plot_merging_results(results_db, base_ref, sft_ref, output_dir, model_name):
    """
    绘制结果，包含 Base 和 SFT 的基准线
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = list(results_db.keys())
    strategies = sorted(list(set(k for d in results_db for k in results_db[d])))

    colors = sns.color_palette("viridis", len(strategies))
    x = np.arange(len(datasets))
    width = 0.8 / len(strategies)

    plt.figure(figsize=(12, 7), dpi=300)

    # 1. 绘制柱状图 (Strategies)
    for i, strat in enumerate(strategies):
        y_vals = [results_db[d].get(strat, {}).get('acc', 0) for d in datasets]
        plt.bar(x + i*width, y_vals, width, label=strat, color=colors[i], alpha=0.85, edgecolor='white')

    # 2. 绘制基准线 (Base & SFT) - 使用 Scatter 点或横线
    # 为了清晰，我们在每个 dataset 的中心位置画短横线
    for i, d in enumerate(datasets):
        b_acc = base_ref[d]['acc']
        s_acc = sft_ref[d]['acc']

        # Base Line (Grey)
        plt.hlines(b_acc, x[i]-0.4, x[i]+0.4, colors='gray', linestyles='--', linewidth=2,
                   label='Base Ref' if i==0 else "")
        plt.text(x[i], b_acc + 0.01, 'Base', color='gray', ha='center', fontsize=8, fontweight='bold')

        # SFT Line (Red)
        plt.hlines(s_acc, x[i]-0.4, x[i]+0.4, colors='red', linestyles='--', linewidth=2,
                   label='SFT Ref' if i==0 else "")
        plt.text(x[i], s_acc + 0.01, 'SFT', color='red', ha='center', fontsize=8, fontweight='bold')

    plt.xlabel('Datasets', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'Layer-Adaptive Merging vs Base/SFT ({model_name})', fontsize=16)
    plt.xticks(x + width * (len(strategies) - 1) / 2, datasets, fontsize=12)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Strategies")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'merging_acc_{model_name.replace("/", "_")}.pdf')
    plt.savefig(save_path, dpi=300)
    cprint(f"📸 Saved plot: {save_path}", 'green')
    plt.close()

# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Layer-Adaptive Merging Experiment")

    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/1b'])
    parser.add_argument('--dataset', type=str, nargs='+', default=['gsm8k', 'mmlu', 'ifeval', 'humaneval'])

    # 策略配置：0=Base, 1=SFT
    parser.add_argument('--alpha_configs', type=str, nargs='+',
                        default=[
                            "0,1,1,1,0",          # Paper Strategy
                            "0,1,1,1,0.5",        # Optimized
                            "0.5,1,1,1,0.5",      # Smooth
                        ],
                        help='List of alpha configurations per segment.')

    # [NEW] 深度控制参数
    parser.add_argument('--num_segments', type=int, default=5,
                        help='Number of segments to divide the model into.')
    parser.add_argument('--swap_depth', type=int, default=100,
                        help='Number of layers to merge within each segment. Default 100 means full segment.')

    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'merging'))

    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging()

    cprint("="*50, 'blue')
    cprint("🧬 Layer-Adaptive Merging (Fixed SFT & Depth)", 'blue')
    cprint(f"📦 Models: {args.models}", 'blue')
    cprint(f"🔢 Segments: {args.num_segments} | Depth: {args.swap_depth}", 'blue')
    cprint("="*50, 'blue')

    for model_str in args.models:
        family, scale = model_str.split('/')
        num_layers = get_num_layers(family, scale)
        cprint(f"\nProcessing {model_str} ({num_layers} layers)...", 'purple')

        # 1. Load SFT (Donor) - 确保类型正确
        cprint(f"  📥 Loading SFT (Donor) to CPU...", 'yellow')
        sft_model, tokenizer = load_model(family, scale, 'sft', device_map='cpu', dtype='float16')
        sft_model.eval()

        # 2. Load Base (Patient)
        cprint(f"  📥 Loading Base (Patient) to GPU...", 'yellow')
        base_model, _ = load_model(family, scale, 'base', device_map='cuda', dtype='float16')
        base_model.eval()
        device = next(base_model.parameters()).device

        # 3. Snapshot Base
        cprint(f"  💾 Creating Base snapshot in RAM...", 'white')
        base_snapshot = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

        # 4. Prepare Data
        dataloaders = {}
        for ds_name in args.dataset:
            dataset = download_dataset(ds_name)
            sampled = sample_dataset(dataset, args.n_samples)
            dataloaders[ds_name] = create_dataloader(sampled, ds_name, tokenizer, args.batch_size, 512)

        # 5. Calculate References (Independently)
        base_ref = {}
        sft_ref = {}

        # 5.1 Base Ref
        cprint(f"\n📏 Calculating Base Reference...", 'white')
        for ds_name, loader in dataloaders.items():
            ppl, acc = evaluate_model(base_model, loader, device, tokenizer)
            base_ref[ds_name] = {'acc': acc, 'ppl': ppl}
            cprint(f"   -> {ds_name}: Acc={acc:.4f}", 'green')

        # 5.2 SFT Ref (Load full SFT weights)
        cprint(f"\n📏 Calculating SFT Reference...", 'white')
        base_model.load_state_dict(sft_model.state_dict(), strict=False)
        for ds_name, loader in dataloaders.items():
            ppl, acc = evaluate_model(base_model, loader, device, tokenizer)
            sft_ref[ds_name] = {'acc': acc, 'ppl': ppl}
            cprint(f"   -> {ds_name}: Acc={acc:.4f}", 'green')

        # 6. Run Experiments
        results_db = defaultdict(lambda: defaultdict(dict))

        for config_str in args.alpha_configs:
            try:
                # 检查 segment 数量是否匹配
                seg_counts = len(config_str.split(','))
                if seg_counts != args.num_segments:
                    cprint(f"⚠️ Config {config_str} length ({seg_counts}) != num_segments ({args.num_segments}), skipping.", 'red')
                    continue

                alpha_map = generate_layer_alpha_map(num_layers, config_str, args.swap_depth)
                strat_name = f"Alpha [{config_str}]"
            except Exception as e:
                cprint(f"Skipping invalid config {config_str}: {e}", 'red')
                continue

            cprint(f"\n🧪 Testing Strategy: {strat_name}", 'blue')

            # Restore Base
            base_model.load_state_dict(base_snapshot, strict=False)

            # Perform Merging
            perform_adaptive_merging(base_model, sft_model, alpha_map)

            # Evaluate
            for ds_name, loader in dataloaders.items():
                ppl, acc = evaluate_model(base_model, loader, device, tokenizer)
                results_db[ds_name][strat_name] = {'acc': acc, 'ppl': ppl}
                cprint(f"   -> {ds_name}: Acc={acc:.4f}, PPL={ppl:.2f}", 'green')

        # 7. Save & Plot
        json_path = os.path.join(args.output_dir, f'merging_results_{family}_{scale}.json')
        save_data = {
            'base_ref': base_ref,
            'sft_ref': sft_ref,
            'experiments': json.loads(json.dumps(results_db))
        }
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        plot_merging_results(results_db, base_ref, sft_ref, args.output_dir, model_str)

        # Cleanup
        del sft_model, base_model, base_snapshot
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
"""
03_vector_patching_v3.py
向量干预 V3：全参数列表化 (Grid Search Support)
支持批量对比：Models x Datasets x Alphas x Segments
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import argparse
import logging
import math
import gc # 引入垃圾回收，防止批量跑模型时显存爆炸
from glob import glob
from tqdm import tqdm
import sys

sys.path.insert(0, os.getcwd())
from config import REPRESENTATION_DIR
from model_utils import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- (保持 compute_mean_shift_vector, get_target_layers, patch_model_and_generate 不变) ---
# 为节省篇幅，这里省略重复的函数定义，逻辑与上一版相同
# 请确保 compute_mean_shift_vector, get_target_layers, patch_model_and_generate 依然在代码中
# ---------------------------------------------------------------------------

def compute_mean_shift_vector(family, scale, dataset, top_k_ratio=1.0):
    """(同上个版本: 计算 Mean Shift Vector)"""
    logger.info(f"Computing Mean Shift Vector for [{family}/{scale}] on [{dataset}]...")

    def load_vectors(variant):
        path = os.path.join(REPRESENTATION_DIR, family, f"{scale}_{variant}", dataset)
        files = glob(os.path.join(path, "*.pt"))
        files = files[:100] if len(files) > 100 else files
        if not files: return None # 修改：找不到返回 None 而不是报错，防止打断整个循环

        agg = {}
        for f in tqdm(files, desc=f"Loading {variant}", leave=False):
            d = torch.load(f, map_location='cpu')
            for layer, tensor in d['pooled_states'].items():
                if layer not in agg: agg[layer] = []
                agg[layer].append(tensor.float())
        return {l: torch.stack(tensors).mean(dim=0) for l, tensors in agg.items()}

    base_means = load_vectors('base')
    sft_means = load_vectors('sft')

    if not base_means or not sft_means:
        logger.warning(f"Missing data for {family}/{scale} on {dataset}. Skipping.")
        return None

    shift_vectors = {}
    for l in base_means.keys():
        if l in sft_means:
            vec = sft_means[l] - base_means[l]
            if top_k_ratio < 1.0:
                k = int(vec.numel() * top_k_ratio)
                if k > 0:
                    topk_vals, _ = torch.topk(torch.abs(vec), k)
                    vec[torch.abs(vec) < topk_vals[-1]] = 0.0
            shift_vectors[l] = vec
    return shift_vectors


def get_target_layers(total_layers, num_segments, segments_indx, swap_depth):
    """(同上个版本: 计算目标层)"""
    if num_segments <= 0: return list(range(total_layers))
    layers_per_seg = math.ceil(total_layers / num_segments)
    target_layers = []
    for seg_idx in segments_indx:
        if seg_idx < 0 or seg_idx >= num_segments: continue
        start_idx = seg_idx * layers_per_seg
        end_idx = min(start_idx + layers_per_seg, total_layers)
        patch_start = max(start_idx, end_idx - swap_depth)
        target_layers.extend(list(range(patch_start, end_idx)))
    return sorted(list(set(target_layers)))


def patch_model_and_generate(model, tokenizer, shift_vectors, prompt, alpha, target_layers):
    """(同上个版本: Hook 注入与生成)"""
    hooks = []
    device = next(model.parameters()).device
    def get_hook_fn(vec):
        def hook(module, args, output):
            if isinstance(output, tuple):
                h = output[0]
                perturbation = vec.to(dtype=h.dtype, device=h.device).view(1, 1, -1)
                h = h + alpha * perturbation
                return (h,) + output[1:]
            else:
                perturbation = vec.to(dtype=output.dtype, device=output.device).view(1, 1, -1)
                return output + alpha * perturbation
        return hook

    for i, layer_module in enumerate(model.model.layers):
        if i in target_layers and i in shift_vectors:
            hooks.append(layer_module.register_forward_hook(get_hook_fn(shift_vectors[i])))

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    for h in hooks: h.remove()
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Targeted Steering V3: Batch Experiments")

    # 1. 列表化 Model 和 Dataset
    parser.add_argument('--model', type=str, nargs='+', default=['olmo2/1b'],
                        help="List of models (e.g. olmo2/1b olmo2/7b)")

    parser.add_argument('--dataset', type=str, nargs='+', default=['mmlu'],
                        help="List of datasets (e.g. mmlu gsm8k)")

    # 2. 列表化 Alpha 和 Segment 配置
    parser.add_argument('--alpha', type=float, nargs='+', default=[-0.01, -0.1, 0.01, 0.1],
                        help="List of alphas")

    parser.add_argument('--num_segments', type=int, default=4)
    parser.add_argument('--segments_indx', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--swap_depth', type=int, default=1)

    args = parser.parse_args()

    test_prompt = "Question: What is the capital of France?\nAnswer:"

    # --- 第一层循环：遍历模型 ---
    for model_full_name in args.model:
        logger.info(f"\n{'='*60}\nProcessing Model: {model_full_name}\n{'='*60}")

        try:
            family, scale = model_full_name.split('/')

            # 加载模型 (在 Dataset 循环外加载，避免重复 IO)
            logger.info(f"Loading base model {model_full_name}...")
            model, tokenizer = load_model(family, scale, 'base', dtype='bfloat16')
            num_total_layers = len(model.model.layers)

            # 计算目标层 (基于当前模型的层数)
            target_layers = get_target_layers(num_total_layers, args.num_segments, args.segments_indx, args.swap_depth)
            logger.info(f"Target Layers for Steering: {target_layers}")

            # --- 第二层循环：遍历数据集 ---
            for d_name in args.dataset:
                logger.info(f"--- Dataset: {d_name} ---")

                # 计算向量 (依赖 Model + Dataset)
                shift_vecs = compute_mean_shift_vector(family, scale, d_name)

                if shift_vecs is None:
                    continue # 跳过无效数据集

                # Baseline (每个数据集跑一次 baseline 对照)
                print(f"\n[Model: {scale} | Data: {d_name} | Baseline]")
                print(patch_model_and_generate(model, tokenizer, shift_vecs, test_prompt, 0.0, []))

                # --- 第三层循环：遍历 Alpha ---
                for a in args.alpha:
                    print(f"\n[Model: {scale} | Data: {d_name} | Alpha: {a} | Layers: {args.segments_indx}]")
                    res = patch_model_and_generate(model, tokenizer, shift_vecs, test_prompt, a, target_layers)
                    print(res)

            # 模型切换前清理显存
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing model {model_full_name}: {e}")
            continue

if __name__ == "__main__":
    main()
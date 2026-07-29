"""
13_segment_lora_adapter_fixed.py
E13: Segment-wise Adaptive Adapter (Transfer Learning & Depth Control)
** 支持断点续传 (Resume Capability) + 唯一化文件命名 **
"""
import os

# # ================= 强制离线模式 =================
# # 告诉 HF 只能使用本地缓存，绝对禁止联网检查更新
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# # ===============================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import argparse
import logging
import gc
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# HF & PEFT
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# === Project Imports ===
sys.path.insert(0, os.getcwd())
from config import (
    MODEL_CONFIGS, DATASET_CONFIGS,
    OUTPUT_DIR, DATASET_CACHE_DIR, MODEL_CACHE_DIR
)
from model_utils import load_model, get_num_layers
from data_utils import format_sample

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Plot Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")


def cprint(text, color='default'):
    """只在主进程打印"""
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        colors = {'green': '\033[92m', 'red': '\033[91m', 'blue': '\033[94m',
                  'yellow': '\033[93m', 'default': '\033[0m', 'purple': '\033[95m'}
        print(f"{colors.get(color, '')}{text}{colors['default']}")


def parse_args():
    parser = argparse.ArgumentParser(description='E13: Segment-wise Adaptive Adapter (DDP Fixed)')

    # Model
    parser.add_argument('--model', type=str, default='olmo2/1b', help='Format: family/scale')

    # Data
    parser.add_argument('--train_dataset', type=str, default='gsm8k',
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for SFT')
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['gsm8k', 'mmlu'],
                        help='Datasets used for evaluation')

    # Strategy Params
    parser.add_argument('--adapter_configs', type=str, nargs='+',
                        # default=[
                        #     '8,8,8,8,8',           # 1. Baseline -> 128 (Budget)
                        #     '-1,-1,21,21,-1',  # 6. 中间靠右+ (Seg 3-4, 6L): 21*6=126 ≈ 128
                        #     '-1,-1,42,-1,-1',      # 2. 中间单峰 (Seg 3, 3L): 42*3=126 ≈ 128
                        #     '-1,14,14,14,-1',      # 3. 中间宽幅 (Seg 2-4, 9L): 14*9=126 ≈ 128
                        #     '-1,21,21,-1,-1',      # 4. 中间靠左+ (Seg 2-3, 6L): 21*6=126 ≈ 128
                        #     '-1,42,-1,-1,-1',      # 5. 中间靠左 (Seg 2, 3L): 42*3=126 ≈ 128
                        #     '-1,-1,-1,42,-1',      # 7. 中间靠右 (Seg 4, 3L): 42*3=126 ≈ 128
                        #     '-1,-1,-1,-1,32',      # 8. 高层强化 (Seg 5, 4L): 32*4=128 (=Budget)
                        #     '42,-1,-1,-1,-1',      # 9. 低层强化 (Seg 1, 3L): 42*3=126 ≈ 128
                        # ],
                        default = [
                            '64,64,64,64,64',  # 1. Baseline -> 1024
                            '-1,-1,341,-1,-1',  # 2. 中间单峰 (Seg 3, 3L): 341*3=1023 ≈ 1024
                            '-1,113,113,113,-1',  # 3. 中间宽幅 (Seg 2-4, 9L): 113*9=1017 ≈ 1024
                            '-1,170,170,-1,-1',  # 4. 中间靠左+ (Seg 2-3, 6L): 170*6=1020 ≈ 1024
                            '-1,341,-1,-1,-1',  # 5. 中间靠左 (Seg 2, 3L): 341*3=1023 ≈ 1024
                            '-1,-1,170,170,-1',  # 6. 中间靠右+ (Seg 3-4, 6L): 170*6=1020 ≈ 1024
                            '-1,-1,-1,341,-1',  # 7. 中间靠右 (Seg 4, 3L): 341*3=1023 ≈ 1024
                            '-1,-1,-1,-1,256',  # 8. 高层强化 (Seg 5, 4L): 256*4=1024 (=Budget)
                            '341,-1,-1,-1,-1',  # 9. 低层强化 (Seg 1, 3L): 341*3=1023 ≈ 1024
                        ],
                        help='List of comma-separated ranks.')

    parser.add_argument('--adapter_depth', type=int, default=100)

    # LoRA Params
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Training Config
    parser.add_argument('--num_epochs', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Per-GPU batch size. Effective = this × num_gpus × grad_accum')
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--grad_accum', type=int, default=8)
    # ================================================

    parser.add_argument('--n_train_samples', type=int, default=2000)
    parser.add_argument('--n_test_samples', type=int, default=100)

    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'adaptive_adapter_viz'))
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ============================================================================
# Helper Functions
# ============================================================================

def is_main_process():
    """检查是否为主进程（用于分布式训练）"""
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def get_local_rank():
    """获取当前进程的本地rank"""
    return int(os.environ.get('LOCAL_RANK', 0))


def parse_rank_string(config_str):
    try:
        return [int(x) for x in config_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid rank config string: {config_str}")


def calculate_segmented_rank_pattern(num_layers, rank_list, depth_per_seg):
    num_segments = len(rank_list)
    segment_size = num_layers / num_segments
    rank_pattern = {}
    active_layers = []

    for i in range(num_layers):
        rank_pattern[i] = 0

    for i, rank_val in enumerate(rank_list):
        if rank_val <= 0:
            continue

        seg_start_idx = int(i * segment_size)
        seg_end_idx = int((i + 1) * segment_size)
        seg_end_idx = min(seg_end_idx, num_layers)
        active_start = max(seg_start_idx, seg_end_idx - depth_per_seg)

        for layer_idx in range(active_start, seg_end_idx):
            rank_pattern[layer_idx] = rank_val
            active_layers.append(layer_idx)

    return rank_pattern, active_layers


def load_data_custom(dataset_name, split_key, n_samples):
    from datasets import load_dataset as hf_load

    if dataset_name not in DATASET_CONFIGS:
        cprint(f"Warning: {dataset_name} not in DATASET_CONFIGS, skipping...", 'red')
        return None

    cfg = DATASET_CONFIGS[dataset_name]
    target_split = cfg.get(split_key)
    if not target_split:
        target_split = 'train' if split_key == 'split_train' else 'test'

    cprint(f"Loading {dataset_name} | Split: {target_split}", 'yellow')
    try:
        if cfg['subset']:
            dataset = hf_load(cfg['hf_name'], cfg['subset'], split=target_split, cache_dir=DATASET_CACHE_DIR)
        else:
            dataset = hf_load(cfg['hf_name'], split=target_split, cache_dir=DATASET_CACHE_DIR)

        if 0 < n_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(n_samples))
        return dataset
    except Exception as e:
        cprint(f"Failed to load dataset {dataset_name}: {e}", 'red')
        return None


def prepare_dataset_for_sft(dataset, tokenizer, dataset_name):
    max_len = DATASET_CONFIGS[dataset_name].get('max_length', 512)

    def process_fn(example):
        text = format_sample(dataset_name, example)
        if not text:
            text = ""
        text = text + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=max_len)

    return dataset.map(process_fn, remove_columns=dataset.column_names)


# ============================================================================
# Evaluator (Metric Mode: Exact Match & Perplexity)
# ============================================================================

class DatasetEvaluator:
    def __init__(self, tokenizer, dataset_name, device):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS.get(dataset_name, {})
        self.metric_mode = self.config.get('metric_mode', 'loss')
        self.device = device
        self.max_len = self.config.get('max_length', 512)

        tokenizer.padding_side = "left"  # 推理必须左填充
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def extract_answer(self, text):
        text = text.strip()
        if self.dataset_name == 'gsm8k':
            nums = re.findall(r'-?[\d,]+\.?\d*', text)
            if not nums:
                return None
            clean_num = nums[-1].replace(',', '').replace('$', '')
            try:
                return float(clean_num)
            except:
                return None
        elif self.dataset_name == 'mmlu':
            match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match_end = re.search(r'\b([A-D])\s*$', text, re.IGNORECASE)
            if match_end:
                return match_end.group(1).upper()
            return None
        return text

    def run_evaluation(self, model, dataset, batch_size):
        if self.metric_mode == 'exact_match':
            return self._evaluate_generation_batch(model, dataset, batch_size)
        else:
            return self._evaluate_perplexity(model, dataset, batch_size)

    def _evaluate_generation_batch(self, model, dataset, batch_size):
        correct = 0
        total = 0
        model.eval()
        formatted_prompts = []
        golds = []

        for example in dataset:
            prompt = format_sample(self.dataset_name, example)
            if not prompt:
                continue
            if 'Answer:' in prompt:
                input_prompt = prompt.split('Answer:')[0] + 'Answer:'
                gold_raw = prompt.split('Answer:')[1].strip()
            else:
                input_prompt = prompt
                gold_raw = example.get('answer', '') or example.get('canonical_solution', '')
            formatted_prompts.append(input_prompt)
            golds.append(gold_raw)

        num_samples = len(formatted_prompts)
        for i in range(0, num_samples, batch_size):
            batch_prompts = formatted_prompts[i: i + batch_size]
            batch_golds = golds[i: i + batch_size]
            if not batch_prompts:
                continue

            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True,
                                    max_length=self.max_len).to(self.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)

            input_len = inputs.input_ids.shape[1]
            gen_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            for gen_text, gold_raw in zip(gen_texts, batch_golds):
                pred_val = self.extract_answer(gen_text)
                is_correct = False
                if self.dataset_name == 'gsm8k':
                    gold_val = self.extract_answer(gold_raw)
                    if pred_val is not None and gold_val is not None:
                        if abs(pred_val - gold_val) < 1e-3:
                            is_correct = True
                elif self.dataset_name == 'mmlu':
                    gold_val = self.extract_answer(gold_raw)
                    if pred_val == gold_val:
                        is_correct = True
                if is_correct:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0

    def _evaluate_perplexity(self, model, dataset, batch_size):
        total_loss = 0.0
        total_steps = 0
        model.eval()
        valid_texts = []
        for i in range(len(dataset)):
            t = format_sample(self.dataset_name, dataset[i])
            if isinstance(t, list):
                t = t[0] if t else ""
            if t:
                valid_texts.append(t + self.tokenizer.eos_token)

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i: i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_len,
                                    return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                labels=inputs.input_ids)
            total_loss += outputs.loss.item() * len(batch_texts)
            total_steps += len(batch_texts)

        avg_loss = total_loss / total_steps if total_steps > 0 else 99.0
        if self.metric_mode == 'perplexity':
            try:
                return math.exp(avg_loss)
            except:
                return 9999.0
        return avg_loss


# ============================================================================
# Visualization
# ============================================================================

def visualize_multi_dataset_results(results_df, output_dir, train_dataset, args, rank):
    os.makedirs(output_dir, exist_ok=True)

    # 唯一化文件名参数
    model_safe_name = args.model.replace("/", "-")

    # # 1. Bar Chart
    # plt.figure(figsize=(10, 6), dpi=300)
    # sns.barplot(data=results_df, x='TestDataset', y='Score', hue='Strategy', palette='viridis')
    # plt.title(f"Transfer Learning: {model_safe_name} trained on {train_dataset}")
    # plt.ylabel("Score (Acc or Low PPL)")
    # plt.grid(axis='y', alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f'lora_adapter_{model_safe_name}_r{rank}.png'))
    # plt.show()
    # plt.close()

    # 2. Efficiency Scatter
    train_subset = results_df[results_df['TestDataset'] == train_dataset].copy()
    if not train_subset.empty:
        plt.figure(figsize=(8, 6), dpi=300)
        sns.scatterplot(data=train_subset, x='Params', y='Score', hue='Strategy', s=400, style='Strategy')
        for i, row in train_subset.iterrows():
            plt.text(row['Params'], row['Score'], f"{row['Strategy']}", fontsize=14, alpha=0.7)
        plt.title(f"Param Efficiency: {model_safe_name} on {train_dataset}")
        plt.xlabel("Trainable Parameters")
        plt.ylabel(f"Score on {train_dataset}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'lora_adapter_{model_safe_name}_r{rank}.png'), dpi=300)
        plt.show()
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    args = parse_args()

    local_rank = get_local_rank()
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)

    cprint(f"🚀 DDP Training: {world_size} GPUs | Local Rank: {local_rank}", 'purple')
    cprint(f"   Effective Batch Size: {args.train_batch_size} × {world_size} × {args.grad_accum} = "
           f"{args.train_batch_size * world_size * args.grad_accum}", 'blue')

    if args.use_wandb and is_main_process():
        wandb.init(project="layerwise_adapter_transfer", config=vars(args))

    cprint(f"🚀 Experiment: {args.model} | Train: {args.train_dataset}", 'purple')
    cprint(f"   Test Transfer: {args.test_datasets}", 'purple')
    cprint(f"   Adapter Depth per Segment: {args.adapter_depth}", 'blue')

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ================= 关键修改：唯一化文件名 =================
    # 获取 Base Rank 方便命名
    first_rank_config = parse_rank_string(args.adapter_configs[0])
    base_rank = max(first_rank_config)  # 假设第一个配置通常包含基准rank

    model_safe_name = args.model.replace("/", "-")
    # csv_filename = f'lora_{model_safe_name}_{args.train_dataset}_rank{base_rank}_lr{args.learning_rate}_target.csv'
    csv_filename = f'lora_{model_safe_name}_{args.train_dataset}_rank{base_rank}_lr{args.learning_rate}.csv'
    csv_path = os.path.join(args.output_dir, csv_filename)

    finished_strategies = set()
    results_list = []

    # 尝试加载之前的进度
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                results_list = existing_df.to_dict('records')
                finished_strategies = set(existing_df['Strategy'].unique())

                # ==== 打印加载的数据内容供检查 ====
                if is_main_process():
                    cprint(f"\n🔄 Resuming from {csv_filename}.", 'yellow')
                    cprint(f"   Found {len(finished_strategies)} completed strategies.", 'yellow')
                    cprint(f"   Loaded Data Preview (First 5 rows):", 'yellow')
                    print(existing_df.head().to_string(index=False))
                    print("-" * 50)
                # ==================================
        except Exception as e:
            cprint(f"⚠️ Failed to load existing results: {e}", 'red')
    else:
        if is_main_process():
            cprint(f"🆕 No existing checkpoint found. Starting fresh: {csv_filename}", 'green')
    # ========================================================

    family, scale = args.model.split('/')
    num_layers = get_num_layers(family, scale)
    cprint(f"   Model Layers: {num_layers}", 'blue')

    # 1. Load Data
    train_ds = load_data_custom(args.train_dataset, 'split_train', args.n_train_samples)
    if train_ds is None:
        raise ValueError("Training dataset failed to load.")

    test_datasets_map = {}
    for ds_name in args.test_datasets:
        ds = load_data_custom(ds_name, 'split_test', args.n_test_samples)
        if ds:
            test_datasets_map[ds_name] = ds

    # 2. Define Strategies
    strategies = {}
    strategies['Pretrained'] = {'type': 'pretrained', 'allocation': {}, 'active_layers': []}

    for config_str in args.adapter_configs:
        ranks = parse_rank_string(config_str)
        if all(r == ranks[0] for r in ranks):
            name = f"Uniform_r{ranks[0]}"
        elif ranks[0] == -1 and ranks[-1] == -1:
            name = f"MidOnly_{config_str.replace(',', '_')}"
        else:
            name = f"Pattern_{config_str.replace(',', '_')}"

        alloc, active = calculate_segmented_rank_pattern(num_layers, ranks, args.adapter_depth)
        strategies[name] = {'type': 'lora', 'allocation': alloc, 'active_layers': active}

    cprint(f"📋 Strategies: {list(strategies.keys())}", 'blue')

    # 3. Execution Loop
    for name, strategy_cfg in strategies.items():
        if name in finished_strategies:
            if is_main_process():
                cprint(f"⏩ Skipping {name} (Already in {csv_filename})", 'green')
            continue

        cprint(f"\n{'=' * 20} Running {name} {'=' * 20}", 'blue')

        if world_size > 1:
            device_map = {"": local_rank}
        else:
            device_map = "auto"

        # Load Model
        try:
            model, tokenizer = load_model(family, scale, 'base', dtype='bfloat16', device_map=device_map)
        except Exception as e:
            cprint(f"❌ Error loading model: {e}", 'red')
            continue

        trainable_params = 0
        total_rank_units = 0

        # Apply LoRA
        if strategy_cfg['type'] == 'lora':
            allocation = strategy_cfg['allocation']
            active_layers = strategy_cfg['active_layers']
            total_rank_units = sum(allocation.values())
            max_rank = max(allocation.values()) if allocation else 0

            if max_rank > 0 and len(active_layers) > 0:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=max_rank,
                    rank_pattern=allocation,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    layers_to_transform=active_layers
                )
                model = get_peft_model(model, peft_config)

                trainable_params, all_params = model.get_nb_trainable_parameters()
                cprint(f"  > Active Layers: {len(active_layers)} | Params: {trainable_params:,}", 'purple')

                cprint(f"  > SFT on {args.train_dataset}...", 'yellow')
                train_tokenized = prepare_dataset_for_sft(train_ds, tokenizer, args.train_dataset)

                training_args = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, f"{name}_ckpt"),
                    num_train_epochs=args.num_epochs,
                    per_device_train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.learning_rate,
                    logging_steps=10,
                    save_strategy="no",
                    bf16=True,
                    ddp_find_unused_parameters=False,
                    ddp_bucket_cap_mb=25,
                    dataloader_num_workers=4,
                    dataloader_pin_memory=True,
                    dataloader_prefetch_factor=2,
                    gradient_checkpointing=False,
                    report_to="wandb" if args.use_wandb else "none",
                    logging_first_step=True,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_tokenized,
                    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
                )
                trainer.train()
            else:
                cprint("  > Config resulted in 0 active layers (Frozen). Treating as Pretrained.", 'yellow')

        # Evaluate (只在主进程评估)
        if is_main_process():
            # 评估时获取模型所在设备
            device = next(model.parameters()).device

            for test_name, test_ds in test_datasets_map.items():
                try:
                    evaluator = DatasetEvaluator(tokenizer, test_name, device)
                    score = evaluator.run_evaluation(model, test_ds, args.eval_batch_size)

                    # 构建结果字典 (包含 Model 和 Rank 信息)
                    res = {
                        'Model': args.model,
                        'TrainDataset': args.train_dataset,
                        'Strategy': name,
                        'TestDataset': test_name,
                        'Score': score,
                        'Params': trainable_params,
                        'RankUnits': total_rank_units
                    }
                    results_list.append(res)
                    cprint(f"    -> {test_name}: {score:.4f}", 'green')
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        cprint(f"❌ OOM during evaluation of {test_name}. Skipping this metric.", 'red')
                        torch.cuda.empty_cache()
                    else:
                        raise e

            # ================= 保存逻辑 =================
            try:
                temp_df = pd.DataFrame(results_list)
                temp_df.to_csv(csv_path, index=False)
                cprint(f"💾 Checkpoint updated: {csv_filename}", 'blue')
            except Exception as e:
                cprint(f"⚠️ Failed to save checkpoint: {e}", 'red')
            # ===========================================

        # 同步所有进程
        if world_size > 1:
            torch.distributed.barrier()

        del model, tokenizer
        if 'trainer' in locals():
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Save & Summary (只在主进程)
    if is_main_process():
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(csv_path, index=False)  # Final save
        cprint(f"\nFinal results saved to {csv_path}", 'blue')

        if not results_df.empty:
            visualize_multi_dataset_results(results_df, args.output_dir, args.train_dataset, args, base_rank)

            try:
                pivot_df = results_df.pivot(index='Strategy', columns='TestDataset', values='Score')
                print("\n" + "=" * 50)
                print(f"📊 FINAL RESULTS ({args.model})")
                print("=" * 50)
                print(pivot_df.to_string(float_format="{:.4f}".format))
            except Exception as e:
                cprint(f"Error printing summary: {e}", 'red')

            if args.use_wandb:
                wandb.log({"final_table": wandb.Table(dataframe=results_df)})
                wandb.finish()


if __name__ == '__main__':
    main()
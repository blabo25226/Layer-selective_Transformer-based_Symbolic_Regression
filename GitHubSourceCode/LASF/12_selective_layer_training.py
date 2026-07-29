"""
12_selective_layer_training.py
E11: Selective Layer Training - 选择性层训练

功能：
1. 测试多种层冻结策略
2. 对比训练成本和性能
3. 找到最优的冻结配置
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import torch
import argparse
import logging
import json
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from transformers import get_linear_schedule_with_warmup, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import time

sys.path.insert(0, os.getcwd())
from config import MODEL_CACHE_DIR, OUTPUT_DIR, DATASET_CACHE_DIR
from model_utils import load_model
from data_utils import download_dataset, sample_dataset, format_sample

plt.style.use('seaborn-v0_8-paper')

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
    log_file = os.path.join(output_dir, f"selective_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='E11: Selective Layer Training')
    
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/1b'],
                        help='Model configs in format "family/scale"')
    parser.add_argument('--datasets', type=str, nargs='+', default=['mmlu'],
                        help='Training datasets')
    parser.add_argument('--eval_datasets', type=str, nargs='+', default=['mmlu', 'gsm8k', 'ifeval'],
                        help='Evaluation datasets')
    
    # Freezing strategies
    parser.add_argument('--freeze_ratios', type=float, nargs='+', default=[0.0, 0.33, 0.5, 0.66],
                        help='Ratios of layers to freeze from the beginning (0.0 = no freeze, full training)')
    
    # Training config
    parser.add_argument('--n_train_samples', type=int, default=15000,
                        help='Number of training samples')
    parser.add_argument('--n_eval_samples', type=int, default=200,
                        help='Number of evaluation samples per dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    
    # System
    parser.add_argument('--model_dtype', type=str, default='bfloat16')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'selective_training'))
    
    return parser.parse_args()

# ============================================================================
# 冻结层函数
# ============================================================================
def freeze_layers(model, freeze_ratio, num_layers):
    """冻结指定比例的前N层"""
    num_freeze = int(num_layers * freeze_ratio)
    
    cprint(f"Freezing first {num_freeze}/{num_layers} layers (ratio={freeze_ratio:.2f})", 'yellow')
    
    # 找到layer模块
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        layers = model.layers
    
    # 冻结前N层
    frozen_params = 0
    total_params = 0
    
    for i, layer in enumerate(layers):
        for param in layer.parameters():
            total_params += param.numel()
            if i < num_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
    
    freeze_percent = 100.0 * frozen_params / total_params if total_params > 0 else 0
    cprint(f"Frozen {frozen_params:,} / {total_params:,} params ({freeze_percent:.1f}%)", 'yellow')
    
    return num_freeze

# ============================================================================
# 评估函数
# ============================================================================
def evaluate_model(model, tokenizer, dataset_name, n_samples, max_length, device):
    """评估模型性能"""
    from torch.nn import CrossEntropyLoss
    
    dataset = download_dataset(dataset_name, DATASET_CACHE_DIR)
    sampled = sample_dataset(dataset, n_samples, strategy='random')
    
    # 格式化数据
    texts = []
    for sample in sampled:
        text = format_sample(dataset_name, sample)
        if text:
            texts.append(text)
    
    if len(texts) == 0:
        return {'loss': 0.0, 'perplexity': 0.0, 'accuracy': 0.0}
    
    # Tokenize
    encodings = tokenizer(texts, truncation=True, max_length=max_length, 
                         padding='max_length', return_tensors='pt')
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    loss_fct = CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token_id)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 8), desc=f"Evaluating {dataset_name}", leave=False):
            batch_encodings = {k: v[i:i+8].to(device) for k, v in encodings.items()}
            
            outputs = model(**batch_encodings)
            logits = outputs.logits
            
            labels = batch_encodings['input_ids']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Accuracy
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != tokenizer.pad_token_id)
            correct = (preds == shift_labels) & mask
            
            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss < 10 else 1e10
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': accuracy
    }

# ============================================================================
# 训练函数
# ============================================================================
def train_model(model, tokenizer, train_dataset, args, freeze_ratio, num_layers):
    """训练模型"""
    # 冻结层
    num_frozen = freeze_layers(model, freeze_ratio, num_layers)
    
    # 准备数据
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length)
    
    # 格式化训练数据
    texts = []
    for sample in train_dataset:
        text = format_sample(args.datasets[0], sample)
        if text:
            texts.append(text)
    
    # 创建HF Dataset
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_dict({'text': texts})
    tokenized_dataset = hf_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=hf_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, 'checkpoints'),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=100,
        save_strategy='epoch',
        bf16=(args.model_dtype == 'bfloat16'),
        fp16=(args.model_dtype == 'float16'),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to='none',
        disable_tqdm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练
    cprint("Starting training...", 'yellow')
    trainer.train()
    
    training_time = time.time() - start_time
    
    cprint(f"Training completed in {training_time/60:.2f} minutes", 'green')
    
    return training_time, num_frozen

# ============================================================================
# 可视化
# ============================================================================
def visualize_results(all_results, output_dir):
    """可视化结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    data = []
    for model_key, strategies in all_results.items():
        for freeze_ratio, metrics in strategies.items():
            for dataset, scores in metrics['eval_results'].items():
                data.append({
                    'model': model_key,
                    'freeze_ratio': freeze_ratio,
                    'dataset': dataset,
                    'accuracy': scores['accuracy'],
                    'perplexity': scores['perplexity'],
                    'training_time': metrics['training_time'],
                    'frozen_layers': metrics['num_frozen_layers']
                })
    
    df = pd.DataFrame(data)
    
    # 绘图1: Accuracy vs Freeze Ratio
    for model_key in df['model'].unique():
        model_df = df[df['model'] == model_key]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        for dataset in model_df['dataset'].unique():
            dataset_df = model_df[model_df['dataset'] == dataset]
            axes[0].plot(dataset_df['freeze_ratio'], dataset_df['accuracy'], 
                        marker='o', label=dataset, linewidth=2)
        
        axes[0].set_xlabel('Freeze Ratio', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title(f'Accuracy vs Freeze Ratio ({model_key})', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Training Time
        unique_freeze = model_df.groupby('freeze_ratio').first().reset_index()
        axes[1].plot(unique_freeze['freeze_ratio'], unique_freeze['training_time']/60, 
                    marker='s', color='red', linewidth=2)
        axes[1].set_xlabel('Freeze Ratio', fontsize=12)
        axes[1].set_ylabel('Training Time (minutes)', fontsize=12)
        axes[1].set_title(f'Training Time vs Freeze Ratio ({model_key})', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'selective_training_{model_key}.png'), dpi=300)
        plt.close()
    
    # 保存数据
    df.to_csv(os.path.join(output_dir, 'selective_training_results.csv'), index=False)
    cprint(f"Saved results to {output_dir}", 'green')

# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    output_dir = args.output_dir
    logger = setup_logging(output_dir)
    
    cprint("="*60, 'blue')
    cprint("E11: Selective Layer Training", 'blue')
    cprint(f"Models: {args.models}", 'blue')
    cprint(f"Freeze Ratios: {args.freeze_ratios}", 'blue')
    cprint("="*60, 'blue')
    
    all_results = {}
    
    for model_str in args.models:
        family, scale = model_str.split('/')
        model_key = f"{family}_{scale}"
        all_results[model_key] = {}
        
        cprint(f"\n{'='*60}", 'purple')
        cprint(f"Processing: {family}/{scale}", 'purple')
        cprint(f"{'='*60}", 'purple')
        
        # 获取层数
        from model_utils import get_num_layers
        num_layers = get_num_layers(family, scale)
        
        # 准备训练数据（只加载一次）
        cprint("Loading training data...", 'yellow')
        train_dataset = download_dataset(args.datasets[0], DATASET_CACHE_DIR)
        train_sampled = sample_dataset(train_dataset, args.n_train_samples, strategy='random')
        
        for freeze_ratio in args.freeze_ratios:
            cprint(f"\n--- Freeze Ratio: {freeze_ratio:.2f} ---", 'blue')
            
            try:
                # 加载模型
                cprint("Loading model...", 'yellow')
                model, tokenizer = load_model(
                    family, scale, 'base',
                    cache_dir=MODEL_CACHE_DIR,
                    device_map=args.device_map,
                    dtype=args.model_dtype
                )
                
                # 训练
                training_time, num_frozen = train_model(
                    model, tokenizer, train_sampled, 
                    args, freeze_ratio, num_layers
                )
                
                # 评估
                cprint("Evaluating model...", 'yellow')
                eval_results = {}
                for eval_dataset in args.eval_datasets:
                    device = next(model.parameters()).device
                    scores = evaluate_model(
                        model, tokenizer, eval_dataset,
                        args.n_eval_samples, args.max_length, device
                    )
                    eval_results[eval_dataset] = scores
                    cprint(f"{eval_dataset}: Acc={scores['accuracy']:.4f}, PPL={scores['perplexity']:.2f}", 'green')
                
                # 保存结果
                all_results[model_key][freeze_ratio] = {
                    'training_time': training_time,
                    'num_frozen_layers': num_frozen,
                    'total_layers': num_layers,
                    'eval_results': eval_results
                }
                
            except Exception as e:
                cprint(f"❌ Error with freeze_ratio={freeze_ratio}: {e}", 'red')
                logger.error(f"Error: {e}", exc_info=True)
            
            finally:
                # 清理
                if 'model' in locals():
                    del model
                if 'tokenizer' in locals():
                    del tokenizer
                gc.collect()
                torch.cuda.empty_cache()
    
    # 保存汇总
    summary_path = os.path.join(output_dir, 'selective_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 可视化
    visualize_results(all_results, output_dir)
    
    cprint("="*60, 'blue')
    cprint("✅ E11: Selective Layer Training Completed", 'blue')
    cprint(f"Results saved to: {output_dir}", 'blue')
    cprint("="*60, 'blue')

if __name__ == '__main__':
    main()

# base_sft: A Layer-wise Analysis of Supervised Fine-Tuning

This repository provides an end-to-end pipeline to **extract layer-wise representations** from causal LMs and to **quantify representation shifts** between a **base model** and its **aligned variants** (e.g., **SFT / DPO / RLVR / Instruct**). It also includes auxiliary analyses such as **seed robustness checks**, **metric correlation analysis**, **layer swapping**, and **gradient-flow diagnostics**.

The code is designed for  reproducibility: config-driven model/dataset registration, deterministic sampling, cached intermediate artifacts, and standardized CSV/figure outputs.

---

## What this repo does

### Core workflow
1. **Extract representations** (per layer) for a model and dataset:
   - token-level hidden states for a small subset (for sequence metrics),
   - pooled representations for a larger subset (for global metrics).
2. **Compute & visualize representation metrics**:
   - information-theoretic, geometric, and alignment metrics,
   - base vs variant comparisons,
   - PDF plots + CSV reports.

### Additional experiments (optional)
- **Random seed stability** of metrics (02_random_seed_check.py)
- **Metric correlation analysis** (10_metrics_correlation.py)
- **Layer-wise early-exit evaluation** (11_layerwise_probing.py)
- **Layer swapping analysis** between base and aligned models (12_layer_swapping.py)
- **Gradient flow analysis** with optional correlation to representation metrics (13_gradient_flow_analysis.py)
- **Parameter-efficient adapters (LoRA)** for transfer experiments (14_*.py)

---

## Repository structure

```
.
├── 00_extract_representations.py
├── 01_compute_and_visualize_metrics.py
├── 02_random_seed_check.py
├── 10_metrics_correlation.py
├── 11_layerwise_probing.py
├── 12_layer_swapping.py
├── 13_gradient_flow_analysis.py
├── 14_layerwise_adaptive_adapter_7B.py
├── 14_layerwise_lora_adapter_32B.py
├── 14_segment_lora_adapter_13B.py
├── 14_segment_lora_adapter_1B.py
├── config.py
├── data_utils.py
├── metric_utils.py
├── model_utils.py
└── save_utils.py
```

---

## Environment

### Python
- Python **3.9+** (recommended: 3.10)

### Core dependencies
Install the following packages (you may prefer to pin exact versions for archival):
```bash
pip install torch transformers datasets huggingface_hub   numpy pandas scipy tqdm matplotlib seaborn
```

### Optional dependencies
Only needed for specific scripts:
- Adapter training scripts (`14_*.py`): `peft`, `wandb`
- Layer swapping script (`12_layer_swapping.py`): `sympy`

```bash
pip install peft wandb sympy
```

---

## Configuration

### Cache/output directories
This repo uses environment variables (see `config.py`):
- `DATASET_CACHE_DIR` (default: `/xxx/xxx/base_sft/dataset_cache`)
- `MODEL_CACHE_DIR` (default: `/xxx/xxx/base_sft/model_cache`)
- `OUTPUT_DIR` (default: `/xxx/xxx/base_sft/outputs`)

Example:
```bash
export DATASET_CACHE_DIR=/path/to/dataset_cache
export MODEL_CACHE_DIR=/path/to/model_cache
export OUTPUT_DIR=/path/to/outputs
```

Outputs are organized under:
- `outputs/representations/`  (per-sample `.pt` files)
- `outputs/metrics/`          (cached metric tensors, plots, CSV reports)
- `outputs/checkpoints/`      (resume state for extraction)
- plus additional experiment folders (swapping, probing, gradient_flow, …)



## Supported models & variants

Registered in `config.py` under `MODEL_CONFIGS`.

### Model families/scales
- `mistral/7b`  → `mistralai/Mistral-7B-v0.1` (+ instruct as SFT)
- `olmo2/1b`, `olmo2/7b`, `olmo2/13b`, `olmo2/32b`  → `allenai/OLMo-2-*`

### Variants
Depending on availability in `MODEL_CONFIGS`:
- `base`
- `sft`
- `dpo`
- `rlvr`
- `instruct`

---

## Datasets

- `mmlu` (multiple-choice; token-level)
- `gsm8k` (QA; token-level)
- `gsm8kgradient` (QA; train split for gradient experiments)
- `wikitext` (text; token-level)
- `ifeval` (instruction; token-level)
- `humaneval` (code; pooled)
- `mt_bench` (conversation; pooled)
- `toxigen` (text classification; pooled)

> Formatting logic is in `data_utils.py` (`format_sample`) and controlled by each dataset’s `format_type`.

---

## Quickstart

### 0) (Optional) Download models only
```bash
python 00_extract_representations.py   --models olmo2/7b   --dataset mmlu   --variant base sft   --download_only
```

### 1) Extract representations
This generates per-sample `.pt` files with pooled states (and token-level hidden states for the first `N_TOKEN` samples on token-level datasets).

```bash
python 00_extract_representations.py   --models olmo2/7b   --dataset mmlu gsm8k wikitext   --variant base sft   --layer_indices all   --batch_size 16   --model_dtype bfloat16   --save_dtype float16   --resume
```

Layer sampling strategies:
- `all`: all layers
- `key`: roughly every ~1/8 depth + last layer
- `sparse`: {0, 1/4, 1/2, 3/4, last} (deduped)

### 2) Compute & visualize representation metrics
Compute metrics comparing **Base vs Variants** for a dataset. Results are cached in `outputs/metrics/` and exported as PDFs/CSVs.

```bash
python 01_compute_and_visualize_metrics.py   --models olmo2/7b olmo2/13b   --dataset toxigen   --variants sft dpo rlvr instruct   --metrics all   --normalization maxEntropy   --max_samples 1000
```

Metrics supported by `01_compute_and_visualize_metrics.py`:
- `prompt_entropy`, `dataset_entropy`
- `curvature`
- `effective_rank`, `l2_norm`, `spectral_metrics`, `sparsity`
- `cka`, `cosine_similarity`, `mean_shift`

Outputs:
- Plots: `outputs/metrics/plots/<dataset>/*.pdf`
- CSV reports: `outputs/metrics/csv_reports/*.csv`
- Cache tensors: `outputs/metrics/cache_<dataset>_<model>.pt`

---

## Output formats

### Representation files
Each sample is saved as a PyTorch file (`.pt`) at:

```
outputs/representations/{model_family}/{scale}_{variant}/{dataset}/{sample_id:05d}.pt
```

Example:
```
outputs/representations/mistral/7b_base/mmlu/00042.pt
```

Each file contains:
- `sample_id`
- `input_text`
- `input_ids`
- `pooled_states`: `{layer_idx: tensor(D)}`
- `hidden_states`: `{layer_idx: tensor(T, D)}` (only for token-level subset)
- `metadata` (model/dataset identifiers)

---

## Additional scripts (optional)

### Seed robustness
```bash
python 02_random_seed_check.py   --models olmo2/7b   --dataset wikitext   --seeds 42 1234 2024   --metrics prompt_entropy curvature sparsity   --max_samples 100   --layer_indices all
```

### Correlation analysis (metric–metric)
```bash
python 10_metrics_correlation.py   --models olmo2/7b   --dataset mmlu   --variants sft dpo rlvr instruct   --metrics all
```
Outputs:
- `outputs/metrics/plots/correlation_analysis/correlation_matrix_*_enhanced.png`
- `outputs/metrics/plots/correlation_analysis/correlation_clustermap_*_enhanced.png`

### Layer swapping (base ↔ aligned)
```bash
python 12_layer_swapping.py   --models olmo2/7b   --dataset mmlu   --variant sft   --output_dir $OUTPUT_DIR/swapping
```

### Gradient flow analysis
```bash
python 13_gradient_flow_analysis.py   --models olmo2/7b   --datasets gsm8kgradient   --n_samples 5000   --batch_size 8   --gradient_accumulation_steps 8   --learning_rate 2e-5   --num_epochs 0.01   --max_length 512
```

---

## Reproducibility notes

- For deterministic sampling/behavior, control:
  - `--seed` (representation extraction)
  - `--seeds` list (seed check)
- Extraction supports checkpointed resume via `--resume`.
- Large models (e.g., 32B) may require multi-GPU, CPU offload, or high VRAM. Use `--device_map auto` and adjust batch size/dtypes accordingly.

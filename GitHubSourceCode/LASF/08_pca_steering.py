
import os
import argparse
import torch
import logging
from tqdm import tqdm
from glob import glob
from sklearn.decomposition import PCA
import numpy as np

from config import REPRESENTATION_DIR, MODEL_CACHE_DIR
from model_utils import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 5: PCA Subspace Steering")

    # Task Selection
    parser.add_argument('--mode', type=str, default='extract', choices=['extract', 'steer'],
                        help='Step 1: extract vectors, Step 2: steer generation')

    # Model Config
    parser.add_argument('--model_family', type=str, default='olmo2')
    parser.add_argument('--scale', type=str, default='1b')
    parser.add_argument('--dataset', type=str, default='mmlu')

    # Extraction Params
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100)

    # Steering Params
    parser.add_argument('--alpha', type=float, default=1.0, help='Steering strength')
    parser.add_argument('--layer_start', type=int, default=10)
    parser.add_argument('--layer_end', type=int, default=30)
    parser.add_argument('--test_prompt', type=str, default="Question: What is the capital of France?\nAnswer:")

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def extract_vectors(args):
    logger.info("Step 1: Extracting PCA Vectors...")

    # Load Reps
    base_dir = os.path.join(REPRESENTATION_DIR, args.model_family, f"{args.scale}_base", args.dataset)
    sft_dir = os.path.join(REPRESENTATION_DIR, args.model_family, f"{args.scale}_sft", args.dataset)

    files = sorted(glob(os.path.join(base_dir, "*.pt")))[:args.num_samples]

    diff_storage = {} # {layer: [diff_vectors]}

    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        sft_path = os.path.join(sft_dir, fname)
        if not os.path.exists(sft_path): continue

        base_data = torch.load(fpath, map_location='cpu')
        sft_data = torch.load(sft_path, map_location='cpu')

        layers = base_data['hidden_states'].keys()

        for layer in layers:
            # Taking the LAST token difference
            h_base = base_data['hidden_states'][layer][-1, :].float()
            h_sft = sft_data['hidden_states'][layer][-1, :].float()
            diff = h_sft - h_base

            if layer not in diff_storage: diff_storage[layer] = []
            diff_storage[layer].append(diff.numpy())

    # Compute PCA
    pca_vectors = {}
    for layer, diffs in diff_storage.items():
        X = np.stack(diffs) # (N, D)
        pca = PCA(n_components=args.n_components)
        pca.fit(X)
        # Store the first component direction
        # Normalize just in case, though PCA components are unit norm
        comp = torch.tensor(pca.components_[0], dtype=torch.float32)
        pca_vectors[layer] = comp

    save_path = f"pca_vectors_{args.model_family}_{args.scale}.pt"
    torch.save(pca_vectors, save_path)
    logger.info(f"✅ PCA vectors saved to {save_path}")

def steer_model(args):
    logger.info(f"Step 2: Steering Model (Alpha={args.alpha})...")

    # Load Vectors
    vec_path = f"pca_vectors_{args.model_family}_{args.scale}.pt"
    if not os.path.exists(vec_path):
        logger.error("Vectors not found! Run --mode extract first.")
        return
    steering_vectors = torch.load(vec_path, map_location=args.device)

    # Load Base Model
    model, tokenizer = load_model(args.model_family, args.scale, 'base', device_map=args.device)

    # Define Hook
    def steering_hook(module, args_, output):
        # output is usually (batch, seq, hidden)
        # We add vector to the last token position across the sequence?
        # Or add to all positions? usually all positions for simplicity in generation.
        layer_idx = module.layer_idx
        if layer_idx in steering_vectors:
            vec = steering_vectors[layer_idx].to(output.device)
            # Add [1, 1, D] to broadcast
            output = output + args.alpha * vec.view(1, 1, -1)
        return output

    # Register Hooks
    hooks = []
    # Identify layers. This depends on model architecture.
    # For Olmo/Llama, usually model.model.layers
    layers_module = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_module = model.model.layers
    elif hasattr(model, 'layers'):
        layers_module = model.layers

    if layers_module:
        for i, layer in enumerate(layers_module):
            if args.layer_start <= i <= args.layer_end:
                # Monkey patch layer index for the hook to know
                layer.layer_idx = i 
                h = layer.register_forward_hook(steering_hook)
                hooks.append(h)

    logger.info(f"🪝 Registered hooks on layers {args.layer_start}-{args.layer_end}")

    # Generate
    inputs = tokenizer(args.test_prompt, return_tensors="pt").to(args.device)

    logger.info("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False, # Greedy for determinism
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("-" * 40)
    print(f"Prompt: {args.test_prompt}")
    print(f"Result (Alpha={args.alpha}):\n{output_text}")
    print("-" * 40)

    # Remove hooks
    for h in hooks: h.remove()

def main():
    args = parse_args()
    if args.mode == 'extract':
        extract_vectors(args)
    elif args.mode == 'steer':
        steer_model(args)

if __name__ == "__main__":
    main()
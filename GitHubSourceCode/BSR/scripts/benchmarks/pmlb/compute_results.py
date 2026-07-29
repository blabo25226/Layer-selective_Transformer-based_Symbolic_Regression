import pandas as pd
import numpy as np
import torch
import random
import json
import argparse
from pmlb import fetch_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

from download_datasets import compatible_datasets
from preprocess_dataset import binarize_all_features

# Import Boolformer-related modules
from src.formula import create_both_vocabs, Formula
from src.ConfigClasses import ConfigFormula
from src.transformer import LtnTransformer

def get_data_pmlb(dataset, train_ratio=0.75, min_points=30, seed=0, binarize_categorical=True, max_features=100, verbose=False):
    np.random.seed(seed)
    df = fetch_data(dataset, return_X_y=False, local_cache_dir='pmlb_cache')
    outputs, df = df.iloc[:, -1].values, df.drop(df.columns[-1], axis=1)

    if verbose:
        print(f"Dataset: {dataset}, Initial Shape: {df.shape}")

    # Ensure all columns are numeric
    for col in df.columns:
        if df[col].dtype == 'object' or not np.issubdtype(df[col].dtype, np.number):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(axis=1, how='all')  # Drop columns with all NaN values

    if binarize_categorical:
        # Binarize categorical features
        df = binarize_all_features(df, max_unique_values=5)
        if len(df.columns) > max_features:
            print(f"Too many features ({len(df.columns)}), skipping dataset.")
            return None, None, None, None

    else:
        binary_columns = [col for col in df.columns if df[col].nunique() == 2]
        df = df[binary_columns]

    # Ensure sufficient data points
    if len(df) < 1:
        print(f"Dataset {dataset} is empty after preprocessing.")
        return None, None, None, None

    inputs = df.values.astype(float)  # Convert to a consistent numeric type
    shuffle_idx = np.random.permutation(len(inputs))
    inputs, outputs = inputs[shuffle_idx], outputs[shuffle_idx]
    
    # Calculate number of samples and ensure it's non-negative
    n_samples = int(train_ratio * len(inputs))
    if n_samples <= 0 or len(inputs) <= min_points:
        print(f"Dataset {dataset} has insufficient data points for the split.")
        return None, None, None, None

    # Ensure binary outputs
    unique_outputs = np.unique(outputs)
    if len(unique_outputs) != 2:
        print(f"Non-binary outputs in {dataset}, skipping.")
        return None, None, None, None
    output_mapping = {unique_outputs[0]: 0, unique_outputs[1]: 1}
    outputs = np.vectorize(output_mapping.get)(outputs)

    inputs_train, inputs_val = inputs[:n_samples], inputs[n_samples:]
    outputs_train, outputs_val = outputs[:n_samples], outputs[n_samples:]

    return inputs_train, outputs_train, inputs_val, outputs_val


# Function to evaluate models
def evaluate_models(inputs_train, outputs_train, inputs_val, outputs_val, model: LtnTransformer, c_formula, input_vocab, beam, seed, nb_smpl):
    results = {}


    # Random Forest (1 tree)
    clf = RandomForestClassifier(n_estimators=1, random_state=42)
    clf.fit(inputs_train, outputs_train)
    preds = clf.predict(inputs_val)
    results['RandomForest_1'] = {
        'accuracy': accuracy_score(outputs_val, preds),
        'f1_score': f1_score(outputs_val, preds)
    }

    add_to_dataset = f1_score(outputs_val, preds) > 0.75

    if add_to_dataset:
        # Logistic Regression
        clf = LogisticRegression(random_state=42)
        clf.fit(inputs_train, outputs_train)
        preds = clf.predict(inputs_val)
        results['LogisticRegression'] = {
            'accuracy': accuracy_score(outputs_val, preds),
            'f1_score': f1_score(outputs_val, preds)
        }

        # Random Forest (100 trees)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(inputs_train, outputs_train)
        preds = clf.predict(inputs_val)
        results['RandomForest_100'] = {
            'accuracy': accuracy_score(outputs_val, preds),
            'f1_score': f1_score(outputs_val, preds)
        }

        device = model.device
        inputs_train_tensor = torch.tensor(inputs_train, dtype=torch.long).to(device)
        outputs_train_tensor = torch.tensor(outputs_train, dtype=torch.long).to(device)
        inputs_val_tensor = torch.tensor(inputs_val, dtype=torch.long).to(device)
        outputs_val_tensor = torch.tensor(outputs_val, dtype=torch.long).to(device)
        
        full_train_eval = torch.cat([inputs_train_tensor, outputs_train_tensor.unsqueeze(1)], dim=1)
        full_val_eval = torch.cat([inputs_val_tensor, outputs_val_tensor.unsqueeze(1)], dim=1)
        
        sampled_batches = []
        for _ in range(nb_smpl):
            train_nb_points = min(random.randint(250, 299), len(inputs_train_tensor))
            train_indices = torch.randperm(inputs_train_tensor.size(0))[:train_nb_points]
            batch_sample = full_train_eval[train_indices]
            if batch_sample.size(0) > 0:
                sampled_batches.append(batch_sample)
            else:
                print("Warning: Empty batch encountered, skipping.")
        
        if not sampled_batches:
            print("Error: No valid batches created for Boolformer prediction.")
            return results, add_to_dataset
        
        best_formula = None
        best_train_f1 = 0
        
        # Perform batch predictions
        if seed == 0:
            batch_predictions = model.predict(c_formula, sampled_batches, beam=beam, temperature=0.5, use_beam=False)
        else:
            batch_predictions = [model.genetic(c_formula, batch, seed=seed, beam=beam, temperature=0.5) for batch in sampled_batches]
            batch_predictions = [([pred[0]], [pred[1]], pred[2], pred[3]) for pred in batch_predictions]
        
        for formulas, _, _, _ in batch_predictions:
            for formula in formulas:
                if formula is None:
                    continue
                
                try:
                    train_evaluations = formula.evaluate_pts(points=inputs_train_tensor)
                    train_results = train_evaluations[:, -1]
                    train_f1 = f1_score(outputs_train_tensor.cpu().numpy(), train_results.cpu().numpy())
                    
                    if train_f1 > best_train_f1:
                        best_train_f1 = train_f1
                        best_formula = formula
                except Exception as e:
                    print(f"Error in Boolformer training evaluation: {e}")
        
        best_f1_score, best_accuracy = 0, 0
        if best_formula:
            try:
                val_evaluations = best_formula.evaluate_pts(points=inputs_val_tensor)
                val_results = val_evaluations[:, -1]
                best_f1_score = f1_score(outputs_val_tensor.cpu().numpy(), val_results.cpu().numpy())
                best_accuracy = accuracy_score(outputs_val_tensor.cpu().numpy(), val_results.cpu().numpy())
            except Exception as e:
                print(f"Error in Boolformer validation evaluation: {e}")
        
        results['Boolformer'] = {'f1_score': best_f1_score, 'accuracy': best_accuracy}
        print("Result Boolformer =", results['Boolformer'])
    else: 
        print("RF 1 didnt get the score!")

    return results, add_to_dataset



# Main evaluation loop
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Boolformer and baseline models on PMLB datasets.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained Boolformer model checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to the ConfigFormula JSON file.")
    parser.add_argument("--seed", type=int, default=0, help="Number of seeds for the genetic. 0 = predict")
    parser.add_argument("--beam", type=int, default=1, help="Device to use for evaluation (e.g., cuda:0 or cpu).")
    parser.add_argument("--nb_smpl", type=int, default=4, help="Number of set of points selected for the generation.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for evaluation (e.g., cuda:0 or cpu).")
    args = parser.parse_args()

    # Load Boolformer model and config
    c_formula = ConfigFormula(py_config_path=args.config)
    beam = args.beam
    nb_smpl = args.nb_smpl
    seed = args.seed
    model = LtnTransformer.load_from_checkpoint(args.model, map_location=args.device)
    model.eval()
    input_vocab, _ = create_both_vocabs(config=c_formula)

    results_summary = {}

    for i, dataset_name in enumerate(compatible_datasets):
        print(f"\n\nProcessing dataset {i+1}/{len(compatible_datasets)}: {dataset_name}")
        inputs_train, outputs_train, inputs_val, outputs_val = get_data_pmlb(dataset_name)
        if inputs_train is None:
            print(f"Skipping {dataset_name} due to preprocessing issues.")
            continue

        # Evaluate models
        results, add_to_dataset = evaluate_models(inputs_train, outputs_train, inputs_val, outputs_val, model, c_formula, input_vocab, beam, seed, nb_smpl)
            
        if add_to_dataset:
            results_summary[dataset_name] = results            

    # Save results to JSON
    with open(args.output, "w") as json_file:
        json.dump(results_summary, json_file, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

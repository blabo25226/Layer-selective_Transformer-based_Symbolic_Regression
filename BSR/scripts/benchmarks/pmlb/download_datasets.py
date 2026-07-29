from pmlb import dataset_names, fetch_data
import numpy as np
import pandas as pd
import os
import shutil

from preprocess_dataset import binarize_all_features

def get_compatible_datasets(max_features=120, verbose=False):
    compatible_datasets = []
    
    for dataset in dataset_names:
        try:
            # Load the dataset
            df = fetch_data(dataset, return_X_y=False, local_cache_dir='pmlb_cache')
            outputs, df = df.iloc[:, -1].values, df.drop(df.columns[-1], axis=1)
            
            # Ensure binary output
            if set(np.unique(outputs)) != {0, 1}:
                values = np.unique(outputs)
                if len(values) != 2:
                    if verbose:
                        print(f"Skipping {dataset}: Non-binary output")
                    continue
            
            # Binarize features and check count
            df = binarize_all_features(df)
            if len(df.columns) > max_features:
                if verbose:
                    print(f"Skipping {dataset}: Too many features after binarization ({len(df.columns)} features)")
                continue
            
            # Add to compatible datasets
            compatible_datasets.append(dataset)
            if verbose:
                print(f"Compatible dataset found: {dataset}")
        
        except Exception as e:
            if verbose:
                print(f"Error processing {dataset}: {e}")
    
    return compatible_datasets

compatible_datasets = [
    'agaricus_lepiota',
    'analcatdata_asbestos',
    'australian',
    'buggyCrx',
    'car_evaluation',
    'chess',
    'colic',
    'corral',
    'credit_a',
    'credit_g',
    'crx',
    'heart_c',
    'heart_h',
    'heart_statlog',
    'house_votes_84',
    'hungarian',
    'irish',
    'kr_vs_kp',
    'labor',
    'mofn_3_7_10',
    'monk1',
    'monk2',
    'monk3',
    'mushroom',
    'mux6',
    'parity5',
    'parity5+5',
    'postoperative_patient_data',
    'prnn_crabs',
    'profb',
    'saheart',
    'schizo',
    'spect',
    'threeOf9',
    'tic_tac_toe',
    'tokyo1',
    'vote',
    'xd6',
    'titanic'
    ]

if __name__ == "__main__":
    datasets = get_compatible_datasets(verbose=True)
    # Path to the local cache directory
    cache_dir = 'pmlb_cache'

    # Ensure the cache directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Cache compatible datasets
    for dataset in datasets:
        try:
            fetch_data(dataset,
             return_X_y=False,
             local_cache_dir=cache_dir)
            print(f"Cached dataset: {dataset}")
        except Exception as e:
            print(f"Failed to cache dataset {dataset}: {e}")

    # Remove incompatible datasets
    for dataset_dir in os.listdir(cache_dir):
        dataset_path = os.path.join(cache_dir,
         dataset_dir)
        if dataset_dir not in datasets:
            try:
                if os.path.isdir(dataset_path):
                    shutil.rmtree(dataset_path)
                else:
                    os.remove(dataset_path)
                print(f"Removed incompatible dataset: {dataset_dir}")
            except Exception as e:
                print(f"Failed to remove {dataset_dir}: {e}")
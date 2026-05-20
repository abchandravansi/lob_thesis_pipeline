import os
import glob
import numpy as np
import torch

def load_fi2010_file(file_path):
    """
    Parses a single FI-2010 raw text file.
    Rows 0-39: Spatial structural features (prices and volumes).
    Rows 144-148: Multi-horizon labels (k=1, 2, 3, 5, 10).
    """
    # Load matrix and transpose because data is stored columns-as-ticks
    data = np.loadtxt(file_path)
    features = data[:40, :].T
    
    # Extract the 5 prediction horizons (last 5 rows)
    # Convert labels from 1,2,3 format to 0,1,2 standard indices
    labels = data[-5:, :].T - 1
    
    return features, labels

def get_fi2010_data(data_dir="data", split="train"):
    """
    Scans folders and aggregates text logs into combined training/testing tensors.
    """
    base_path = os.path.join(data_dir, "BenchmarkDatasets", "NoAuction")
    
    if split == "train":
        search_path = os.path.join(base_path, "1.NoAuction_Zscore\\NoAuction_Zscore_Training", "*.txt")
    else:
        search_path = os.path.join(base_path, "1.NoAuction_Zscore\\NoAuction_Zscore_Testing", "*.txt")
        
    file_list = glob.glob(search_path)
    if not file_list:
        raise FileNotFoundError(f"No FI-2010 text logs discovered matching: {search_path}. Please execute download_data.py first.")
        
    all_features = []
    all_labels = []
    
    for f in sorted(file_list):
        feats, labs = load_fi2010_file(f)
        all_features.append(feats)
        all_labels.append(labs)
        
    # Concatenate tracking chunks across days
    X = np.vstack(all_features)
    Y = np.vstack(all_labels)
    
    # Stationary Z-score normalization across columns
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 # Protect against zero variance division errors
    X = (X - mean) / std
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

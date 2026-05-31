# Save this file inside your repository as: src/utils.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LOBDataset(Dataset):
    """Custom wrapper to format high-frequency order books into structural blocks."""
    def __init__(self, data_path, horizons=[1, 2, 3, 5, 10]):
        self.horizons = horizons
        print(f"Parsing raw text datasets from path: {data_path}...")
        
        # Load raw text matrix (adjust delimiter if your text uses commas instead of spaces)
        raw_data = np.loadtxt(data_path)
        
        # Extract features (FI-2010 uses the first 40 columns for LOB feature points)
        self.features = raw_data[:, :40].astype(np.float32)
        
        # Extract multiple target labels (FI-2010 maps multi-horizons to the last few columns)
        # Note: Adjust column indexes if your custom file structure changes them
        self.labels = raw_data[:, -len(horizons):].astype(np.int64) - 1  # Map 1,2,3 values to 0,1,2 index positions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Format feature arrays as PyTorch Tensors
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_lob_dataloader(data_path, batch_size=128, shuffle=False):
    """Initializes optimized operational memory streams."""
    dataset = LOBDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

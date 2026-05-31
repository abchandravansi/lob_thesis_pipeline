import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import importlib.util
import glob

# 1. ESTABLISH EXACT PATH TO LOB_MODELS.PY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE_PATH = os.path.join(BASE_DIR, "src", "models", "lob_models.py")

if not os.path.exists(MODELS_FILE_PATH):
    print(f"\n❌ ERROR: Cannot find your model definition script at the expected path:")
    print(f"👉 Checked location: {MODELS_FILE_PATH}")
    sys.exit(1)

print(f"🔍 Successfully targeted model definitions at: {MODELS_FILE_PATH}")

# 2. FORCE PYTHON TO IMPORT DIRECTLY FROM THAT VERIFIED TARGET FILE
spec = importlib.util.spec_from_file_location("local_lob_models", MODELS_FILE_PATH)
local_models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_models_module)

DeepLOB = local_models_module.DeepLOB
LOBTransformer = local_models_module.LOBTransformer
LOBCAST = local_models_module.LOBCAST
MambaLOB = local_models_module.MambaLOB
print("✅ Success! Local LOB model architectures extracted without conflict.")

# =====================================================================
# 🔥 ULTIMATE CONVOLUTION INTERCEPT PATCH
# =====================================================================
class FoolproofConv2dWrapper(nn.Module):
    """
    Intercepts the first convolutional layer of the model directly.
    Forces ANY 5D tensor down to a clean 4D batched tensor right at impact.
    """
    def __init__(self, original_conv):
        super().__init__()
        self.original_conv = original_conv

    def forward(self, x):
        if x.dim() == 5:
            if x.size(1) == 1 and x.size(2) == 1:
                x = x.squeeze(2)
            else:
                x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
                if x.size(1) > 1:
                    x = x[:, :1, :, :]
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return self.original_conv(x)

def patch_model_layers(model_instance):
    """Finds the first Conv2d layer in the architecture dynamically and wraps it."""
    for name, module in model_instance.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model_instance.get_submodule(parent_name) if parent_name else model_instance
            
            setattr(parent, layer_name, FoolproofConv2dWrapper(module))
            print(f"🎯 Successfully wrapped layer: {name}")
            break 
    return model_instance

print("🛠️ Deep interception wrappers initialized successfully.")

# =====================================================================
# 📦 ROLLING SEQUENTIAL MULTI-FILE DATA LOADING MECHANISM
# =====================================================================
class FI2010RollingDataset(Dataset):
    def __init__(self, folder_path, horizons=5, T=100):
        self.T = T  
        print(f"Scanning folder for dataset chunks: {folder_path}")
        
        search_path = os.path.join(folder_path, "*.txt")
        file_list = sorted(glob.glob(search_path))
        
        if not file_list:
            raise FileNotFoundError(f"❌ No text files found inside matching pattern: {search_path}")
            
        all_features = []
        all_labels = []
        
        for file_path in file_list:
            print(f" 📂 Loading segment: {os.path.basename(file_path)}...")
            raw_data = np.loadtxt(file_path)
            
            features = raw_data[:, :40].astype(np.float32)
            labels = raw_data[:, -horizons:].astype(np.int64)
            
            all_features.append(features)
            all_labels.append(labels)
            
        raw_features = np.vstack(all_features)
        raw_labels = np.vstack(all_labels)
        
        n_samples = len(raw_features) - T + 1
        if n_samples <= 0:
            raise ValueError(f"Dataset is too small to form a rolling window of size T={T}.")
            
        self.features = np.zeros((n_samples, T, 40), dtype=np.float32)
        self.labels = np.zeros((n_samples, horizons), dtype=np.int64)
        
        for i in range(n_samples):
            self.features[i] = raw_features[i : i + T, :]
            self.labels[i] = raw_labels[i + T - 1, :]
            
        print(f"✅ Dataset matrix created. Base features array shape: {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

def get_clean_loaders(train_folder, val_folder, batch_size=128):
    # FIXED: Configured to T=100 to meet structural constraints for Transformers
    train_dataset = FI2010RollingDataset(train_folder, T=100)
    val_dataset = FI2010RollingDataset(val_folder, T=100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader

# =====================================================================
# 🏗️ MODEL TRAINING ROUTINE (NO DOWNSTREAM TESTING)
# =====================================================================
def get_model_by_name(name):
    mapping = {"DeepLOB": DeepLOB, "LOBCAST": LOBCAST, "LOBTransformer": LOBTransformer, "MambaLOB": MambaLOB}
    model = mapping[name]()
    return patch_model_layers(model)

def train_and_export_pipeline(epochs, use_amp):
    print("🚀 Pipeline Active! Validating system data path targets...")
    
    train_folder = os.path.join(BASE_DIR, "data", "BenchmarkDatasets", "NoAuction", "1.NoAuction_Zscore", "NoAuction_Zscore_Training")
    val_folder = os.path.join(BASE_DIR, "data", "BenchmarkDatasets", "NoAuction", "1.NoAuction_Zscore", "NoAuction_Zscore_Testing")
    save_dir = os.path.join(BASE_DIR, "saved_lob_models")
    
    if not os.path.exists(train_folder):
        print(f"❌ Critical Path Error: Target folder missing at: {train_folder}")
        return
        
    train_loader, val_loader = get_clean_loaders(train_folder, val_folder, batch_size=128)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("ℹ️ CPU Execution Target selected. Disabling AMP (Mixed Precision) to prevent runtime overhead.")
        use_amp = False
        
    print(f"💻 Computational Processing Target Hardware: {device}")
    
    model_names = ["DeepLOB", "LOBCAST", "LOBTransformer", "MambaLOB"]
    
    for name in model_names:
        print(f"\n" + "="*50)
        print(f"=== Model Training Initialization: {name} ===")
        print("="*50)
        
        model = get_model_by_name(name).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        scaler = torch.amp.GradScaler(enabled=use_amp)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                y_batch = torch.clamp(y_batch - 1, min=0, max=2)

                optimizer.zero_grad()
                
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                    outputs = model(x_batch)
                    
                    if isinstance(outputs, list):
                        loss = 0.0
                        for idx, horizon_output in enumerate(outputs):
                            loss += criterion(horizon_output, y_batch[:, idx])
                    elif outputs.dim() == 3: 
                        loss = criterion(outputs.view(-1, outputs.size(-1)), y_batch.view(-1))
                    else: 
                        loss = criterion(outputs, y_batch[:, 0])
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                    
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | Train Loss: {running_loss / len(train_loader):.4f}")
            
            # --- Checkpoint Export ---
            if (epoch + 1) == epochs: 
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"{name}_best.pt")
                
                for m_name, module in list(model.named_modules()):
                    if isinstance(module, FoolproofConv2dWrapper):
                        p_name = ".".join(m_name.split(".")[:-1])
                        l_name = m_name.split(".")[-1]
                        p_mod = model.get_submodule(p_name) if p_name else model
                        setattr(p_mod, l_name, module.original_conv)
                    
                torch.save({'model_name': name, 'model_state_dict': model.state_dict(), 'epoch': epoch + 1}, checkpoint_path)
                print(f"💾 Checkpoint saved for {name} -> {checkpoint_path}")
        
        del model
        torch.cuda.empty_cache()

# =====================================================================
# ⚡ ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("🎮 Editor Run Triggered! Initializing model training pipeline...")
    
    # Run loop strictly on CPU resources with 5 epoch loops
    train_and_export_pipeline(epochs=5, use_amp=False)
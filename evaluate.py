import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import importlib.util
import glob
from sklearn.metrics import accuracy_score, f1_score

# 1. ESTABLISH PATH TO YOUR LOCAL MODEL ARCHITECTURES
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FILE_PATH = os.path.join(BASE_DIR, "src", "models", "lob_models.py")

if not os.path.exists(MODELS_FILE_PATH):
    print(f"\n❌ ERROR: Cannot find your model definition script at the expected path:")
    print(f"👉 Checked location: {MODELS_FILE_PATH}")
    sys.exit(1)

# Force Python to import directly from that verified target file
spec = importlib.util.spec_from_file_location("local_lob_models", MODELS_FILE_PATH)
local_models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_models_module)

DeepLOB = local_models_module.DeepLOB
LOBTransformer = local_models_module.LOBTransformer
LOBCAST = local_models_module.LOBCAST
MambaLOB = local_models_module.MambaLOB

# =====================================================================
# 🔥 ULTIMATE CONVOLUTION INTERCEPT PATCH
# =====================================================================
class FoolproofConv2dWrapper(nn.Module):
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
    for name, module in model_instance.named_modules():
        if isinstance(module, nn.Conv2d):
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model_instance.get_submodule(parent_name) if parent_name else model_instance
            setattr(parent, layer_name, FoolproofConv2dWrapper(module))
            break 
    return model_instance

# =====================================================================
# 📦 ROLLING SEQUENTIAL MULTI-FILE DATA LOADING MECHANISM
# =====================================================================
class FI2010RollingTestDataset(Dataset):
    def __init__(self, folder_path, horizons=5, T=100):
        self.T = T  
        print(f"Scanning testing folder for dataset chunks: {folder_path}")
        
        search_path = os.path.join(folder_path, "*.txt")
        file_list = sorted(glob.glob(search_path))
        
        if not file_list:
            raise FileNotFoundError(f"❌ No test text files found inside matching pattern: {search_path}")
            
        all_features = []
        all_labels = []
        
        for file_path in file_list:
            print(f" 📂 Loading test segment: {os.path.basename(file_path)}...")
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
            
        print(f"Testing dataset created. Matrix Shape: {self.features.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# =====================================================================
# 📊 PERFORMANCE REPORTING PIPELINE
# =====================================================================
def load_saved_checkpoint(name, checkpoint_dir, device):
    mapping = {"DeepLOB": DeepLOB, "LOBCAST": LOBCAST, "LOBTransformer": LOBTransformer, "MambaLOB": MambaLOB}
    
    ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Warning: Checkpoint asset missing for {name} at {ckpt_path}. Skipping.")
        return None
        
    model = mapping[name]()
    
    # FIXED: Load standard state dictionary parameters BEFORE patching the layer architecture shape
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply the dynamic layer wrapping interceptor after parameters are injected
    model = patch_model_layers(model)
    
    model.to(device)
    model.eval()
    return model

def execute_unseen_evaluation_report():
    print("📋 Testing Pipeline Active! Loading validation text profiles...")
    
    test_folder = os.path.join(BASE_DIR, "data", "BenchmarkDatasets", "NoAuction", "1.NoAuction_Zscore", "NoAuction_Zscore_Testing")
    checkpoint_dir = os.path.join(BASE_DIR, "saved_lob_models")
    
    if not os.path.exists(test_folder):
        print(f"❌ Critical Path Error: Testing folder missing at: {test_folder}")
        return
        
    test_dataset = FI2010RollingTestDataset(test_folder, T=100)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)
    
    device = torch.device("cpu")
    print(f"💻 Deployment Engine: {device}")
    
    model_names = ["DeepLOB", "LOBCAST", "LOBTransformer", "MambaLOB"]
    criterion = torch.nn.CrossEntropyLoss()
    report_card = {}

    for name in model_names:
        model = load_saved_checkpoint(name, checkpoint_dir, device)
        if model is None:
            continue
            
        print(f"⚙️ Running out-of-sample inference sweep for: {name}...")
        
        horizons_preds = {k: [] for k in range(5)}
        horizons_targets = {k: [] for k in range(5)}
        total_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_batch = torch.clamp(y_batch - 1, min=0, max=2)
                outputs = model(x_batch)
                
                if isinstance(outputs, list):
                    loss = 0.0
                    for idx, horizon_output in enumerate(outputs):
                        loss += criterion(horizon_output, y_batch[:, idx])
                        preds = torch.argmax(horizon_output, dim=-1)
                        horizons_preds[idx].extend(preds.cpu().numpy())
                        horizons_targets[idx].extend(y_batch[:, idx].cpu().numpy())
                    total_loss += (loss / len(outputs)).item()
                else:
                    if outputs.dim() == 3:
                        loss = criterion(outputs.view(-1, outputs.size(-1)), y_batch.view(-1))
                        total_loss += loss.item()
                        for idx in range(5):
                            preds = torch.argmax(outputs[:, idx, :], dim=-1)
                            horizons_preds[idx].extend(preds.cpu().numpy())
                            horizons_targets[idx].extend(y_batch[:, idx].cpu().numpy())
                    else:
                        loss = criterion(outputs, y_batch[:, 0])
                        total_loss += loss.item()
                        preds = torch.argmax(outputs, dim=-1)
                        horizons_preds.extend(preds.cpu().numpy())
                        horizons_targets.extend(y_batch[:, 0].cpu().numpy())

        report_card[name] = {
            "Avg_Loss": total_loss / len(test_loader),
            "Horizons": {}
        }
        
        for idx in range(5):
            if len(horizons_preds[idx]) == 0:
                continue
            acc = accuracy_score(horizons_targets[idx], horizons_preds[idx])
            f1 = f1_score(horizons_targets[idx], horizons_preds[idx], average='macro', zero_division=0)
            report_card[name]["Horizons"][idx + 1] = {"Accuracy": acc, "Macro_F1": f1}

    print("\n" + "="*75)
    print("🏆 FINAL OUT-OF-SAMPLE BENCHMARK REPORT (UNSEEN TESTING DATA) 🏆")
    print("="*75)
    
    for name, metrics in report_card.items():
        print(f"\n🧱 Model Architecture: {name}")
        print(f"📉 Combined Test Evaluation Loss: {metrics['Avg_Loss']:.4f}")
        print(f"| Evaluation Horizon (k) | Out-of-Sample Accuracy | Macro F1-Score |")
        print(f"|------------------------|------------------------|----------------|")
        for k, h_metrics in metrics["Horizons"].items():
            print(f"| Target k = {k:02d}          | {h_metrics['Accuracy']:.4f}                 | {h_metrics['Macro_F1']:.4f}         |")
    print("="*75)

if __name__ == "__main__":
    execute_unseen_evaluation_report()

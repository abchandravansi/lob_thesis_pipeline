import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.parser import get_fi2010_data
from src.dataset import MultiHorizonFI2010Dataset
from src.models.lob_models import DeepLOB, LOBCAST, LOBTransformer, MambaLOB
from src.engine import train_one_epoch, evaluate_pipeline

def run_benchmarks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Real FI-2010 Benchmark Pipeline on Windows: {device}\n")
    
    # 1. Parse and extract the real text logs
    try:
        print("Parsing raw FI-2010 text datasets into memory blocks...")
        train_x, train_y = get_fi2010_data(data_dir="data", split="train")
        test_x, test_y = get_fi2010_data(data_dir="data", split="test")
        print(f"Successfully loaded. Train steps: {train_x.shape[0]} | Test steps: {test_x.shape[0]}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please check that your 'data' folder contains the extracted text files.")
        return

    # 2. Package into sequence loader generators
    train_dataset = MultiHorizonFI2010Dataset(train_x, train_y, T=100)
    test_dataset = MultiHorizonFI2010Dataset(test_x, test_y, T=100)
    
    # Large batch size helps process the extensive market logs efficiently
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model_suite = {
        "DeepLOB": DeepLOB(),
        "LOBCAST": LOBCAST(),
        "LOBTransformer": LOBTransformer(T=100),
        "MambaLOB": MambaLOB()
    }
    
    criterion = nn.CrossEntropyLoss()
    horizons = [1, 2, 3, 5, 10]
    
    for name, model in model_suite.items():
        print(f"\n=== Model Training Initialization: {name} ===")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test 1 baseline verification epoch across the raw financial sequence
        avg_loss, latency = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {avg_loss:.4f} | Complete Epoch Latency: {latency:.2f}s")
        
        # Evaluate model prediction stability across test logs
        metrics = evaluate_pipeline(model, test_loader, device, horizons)
        for h, scored in metrics.items():
            print(f"  -> Test Horizon k={h:02d} | Accuracy: {scored['Accuracy']:.4f} | Macro F1: {scored['Macro_F1']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    run_benchmarks()

import time
import torch
from sklearn.metrics import f1_score, accuracy_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        
        joint_loss = 0.0
        for i in range(len(outputs)):
            joint_loss += criterion(outputs[i], y_batch[:, i])
            
        joint_loss = joint_loss / len(outputs)
        joint_loss.backward()
        optimizer.step()
        total_loss += joint_loss.item()
        
    return total_loss / len(dataloader), time.time() - start_time

@torch.no_grad()
def evaluate_pipeline(model, dataloader, device, horizons=[1, 2, 3, 5, 10]):
    model.eval()
    trackers = {i: {"preds": [], "labels": []} for i in range(len(horizons))}
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        
        for i in range(len(horizons)):
            preds = torch.argmax(outputs[i], dim=1).cpu().numpy()
            labels = y_batch[:, i].numpy()
            trackers[i]["preds"].extend(preds)
            trackers[i]["labels"].extend(labels)
            
    results = {}
    for i, h in enumerate(horizons):
        acc = accuracy_score(trackers[i]["labels"], trackers[i]["preds"])
        f1 = f1_score(trackers[i]["labels"], trackers[i]["preds"], average='macro')
        results[h] = {"Accuracy": acc, "Macro_F1": f1}
    return results
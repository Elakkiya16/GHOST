import torch
import torch.nn.functional as F

def run_tramer_extraction(target_model, query_loader, test_loader, device):
    """Tramer et al. Model Stealing (Shadow Model Copying)"""
    # 1. Student/Clone model — 2-conv with padding=1 to preserve spatial dims (Section IV-C)
    clone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(), torch.nn.Linear(64 * 8 * 8, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(clone.parameters(), lr=1e-3)
    target_model.eval()
    
    for _ in range(5):
        for x, _ in query_loader:
            x = x.to(device)
            with torch.no_grad():
                soft_labels = F.softmax(target_model(x), dim=1)
            
            optimizer.zero_grad()
            clone_out = F.log_softmax(clone(x), dim=1)
            # Use KL Divergence to match the target's output distribution
            loss = torch.nn.KLDivLoss(reduction='batchmean')(clone_out, soft_labels)
            loss.backward()
            optimizer.step()
            
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            clone_p = clone(x).argmax(dim=1)
            # Ground-truth accuracy (Table III Ext%), not agreement with target
            correct += (clone_p == y).sum().item()
            total += x.size(0)

    return (correct / total) * 100

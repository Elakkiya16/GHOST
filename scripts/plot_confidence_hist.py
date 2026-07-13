"""
Softmax confidence histograms over 10,000 unseen CIFAR-10 test samples (Figure 2).
Shows GHOST pushes confidence distributions toward uniform vs Baseline's peaked confidence.
Requires: hybrid_cnn2.pth and baseline_cnn2.pth (from scripts/train.py).
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_map(path):
    with open(path) as f:
        return json.load(f)

def get_max_confidences(model, loader, is_hybrid=False, key=0):
    model.eval()
    confs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            logits = model(x, key=key) if is_hybrid else model(x)
            probs = F.softmax(logits, dim=1)
            confs.extend(probs.max(dim=1).values.cpu().numpy())
    return np.array(confs)

def main():
    map_file = 'shuffle_map.json'
    if not os.path.exists(map_file):
        print(f"ERROR: {map_file} not found. Run scripts/train.py first.")
        return
    mapping = load_map(map_file)

    def shuffle_img(img, m):
        c,h,w = img.shape
        return img.view(c,-1)[:,m].view(c,h,w)

    tf_ghost = transforms.Compose([
        transforms.Resize((32,32)), transforms.ToTensor(),
        transforms.Lambda(lambda x: shuffle_img(x, mapping)),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tf_base = transforms.Compose([
        transforms.Resize((32,32)), transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    loader_g = DataLoader(datasets.CIFAR10('./data', False, transform=tf_ghost, download=True), batch_size=256, shuffle=False)
    loader_b = DataLoader(datasets.CIFAR10('./data', False, transform=tf_base,  download=False), batch_size=256, shuffle=False)

    # Minimal model stubs (must match train.py architecture)
    class _Unshuffle(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.register_buffer('inv', torch.argsort(torch.tensor(m, dtype=torch.long)))
        def forward(self, x):
            B,C,H,W=x.shape
            return x.view(B,C,-1)[:,:,self.inv].view(B,C,H,W)

    class HybridCNN(nn.Module):
        def __init__(self, fake_layers=4):
            super().__init__()
            self.unshuffle=_Unshuffle(mapping)
            self.conv1=nn.Conv2d(3,32,3,padding=1); self.bn1=nn.BatchNorm2d(32)
            self.conv2=nn.Conv2d(32,64,3,padding=1); self.bn2=nn.BatchNorm2d(64)
            self.pool=nn.AvgPool2d(2,2)
            self.fakes=nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU()) for _ in range(fake_layers)])
            self.conv3=nn.Conv2d(64,128,3,padding=1); self.bn3=nn.BatchNorm2d(128)
            self.fc1=nn.Linear(128*8*8,256); self.dropout=nn.Dropout(0.5); self.fc2=nn.Linear(256,10)
        def forward(self, x, key=0):
            x=self.unshuffle(x)
            x=F.relu(self.bn1(self.conv1(x)))
            x=self.pool(F.relu(self.bn2(self.conv2(x))))
            x=self.fakes[key%len(self.fakes)](x)
            x=self.pool(F.relu(self.bn3(self.conv3(x))))
            x=x.view(x.size(0),-1)
            return self.fc2(self.dropout(F.relu(self.fc1(x))))

    class BaselineCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1=nn.Conv2d(3,32,3,padding=1); self.bn1=nn.BatchNorm2d(32)
            self.pool=nn.AvgPool2d(2,2)
            self.conv2=nn.Conv2d(32,64,3,padding=1); self.bn2=nn.BatchNorm2d(64)
            self.fc1=nn.Linear(64*8*8,256); self.dropout=nn.Dropout(0.5); self.fc2=nn.Linear(256,10)
        def forward(self, x):
            x=self.pool(F.relu(self.bn1(self.conv1(x))))
            x=self.pool(F.relu(self.bn2(self.conv2(x))))
            x=x.view(x.size(0),-1)
            return self.fc2(self.dropout(F.relu(self.fc1(x))))

    ghost_model = HybridCNN().to(DEVICE)
    base_model  = BaselineCNN().to(DEVICE)
    ghost_model.load_state_dict(torch.load('hybrid_cnn2.pth',   map_location=DEVICE))
    base_model.load_state_dict(torch.load('baseline_cnn2.pth', map_location=DEVICE))

    print("Computing confidence distributions over 10,000 test samples...")
    ghost_confs = get_max_confidences(ghost_model, loader_g, is_hybrid=True)
    base_confs  = get_max_confidences(base_model,  loader_b, is_hybrid=False)
    print(f"Baseline mean confidence: {base_confs.mean():.4f}")
    print(f"GHOST mean confidence:    {ghost_confs.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    axes[0].hist(base_confs,  bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_title('Baseline — peaked near 1.0', fontsize=11)
    axes[0].set_xlabel('Max softmax confidence')
    axes[0].set_ylabel('Count')
    axes[1].hist(ghost_confs, bins=bins, color='darkorange', edgecolor='white', alpha=0.85)
    axes[1].set_title('GHOST — flattened / disrupted', fontsize=11)
    axes[1].set_xlabel('Max softmax confidence')
    for ax in axes:
        ax.axvline(0.1, color='red', linestyle='--', linewidth=0.8, label='random (0.1)')
        ax.legend(fontsize=8)
    plt.suptitle('Softmax Confidence Distributions — 10,000 CIFAR-10 test samples (Figure 2)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = 'confidence_histograms.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()

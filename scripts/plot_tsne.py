"""
t-SNE embedding visualization comparing GHOST vs Baseline feature spaces (Figure 3).
Requires trained models: hybrid_cnn2.pth and baseline_cnn2.pth (produced by scripts/train.py).
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SAMPLES = 2000
PERPLEXITY = 30
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

def load_map(path):
    with open(path) as f:
        return json.load(f)

def shuffle_image(img, mapping):
    c, h, w = img.shape
    return img.view(c, -1)[:, mapping].view(c, h, w)

def extract_penultimate(model, loader, n_samples, is_hybrid=False, key=0):
    """Return (embeddings, labels) from the layer just before the final FC."""
    feats, labs = [], []
    model.eval()
    hooks = []
    captured = {}

    def make_hook(name):
        def h(m, inp, out):
            captured[name] = out.detach().cpu()
        return h

    # Identify the penultimate activation
    if hasattr(model, 'fc1'):
        hooks.append(model.fc1.register_forward_hook(make_hook('feat')))
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        hooks.append(model.classifier[-2].register_forward_hook(make_hook('feat')))
    else:
        hooks.append(list(model.modules())[-2].register_forward_hook(make_hook('feat')))

    with torch.no_grad():
        for x, y in loader:
            if len(feats) * loader.batch_size >= n_samples:
                break
            x = x.to(DEVICE)
            if is_hybrid:
                model(x, key=key)
            else:
                model(x)
            feats.append(captured['feat'])
            labs.append(y)

    for h in hooks:
        h.remove()

    feats = torch.cat(feats)[:n_samples].numpy()
    labs  = torch.cat(labs)[:n_samples].numpy()
    return feats, labs


def plot_tsne(ax, embeddings, labels, title):
    reducer = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=42)
    proj = reducer.fit_transform(embeddings)
    cmap = plt.cm.get_cmap('tab10', 10)
    for cls_idx in range(10):
        mask = labels == cls_idx
        ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6,
                   color=cmap(cls_idx), label=CIFAR10_CLASSES[cls_idx])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    map_file = 'shuffle_map.json'
    if not os.path.exists(map_file):
        print(f"ERROR: {map_file} not found. Run scripts/train.py first.")
        return
    mapping = load_map(map_file)

    tf_ghost = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Lambda(lambda x: shuffle_image(x, mapping)),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tf_base = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    cifar_ghost = Subset(datasets.CIFAR10('./data', False, transform=tf_ghost, download=True), range(N_SAMPLES))
    cifar_base  = Subset(datasets.CIFAR10('./data', False, transform=tf_base,  download=False), range(N_SAMPLES))
    loader_g = DataLoader(cifar_ghost, batch_size=256, shuffle=False)
    loader_b = DataLoader(cifar_base,  batch_size=256, shuffle=False)

    # Minimal model definitions matching train.py
    class _Unshuffle(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.register_buffer('inv', torch.argsort(torch.tensor(m, dtype=torch.long)))
        def forward(self, x):
            B,C,H,W = x.shape
            return x.view(B,C,-1)[:,:,self.inv].view(B,C,H,W)

    class HybridCNN(nn.Module):
        def __init__(self, fake_layers=4):
            super().__init__()
            self.unshuffle = _Unshuffle(mapping)
            self.conv1 = nn.Conv2d(3,32,3,padding=1); self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32,64,3,padding=1); self.bn2 = nn.BatchNorm2d(64)
            self.pool  = nn.AvgPool2d(2,2)
            self.fakes = nn.ModuleList([nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU()) for _ in range(fake_layers)])
            self.conv3 = nn.Conv2d(64,128,3,padding=1); self.bn3 = nn.BatchNorm2d(128)
            self.fc1 = nn.Linear(128*8*8,256); self.dropout = nn.Dropout(0.5); self.fc2 = nn.Linear(256,10)
        def forward(self, x, key=0):
            x = self.unshuffle(x)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.fakes[key % len(self.fakes)](x)
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0),-1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x)

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

    print("Extracting GHOST embeddings...")
    g_feats, g_labs = extract_penultimate(ghost_model, loader_g, N_SAMPLES, is_hybrid=True)
    print("Extracting Baseline embeddings...")
    b_feats, b_labs = extract_penultimate(base_model,  loader_b, N_SAMPLES, is_hybrid=False)

    print("Running t-SNE (this may take ~1-2 min)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_tsne(axes[0], b_feats, b_labs, 'Baseline — well-separated clusters')
    plot_tsne(axes[1], g_feats, g_labs, 'GHOST — disrupted class-wise compactness')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8, frameon=False)
    plt.suptitle('t-SNE Feature Space (Figure 3)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = 'tsne_comparison.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")

if __name__ == '__main__':
    main()

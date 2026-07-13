"""
Permutation granularity sweep on ResNet-18 / CIFAR-100 (Table VII, Section IV-F).

Spatial grid sizes evaluated: None (no perm), 2×2, 4×4, 8×8, pixel-level (32×32).
Metrics: test accuracy, MIA AUC, extraction %, latency (ms).
Results written to outputs/perm_granularity.csv.
"""
import os, sys, csv, time, hashlib, secrets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ghost.utils import Unshuffle, TokenGate, get_mapping, measure_latency

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100; BATCH = 128; LR = 1e-3; NUM_CLASSES = 100
MAP_FILE = 'ghost_map_32.json'
OUT_CSV  = os.path.join('outputs', 'perm_granularity.csv')
os.makedirs('outputs', exist_ok=True)

TF = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,padding=4),
                          transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
TF_TEST = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])


def block_perm(img, grid_size, mapping_cache, rng_seed=42):
    """Permute grid_size×grid_size blocks within a 32×32 image."""
    if grid_size is None:
        return img
    C, H, W = img.shape
    g = grid_size
    n_blocks_h, n_blocks_w = H // g, W // g
    key = (grid_size, H, W)
    if key not in mapping_cache:
        n_blocks = n_blocks_h * n_blocks_w
        rng = torch.Generator(); rng.manual_seed(rng_seed)
        mapping_cache[key] = torch.randperm(n_blocks, generator=rng)
    perm = mapping_cache[key]
    blocks = img.unfold(1, g, g).unfold(2, g, g)  # C, nh, nw, g, g
    blocks = blocks.contiguous().view(C, -1, g, g)
    blocks = blocks[:, perm]
    return blocks.view(C, n_blocks_h, n_blocks_w, g, g).permute(0,1,3,2,4).contiguous().view(C, H, W)


def _sub(m, path):
    """Traverse dotted attribute path, handling integer indices."""
    for p in path.split('.'):
        try:
            m = m[int(p)]
        except (ValueError, TypeError):
            m = getattr(m, p)
    return m

def _batch_block_perm(x: torch.Tensor, grid_size, cache) -> torch.Tensor:
    """Apply block_perm to each sample in a (B,C,H,W) batch."""
    if grid_size is None:
        return x
    device = x.device
    return torch.stack([block_perm(x[i].cpu(), grid_size, cache)
                        for i in range(x.size(0))]).to(device)

# Layers targeted for spatial permutation — conv2 outputs, before identity add (Section III-B)
_PERM_ATTRS = [
    'conv1', 'layer1.0.conv2', 'layer1.1.conv2',
    'layer2.0.conv2', 'layer2.0.downsample.0', 'layer2.1.conv2',
    'layer3.0.conv2', 'layer3.0.downsample.0', 'layer3.1.conv2',
    'layer4.0.conv2', 'layer4.0.downsample.0', 'layer4.1.conv2',
]   # L = 12

class GridPermResNet18(nn.Module):
    """ResNet-18 with variable block permutation granularity applied to activations,
    N=8 decoys, and M=4 token gates held constant (Table VII, Section IV-F)."""

    def __init__(self, mapping, grid_size, num_decoys=8, num_classes=NUM_CLASSES,
                 token_hash: str = ''):
        super().__init__()
        self.grid_size = grid_size
        self.unshuffle = Unshuffle(mapping)
        self._cache    = {}
        resnet = models.resnet18(weights=None)
        resnet.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)  # Section IV-A: AvgPool
        resnet.fc = nn.Linear(512, num_classes)
        self.backbone = resnet

        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(num_decoys)
        ])

        # M=4 token-gated access modules held constant across granularity sweep (Section IV-F)
        self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])

        # Block-permutation applied to activations via forward hooks (not to input)
        self._perm_hooks = []
        for attr in _PERM_ATTRS:
            layer = _sub(self.backbone, attr)
            handle = layer.register_forward_hook(
                lambda m, inp, out, gs=grid_size, c=self._cache:
                    _batch_block_perm(out, gs, c)
            )
            self._perm_hooks.append(handle)

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)
        x = self.gates[0](x, token)
        # backbone.layer1..layer4 forward triggers permutation hooks on activations
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        x = self.gates[1](x, token)
        x = self.backbone.layer1(x)
        x = self.fakes[key % len(self.fakes)](x)
        x = self.gates[2](x, token)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.gates[3](x, token)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return self.backbone.fc(torch.flatten(x, 1))


def train(model, loader, token=''):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Shared-BN: forward through all N decoy paths per batch (Section IV-A)
            opt.zero_grad()
            loss = sum(
                crit(model(x, key=k, token=token), y)
                for k in range(len(model.fakes))
            ) / len(model.fakes)
            loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 20 == 0: print(f"  ep {ep+1}/{EPOCHS}")


def eval_acc(model, loader, token=''):
    model.eval(); c = t = 0
    with torch.no_grad():
        for x,y in loader:
            p=model(x.to(DEVICE), token=token).argmax(1)
            c+=(p==y.to(DEVICE)).sum().item(); t+=y.size(0)
    return c/t*100


def eval_mia(model, mem_loader, non_loader, token=''):
    model.eval()
    def probs(ldr):
        out=[]
        with torch.no_grad():
            for x,_ in ldr:
                out.extend(F.softmax(model(x.to(DEVICE), token=token),1).cpu().numpy())
        return np.array(out)
    mp=probs(mem_loader); np_=probs(non_loader)
    X=np.concatenate([mp,np_]); y=np.array([1]*len(mp)+[0]*len(np_))
    # Train/test split for unbiased AUC (Section IV-B)
    n=len(X); idx=np.random.RandomState(42).permutation(n); split=n//2
    clf=RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X[idx[:split]],y[idx[:split]])
    return roc_auc_score(y[idx[split:]], clf.predict_proba(X[idx[split:]])[:,1])


def eval_ext(model, loader, token='', n_q=10000):
    model.eval()
    clone = nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                          nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                          nn.Flatten(),nn.Linear(64*8*8,256),nn.ReLU(),nn.Linear(256,NUM_CLASSES)).to(DEVICE)
    opt=torch.optim.Adam(clone.parameters(),lr=1e-3); q=0
    for _ in range(5):
        for x,_ in loader:
            if q>=n_q: break
            x=x.to(DEVICE)
            with torch.no_grad(): soft=F.softmax(model(x, token=token),1)
            opt.zero_grad(); F.kl_div(F.log_softmax(clone(x),1),soft,reduction='batchmean').backward(); opt.step(); q+=x.size(0)
    clone.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            c+=(clone(x.to(DEVICE)).argmax(1)==y.to(DEVICE)).sum().item(); t+=y.size(0)
    return c/t*100


def main():
    mapping = get_mapping(MAP_FILE, size=32)

    # κ=256-bit token held constant across all granularity conditions (Section IV-F)
    tok_bytes   = secrets.token_bytes(32)
    token_hex   = tok_bytes.hex()
    token_hash  = hashlib.sha256(tok_bytes).hexdigest()

    train_loader = DataLoader(datasets.CIFAR100('./data',True, transform=TF,      download=True),  batch_size=BATCH,shuffle=True,  num_workers=2)
    test_loader  = DataLoader(datasets.CIFAR100('./data',False,transform=TF_TEST, download=False), batch_size=BATCH,shuffle=False, num_workers=2)
    # Members = CIFAR-100 train (the set the model was trained on)
    # Non-members = CIFAR-10 test (out-of-distribution)
    cifar100_mem = DataLoader(Subset(datasets.CIFAR100('./data',True, transform=TF_TEST,download=False),range(5000)),batch_size=BATCH,shuffle=False)
    cifar10_non  = DataLoader(Subset(datasets.CIFAR10('./data',False,transform=TF_TEST,download=True), range(5000)),batch_size=BATCH,shuffle=False)

    grid_sizes = [None, 2, 4, 8, 32]  # None=no perm, 32=pixel-level
    labels     = ['None', '2×2', '4×4', '8×8', 'pixel']

    rows = []
    for gs, label in zip(grid_sizes, labels):
        print(f"\n=== Granularity: {label} ===")
        m = GridPermResNet18(mapping, grid_size=gs, token_hash=token_hash).to(DEVICE)
        train(m, train_loader, token_hex)
        acc  = eval_acc(m, test_loader, token_hex)
        mia  = eval_mia(m, cifar100_mem, cifar10_non, token_hex)
        extr = eval_ext(m, test_loader, token_hex)
        lat  = measure_latency(m, DEVICE)
        print(f"  Acc={acc:.2f}  MIA={mia:.4f}  Ext={extr:.2f}  Lat={lat:.2f}ms")
        rows.append({'grid': label, 'accuracy': acc, 'mia_auc': mia,
                     'extraction_pct': extr, 'latency_ms': lat})

    with open(OUT_CSV,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['grid','accuracy','mia_auc','extraction_pct','latency_ms'])
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {OUT_CSV}")

if __name__ == '__main__':
    main()

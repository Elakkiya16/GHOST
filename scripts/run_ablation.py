"""
Ablation study: selectively disable GHOST tiers on ResNet-18 (Table VI, Section IV-F).

Conditions evaluated:
  Full GHOST  — all four tiers active
  w/o Decoys  — N=0 (single true path, no encrypted decoys)
  w/o Perm    — spatial permutation hooks disabled
  w/o Token   — token gate replaced with pass-through
  w/o Split   — no AES-GCM edge-cloud split (equivalent to baseline with unshuffle)
  Baseline    — plain ResNet-18, no GHOST components

Metrics per condition: test accuracy, MIA AUC (OOD), extraction %, latency (ms).
Results written to outputs/ablation_results.csv (Table VI).
"""
import os, sys, csv, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ghost.utils import Unshuffle, SpatialPerm, TokenGate, get_mapping, measure_latency

_AES_KEY = os.urandom(32)

def _encrypt(t):
    nonce = os.urandom(12)
    data  = t.detach().cpu().float().numpy().tobytes()
    ct    = AESGCM(_AES_KEY).encrypt(nonce, data, None)
    return nonce, ct, t.shape

def _decrypt(nonce, ct, shape, device):
    data = AESGCM(_AES_KEY).decrypt(nonce, ct, None)
    arr  = np.frombuffer(data, dtype=np.float32).reshape([s for s in shape])
    return torch.from_numpy(arr.copy()).to(device)

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS    = 100
BATCH     = 128
LR        = 1e-3
NUM_DECOYS = 8
MAP_FILE  = 'ghost_map_32.json'
OUT_CSV   = os.path.join('outputs', 'ablation_results.csv')
os.makedirs('outputs', exist_ok=True)

CIFAR10_TF = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3),
])
TEST_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
AUX_TF  = TEST_TF


def get_loaders():
    tr = DataLoader(datasets.CIFAR10('./data', True,  transform=CIFAR10_TF, download=True),  batch_size=BATCH, shuffle=True,  num_workers=2)
    te = DataLoader(datasets.CIFAR10('./data', False, transform=TEST_TF,    download=False), batch_size=BATCH, shuffle=False, num_workers=2)
    ax = DataLoader(Subset(datasets.CIFAR100('./data', False, transform=AUX_TF, download=True), range(5000)), batch_size=BATCH, shuffle=False, num_workers=2)
    return tr, te, ax


class AblationResNet18(nn.Module):
    """ResNet-18 with selectable GHOST tier toggles."""

    def __init__(self, mapping, use_decoys=True, use_perm=True, use_token=True,
                 use_split=True, num_decoys=NUM_DECOYS, token_hash='', num_classes=10):
        super().__init__()
        self.use_decoys = use_decoys
        self.use_perm   = use_perm
        self.use_token  = use_token
        self.use_split  = use_split

        self.unshuffle = Unshuffle(mapping)
        resnet = models.resnet18(weights=None)
        self.conv1  = resnet.conv1
        self.bn1    = resnet.bn1
        self.relu   = resnet.relu
        self.pool   = nn.AvgPool2d(3, 2, 1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(512, num_classes)

        if use_decoys:
            self.fakes = nn.ModuleList([
                nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
                for _ in range(num_decoys)
            ])
        if use_perm:
            self.spatial_perms = nn.ModuleList([SpatialPerm(1337+l) for l in range(12)])
            # Hooks target conv2 outputs (immediately before identity add) — Section III-B
            _perm_targets = [
                self.conv1,
                self.layer1[0].conv2, self.layer1[1].conv2,
                self.layer2[0].conv2, self.layer2[0].downsample[0],
                self.layer2[1].conv2,
                self.layer3[0].conv2, self.layer3[0].downsample[0],
                self.layer3[1].conv2,
                self.layer4[0].conv2, self.layer4[0].downsample[0],
                self.layer4[1].conv2,
            ]
            self._hooks = []
            for i, layer in enumerate(_perm_targets):
                pm = self.spatial_perms[i]
                self._hooks.append(layer.register_forward_hook(lambda m,i,o,p=pm: p(o)))
        if use_token:
            self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        if self.use_token:
            x = self.gates[0](x, token)
        x = self.layer1(x)

        if self.use_token:
            x = self.gates[1](x, token)
        if self.use_decoys:
            x = self.fakes[key % len(self.fakes)](x)

        # AES-GCM edge→cloud encrypted transfer (Section III-D); eval-only
        if self.use_split and not self.training:
            nonce, ct, shape = _encrypt(x)
            x = _decrypt(nonce, ct, shape, x.device)

        if self.use_token:
            x = self.gates[2](x, token)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.use_token:
            x = self.gates[3](x, token)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def train_model(model, loader, epochs, token=''):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if hasattr(model, 'fakes') and len(model.fakes) > 0:
                # Shared-BN: forward through all N decoy paths per batch (Section IV-A)
                loss = sum(
                    crit(model(x, key=k, token=token), y)
                    for k in range(len(model.fakes))
                ) / len(model.fakes)
            else:
                loss = crit(model(x, token=token), y)
            loss.backward()
            opt.step()
        sch.step()
        if (epoch+1) % 20 == 0:
            print(f"  epoch {epoch+1}/{epochs}")


def evaluate_accuracy(model, loader, token=''):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x, token=token).argmax(1)
            correct += (preds == y).sum().item(); total += y.size(0)
    return correct / total * 100


def evaluate_mia(model, member_loader, nonmember_loader, token=''):
    model.eval()
    def get_probs(loader):
        out = []
        with torch.no_grad():
            for x, _ in loader:
                p = F.softmax(model(x.to(DEVICE), token=token), dim=1).cpu().numpy()
                out.extend(p)
        return np.array(out)
    mem_p   = get_probs(member_loader)
    non_p   = get_probs(nonmember_loader)
    X = np.concatenate([mem_p, non_p])
    y = np.array([1]*len(mem_p) + [0]*len(non_p))
    # Train/test split for unbiased AUC (Section IV-B: evaluated on held-out test set)
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = n // 2
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X[idx[:split]], y[idx[:split]])
    return roc_auc_score(y[idx[split:]], clf.predict_proba(X[idx[split:]])[:, 1])


def evaluate_extraction(model, test_loader, n_queries=10000, token=''):
    model.eval()
    clone = nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.AvgPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AvgPool2d(2),
        nn.Flatten(), nn.Linear(64*8*8, 256), nn.ReLU(), nn.Linear(256,10)
    ).to(DEVICE)
    opt = torch.optim.Adam(clone.parameters(), lr=1e-3)
    queries = 0
    for _ in range(5):
        for x, _ in test_loader:
            if queries >= n_queries: break
            x = x.to(DEVICE)
            with torch.no_grad():
                soft = F.softmax(model(x, token=token), dim=1)
            opt.zero_grad()
            F.kl_div(F.log_softmax(clone(x), dim=1), soft, reduction='batchmean').backward()
            opt.step()
            queries += x.size(0)
    clone.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (clone(x).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total * 100


def main():
    mapping = get_mapping(MAP_FILE, size=32)

    import hashlib, secrets
    tok_b = secrets.token_bytes(32)
    token_hex  = tok_b.hex()
    token_hash = hashlib.sha256(tok_b).hexdigest()

    train_loader, test_loader, aux_loader = get_loaders()
    mem_loader = DataLoader(Subset(datasets.CIFAR10('./data', True, transform=TEST_TF, download=False), range(5000)), batch_size=BATCH, shuffle=False)

    conditions = [
        ('Full GHOST',  dict(use_decoys=True,  use_perm=True,  use_token=True,  use_split=True,  token_hash=token_hash)),
        ('w/o Decoys',  dict(use_decoys=False, use_perm=True,  use_token=True,  use_split=True,  token_hash=token_hash)),
        ('w/o Perm',    dict(use_decoys=True,  use_perm=False, use_token=True,  use_split=True,  token_hash=token_hash)),
        ('w/o Token',   dict(use_decoys=True,  use_perm=True,  use_token=False, use_split=True,  token_hash='')),
        ('w/o Split',   dict(use_decoys=True,  use_perm=True,  use_token=True,  use_split=False, token_hash=token_hash)),
        ('Baseline',    dict(use_decoys=False, use_perm=False, use_token=False, use_split=False, token_hash='')),
    ]

    rows = []
    for cond_name, kwargs in conditions:
        print(f"\n=== {cond_name} ===")
        m = AblationResNet18(mapping, **kwargs).to(DEVICE)
        tok = token_hex if kwargs.get('use_token') else ''
        train_model(m, train_loader, EPOCHS, token=tok)
        acc   = evaluate_accuracy(m, test_loader, token=tok)
        mia   = evaluate_mia(m, mem_loader, aux_loader, token=tok)
        extr  = evaluate_extraction(m, test_loader, token=tok)
        lat   = measure_latency(m, DEVICE)
        print(f"  Acc={acc:.2f}%  MIA AUC={mia:.4f}  Ext%={extr:.2f}  Latency={lat:.2f}ms")
        rows.append({'condition': cond_name, 'accuracy': acc, 'mia_auc': mia,
                     'extraction_pct': extr, 'latency_ms': lat})

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['condition','accuracy','mia_auc','extraction_pct','latency_ms'])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {OUT_CSV}")

if __name__ == '__main__':
    main()

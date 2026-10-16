"""
Decoy density sweep: vary N ∈ {0, 2, 4, 8, 16} on ResNet-18 / CIFAR-10 (Table VIII, Section IV-F).
Metrics: test accuracy, MIA AUC, extraction %, latency (ms).
Results written to outputs/decoy_density.csv.
"""
import os, sys, csv, hashlib, secrets
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

from src.ghost.utils import Unshuffle, SpatialPerm, TokenGate, get_mapping, measure_latency

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100; BATCH = 128; LR = 1e-3
MAP_FILE = 'ghost_map_32.json'
OUT_CSV  = os.path.join('outputs', 'decoy_density.csv')
os.makedirs('outputs', exist_ok=True)

TF      = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4),
                               transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])
TF_TEST = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,)*3,(0.5,)*3)])


class DecoyResNet18(nn.Module):
    """ResNet-18 GHOST with configurable N decoy paths; M=4 gates held constant (Table VIII)."""
    def __init__(self, mapping, num_decoys=8, num_classes=10, token_hash: str = ''):
        super().__init__()
        self.num_decoys = num_decoys
        self.unshuffle  = Unshuffle(mapping)
        resnet = models.resnet18(weights=None)
        self.conv1  = resnet.conv1; self.bn1 = resnet.bn1; self.relu = resnet.relu
        self.pool   = nn.AvgPool2d(3,2,1)
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(512, num_classes)
        # N decoy paths (N=0 means true path only)
        if num_decoys > 0:
            self.fakes = nn.ModuleList([
                nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
                for _ in range(num_decoys)
            ])
        # M=4 token-gated access modules held constant (Section IV-F, Table VIII)
        self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])
        # L=12 spatial permutations
        self.spatial_perms = nn.ModuleList([SpatialPerm(1337+l) for l in range(12)])
        # conv2 outputs immediately before identity add — Section III-B
        _perm_layers = [
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
        for i, lyr in enumerate(_perm_layers):
            pm = self.spatial_perms[i]
            self._hooks.append(lyr.register_forward_hook(lambda m,inp,out,p=pm: p(out)))

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.gates[0](x, token)
        x = self.layer1(x)
        x = self.gates[1](x, token)
        if self.num_decoys > 0:
            x = self.fakes[key % self.num_decoys](x)
        x = self.gates[2](x, token)
        x = self.layer2(x); x = self.layer3(x)
        x = self.gates[3](x, token)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def train(model, loader, num_decoys, token=''):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if num_decoys > 0:
                # Shared-BN: forward through all N decoy paths per batch (Section IV-A)
                loss = sum(
                    crit(model(x, key=k, token=token), y)
                    for k in range(num_decoys)
                ) / num_decoys
            else:
                loss = crit(model(x, token=token), y)
            loss.backward(); opt.step()
        sch.step()
        if (ep+1) % 20 == 0: print(f"  ep {ep+1}/{EPOCHS}")


def eval_acc(model, loader, num_decoys, token=''):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            p=model(x.to(DEVICE), token=token).argmax(1)
            c+=(p==y.to(DEVICE)).sum().item(); t+=y.size(0)
    return c/t*100


def eval_mia(model, mem_l, non_l, token=''):
    model.eval()
    def probs(ldr):
        out=[]
        with torch.no_grad():
            for x,_ in ldr:
                out.extend(F.softmax(model(x.to(DEVICE), token=token),1).cpu().numpy())
        return np.array(out)
    mp=probs(mem_l); np_=probs(non_l)
    X=np.concatenate([mp,np_]); y=np.array([1]*len(mp)+[0]*len(np_))
    # Train/test split for unbiased AUC (Section IV-B)
    n=len(X); idx=np.random.RandomState(42).permutation(n); split=n//2
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X[idx[:split]],y[idx[:split]])
    return roc_auc_score(y[idx[split:]], clf.predict_proba(X[idx[split:]])[:,1])


def eval_ext(model, loader, token='', n_q=10000):
    model.eval()
    clone=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                        nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                        nn.Flatten(),nn.Linear(64*8*8,256),nn.ReLU(),nn.Linear(256,10)).to(DEVICE)
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

    # κ=256-bit token held constant across all N conditions (Section IV-F, Table VIII)
    tok_bytes  = secrets.token_bytes(32)
    token_hex  = tok_bytes.hex()
    token_hash = hashlib.sha256(tok_bytes).hexdigest()

    train_l = DataLoader(datasets.CIFAR10('./data',True, transform=TF,      download=True), batch_size=BATCH,shuffle=True, num_workers=2)
    test_l  = DataLoader(datasets.CIFAR10('./data',False,transform=TF_TEST, download=False),batch_size=BATCH,shuffle=False,num_workers=2)
    mem_l   = DataLoader(Subset(datasets.CIFAR10('./data',True, transform=TF_TEST,download=False),range(5000)),batch_size=BATCH,shuffle=False)
    non_l   = DataLoader(Subset(datasets.CIFAR100('./data',False,transform=TF_TEST,download=True), range(5000)),batch_size=BATCH,shuffle=False)

    rows = []
    for N in [0, 2, 4, 8, 16]:
        print(f"\n=== N={N} decoys ===")
        m = DecoyResNet18(mapping, num_decoys=N, token_hash=token_hash).to(DEVICE)
        train(m, train_l, N, token=token_hex)
        acc  = eval_acc(m, test_l, N, token=token_hex)
        mia  = eval_mia(m, mem_l, non_l, token=token_hex)
        extr = eval_ext(m, test_l, token=token_hex)
        lat  = measure_latency(m, DEVICE)
        print(f"  Acc={acc:.2f}  MIA={mia:.4f}  Ext={extr:.2f}  Lat={lat:.2f}ms")
        rows.append({'N': N,'accuracy':acc,'mia_auc':mia,'extraction_pct':extr,'latency_ms':lat})

    with open(OUT_CSV,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['N','accuracy','mia_auc','extraction_pct','latency_ms'])
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {OUT_CSV}")

if __name__ == '__main__':
    main()

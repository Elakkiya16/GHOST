"""Shared data, models and training utilities for the GHOST v2 experiments."""
import os, time, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'results')
CKPT = os.path.join(HERE, 'checkpoints')
os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT, exist_ok=True)

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def load(name):
    d = np.load(os.path.join(DATA, name + '.npz'))
    X = torch.from_numpy(d['X']).permute(0, 3, 1, 2).float().div(255.)
    X = (X - MEAN) / STD
    return X, torch.from_numpy(d['y'])


def splits(seed=0):
    """Victim members: 40k of the CIFAR-10 training set.
    Attacker pool: the disjoint remaining 10k (same distribution).
    Non-members / evaluation: the 10k CIFAR-10 test set."""
    X, y = load('c10_train')
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    mem, pool = idx[:40000], idx[40000:]
    Xt, yt = load('c10_test')
    return (X[mem], y[mem]), (X[pool], y[pool]), (Xt, yt)


def augment(x):
    # random crop with 4-pixel padding and horizontal flip
    B = x.size(0)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    i = torch.randint(0, 9, (B,)); j = torch.randint(0, 9, (B,))
    out = torch.stack([xp[b, :, i[b]:i[b] + 32, j[b]:j[b] + 32] for b in range(B)])
    flip = torch.rand(B) < 0.5
    out[flip] = out[flip].flip(3)
    return out


# ----------------------------------------------------------------- models
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        s = x if self.short is None else self.short(x)
        return F.relu(o + s)


class ResNetC(nn.Module):
    """CIFAR ResNet (ResNet-20 for n=3, width 16/32/64)."""
    def __init__(self, n=3, widths=(16, 32, 64), num_classes=10):
        super().__init__()
        w0 = widths[0]
        self.conv1 = nn.Conv2d(3, w0, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(w0)
        layers, cin = [], w0
        for si, w in enumerate(widths):
            for b in range(n):
                layers.append(BasicBlock(cin, w, 2 if (b == 0 and si > 0) else 1)); cin = w
        self.blocks = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class VGGC(nn.Module):
    """VGG-11-style network for 32x32 inputs (half width)."""
    cfg = [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']

    def __init__(self, num_classes=10):
        super().__init__()
        layers, cin = [], 3
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, 1, 1, bias=False), nn.BatchNorm2d(v), nn.ReLU()]
                cin = v
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


ARCHS = {'resnet20': lambda nc=10: ResNetC(3, num_classes=nc),
         'vgg11h': lambda nc=10: VGGC(nc)}


# ----------------------------------------------------------------- training
def train(model, X, y, epochs=30, bs=128, lr=0.1, wd=5e-4, aug=True, log=None, seed=0, resume=None):
    """Training with per-epoch checkpoints (resume = path) so an interrupted run continues."""
    seed_all(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    start = 0
    if resume and os.path.exists(resume):
        st = torch.load(resume)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        torch.set_rng_state(st['rng']); start = st['epoch']
        if log: log(f'resumed at epoch {start}')
    for ep in range(start, epochs):
        model.train(); perm = torch.randperm(len(X)); t0 = time.time(); tl = 0.
        for k in range(0, len(X), bs):
            b = perm[k:k + bs]; xb = X[b]; yb = y[b]
            if aug: xb = augment(xb)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(b)
        if log: log(f'ep {ep+1}/{epochs} loss {tl/len(X):.4f} {time.time()-t0:.0f}s')
        if resume:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                        'rng': torch.get_rng_state(), 'epoch': ep + 1}, resume)
    model.eval(); return model


@torch.no_grad()
def logits_of(model, X, bs=500):
    model.eval()
    return torch.cat([model(X[k:k + bs]) for k in range(0, len(X), bs)])


@torch.no_grad()
def accuracy(model, X, y):
    return (logits_of(model, X).argmax(1) == y).float().mean().item() * 100


def save_json(obj, name):
    with open(os.path.join(OUT, name), 'w') as f: json.dump(obj, f, indent=1)


def train_steps(model, X, y, steps=1500, bs=128, lr=0.05, wd=5e-4, aug=True, seed=0, soft=None, params=None):
    """Fixed-iteration training used by the attackers (soft = soft-label targets)."""
    seed_all(seed)
    opt = torch.optim.SGD(params if params is not None else model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    n = len(X); model.train(); it = 0
    while it < steps:
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            if it >= steps: break
            b = perm[k:k + bs]; xb = X[b]
            if aug: xb = augment(xb)
            out = model(xb)
            if soft is not None:
                loss = F.kl_div(F.log_softmax(out, 1), soft[b], reduction='batchmean')
            else:
                loss = F.cross_entropy(out, y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
    model.eval(); return model

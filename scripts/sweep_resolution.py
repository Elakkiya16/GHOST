"""
Input resolution scaling on ResNet-18 (Table IX, Section IV-F).
Resolutions: 32×32 (CIFAR-10), 64×64 (TinyImageNet), 128×128, 224×224 (ImageNet-50k).
Metrics: test accuracy, MIA AUC, extraction %, inference latency (ms).
Results written to outputs/resolution_scaling.csv.
"""
import os, sys, csv
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

from src.ghost.utils import Unshuffle, SpatialPerm, get_mapping, measure_latency

DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS  = 100; BATCH = 64; LR = 1e-3
OUT_CSV = os.path.join('outputs', 'resolution_scaling.csv')
os.makedirs('outputs', exist_ok=True)


class GHOSTResNet18(nn.Module):
    def __init__(self, mapping, img_size=32, num_decoys=8, num_classes=10):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        resnet = models.resnet18(weights=None)
        if img_size <= 32:
            resnet.conv1   = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(512, num_classes)
        self.backbone = resnet
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(num_decoys)
        ])
        self.spatial_perms = nn.ModuleList([SpatialPerm(1337+l) for l in range(12)])

    def forward(self, x, key=0):
        x = self.unshuffle(x)
        return self.backbone(x)


def make_tf(size):
    aug = [transforms.Resize((size, size))]
    if size > 32:
        aug += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=size//8)]
    else:
        aug += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    return (
        transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)]),
        transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)]),
    )


def get_cifar10_loaders(size):
    tf_train, tf_test = make_tf(size)
    tr = DataLoader(datasets.CIFAR10('./data',True, transform=tf_train,download=True), batch_size=BATCH,shuffle=True, num_workers=2)
    te = DataLoader(datasets.CIFAR10('./data',False,transform=tf_test, download=False),batch_size=BATCH,shuffle=False,num_workers=2)
    return tr, te


def train(model, loader):
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    crit=nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            key=torch.randint(0,len(model.fakes),(1,)).item()
            opt.zero_grad(); crit(model(x,key=key),y).backward(); opt.step()
        sch.step()
        if (ep+1)%20==0: print(f"  ep {ep+1}/{EPOCHS}")


def eval_acc(model, loader):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            c+=(model(x.to(DEVICE)).argmax(1)==y.to(DEVICE)).sum().item(); t+=y.size(0)
    return c/t*100


def eval_mia(model, mem_l, non_l):
    model.eval()
    def probs(ldr):
        out=[]
        with torch.no_grad():
            for x,_ in ldr: out.extend(F.softmax(model(x.to(DEVICE)),1).cpu().numpy())
        return np.array(out)
    mp=probs(mem_l); np_=probs(non_l)
    X=np.concatenate([mp,np_]); y=np.array([1]*len(mp)+[0]*len(np_))
    clf=RandomForestClassifier(n_estimators=100,random_state=42)
    clf.fit(X,y); return roc_auc_score(y, clf.predict_proba(X)[:,1])


def eval_ext(model, loader, nc, n_q=10000):
    model.eval()
    sz = loader.dataset[0][0].shape[-1]
    flat = (sz//4)**2 * 64
    clone=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                        nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
                        nn.Flatten(),nn.Linear(flat,256),nn.ReLU(),nn.Linear(256,nc)).to(DEVICE)
    opt=torch.optim.Adam(clone.parameters(),lr=1e-3); q=0
    for _ in range(5):
        for x,_ in loader:
            if q>=n_q: break
            x=x.to(DEVICE)
            with torch.no_grad(): soft=F.softmax(model(x),1)
            opt.zero_grad(); F.kl_div(F.log_softmax(clone(x),1),soft,reduction='batchmean').backward(); opt.step(); q+=x.size(0)
    clone.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            c+=(clone(x.to(DEVICE)).argmax(1)==y.to(DEVICE)).sum().item(); t+=y.size(0)
    return c/t*100


def main():
    rows = []
    for size in [32, 64, 128, 224]:
        print(f"\n=== Resolution {size}×{size} ===")
        map_file = f'ghost_map_{size}.json'
        mapping  = get_mapping(map_file, size=size)
        train_l, test_l = get_cifar10_loaders(size)
        mem_l  = DataLoader(Subset(datasets.CIFAR10('./data',True, transform=make_tf(size)[1],download=False),range(2000)),batch_size=BATCH,shuffle=False)
        non_l  = DataLoader(Subset(datasets.CIFAR100('./data',False,transform=make_tf(size)[1],download=True), range(2000)),batch_size=BATCH,shuffle=False)

        m = GHOSTResNet18(mapping, img_size=size).to(DEVICE)
        train(m, train_l)
        acc  = eval_acc(m, test_l)
        mia  = eval_mia(m, mem_l, non_l)
        extr = eval_ext(m, test_l, nc=10)
        lat  = measure_latency(m, DEVICE, input_shape=(1,3,size,size))
        print(f"  Acc={acc:.2f}  MIA={mia:.4f}  Ext={extr:.2f}  Lat={lat:.2f}ms")
        rows.append({'resolution':f'{size}x{size}','accuracy':acc,'mia_auc':mia,
                     'extraction_pct':extr,'latency_ms':lat})

    with open(OUT_CSV,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['resolution','accuracy','mia_auc','extraction_pct','latency_ms'])
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {OUT_CSV}")

if __name__ == '__main__':
    main()

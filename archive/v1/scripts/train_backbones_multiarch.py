"""
scripts/train_backbones.py

Trains GHOST-augmented ResNet-18, ResNet-50, and MobileNetV3 across five
datasets and reproduces Table V (Section IV-E: Scalability and Performance).

Backbone GHOST configuration (Section IV-E):
  N = 8  encrypted decoy paths
  L = 12 layer-wise spatial permutations (applied at residual-stage outputs)
  M = 4  token-gated access modules  (κ = 256 bits)

Training protocol (Section IV):
  Optimiser : Adam, lr = 0.001
  Scheduler : CosineAnnealingLR
  Batch size: 128
  Epochs    : 100  (backbone_num_epochs in configs/default_config.py)
  Runs      : results averaged across 5 independent seeds

Hardware: Results in the paper were obtained on NVIDIA Tesla V100
          (32 GB VRAM, CUDA 12.2). This script runs on CUDA, Apple MPS,
          and CPU.

Datasets
--------
CIFAR-10 / CIFAR-100 / SVHN  — downloaded automatically via torchvision.
TinyImageNet                  — downloaded automatically from the Stanford
                                CS231N mirror (~237 MB, first run only).
ImageNet-50k                  — must be provided locally. Set the env var
                                IMAGENET_DIR to the root of the ImageNet
                                ILSVRC2012 directory (containing train/ and
                                val/). If unset, this dataset is skipped.

Usage
-----
# All 15 (arch × dataset) combinations:
python scripts/train_backbones.py

# Single combination:
python scripts/train_backbones.py --arch ResNet-18 --dataset CIFAR-10

Outputs
-------
outputs/backbone_results.csv        — MIA AUC, Ext.%, latency per run
outputs/<arch>_<dataset>_ghost.pth  — GHOST model checkpoint
outputs/<arch>_<dataset>_base.pth   — baseline model checkpoint
"""

import os
import sys
import csv
import json
import time
import random
import hashlib
import secrets
import urllib.request
import zipfile
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.default_config import CONFIG as DEFAULT_CONFIG
from src.ghost.utils import Unshuffle, get_mapping
from src.ghost.models import (
    GHOST_ResNet18, GHOST_ResNet50, GHOST_MobileNetV3,
    BaselineResNet18, BaselineResNet50, BaselineMobileNetV3,
)
from src.ghost.attacks.extraction import run_tramer_extraction
from src.ghost.attacks.privacy import PrivacyAttackSuite

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

def get_device():
    # Paper results: NVIDIA Tesla V100 (32 GB VRAM, CUDA 12.2).
    # Also runs on Apple MPS and CPU for local development.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EPOCHS      = DEFAULT_CONFIG["backbone_num_epochs"]  # 100 (Section IV)
BATCH_SIZE  = DEFAULT_CONFIG["batch_size"]            # 128
LR          = DEFAULT_CONFIG["lr"]                    # 1e-3
NUM_DECOYS  = 8                                       # N=8 for backbone models (Section IV-E)
NUM_SEEDS   = 5                                       # 5 independent runs (Section IV)
DATA_ROOT   = "./data"
OUT_DIR     = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_META = {
    "CIFAR-10":      {"num_classes": 10,   "img_size": 32},
    "CIFAR-100":     {"num_classes": 100,  "img_size": 32},
    "SVHN":          {"num_classes": 10,   "img_size": 32},
    "TinyImageNet":  {"num_classes": 200,  "img_size": 64},
    "ImageNet-50k":  {"num_classes": 1000, "img_size": 224},
}

ARCH_NAMES  = ["ResNet-18", "ResNet-50", "MobileNetV3"]
DATA_NAMES  = list(DATASET_META.keys())

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _norm(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return transforms.Normalize(mean, std)

def _aug(size):
    return [
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size, padding=4),
        transforms.ToTensor(),
    ]

def _test_tf(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), _norm()])


def _download_tinyimagenet(root):
    dest = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(dest):
        return dest
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print(f"Downloading TinyImageNet to {zip_path} …")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)
    # Re-organise val/ into ImageFolder-compatible structure
    val_dir = os.path.join(dest, "val")
    annot   = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isfile(annot):
        img_dir = os.path.join(val_dir, "images")
        with open(annot) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(img_dir, fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.isfile(src) and not os.path.isfile(dst):
                    os.rename(src, dst)
    print("TinyImageNet ready.")
    return dest


def get_loaders(dataset_name, img_size, mapping=None):
    """Return (train_loader, test_loader, aux_loader) for OOD MIA evaluation."""
    aug_tf  = transforms.Compose(_aug(img_size) + [_norm()])
    test_tf = _test_tf(img_size)

    if dataset_name == "CIFAR-10":
        train_ds = datasets.CIFAR10(DATA_ROOT, True,  download=True,  transform=aug_tf)
        test_ds  = datasets.CIFAR10(DATA_ROOT, False, download=True,  transform=test_tf)
        aux_ds   = datasets.CIFAR100(DATA_ROOT, False, download=True, transform=test_tf)

    elif dataset_name == "CIFAR-100":
        train_ds = datasets.CIFAR100(DATA_ROOT, True,  download=True,  transform=aug_tf)
        test_ds  = datasets.CIFAR100(DATA_ROOT, False, download=True,  transform=test_tf)
        # OOD non-members from CIFAR-10 when primary dataset is CIFAR-100
        aux_ds   = datasets.CIFAR10(DATA_ROOT, False,  download=True, transform=test_tf)

    elif dataset_name == "SVHN":
        train_ds = datasets.SVHN(DATA_ROOT, split="train", download=True, transform=aug_tf)
        test_ds  = datasets.SVHN(DATA_ROOT, split="test",  download=True, transform=test_tf)
        aux_ds   = datasets.CIFAR100(DATA_ROOT, False, download=True, transform=test_tf)

    elif dataset_name == "TinyImageNet":
        tin_root = _download_tinyimagenet(DATA_ROOT)
        train_ds = datasets.ImageFolder(os.path.join(tin_root, "train"), transform=aug_tf)
        test_ds  = datasets.ImageFolder(os.path.join(tin_root, "val"),   transform=test_tf)
        aux_ds   = datasets.CIFAR100(DATA_ROOT, False, download=True, transform=test_tf)

    elif dataset_name == "ImageNet-50k":
        imagenet_dir = os.environ.get("IMAGENET_DIR", "")
        if not imagenet_dir or not os.path.isdir(imagenet_dir):
            raise RuntimeError(
                "ImageNet-50k: set the IMAGENET_DIR environment variable to "
                "the root of the ILSVRC2012 directory (containing train/ and val/)."
            )
        full_train = datasets.ImageFolder(os.path.join(imagenet_dir, "train"), transform=aug_tf)
        # Sample 50 k images uniformly at random (fixed seed for reproducibility)
        rng = random.Random(42)
        indices = rng.sample(range(len(full_train)), 50_000)
        train_ds = Subset(full_train, indices)
        test_ds  = datasets.ImageFolder(os.path.join(imagenet_dir, "val"), transform=test_tf)
        aux_ds   = datasets.CIFAR100(DATA_ROOT, False, download=True, transform=test_tf)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        DataLoader(aux_ds,   shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(arch, mapping, num_classes):
    # κ=256-bit token for backbone models (Section IV-E / Section III-C)
    tok_bytes  = secrets.token_bytes(32)
    tok_hex    = tok_bytes.hex()
    tok_hash   = hashlib.sha256(tok_bytes).hexdigest()

    if arch == "ResNet-18":
        ghost = GHOST_ResNet18(mapping, num_classes=num_classes, num_decoys=NUM_DECOYS,
                               token_hash=tok_hash)
        base  = BaselineResNet18(num_classes=num_classes)
    elif arch == "ResNet-50":
        ghost = GHOST_ResNet50(mapping, num_classes=num_classes, num_decoys=NUM_DECOYS,
                               token_hash=tok_hash)
        base  = BaselineResNet50(num_classes=num_classes)
    elif arch == "MobileNetV3":
        ghost = GHOST_MobileNetV3(mapping, num_classes=num_classes, num_decoys=NUM_DECOYS,
                                  token_hash=tok_hash)
        base  = BaselineMobileNetV3(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ghost = ghost.to(DEVICE)
    # Store token_hex on the model so callers can pass it without extra plumbing
    ghost._ghost_token_hex = tok_hex
    return ghost, base.to(DEVICE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    token = getattr(model, '_ghost_token_hex', '')
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if hasattr(model, "fakes"):
            # Shared-BN: forward through all N decoy paths per batch (Section IV-A)
            loss = sum(
                criterion(model(x, key=k, token=token), y)
                for k in range(len(model.fakes))
            ) / len(model.fakes)
        else:
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def train_model(model, loader, tag):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    print(f"  Training {tag} for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()
        if epoch % 20 == 0:
            print(f"    epoch {epoch}/{EPOCHS}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    token = getattr(model, '_ghost_token_hex', '')
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if hasattr(model, "fakes"):
            preds = model(x, key=0, token=token).argmax(1)
        else:
            preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def eval_mia_auc(ghost_model, shadow_models, member_loader, nonmember_loader):
    suite = PrivacyAttackSuite(ghost_model, shadow_models, DEVICE)
    return suite.shokri_shadow_mia(member_loader, nonmember_loader)


def eval_extraction(model, query_loader, test_loader):
    return run_tramer_extraction(model, query_loader, test_loader, DEVICE)


def measure_latency(model, img_size, n_runs=200):
    model.eval()
    token = getattr(model, '_ghost_token_hex', '')
    dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE)
    # warm-up
    for _ in range(10):
        if hasattr(model, "fakes"):
            _ = model(dummy, key=0, token=token)
        else:
            _ = model(dummy)
    # timed
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        if hasattr(model, "fakes"):
            _ = model(dummy, key=0, token=token)
        else:
            _ = model(dummy)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000  # ms


# ---------------------------------------------------------------------------
# Per-combination runner
# ---------------------------------------------------------------------------

def run_combination(arch, dataset_name, csv_writer):
    meta     = DATASET_META[dataset_name]
    img_size = meta["img_size"]
    n_cls    = meta["num_classes"]

    print(f"\n{'='*60}")
    print(f"  {arch} on {dataset_name}  ({n_cls} classes, {img_size}×{img_size})")
    print(f"{'='*60}")

    map_path = os.path.join(DATA_ROOT, f"ghost_map_{img_size}.json")
    mapping  = get_mapping(map_path, size=img_size)

    try:
        train_loader, test_loader, aux_loader = get_loaders(dataset_name, img_size, mapping)
    except RuntimeError as exc:
        print(f"  SKIP — {exc}")
        return

    # Accumulate metrics across NUM_SEEDS runs
    metrics = {"ghost_acc": [], "base_acc": [], "ghost_mia": [], "base_mia": [],
               "ghost_ext": [], "base_ext": [], "ghost_lat": [], "base_lat": []}

    for seed in range(NUM_SEEDS):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        ghost, base = build_models(arch, mapping, n_cls)

        print(f"  Seed {seed+1}/{NUM_SEEDS} — training …")
        train_model(ghost, train_loader, f"GHOST {arch}")
        train_model(base,  train_loader, f"Baseline {arch}")

        # Save checkpoints (last seed overwrites; all seeds are averaged for the table)
        slug = f"{arch.replace('-','').replace('V','v')}_{dataset_name.replace('-','').replace(' ','_')}"
        ghost_ckpt = os.path.join(OUT_DIR, f"{slug}_ghost_seed{seed}.pth")
        base_ckpt  = os.path.join(OUT_DIR, f"{slug}_base_seed{seed}.pth")
        torch.save(ghost.state_dict(), ghost_ckpt)
        torch.save(base.state_dict(),  base_ckpt)

        # Classification accuracy
        ghost_acc = eval_accuracy(ghost, test_loader)
        base_acc  = eval_accuracy(base,  test_loader)

        # MIA AUC (OOD protocol — Section IV-B)
        shadow = [BaselineResNet18(num_classes=n_cls).to(DEVICE) for _ in range(2)]
        for s in shadow:
            train_model(s, train_loader, "shadow")
        ghost_mia = eval_mia_auc(ghost, shadow, test_loader, aux_loader)
        base_mia  = eval_mia_auc(base,  shadow, test_loader, aux_loader)

        # Model extraction accuracy
        ghost_ext = eval_extraction(ghost, test_loader, test_loader)
        base_ext  = eval_extraction(base,  test_loader, test_loader)

        # Latency
        ghost_lat = measure_latency(ghost, img_size)
        base_lat  = measure_latency(base,  img_size)

        metrics["ghost_acc"].append(ghost_acc)
        metrics["base_acc"].append(base_acc)
        metrics["ghost_mia"].append(ghost_mia)
        metrics["base_mia"].append(base_mia)
        metrics["ghost_ext"].append(ghost_ext)
        metrics["base_ext"].append(base_ext)
        metrics["ghost_lat"].append(ghost_lat)
        metrics["base_lat"].append(base_lat)

        print(f"    Acc  ghost={ghost_acc:.2f}%  base={base_acc:.2f}%")
        print(f"    MIA  ghost={ghost_mia:.4f}   base={base_mia:.4f}")
        print(f"    Ext  ghost={ghost_ext:.2f}%  base={base_ext:.2f}%")
        print(f"    Lat  ghost={ghost_lat:.2f}ms  base={base_lat:.2f}ms")

    # Write averaged row to CSV
    def avg(lst): return float(np.mean(lst))
    def std(lst): return float(np.std(lst))

    csv_writer.writerow({
        "arch": arch, "dataset": dataset_name,
        "ghost_acc_mean": f"{avg(metrics['ghost_acc']):.2f}",
        "ghost_acc_std":  f"{std(metrics['ghost_acc']):.2f}",
        "base_acc_mean":  f"{avg(metrics['base_acc']):.2f}",
        "ghost_mia_auc":  f"{avg(metrics['ghost_mia']):.4f}",
        "base_mia_auc":   f"{avg(metrics['base_mia']):.4f}",
        "ghost_ext_pct":  f"{avg(metrics['ghost_ext']):.2f}",
        "base_ext_pct":   f"{avg(metrics['base_ext']):.2f}",
        "ghost_lat_ms":   f"{avg(metrics['ghost_lat']):.2f}",
        "base_lat_ms":    f"{avg(metrics['base_lat']):.2f}",
    })

    print(f"  Done. Averaged over {NUM_SEEDS} seeds.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GHOST backbone models (Table V)")
    parser.add_argument("--arch",    choices=ARCH_NAMES, default=None,
                        help="Architecture to train (default: all)")
    parser.add_argument("--dataset", choices=DATA_NAMES, default=None,
                        help="Dataset to use (default: all)")
    args = parser.parse_args()

    archs    = [args.arch]    if args.arch    else ARCH_NAMES
    datasets_ = [args.dataset] if args.dataset else DATA_NAMES

    csv_path = os.path.join(OUT_DIR, "backbone_results.csv")
    fieldnames = [
        "arch", "dataset",
        "ghost_acc_mean", "ghost_acc_std", "base_acc_mean",
        "ghost_mia_auc", "base_mia_auc",
        "ghost_ext_pct", "base_ext_pct",
        "ghost_lat_ms",  "base_lat_ms",
    ]
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for arch in archs:
            for ds in datasets_:
                run_combination(arch, ds, writer)
                f.flush()

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()

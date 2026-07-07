"""Train the GHOST backbone model (Section IV-E).

Produces the checkpoint consumed by the attack suite in
src/ghost/attacks/attack_adaptive_perm.py and src/ghost/attacks/attack_lira.py:
  - checkpoints/ghost_<arch>_<dataset>.pt   (GHOST_* state dict)
  - shuffle_map.json                        (input-pixel shuffle map)
  - ghost_token.json                        (kappa=256-bit gate token, hex + sha256 hash)

Saves a checkpoint every --save-every epochs so the run can be resumed with
--resume if interrupted. Pass --train-baseline to also train BaselineResNet18
alongside it (not required by the attack scripts, doubles the per-epoch cost).

USAGE (from repo root):
    python scripts/train_backbones.py --arch resnet18 --dataset cifar10
"""

import os
import sys
import json
import hashlib
import secrets
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.default_config import CONFIG
from src.ghost.utils import get_mapping
from src.ghost.models import (
    GHOST_ResNet18, GHOST_ResNet50, GHOST_MobileNetV3,
    BaselineResNet18, BaselineResNet50, BaselineMobileNetV3,
)
from src.ghost.attacks.hf_datasets import HFImageDataset

_GHOST = {"resnet18": GHOST_ResNet18, "resnet50": GHOST_ResNet50, "mobilenetv3": GHOST_MobileNetV3}
_BASE = {"resnet18": BaselineResNet18, "resnet50": BaselineResNet50, "mobilenetv3": BaselineMobileNetV3}

NUM_DECOYS = 8  # N=8 decoys for backbone models (Section IV-E)


def _tf(augment):
    ops = [transforms.Resize((32, 32))]
    if augment:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    ops += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    return transforms.Compose(ops)


def build_loaders(dataset, data_root, batch_size, num_workers=0):
    train_ds = HFImageDataset(dataset, data_root, train=True, transform=_tf(augment=True))
    test_ds = HFImageDataset(dataset, data_root, train=False, transform=_tf(augment=False))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate(model, loader, device, is_ghost, token_hex=""):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, token=token_hex) if is_ghost else model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser(description="Train the GHOST backbone model (Section IV-E)")
    ap.add_argument("--arch", choices=list(_GHOST), default="resnet18")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=CONFIG["backbone_num_epochs"])
    ap.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"])
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--shuffle-map", default="shuffle_map.json")
    ap.add_argument("--token-file", default="ghost_token.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train-baseline", action="store_true",
                     help="also train BaselineResNet18 alongside GHOST (not needed by the attacks)")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}", flush=True)

    torch.manual_seed(42)
    mapping = get_mapping(args.shuffle_map)

    if os.path.exists(args.token_file):
        with open(args.token_file) as f:
            td = json.load(f)
        token_hex, token_hash = td["token_hex"], td["token_hash"]
    else:
        tb = secrets.token_bytes(32)  # kappa = 256-bit gate token (Section III-C, backbones)
        token_hex, token_hash = tb.hex(), hashlib.sha256(tb).hexdigest()
        with open(args.token_file, "w") as f:
            json.dump({"token_hex": token_hex, "token_hash": token_hash}, f)

    train_loader, test_loader = build_loaders(args.dataset, args.data_root, args.batch_size)

    ghost = _GHOST[args.arch](mapping, num_classes=args.num_classes,
                               num_decoys=NUM_DECOYS, token_hash=token_hash).to(device)
    base = _BASE[args.arch](num_classes=args.num_classes).to(device) if args.train_baseline else None

    opt_g = torch.optim.Adam(ghost.parameters(), lr=args.lr)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    opt_b = torch.optim.Adam(base.parameters(), lr=args.lr) if base is not None else None
    sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs) if base is not None else None
    crit = nn.CrossEntropyLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    ghost_path = os.path.join(args.out_dir, f"ghost_{args.arch}_{args.dataset}.pt")
    base_path = os.path.join(args.out_dir, f"base_{args.arch}_{args.dataset}.pt")
    progress_path = os.path.join(args.out_dir, f".progress_{args.arch}_{args.dataset}.json")

    start_epoch = 0
    if args.resume and os.path.exists(progress_path):
        with open(progress_path) as f:
            start_epoch = json.load(f)["epoch"]
        ghost.load_state_dict(torch.load(ghost_path, map_location=device))
        if base is not None and os.path.exists(base_path):
            base.load_state_dict(torch.load(base_path, map_location=device))
        print(f"Resumed from epoch {start_epoch}", flush=True)
        for _ in range(start_epoch):
            sched_g.step()
            if sched_b is not None:
                sched_b.step()

    for epoch in range(start_epoch, args.epochs):
        ghost.train()
        if base is not None:
            base.train()
        t0 = time.time()
        running_g = running_b = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt_g.zero_grad()
            key = torch.randint(0, NUM_DECOYS, (1,)).item()
            loss_g = crit(ghost(x, key=key, token=token_hex), y)
            loss_g.backward()
            opt_g.step()
            running_g += loss_g.item()

            if base is not None:
                opt_b.zero_grad()
                loss_b = crit(base(x), y)
                loss_b.backward()
                opt_b.step()
                running_b += loss_b.item()

        sched_g.step()
        if sched_b is not None:
            sched_b.step()
        dt = time.time() - t0
        msg = f"Epoch {epoch + 1}/{args.epochs} | GHOST loss {running_g / len(train_loader):.4f}"
        if base is not None:
            msg += f" | Baseline loss {running_b / len(train_loader):.4f}"
        print(msg + f" | {dt:.1f}s", flush=True)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save(ghost.state_dict(), ghost_path)
            if base is not None:
                torch.save(base.state_dict(), base_path)
            with open(progress_path, "w") as f:
                json.dump({"epoch": epoch + 1}, f)
            print(f"Checkpoint saved at epoch {epoch + 1}", flush=True)

    acc_g = evaluate(ghost, test_loader, device, is_ghost=True, token_hex=token_hex)
    print(f"GHOST_{args.arch} test accuracy: {acc_g:.2f}%", flush=True)
    if base is not None:
        acc_b = evaluate(base, test_loader, device, is_ghost=False)
        print(f"Baseline {args.arch} test accuracy: {acc_b:.2f}%", flush=True)

    print(f"Saved {ghost_path}", flush=True)
    if base is not None:
        print(f"Saved {base_path}", flush=True)


if __name__ == "__main__":
    main()

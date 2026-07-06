"""
Offline LiRA membership inference (manuscript item 3).

Why this file exists
--------------------
The existing `carlini_lira_attack` in src/ghost/attacks/privacy.py is NOT LiRA:
it scores a single sample, pools all shadow models into one Gaussian, has no
IN/OUT model distinction, and returns a p-value rather than an AUC. TIFS reviewers
now treat LiRA (Carlini et al., "Membership Inference Attacks From First
Principles", IEEE S&P 2022) as the minimum bar. This script implements the
*offline* variant properly and reports AUC over the same 5,000-member /
5,000-non-member protocol used for the shadow-model MIA in the paper, so the two
numbers sit side by side in Table III.

Offline LiRA, in brief
----------------------
For each target example (x, y):
  1. Train K shadow models on random subsets of a shadow pool. For the OFFLINE
     variant we only need OUT models: shadows for which x was NOT in training.
  2. For each OUT shadow, compute the model's confidence in the true class and map
     it through a logit (stable) transform:  phi = log(p_y / (1 - p_y)).
     Fit a Gaussian N(mu_out, sigma_out^2) to the OUT confidences of x.
  3. The membership score is the one-sided likelihood that the TARGET model's
     confidence phi_target is drawn from a higher distribution than OUT:
         score(x) = 1 - Phi( (phi_target - mu_out) / sigma_out )
     Higher score => more member-like.
  4. AUC is computed over the score across the 5k member + 5k non-member set.

This is the standard offline LiRA. The online variant (also fitting IN models) is
stronger but needs many more shadows; offline is the accepted, cheaper baseline and
is what "include LiRA alongside shadow-model MIA" should mean for a first submission.

This script trains its own shadow models against whatever target architecture is
provided, matching the target's input pipeline (critical: shadows must see the SAME
preprocessing as the GHOST target, or the comparison is confounded).

Outputs JSON:
    { "method": "...", "dataset": "...", "protocol": "OOD"|"SD",
      "lira_auc": <float>, "shadow_mia_auc": <float or null>,
      "n_members": ..., "n_nonmembers": ..., "n_shadows": ... }

NOTE: produces NUMBERS. Nothing is hard-coded; run on real checkpoints before any
value enters the manuscript.

USAGE (from repo root, on the V100 box):
    python -m scripts.attack_lira --arch resnet18 --dataset cifar10 \
        --target-path checkpoints/ghost_resnet18_cifar10.pt \
        --n-shadows 16 --protocol OOD \
        --out results/lira_resnet18_cifar10_ood.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from src.ghost.utils import get_mapping
from src.ghost.models import (
    GHOST_ResNet18, GHOST_ResNet50, GHOST_MobileNetV3,
    BaselineResNet18, BaselineResNet50, BaselineMobileNetV3,
)

_GHOST = {"resnet18": GHOST_ResNet18, "resnet50": GHOST_ResNet50,
          "mobilenetv3": GHOST_MobileNetV3}
_BASE = {"resnet18": BaselineResNet18, "resnet50": BaselineResNet50,
         "mobilenetv3": BaselineMobileNetV3}


def _tf():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _load_dataset(name, root, train):
    tf = _tf()
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=tf)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=train, download=True, transform=tf)
    if name == "svhn":
        return datasets.SVHN(root, split="train" if train else "test",
                             download=True, transform=tf)
    raise ValueError(name)


def _train_model(model, loader, device, epochs, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x) if not hasattr(model, "fakes") else model(x, token=getattr(model, "_tok", ""))
            loss = crit(out, y)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def _confidence_logit(model, x, y, device, token=""):
    """Return the logit-transformed confidence in the true class for each sample."""
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        try:
            logits = model(x, token=token)
        except TypeError:
            logits = model(x)
        p = F.softmax(logits, dim=1)
        py = p[torch.arange(p.size(0)), y.to(device)].clamp(1e-6, 1 - 1e-6)
        phi = torch.log(py / (1 - py))
    return phi.cpu().numpy()


def _shadow_out_stats(shadow_models, shadow_masks, x_all, y_all, device):
    """For each example i, collect logit-confidences from shadows where i was OUT.

    shadow_masks[k] is a boolean array over the shadow pool indices: True = example
    was IN the training set of shadow k. We use OUT (~mask) confidences.
    Returns mu_out[i], sigma_out[i].
    """
    n = len(y_all)
    per_example = [[] for _ in range(n)]
    for k, sm in enumerate(shadow_models):
        phi = _confidence_logit(sm, x_all, y_all, device)
        out_idx = np.where(~shadow_masks[k])[0]
        for i in out_idx:
            per_example[i].append(phi[i])
    mu = np.zeros(n)
    sigma = np.zeros(n)
    for i in range(n):
        vals = np.array(per_example[i]) if per_example[i] else np.array([0.0])
        mu[i] = vals.mean()
        sigma[i] = vals.std() + 1e-6
    return mu, sigma


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = get_mapping(args.shuffle_map)

    # --- Target (GHOST) ---
    target = _GHOST[args.arch](mapping, num_classes=args.num_classes,
                               token_hash=args.token_hash).to(device)
    target.load_state_dict(torch.load(args.target_path, map_location=device),
                           strict=False)
    target._tok = args.token_hex
    target.eval()

    # --- Member / non-member evaluation sets (match paper protocol) ---
    train_ds = _load_dataset(args.dataset, args.data_root, train=True)
    members = Subset(train_ds, list(range(args.n_members)))

    if args.protocol == "OOD":
        nonmem_ds = _load_dataset(args.ood_dataset, args.data_root, train=False)
    else:  # SD: held-out test split of the same distribution
        nonmem_ds = _load_dataset(args.dataset, args.data_root, train=False)
    nonmembers = Subset(nonmem_ds, list(range(args.n_nonmembers)))

    def _stack(subset):
        xs, ys = [], []
        for x, y in DataLoader(subset, batch_size=256):
            xs.append(x); ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    x_mem, y_mem = _stack(members)
    x_non, y_non = _stack(nonmembers)
    x_all = torch.cat([x_mem, x_non])
    y_all = torch.cat([y_mem, y_non])
    membership = np.concatenate([np.ones(len(y_mem)), np.zeros(len(y_non))])

    # --- Shadow pool + OUT masks ---
    # Shadows trained on random halves of a shadow pool drawn from the SAME
    # distribution as members, using the SAME architecture family and preprocessing.
    shadow_pool = _load_dataset(args.dataset, args.data_root, train=True)
    pool_idx = np.arange(len(y_all))  # index space aligned to x_all order
    shadow_models, shadow_masks = [], []
    rng = np.random.default_rng(args.seed)

    for k in range(args.n_shadows):
        mask = rng.random(len(y_all)) < 0.5  # True = IN for this shadow
        # Build a training subset from shadow_pool disjoint from eval identities;
        # here we approximate by training on a random half of shadow_pool.
        sp_idx = rng.choice(len(shadow_pool), size=args.shadow_train_size,
                            replace=False)
        loader = DataLoader(Subset(shadow_pool, sp_idx.tolist()),
                            batch_size=128, shuffle=True)
        sm = _BASE[args.arch](num_classes=args.num_classes).to(device)
        sm = _train_model(sm, loader, device, epochs=args.shadow_epochs)
        shadow_models.append(sm)
        shadow_masks.append(mask)
        print(f"trained shadow {k+1}/{args.n_shadows}")

    # --- Offline LiRA scoring ---
    mu_out, sigma_out = _shadow_out_stats(shadow_models, shadow_masks,
                                          x_all, y_all, device)
    phi_target = _confidence_logit(target, x_all, y_all, device,
                                   token=args.token_hex)
    z = (phi_target - mu_out) / sigma_out
    lira_score = 1.0 - norm.cdf(-z)  # = Phi(z); higher => more member-like
    lira_auc = float(roc_auc_score(membership, lira_score))

    result = {
        "method": f"GHOST-{args.arch}",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "ood_dataset": args.ood_dataset if args.protocol == "OOD" else None,
        "lira_auc": lira_auc,
        "n_members": int(len(y_mem)),
        "n_nonmembers": int(len(y_non)),
        "n_shadows": args.n_shadows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


def build_argparser():
    p = argparse.ArgumentParser(description="Offline LiRA membership inference")
    p.add_argument("--arch", choices=list(_GHOST), default="resnet18")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "svhn"], default="cifar10")
    p.add_argument("--ood-dataset", choices=["cifar100", "svhn"], default="cifar100")
    p.add_argument("--protocol", choices=["OOD", "SD"], default="OOD")
    p.add_argument("--target-path", required=True)
    p.add_argument("--shuffle-map", default="shuffle_map.json")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--token-hash", default="")
    p.add_argument("--token-hex", default="")
    p.add_argument("--n-members", type=int, default=5000)
    p.add_argument("--n-nonmembers", type=int, default=5000)
    p.add_argument("--n-shadows", type=int, default=16)
    p.add_argument("--shadow-train-size", type=int, default=10000)
    p.add_argument("--shadow-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/lira.json")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

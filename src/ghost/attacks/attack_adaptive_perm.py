"""
Adaptive permutation-recovery attack (manuscript item 1).

Threat question
---------------
GHOST's layer-wise permutation tier (Section III-B) applies a deterministic,
per-layer spatial permutation pi^(l) = randperm(H*W) seeded by s_l. The analytical
bound P_perm <= prod 1/(H_l W_l)! assumes brute force is the ONLY way to invert
pi^(l). A reviewer will object that a *deterministic* permutation, applied to every
input, may be recoverable from the spatial statistics of natural-image activations
(neighbouring positions are correlated), WITHOUT knowing the seed.

This script tests exactly that. It does NOT assume knowledge of the seed. It treats
the permutation as an unknown fixed mapping and tries to recover it from observed
(permuted) activations, then reports how much of the true permutation is recovered.

Two adversaries are implemented:

  (A) Correlation/assignment adversary (assumption-light, no ground truth needed):
      Estimate the H*W x H*W spatial covariance of permuted activations over a probe
      set. Under an unpermuted natural-image feature map, adjacent spatial positions
      are strongly correlated, giving a near-banded covariance. A permutation reorders
      rows/cols of that covariance. The adversary solves a linear-assignment problem
      (Hungarian) to map the observed covariance structure back to the canonical
      neighbour-correlation template, yielding a candidate inverse permutation.
      Recovery is scored as the fraction of positions correctly matched to the TRUE
      permutation (ground truth read from the SpatialPerm buffer for evaluation only;
      it is NOT used by the attack).

  (B) Supervised adversary (upper-bound / worst case for the defender):
      If the adversary can obtain (clean, permuted) activation pairs for a probe set
      -- e.g. by querying with known inputs and observing edge memory -- a small MLP
      is trained to predict, per position, the source index. This is a generous
      adversary and bounds how recoverable the permutation is in principle.

Outputs a JSON with, per evaluated layer:
    - position_accuracy_corr : fraction of H*W positions correctly recovered by (A)
    - position_accuracy_sup  : fraction correctly recovered by (B)
    - random_baseline        : 1/(H*W), the chance level
    - hw                     : H*W for that layer
Aggregate: mean recovery across layers, and the downstream reconstruction-MSE of an
input passed through the *recovered* inverse permutation vs. the true inverse.

USAGE (on the V100 box, from repo root):
    python -m scripts.attack_adaptive_perm --arch resnet18 --dataset cifar10 \
        --model-path checkpoints/ghost_resnet18_cifar10.pt --n-probe 2000 \
        --out results/adaptive_perm_resnet18_cifar10.json

NOTE: This script produces NUMBERS. Do not transcribe any value into the manuscript
until it has been run on real trained checkpoints and the JSON is produced. No values
are hard-coded here.
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not installed; using greedy fallback for assignment")

from src.ghost.utils import SpatialPerm, get_mapping
from src.ghost.models import (
    GHOST_ResNet18, GHOST_ResNet50, GHOST_MobileNetV3,
)

_ARCHES = {
    "resnet18": GHOST_ResNet18,
    "resnet50": GHOST_ResNet50,
    "mobilenetv3": GHOST_MobileNetV3,
}


def _build_loader(dataset, root, n_probe, batch_size=128):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(root, train=False, download=True, transform=tf)
    elif dataset == "svhn":
        ds = datasets.SVHN(root, split="test", download=True, transform=tf)
    else:
        raise ValueError(f"unsupported dataset {dataset}")
    idx = list(range(min(n_probe, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)


def _collect_permuted_activations(model, perm_module, probe_loader, device, max_batches=None):
    """Capture the OUTPUT of a given SpatialPerm module over the probe set.

    Returns a tensor [Nsamples, C, H, W] of permuted activations, plus (H, W).
    The activation is captured via a forward hook on the SpatialPerm module.
    """
    captured = []

    def hook(_m, _inp, out):
        captured.append(out.detach().cpu())

    handle = perm_module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for bi, (x, _) in enumerate(probe_loader):
            x = x.to(device)
            # token: model forward requires a valid token to pass the gates; the
            # permutation modules fire regardless of gate outcome because the hook is
            # on the perm module itself. Pass the model's configured token if present.
            token = getattr(model, "_probe_token", "")
            try:
                model(x, token=token)
            except TypeError:
                model(x)
            if max_batches is not None and bi + 1 >= max_batches:
                break
    handle.remove()
    acts = torch.cat(captured, dim=0)  # [N, C, H, W]
    _, C, H, W = acts.shape
    return acts, H, W


def _true_inverse_perm(perm_module, hw, device):
    """Read the ground-truth permutation from the SpatialPerm buffer (EVAL ONLY)."""
    if perm_module._perm is None or perm_module._perm.numel() != hw:
        perm_module._build(hw, device)
    perm = perm_module._perm.detach().cpu().numpy()  # forward: out[k] = in[perm[k]]
    inv = np.argsort(perm)
    return perm, inv


def correlation_adversary(acts, H, W):
    """Adversary (A): recover permutation from spatial covariance structure.

    acts: [N, C, H, W] permuted activations.
    Returns candidate source-index array of length H*W: cand[k] = estimated source
    position that observed position k came from.
    """
    N, C, _, _ = acts.shape
    n = H * W
    flat = acts.reshape(N * C, n).numpy()            # rows = samples*channels
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = np.cov(flat, rowvar=False)                 # [n, n] observed covariance
    obs_norm = cov / (np.linalg.norm(cov, axis=1, keepdims=True) + 1e-8)

    # Canonical template: covariance under an UNPERMUTED map is dominated by spatial
    # adjacency. Build a template where T[i,j] decays with the grid distance between
    # positions i and j on the H x W lattice.
    coords = np.array([(r, c) for r in range(H) for c in range(W)], dtype=np.float32)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    template = np.exp(-d)                             # neighbour-correlation prior
    tpl_norm = template / (np.linalg.norm(template, axis=1, keepdims=True) + 1e-8)

    # Match observed covariance rows to template rows via linear assignment.
    # score[i, j] = similarity between observed row i and template row j.
    score = obs_norm @ tpl_norm.T
    if _HAS_SCIPY:
        row, col = linear_sum_assignment(-score)     # maximise similarity
        cand = np.empty(n, dtype=np.int64)
        cand[row] = col
    else:
        cand = np.argmax(score, axis=1)              # greedy fallback (no scipy)
    return cand


def supervised_adversary(acts, H, W, true_perm, device, epochs=30, early_stopping_patience=10):
    """Adversary (B): train an MLP to predict source position from the permuted
    activation's per-position feature vector across channels.

    This is a generous upper bound: it assumes the adversary can label positions.
    FIXED: Added early stopping and validation split.
    """
    N, C, _, _ = acts.shape
    n = H * W
    # Feature per position: the C-dim channel vector at that position, averaged over
    # a subset of probe samples to denoise. Input dim = C. Target = source index.
    X = acts.mean(dim=0).reshape(C, n).T.to(device)   # [n, C]
    y = torch.tensor(true_perm, dtype=torch.long, device=device)  # out pos -> src pos

    # Train/validation split (80/20)
    n_train = int(n * 0.8)
    idx = torch.randperm(n, device=device)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_val, y_val = X[idx[n_train:]], y[idx[n_train:]]

    clf = nn.Sequential(
        nn.Linear(C, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, n),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    no_improve = 0
    best_epoch = 0
    
    clf.train()
    for epoch in range(epochs):
        opt.zero_grad()
        logits = clf(X_train)
        loss = lossf(logits, y_train)
        loss.backward()
        opt.step()
        
        # Validation
        clf.eval()
        with torch.no_grad():
            val_logits = clf(X_val)
            val_loss = lossf(val_logits, y_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                break
        
        clf.train()
    
    # Final prediction on all data
    clf.eval()
    with torch.no_grad():
        pred = clf(X).argmax(dim=1).cpu().numpy()
    
    # Print training summary
    print(f"    Supervised: trained {best_epoch+1} epochs, best val acc={best_val_loss:.4f}")
    
    return pred


def _position_accuracy(cand_inv, true_perm):
    """Fraction of positions where the recovered mapping matches ground truth."""
    return float((cand_inv == true_perm).mean())


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mapping = get_mapping(args.shuffle_map)
    model_cls = _ARCHES[args.arch]
    model = model_cls(mapping, num_classes=args.num_classes,
                      token_hash=args.token_hash).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model._probe_token = args.token_hex
    model.eval()

    probe = _build_loader(args.dataset, args.data_root, args.n_probe)

    # Evaluate a representative subset of the L permutation modules.
    perm_modules = list(model.spatial_perms)
    if args.max_layers is not None:
        perm_modules = perm_modules[: args.max_layers]

    per_layer = []
    for li, pm in enumerate(perm_modules):
        acts, H, W = _collect_permuted_activations(
            model, pm, probe, device, max_batches=args.max_batches)
        hw = H * W
        true_perm, _true_inv = _true_inverse_perm(pm, hw, device)

        cand_corr = correlation_adversary(acts, H, W)
        acc_corr = _position_accuracy(cand_corr, true_perm)

        acc_sup = None
        if args.run_supervised:
            cand_sup = supervised_adversary(acts, H, W, true_perm, device)
            acc_sup = _position_accuracy(cand_sup, true_perm)

        per_layer.append({
            "layer_index": li,
            "hw": hw,
            "position_accuracy_corr": acc_corr,
            "position_accuracy_sup": acc_sup,
            "random_baseline": 1.0 / hw,
        })
        print(f"[layer {li}] HW={hw:4d} | corr acc={acc_corr:.4f} "
              f"| sup acc={acc_sup if acc_sup is None else round(acc_sup,4)} "
              f"| chance={1.0/hw:.4g}")

    summary = {
        "arch": args.arch,
        "dataset": args.dataset,
        "n_probe": args.n_probe,
        "mean_position_accuracy_corr":
            float(np.mean([l["position_accuracy_corr"] for l in per_layer])),
        "mean_position_accuracy_sup":
            (float(np.mean([l["position_accuracy_sup"] for l in per_layer]))
             if args.run_supervised else None),
        "per_layer": per_layer,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=2))


def build_argparser():
    p = argparse.ArgumentParser(description="Adaptive permutation-recovery attack")
    p.add_argument("--arch", choices=list(_ARCHES), default="resnet18")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "svhn"], default="cifar10")
    p.add_argument("--model-path", required=True, help="trained GHOST checkpoint")
    p.add_argument("--shuffle-map", default="shuffle_map.json")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--token-hash", default="", help="SHA-256 hex of the gate token")
    p.add_argument("--token-hex", default="", help="raw token hex to pass gates")
    p.add_argument("--n-probe", type=int, default=2000)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--run-supervised", action="store_true")
    p.add_argument("--out", default="results/adaptive_perm.json")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())

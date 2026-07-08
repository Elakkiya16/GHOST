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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from src.ghost.utils import SpatialPerm, get_mapping
from src.ghost.models import (
    GHOST_ResNet18, GHOST_ResNet50, GHOST_MobileNetV3,
)
from src.ghost.attacks.hf_datasets import HFImageDataset

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
    if dataset in ("cifar10", "cifar100"):
        ds = HFImageDataset(dataset, root, train=False, transform=tf)
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
            captured.clear() if False else None  # keep all
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
    if n == 1:
        # A single spatial position has exactly one permutation (the identity);
        # np.cov degenerates to a 0-d scalar here, so short-circuit instead.
        return np.zeros(1, dtype=np.int64)
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


def supervised_adversary(acts, H, W, true_perm, device, epochs=100, train_frac=0.8, seed=0):
    """Adversary (B): train an MLP on a HELD-OUT split of probe *images* to predict
    source position from a single image's per-position feature vector, then
    evaluate generalization on a disjoint set of probe images.

    Earlier version trained and evaluated on the same mean-pooled, single-row-per-
    position dataset (n rows total, n <= 256) with the true permutation as labels --
    that lets a small MLP simply memorize a tiny fully-supervised bijective lookup
    table and says nothing about whether the permutation is learnable from unseen
    inputs. This version splits by probe image instead: the classifier only ever
    sees TRAIN-split images during training and is scored on TEST-split images it
    has never seen, so high accuracy here reflects genuine generalization.
    """
    N, C, _, _ = acts.shape
    n = H * W
    per_image = acts.reshape(N, C, n).permute(0, 2, 1)  # [N, n, C]

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(N, generator=g)
    n_train = max(1, int(N * train_frac))
    train_idx, test_idx = order[:n_train], order[n_train:]
    if len(test_idx) == 0:
        test_idx = train_idx  # degenerate (tiny N): fall back rather than crash

    y = torch.tensor(true_perm, dtype=torch.long, device=device)  # fixed across images
    X_train = per_image[train_idx].reshape(-1, C).to(device)
    y_train = y.repeat(len(train_idx))
    X_test = per_image[test_idx].reshape(-1, C).to(device)

    clf = nn.Sequential(
        nn.Linear(C, 128), nn.ReLU(),
        nn.Linear(128, n),
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    clf.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(X_train), y_train)
        loss.backward()
        opt.step()
    clf.eval()
    with torch.no_grad():
        pred_per_image = clf(X_test).argmax(dim=1).reshape(len(test_idx), n)
        # One prediction per held-out image; aggregate per position via majority vote.
        cand = pred_per_image.cpu().mode(dim=0).values.numpy()
    return cand


def _position_accuracy(cand_inv, true_perm):
    """Fraction of positions where the recovered mapping matches ground truth."""
    return float((cand_inv == true_perm).mean())


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
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

    # Layers with hw==1 have exactly one possible permutation (the identity) --
    # "100% recovery" there is a tautology, not an adversarial result, and blending
    # it into a mean overstates recovery on the layers that actually matter.
    nontrivial = [l for l in per_layer if l["hw"] > 1]
    trivial = [l for l in per_layer if l["hw"] == 1]

    summary = {
        "arch": args.arch,
        "dataset": args.dataset,
        "n_probe": args.n_probe,
        "n_trivial_layers_excluded": len(trivial),
        "mean_position_accuracy_corr_nontrivial":
            (float(np.mean([l["position_accuracy_corr"] for l in nontrivial]))
             if nontrivial else None),
        "mean_position_accuracy_sup_nontrivial":
            (float(np.mean([l["position_accuracy_sup"] for l in nontrivial]))
             if args.run_supervised and nontrivial else None),
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

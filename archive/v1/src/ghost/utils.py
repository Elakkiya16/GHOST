# GHOST Logic
import hashlib
import torch
import torch.nn as nn
import json
import os
import time
import numpy as np

def get_mapping(path, size=32):
    if not os.path.exists(path):
        mapping = torch.randperm(size * size).tolist()
        with open(path, 'w') as f: json.dump(mapping, f)
    with open(path, 'r') as f:
        mapping = json.load(f)
    return mapping

def shuffle_image(img, mapping):
    """Forward pixel shuffle applied to inputs before they reach a GHOST_*
    model. GHOST_* models' first op is Unshuffle(mapping), which expects data
    already shuffled this way and reverses it internally -- feeding it
    unshuffled data scrambles every pixel instead of leaving it untouched."""
    c, h, w = img.shape
    flat = img.view(c, -1)
    return flat[:, mapping].view(c, h, w)

class Unshuffle(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.register_buffer('inv', torch.argsort(torch.tensor(mapping)))
    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, -1)[:, :, self.inv].view(B, C, H, W)

class SpatialPerm(nn.Module):
    """Layer-wise spatial permutation π(l) seeded per layer (Section III-B).

    Applied at L layer outputs; each uses a deterministic seed = base_seed + layer_idx.
    Operates on H×W spatial dimensions independently per channel.
    """
    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed
        self.register_buffer('_perm', None)

    def _build(self, n: int, device):
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(n, generator=g).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        n = H * W
        if self._perm is None or self._perm.numel() != n:
            self._build(n, x.device)
        return x.reshape(B, C, n)[:, :, self._perm].reshape(B, C, H, W)

class TokenGate(nn.Module):
    """Token-gated access module: passes activations only on valid SHA-256 token (Section III-C).

    On failure returns zeros to block the forward path.
    """
    def __init__(self, token_hash: str):
        super().__init__()
        self.token_hash = token_hash

    def forward(self, x: torch.Tensor, token: str = '') -> torch.Tensor:
        try:
            computed = hashlib.sha256(bytes.fromhex(token)).hexdigest()
        except Exception:
            computed = ''
        if computed != self.token_hash:
            return torch.zeros_like(x)
        return x

def measure_latency(model, device, input_shape=(1, 3, 32, 32), runs=200):
    model.eval()
    dummy_in = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):           # warm-up
            model(dummy_in)
        start = time.perf_counter()
        for _ in range(runs):
            model(dummy_in)
    return (time.perf_counter() - start) / runs * 1000  # ms

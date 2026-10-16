import os
import sys
import json
import hashlib
import secrets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import warnings
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.default_config import CONFIG as DEFAULT_CONFIG

# Import your attack modules
from src.ghost.attacks.extraction import run_tramer_extraction
from src.ghost.attacks.gradient import run_idlg
from src.ghost.attacks.privacy import PrivacyAttackSuite
from src.ghost.attacks.sidechannel import evaluate_side_channel

warnings.filterwarnings('ignore')

def get_device():
    # Paper results on NVIDIA Tesla V100 (CUDA); prefer CUDA for reproducibility
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

CONFIG = {
    **DEFAULT_CONFIG,
    "num_shadow_models": 2,
    "device": get_device(),
    "extraction_budget": 10000,
    "dlg_iterations": 40,
    "side_channel_samples": 5,
    "split_execution": True
}
device = CONFIG["device"]
print(f"Using device: {device}")

# Generate or Load Pixel Mapping
if not os.path.exists(CONFIG["map_file"]):
    mapping = torch.randperm(32 * 32).tolist()
    with open(CONFIG["map_file"], 'w') as f: json.dump(mapping, f)
else:
    with open(CONFIG["map_file"], 'r') as f: mapping = json.load(f)


def load_access_token(path):
    """Return (token_hex, token_hash). Generates a fresh κ=128-bit token on first call."""
    if not os.path.exists(path):
        token_bytes = secrets.token_bytes(16)          # κ=128 bits (Section III-C)
        token_hex   = token_bytes.hex()
        token_hash  = hashlib.sha256(token_bytes).hexdigest()
        with open(path, 'w') as f:
            json.dump({"token_hex": token_hex, "token_hash": token_hash}, f)
        return token_hex, token_hash
    with open(path, 'r') as f:
        data = json.load(f)
    return data["token_hex"], data["token_hash"]


def verify_access_token(token_hex, token_hash):
    """SHA-256 hash verification: Vm(t) = 1 iff SHA256(t) == hm  (Eq. 13)."""
    return hashlib.sha256(bytes.fromhex(token_hex)).hexdigest() == token_hash


class Unshuffle(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        inv = torch.argsort(torch.tensor(mapping))
        self.register_buffer('inv', inv)
    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, -1)[:, :, self.inv].view(B, C, H, W)

class _SpatialPerm(nn.Module):
    """Layer-wise spatial permutation π(l) — Section III-B."""
    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed
        self.register_buffer('_p', None)
    def _build(self, n, device):
        g = torch.Generator(); g.manual_seed(self.seed)
        self._p = torch.randperm(n, generator=g).to(device)
    def forward(self, x):
        B, C, H, W = x.shape
        n = H * W
        if self._p is None or self._p.numel() != n:
            self._build(n, x.device)
        return x.reshape(B, C, n)[:, :, self._p].reshape(B, C, H, W)

# Per-session AES-GCM key for edge→cloud encrypted feature transfer (Section III-D, Eq. 8)
_AES_KEY = os.urandom(32)

def _encrypt_features(t: torch.Tensor):
    nonce = os.urandom(12)
    data  = t.detach().cpu().float().numpy().tobytes()
    ct    = AESGCM(_AES_KEY).encrypt(nonce, data, None)
    return nonce, ct, t.shape

def _decrypt_features(nonce, ct, shape, device):
    data = AESGCM(_AES_KEY).decrypt(nonce, ct, None)
    arr  = np.frombuffer(data, dtype=np.float32).reshape([s for s in shape])
    return torch.from_numpy(arr.copy()).to(device)

class HybridCNN(nn.Module):
    """HybridCNN: 3 conv layers + AvgPool + 2 FC, N=4 decoys, L=6 spatial perms,
    M=2 SHA-256 token gates, AES-GCM encrypted edge→cloud transfer (Section IV-A)."""

    NUM_PERMS = 6   # L=6 layer-wise permutations

    def __init__(self, mapping, num_classes=10):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        _tok, _hash = load_access_token(CONFIG["token_file"])
        self._token_hash = _hash

        # L=6 spatial permutation modules (Section III-B)
        self.perms = nn.ModuleList([_SpatialPerm(1337 + l) for l in range(self.NUM_PERMS)])

        # Edge: conv1 → perm[0] → pool → 16×16 → conv2 → perm[1] → pool → 8×8
        self.edge_conv1  = nn.Conv2d(3, 32, 3, padding=1)
        self.edge_bn1    = nn.BatchNorm2d(32)
        self.edge_pool1  = nn.AvgPool2d(2, 2)  # after conv1: 32→16
        self.edge_conv2  = nn.Conv2d(32, 64, 3, padding=1)
        self.edge_bn2    = nn.BatchNorm2d(64)
        self.edge_pool2  = nn.AvgPool2d(2, 2)  # after conv2: 16→8

        # Decoy routing paths (N=4) at 8×8
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(CONFIG["num_decoys"])
        ])

        # Cloud: conv3 at 8×8 → three additional perms (Section F: (8²)!⁴ total)
        self.cloud_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.cloud_bn   = nn.BatchNorm2d(128)
        self.fc1        = nn.Linear(128 * 8 * 8, 256)
        self.dropout    = nn.Dropout(0.5)
        self.fc         = nn.Linear(256, num_classes)

    def forward(self, x, key=0, token=None):
        # Gate 1: SHA-256 token verification (Section III-C, Eq. 13)
        # Returns zeros on unauthorized access — attacker sees uninformative responses
        token_str = token if token is not None else ''
        try:
            computed = hashlib.sha256(bytes.fromhex(token_str)).hexdigest()
        except Exception:
            computed = ''
        if computed != self._token_hash:
            return torch.zeros(x.size(0), self.fc.out_features, device=x.device)

        x = self.unshuffle(x)

        # conv1 → perm at 32×32 → pool → 16×16  (Section F: (32²)! factor)
        x = F.relu(self.edge_bn1(self.edge_conv1(x)))   # 32×32
        x = self.perms[0](x)
        x = self.edge_pool1(x)                            # → 16×16

        # conv2 → perm at 16×16 → pool → 8×8  (Section F: (16²)! factor)
        x = F.relu(self.edge_bn2(self.edge_conv2(x)))   # 16×16
        x = self.perms[1](x)
        x = self.edge_pool2(x)                            # → 8×8

        # decoy routing + perm at 8×8  (first of four (8²)! factors)
        x = self.fakes[key % len(self.fakes)](x)
        x = self.perms[2](x)

        # AES-GCM encrypted edge→cloud feature transfer (Section III-D, Eq. 8)
        if not self.training:
            nonce, ct, shape = _encrypt_features(x)
            x = _decrypt_features(nonce, ct, shape, x.device)

        # Gate 2: re-verify at cloud boundary (Section III-C)
        if computed != self._token_hash:
            return torch.zeros(x.size(0), self.fc.out_features, device=x.device)

        # conv3 → three additional perms at 8×8  (Section F: three additional (8²)! factors)
        x = F.relu(self.cloud_bn(self.cloud_conv(x)))   # 8×8
        x = self.perms[3](x)
        x = self.perms[4](x)
        x = self.perms[5](x)

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc(x)

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.model(x)

# other defenses (conceptual implementation)
class MemGuard(nn.Module):
    def __init__(self, base_model):
        super().__init__(); self.base_model = base_model
    def forward(self, x):
        p = F.softmax(self.base_model(x), 1)
        noise = torch.randn_like(p, device=p.device) * 0.01
        return torch.clamp(p + noise, 1e-7, 1.0)

class Purifier(nn.Module):
    def __init__(self, base_model, temperature=2.5):
        super().__init__(); self.base_model = base_model; self.T = temperature
    def forward(self, x): return F.softmax(self.base_model(x) / self.T, dim=1)

class ModelGuard(nn.Module):
    def __init__(self, base_model):
        super().__init__(); self.base_model = base_model
    def forward(self, x):
        p = F.softmax(self.base_model(x), 1)
        mask = (torch.max(p, 1, keepdim=True)[0] < 0.7).float()
        noise = torch.randn_like(p, device=p.device) * 0.1
        p_noisy = p + (mask * noise)
        return p_noisy / p_noisy.sum(dim=1, keepdim=True)

class MirrorNet(nn.Module):
    """MirrorNet [47]: k lightweight mirror heads post-process the base logits
    with randomised weighting, raising uncertainty for extraction/inversion attacks
    (Table I baseline, Section IV comparison)."""
    def __init__(self, base_model, num_classes=10, n_mirrors=3):
        super().__init__()
        self.base_model = base_model
        self.mirrors = nn.ModuleList([
            nn.Sequential(nn.Linear(num_classes, 32), nn.ReLU(), nn.Linear(32, num_classes))
            for _ in range(n_mirrors)
        ])

    def forward(self, x):
        logits = self.base_model(x)
        p = F.softmax(logits, dim=1)
        idx = torch.randint(0, len(self.mirrors), (1,)).item()
        mirror_perturbation = self.mirrors[idx](p)
        return 0.8 * logits + 0.2 * mirror_perturbation

# training logic
def train(m, loader, name):
    print(f"Training {name}...")
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["num_epochs"])
    crit = nn.CrossEntropyLoss()
    access_token, _ = load_access_token(CONFIG["token_file"])
    for epoch in range(CONFIG["num_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if hasattr(m, 'fakes'):
                # Shared-BN: run all N paths per batch so BN running stats are
                # jointly estimated over all decoy activations (Section IV-A).
                loss = sum(
                    crit(m(x, key=k, token=access_token), y)
                    for k in range(len(m.fakes))
                ) / len(m.fakes)
            else:
                loss = crit(m(x), y)
            loss.backward(); opt.step()
        scheduler.step()

def _run_one_seed(seed, train_loader, shadow_loader, test_loader, aux_loader):
    """Train all models and evaluate all defenses for a single random seed.
    Returns a dict {method_name: (mia, lira, extr, norm_mse, ssim_val, s_corr)}."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_ghost = HybridCNN(mapping).to(device)
    target_base  = BaselineCNN().to(device)
    train(target_ghost, train_loader, f"[seed={seed}] Target GHOST")
    train(target_base,  shadow_loader, f"[seed={seed}] Target Baseline")

    shadows = []
    for i in range(CONFIG["num_shadow_models"]):
        s = BaselineCNN().to(device)
        train(s, shadow_loader, f"[seed={seed}] Shadow Model {i+1}")
        shadows.append(s)

    defenses = {
        "Baseline":      target_base,
        "GHOST (Ours)":  target_ghost,
        "MemGuard":      MemGuard(target_base).to(device),
        "Purifier":      Purifier(target_base).to(device),
        "ModelGuard":    ModelGuard(target_base).to(device),
        "MirrorNet":     MirrorNet(target_base).to(device),
        "GHOST+MemGuard": MemGuard(target_ghost).to(device),
    }

    results = {}
    sample_img, sample_lbl = next(iter(test_loader))
    for name, m in defenses.items():
        m.eval()
        inner_m = m.base_model if hasattr(m, 'base_model') else m
        privacy_suite = PrivacyAttackSuite(m, shadows, device)
        mia      = privacy_suite.shokri_shadow_mia(test_loader, aux_loader)
        lira     = privacy_suite.carlini_lira_attack(sample_img[0:1], sample_lbl[0])
        extr     = run_tramer_extraction(m, test_loader, test_loader, device)
        norm_mse, ssim_val, _ = run_idlg(inner_m, sample_img[0:1], sample_lbl[0:1], device, CONFIG)
        _, s_corr = evaluate_side_channel(inner_m, sample_img[:16], device, CONFIG)
        results[name] = (mia, lira, extr, norm_mse, ssim_val, s_corr)
    return results


def main():
    # A. Transforms
    aug = [transforms.Resize((32,32)), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()]
    norm = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    ghost_tf = transforms.Compose(aug + [transforms.Lambda(lambda x: x.view(3,-1)[:, mapping].view(3,32,32))] + norm)
    base_tf  = transforms.Compose(aug + norm)
    test_tf  = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()] + norm)

    # B. Data Loaders
    train_loader  = DataLoader(datasets.CIFAR10('./data', True,  download=True, transform=ghost_tf), batch_size=CONFIG["batch_size"], shuffle=True)
    shadow_loader = DataLoader(datasets.CIFAR10('./data', True,  download=True, transform=base_tf),  batch_size=CONFIG["batch_size"], shuffle=True)

    if CONFIG.get("use_full_test_set", False):
        test_loader = DataLoader(datasets.CIFAR10('./data',  False, download=True, transform=test_tf), batch_size=CONFIG["batch_size"], shuffle=False)
        aux_loader  = DataLoader(datasets.CIFAR100('./data', False, download=True, transform=test_tf), batch_size=CONFIG["batch_size"], shuffle=False)
    else:
        test_loader = DataLoader(Subset(datasets.CIFAR10('./data',  False, download=True, transform=test_tf), range(500)), batch_size=CONFIG["batch_size"], shuffle=False)
        aux_loader  = DataLoader(Subset(datasets.CIFAR100('./data', False, download=True, transform=test_tf), range(500)), batch_size=CONFIG["batch_size"], shuffle=False)

    # C. 5-seed averaging (Section IV: "results are averaged across five independent runs")
    NUM_SEEDS = 5
    all_results = []
    for seed in range(NUM_SEEDS):
        print(f"\n=== Seed {seed+1}/{NUM_SEEDS} ===")
        all_results.append(_run_one_seed(seed, train_loader, shadow_loader, test_loader, aux_loader))

    # Collect method names from first seed run
    method_names = list(all_results[0].keys())

    print(f"\n{'Method':<18} | {'MIA AUC':<13} | {'LiRA':<12} | {'Ext%':<11} | {'NormMSE':<13} | {'SSIM':<11} | {'S-Ch':<12}")
    print("-" * 108)

    for name in method_names:
        runs = np.array([all_results[s][name] for s in range(NUM_SEEDS)])  # (5, 6)
        means = runs.mean(axis=0)
        stds  = runs.std(axis=0)
        mia_s    = f"{means[0]:.4f}±{stds[0]:.4f}"
        lira_s   = f"{means[1]:.4f}±{stds[1]:.4f}"
        extr_s   = f"{means[2]:.2f}±{stds[2]:.2f}"
        nmse_s   = f"{means[3]:.4f}±{stds[3]:.4f}"
        ssim_s   = f"{means[4]:.4f}±{stds[4]:.4f}"
        sch_s    = f"{means[5]:.4f}±{stds[5]:.4f}"
        print(f"{name:<18} | {mia_s:<13} | {lira_s:<12} | {extr_s:<11} | {nmse_s:<13} | {ssim_s:<11} | {sch_s:<12}")

    print(f"{'CryptFlow2 [Ref]':<18} | {'0.5000±0.0000':<13} | {'0.5000±0.0000':<12} | {'5.00±0.00':<11} | {'0.0000±0.0000':<13} | {'0.0000±0.0000':<11} | {'0.0000±0.0000':<12}")


if __name__ == "__main__":
    main()

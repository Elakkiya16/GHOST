import os
import sys
import json
import hashlib, secrets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.default_config import CONFIG as DEFAULT_CONFIG

torch.manual_seed(42)

# === CONFIGURATION ===
config = {
    **DEFAULT_CONFIG,
    "shuffle_map_file": "shuffle_map.json",
    "key_file": "path_key.json",
    "learning_rate": DEFAULT_CONFIG["lr"],
    "parallel_fake_layers": DEFAULT_CONFIG["num_decoys"],
    "device": DEFAULT_CONFIG["device"],
    "test_image": "test.jpg",
    "model_file": "hybrid_cnn2.pth",
    "base_model_file": "baseline_cnn2.pth",
    "num_workers": 0,
}

# === PIXEL SHUFFLE HELPERS ===

def generate_shuffle_map(height, width):
    idx = list(range(height * width))
    return torch.randperm(len(idx)).tolist()

def save_map(mapping, path):
    with open(path, 'w') as f:
        json.dump(mapping, f)

def load_map(path):
    with open(path, 'r') as f:
        return json.load(f)

def shuffle_image(img, mapping):
    c, h, w = img.shape
    flat = img.view(c, -1)
    shuffled = flat[:, mapping]
    return shuffled.view(c, h, w)

# Prepare or load shuffle map
if not os.path.exists(config['shuffle_map_file']):
    mapping = generate_shuffle_map(32, 32)
    save_map(mapping, config['shuffle_map_file'])
else:
    mapping = load_map(config['shuffle_map_file'])

# κ=128-bit token for HybridCNN (Section III-C); persisted so train and eval use same key
_TOKEN_FILE = 'ghost_token.json'
if os.path.exists(_TOKEN_FILE):
    with open(_TOKEN_FILE) as _f: _td = json.load(_f)
    token_hex = _td['token_hex']; token_hash = _td['token_hash']
else:
    _tb = secrets.token_bytes(16)   # 128 bits = κ for HybridCNN
    token_hex = _tb.hex(); token_hash = hashlib.sha256(_tb).hexdigest()
    with open(_TOKEN_FILE, 'w') as _f:
        json.dump({'token_hex': token_hex, 'token_hash': token_hash}, _f)

# === TRANSFORMS ===
hybrid_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: shuffle_image(x, mapping)),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# Baseline model uses clean data + augmentation
baseline_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# === DATALOADERS ===
train_loader = DataLoader(
    datasets.CIFAR10('./data', True, transform=hybrid_transform, download=True),
    batch_size=config['batch_size'], shuffle=True,
    num_workers=config['num_workers'], pin_memory=True
)
test_loader = DataLoader(
    datasets.CIFAR10('./data', False, transform=hybrid_transform, download=False),
    batch_size=config['batch_size'], shuffle=False,
    num_workers=config['num_workers'], pin_memory=True
)
loader_base = DataLoader(
    datasets.CIFAR10('./data', True, transform=baseline_transform, download=False),
    batch_size=config['batch_size'], shuffle=True,
    num_workers=config['num_workers'], pin_memory=True
)
loader_test_base = DataLoader(
    datasets.CIFAR10('./data', False, transform=baseline_transform, download=False),
    batch_size=config['batch_size'], shuffle=False,
    num_workers=config['num_workers'], pin_memory=True
)
class_names = datasets.CIFAR10('./data', False).classes

# === MODEL DEFINITION ===

# === UNSHUFFLE MODULE ===
class Unshuffle(nn.Module):
    """
    Inverts the pixel shuffle using the inverse mapping stored as a buffer.
    """
    def __init__(self, mapping):
        super().__init__()
        inv = torch.argsort(torch.tensor(mapping, dtype=torch.long))
        self.register_buffer('inv_map', inv)

    def forward(self, x):
        # x: [B, C, H, W]
        B,C,H,W = x.shape
        flat = x.view(B, C, -1)
        unshuffled = flat[:, :, self.inv_map]
        return unshuffled.view(B, C, H, W)

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
        B, C, H, W = x.shape; n = H * W
        if self._p is None or self._p.numel() != n:
            self._build(n, x.device)
        return x.reshape(B, C, n)[:, :, self._p].reshape(B, C, H, W)

class _TokenGate(nn.Module):
    """SHA-256 hash verification gate — Section III-C."""
    def __init__(self, token_hash: str):
        super().__init__()
        self.token_hash = token_hash
    def forward(self, x, token: str = ''):
        try:
            computed = hashlib.sha256(bytes.fromhex(token)).hexdigest()
        except Exception:
            computed = ''
        if computed != self.token_hash:
            return torch.zeros_like(x)
        return x

class HybridCNN(nn.Module):
    """HybridCNN: N=4 decoys, L=6 SpatialPerm, M=2 token gates, κ=128-bit (Section IV-A)."""
    NUM_PERMS = 6   # L=6

    def __init__(self, fake_layers, token_hash: str = ''):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)

        # L=6 layer-wise spatial permutation modules (Section III-B)
        self.perms = nn.ModuleList([_SpatialPerm(1337 + l) for l in range(self.NUM_PERMS)])

        # M=2 token-gated access modules, κ=128-bit SHA-256 (Section III-C)
        self.gates = nn.ModuleList([_TokenGate(token_hash) for _ in range(2)])

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.AvgPool2d(2, 2)   # Section IV-A: global average pooling
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(fake_layers)
        ])
        self.conv3   = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.fc1     = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 10)

    def forward(self, x, path_key=0, token=''):
        x = self.unshuffle(x)
        x = self.gates[0](x, token)          # gate 1: entry (Section III-C)

        # conv1 → perm at 32×32 → pool → 16×16  (Section F: (32²)! factor)
        x = F.relu(self.bn1(self.conv1(x)))   # 32×32
        x = self.perms[0](x)
        x = self.pool(x)                       # → 16×16

        # conv2 → perm at 16×16 → pool → 8×8  (Section F: (16²)! factor)
        x = F.relu(self.bn2(self.conv2(x)))   # 16×16
        x = self.perms[1](x)
        x = self.pool(x)                       # → 8×8

        # decoy routing + perm at 8×8  (first of four (8²)! factors)
        x = self.fakes[path_key % len(self.fakes)](x)
        x = self.perms[2](x)

        x = self.gates[1](x, token)           # gate 2: cloud boundary (Section III-C)

        # conv3 → three additional perms at 8×8  (Section F: three additional (8²)! factors)
        x = F.relu(self.bn3(self.conv3(x)))   # 8×8
        x = self.perms[3](x)
        x = self.perms[4](x)
        x = self.perms[5](x)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# === COMPARE TO ===
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.AvgPool2d(2, 2)   # Section IV-A: global average pooling
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# === KEY MANAGEMENT ===
def save_key(key, path):
    with open(path, 'w') as f:
        json.dump({"key": key}, f)

def load_key(path):
    with open(path, 'r') as f:
        return json.load(f)["key"]

if not os.path.exists(config['key_file']):
    save_key(0, config['key_file'])

# === SPLIT RUN FUNCTIONS ===
def run_local(model, x, path_key=None):
    x = model.unshuffle(x)
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.pool(F.relu(model.bn2(model.conv2(out))))
    if path_key is not None:
        idx = path_key % len(model.fakes)
        out = model.fakes[idx](out)
    return out

def run_cloud(model, out):
    out = F.relu(model.bn3(model.conv3(out)))   # conv3 stays at 8×8; no pool (Section F)
    out = out.view(out.size(0), -1)
    out = model.dropout(F.relu(model.fc1(out)))
    return out

def predict(model, x, key=None):
    if key is None:
        key = torch.randint(0, len(model.fakes), (1,)).item()
    return model(x, path_key=key, token=token_hex)

# === TRAIN, EVALUATE, AND TEST IMAGE ===
def main():
    device = config['device']
    model = HybridCNN(config['parallel_fake_layers'], token_hash=token_hash).to(device)
    base = BaselineCNN().to(device)

    opt_h = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    opt_b = torch.optim.Adam(base.parameters(), lr=config['learning_rate'])
    scheduler_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=config['num_epochs'])
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=config['num_epochs'])
    crit = nn.CrossEntropyLoss()

    if os.path.exists(config['model_file']) and os.path.exists(config['base_model_file']):
        model.load_state_dict(torch.load(config['model_file'], map_location=device))
        base.load_state_dict(torch.load(config['base_model_file'], map_location=device))
        print(f"Loaded model from {config['model_file']}")
        print(f"Loaded baseline model from {config['base_model_file']}")
    else:
        epoch_losses = []
        for epoch in range(config['num_epochs']):
            model.train()
            base.train()
            for (x, y) in train_loader:
                x, y = x.to(device), y.to(device)
                # Shared-BN: run all N decoy paths per batch (Section IV-A)
                opt_h.zero_grad()
                loss_h = sum(
                    crit(predict(model, x, key=k), y)
                    for k in range(len(model.fakes))
                ) / len(model.fakes)
                loss_h.backward()
                opt_h.step()
            for (x, y) in loader_base:
                # BaselineCNN
                x, y = x.to(device), y.to(device)
                opt_b.zero_grad()
                out_b = base(x)
                loss_b = crit(out_b, y)
                loss_b.backward()
                opt_b.step()
            scheduler_h.step()
            scheduler_b.step()
            # Save losses for each epoch
            epoch_losses.append({
                "epoch": epoch + 1,
                "hybrid_loss": loss_h.item(),
                "baseline_loss": loss_b.item()
            })
            print(f"Epoch {epoch+1}/{config['num_epochs']} | Hybrid Loss: {loss_h.item():.4f} | Baseline Loss: {loss_b.item():.4f}")

        # Write losses to CSV after training
        with open("training_statistics.csv", "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "hybrid_loss", "baseline_loss"])
            writer.writeheader()
            writer.writerows(epoch_losses)
        # save models
        torch.save(model.state_dict(), config['model_file'])
        torch.save(base.state_dict(), config['base_model_file'])

    # Testing accuracy for our model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = predict(model, inputs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"HybridCNN Test Accuracy: {100 * correct/total:.2f}%")

    # Baseline accuracy
    base.eval()
    correct_b = 0
    with torch.no_grad():
        for x, y in loader_test_base:
            x, y = x.to(device), y.to(device)
            preds_b = base(x).argmax(1)
            correct_b += (preds_b==y).sum().item()
    print(f"BaselineCNN Test Accuracy: {100*correct_b/total:.2f}%")

    with open("test_accuracy.csv", "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["model", "accuracy"])
        writer.writeheader()
        writer.writerow({"model": "HybridCNN", "accuracy": f"{100 * correct/total:.2f}"})
        writer.writerow({"model": "BaselineCNN", "accuracy": f"{100 * correct_b/total:.2f}"})

    # Run on single image
    img = Image.open(config['test_image']).convert('RGB')
    x_hybrid = hybrid_transform(img).unsqueeze(0).to(device)
    x_base = baseline_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_hybrid = predict(model, x_hybrid)
        _, pred_hybrid = logits_hybrid.max(1)
        logits_base = base(x_base)
        _, pred_base = logits_base.max(1)
    print(f"HybridCNN prediction for {config['test_image']}: {class_names[pred_hybrid.item()]} ({pred_hybrid.item()})")
    print(f"BaselineCNN prediction for {config['test_image']}: {class_names[pred_base.item()]} ({pred_base.item()})")

if __name__ == '__main__':
    main()

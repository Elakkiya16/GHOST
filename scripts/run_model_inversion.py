"""
MIFace model-inversion attack demonstrated on MNIST (Figure 4, Section IV-C).

MNIST is used intentionally: its low-resolution, high-contrast structure makes
inversion success and failure visually unambiguous to the reader (see paper caption,
Fig. 4). SSIM reductions in Table III are from iDLG on CIFAR-10, not from this
script; this script provides the *visual* validation of structural obfuscation.
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from art.attacks.inference.model_inversion import MIFace
from art.estimators.classification import PyTorchClassifier
torch.manual_seed(42)

# === CONFIGURATION ===
config = {
    "batch_size": 128,
    "shuffle_map_file": "shuffle_map_simple.json",
    "key_file": "path_key_simple.json",
    "num_epochs": 20,          # MNIST converges faster than CIFAR; 20 epochs sufficient
    "learning_rate": 1e-3,
    "parallel_fake_layers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_file": "hybrid_cnn_simple.pth",
    "base_model_file": "baseline_cnn_simple.pth",
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

# Prepare or load shuffle map (28×28 = 784 pixels for MNIST)
height, width = 28, 28
if not os.path.exists(config['shuffle_map_file']):
    mapping = generate_shuffle_map(height, width)
    save_map(mapping, config['shuffle_map_file'])
else:
    mapping = load_map(config['shuffle_map_file'])
    if len(mapping) != height * width:
        mapping = generate_shuffle_map(height, width)
        save_map(mapping, config['shuffle_map_file'])

# === TRANSFORMS (MNIST, 28×28 grayscale) ===
hybrid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: shuffle_image(x, mapping)),
    transforms.Normalize((0.5,), (0.5,))
])
baseline_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === DATALOADERS ===
train_loader = DataLoader(
    datasets.MNIST('./data', True,  transform=hybrid_transform,   download=True),
    batch_size=config['batch_size'], shuffle=True,  num_workers=config['num_workers'])
test_loader = DataLoader(
    datasets.MNIST('./data', False, transform=hybrid_transform,   download=False),
    batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
loader_base = DataLoader(
    datasets.MNIST('./data', True,  transform=baseline_transform, download=False),
    batch_size=config['batch_size'], shuffle=True,  num_workers=config['num_workers'])
loader_test_base = DataLoader(
    datasets.MNIST('./data', False, transform=baseline_transform, download=False),
    batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
class_names = datasets.MNIST('./data', False).classes

# === UNSHUFFLE MODULE ===
class Unshuffle(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        inv = torch.argsort(torch.tensor(mapping, dtype=torch.long))
        self.register_buffer('inv_map', inv)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.view(B, C, -1)
        return flat[:, :, self.inv_map].view(B, C, H, W)

# === MODEL DEFINITIONS (MNIST-scale; single conv as in original paper demo) ===
class HybridCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        self.conv = nn.Conv2d(1, 16, 3, 1)
        self.fc = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = self.unshuffle(x)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, 1)
        self.fc = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === MIFace ATTACK ===
def model_inversion_attack_miface(target_model, mtype):
    """Run MIFace inversion on MNIST (Figure 4). Generates one reconstructed
    image per class; saves grid to miface_{mtype}.png."""
    target_model.eval()
    classifier = PyTorchClassifier(
        model=target_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )
    attack = MIFace(classifier=classifier, max_iter=100000, window_length=400)
    print(f"Running MIFace inversion on {mtype} model (MNIST)...")
    x_inverted = attack.infer(x=None, y=np.eye(10, dtype=np.float32))

    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        img = np.clip(x_inverted[i][0], 0, 1)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Class {i}", fontsize=7)
    plt.suptitle(f"MIFace Inversion — {mtype} (MNIST)", fontsize=10)
    plt.tight_layout()
    out = f"miface_{mtype}.png"
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    return x_inverted

# === TRAIN AND EVALUATE ===
def main():
    device = torch.device(config['device'])
    model = HybridCNN().to(device)
    base  = BaselineCNN().to(device)

    opt_h  = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    opt_b  = torch.optim.Adam(base.parameters(),  lr=config['learning_rate'])
    crit   = nn.CrossEntropyLoss()

    if os.path.exists(config['model_file']) and os.path.exists(config['base_model_file']):
        model.load_state_dict(torch.load(config['model_file'],      map_location=device))
        base.load_state_dict(torch.load(config['base_model_file'],  map_location=device))
        print(f"Loaded models from disk.")
    else:
        epoch_losses = []
        for epoch in range(config['num_epochs']):
            model.train(); base.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt_h.zero_grad()
                crit(model(x), y).backward()
                opt_h.step()
            for x, y in loader_base:
                x, y = x.to(device), y.to(device)
                opt_b.zero_grad()
                crit(base(x), y).backward()
                opt_b.step()
            epoch_losses.append({"epoch": epoch + 1})
            print(f"Epoch {epoch+1}/{config['num_epochs']}")

        torch.save(model.state_dict(), config['model_file'])
        torch.save(base.state_dict(),  config['base_model_file'])

    # Test accuracy
    model.eval(); base.eval()
    for m, name, ldr in [(model, 'HybridCNN', test_loader), (base, 'Baseline', loader_test_base)]:
        correct = total = 0
        with torch.no_grad():
            for x, y in ldr:
                x, y = x.to(device), y.to(device)
                correct += (m(x).argmax(1) == y).sum().item()
                total   += y.size(0)
        print(f"{name} Test Accuracy: {100*correct/total:.2f}%")

    print("Running MIFace inversion attacks (may take a while)...")
    model_inversion_attack_miface(base,  'Baseline')
    model_inversion_attack_miface(model, 'HybridCNN')

if __name__ == '__main__':
    main()

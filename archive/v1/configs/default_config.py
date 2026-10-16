import torch
import os

# Hardware used for all paper results: NVIDIA Tesla V100 (32 GB VRAM, CUDA 12.2).
# The codebase also runs on Apple MPS and CPU for local development.

CONFIG = {
    # HybridCNN training epochs (Section IV-A)
    "num_epochs": 50,
    "hybrid_num_epochs": 50,
    # ResNet-18 / ResNet-50 / MobileNetV3 training epochs (Section IV-E).
    # Used by scripts/train_backbones.py.
    "backbone_num_epochs": 100,
    "batch_size": 128,
    "lr": 1e-3,
    "optimizer": "Adam",
    "scheduler": "CosineAnnealingLR",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "map_file": "ghost_map.json",
    "model_path": "ghost_resnet18.pth",
    "base_path": "base_resnet18.pth",
    "query_budget": 10000,
    "img_size": 32,
    # N=4 decoys for HybridCNN; backbone models use N=8 (set in train_backbones.py)
    "num_decoys": 4,
    "token_file": "ghost_token.json",
    "split_execution": True,
    "train_dataset": "CIFAR-10",
    "aux_dataset": "CIFAR-100",
    "supported_backbones": ["HybridCNN", "ResNet-18", "ResNet-50", "MobileNetV3"],
    "evaluation_datasets": ["CIFAR-10", "CIFAR-100", "SVHN", "TinyImageNet", "ImageNet-50k"],
    "use_full_test_set": True,
}

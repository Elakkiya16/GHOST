import torch
import os

CONFIG = {
    "num_epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "map_file": "ghost_map.json",
    "model_path": "ghost_resnet18.pth",
    "base_path": "base_resnet18.pth",
    "query_budget": 10000,
    "img_size": 32,
    "num_decoys": 4,
    "token_file": "ghost_token.json",
    "split_execution": True
}

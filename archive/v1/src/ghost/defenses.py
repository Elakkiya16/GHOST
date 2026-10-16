import torch
import torch.nn as nn
import torch.nn.functional as F

# conceptual implementation of defenses
class MemGuard(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        logits = self.base_model(x)
        probs = F.softmax(logits, dim=1)
        noise = torch.randn_like(probs) * 0.01
        return torch.clamp(probs + noise, 1e-7, 1.0)

class Purifier(nn.Module):
    def __init__(self, base_model, temperature=2.5):
        super().__init__()
        self.base_model = base_model
        self.T = temperature
    def forward(self, x):
        return F.softmax(self.base_model(x) / self.T, dim=1)

class ModelGuard(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    def forward(self, x):
        probs = F.softmax(self.base_model(x), dim=1)
        max_p, _ = torch.max(probs, dim=1, keepdim=True)
        mask = (max_p < 0.7).float()
        noise = torch.randn_like(probs) * 0.1
        protected_probs = probs + (mask * noise)
        return protected_probs / protected_probs.sum(dim=1, keepdim=True)

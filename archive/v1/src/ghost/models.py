import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.ghost.utils import Unshuffle, SpatialPerm, TokenGate

_PERM_SEED_BASE = 1337   # master seed; layer l uses seed base + l

# ResNet-18: L=12 permutation points.
# Target = conv2 (output of residual function F, immediately preceding the identity
# add) for each BasicBlock, plus the downsample.0 projection convs on stages 2-4
# (where the shortcut itself is a conv), plus the stem conv1.
# This matches Section III-B: "integrated at the output of each residual stage
# immediately preceding the identity mapping."
_R18_PERM_LAYER_ATTRS = [
    'conv1',                     # stem (no identity branch)
    'layer1.0.conv2',            # stage-1 block-0 residual output
    'layer1.1.conv2',            # stage-1 block-1 residual output
    'layer2.0.conv2',            # stage-2 block-0 residual output
    'layer2.0.downsample.0',     # stage-2 projection shortcut
    'layer2.1.conv2',            # stage-2 block-1 residual output
    'layer3.0.conv2',            # stage-3 block-0 residual output
    'layer3.0.downsample.0',     # stage-3 projection shortcut
    'layer3.1.conv2',            # stage-3 block-1 residual output
    'layer4.0.conv2',            # stage-4 block-0 residual output
    'layer4.0.downsample.0',     # stage-4 projection shortcut
    'layer4.1.conv2',            # stage-4 block-1 residual output
]   # L = 12

# ResNet-50: L=12 permutation points targeting conv3 (bottleneck residual output)
# immediately before the identity add, plus projection shortcuts for each stage.
_R50_PERM_LAYER_ATTRS = [
    'conv1',                     # stem
    'layer1.0.conv3',            # stage-1 bottleneck-0 residual output
    'layer1.0.downsample.0',     # stage-1 projection shortcut
    'layer1.1.conv3',            # stage-1 bottleneck-1 residual output
    'layer1.2.conv3',            # stage-1 bottleneck-2 residual output
    'layer2.0.conv3',            # stage-2 bottleneck-0 residual output
    'layer2.0.downsample.0',     # stage-2 projection shortcut
    'layer2.1.conv3',            # stage-2 bottleneck-1 residual output
    'layer3.0.conv3',            # stage-3 bottleneck-0 residual output
    'layer3.0.downsample.0',     # stage-3 projection shortcut
    'layer4.0.conv3',            # stage-4 bottleneck-0 residual output
    'layer4.0.downsample.0',     # stage-4 projection shortcut
]   # L = 12


def _get_submodule(root, attr_path: str) -> nn.Module:
    m = root
    for part in attr_path.split('.'):
        m = getattr(m, part)
    return m


def _attach_spatial_perms(model, layer_attrs, seed_base=_PERM_SEED_BASE):
    """Register forward hooks that apply a unique SpatialPerm after each target layer."""
    model.spatial_perms = nn.ModuleList([
        SpatialPerm(seed_base + l) for l in range(len(layer_attrs))
    ])
    model._perm_hooks = []
    for i, attr in enumerate(layer_attrs):
        perm = model.spatial_perms[i]
        layer = _get_submodule(model, attr)
        handle = layer.register_forward_hook(lambda m, inp, out, p=perm: p(out))
        model._perm_hooks.append(handle)


class GHOST_ResNet18(nn.Module):
    """ResNet-18 with N=8 decoys, L=12 spatial permutations, M=4 token gates (Section IV-E)."""

    def __init__(self, mapping, num_classes=10, num_decoys=8, token_hash: str = ''):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        resnet = models.resnet18(weights=None)

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(512, num_classes)

        # N=8 encrypted decoy paths (Section IV-E)
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(num_decoys)
        ])

        # M=4 token-gated access modules, κ=256-bit SHA-256 verification (Section III-C)
        self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])

        # L=12 layer-wise spatial permutations via forward hooks (Section III-B)
        _attach_spatial_perms(self, _R18_PERM_LAYER_ATTRS)

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.gates[0](x, token)   # gate 1
        x = self.layer1(x)

        x = self.gates[1](x, token)   # gate 2
        x = self.fakes[key % len(self.fakes)](x)

        x = self.gates[2](x, token)   # gate 3
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gates[3](x, token)   # gate 4
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class GHOST_ResNet50(nn.Module):
    """ResNet-50 with N=8 decoys, L=12 spatial permutations, M=4 token gates (Section IV-E)."""

    def __init__(self, mapping, num_classes=10, num_decoys=8, token_hash: str = ''):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        resnet = models.resnet50(weights=None)

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc      = nn.Linear(2048, num_classes)

        # N=8 encrypted decoy paths (Section IV-E)
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
            for _ in range(num_decoys)
        ])

        # M=4 token-gated access modules (Section III-C)
        self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])

        # L=12 layer-wise spatial permutations via forward hooks (Section III-B)
        _attach_spatial_perms(self, _R50_PERM_LAYER_ATTRS)

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.gates[0](x, token)
        x = self.layer1(x)

        x = self.gates[1](x, token)
        x = self.fakes[key % len(self.fakes)](x)

        x = self.gates[2](x, token)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.gates[3](x, token)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


class GHOST_MobileNetV3(nn.Module):
    """MobileNetV3-Small with N=8 decoys, L=12 spatial permutations, M=4 token gates
    (Section IV-E). MobileNetV3-Small features[0..12]: 4 edge blocks + 9 cloud blocks.
    Permutations applied at output of all 4 edge blocks + first 8 cloud blocks = L=12."""

    def __init__(self, mapping, num_classes=10, num_decoys=8, token_hash: str = ''):
        super().__init__()
        self.unshuffle = Unshuffle(mapping)
        base = models.mobilenet_v3_small(weights=None)

        self.edge           = base.features[:4]    # 4 blocks
        self.cloud_features = base.features[4:]    # 9 blocks (features[4..12])
        self.avgpool        = base.avgpool
        self.classifier     = base.classifier
        self.classifier[3]  = nn.Linear(1024, num_classes)

        # N=8 encrypted decoy paths (Section IV-E)
        self.fakes = nn.ModuleList([
            nn.Sequential(nn.Conv2d(24, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU())
            for _ in range(num_decoys)
        ])

        # M=4 token-gated access modules (Section III-C)
        self.gates = nn.ModuleList([TokenGate(token_hash) for _ in range(4)])

        # L=12: 4 edge perms + 8 cloud perms (Section III-B, Section IV-E)
        self.spatial_perms = nn.ModuleList([
            SpatialPerm(_PERM_SEED_BASE + l) for l in range(12)
        ])

    def forward(self, x, key=0, token=''):
        x = self.unshuffle(x)

        x = self.gates[0](x, token)
        # Perms 0-3: at output of each of the 4 edge blocks
        for i, block in enumerate(self.edge):
            x = block(x)
            x = self.spatial_perms[i](x)

        x = self.gates[1](x, token)
        x = self.fakes[key % len(self.fakes)](x)

        x = self.gates[2](x, token)
        # Perms 4-11: at output of first 8 of the 9 cloud blocks
        for i, block in enumerate(self.cloud_features):
            x = block(x)
            if i < 8:   # L=12: 4 edge + 8 cloud = 12 total
                x = self.spatial_perms[4 + i](x)

        x = self.gates[3](x, token)
        x = self.avgpool(x)
        return self.classifier(torch.flatten(x, 1))


class BaselineResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

class BaselineResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.model(x)

class BaselineMobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(1024, num_classes)
    def forward(self, x):
        return self.model(x)

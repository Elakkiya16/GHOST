import torch
from common import *
from ghost import *
torch.set_num_threads(1)
seed_all(1)
for arch, split in [('resnet20', 3), ('vgg11h', 4)]:
    m = ARCHS[arch]().eval()
    # random BN stats so folding is non-trivial
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-.5,.5); mod.running_var.uniform_(.5,2); mod.weight.data.uniform_(.5,1.5); mod.bias.data.uniform_(-.2,.2)
    X = torch.randn(64,3,32,32)
    ref = m(X)
    for level in ['ps','left','full']:
      for rho in [0, 10, 1000]:
        dep = GhostDeployment(m, arch, split, rho=rho, seed=0, Xcal=X[:32], level=level)
        out = dep.query(X)
        print(arch, level, rho, len(dep.names), 'relerr', ((out-ref).norm()/ref.norm()).item(), 'agree', (out.argmax(1)==ref.argmax(1)).float().mean().item())
    # token refusal
    try:
        dep.query(X, token=b'0'*32); print('NOT REFUSED')
    except PermissionError as e: print('refused ok')

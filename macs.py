"""Platform-independent cost of GHOST per request: multiply-accumulate counts of the TEE-executed
first layer, the TEE online work (input mixing, output un-mixing), the TEE offline mask correction,
and the REE convolutions, versus the original convolutions they replace."""
import sys, json
from common import *
from ghost import GhostDeployment, fold_bn, ObfConv, _get
torch.set_num_threads(1)
ARCH = sys.argv[1]
SPLITS = {'resnet20': [1, 3, 6], 'vgg11h': [2, 4, 6]}[ARCH]
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{ARCH}_victim_s0.pt')); victim.eval()
folded = fold_bn(victim)
# output spatial sizes of every conv in the folded model
sizes = {}
hooks = [m.register_forward_hook(lambda mod, i, o, n=n: sizes.__setitem__(n, (i[0].shape, o.shape)))
         for n, m in folded.named_modules() if isinstance(m, nn.Conv2d)]
with torch.no_grad(): folded(Xt[:1])
for h in hooks: h.remove()
total_orig = sum(m.weight.numel() * sizes[n][1][2] * sizes[n][1][3] for n, m in folded.named_modules() if isinstance(m, nn.Conv2d))
rows = []
for split in SPLITS:
    dep = GhostDeployment(victim, ARCH, split, rho=10, seed=7, Xcal=Xp[:64], encrypt=False)
    tee_first = sum(_get(folded, n).weight.numel() * sizes[n][1][2] * sizes[n][1][3] for n in dep.tee_names)
    online = offline = ree = orig = 0
    for n in dep.names:
        oc = _get(dep.model, n); (B, ci, H, W), (_, co, Ho, Wo) = sizes[n]
        orig += _get(folded, n).weight.numel() * Ho * Wo
        ree += oc.Wt.numel() * Ho * Wo                      # obfuscated conv (with decoy filters)
        offline += oc.Wt.numel() * Ho * Wo                  # mask correction W~ * R, same size
        online += ci * ci * H * W + (co + oc.nd) * co * Ho * Wo   # input mixing B a, output un-mixing A^{-1}
    rows.append(dict(arch=ARCH, split=split, M_orig_total=total_orig / 1e6, M_orig_edge=orig / 1e6, M_tee_first=tee_first / 1e6,
                     M_tee_online=online / 1e6, M_tee_offline=offline / 1e6, M_ree=ree / 1e6))
    print(rows[-1])
save_json(rows, f'macs_{ARCH}.json')

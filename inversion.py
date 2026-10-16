"""E4: input reconstruction from REE-visible activations.
The adversary submits its own images through authorised queries, records the masked,
permuted operands x~ that the TEE hands to the REE, trains a decoder x~ -> image on
these pairs, and applies it to other users' queries.  Usage: python inversion.py ARCH"""
import sys
import numpy as np
from skimage.metrics import structural_similarity
from common import *
from ghost import GhostDeployment

torch.set_num_threads(int(os.environ.get('NT', '2')))
ARCH = sys.argv[1]
MAIN = {'resnet20': 3, 'vgg11h': 4}[ARCH]
STEPS = int(os.environ.get('STEPS', '1500'))
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{ARCH}_victim_s0.pt')); victim.eval()


class Decoder(nn.Module):
    def __init__(self, cin, up):
        super().__init__()
        L = [nn.Conv2d(cin, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()]
        c = 128
        for _ in range(up):
            L += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1), nn.BatchNorm2d(c // 2), nn.ReLU()]; c //= 2
        L += [nn.Conv2d(c, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 3, 3, 1, 1)]
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


def collect(dep, X, name, bs=250):
    dep.view.enabled = True; dep.view.only = name; dep.view.inputs = {}; dep.view.outputs = {}
    for k in range(0, len(X), bs): dep.query(X[k:k + bs])
    dep.view.enabled = False
    return torch.cat(dep.view.inputs[name])


def to01(x): return (x * STD + MEAN).clamp(0, 1)


def metrics(rec, X):
    a = to01(rec).numpy(); b = to01(X).numpy()
    mse = ((a - b) ** 2).mean(axis=(1, 2, 3))
    psnr = float(np.mean(10 * np.log10(1.0 / np.maximum(mse, 1e-10))))
    ssim = float(np.mean([structural_similarity(a[i].transpose(1, 2, 0), b[i].transpose(1, 2, 0),
                                                channel_axis=2, data_range=1.0) for i in range(len(a))]))
    return psnr, ssim


rows = []
NTR = int(os.environ.get('NTR', '8000')); Xtr, Xte = Xp[:NTR], Xt[:1000]
# uninformed baseline: predict the mean image of the adversary's data for every input
mean_img = Xtr.mean(0, keepdim=True).expand_as(Xte)
p0, s0 = metrics(mean_img, Xte)
save_json({'psnr_mean_image': p0, 'ssim_mean_image': s0}, 'inversion_baseline.json')
print('mean-image baseline', p0, s0, flush=True)
for rho in [0, 1, 10, 100]:
    dep = GhostDeployment(victim, ARCH, MAIN, rho=rho, seed=11, Xcal=Xp[:256], encrypt=False)
    for li, name in [(0, dep.names[0]), (len(dep.names) - 1, dep.names[-1])]:
        Otr = collect(dep, Xtr, name); Ote = collect(dep, Xte, name)
        up = int(round(np.log2(32 / Otr.shape[-1])))
        dec = Decoder(Otr.shape[1], up)
        seed_all(0)
        opt = torch.optim.Adam(dec.parameters(), 1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        dec.train()
        for it in range(STEPS):
            b = torch.randint(0, len(Otr), (64,))
            loss = F.mse_loss(dec(Otr[b]), Xtr[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        dec.eval()
        with torch.no_grad(): rec = torch.cat([dec(Ote[k:k + 250]) for k in range(0, len(Ote), 250)])
        psnr, ssim = metrics(rec, Xte)
        r = dict(arch=ARCH, rho=rho, layer=name, depth=li + 1, shape=list(Otr.shape[1:]), psnr=psnr, ssim=ssim)
        rows.append(r); print(r, flush=True)
        if rho in (0, 10):
            torch.save({'rec': to01(rec[:8]), 'orig': to01(Xte[:8])}, f'{OUT}/inv_examples_{ARCH}_{name}_rho{rho}.pt')
save_json(rows, f'inversion_{ARCH}.json')

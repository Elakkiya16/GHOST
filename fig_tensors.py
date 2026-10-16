"""Extract the real intermediate tensors of one GHOST request (ResNet-20, split 3, rho=10)
for the system figure. Writes PNG tiles and a TeX macro file to paper/figures/arch_assets (override with OUTD)."""
import os, hashlib, struct, json
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from common import *
from ghost import GhostDeployment, fold_bn, _get
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

OUTD = os.environ.get('OUTD', 'paper/figures/arch_assets'); os.makedirs(OUTD, exist_ok=True)
import matplotlib.cm as mcm
CM = matplotlib.colormaps['viridis']
DV = matplotlib.colormaps['RdBu_r']
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
torch.manual_seed(0)

(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
victim = ARCHS['resnet20'](); victim.load_state_dict(torch.load(f'{CKPT}/resnet20_victim_s0.pt')); victim.eval()
with torch.no_grad(): P = torch.softmax(victim(Xt[:500]), 1)
cand = [i for i in range(500) if yt[i] == 7 and P[i].argmax() == 7 and P[i, 7] > 0.9]
idx = int(os.environ.get('IDX', cand[0]))
x = Xt[idx:idx + 1]

dep = GhostDeployment(victim, 'resnet20', 3, rho=10, seed=7, Xcal=Xp[:256], encrypt=False)
L = 'blocks.0.conv1'
dep.view.enabled = True; dep.view.only = L
cap = {}
def hk(name):
    def h(m, i, o): cap[name] = (i[0].detach().clone(), o.detach().clone())
    return h
_get(dep.model, L).register_forward_hook(hk('layer'))
_get(dep.model, 'conv1').register_forward_hook(hk('stem'))
_get(dep.model, 'blocks.2').register_forward_hook(hk('feat'))
out = dep.query(x)
prob = torch.softmax(out, 1)[0]
plain = fold_bn(victim)
with torch.no_grad(): pref = torch.softmax(plain(x), 1)[0]
print('idx', idx, 'pred', CLASSES[prob.argmax()], float(prob.max()), 'maxdiff', float((prob - pref).abs().max()))

a = torch.relu(cap['stem'][1])[0]          # first-conv activation (TEE)
xt = dep.view.inputs[L][0][0]               # masked mixed activation released to REE
yt = dep.view.outputs[L][0][0]              # REE convolution output
y = cap['layer'][1][0]                      # recovered pre-activation
feat = cap['feat'][1][0]                    # split feature sent to cloud
oc = _get(dep.model, L)
W = _get(dep.folded, L).weight.detach(); Wt = oc.Wt

def save_map(t, name, size=96):
    t = t.numpy(); lo, hi = np.percentile(t, [1, 99]); t = np.clip((t - lo) / (hi - lo + 1e-9), 0, 1)
    im = (CM(t)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(im).resize((size, size), Image.NEAREST).save(f'{OUTD}/{name}.png')

def save_mat(m, name, scale=6):
    m = m.numpy(); v = np.percentile(np.abs(m), 90, axis=1, keepdims=True); t = np.clip((m / v + 1) / 2, 0, 1)
    im = (DV(t)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(im).resize((m.shape[1] * scale, m.shape[0] * scale), Image.NEAREST).save(f'{OUTD}/{name}.png')

# input image (de-normalised)
img = (x * STD + MEAN)[0].clamp(0, 1).permute(1, 2, 0).numpy()
Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128), Image.LANCZOS).save(f'{OUTD}/input.png')
ch = [0, 3, 7, 11]
for k, c in enumerate(ch):
    save_map(a[c], f'a{k}'); save_map(xt[c], f'xt{k}'); save_map(yt[c], f'yt{k}'); save_map(y[c], f'y{k}'); save_map(feat[c], f'f{k}')
def save_kernels(w, name, n=8, s=6):
    w = w[:n, :n].numpy(); vr = np.percentile(np.abs(w.reshape(n, -1)), 90, axis=1)
    g = np.ones((n * 4 - 1, n * 4 - 1, 3)) * np.array([1, 1, 1])
    for i in range(n):
        for j in range(n):
            t = np.clip((w[i, j] / vr[i] + 1) / 2, 0, 1)
            g[i * 4:i * 4 + 3, j * 4:j * 4 + 3] = DV(t)[..., :3]
    Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * s, g.shape[0] * s), Image.NEAREST).save(f'{OUTD}/{name}.png')
save_kernels(W, 'W'); save_kernels(Wt, 'Wt')
R = xt - torch.einsum('ij,jhw->ihw', oc.B, cap['layer'][0][0])
[save_map(R[c], f'R{k}') for k, c in enumerate(ch)]
save_mat(oc.B, 'B', 8); save_mat(oc.A, 'A', 8)
# ciphertext of the split feature
key = dep.tee.k_t; ct = AESGCM(key).encrypt(os.urandom(12), feat.numpy().tobytes(), None)
cb = np.frombuffer(ct[:24 * 24], dtype=np.uint8).reshape(24, 24) / 255.
Image.fromarray((matplotlib.colormaps['gray'](cb)[..., :3] * 255).astype(np.uint8)).resize((96, 96), Image.NEAREST).save(f'{OUTD}/ct.png')
digest = hashlib.sha256(x.numpy().tobytes()).digest()
tok = dep.tee.issue_token(1, digest).hex()
with open(f'{OUTD}/values.tex', 'w') as f:
    f.write('\\def\\TokHex{%s}\n' % tok[:8])
    f.write('\\def\\CtHex{%s}\n' % ct[:4].hex())
    f.write('\\def\\PredClass{%s}\n' % CLASSES[prob.argmax()].capitalize())
    f.write('\\def\\ProbList{%s}\n' % ','.join('%.3f' % p for p in prob.tolist()))
    f.write('\\def\\WRows{%d}\\def\\WtRows{%d}\n' % (W.shape[0], Wt.shape[0]))
print(tok[:8], ct[:4].hex(), [round(p, 3) for p in prob.tolist()])

# ---- adversary side information (threat model): auxiliary data and own reference model
own = ARCHS['resnet20'](); own.load_state_dict(torch.load(f'{CKPT}/steal_resnet20_n10000_scratch.pt')); own.eval()
save_kernels(_get(fold_bn(own), L).weight.detach(), 'Wref')
aux = (Xp[[3, 11, 25, 40]] * STD + MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
g = np.ones((66, 66, 3))
for k in range(4):
    i, j = divmod(k, 2); g[i * 34:i * 34 + 32, j * 34:j * 34 + 32] = aux[k]
Image.fromarray((g * 255).astype(np.uint8)).resize((132, 132), Image.NEAREST).save(f'{OUTD}/aux.png')

# ---- every intermediate of the outsourced layer, and the setup pipeline
a_in = cap['layer'][0][0]                                   # layer input a (TEE)
Ba = torch.einsum('ij,jhw->ihw', oc.B, a_in)                # channel mixing
M = oc.ree_conv((xt - Ba).unsqueeze(0))[0]                  # precomputed correction W~*R
z = (yt - M)[oc.Pinv]                                       # unmasked, still mixed
u = cap['layer'][1][0]                                      # unmixed real channels + bias
act = torch.relu(u)
for k, c in enumerate(ch):
    save_map(Ba[c], f'Ba{k}'); save_map(M[c], f'M{k}'); save_map(z[c], f'z{k}'); save_map(u[c], f'u{k}'); save_map(act[c], f'act{k}')
Wraw = _get(victim, L).weight.detach()
save_kernels(Wraw, 'Wraw'); save_kernels(W, 'Wfold')
Wx = torch.einsum('oc,cikl->oikl', torch.linalg.inv(oc.A.double()).float(), Wt[oc.Pinv])   # [W; F] in input-mixed coords
Wx = torch.einsum('oikl,ij->ojkl', Wx, oc.B)                                                # undo input mixing
print('stack recon err', float((Wx[:oc.cout] - W).abs().max()))
WF = torch.cat([W[:6], Wx[oc.cout:oc.cout + 2]])
save_kernels(WF, 'WF')
PAWF = torch.einsum('oc,cikl->oikl', oc.A, Wx)[oc.P]
save_kernels(PAWF, 'PAWF')

# ---- setup-row tiles on ONE shared colour scale, 6 filters x 6 input channels, larger kernels
Wr_ = _get(victim, L).weight.detach()
fs = (W.reshape(W.shape[0], -1).norm(dim=1) / Wr_.reshape(W.shape[0], -1).norm(dim=1))
o = torch.argsort(fs); rows = sorted([int(o[0]), int(o[1]), int(o[-1]), int(o[-2])])   # filters most shrunk and most enlarged by folding
VMAX = float(np.percentile(np.abs(W.numpy()), 75))
NMAX = 2.75     # common filter-norm scale for the bar column (largest mixed filter norm is 2.69)
def save_grid(w, name, s=7, decoy_from=None):
    n_r, n_c = w.shape[0], 6
    nrm = w.reshape(w.shape[0], -1).norm(dim=1).numpy()        # norm of the full filter (all input channels)
    g = np.ones((n_r * 4 - 1, n_c * 4 - 1 + 10, 3))
    for i in range(n_r):
        for j in range(n_c):
            t = np.clip((w[i, j].numpy() / VMAX + 1) / 2, 0, 1)
            g[i * 4:i * 4 + 3, j * 4:j * 4 + 3] = DV(t)[..., :3]
        L_ = max(1, int(round(nrm[i] / NMAX * 9)))
        g[i * 4:i * 4 + 3, n_c * 4 + 1:n_c * 4 + 1 + L_] = np.array([0x26, 0x32, 0x38]) / 255.
    Image.fromarray((g * 255).astype(np.uint8)).resize((g.shape[1] * s, g.shape[0] * s), Image.NEAREST).save(f'{OUTD}/{name}.png')
C6 = list(range(6))
save_grid(Wr_[rows], 'S_raw')
save_grid(W[rows], 'S_fold')
save_grid(torch.cat([W[rows], Wx[oc.cout:oc.cout + 2]]), 'S_wf')
save_grid(PAWF[:6], 'S_mix')
save_grid(Wt[:6], 'S_wt')
print('setup rows', rows, 'fold scales', fs[rows].numpy().round(2), 'mixed norms', PAWF[:6].reshape(6, -1).norm(dim=1).numpy().round(2))

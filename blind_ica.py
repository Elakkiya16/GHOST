"""Blind estimation of the input-side mixing B from released operands x~ = B a + R.
The adversary submits its own images, records the operand of one edge convolution,
averages `repeats` observations of the same input (allowed by the query budget), runs FastICA
to estimate B^{-1} up to row order, sign and scale, and then uses the estimate in the
row-space projection attack against the stored tensor of that layer.  Reported: the relative
error of the recovered filters against the victim's folded filters, compared with the error of
the reference model alone (what the adversary has without any attack) and with the projection
attack given the true B (upper bound).   Usage: python blind_ica.py ARCH [layer index]"""
import sys, numpy as np, torch
from sklearn.decomposition import FastICA
from common import *
from ghost import GhostDeployment, _get, fold_bn
from matching import project
torch.set_num_threads(int(os.environ.get('NT', '2')))
ARCH = sys.argv[1]; LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SPLIT = {'resnet20': 3, 'vgg11h': 4}[ARCH]
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
VICTIM = os.environ.get('VICTIM', f'{ARCH}_victim_s0'); ORACLE = os.environ.get('ORACLE', f'{ARCH}_victim_s1'); TAG = os.environ.get('TAG', '')
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{VICTIM}.pt')); victim.eval()
ref = ARCHS[ARCH](); ref.load_state_dict(torch.load(f'{CKPT}/{ORACLE}.pt')); ref.eval()
VF, RF = fold_bn(victim), fold_bn(ref)
REF_ACT, REF_STD = {}, {}


def _ref_inputs(name, X, idx_sel=None):
    acts = []
    def h(mod, inp, out): acts.append(inp[0].detach())
    hd = _get(RF, name).register_forward_hook(h)
    with torch.no_grad():
        for k in range(0, len(X), 250): RF(X[k:k + 250])
    hd.remove()
    A = torch.cat(acts)
    return A


rows = []
for rho, repeats in [(1, 1), (10, 1), (10, 10), (10, 100), (100, 10), (100, 100)]:
    dep = GhostDeployment(victim, ARCH, SPLIT, rho=rho, seed=21, Xcal=Xp[:256], encrypt=False)
    name = dep.names[LAYER]; oc = _get(dep.model, name)
    Wv = _get(VF, name).weight.detach(); Wr = _get(RF, name).weight.detach(); Wo = dep.obfuscated_weights()[name]
    ref_err = ((Wr - Wv).norm() / Wv.norm()).item()
    n = 2000 // repeats; bs = min(200, n)
    dep.view.enabled = True; dep.view.only = name; dep.view.inputs = {}; dep.view.outputs = {}
    for k in range(0, n, bs):
        for _ in range(repeats): dep.query(Xp[k:k + bs])
    dep.view.enabled = False
    if name not in REF_ACT:
        A = _ref_inputs(name, Xp[:n])
        REF_STD[name] = A.permute(0, 2, 3, 1).reshape(-1, A.shape[1]).std(0)
    O = torch.stack(dep.view.inputs[name]); nb = n // bs
    O = O.view(nb, repeats, *O.shape[1:]).mean(1).reshape(-1, *O.shape[2:])
    C = O.shape[1]
    S = O.permute(0, 2, 3, 1).reshape(-1, C).numpy()
    idx = np.random.default_rng(0).choice(len(S), min(len(S), 60000), replace=False)
    S_full = S[idx]; S = S_full
    A = _ref_inputs(name, Xp[:n])
    REF_ACT[name] = A.permute(0, 2, 3, 1).reshape(-1, C).numpy()[idx]
    ica = FastICA(n_components=C, random_state=0, max_iter=400, tol=1e-3, whiten='unit-variance').fit(S)
    U = ica.components_                                   # estimate of B^{-1} up to row order, sign, scale
    Bhat = np.linalg.pinv(U)                              # estimate of B up to column order, sign, scale
    Wk = torch.einsum('oikl,ij->ojkl', Wo, torch.from_numpy(Bhat).float())
    est = project(Wk, Wr)                                 # searches the column order itself
    err_ica = ((est - Wv).norm() / Wv.norm()).item()
    # refinement: the adversary fixes the unknown source scales with the per-channel activation
    # statistics of its reference model (sources have unit variance after ICA), and the signs and
    # column order with the projection search
    ref_std = REF_STD[name].numpy()
    Bhat2 = np.linalg.pinv(U) / np.maximum(ref_std.mean(), 1e-8)   # provisional: uniform scale
    # match ICA sources to reference channels by |correlation| of source signals with reference activations
    Sref = REF_ACT[name]                                  # samples x C (reference activations, same inputs)
    src = (S_full @ U.T)                                  # ICA sources for the same samples
    Cm = np.abs(np.corrcoef(src.T, Sref.T)[:C, C:])       # C x C
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-Cm)                   # source ri[k] <-> reference channel ci[k]
    scale = np.zeros(C); order = np.zeros(C, dtype=int)
    for k in range(C):
        sgn = np.sign(np.corrcoef(src[:, ri[k]], Sref[:, ci[k]])[0, 1])
        scale[ri[k]] = sgn * ref_std[ci[k]] / (src[:, ri[k]].std() + 1e-8); order[ri[k]] = ci[k]
    Bhat3 = np.linalg.pinv(U) * scale[None, :]            # columns rescaled to reference channel scale
    Wk3 = torch.einsum('oikl,ij->ojkl', Wo, torch.from_numpy(Bhat3).float())
    est3 = project(Wk3, Wr, pi=np.argsort(order))         # column order from the correlation matching
    err_ica_ref = ((est3 - Wv).norm() / Wv.norm()).item()
    Wk_true = torch.einsum('oikl,ij->ojkl', Wo, oc.B)
    err_true = ((project(Wk_true, Wr, pi=np.arange(C)) - Wv).norm() / Wv.norm()).item()
    Bi = oc.Binv.numpy(); Bi = Bi / np.linalg.norm(Bi, axis=1, keepdims=True)
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    r = dict(arch=ARCH, layer=name, c_in=int(C), rho=rho, repeats=repeats, n_inputs=n, queries=n * repeats,
             alignment=float(np.abs(Bi @ Un.T).max(1).mean()),
             err_reference=ref_err, err_projection_estimated_B=err_ica, err_projection_estimated_B_refscaled=err_ica_ref,
             err_projection_true_B=err_true)
    rows.append(r); print(r, flush=True)
save_json(rows, f'blind_ica{TAG}_{ARCH}_L{LAYER}.json')

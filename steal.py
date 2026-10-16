"""E3: model-stealing by a device-level adversary who reads the edge-side weights.

Usage: python steal.py ARCH N [conds]
Conditions (attacker owns N labelled in-distribution samples, disjoint from members):
  scratch        train the full architecture from random initialisation
  public         fine-tune the reference checkpoint itself (public-model scenario baseline)
  plain          initialise the edge prefix from the plaintext victim weights (split only, no obfuscation)
  ghost          initialise the edge prefix from the stored GHOST tensors as read from REE memory
  [lvl_]atk_ref  adaptive weight recovery: atk in {arrow (direction matching), unmix (least-squares
                 un-mixing), projK (row-space projection with the input-side transform given, upper
                 bound)}; ref in {own (attacker's scratch model), oracle (independent model trained on
                 the victim's data, or the public base model)}; lvl in {ps, left} selects an ablated
                 obfuscation level (default: full GHOST).
All conditions then fine-tune every layer for the same number of steps.
"""
import sys, copy
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import *
from ghost import GhostDeployment, edge_conv_names, _get, fold_bn
from matching import match, unmix, project, unmix_bilinear

torch.set_num_threads(int(os.environ.get('NT', '2')))
ARCH = sys.argv[1]; N = int(sys.argv[2])
CONDS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['scratch', 'plain', 'ghost', 'arrow_own', 'unmix_own', 'arrow_oracle', 'unmix_oracle', 'projK_oracle', 'ps_arrow_oracle', 'left_projK_oracle']
STEPS = int(os.environ.get('STEPS', '1200'))
SPLIT = {'resnet20': 3, 'vgg11h': 4}[ARCH]
SEED = 0

(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
Xa, ya = Xp[:N], yp[:N]
VICTIM = os.environ.get('VICTIM', f'{ARCH}_victim_s0'); ORACLE = os.environ.get('ORACLE', f'{ARCH}_victim_s1'); TAG = os.environ.get('TAG', ''); BASE = os.environ.get('BASE', 'random')
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{VICTIM}.pt')); victim.eval()
DEPS = {lv: GhostDeployment(victim, ARCH, SPLIT, rho=10.0, seed=1234, Xcal=Xa[:256], level=lv)
        for lv in ('full', 'left', 'ps')}          # GHOST and the two ablations
WTS = {lv: d.obfuscated_weights() for lv, d in DEPS.items()}
dep, Wt = DEPS['full'], WTS['full']
names = dep.names
VFOLD = fold_bn(victim)
WERR = {}


def rel_err(est, n):
    Wv = _get(VFOLD, n).weight.detach()
    return ((est[:Wv.shape[0]] - Wv).norm() / Wv.norm()).item()


def prefix_modules(model):
    """(conv, bn) module pairs of the edge prefix in an unfolded model, keyed by folded conv name."""
    out = {}
    for n in (names if ARCH == 'resnet20' else []):
        if True:
            if n == 'conv1': out[n] = (model.conv1, model.bn1)
            elif n.endswith('short.0'):
                blk = _get(model, n.rsplit('.', 2)[0]); out[n] = (blk.short[0], blk.short[1])
            else:
                blk = _get(model, n.rsplit('.', 1)[0]); k = n[-1]
                out[n] = (getattr(blk, 'conv' + k), getattr(blk, 'bn' + k))
    if ARCH != 'resnet20':
        f = list(model.features)
        convs = [i for i, l in enumerate(f) if isinstance(l, nn.Conv2d)]                     # unfolded conv positions
        fconvs = [i for i, l in enumerate(VFOLD.features) if isinstance(l, nn.Conv2d)]       # folded conv positions
        for n in names:                         # same ordinal in the folded and unfolded sequences
            k = fconvs.index(int(n.split('.')[1])); i = convs[k]; out[n] = (f[i], f[i + 1])
    return out


def build(cond, ref=None):
    m = ARCHS[ARCH]()
    seed_all(SEED)  # identical random initialisation of the non-prefix part across conditions
    m = ARCHS[ARCH]()
    if cond == 'scratch':
        return m
    if cond == 'public':          # fine-tune the reference checkpoint itself, no stolen weights
        m.load_state_dict(ref.state_dict()); return m
    if BASE == 'ref' and ref is not None:   # public-model scenario: non-prefix layers start from the public checkpoint
        m.load_state_dict(ref.state_dict())
    pm = prefix_modules(m)
    if cond == 'plain':
        vp = prefix_modules(victim)
        for n in names:
            pm[n][0].load_state_dict(vp[n][0].state_dict()); pm[n][1].load_state_dict(vp[n][1].state_dict())
        return m
    if cond == 'ghost':
        for n in names: pm[n][0].weight.data = Wt[n][:pm[n][0].weight.shape[0]].clone()   # naive: first C_out stored rows
        WERR[cond] = float(np.mean([rel_err(Wt[n], n) for n in names]))
        return m
    # adaptive weight-recovery attacks: [level_]attack_reference
    parts = cond.split('_')
    level = parts[0] if parts[0] in ('ps', 'left') else 'full'
    attack = parts[-2]
    D, W_src = DEPS[level], WTS[level]
    rp = prefix_modules(ref); rf = fold_bn(ref)
    for n in names:
        Wr = _get(rf, n).weight.detach()        # reference in the same (BN-folded) space as the stored tensors
        Wo = W_src[n]
        if attack == 'arrow':                                   # direction matching
            est = match(Wo, Wr)
        elif attack == 'unmix':                                 # least-squares un-mixing
            est = unmix_bilinear(Wo, Wr)[0] if level == 'full' else unmix(Wo, Wr)[0]
        elif attack == 'projK':                                 # upper bound: input-side transform known
            oc = _get(D.model, n)
            if level == 'full':
                Wk = torch.einsum('oikl,ij->ojkl', Wo, oc.B)    # undo B^{-1}: rows = P A [W; F]
                est = project(Wk, Wr, pi=np.arange(Wr.shape[1]))
            else:
                est = project(Wo, Wr, pi=torch.argsort(oc.Q).numpy())
        else:
            raise ValueError(cond)
        pm[n][0].weight.data = est
        pm[n][1].load_state_dict(rp[n][1].state_dict())       # reference BN statistics as the best available guess
        WERR.setdefault(cond, []).append(rel_err(est, n))
    WERR[cond] = float(np.mean(WERR[cond]))
    WERR['ref_' + cond] = float(np.mean([rel_err(_get(rf, n).weight.detach(), n) for n in names]))
    return m


res = {}
ref_own = None
for cond in CONDS:
    t0 = time.time()
    if cond.endswith('_own'):
        path = f'{CKPT}/steal{TAG}_{ARCH}_n{N}_scratch.pt'
        ref_own = ARCHS[ARCH](); ref_own.load_state_dict(torch.load(path)); ref = ref_own
    elif cond.endswith('_oracle') or cond == 'public' or BASE == 'ref':
        ref = ARCHS[ARCH](); ref.load_state_dict(torch.load(f'{CKPT}/{ORACLE}.pt'))
    else:
        ref = None
    m = build(cond, ref)
    acc0 = accuracy(m, Xt, yt)
    train_steps(m, Xa, ya, steps=STEPS, seed=SEED)
    acc = accuracy(m, Xt, yt)
    agree = (logits_of(m, Xt).argmax(1) == logits_of(victim, Xt).argmax(1)).float().mean().item() * 100
    if cond == 'scratch': torch.save(m.state_dict(), f'{CKPT}/steal{TAG}_{ARCH}_n{N}_scratch.pt')
    r = dict(arch=ARCH, n=N, cond=cond, steps=STEPS, acc_init=acc0, acc=acc, fidelity=agree, secs=time.time() - t0)
    print(r, flush=True)
    r['victim'] = VICTIM; r['oracle_ref'] = ORACLE; r['base'] = BASE
    r['w_rel_err'] = WERR.get(cond); r['ref_rel_err'] = WERR.get('ref_' + cond)
    save_json(r, f'steal{TAG}_{ARCH}_n{N}_{cond}.json')

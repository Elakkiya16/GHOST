"""E5 query-based extraction under a TEE-enforced budget and E6 membership inference
under output-release policies.  Usage: python extract_mia.py ARCH [mia|extract]"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from common import *
from ghost import GhostDeployment

torch.set_num_threads(int(os.environ.get('NT', '2')))
ARCH = sys.argv[1]; WHAT = sys.argv[2] if len(sys.argv) > 2 else 'both'
MAIN = {'resnet20': 3, 'vgg11h': 4}[ARCH]
STEPS = int(os.environ.get('STEPS', '1200'))
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{ARCH}_victim_s0.pt')); victim.eval()
dep = GhostDeployment(victim, ARCH, MAIN, rho=10, seed=5, Xcal=Xp[:256], encrypt=False)


def release(logits, policy):
    p = F.softmax(logits, 1)
    if policy == 'probs': return p
    if policy == 'top1_rounded':     # label plus top-1 confidence rounded to one decimal
        q = torch.zeros_like(p); c, i = p.max(1)
        q[torch.arange(len(p)), i] = torch.round(c * 10) / 10
        return q
    if policy == 'label':
        return F.one_hot(p.argmax(1), p.size(1)).float()


def ghost_logits(X):
    return torch.cat([dep.query(X[k:k + 500]) for k in range(0, len(X), 500)])


if WHAT in ('mia', 'both'):
    rows = []
    g = torch.Generator().manual_seed(3)
    mi = torch.randperm(len(Xm), generator=g)[:10000]
    Xin, yin = Xm[mi], ym[mi]
    for engine in ['plain', 'ghost']:
        Lin = logits_of(victim, Xin) if engine == 'plain' else ghost_logits(Xin)
        Lout = logits_of(victim, Xt) if engine == 'plain' else ghost_logits(Xt)
        for policy in ['probs', 'top1_rounded', 'label']:
            Pin, Pout = release(Lin, policy), release(Lout, policy)
            lab = np.r_[np.ones(len(Pin)), np.zeros(len(Pout))]
            P = torch.cat([Pin, Pout]).double(); Y = torch.cat([yin, yt])
            conf_true = P[torch.arange(len(P)), Y].clamp(1e-12, 1)
            scores = {
                'loss': (conf_true.log()).numpy(),                      # Yeom et al. loss attack
                'confidence': P.max(1).values.numpy(),
                'correctness': (P.argmax(1) == Y).float().numpy(),      # label-only gap attack
            }
            # modified entropy (Song & Mittal) needs the full vector
            pc = P.clamp(1e-12, 1 - 1e-12)
            py = pc[torch.arange(len(P)), Y]
            other = pc.clone(); other[torch.arange(len(P)), Y] = 0
            ment = -((1 - py) * py.log() + (other * (1 - other).log()).sum(1))
            scores['mentropy'] = (-ment).numpy()
            r = dict(arch=ARCH, engine=engine, policy=policy)
            for k, s in scores.items(): r['auc_' + k] = float(roc_auc_score(lab, s))
            r['train_acc'] = (Pin.argmax(1) == yin).float().mean().item() * 100
            r['test_acc'] = (Pout.argmax(1) == yt).float().mean().item() * 100
            rows.append(r); print(r, flush=True)
    save_json(rows, f'mia_{ARCH}.json')

if WHAT in ('extract', 'both'):
    rows = []
    for budget in [1000, 5000, 10000]:
        Xq = Xp[:budget]
        Lq = ghost_logits(Xq)
        for policy in ['probs', 'label']:
            soft = release(Lq, policy)
            sur = ARCHS[ARCH]()
            train_steps(sur, Xq, soft.argmax(1), steps=STEPS, seed=0, soft=soft if policy == 'probs' else None)
            acc = accuracy(sur, Xt, yt)
            fid = (logits_of(sur, Xt).argmax(1) == logits_of(victim, Xt).argmax(1)).float().mean().item() * 100
            r = dict(arch=ARCH, budget=budget, policy=policy, acc=acc, fidelity=fid, steps=STEPS)
            rows.append(r); print(r, flush=True)
            save_json(rows, f'extract_{ARCH}.json')

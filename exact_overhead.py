"""E1 functional exactness and E2 overhead.  Usage: python exact_overhead.py ARCH"""
import sys
import numpy as np
from common import *
from ghost import GhostDeployment, fold_bn

torch.set_num_threads(int(os.environ.get('NT', '2')))
ARCH = sys.argv[1]
SPLITS = {'resnet20': [1, 3, 6], 'vgg11h': [2, 4, 6]}[ARCH]
MAIN = {'resnet20': 3, 'vgg11h': 4}[ARCH]
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
victim = ARCHS[ARCH](); victim.load_state_dict(torch.load(f'{CKPT}/{ARCH}_victim_s0.pt')); victim.eval()
ref = logits_of(victim, Xt)
base_acc = (ref.argmax(1) == yt).float().mean().item() * 100

rows = []
for rho in [0, 1, 10, 100, 1000, 10000]:
    dep = GhostDeployment(victim, ARCH, MAIN, rho=rho, seed=7, Xcal=Xp[:256], encrypt=False)
    outs = []
    for k in range(0, len(Xt), 500):
        outs.append(dep.query(Xt[k:k + 500]))
    out = torch.cat(outs)
    rows.append(dict(arch=ARCH, rho=rho, base_acc=base_acc,
                     ghost_acc=(out.argmax(1) == yt).float().mean().item() * 100,
                     agreement=(out.argmax(1) == ref.argmax(1)).float().mean().item() * 100,
                     max_abs_logit_err=(out - ref).abs().max().item(),
                     rel_logit_err=((out - ref).norm() / ref.norm()).item()))
    print(rows[-1], flush=True)
save_json(rows, f'exact_{ARCH}.json')

# ---- E2: per-query cost at batch size 1
torch.set_num_threads(1)
folded = fold_bn(victim)
xq = Xt[:1000]
lat = []
for split in SPLITS:
    dep = GhostDeployment(victim, ARCH, split, rho=10, seed=7, Xcal=Xp[:256], encrypt=True)
    with torch.no_grad():
        for i in range(10): dep.query(xq[i:i + 1]); folded(xq[i:i + 1])     # warm-up
    dep.reset_timers()
    tg, tp = [], []
    with torch.no_grad():
        for b in range(0, len(xq), 50):               # alternate blocks of 50 to cancel drift
            for i in range(b, b + 50):
                t0 = time.perf_counter(); folded(xq[i:i + 1]); tp.append(time.perf_counter() - t0)
            for i in range(b, b + 50):
                off0 = dep.timers['tee_offline']
                t0 = time.perf_counter(); dep.query(xq[i:i + 1]); dt = time.perf_counter() - t0
                tg.append(dt - (dep.timers['tee_offline'] - off0))
    plain = float(np.median(tp)) * 1e3
    total = float(np.median(tg)) * 1e3
    T = {k: v / len(xq) for k, v in dep.timers.items()}
    nparams_edge = sum(dep.obfuscated_weights()[n].numel() for n in dep.names)
    nparams = sum(p.numel() for p in folded.parameters())
    r = dict(arch=ARCH, split=split, n_edge_convs=len(dep.names) + len(dep.tee_names), n_outsourced=len(dep.names), edge_param_frac=nparams_edge / nparams * 100,
             plain_ms=plain, ghost_total_ms=total,
             tee_online_ms=T['tee_online'] * 1e3, ree_ms=T['ree'] * 1e3, tee_offline_ms=T['tee_offline'] * 1e3,
             aes_ms=T['aes'] * 1e3, kb_tee_to_ree=T['bytes_out'] / 1024, kb_ree_to_tee=T['bytes_in'] / 1024,
             kb_to_cloud=T['bytes_cloud'] / 1024)
    r['overhead_pct'] = (r['ghost_total_ms'] / plain - 1) * 100
    r['added_work_pct'] = (r['tee_online_ms'] + r['aes_ms']) / plain * 100      # directly measured additional work
    lat.append(r); print(r, flush=True)
save_json(lat, f'overhead_{ARCH}.json')

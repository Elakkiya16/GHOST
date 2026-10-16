"""E7: behaviour of the token-gated TEE under adversarial request patterns."""
import hashlib, os
from common import *
from ghost import GhostDeployment

torch.set_num_threads(1)
victim = ARCHS['resnet20'](); victim.load_state_dict(torch.load(f'{CKPT}/resnet20_victim_s0.pt')); victim.eval()
(Xm, ym), (Xp, yp), (Xt, yt) = splits(0)
dep = GhostDeployment(victim, 'resnet20', 3, rho=10, seed=3, budget=5, Xcal=Xp[:64])
x = Xt[:1]; x2 = Xt[1:2]
dg = lambda z: hashlib.sha256(z.numpy().tobytes()).digest()


def attempt(label, fn):
    try:
        fn(); out = 'served'
    except PermissionError:
        out = 'refused'
    rows.append(dict(scenario=label, outcome=out)); print(label, out)


rows = []
tok1 = dep.tee.issue_token(1, dg(x))
attempt('Valid token, fresh counter', lambda: dep.query(x, token=tok1, counter=1))
attempt('Replay of the same token and counter', lambda: dep.query(x, token=tok1, counter=1))
attempt('Valid token reused for a different input', lambda: dep.query(x2, token=dep.tee.issue_token(2, dg(x)), counter=2))
attempt('Forged token (random 256-bit string)', lambda: dep.query(x, token=os.urandom(32), counter=3))
attempt('Direct REE execution without authorisation', lambda: dep.model(x))
for c in range(4, 12):
    attempt(f'Valid token, counter {c} (budget 5)', lambda c=c: dep.query(x, token=dep.tee.issue_token(c, dg(x)), counter=c))
save_json(rows, 'token_tests.json')

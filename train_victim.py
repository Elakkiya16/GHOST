import sys, torch
from common import *
torch.set_num_threads(int(os.environ.get('NT', '2')))
arch, ep, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
(Xm, ym), _, (Xt, yt) = splits(0)
m = ARCHS[arch]()
logf = open(f'{OUT}/train_{arch}_s{seed}.log', 'a')
def log(s): logf.write(s + '\n'); logf.flush()
train(m, Xm, ym, epochs=ep, log=log, seed=seed, resume=f'{CKPT}/{arch}_s{seed}_resume.pt')
torch.save(m.state_dict(), f'{CKPT}/{arch}_victim_s{seed}.pt')
acc = accuracy(m, Xt, yt); log(f'test acc {acc:.2f}')
save_json({'arch': arch, 'epochs': ep, 'seed': seed, 'test_acc': acc}, f'victim_{arch}_s{seed}.json')

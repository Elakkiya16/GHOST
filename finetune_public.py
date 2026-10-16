"""Victim obtained by fine-tuning a 'public' checkpoint (seed-1 model) on the member set,
the setting in which direction-matching attacks against permute-and-scale schemes succeed."""
import torch
from common import *
torch.set_num_threads(int(os.environ.get('NT', '2')))
(Xm, ym), _, (Xt, yt) = splits(0)
m = ARCHS['resnet20'](); m.load_state_dict(torch.load(f'{CKPT}/resnet20_victim_s1.pt'))
train(m, Xm, ym, epochs=3, lr=0.01, seed=5)
torch.save(m.state_dict(), f'{CKPT}/resnet20_victimft.pt')
save_json({'test_acc': accuracy(m, Xt, yt)}, 'victimft_resnet20.json')

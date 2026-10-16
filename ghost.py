"""GHOST v2: exact, TEE-anchored obfuscation of the edge-side subnetwork.

For every convolution (batch normalisation folded in) of the edge prefix, the untrusted rich
execution environment (REE) stores only

    W~ = P A [W; F] (B^{-1} (x) I_k)

where F are random decoy filters, A and B are secret invertible output- and input-side mixing
matrices and P a secret output permutation, all sealed in the TEE together with the bias b.
Per inference the TEE releases only x~ = B a + R with a fresh one-time mask R, the REE computes
y~ = W~ * x~, and the TEE recovers the exact pre-activation

    W * a + b = [A^{-1} P^T (y~ - W~ * R)]_real + b

with W~ * R precomputed offline. Non-linearities, pooling, residual additions and the bias stay
inside the TEE. At the split point the TEE encrypts the activation with AES-GCM for the cloud.
"""
import copy, hmac, hashlib, os, time, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ------------------------------------------------------------------ BN folding
def _fold(conv, bn):
    w = conv.weight.detach().clone()
    b = conv.bias.detach().clone() if conv.bias is not None else torch.zeros(w.size(0))
    g = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=True)
    new.weight.data = w * g.view(-1, 1, 1, 1)
    new.bias.data = (b - bn.running_mean) * g + bn.bias.detach()
    return new


def fold_bn(model):
    """Return an eval-mode copy with every Conv-BN pair folded into a biased Conv."""
    m = copy.deepcopy(model).eval()
    if hasattr(m, 'blocks'):  # ResNetC
        m.conv1 = _fold(m.conv1, m.bn1); m.bn1 = nn.Identity()
        for blk in m.blocks:
            blk.conv1 = _fold(blk.conv1, blk.bn1); blk.bn1 = nn.Identity()
            blk.conv2 = _fold(blk.conv2, blk.bn2); blk.bn2 = nn.Identity()
            if blk.short is not None:
                blk.short = nn.Sequential(_fold(blk.short[0], blk.short[1]))
    else:  # VGGC
        layers, f = [], list(m.features)
        i = 0
        while i < len(f):
            if isinstance(f[i], nn.Conv2d) and i + 1 < len(f) and isinstance(f[i + 1], nn.BatchNorm2d):
                layers.append(_fold(f[i], f[i + 1])); i += 2
            else:
                layers.append(f[i]); i += 1
        m.features = nn.Sequential(*layers)
    return m


def edge_conv_names(model, arch, split):
    """Names of the convolutions that execute on the edge device.
    resnet20: stem + first `split` residual blocks. vgg11h: first `split` convs."""
    names = []
    if arch == 'resnet20':
        names.append('conv1')
        for k in range(split):
            names += [f'blocks.{k}.conv1', f'blocks.{k}.conv2']
            if model.blocks[k].short is not None:
                names.append(f'blocks.{k}.short.0')
    else:
        c = 0
        for i, l in enumerate(model.features):
            if isinstance(l, nn.Conv2d):
                if c < split: names.append(f'features.{i}')
                c += 1
    return names


def split_module_name(arch, split):
    """Module whose output is the split-point activation sent to the cloud."""
    if arch == 'resnet20':
        return f'blocks.{split - 1}'
    return None  # handled by index for VGG


# ------------------------------------------------------------------ TEE model
class TEE:
    """Simulated trusted execution environment: sealed secrets, mask generation,
    token verification with replay protection and a query budget."""

    def __init__(self, seed=None, budget=None, key=None):
        self.rng = np.random.default_rng(seed if seed is not None else int.from_bytes(os.urandom(8), 'little'))
        self.tg = torch.Generator().manual_seed(int(self.rng.integers(2**62)))
        self.k_tok = key or os.urandom(32)       # provisioned token key (HMAC-SHA256)
        self.k_t = AESGCM.generate_key(256)      # edge->cloud transport key
        self.budget = budget
        self.counter = -1
        self.served = 0
        self.authorised = False

    # authorisation ------------------------------------------------------
    def issue_token(self, counter, digest):
        """Run by the model owner / provisioning service, not by the device."""
        return hmac.new(self.k_tok, struct.pack('>Q', counter) + digest, hashlib.sha256).digest()

    def authorise(self, counter, digest, token, n_inputs=1):
        ok = hmac.compare_digest(token, self.issue_token(counter, digest))
        ok = ok and counter > self.counter                       # replay protection
        ok = ok and (self.budget is None or self.served + n_inputs <= self.budget)   # budget counts inputs
        if ok:
            self.counter = counter; self.served += n_inputs
        self.authorised = ok
        return ok

    # secrets -------------------------------------------------------------
    def perm(self, n):
        return torch.from_numpy(self.rng.permutation(n)).long()

    def scales(self, n, lo=0.25, hi=4.0):
        mag = np.exp(self.rng.uniform(np.log(lo), np.log(hi), n))
        sgn = self.rng.choice([-1.0, 1.0], n)
        return torch.from_numpy(mag * sgn).float()

    def mask(self, shape, std):
        return torch.randn(shape, generator=self.tg) * std


class REEView:
    """Everything the untrusted side observes, for the attack experiments."""
    def __init__(self):
        self.enabled = False
        self.only = None    # record a single layer when set
        self.inputs = {}    # masked inputs x~ per layer
        self.outputs = {}

    def record(self, name, xt, yt):
        if self.enabled and (self.only is None or self.only == name):
            self.inputs.setdefault(name, []).append(xt.detach().clone())
            self.outputs.setdefault(name, []).append(yt.detach().clone())


def _mixing(tee, n, scaled=True):
    """Random well-conditioned invertible matrix: a Haar-random orthogonal matrix, optionally
    multiplied by random signed log-uniform scales (output side only)."""
    G = torch.from_numpy(tee.rng.standard_normal((n, n))).float()
    O, R = torch.linalg.qr(G); O = O * torch.sign(torch.diagonal(R)).view(1, -1)
    return tee.scales(n).view(-1, 1) * O if scaled else O


class ObfConv(nn.Module):
    """One edge convolution under GHOST.

    level='full' (GHOST): decoy filters F are appended to the real filters W, the stack is mixed on
    the output side by a secret invertible A and permuted by P, and mixed on the input-channel side by
    a secret invertible B, so REE memory holds  W~ = P A [W; F] (B^{-1} (x) I_k).
    level='left': output mixing and decoys only, input channels permuted by Q (ablation).
    level='ps'  : permute-and-scale only, W~ = P D W Q^T (ablation; the class attacked by
                  direction matching).
    The TEE sends x~ = B a + R (or Q a + R) and recovers W*a + b = [A^{-1} P^T (y~ - W~*R)]_real + b."""

    def __init__(self, name, conv, tee, view, rho, rms_in, timers, level='full', decoy=0.25):
        super().__init__()
        self.name, self.tee, self.view, self.timers, self.level = name, tee, view, timers, level
        W = conv.weight.detach()
        self.stride, self.padding = conv.stride, conv.padding
        cout, cin = W.shape[:2]
        nd = int(np.ceil(decoy * cout)) if (decoy > 0 and level != 'ps') else 0
        if nd:
            rn = W.reshape(cout, -1).norm(dim=1)
            Fk = torch.from_numpy(tee.rng.standard_normal((nd,) + tuple(W.shape[1:]))).float()
            Fk = Fk / Fk.reshape(nd, -1).norm(dim=1).view(-1, 1, 1, 1)
            Fk = Fk * rn[torch.from_numpy(tee.rng.integers(0, cout, nd))].view(-1, 1, 1, 1)
            Wx = torch.cat([W, Fk])
        else:
            Wx = W
        co = cout + nd
        self.cout, self.nd = cout, nd
        # ---- secrets sealed in the TEE
        self.P = tee.perm(co); self.Pinv = torch.argsort(self.P)
        if level == 'ps':
            self.A = torch.diag(tee.scales(co))
        else:
            self.A = _mixing(tee, co)
        self.Ainv = torch.linalg.inv(self.A.double()).float()[:cout]     # rows for the real channels only
        if level == 'full':
            self.B = _mixing(tee, cin, scaled=False); self.Binv = self.B.t().contiguous()   # orthogonal: preserves the RMS the mask is calibrated to
            self.Q = None
        else:
            self.B = None; self.Q = tee.perm(cin)
        self.b = conv.bias.detach().clone()
        self.rho, self.rms_in = rho, rms_in
        # ---- stored in REE memory
        Wm = torch.einsum('oc,cikl->oikl', self.A, Wx)[self.P]
        if level == 'full':
            Wm = torch.einsum('oikl,ij->ojkl', Wm, self.Binv)
        else:
            Wm = Wm[:, self.Q]
        self.Wt = Wm.contiguous()

    def ree_conv(self, x):
        return F.conv2d(x, self.Wt, None, self.stride, self.padding)

    def _in(self, a):
        return torch.einsum('ij,bjhw->bihw', self.B, a) if self.B is not None else a[:, self.Q]

    def forward(self, a):
        if not self.tee.authorised:
            raise PermissionError('TEE refused: no valid authorisation token for this query')
        t = time.perf_counter
        # --- TEE (online): input mixing / permutation and one-time mask
        t0 = t()
        aq = self._in(a)
        R = self.tee.mask(aq.shape, self.rho * self.rms_in) if self.rho > 0 else None
        xt = aq + R if R is not None else aq
        t1 = t()
        # --- TEE (offline, idle time): mask correction W~ * R
        M = self.ree_conv(R) if R is not None else None
        t2 = t()
        # --- REE: heavy convolution on obfuscated operands
        yt = self.ree_conv(xt)
        t3 = t()
        self.view.record(self.name, xt, yt)
        # --- TEE (online): unmask, unpermute, unmix, drop decoys, add bias
        z = (yt - M if M is not None else yt)[:, self.Pinv]
        y = torch.einsum('oc,bchw->bohw', self.Ainv, z) + self.b.view(1, -1, 1, 1)
        t4 = t()
        self.timers['tee_online'] += (t1 - t0) + (t4 - t3)
        self.timers['tee_offline'] += (t2 - t1)
        self.timers['ree'] += (t3 - t2)
        self.timers['bytes_out'] += xt.numel() * 4
        self.timers['bytes_in'] += yt.numel() * 4
        return y


def _get(root, name):
    m = root
    for p in name.split('.'): m = getattr(m, p)
    return m


def _set(root, name, new):
    parts = name.split('.'); m = root
    for p in parts[:-1]: m = getattr(m, p)
    setattr(m, parts[-1], new)


@torch.no_grad()
def calibrate_rms(folded, names, Xcal):
    """RMS of the input activation of each edge convolution (sets the mask scale)."""
    rms, hooks = {}, []
    for n in names:
        def h(mod, inp, out, n=n):
            rms[n] = inp[0].pow(2).mean().sqrt().item()
        hooks.append(_get(folded, n).register_forward_hook(h))
    folded(Xcal)
    for h in hooks: h.remove()
    return rms


class GhostDeployment:
    """A folded model whose edge prefix is obfuscated and whose split activation is
    AES-GCM encrypted for the cloud suffix."""

    def __init__(self, model, arch, split, rho=10.0, seed=0, budget=None, Xcal=None, encrypt=True, level='full', decoy=0.25, tee_first=True):
        self.folded = fold_bn(model)
        self.arch, self.split, self.rho = arch, split, rho
        all_names = edge_conv_names(self.folded, arch, split)
        # The first convolution reads the public input, so its input-side mixing could be solved from
        # chosen queries. It is executed inside the TEE (negligible cost) and only later layers are outsourced.
        self.tee_names = all_names[:1] if tee_first else []
        self.names = all_names[1:] if tee_first else all_names
        self.tee = TEE(seed=seed, budget=budget)
        self.view = REEView()
        self.timers = dict(tee_online=0., tee_offline=0., ree=0., aes=0., bytes_out=0, bytes_in=0, bytes_cloud=0)
        rms = calibrate_rms(self.folded, self.names, Xcal) if Xcal is not None else {n: 1.0 for n in self.names}
        self.model = copy.deepcopy(self.folded)
        for n in self.names:
            _set(self.model, n, ObfConv(n, _get(self.folded, n), self.tee, self.view, rho, rms[n], self.timers, level, decoy))
        for n in self.tee_names:                       # time the TEE-executed first layer
            def pre(mod, inp, d=self): d._t0 = time.perf_counter()
            def post(mod, inp, out, d=self): d.timers['tee_online'] += time.perf_counter() - d._t0
            _get(self.model, n).register_forward_pre_hook(pre); _get(self.model, n).register_forward_hook(post)
        self.encrypt = encrypt
        self._install_split_hook()
        self.counter = 0

    def _install_split_hook(self):
        if self.arch == 'resnet20':
            mod = _get(self.model, f'blocks.{self.split - 1}')
        else:
            # output of the last edge conv's ReLU (and pool if it follows)
            idx = int(self.names[-1].split('.')[1])
            f = self.model.features
            j = idx + 1
            while j < len(f) and not isinstance(f[j], (nn.Conv2d,)) and isinstance(f[j], (nn.ReLU, nn.MaxPool2d)):
                j += 1
            mod = f[j - 1]
        dep = self

        def hook(m, inp, out):
            if not dep.encrypt: return out
            t0 = time.perf_counter()
            nonce = os.urandom(12)
            ct = AESGCM(dep.tee.k_t).encrypt(nonce, out.detach().numpy().tobytes(), None)
            dep.timers['bytes_cloud'] += len(ct)
            pt = AESGCM(dep.tee.k_t).decrypt(nonce, ct, None)       # cloud side
            y = torch.from_numpy(np.frombuffer(pt, dtype=np.float32).copy()).view_as(out)
            dep.timers['aes'] += time.perf_counter() - t0
            return y
        mod.register_forward_hook(hook)

    @torch.no_grad()
    def query(self, x, token=None, counter=None):
        """One authorised request of x.shape[0] inputs; the budget is charged per input."""
        digest = hashlib.sha256(x.numpy().tobytes()).digest()
        if counter is None:
            self.counter += 1; counter = self.counter
        if token is None:      # honest client: token issued by the model owner
            token = self.tee.issue_token(counter, digest)
        if not self.tee.authorise(counter, digest, token, n_inputs=x.shape[0]):
            raise PermissionError('TEE refused the query')
        try:
            return self.model(x)
        finally:
            self.tee.authorised = False      # authorisation covers exactly one request

    def obfuscated_weights(self):
        """What a device-level adversary can read from REE memory."""
        return {n: _get(self.model, n).Wt.clone() for n in self.names}

    def reset_timers(self):
        for k in self.timers: self.timers[k] = 0

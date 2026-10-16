import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _alternate(A, B, col, iters):
    coa, ci = A.shape[:2]; co = B.shape[0]
    for _ in range(iters):
        a = _unit(A[:, col].reshape(coa, -1)); b = _unit(B.reshape(co, -1))
        S = np.abs(b @ a.T)
        _, row = linear_sum_assignment(-S)
        sg = np.sign(np.einsum('ij,ij->i', b, a[row]) + 1e-12)
        Ar = A[row] * sg[:, None, None]
        a2 = _unit(np.transpose(Ar, (1, 0, 2)).reshape(ci, -1)); b2 = _unit(np.transpose(B, (1, 0, 2)).reshape(ci, -1))
        _, newcol = linear_sum_assignment(-(b2 @ a2.T))
        if np.array_equal(newcol, col): break
        col = newcol
    a = _unit(A[row][:, col].reshape(co, -1) * sg[:, None]); b = _unit(B.reshape(co, -1))
    score = np.einsum('ij,ij->i', a, b).sum()
    return row, col, sg, score


def match(Wo, Wr, iters=20, restarts=24, seed=0):
    """Weight-matching attack: recover the output and input channel permutations and the
    output signs of an obfuscated tensor Wo = D P W Q^T against a reference tensor Wr by
    alternating Hungarian assignment on sign-invariant cosine similarity, with random
    restarts. Output scales are unidentifiable and are set to the reference row norms."""
    coa, ci = Wo.shape[:2]; co = Wr.shape[0]
    A = Wo.reshape(coa, ci, -1).numpy().astype(np.float64)
    B = Wr.reshape(co, ci, -1).numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    inits = [np.arange(ci)] + [rng.permutation(ci) for _ in range(restarts)]
    best = None
    for col in inits:
        r = _alternate(A, B, col, iters)
        if best is None or r[3] > best[3]: best = r
    row, col, sg, _ = best
    W = torch.from_numpy(A[row][:, col] * sg[:, None, None]).reshape(Wr.shape).float()
    rn = Wr.reshape(co, -1).norm(dim=1); wn = W.reshape(co, -1).norm(dim=1) + 1e-8
    return W * (rn / wn).view(-1, *([1] * (W.dim() - 1)))


def unmix(Wo, Wr, iters=15, restarts=24, seed=0, return_col=False):
    """Least-squares un-mixing attack against mixed obfuscation: jointly estimate the input
    permutation (Hungarian matching) and a general output transform T (least squares) such that
    Wo[:, Q] ~= T Wr, and return T^+ Wo[:, Q] as the attacker's estimate in the reference basis."""
    coa, ci = Wo.shape[:2]; co = Wr.shape[0]
    A = Wo.reshape(coa, ci, -1).numpy().astype(np.float64)
    B = Wr.reshape(co, ci, -1).numpy().astype(np.float64)
    Bf = B.reshape(co, -1)
    rng = np.random.default_rng(seed)
    best = None
    for col in [np.arange(ci)] + [rng.permutation(ci) for _ in range(restarts)]:
        for _ in range(iters):
            Af = A[:, col].reshape(coa, -1)
            T = np.linalg.lstsq(Bf.T, Af.T, rcond=None)[0].T          # Af ~= T Bf
            U = np.linalg.pinv(T) @ A.reshape(coa, -1)                 # estimate in reference basis
            U = U.reshape(co, ci, -1)
            a2 = _unit(np.transpose(U, (1, 0, 2)).reshape(ci, -1)); b2 = _unit(np.transpose(B, (1, 0, 2)).reshape(ci, -1))
            _, newcol = linear_sum_assignment(-(b2 @ a2.T))
            if np.array_equal(newcol, col): break
            col = newcol
        Af = A[:, col].reshape(coa, -1)
        T = np.linalg.lstsq(Bf.T, Af.T, rcond=None)[0].T
        res = np.linalg.norm(Af - T @ Bf) / (np.linalg.norm(Af) + 1e-12)
        if best is None or res < best[0]: best = (res, col, T)
    res, col, T = best
    U = (np.linalg.pinv(T) @ A[:, col].reshape(coa, -1)).reshape(Wr.shape)
    if return_col:
        return torch.from_numpy(U).float(), float(res), col
    return torch.from_numpy(U).float(), float(res)


def project(Wo, Wr, iters=30, restarts=12, seed=0, pi=None):
    """Subspace-projection attack. Left mixing and decoy rows leave the row space of the stored
    tensor equal to span(real rows, decoy rows). The adversary searches for the input permutation
    that maximises the energy of the reference rows inside the observed row space (linearised
    quadratic assignment with restarts, initialised from the un-mixing estimate) and returns the
    projection of the reference rows onto that space."""
    coa, ci = Wo.shape[:2]; co = Wr.shape[0]
    A = Wo.reshape(coa, ci, -1).numpy().astype(np.float64); k = A.shape[2]
    _, sv, Vt = np.linalg.svd(A.reshape(coa, -1), full_matrices=False)
    V = Vt[sv > sv[0] * 1e-6].T                              # (ci*k, r) basis in obfuscated column order
    M = (V @ V.T).reshape(ci, k, ci, k)                      # projector blocks M[i, :, i', :]
    B = Wr.reshape(co, ci, k).numpy().astype(np.float64)
    C = np.einsum('ojk,opl->jkpl', B, B)                     # reference cross-Gram blocks (j,k,j',l)
    def energy(pi):                                           # pi[j] = obfuscated column of reference column j
        return np.einsum('jkpl,jkpl->', C, M[pi][:, :, pi, :])
    if pi is not None:                                        # upper bound: input permutation given
        best = (None, np.asarray(pi)); inits = []
    else:
        _, _, col0 = unmix(Wo, Wr, return_col=True, restarts=6)
        rng = np.random.default_rng(seed); best = None
        inits = [np.asarray(col0)] + [rng.permutation(ci) for _ in range(restarts)]
    for pi in inits:
        for _ in range(iters):
            Mp = M[:, :, pi, :]                                   # (i, k, j', l)
            cost = np.einsum('jkpl,ikpl->ji', C, Mp)              # gain of placing ref column j at obf column i
            _, newpi = linear_sum_assignment(-cost)
            if np.array_equal(newpi, pi): break
            pi = newpi
        e = energy(pi)
        if best is None or e > best[0]: best = (e, pi)
    pi = best[1]
    # rows of the projection, expressed in reference column order
    Vp = V.reshape(ci, k, -1)[pi].reshape(ci * k, -1)
    Bf = B.reshape(co, -1)
    return torch.from_numpy(Bf @ Vp @ Vp.T).reshape(Wr.shape).float()


def unmix_bilinear(Wo, Wr, iters=50, restarts=8, seed=0):
    """Bilinear least-squares un-mixing against two-sided mixing: find T, S with
    Wo_t ~= T Wr_t S for every kernel tap t, then return T^+ Wo_t S^{-1} (reference basis)."""
    coa, ci = Wo.shape[:2]; co = Wr.shape[0]
    A = Wo.reshape(coa, ci, -1).numpy().astype(np.float64); k = A.shape[2]
    B = Wr.reshape(co, ci, -1).numpy().astype(np.float64)
    Ah = np.concatenate([A[:, :, t] for t in range(k)], 1)            # coa x (ci k)
    Av = np.concatenate([A[:, :, t] for t in range(k)], 0)            # (k coa) x ci
    rng = np.random.default_rng(seed)
    best = None
    inits = [np.eye(ci)] + [np.linalg.qr(rng.standard_normal((ci, ci)))[0] for _ in range(restarts)]
    for S in inits:
        for _ in range(iters):
            Xh = np.concatenate([B[:, :, t] @ S for t in range(k)], 1)   # co x (ci k)
            T = Ah @ np.linalg.pinv(Xh)
            Yv = np.concatenate([T @ B[:, :, t] for t in range(k)], 0)   # (k coa) x ci
            S = np.linalg.pinv(Yv) @ Av
        Xh = np.concatenate([B[:, :, t] @ S for t in range(k)], 1)
        res = np.linalg.norm(Ah - T @ Xh) / (np.linalg.norm(Ah) + 1e-12)
        if best is None or res < best[0]: best = (res, T, S)
    res, T, S = best
    Tp = np.linalg.pinv(T); Si = np.linalg.pinv(S)
    U = np.stack([Tp @ A[:, :, t] @ Si for t in range(k)], 2)
    return torch.from_numpy(U).reshape(Wr.shape).float(), float(res)

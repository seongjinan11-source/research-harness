"""Bounded-context inference. Query labels are not accepted by this module."""
import sklearn  # Windows runtime import order
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
import torch
from torch import nn

ARMS = ('tail', 'proportional', 'top', 'full')
RULES = ('ppr', 'centroid', 'ridge', 'uniform')
BLOCK, QUERY_BLOCK = 4, 64


@dataclass(frozen=True)
class Input:
    x: np.ndarray
    edge: np.ndarray
    context: np.ndarray
    yc: np.ndarray
    query: np.ndarray


def transition(n, edge, dtype=np.float32):
    edge = np.asarray(edge, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2 or np.any((edge < 0) | (edge >= n)):
        raise ValueError('expected graph-local edge array [2,E]')
    a = sp.csr_matrix((np.ones(edge.shape[1], dtype), tuple(edge)), shape=(n, n))
    a = a.maximum(a.T) + sp.eye(n, dtype=dtype, format='csr')
    a.data.fill(1)
    a.sort_indices()
    return sp.diags((1 / np.asarray(a.sum(1)).ravel()).astype(dtype), format='csr') @ a


def standardize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('complete finite node features required')
    s = x.std(0, dtype=np.float64).astype(np.float32)
    return np.clip((x - x.mean(0, dtype=np.float64).astype(np.float32)) / np.maximum(s, 1e-6), -10, 10)


def diffuse(p, seed):
    f = seed.copy()
    restart = .15 * seed
    for _ in range(8):
        f = restart + .85 * (p @ f)
    return f


def preproc(inp):
    x = standardize(inp.x)
    n = len(x)
    if (len(inp.context) != len(inp.yc) or len(inp.context) == 0 or len(inp.query) == 0
            or len(np.unique(inp.context)) != len(inp.context) or len(np.unique(inp.query)) != len(inp.query)
            or np.any((inp.context < 0) | (inp.context >= n)) or np.any((inp.query < 0) | (inp.query >= n))
            or len(np.intersect1d(inp.context, inp.query))):
        raise ValueError('invalid context/query identities')
    if not np.array_equal(np.unique(inp.yc), np.arange(len(np.unique(inp.yc)))):
        raise ValueError('all contiguous classes must have an observed anchor')
    p = transition(n, inp.edge)
    x1 = p @ x
    h = np.stack((x, x1, p @ x1), axis=1)
    classes = [np.sort(inp.context[inp.yc == c]) for c in range(int(inp.yc.max()) + 1)]
    return p, h, classes


def quotas(classes, arm):
    n = np.array([len(v) for v in classes])
    tail = np.where(n <= 4, n, 2)
    if arm in ('tail', 'top'):
        return tail
    if arm == 'full':
        return n
    if arm != 'proportional':
        raise ValueError(arm)
    out = np.ones_like(n)
    for _ in range(int(tail.sum() - len(n))):
        selected = min(np.flatnonzero(out < n), key=lambda c: (-n[c] / out[c], int(classes[c][0])))
        out[selected] += 1
    return out


def partition(ids, n_groups, h):
    if n_groups == len(ids):
        return [ix.reshape(1) for ix in np.split(ids, len(ids))]
    if n_groups == 1:
        return [ids]
    x = h[ids].reshape(len(ids), -1).astype(np.float64)
    x -= x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    if s[0] <= 1e-8 or s[0] - s[1] <= 1e-7 * s[0]:
        order = np.arange(len(ids))
    else:
        proj = u[:, 0] * s[0]
        anchor = np.flatnonzero(np.abs(proj) >= np.max(np.abs(proj)) - 1e-9 * s[0])[0]
        proj *= 1 if proj[anchor] >= 0 else -1
        order = np.lexsort((ids, np.round(proj / s[0], 10)))
    return list(np.array_split(ids[order], n_groups))


def moments(p, h, query, groups):
    """Group masses/centroids. Singleton features never become diffusion columns."""
    q, g, d = len(query), len(groups), int(np.prod(h.shape[1:]))
    flat = h.reshape(len(h), d)
    cnt = np.array([len(a) for a in groups], dtype=np.float32)
    mu = np.stack([flat[a].mean(0) for a in groups])
    mass = np.empty((q, g), dtype=np.float32)
    local = np.broadcast_to(mu, (q, g, d)).copy()
    max_dims, total = 0, 0
    for lo in range(0, g, BLOCK):
        ids = groups[lo:lo + BLOCK]
        dims = len(ids) + d * sum(len(v) > 1 for v in ids)
        seed = np.zeros((len(h), dims), dtype=np.float32)
        start = len(ids)
        for j, a in enumerate(ids):
            seed[a, j] = 1
            if len(a) > 1:
                seed[a, start:start + d] = flat[a]
                start += d
        f = diffuse(p, seed)[query]
        mass[:, lo:lo + len(ids)] = f[:, :len(ids)]
        start = len(ids)
        for j, a in enumerate(ids):
            if len(a) > 1:
                m = f[:, j:j + 1]
                np.divide(f[:, start:start + d], np.where(m > 0, m, 1), out=local[:, lo + j], where=m > 0)
                start += d
        max_dims = max(dims, max_dims)
        total += dims
    return mass, local.reshape(q, g, *h.shape[1:]), mu.reshape(g, *h.shape[1:]), cnt, max_dims, total


def singleton_masses(p, query, ids):
    out = np.empty((len(query), len(ids)), np.float32)
    for lo in range(0, len(ids), BLOCK):
        bs = ids[lo:lo + BLOCK]
        seed = np.zeros((p.shape[0], len(bs)), dtype=np.float32)
        seed[bs, np.arange(len(bs))] = 1
        out[:, lo:lo + len(bs)] = diffuse(p, seed)[query]
    return out


def cosine(a, b):
    return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-8)


def pairs(xq, local, mu, mass, sizes, cls_counts):
    a = xq[:, None]
    channels = []
    for b in (local, mu):
        channels += [cosine(a, b), np.log1p(np.square(a - b).mean(-1))]
    size = np.broadcast_to(sizes, mass.shape)
    nc = np.broadcast_to(cls_counts, mass.shape)
    channels += [np.stack((np.log1p(100 * mass), np.log1p(size), np.log1p(nc), mass > 0), axis=-1)]
    return np.concatenate(channels, axis=-1).astype(np.float32)


def prepare(inp, arm, shared=None):
    """Raw graph → compact query/group statistics, with measured cost stages."""
    t0 = perf_counter()
    p, h, classes = preproc(inp) if shared is None else shared
    t1 = perf_counter()
    quota = quotas(classes, arm)
    group_y = np.repeat(np.arange(len(classes), dtype=np.int64), quota)
    nc = np.repeat(np.array([len(v) for v in classes]), quota)
    q = inp.query
    if arm in ('full', 'top'):
        ids = np.concatenate(classes)
        t2 = perf_counter()
        mass = singleton_masses(p, q, ids)
        t3 = perf_counter()
        if arm == 'top':
            picks = []
            offset = 0
            for cls, k in zip(classes, quota):
                m = mass[:, offset:offset + len(cls)]
                sim = cosine(h[q, None], h[cls]).mean(-1)
                score = rankdata(m, axis=1) + rankdata(sim, axis=1)
                # ID order, followed by stable descending score, resolves final ties.
                top = np.argsort(-score, axis=1, kind='stable')[:, :k] + offset
                picks.append(top)
                offset += len(cls)
            pick = np.concatenate(picks, axis=1)
            chosen, mass = h[ids[pick]], np.take_along_axis(mass, pick, 1)
        else:
            chosen = h[ids][None]
        z = pairs(h[q], chosen, chosen, mass, np.ones(len(group_y), np.float32), nc)
        sizes = np.ones(len(group_y), dtype=np.float32)
        max_dims, cols = min(BLOCK, len(ids)), len(ids)
    else:
        groups = [g for cc, k in zip(classes, quota) for g in partition(cc, int(k), h)]
        assert len(groups) == int(quota.sum()) and sum(map(len, groups)) == len(inp.context)
        t2 = perf_counter()
        m, local, mu, sizes, max_dims, cols = moments(p, h, q, groups)
        t3 = perf_counter()
        z = pairs(h[q], local, mu[None], m, sizes, nc)
    return (z, group_y, sizes), dict(tokens=len(group_y), counts=quota.tolist(),
                preproc= t1 - t0, grouping=t2 - t1, diffusion=t3 - t2,
                pair_retrieval=perf_counter()-t3, edges=int(p.nnz-len(h)),
                diffusion_total_columns=int(cols), diffusion_max_columns=int(max_dims),
                seed_block_bytes=int(len(h)*max_dims*4), pair_bytes=int(z.nbytes))


class Head(nn.Module):
    """Class-shared multihead query-to-group attention, no class-ID embeddings."""
    def __init__(self):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(16,48), nn.GELU(), nn.Linear(48,48), nn.GELU(), nn.Linear(48,8))
        self.classifier = nn.Sequential(nn.Linear(4,24), nn.GELU(), nn.Linear(24,1))

    def forward(self, pair, group_y, sizes, k):
        s, v = self.pair(pair).chunk(2, -1)
        idx = group_y.view(1,-1,1).expand_as(s)
        s = s + sizes.log()[None,:,None]
        maxes = s.new_full((len(s), k, 4), -torch.inf).scatter_reduce_(-2, idx, s.detach(), reduce='amax')
        w = torch.exp(s - maxes.gather(1, idx))
        den = torch.zeros_like(maxes).scatter_add_(1, idx, w)
        p = torch.zeros_like(maxes).scatter_add_(1, idx, w * v) / den.clamp_min(1e-10)
        return self.classifier(p).squeeze(-1)


def probabilities(inp, shared):
    p,h,classes = shared
    ids=np.concatenate(classes); c=np.repeat(np.arange(len(classes)), [len(v) for v in classes]); k=len(classes)
    x=h.reshape(len(h),-1).astype(np.float64); q=x[inp.query]
    indicator=np.eye(k)[c]
    seed=np.zeros((len(x),k),np.float32); seed[ids,c]=1./np.bincount(c)[c]
    support=diffuse(p,seed)[inp.query].astype(np.float64)
    uniform=np.full_like(support,1/k)
    mass=support.sum(-1,keepdims=True)
    ppr=np.divide(support,mass,out=uniform.copy(),where=mass>0)
    cent=np.stack([x[a].mean(0) for a in classes])
    d2=np.square(q[:,None]-cent).mean(-1)
    def softmax(z):
        e=np.exp(z-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
    pcent=softmax(-d2)
    w=1./np.bincount(c)[c];w=w/w.sum()
    xbar=w @ x[ids]; ybar=w @ indicator
    x0=x[ids]-xbar
    coef=np.linalg.solve(x0.T @ (w[:,None]*x0)+.1*np.eye(x0.shape[1]), x0.T @ (w[:,None]*(indicator-ybar)))
    return dict(ppr=ppr, centroid=pcent, ridge=softmax((q-xbar) @ coef+ybar), uniform=uniform)

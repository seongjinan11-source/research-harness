"""CPU sparse implementation of FP, PCFI and the public FISF algorithm.

Algorithm reference: vendor/fisf.py, Copyright Daeho Um, Apache-2.0.
This implementation uses SciPy SpMM across channels instead of rebuilding one
Torch sparse operator per channel. Observed values are clamped at each step.
FISF deliberately retains the source's all-node anchor sampling behavior.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra


def adjacency(edges, n, self_loops=True):
    src, dst = np.asarray(edges, dtype=np.int64)
    a = sp.coo_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n)).tocsr()
    a = a.maximum(a.T)
    a.data[:] = 1
    if self_loops:
        a.setdiag(1)
    else:
        a.setdiag(0); a.eliminate_zeros()
    a.sort_indices()
    return a


def distances(a, known, structural=False):
    n, d = known.shape
    def one(sources):
        if not len(sources):
            return np.zeros(n, np.float32)  # author's convention for no-source columns
        z = dijkstra(a, directed=False, indices=sources, min_only=True, unweighted=True)
        return np.where(np.isfinite(z), z, 0).astype(np.float32)
    if structural:
        return np.repeat(one(np.flatnonzero(known[:, 0]))[:, None], d, axis=1)
    return np.column_stack([one(np.flatnonzero(known[:, j])) for j in range(d)])


def diffuse(a, x, known, pc, iterations=100):
    out = np.where(known, x, 0).astype(np.float32)
    den = a @ pc
    den = np.maximum(den, np.finfo(np.float32).tiny)
    for _ in range(iterations):
        out = (a @ (out * pc)) / den
        np.copyto(out, x, where=known)
    return out


def impute(a, x_observed, method, mode, iterations=100, alpha=.9, beta=.9, gamma=.7):
    import torch
    t0 = time.perf_counter()
    x = np.asarray(x_observed, np.float32).copy()
    known = np.isfinite(x)
    x[~known] = 0
    original_known = known.copy()
    info = dict(method=method, no_source_columns=int((~known.any(0)).sum()),
                anchor_overwrites=0, injected=0)
    if method == 'most_frequent':
        out = x.copy()
        for j in range(x.shape[1]):
            vals, counts = np.unique(x[known[:, j], j], return_counts=True)
            out[~known[:, j], j] = vals[counts.argmax()] if len(vals) else 0.
    elif method == 'fp':
        deg = np.asarray(a.sum(1)).ravel()
        inv = 1 / np.sqrt(np.maximum(deg, 1))
        norm = sp.diags(inv) @ a @ sp.diags(inv)
        out = x.copy()
        for _ in range(iterations):
            out = norm @ out
            np.copyto(out, x, where=known)
    else:
        dist = distances(a, known, mode == 'structural')
        pc = np.power(np.float32(alpha), dist)
        out = diffuse(a, x, known, pc, iterations)
        info['pre_variance_mean'] = float(out.var(0, ddof=1).mean())
        info['pre_zero_variance_columns'] = int((out.var(0) <= 1e-12).sum())
        if method == 'pcfi':
            # Match torch.corrcoef, including zero-variance columns -> zero correlation.
            h = out - out.mean(0)
            norms = np.sqrt((h * h).sum(0))
            h_unit = h / np.maximum(norms, 1e-30)
            cor = h_unit.T @ h_unit
            np.clip(cor, -1, 1, out=cor); np.fill_diagonal(cor, 0)
            out = out + (1 - pc) * ((pc * h) @ cor)  # public PCFI beta=1.
        elif method == 'fisf':
            # Torch topk/RNG are used to match the author's tie-breaking and anchors.
            low = torch.topk(torch.var(torch.from_numpy(out), dim=0), int(x.shape[1] * gamma), largest=False).indices.numpy()
            rng = np.random.RandomState(0)
            gen = torch.Generator(device='cpu').manual_seed(0)
            ds = np.zeros_like(x)
            for j in low:
                anchor = int(rng.choice(x.shape[0], 1, replace=False)[0])
                info['anchor_overwrites'] += int(known[anchor, j])
                x[anchor, j] = float(torch.rand(1, generator=gen).item())
                known[anchor, j] = True
                z = dijkstra(a, directed=False, indices=anchor, unweighted=True)
                ds[:, j] = np.where(np.isfinite(z), z, 0)
            # Original structural code uses any non-injected channel as the base source set.
            # For uniform masks each channel has its own source set including the new anchor.
            if mode != 'structural':
                dist = distances(a, known, False)
            pc = np.power(np.float32(alpha), dist)
            pc[:, low] *= np.power(np.float32(beta), ds[:, low])
            out = diffuse(a, x, known, pc, iterations)
            info['injected'] = int(len(low))
        else:
            raise ValueError(method)
    assert np.isfinite(out).all(), method
    info.update(seconds=time.perf_counter() - t0,
                variance_mean=float(out.var(0, ddof=1).mean()),
                zero_variance_columns=int((out.var(0) <= 1e-12).sum()),
                original_observed_max_change=float(np.max(np.abs(out[original_known] - np.asarray(x_observed)[original_known]))) if original_known.any() else 0.)
    return out.astype(np.float32), info

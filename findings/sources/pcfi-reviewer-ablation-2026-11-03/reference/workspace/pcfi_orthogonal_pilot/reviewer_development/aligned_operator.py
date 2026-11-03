"""An equivalent implementation of the existing static Node-frame frontend.

This module does not change the locked experiment or introduce learned parameters.
It applies only to T_ij = Q_i.T Q_j with fixed SO(2) node/column frames.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import sklearn  # Must precede torch in this Windows environment.
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import correlation


def node_theta(front, extra):
    u = extra['descriptor']
    affinity = extra['node_affinity'][:, None, None].expand(-1, u.shape[1], 1)
    descriptor = torch.cat((u, extra['neighbor'], affinity), -1)
    return np.pi * torch.tanh(front.phi(descriptor).squeeze(-1))


def prepare_sparse(extra, n):
    """Factor P_ij,f as A_ij c_jf / (A c_f)_i.

    Cached P was rounded to float32. Refactorization can therefore change the
    last floating-point bits even on CPU; index_add uses the exact cached P.
    """
    indices = torch.stack((extra['ii'], extra['jj']))
    values = torch.ones(len(extra['ii']), dtype=extra['pc'].dtype,
                        device=extra['pc'].device)
    adjacency = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    denominator = torch.sparse.mm(adjacency, extra['pc']).clamp_min(1e-30)
    return adjacency, denominator


def aligned_forward(front, xo, observed, extra, *, backend='index_add',
                    steps=100, prepared=None):
    """Rotate boundary values, diffuse two scalar channels, rotate back.

    Observed boundaries are rotated too and clamped at every iteration. The
    original channel correction is applied once after rotating back.
    """
    theta = node_theta(front, extra)
    cosine, sine = theta.cos(), theta.sin()
    boundary0, boundary1 = xo * cosine, xo * sine
    h0, h1 = boundary0, boundary1
    if backend == 'index_add':
        ii, jj, weights = extra['ii'], extra['jj'], extra['weights']
        for _ in range(steps):
            message0 = torch.zeros_like(h0).index_add(0, ii, weights * h0[jj])
            message1 = torch.zeros_like(h1).index_add(0, ii, weights * h1[jj])
            h0 = torch.where(observed, boundary0, message0)
            h1 = torch.where(observed, boundary1, message1)
    elif backend == 'sparse':
        adjacency, denominator = prepared or prepare_sparse(extra, len(xo))
        features = xo.shape[1]
        for _ in range(steps):
            joined = torch.cat((extra['pc'] * h0, extra['pc'] * h1), dim=1)
            messages = torch.sparse.mm(adjacency, joined)
            h0 = torch.where(observed, boundary0,
                             messages[:, :features] / denominator)
            h1 = torch.where(observed, boundary1,
                             messages[:, features:] / denominator)
    else:
        raise ValueError(backend)
    spatial = cosine * h0 + sine * h1
    return torch.where(observed, xo, correlation(spatial, extra['pc']))


def source_kernel_forward(front, xo, observed, extra, *, steps=100):
    """Independent dense source-influence reference for tiny graphs only."""
    theta = node_theta(front, extra)
    ii, jj, weights = extra['ii'], extra['jj'], extra['weights']
    n, features = xo.shape
    columns = []
    for feature in range(features):
        transition = xo.new_zeros((n, n))
        transition.index_put_((ii, jj), weights[:, feature], accumulate=True)
        source = torch.diag(observed[:, feature].to(xo.dtype))
        influence = source
        for _ in range(steps):
            influence = torch.where(observed[:, feature, None], source,
                                    transition @ influence)
        endpoint_cosine = (theta[None, :, feature] - theta[:, None, feature]).cos()
        columns.append((influence * endpoint_cosine) @ xo[:, feature])
    spatial = torch.stack(columns, dim=1)
    return torch.where(observed, xo, correlation(spatial, extra['pc']))

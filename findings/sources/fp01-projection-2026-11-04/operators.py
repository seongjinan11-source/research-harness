"""FP01 feature operators; graph parameters and role weights are external/frozen.

centroid_bp(j, valid, context, classes, observed, log_likelihood, log_weights,
            iterations=24, damping=.5, chunk=128, return_messages=False)
  j [1,H,N,N], valid [1,N,N], context [1,N], observed [N,F].
  log_likelihood [N,F,R] is the literal Normal table, shared by class/role.
  log_weights [R] or [F,R] are normalized discrete centroid prior masses.
  Returns (node [H,N,K], info); info contains centroid [H,K,F,R].
  With return_messages=True it also contains graph/to_node/to_centroid messages
  and final active-class/centroid pair-consistency residuals.

FeatureModel.hyperparameters returns (location [F], tau [F], sigma [F]).
FeatureModel(..., details=True) returns (p, dict) with branches [H,N,K],
weights [H], and, for C arms, location/tau/sigma/grid/log_weights.
The zero-source shortcut instead returns branches=None and sources=0;
hyperparameters() can still be called separately for a C arm if needed.
The literal class-gated finite-grid score uses one Normal per observed cell.
All updates are synchronous, log-damped, with strict opposite-message exclusion.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order.
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _validate_graph(j, valid, context, classes):
    if j.ndim != 4 or j.shape[0] != 1 or j.shape[-2] != j.shape[-1]:
        raise ValueError('j must have shape [1,H,N,N]')
    n = j.shape[-1]
    if valid.shape != (1, n, n) or context.shape != (1, n):
        raise ValueError('invalid graph mask/context shape')
    if int(classes) < 2:
        raise ValueError('classes must be at least 2')


def _node_belief(field, context, classes):
    p = field.softmax(-1)
    known = context[0] >= 0
    one = F.one_hot(context[0].clamp_min(0), int(classes)).to(field.dtype)
    return torch.where(known[None, :, None], one[None], p)


def _graph_update(messages, field, coupling, valid, context, damping, chunk):
    """messages[h, source, target, class], old fields/messages only."""
    n, k = field.shape[1:]
    known = context[0] >= 0
    one = F.one_hot(context[0].clamp_min(0), k).to(field.dtype)
    parts = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        cavity = field[:, start:end, None] - messages[:, :, start:end].transpose(1, 2)
        p = cavity.softmax(-1)
        p = torch.where(known[None, start:end, None, None], one[None, start:end, None], p)
        proposal = torch.log1p((coupling[:, start:end, :, None] * p).clamp_min(-1 + 1e-7))
        proposal = proposal - proposal.mean(-1, keepdim=True)
        proposal = proposal * valid[:, start:end, :, None]
        parts.append((1 - damping) * messages[:, start:end] + damping * proposal)
    return torch.cat(parts, 1)


def potts_conditional_bp(j, valid, context, classes, iterations=24, damping=.5, chunk=128):
    """Old categorical BP update without computing an unused Bethe score."""
    _validate_graph(j, valid, context, classes)
    h, n = j.shape[1:3]
    u = j.new_zeros(h, n, n, int(classes))
    coupling = torch.expm1(j[0])
    chunk = n if chunk is None else max(1, int(chunk))
    for _ in range(int(iterations)):
        u = _graph_update(u, u.sum(1), coupling, valid, context, damping, chunk)
    return _node_belief(u.sum(1), context, classes)


def _logsum_except(logits):
    """Stable O(K) leave-one-state logsumexp; no 1-softmax cancellation."""
    left = torch.logcumsumexp(logits, -1)
    right = torch.logcumsumexp(logits.flip(-1), -1).flip(-1)
    empty = torch.full_like(logits[..., :1], -torch.inf)
    return torch.logaddexp(torch.cat((empty, left[..., :-1]), -1),
                           torch.cat((right[..., 1:], empty), -1))


def _feature_fields(g, to_node, to_centroid, ei, ef, prior, n, f):
    field = g.sum(1).index_add(1, ei, to_node)
    total = prior[None, None].expand(g.shape[0], to_node.shape[-1], f, -1)
    total = total.index_add(2, ef, to_centroid.permute(0, 2, 1, 3))
    return field, total


def _feature_cavities(field, total, to_node, to_centroid, ei, ef, context):
    local = field[:, ei]
    off = _logsum_except(local)
    active = local - to_node
    den = torch.logaddexp(off, active)
    log_rho, log_not_rho = active - den, off - den
    known = context[0, ei] >= 0
    labels = context[0, ei].clamp_min(0)
    matched = F.one_hot(labels, local.shape[-1]).bool()[None]
    zero = torch.zeros_like(log_rho)
    neginf = torch.full_like(log_rho, -torch.inf)
    log_rho = torch.where(known[None, :, None], torch.where(matched, zero, neginf), log_rho)
    log_not_rho = torch.where(known[None, :, None], torch.where(matched, neginf, zero), log_not_rho)
    raw = total[:, :, ef].permute(0, 2, 1, 3) - to_centroid
    log_q = raw.log_softmax(-1)
    return log_rho, log_not_rho, log_q


def centroid_bp(j, valid, context, classes, observed, log_likelihood, log_weights,
                iterations=24, damping=.5, chunk=128, return_messages=False):
    _validate_graph(j, valid, context, classes)
    h, n = j.shape[1:3]
    k, f, r = int(classes), observed.shape[1], log_likelihood.shape[-1]
    if observed.shape[0] != n or log_likelihood.shape != (n, f, r):
        raise ValueError('invalid observed/log_likelihood shape')
    if log_weights.ndim == 1:
        prior = log_weights[None].expand(f, r)
    elif log_weights.shape == (f, r):
        prior = log_weights
    else:
        raise ValueError('log_weights must be [R] or [F,R]')
    # Normalize the supplied discrete prior once; never add a second Normal prior.
    prior = prior.to(j).log_softmax(-1)
    ei, ef = observed.nonzero(as_tuple=True)
    likelihood = log_likelihood[ei, ef].to(j)
    g = j.new_zeros(h, n, n, k)
    to_node = j.new_zeros(h, len(ei), k)
    to_centroid = j.new_zeros(h, len(ei), k, r)
    coupling = torch.expm1(j[0])
    chunk = n if chunk is None else max(1, int(chunk))
    for _ in range(int(iterations)):
        field, total = _feature_fields(g, to_node, to_centroid, ei, ef, prior, n, f)
        # All three proposals use only the previous iteration.
        next_g = _graph_update(g, field, coupling, valid, context, damping, chunk)
        if len(ei):
            lp, lnotp, lq = _feature_cavities(field, total, to_node, to_centroid, ei, ef, context)
            proposal_a = torch.logsumexp(lq + likelihood[None, :, None], -1)
            proposal_v = torch.logaddexp(lnotp[..., None], lp[..., None] + likelihood[None, :, None])
            proposal_v = proposal_v - proposal_v.mean(-1, keepdim=True)
            next_a = (1 - damping) * to_node + damping * proposal_a
            next_v = (1 - damping) * to_centroid + damping * proposal_v
        else:
            next_a, next_v = to_node, to_centroid
        g, to_node, to_centroid = next_g, next_a, next_v
    field, total = _feature_fields(g, to_node, to_centroid, ei, ef, prior, n, f)
    node = _node_belief(field, context, k)
    info = dict(centroid=total.softmax(-1), observed_nodes=ei, observed_columns=ef)
    if return_messages:
        info.update(graph_messages=g, to_node=to_node, to_centroid=to_centroid)
        if len(ei):
            lp, lnotp, lq = _feature_cavities(field, total, to_node, to_centroid, ei, ef, context)
            la = torch.logsumexp(lq + likelihood[None, :, None], -1)
            lz = torch.logaddexp(lnotp, lp + la)
            pair_active = (lp + la - lz).exp()
            pair_centroid = (lq + torch.logaddexp(lnotp[..., None], lp[..., None] + likelihood[None, :, None]) - lz[..., None]).exp()
            info['pair_active_residual'] = (pair_active - node[:, ei]).abs().amax()
            current_centroid = info['centroid'][:, :, ef].permute(0, 2, 1, 3)
            info['pair_centroid_residual'] = (pair_centroid - current_centroid).abs().amax()
        else:
            info['pair_active_residual'] = j.new_zeros(())
            info['pair_centroid_residual'] = j.new_zeros(())
    return node, info


class FeatureModel(nn.Module):
    def __init__(self, arm, seed=11, iterations=24, damping=.5, grid_points=41):
        super().__init__()
        if arm not in ('A', 'B', 'C_global', 'C'):
            raise ValueError(f'unknown arm {arm}')
        self.arm, self.iterations, self.damping = arm, int(iterations), float(damping)
        self.grid_points = int(grid_points)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            if arm in ('A', 'B'):
                self.feature = nn.Sequential(nn.Linear(28, 24), nn.SiLU(), nn.Linear(24, 1))
                nn.init.zeros_(self.feature[-1].weight)
                nn.init.constant_(self.feature[-1].bias, math.log(math.expm1(.8 / .36)))
            elif arm == 'C_global':
                self.raw = nn.Parameter(torch.zeros(3))
            else:
                self.feature = nn.Sequential(nn.Linear(30, 22), nn.SiLU(), nn.Linear(22, 3))
                nn.init.zeros_(self.feature[-1].weight)
                nn.init.zeros_(self.feature[-1].bias)
        nodes, weights = np.polynomial.hermite.hermgauss(self.grid_points)
        self.register_buffer('gh_nodes', torch.tensor(nodes, dtype=torch.float64))
        self.register_buffer('gh_log_weights', torch.tensor(np.log(weights / np.sqrt(np.pi)), dtype=torch.float64))

    @staticmethod
    def _visible(inp):
        obs = inp['observed'].bool()
        return torch.where(obs, inp['x'], torch.zeros_like(inp['x'])), obs

    def hyperparameters(self, inp, graph):
        if self.arm not in ('C', 'C_global'):
            raise ValueError('centroid hyperparameters only exist for C arms')
        x, obs = self._visible(inp)
        n, f = x.shape
        k = int(inp['classes'])
        count = obs.sum(0).to(x.dtype)
        mean = x.sum(0) / count.clamp_min(1)
        variance = (torch.where(obs, x - mean, 0).square().sum(0) / count.clamp_min(1)).clamp_min(0)
        std = variance.sqrt()
        rms = (x.square().sum(0) / count.clamp_min(1)).sqrt()
        if self.arm == 'C_global':
            raw = self.raw[None].expand(f, 3)
        else:
            z = graph['z'].detach().to(x)
            pooled = obs.to(x.dtype).T @ z / count.clamp_min(1)[:, None]
            stats = torch.stack((mean, std, rms, count / n, x.new_full((f,), 1 / k), x.new_full((f,), 1 / n)), -1)
            raw = self.feature(torch.cat((stats, pooled), -1))
        vclass = .36 + .64 / (k - 1)
        tau0 = math.sqrt((.64 / (k - 1)) / vclass)
        sigma0 = math.sqrt(.36 / vclass)
        def bounded(base, delta):
            offset = x.new_tensor((base - .15) / 1.85).clamp(1e-7, 1 - 1e-7).logit()
            return .15 + 1.85 * torch.sigmoid(offset + delta)
        return mean + std * torch.tanh(raw[:, 0]), bounded(tau0, raw[:, 1]), bounded(sigma0, raw[:, 2])

    def forward(self, inp, graph, details=False):
        x, obs = self._visible(inp)
        n, f = x.shape
        k = int(inp['classes'])
        if int(graph['classes']) != k:
            raise ValueError('feature/graph class count mismatch')
        weights = graph['weights'].detach().to(x)
        if not bool(obs.any()):
            # The low-level zero-source algorithm is independently checked.
            connection = sum((p.sum() * 0 for p in self.parameters()), x.new_zeros(()))
            p = graph['p0'].detach().to(x) + connection
            return (p, dict(weights=weights, branches=None, sources=0)) if details else p
        if self.arm in ('A', 'B'):
            count = obs.sum(0).to(x.dtype)
            rms = (x.square().sum(0) / count.clamp_min(1)).sqrt()
            feats = torch.stack((x.abs(), (count / n).expand(n, f), rms.expand(n, f), x.new_full((n, f), 1 / k)), -1)
            gain = F.softplus(self.feature(torch.cat((graph['z'].detach().to(x)[:, None].expand(n, f, 24), feats), -1)).squeeze(-1)) + .001
            feature_j = (gain * k / (k - 1) * (x - .4 * (k - 2) / (k - 1))).clamp(-12, 12) * obs
            j = x.new_zeros(1, graph['j'].shape[1], n + f, n + f)
            j[0, :, :n, :n] = graph['j'][0].detach()
            j[0, :, :n, n:] = feature_j
            j[0, :, n:, :n] = feature_j.T
            valid = torch.zeros((1, n + f, n + f), dtype=torch.bool, device=x.device)
            valid[0, :n, :n] = graph['valid'][0]
            valid[0, :n, n:] = obs
            valid[0, n:, :n] = obs.T
            context = torch.cat((graph['context'], graph['context'].new_full((1, f), -1)), -1)
            branches = potts_conditional_bp(j, valid, context, k, self.iterations, self.damping)[:, :n]
            info = dict(branches=branches, weights=weights, feature_j=feature_j)
        else:
            location, tau, sigma = self.hyperparameters(inp, graph)
            grid = location[:, None] + math.sqrt(2) * tau[:, None] * self.gh_nodes.to(x)[None]
            log_weights = self.gh_log_weights.to(x)
            log_likelihood = -.5 * math.log(2 * math.pi) - sigma.log()[None, :, None] - .5 * ((x[:, :, None] - grid[None]) / sigma[None, :, None]).square()
            branches, _ = centroid_bp(graph['j'].detach(), graph['valid'], graph['context'], k,
                                       obs, log_likelihood, log_weights, self.iterations, self.damping)
            info = dict(branches=branches, weights=weights, location=location, tau=tau, sigma=sigma,
                        grid=grid, log_weights=log_weights)
        p = torch.einsum('h,hnk->nk', weights, branches)
        return (p, info) if details else p

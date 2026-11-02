"""Conditional node/feature factor PFN; finite BP, not a new Bayes theorem."""
import sys
sys.dont_write_bytecode = True
import sklearn
import math
import torch
from torch import nn
from torch.nn import functional as F

ROLES = [[0, 0], [1, 0], [0, 1], [1, 1]]


def belief_propagation(j, valid, context, iterations=24, damping=.5, block_feature_return=None):
    """j[B,H,L,L], symmetric valid[B,L,L], hard class context[B,L]."""
    b, h, n, _ = j.shape
    u = torch.zeros_like(j)
    known = (context >= 0)[:, None, :, None]
    spin = (2 * context - 1)[:, None, :, None].to(j.dtype)
    mask = valid[:, None].to(j.dtype)
    t = j.tanh()
    for _ in range(iterations):
        cavity = (u.sum(-2)[..., :, None] - u.transpose(-1, -2)).tanh()
        cavity = torch.where(known, spin, cavity)
        proposal = torch.atanh((t * cavity).clamp(-1 + 1e-7, 1 - 1e-7)) * mask
        if block_feature_return is not None:
            permit = torch.ones_like(mask)
            permit[..., block_feature_return:, :block_feature_return] = 0
            proposal = proposal * permit
        u = (1 - damping) * u + damping * proposal
    field = u.sum(-2)
    node_log = torch.stack((-field, field), -1)
    allowed = (context[:, None, :, None] < 0) | (context[:, None, :, None] == j.new_tensor([0, 1]))
    node_log = node_log.masked_fill(~allowed, -torch.inf).log_softmax(-1)
    node = node_log.exp()
    hn = -(node * torch.where(node > 0, node_log, torch.zeros_like(node_log))).sum(-1)
    cavity = (field[..., :, None] - u.transpose(-1, -2)).tanh()
    cavity = torch.where(known, spin, cavity)
    pm = torch.stack(((1 - cavity) / 2, (1 + cavity) / 2), -1)
    pm = pm.clamp_min(torch.finfo(j.dtype).tiny).log()
    # Exact clamps prevent a context seed from re-entering through its own edge.
    allowed_edge = allowed[..., None, :].expand(-1, h, -1, n, -1)
    pm = pm.masked_fill(~allowed_edge, -torch.inf)
    ss = j.new_tensor([[1, -1], [-1, 1]])
    pair_log = j[..., None, None] * ss + pm[..., :, None] + pm.transpose(-3, -2)[..., None, :]
    pair_log = pair_log - torch.logsumexp(pair_log.flatten(-2), -1)[..., None, None]
    pair = pair_log.exp()
    hp = -(pair * torch.where(pair > 0, pair_log, torch.zeros_like(pair_log))).sum((-1, -2))
    corr = (pair * ss).sum((-1, -2))
    edge_term = .5 * ((j * corr + hp) * mask).sum((-1, -2))
    degree = valid.sum(-1)[:, None].to(j.dtype)
    logz = edge_term + ((1 - degree) * hn).sum(-1)
    return node, logz


class StructureEncoder(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.stem = nn.Linear(6, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(2)])

    def forward(self, batch):
        a, valid = batch['adj'].mean(1), batch['pair_valid'].amax(1)
        n = a.shape[-1]
        inp = torch.cat((batch['type'][..., None], a.sum(-1)[..., None] / n, valid.sum(-1)[..., None] / n,
                         batch['prior'][:, None].expand(-1, n, -1)), -1)
        z = self.stem(inp)
        matrices = torch.stack((a, valid - a), 1)
        matrices = matrices / matrices.sum(-1, keepdim=True).clamp_min(1)
        for block in self.blocks:
            messages = torch.einsum('brij,bjd->brid', matrices, z).transpose(1, 2).flatten(2)
            z = z + .2 * block(torch.cat((z, messages), -1))
        return z


class FactorPFN(nn.Module):
    def __init__(self, width=24, iterations=24, constant=False):
        super().__init__()
        self.iterations, self.constant = iterations, constant
        self.encoder = StructureEncoder(width)
        self.graph_factor = nn.Sequential(nn.Linear(3 * width + 4, 32), nn.SiLU(), nn.Linear(32, 2))
        self.feature_factor = nn.Sequential(nn.Linear(width + 3, 24), nn.SiLU(), nn.Linear(24, 1))
        nn.init.zeros_(self.graph_factor[-1].weight)
        nn.init.zeros_(self.graph_factor[-1].bias)
        nn.init.zeros_(self.feature_factor[-1].weight)
        nn.init.constant_(self.feature_factor[-1].bias, math.log(math.expm1(1.5)))

    def potentials(self, batch, feature_off=False, feedback_off=False):
        a, valid = batch['adj'], batch['pair_valid']
        b, r, n, _ = a.shape
        x = torch.where(batch['observed'], batch['x'], torch.zeros_like(batch['x']))
        obs, f = batch['observed'].to(x.dtype), x.shape[-1]
        if self.constant:
            d = x.new_full((b, r, n, n), .25)
            eta = x.new_full((b, r, n, n), .72)
            coef = x.new_full((b, n, f), .8 / (.6 ** 2))
        else:
            z = self.encoder(batch)
            zi, zj = z[:, :, None], z[:, None, :]
            shared = torch.cat(((zi + zj).expand(-1, n, n, -1), (zi - zj).abs().expand(-1, n, n, -1), z.mean(1)[:, None, None].expand(-1, n, n, -1)), -1)
            degree = a.sum(-1) / n
            density = (a * valid).sum((-1, -2)) / valid.sum((-1, -2)).clamp_min(1)
            stats = torch.stack((degree[..., :, None].expand(-1, -1, -1, n), degree[..., None, :].expand(-1, -1, n, -1), density[..., None, None].expand(-1, -1, n, n), torch.ones_like(a)), -1)
            raw = self.graph_factor(torch.cat((shared[:, None].expand(-1, r, -1, -1, -1), stats), -1))
            raw = (raw + raw.transpose(2, 3)) / 2
            dens = density.clamp(.02, .48)
            d = (.49 * torch.sigmoid(torch.logit((dens / .49).clamp(.01, .99))[..., None, None] + .5 * raw[..., 0])).clamp_min(.005)
            eta = .98 * torch.sigmoid(x.new_tensor(math.log(.6 / (.98 - .6))) + raw[..., 1])
            count = obs.sum(1).clamp_min(1)
            rms = ((x.square() * obs).sum(1) / count).sqrt()
            feat = torch.stack((x.abs(), (count / n)[:, None].expand(-1, n, -1), rms[:, None].expand(-1, n, -1)), -1)
            coef = F.softplus(self.feature_factor(torch.cat((z[:, :, None].expand(-1, -1, f, -1), feat), -1)).squeeze(-1)) + .05
        roles = x.new_tensor(ROLES)[None, :, :, None, None]
        same = (d[:, None] * (1 + eta[:, None] * roles)).clamp(1e-5, 1 - 1e-5)
        diff = (d[:, None] * (1 - eta[:, None] * roles)).clamp(1e-5, 1 - 1e-5)
        ls = a[:, None] * same.log() + (1 - a[:, None]) * torch.log1p(-same)
        ld = a[:, None] * diff.log() + (1 - a[:, None]) * torch.log1p(-diff)
        graph_j = (.5 * (ls - ld) * valid[:, None]).sum(2)
        constants = (.25 * (ls + ld) * valid[:, None]).sum((-1, -2, -3))
        feature_j = (x * coef * obs).clamp(-8, 8)
        if feature_off:
            feature_j = feature_j * 0
        j = x.new_zeros((b, 4, n + f, n + f))
        j[:, :, :n, :n] = graph_j
        j[:, :, :n, n:] = feature_j[:, None]
        j[:, :, n:, :n] = feature_j.transpose(-1, -2)[:, None]
        edges = torch.zeros((b, n + f, n + f), device=x.device, dtype=torch.bool)
        edges[:, :n, :n] = valid.amax(1).bool()
        edges[:, :n, n:] = batch['observed']
        edges[:, n:, :n] = batch['observed'].transpose(-1, -2)
        context = torch.cat((batch['context'], batch['context'].new_full((b, f), -1)), -1)
        return j, edges, context, constants

    def forward(self, batch, details=False, feature_off=False, feedback_off=False):
        j, valid, context, constant = self.potentials(batch, feature_off, feedback_off)
        node, logz = belief_propagation(j, valid, context, self.iterations,
                                      block_feature_return=batch['x'].shape[1] if feedback_off else None)
        priors = j.new_tensor([[.25, .25, .25, .25], [0, .5, .5, 0], [.5, 0, 0, .5]])
        mass = batch['prior'] @ priors
        role_logits = (logz + constant - math.log(2) * j.shape[-1]) + mass.log()
        weights = role_logits.softmax(-1)
        prob = torch.einsum('bh,bhnc->bnc', weights, node)[:, :batch['x'].shape[1]]
        return (prob, weights, logz) if details else prob

"""Only observed-cell functions change; graph potentials and finite BP are shared."""
import sys
sys.dont_write_bytecode = True
import sklearn
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import WORK, OLD, C, OLD_MODEL, cfg, sha


class FrozenGraph(nn.Module):
    def __init__(self):
        super().__init__()
        source = OLD_MODEL.FactorPFN()
        checkpoint = torch.load(WORK / cfg()['graph_checkpoint'], map_location='cpu', weights_only=False)['state_dict']
        source.load_state_dict(checkpoint)
        self.encoder, self.graph_factor = source.encoder, source.graph_factor
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, batch):
        a, valid = batch['adj'], batch['pair_valid']
        b, r, n, _ = a.shape
        z = self.encoder(batch)
        zi, zj = z[:, :, None], z[:, None, :]
        shared = torch.cat(((zi + zj).expand(-1, n, n, -1), (zi - zj).abs().expand(-1, n, n, -1),
                            z.mean(1)[:, None, None].expand(-1, n, n, -1)), -1)
        degree = a.sum(-1) / n
        density = (a * valid).sum((-1, -2)) / valid.sum((-1, -2)).clamp_min(1)
        stats = torch.stack((degree[..., :, None].expand(-1, -1, -1, n), degree[..., None, :].expand(-1, -1, n, -1),
                             density[..., None, None].expand(-1, -1, n, n), torch.ones_like(a)), -1)
        raw = self.graph_factor(torch.cat((shared[:, None].expand(-1, r, -1, -1, -1), stats), -1))
        raw = (raw + raw.transpose(2, 3)) / 2
        dens = density.clamp(.02, .48)
        d = (.49 * torch.sigmoid(torch.logit((dens / .49).clamp(.01, .99))[..., None, None] + .5 * raw[..., 0])).clamp_min(.005)
        eta = .98 * torch.sigmoid(a.new_tensor(math.log(.6 / (.98 - .6))) + raw[..., 1])
        role = a.new_tensor(OLD_MODEL.ROLES)[None, :, :, None, None]
        same = (d[:, None] * (1 + eta[:, None] * role)).clamp(1e-5, 1 - 1e-5)
        diff = (d[:, None] * (1 - eta[:, None] * role)).clamp(1e-5, 1 - 1e-5)
        ls = a[:, None] * same.log() + (1 - a[:, None]) * torch.log1p(-same)
        ld = a[:, None] * diff.log() + (1 - a[:, None]) * torch.log1p(-diff)
        j = (.5 * (ls - ld) * valid[:, None]).sum(2)
        constant = (.25 * (ls + ld) * valid[:, None]).sum((-1, -2, -3))
        return j, constant, z


def value_basis(x):
    return torch.stack((x.clamp(-8, 8), x.square().clamp(max=64), x.abs().clamp(max=8),
                        x.asinh().clamp(-8, 8), torch.log1p(x.square()).clamp(max=8), torch.ones_like(x)), -1)


def column_stats(x, observed):
    x = torch.where(observed, x, torch.zeros_like(x))
    m = observed.to(x.dtype)
    count = m.sum(1).clamp_min(1)
    ax = x.abs()
    return torch.stack((m.mean(1), x.clamp(-8, 8).sum(1) / count, ax.clamp(max=8).sum(1) / count,
                        torch.log1p((x.square() * m).sum(1) / count).clamp(max=8), torch.log1p(ax.amax(1)).clamp(max=8),
                        ((ax < .5) * m).sum(1) / count, ((ax > 1) * m).sum(1) / count,
                        ((ax > 2) * m).sum(1) / count), -1)


class Cell(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == 'original':
            self.net = nn.Sequential(nn.Linear(27, 24), nn.SiLU(), nn.Linear(24, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, math.log(math.expm1(1.5)))
        else:
            width, size = (35, 6) if arm == 'static' else (32, 14)
            self.net = nn.Sequential(nn.Linear(size, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.linear = nn.Parameter(torch.tensor([1.55, 0., 0., 0., 0., 0.]))

    def forward(self, batch, embedding):
        obs = batch['observed']
        x = torch.where(obs, batch['x'], torch.zeros_like(batch['x']))
        b, n, f = x.shape
        if self.arm == 'original':
            count = obs.to(x.dtype).sum(1).clamp_min(1)
            rms = (x.square().sum(1) / count).sqrt()
            feature = torch.stack((x.abs(), (count / n)[:, None].expand(-1, n, -1), rms[:, None].expand(-1, n, -1)), -1)
            coef = F.softplus(self.net(torch.cat((embedding[:, :, None].expand(-1, -1, f, -1), feature), -1)).squeeze(-1)) + .05
            rho = x * coef
        else:
            basis = value_basis(x)
            values = basis if self.arm == 'static' else torch.cat((basis, column_stats(x, obs)[:, None].expand(-1, n, -1, -1)), -1)
            rho = self.net(values).squeeze(-1) + (basis * self.linear).sum(-1)
        return rho.clamp(-8, 8) * obs


class CellFactorPFN(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.graph = FrozenGraph()
        torch.manual_seed(seed + 10000)
        self.cell = Cell(arm)
        self.arm = arm

    def forward(self, batch, details=False):
        with torch.no_grad():
            gj, constant, embedding = self.graph(batch)
        b, h, n, _ = gj.shape
        f = batch['x'].shape[-1]
        rho = self.cell(batch, embedding)
        j = gj.new_zeros((b, h, n + f, n + f))
        j[:, :, :n, :n] = gj
        j[:, :, :n, n:] = rho[:, None]
        j[:, :, n:, :n] = rho.transpose(-1, -2)[:, None]
        edges = torch.zeros((b, n + f, n + f), device=j.device, dtype=torch.bool)
        edges[:, :n, :n] = batch['pair_valid'].amax(1).bool()
        edges[:, :n, n:] = batch['observed']
        edges[:, n:, :n] = batch['observed'].transpose(-1, -2)
        context = torch.cat((batch['context'], batch['context'].new_full((b, f), -1)), -1)
        node, logz = OLD_MODEL.belief_propagation(j, edges, context, 24, .5)
        priors = j.new_tensor([[.25, .25, .25, .25], [0, .5, .5, 0], [.5, 0, 0, .5]])
        mass = batch['prior'] @ priors
        weights = (logz + constant + mass.log()).softmax(-1)
        p = torch.einsum('bh,bhnc->bnc', weights, node)[:, :n]
        return (p, weights, rho) if details else p


def make_model(arm, seed):
    if arm in ['original', 'static', 'adaptive']:
        return CellFactorPFN(arm, seed).cuda()
    return C.Model('pcfi_plain' if arm == 'pcfi_pfn' else 'mask_plain', seed)

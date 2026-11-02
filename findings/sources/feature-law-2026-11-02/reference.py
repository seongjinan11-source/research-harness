"""Known-law controls; exact latent sums and same-iteration fixed-factor BP."""
import sys
sys.dont_write_bytecode = True
import sklearn
import itertools
import math
import torch
from common import NATIVE, LAWS, LEVELS, OLD_MODEL


def logpdf(x, law, a, b, u):
    if law == 'variance':
        sd = torch.where(u > 0, b, a)
        return -.5 * (x / sd).square() - sd.log() - .5 * math.log(2 * math.pi)
    residual = x - a * u
    if law == 'gaussian':
        return -.5 * (residual / b).square() - b.log() - .5 * math.log(2 * math.pi)
    scale = b / math.sqrt(3)
    return math.lgamma(2) - math.lgamma(1.5) - .5 * math.log(3 * math.pi) - scale.log() - 2 * torch.log1p(residual.square() / (3 * scale.square()))


class Reference:
    def __init__(self, device='cuda', n=12, nt=8, nh=4):
        self.device, self.n, self.nt, self.nh = device, n, nt, nh
        ids = torch.arange(2 ** n, device=device)
        self.spin = (2 * ((ids[:, None] >> torch.arange(n, device=device)) & 1) - 1).double()
        self.label = (self.spin + 1) / 2
        self.pair = (self.spin[:, :, None] * self.spin[:, None, :]).flatten(1)
        self.candidates = [(law, a, b) for law in LAWS for a in LEVELS[law][0] for b in LEVELS[law][1]]

    def feature_factors(self, batch):
        x = torch.where(batch['observed'], batch['x'], torch.zeros_like(batch['x'])).double()
        m = batch['observed'].double()
        k, constants = [], []
        for law, aa, bb in self.candidates:
            a, b = x.new_tensor(aa), x.new_tensor(bb)
            positive = logpdf(x, law, a, b, x.new_tensor(1))
            negative = logpdf(x, law, a, b, x.new_tensor(-1))
            k.append(.5 * (positive - negative) * m)
            constants.append((.5 * (positive + negative) * m).sum(1))
        return torch.stack(k, 1), torch.stack(constants, 1)

    def feature_score(self, batch):
        k, constants = self.feature_factors(batch)
        scores = torch.einsum('banf,sn->bafs', k, self.spin)
        orientation = torch.stack((scores, -scores), 2) + constants[:, :, None, :, None]
        return torch.logsumexp(orientation.flatten(1, 2) - math.log(2 * len(self.candidates)), 1).sum(1)

    def prior(self, batch):
        return batch['prior'].double() @ self.spin.new_tensor([[.25, .25, .25, .25], [0, .5, .5, 0], [.5, 0, 0, .5]])

    def marginal(self, score, batch):
        allowed = ((batch['context'][:, None] < 0) | (batch['context'][:, None] == self.label[None])).all(-1)
        score = score.masked_fill(~allowed[:, None], -torch.inf)
        assignment = score.flatten(1).softmax(-1).reshape_as(score).sum(1)
        p = assignment @ self.label
        return torch.stack((1 - p, p), -1)

    def point_exact(self, batch, graph):
        j, constant, _ = graph
        score = .5 * j.double().flatten(2) @ self.pair.T
        score = score + constant.double()[..., None] + self.prior(batch).log()[..., None]
        return self.marginal(score + self.feature_score(batch)[:, None], batch)

    def native_exact(self, batch):
        e = batch['incidence'].double().flatten(2)
        ne = 1 - e
        xi = e.new_tensor(NATIVE.PARAMETERS)
        d, eta = xi.unbind(-1)
        plus, minus = d * (1 + eta), d * (1 - eta)
        ce = .5 * (plus.log() + minus.log()) - d.log()
        cn = .5 * (torch.log1p(-plus) + torch.log1p(-minus)) - torch.log1p(-d)
        je, jn = .5 * (plus.log() - minus.log()), .5 * (torch.log1p(-plus) - torch.log1p(-minus))
        roles = e.new_tensor(NATIVE.ROLES)
        products = (self.spin[:, :self.nt, None] * self.spin[:, None, self.nt:]).flatten(1)
        ea, na = e.sum(-1) @ roles.T, ne.sum(-1) @ roles.T
        es = torch.einsum('bre,se,hr->bhs', e, products, roles)
        ns = torch.einsum('bre,se,hr->bhs', ne, products, roles)
        constant = e.sum((1, 2))[:, None] * d.log() + ne.sum((1, 2))[:, None] * torch.log1p(-d)
        score = constant[..., None, None] + ea[:, None, :, None] * ce[None, :, None, None] + na[:, None, :, None] * cn[None, :, None, None]
        score = score + es[:, None] * je[None, :, None, None] + ns[:, None] * jn[None, :, None, None]
        score = score + e.new_tensor(NATIVE.MASS).log()[None, :, None, None] + self.prior(batch).log()[:, None, :, None]
        score = score + self.feature_score(batch)[:, None, None]
        return self.marginal(score.flatten(1, 2), batch)

    def fixed_bp(self, batch, graph, iterations=24):
        gj, gc, _ = graph
        gj, gc = gj.double(), gc.double()
        k, constants = self.feature_factors(batch)
        b, _, n, f = k.shape
        choices = list(itertools.product(range(len(self.candidates)), repeat=f))
        ids = torch.tensor(choices, device=self.device)
        rho = torch.stack([k[:, ids[:, col], :, col] for col in range(f)], -1)
        fc = sum(constants[:, ids[:, col], col] for col in range(f))
        branches = len(choices)
        j = gj.new_zeros(b, branches, 4, n + f, n + f)
        j[:, :, :, :n, :n] = gj[:, None]
        j[:, :, :, :n, n:] = rho[:, :, None]
        j[:, :, :, n:, :n] = rho.transpose(-1, -2)[:, :, None]
        edges = torch.zeros((b, n + f, n + f), device=self.device, dtype=torch.bool)
        edges[:, :n, :n] = batch['pair_valid'].amax(1).bool()
        edges[:, :n, n:] = batch['observed']
        edges[:, n:, :n] = batch['observed'].transpose(-1, -2)
        context = torch.cat((batch['context'], batch['context'].new_full((b, f), -1)), -1)
        node, z = OLD_MODEL.belief_propagation(j.flatten(1, 2), edges, context, iterations, .5)
        mass = self.prior(batch)
        weights = (z + (gc[:, None] + fc[:, :, None] + mass.log()[:, None]).flatten(1)).softmax(-1)
        return torch.einsum('bh,bhnc->bnc', weights, node)[:, :n]

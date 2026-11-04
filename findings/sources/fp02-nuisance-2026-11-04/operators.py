"""FP02 sparse Poisson/centroid inference (PLAN_v2).

No fitting, file I/O, historical-source modification, or graph role selection.
The model is an explicit finite centroid grid and a Poisson working score.
Nonedges use a node-product mean-field approximation; observed edges use BP.

Public API:
  nuisance_bp(src, dst, nodes, context, classes, observed, log_likelihood,
              log_weights, variant='dc_update', iterations=24, damping=.5,
              block=6, details=False) -> (prob[N,K], info)
  gamma_statistics(src, dst, node_belief, edge_belief, theta_mean,
                   affinity_mean, context) -> dict

The second API takes each UNDIRECTED edge once and edge_belief[U,K,K],
with its first/second label axes corresponding to src/dst. Symmetric KxK
count/exposure storage contains the complete unordered off-diagonal H_ab in
both cells; sum its upper triangle, not its full matrix, for a total.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order, before torch.
import torch
from torch.nn import functional as F


def _logsum_except(logits):
    left = torch.logcumsumexp(logits, -1)
    right = torch.logcumsumexp(logits.flip(-1), -1).flip(-1)
    empty = torch.full_like(logits[..., :1], -torch.inf)
    return torch.logaddexp(torch.cat((empty, left[..., :-1]), -1),
                           torch.cat((right[..., 1:], empty), -1))


def _node_belief(field, context, classes):
    p = field.softmax(-1)
    known = context >= 0
    one = F.one_hot(context.clamp_min(0), classes).to(field.dtype)
    return torch.where(known[:, None], one, p)


def _clamped_log_cavity(field, context, classes):
    logp = field.log_softmax(-1)
    known = context >= 0
    one = F.one_hot(context.clamp_min(0), classes).bool()
    clamped = torch.where(one, torch.zeros_like(logp),
                          torch.full_like(logp, -torch.inf))
    return torch.where(known[:, None], clamped, logp)


def _nonneighbor_volume(src, dst, node_belief, theta_mean):
    """Directed symmetric edges; returns sum_{j notin {i,N(i)}} theta_j b_j.

    Only a nonnegative cancellation-roundoff correction is applied. The raw
    minimum is returned for audit; no Poisson intensity is clipped.
    """
    weighted = theta_mean[:, None] * node_belief
    neighbors = torch.zeros_like(weighted).index_add(0, src, weighted[dst])
    raw = weighted.sum(0)[None] - weighted - neighbors
    return raw.clamp_min(0), raw.amin()


def nonneighbor_field(src, dst, node_belief, theta_mean, affinity_mean,
                      density_offset):
    """Public small-check helper; edges are symmetric DIRECTED pairs.

    Returns (positive penalty[N,K], nonneighbor_volume[N,K], raw_min).
    Subtract this penalty from the population unary exactly once.
    """
    volume, raw_min = _nonneighbor_volume(src, dst, node_belief, theta_mean)
    penalty = density_offset * theta_mean[:, None] * (volume @ affinity_mean.T)
    return penalty, volume, raw_min


def _unordered(matrix):
    return matrix + matrix.T - torch.diag_embed(matrix.diagonal())


def gamma_statistics(src, dst, node_belief, edge_belief, theta_mean,
                     affinity_mean, context):
    """Fresh expectations from one node/edge-belief snapshot.

    src/dst contain each undirected edge once (either orientation is valid).
    theta_interaction[i] is sum_{j!=i} Etheta_j E_qij[B_zi,zj], without t.
    All context nodes enter exposures; only unknown nodes enter population.
    Derivatives are retained for possible future unrolled training.
    """
    n, k = node_belief.shape
    if edge_belief.shape != (src.numel(), k, k):
        raise ValueError('edge_belief must be [undirected edges,K,K]')
    if theta_mean.shape != (n,) or context.shape != (n,):
        raise ValueError('invalid theta/context shape')
    all_src = torch.cat((src, dst))
    all_dst = torch.cat((dst, src))
    volume, raw_min = _nonneighbor_volume(all_src, all_dst, node_belief, theta_mean)
    weighted = theta_mean[:, None] * node_belief
    # Ordered nonneighbor expectation is symmetric. Average its two numeric
    # orientations before converting diagonal ordered pairs to unordered pairs.
    nonedge = weighted.T @ volume
    nonedge = .5 * (nonedge + nonedge.T)
    nonedge = nonedge - .5 * torch.diag_embed(nonedge.diagonal())
    edge_count = _unordered(edge_belief.sum(0))
    edge_weight = theta_mean[src] * theta_mean[dst]
    edge_exposure = _unordered(torch.einsum('u,uab->ab', edge_weight, edge_belief))
    pair_exposure = nonedge + edge_exposure
    edge_affinity = (edge_belief * affinity_mean[None]).sum((-1, -2))
    interaction = (node_belief * (volume @ affinity_mean.T)).sum(-1)
    interaction = interaction.index_add(0, src, theta_mean[dst] * edge_affinity)
    interaction = interaction.index_add(0, dst, theta_mean[src] * edge_affinity)
    return dict(edge_count=edge_count, pair_exposure=pair_exposure,
                nonedge_exposure=nonedge, edge_exposure=edge_exposure,
                population_count=node_belief[context < 0].sum(0),
                theta_interaction=interaction, class_volume=weighted.sum(0),
                nonneighbor_volume_raw_min=raw_min)


def _topology(src, dst, n, device):
    src = torch.as_tensor(src, dtype=torch.long, device=device)
    dst = torch.as_tensor(dst, dtype=torch.long, device=device)
    if src.ndim != 1 or dst.shape != src.shape:
        raise ValueError('src/dst must be equal-length directed edge vectors')
    if src.numel() and bool(((src < 0) | (dst < 0) | (src >= n) |
                             (dst >= n) | (src == dst)).any()):
        raise ValueError('invalid endpoint or self edge')
    keys = src * n + dst
    ordered, order = keys.sort()
    if ordered.numel() > 1 and bool((ordered[1:] == ordered[:-1]).any()):
        raise ValueError('duplicate edges are not binary adjacency')
    wanted = dst * n + src
    pos = torch.searchsorted(ordered, wanted)
    if pos.numel() and (bool((pos >= ordered.numel()).any()) or
                        not bool((ordered[pos] == wanted).all())):
        raise ValueError('src/dst must contain both directions of every edge')
    reverse = order[pos]
    undirected = (src < dst).nonzero(as_tuple=True)[0]
    degree = torch.bincount(src, minlength=n)
    return src, dst, reverse, undirected, degree


def _feature_totals(graph_messages, dst, to_node, to_centroid, ei, ef,
                    prior, unary, penalty, n, classes):
    field = unary[None] - penalty
    field = field.index_add(0, dst, graph_messages).index_add(0, ei, to_node)
    total = prior[None].expand(classes, -1, -1)
    total = total.index_add(1, ef, to_centroid.permute(1, 0, 2))
    return field, total


def _feature_cavities(field, total, to_node, to_centroid, ei, ef, context):
    local = field[ei]
    off = _logsum_except(local)
    active = local - to_node
    denominator = torch.logaddexp(off, active)
    logp, lognotp = active - denominator, off - denominator
    known = context[ei] >= 0
    one = F.one_hot(context[ei].clamp_min(0), field.shape[-1]).bool()
    zero, neginf = torch.zeros_like(logp), torch.full_like(logp, -torch.inf)
    logp = torch.where(known[:, None], torch.where(one, zero, neginf), logp)
    lognotp = torch.where(known[:, None], torch.where(one, neginf, zero), lognotp)
    logq = (total[:, ef].permute(1, 0, 2) - to_centroid).log_softmax(-1)
    return logp, lognotp, logq


def _graph_proposal(src, dst, reverse, messages, field, context, theta,
                    meanlog, mean, density, damping, chunk=4096):
    k = field.shape[-1]
    pieces = []
    for start in range(0, src.numel(), chunk):
        stop = min(start + chunk, src.numel())
        si, tj = src[start:stop], dst[start:stop]
        cavity = field[si] - messages[reverse[start:stop]]
        cavity = _clamped_log_cavity(cavity, context[si], k)
        scale = density * theta[si] * theta[tj]
        potential = meanlog[None] - scale[:, None, None] * mean[None]
        proposal = torch.logsumexp(cavity[:, :, None] + potential, -2)
        # Finite zero-mean log gauge; geometric/log damping, not probability damping.
        proposal = proposal - proposal.mean(-1, keepdim=True)
        pieces.append((1 - damping) * messages[start:stop] + damping * proposal)
    return torch.cat(pieces, 0) if pieces else messages


def _edge_beliefs(src, dst, reverse, undirected, messages, field, context,
                   theta, meanlog, mean, density, chunk=4096):
    k = field.shape[-1]
    pieces = []
    for start in range(0, undirected.numel(), chunk):
        ids = undirected[start:start + chunk]
        si, tj = src[ids], dst[ids]
        left = _clamped_log_cavity(field[si] - messages[reverse[ids]], context[si], k)
        right = _clamped_log_cavity(field[tj] - messages[ids], context[tj], k)
        scale = density * theta[si] * theta[tj]
        logq = (left[:, :, None] + right[:, None, :] + meanlog[None]
                - scale[:, None, None] * mean[None])
        pieces.append(logq.flatten(1).softmax(-1).reshape(-1, k, k))
    return torch.cat(pieces, 0) if pieces else field.new_zeros((0, k, k))


def _moments(shape, rate, alpha):
    return (shape / rate, torch.digamma(shape) - rate.log(),
            torch.digamma(alpha) - torch.digamma(alpha.sum()))


def _nuisance_update(variant, shape, rate, theta_shape, theta_rate, alpha,
                     prior_shape, prior_rate, degree, density, stats, damping):
    proposal_shape = prior_shape + stats['edge_count']
    proposal_rate = prior_rate + density * stats['pair_exposure']
    proposal_alpha = torch.ones_like(alpha) + stats['population_count']
    proposal_theta_shape = torch.ones_like(theta_shape) + degree
    proposal_theta_rate = torch.ones_like(theta_rate) + density * stats['theta_interaction']
    before = dict(affinity_shape=shape, affinity_rate=rate, theta_shape=theta_shape,
                  theta_rate=theta_rate, population_alpha=alpha)
    shape = (1 - damping) * shape + damping * proposal_shape
    rate = (1 - damping) * rate + damping * proposal_rate
    alpha = (1 - damping) * alpha + damping * proposal_alpha
    if variant == 'dc_update':
        theta_shape = (1 - damping) * theta_shape + damping * proposal_theta_shape
        theta_rate = (1 - damping) * theta_rate + damping * proposal_theta_rate
    after = dict(affinity_shape=shape, affinity_rate=rate, theta_shape=theta_shape,
                 theta_rate=theta_rate, population_alpha=alpha)
    proposal = dict(affinity_shape=proposal_shape, affinity_rate=proposal_rate,
                    theta_shape=proposal_theta_shape, theta_rate=proposal_theta_rate,
                    population_alpha=proposal_alpha)
    return shape, rate, theta_shape, theta_rate, alpha, dict(
        before=before, statistics=stats, proposal=proposal, after=after)


def _rate_diagnostics(theta, mean, density, usrc, udst, edge_belief):
    # This is the exact maximum over node pairs AND class assignments, and an
    # upper bound on a posterior-averaged pair intensity. No rate clipping.
    with torch.no_grad():
        if theta.numel() > 1:
            maximum = density * theta.topk(2).values.prod() * mean.amax()
        else:
            maximum = theta.new_zeros(())
        edge_rate = (density * theta[usrc] * theta[udst]
                     * (edge_belief * mean[None]).sum((-1, -2)))
        zero = theta.new_zeros(())
        return dict(max_class_conditional_rate=maximum,
                    max_observed_edge_mean_rate=edge_rate.amax() if edge_rate.numel() else zero,
                    mean_observed_edge_rate=edge_rate.mean() if edge_rate.numel() else zero,
                    observed_edge_fraction_rate_gt_1=(edge_rate > 1).to(theta.dtype).mean()
                    if edge_rate.numel() else zero,
                    saturation_probability_at_max_rate=-torch.expm1(-maximum),
                    theta_max=theta.amax(), affinity_max=mean.amax())


def nuisance_bp(src, dst, nodes, context, classes, observed, log_likelihood,
                log_weights, variant='dc_update', iterations=24, damping=.5,
                block=6, details=False):
    """Sparse edge BP + nonneighbor MF + explicit finite-grid centroid BP.

    Damping is the new-proposal fraction for both log messages and nuisance
    distribution parameters. All nuisance proposals use the same old state.
    At each step the field is based on previous node beliefs; the saved final
    field is recombined with the new messages, without an extra MF update.
    Its node/edge beliefs therefore share exactly the same field/cavity state.
    """
    if variant not in ('poisson_fixed', 'sbm_update', 'dc_update'):
        raise ValueError('unknown nuisance variant')
    n, k = int(nodes), int(classes)
    steps, block = int(iterations), int(block)
    if n < 1 or k < 2 or steps < 0 or block < 1 or not 0 < float(damping) <= 1:
        raise ValueError('invalid nodes/classes/iterations/block/damping')
    if log_likelihood.ndim != 3 or log_likelihood.shape[0] != n:
        raise ValueError('log_likelihood must be [N,F,R]')
    if not log_likelihood.is_floating_point():
        raise ValueError('log_likelihood must be floating point')
    device, dtype = log_likelihood.device, log_likelihood.dtype
    f, r = log_likelihood.shape[1:]
    if r < 1:
        raise ValueError('finite centroid grid cannot be empty')
    context = torch.as_tensor(context, device=device, dtype=torch.long)
    observed = torch.as_tensor(observed, device=device, dtype=torch.bool)
    if context.shape != (n,) or observed.shape != (n, f):
        raise ValueError('invalid context/observed shape')
    if bool(((context < -1) | (context >= k)).any()):
        raise ValueError('context must be -1 or a valid class')
    src, dst, reverse, undirected, degree_int = _topology(src, dst, n, device)
    degree = degree_int.to(dtype=dtype)
    usrc, udst = src[undirected], dst[undirected]
    density = degree.mean() / max(n - 1, 1)
    prior = torch.as_tensor(log_weights, device=device, dtype=dtype)
    if prior.ndim == 1 and prior.shape == (r,):
        prior = prior[None].expand(f, r)
    elif prior.shape != (f, r):
        raise ValueError('log_weights must be [R] or [F,R]')
    prior = prior.log_softmax(-1)
    ei, ef = observed.nonzero(as_tuple=True)
    likelihood = log_likelihood[ei, ef]
    if not bool(torch.isfinite(likelihood).all()) or not bool(torch.isfinite(prior).all()):
        raise ValueError('visible likelihood and normalized grid prior must be finite')
    b0 = log_likelihood.new_full((k, k), .4)
    b0 = b0 + torch.eye(k, device=device, dtype=dtype) * (.6 * k)
    prior_shape = torch.full_like(b0, 5.)
    prior_rate = prior_shape / b0
    shape, rate = prior_shape, prior_rate
    theta_shape, theta_rate = degree.new_ones(n), degree.new_ones(n)
    alpha = degree.new_ones(k)
    mean, meanlog, unary = _moments(shape, rate, alpha)
    theta = theta_shape / theta_rate
    messages = log_likelihood.new_zeros(src.numel(), k)
    to_node = log_likelihood.new_zeros(ei.numel(), k)
    to_centroid = log_likelihood.new_zeros(ei.numel(), k, r)
    node = _node_belief(unary[None].expand(n, k), context, k)
    field_node = node
    penalty, volume, raw_min = nonneighbor_field(src, dst, node, theta, mean, density)
    field, total = _feature_totals(messages, dst, to_node, to_centroid,
                                   ei, ef, prior, unary, penalty, n, k)
    if steps == 0:
        node = _node_belief(field, context, k)
    history = []
    for step in range(steps):
        if step > 0 and step % block == 0 and variant != 'poisson_fixed':
            edge = _edge_beliefs(src, dst, reverse, undirected, messages, field,
                                 context, theta, meanlog, mean, density)
            stats = gamma_statistics(usrc, udst, node, edge, theta, mean, context)
            shape, rate, theta_shape, theta_rate, alpha, entry = _nuisance_update(
                variant, shape, rate, theta_shape, theta_rate, alpha,
                prior_shape, prior_rate, degree, density, stats, damping)
            entry['after_steps'] = step
            history.append(entry)
            mean, meanlog, unary = _moments(shape, rate, alpha)
            theta = theta_shape / theta_rate
        # External-field and all three message proposals use OLD node/messages.
        field_node = node
        penalty, volume, raw_min = nonneighbor_field(src, dst, node, theta, mean, density)
        field, total = _feature_totals(messages, dst, to_node, to_centroid,
                                       ei, ef, prior, unary, penalty, n, k)
        next_messages = _graph_proposal(src, dst, reverse, messages, field,
                                        context, theta, meanlog, mean, density, damping)
        if ei.numel():
            lp, lnotp, lq = _feature_cavities(field, total, to_node, to_centroid,
                                             ei, ef, context)
            proposal_a = torch.logsumexp(lq + likelihood[:, None], -1)
            proposal_v = torch.logaddexp(lnotp[..., None], lp[..., None] + likelihood[:, None])
            proposal_v = proposal_v - proposal_v.mean(-1, keepdim=True)
            next_a = (1 - damping) * to_node + damping * proposal_a
            next_v = (1 - damping) * to_centroid + damping * proposal_v
        else:
            next_a, next_v = to_node, to_centroid
        messages, to_node, to_centroid = next_messages, next_a, next_v
        # Preserve the just-used MF penalty; do not run a hidden extra MF step.
        field, total = _feature_totals(messages, dst, to_node, to_centroid,
                                       ei, ef, prior, unary, penalty, n, k)
        node = _node_belief(field, context, k)
    edge = _edge_beliefs(src, dst, reverse, undirected, messages, field,
                         context, theta, meanlog, mean, density)
    # An empty source set has an exactly zero feature derivative, including
    # when a later training caller asks autograd for that zero derivative.
    # Select visible entries before this operation; hidden NaNs are untouched.
    node = node + likelihood.sum() * 0
    stats = gamma_statistics(usrc, udst, node, edge, theta, mean, context)
    diagnostics = _rate_diagnostics(theta, mean, density, usrc, udst, edge)
    info = dict(variant=variant, iterations=steps, block=block, damping=float(damping),
                nuisance_updates=len(history), update_history=history,
                theta_mean=theta, theta_shape=theta_shape, theta_rate=theta_rate,
                affinity_shape=shape, affinity_rate=rate, affinity_mean=mean,
                affinity_meanlog=meanlog, affinity_prior_shape=prior_shape,
                affinity_prior_rate=prior_rate, population_alpha=alpha,
                population_mean=alpha / alpha.sum(), population_unary=unary,
                degree=degree, density_offset=density, fresh_statistics=stats,
                max_rate=diagnostics['max_class_conditional_rate'],
                rate_diagnostics=diagnostics,
                nonneighbor_volume_raw_min=raw_min,
                approximation='nonedge node-product MF; edge BP; independent Gamma nuisance')
    if details:
        info.update(node_belief=node, edge_belief=edge,
                    undirected_src=usrc, undirected_dst=udst,
                    directed_src=src, directed_dst=dst, reverse=reverse,
                    undirected_indices=undirected, graph_messages=messages,
                    to_node=to_node, to_centroid=to_centroid,
                    centroid=total.softmax(-1), field=field,
                    nonneighbor_field=penalty, nonneighbor_volume=volume,
                    field_node_belief=field_node, field_theta_mean=theta,
                    observed_nodes=ei, observed_columns=ef, log_weights=prior)
    return node, info

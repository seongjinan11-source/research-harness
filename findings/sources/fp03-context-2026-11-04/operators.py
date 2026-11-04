"""FP03 conditional contrast BP and context-only calibration helpers.

No I/O, checkpoint/model loading, fitting, or evaluation code is present.
Only symmetric observed edges have directional BP messages. Nonedges use
the explicitly approximate product-belief field in PLAN_v2. Feature cells
use the same literal finite-grid class-centroid factors as FP01/FP02.

contrast_bp(..., context[B,N] or [N], ..., a[B], b[B]) -> p[B,N,K], info
candidate_table(...) -> (a[15], b[15]), a-major/b-minor
balanced_folds(context[N], stable_ids=None, classes=None) -> CPU tensor dict
score_weights(heldp[G,S,K], heldlabels[S]) -> weights[G] float64, info
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order.
import math
import torch
from torch.nn import functional as F


def candidate_table(device='cpu', dtype=torch.float32):
    """Fixed protocol order; fixed-edge index10, graph-off index4."""
    a = torch.tensor([-math.log(2), 0., math.log(2), math.log(4), math.log(16)],
                     device=device, dtype=dtype).repeat_interleave(3)
    b = torch.tensor([-2., 0., 2.], device=device, dtype=dtype).repeat(5)
    return a, b


def _logsum_except(logits):
    """Leave-one-state logsumexp in O(K), without 1-softmax cancellation.

    Calls use finite, unclamped fields; context clamps are handled separately.
    """
    left = torch.logcumsumexp(logits, -1)
    right = torch.logcumsumexp(logits.flip(-1), -1).flip(-1)
    empty = torch.full_like(logits[..., :1], -torch.inf)
    return torch.logaddexp(torch.cat((empty, left[..., :-1]), -1),
                           torch.cat((right[..., 1:], empty), -1))


def _node_belief(field, context, classes):
    one = F.one_hot(context.clamp_min(0), classes).to(field.dtype)
    return torch.where((context >= 0)[..., None], one, field.softmax(-1))


def _topology(src, dst, n, device, dtype):
    src = torch.as_tensor(src, device=device, dtype=torch.long)
    dst = torch.as_tensor(dst, device=device, dtype=torch.long)
    if src.ndim != 1 or dst.shape != src.shape:
        raise ValueError('src/dst must be equal-length directed edge vectors')
    if src.numel() and bool(((src < 0) | (dst < 0) | (src >= n) |
                             (dst >= n) | (src == dst)).any()):
        raise ValueError('invalid endpoint or self edge')
    ordered, order = (src * n + dst).sort()
    if ordered.numel() > 1 and bool((ordered[1:] == ordered[:-1]).any()):
        raise ValueError('duplicate edge in binary adjacency')
    wanted = dst * n + src
    positions = torch.searchsorted(ordered, wanted)
    if positions.numel() and (bool((positions >= ordered.numel()).any()) or
                              not bool((ordered[positions] == wanted).all())):
        raise ValueError('each observed edge must have both directions')
    reverse = order[positions]
    degree = torch.bincount(src, minlength=n).to(dtype=dtype)
    volume = degree.sum()
    theta = degree / volume.sqrt() if bool(volume > 0) else torch.zeros_like(degree)
    return src, dst, reverse, degree, theta


def nonneighbor_field(src, dst, node_belief, theta, b):
    """Public check helper; symmetric DIRECTED edges, q[B,N,K].

    Returns signed field[B,N,K], volume[B,N,K], raw-volume minimum[B].
    Positive b adds coassignment evidence. Self and observed neighbors are
    excluded. Only nonnegative cancellation-roundoff correction is applied;
    neither the field nor a/b is clipped.
    """
    weighted = theta[None, :, None] * node_belief
    neighbors = torch.zeros_like(weighted).index_add(1, src, weighted[:, dst])
    raw = weighted.sum(1)[:, None] - weighted - neighbors
    volume = raw.clamp_min(0)
    field = b[:, None, None] * theta[None, :, None] * volume
    return field, volume, raw.flatten(1).amin(-1)


def _fields(messages, dst, to_node, to_centroid, ei, ef, prior,
            nonedge, batch, classes):
    field = nonedge.index_add(1, dst, messages).index_add(1, ei, to_node)
    total = prior[None, None].expand(batch, classes, -1, -1)
    total = total.index_add(2, ef, to_centroid.permute(0, 2, 1, 3))
    return field, total


def _graph_proposal(src, reverse, messages, field, context, a,
                    damping, bounded_potts=True, edge_chunk=4096):
    """Reviewed-grid Potts sum-product, with a generic extreme-a fallback.

    The reviewed bounded grid uses log1p(expm1(a)*softmax(cavity)). Equal-class
    cavities then remain equal componentwise even in float32, avoiding a tiny
    class-index roundoff seed from prefix/suffix reductions. For extreme a,
    fall back to logaddexp(a+cavity_c, LSE_{s!=c} cavity_s), up to gauge.
    Known sources are handled explicitly in either case.
    """
    k = field.shape[-1]
    parts = []
    for start in range(0, src.numel(), edge_chunk):
        stop = min(start + edge_chunk, src.numel())
        si = src[start:stop]
        cavity = field[:, si] - messages[:, reverse[start:stop]]
        if bounded_potts:
            proposal = torch.log1p(torch.expm1(a)[:, None, None] * cavity.softmax(-1))
        else:
            proposal = torch.logaddexp(cavity + a[:, None, None], _logsum_except(cavity))
        known = context[:, si] >= 0
        one = F.one_hot(context[:, si].clamp_min(0), k).to(field.dtype)
        proposal = torch.where(known[..., None], a[:, None, None] * one, proposal)
        proposal = proposal - proposal.mean(-1, keepdim=True)
        parts.append((1 - damping) * messages[:, start:stop] + damping * proposal)
    return torch.cat(parts, 1) if parts else messages


def _feature_cavities(field, total, to_node, to_centroid, ei, ef, context):
    local = field[:, ei]
    off = _logsum_except(local)
    active = local - to_node
    denominator = torch.logaddexp(off, active)
    logp, lognotp = active - denominator, off - denominator
    known = context[:, ei] >= 0
    matched = F.one_hot(context[:, ei].clamp_min(0), field.shape[-1]).bool()
    zero, neginf = torch.zeros_like(logp), torch.full_like(logp, -torch.inf)
    logp = torch.where(known[..., None], torch.where(matched, zero, neginf), logp)
    lognotp = torch.where(known[..., None], torch.where(matched, neginf, zero), lognotp)
    logq = (total[:, :, ef].permute(0, 2, 1, 3) - to_centroid).log_softmax(-1)
    return logp, lognotp, logq


def contrast_bp(src, dst, nodes, context, classes, observed, log_likelihood,
                log_weights, a, b, iterations=24, damping=.5, details=False):
    """Batched conditional Potts BP + nonneighbor MF + centroid BP.

    Each batch lane is an independent candidate/fold. Context, a, and b may
    have batch length1 or the common B. Topology/observed cells/likelihood
    table/grid prior are label-independent and shared, but ALL messages are
    freshly zero-initialized. Held-label/evaluator metadata is never accepted.

    Device/dtype follow log_likelihood[N,F,R]. Graph-only calls use F=0,R=1.
    Caller chunks candidate/fold lanes (reviewed execution uses at most16).
    Return shape always includes B, including B=1.
    """
    n, k, steps = int(nodes), int(classes), int(iterations)
    if n < 1 or k < 2 or steps < 0 or not 0 < float(damping) <= 1:
        raise ValueError('invalid nodes/classes/iterations/damping')
    if (log_likelihood.ndim != 3 or log_likelihood.shape[0] != n or
            not log_likelihood.is_floating_point()):
        raise ValueError('log_likelihood must be floating point [N,F,R]')
    device, dtype = log_likelihood.device, log_likelihood.dtype
    f, r = log_likelihood.shape[1:]
    if r < 1:
        raise ValueError('finite centroid grid cannot be empty')
    context = torch.as_tensor(context, device=device, dtype=torch.long)
    if context.ndim == 1:
        context = context[None]
    if context.ndim != 2 or context.shape[1] != n:
        raise ValueError('context must be [N] or [B,N]')
    a = torch.as_tensor(a, device=device, dtype=dtype).reshape(-1)
    b = torch.as_tensor(b, device=device, dtype=dtype).reshape(-1)
    batch = max(context.shape[0], a.numel(), b.numel())
    if batch < 1 or any(x not in (1, batch) for x in (context.shape[0], a.numel(), b.numel())):
        raise ValueError('context/a/b batch lengths must be1 or common B')
    context = context.expand(batch, n)
    a, b = a.expand(batch), b.expand(batch)
    if bool(((context < -1) | (context >= k)).any()):
        raise ValueError('context must contain -1 or a valid class')
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise ValueError('a/b must be finite')
    # The actual grid is inside [-log2,log16]. A wider finite safe range keeps
    # the same stable componentwise identity for numerical derivative checks.
    bounded_potts = bool(((a >= -10) & (a <= 10)).all())
    observed = torch.as_tensor(observed, device=device, dtype=torch.bool)
    if observed.shape != (n, f):
        raise ValueError('observed must be [N,F]')
    src, dst, reverse, degree, theta = _topology(src, dst, n, device, dtype)
    prior = torch.as_tensor(log_weights, device=device, dtype=dtype)
    if prior.ndim == 1 and prior.shape == (r,):
        prior = prior[None].expand(f, r)
    elif prior.shape != (f, r):
        raise ValueError('log_weights must be [R] or [F,R]')
    prior = prior.log_softmax(-1)
    ei, ef = observed.nonzero(as_tuple=True)
    likelihood = log_likelihood[ei, ef]
    if not bool(torch.isfinite(likelihood).all() & torch.isfinite(prior).all()):
        raise ValueError('visible likelihood and normalized prior must be finite')
    messages = log_likelihood.new_zeros(batch, src.numel(), k)
    to_node = log_likelihood.new_zeros(batch, ei.numel(), k)
    to_centroid = log_likelihood.new_zeros(batch, ei.numel(), k, r)
    node = _node_belief(log_likelihood.new_zeros(batch, n, k), context, k)
    field_node = node
    nonedge, volume, raw_min = nonneighbor_field(src, dst, node, theta, b)
    field, total = _fields(messages, dst, to_node, to_centroid, ei, ef,
                            prior, nonedge, batch, k)
    if steps == 0:
        node = _node_belief(field, context, k)
    for _ in range(steps):
        field_node = node
        nonedge, volume, raw_min = nonneighbor_field(src, dst, node, theta, b)
        field, total = _fields(messages, dst, to_node, to_centroid, ei, ef,
                                prior, nonedge, batch, k)
        next_messages = _graph_proposal(src, reverse, messages, field,
                                         context, a, damping, bounded_potts)
        if ei.numel():
            lp, lnotp, lq = _feature_cavities(field, total, to_node, to_centroid,
                                             ei, ef, context)
            proposal_a = torch.logsumexp(lq + likelihood[None, :, None], -1)
            proposal_v = torch.logaddexp(lnotp[..., None], lp[..., None] + likelihood[None, :, None])
            proposal_v = proposal_v - proposal_v.mean(-1, keepdim=True)
            next_a = (1 - damping) * to_node + damping * proposal_a
            next_v = (1 - damping) * to_centroid + damping * proposal_v
        else:
            next_a, next_v = to_node, to_centroid
        messages, to_node, to_centroid = next_messages, next_a, next_v
        # Read with the just-used MF field, without an extra hidden MF step.
        field, total = _fields(messages, dst, to_node, to_centroid, ei, ef,
                                prior, nonedge, batch, k)
        node = _node_belief(field, context, k)
    # Empty observations retain an exact zero derivative connection, without
    # touching masked/poisoned likelihood entries.
    node = node + likelihood.sum() * 0
    info = dict(a=a, b=b, batch=batch, iterations=steps, damping=float(damping),
                sources=int(ei.numel()), degree=degree, theta=theta,
                nonneighbor_volume_raw_min=raw_min,
                approximation='nonedge node-product MF; finite edge/centroid BP')
    if details:
        info.update(graph_messages=messages, to_node=to_node, to_centroid=to_centroid,
                    field=field, node_belief=node, field_node_belief=field_node,
                    nonneighbor_field=nonedge, nonneighbor_volume=volume,
                    reverse=reverse, directed_src=src, directed_dst=dst,
                    centroid=total.softmax(-1), observed_nodes=ei,
                    observed_columns=ef, log_weights=prior, context=context)
    return node, info


def balanced_folds(context, stable_ids=None, classes=None):
    """CPU split-only metadata for balanced leave-one-per-class calibration.

    For C+ represented classes let m=min positive count. Select m smallest
    stable IDs per class; fold t hides selected index t in EVERY class.
    Every unselected/extra context is unclamped in every calibration fold.
    Only fold_contexts may be passed to contrast_bp. Held/evaluation metadata
    is used to extract/score predictions after a forward, never as model input.

    Class slots are ordered by their smallest stable node ID, not class name.
    This makes the held-node table itself invariant to class relabeling.
    Node permutation must carry stable_ids along with nodes. Default IDs are
    the original row IDs arange(N); they must not be regenerated after a test
    permutation. `classes` only controls full count-vector length/validation.
    """
    context = torch.as_tensor(context, dtype=torch.long, device='cpu')
    if context.ndim != 1 or bool((context < -1).any()):
        raise ValueError('context must be [N], with -1 for unknown')
    n = context.numel()
    ids = (torch.arange(n, dtype=torch.long) if stable_ids is None else
           torch.as_tensor(stable_ids, dtype=torch.long, device='cpu'))
    if ids.shape != (n,) or ids.unique().numel() != n:
        raise ValueError('stable_ids must be a unique ID per node')
    known = (context >= 0).nonzero(as_tuple=True)[0]
    inferred_k = int(context[known].max()) + 1 if known.numel() else 0
    k = inferred_k if classes is None else int(classes)
    if k < inferred_k or k < 0:
        raise ValueError('classes is smaller than a known label')
    counts = torch.bincount(context[known], minlength=k)
    represented = context[known].unique(sorted=True)
    groups = []
    for c in represented.tolist():
        members = (context == c).nonzero(as_tuple=True)[0]
        members = members[ids[members].argsort()]
        groups.append((int(ids[members[0]]), c, members))
    groups.sort(key=lambda item: item[0])
    represented = torch.tensor([c for _, c, _ in groups], dtype=torch.long)
    m = min((members.numel() for _, _, members in groups), default=0)
    if groups:
        held = torch.stack([members[:m] for _, _, members in groups], 1)
        labels = context[held]
    else:
        held = torch.empty((0, 0), dtype=torch.long)
        labels = held.clone()
    folds = torch.full((m, n), -1, dtype=torch.long)
    selected_flat = held.flatten()
    for t in range(m):
        folds[t, selected_flat] = context[selected_flat]
        folds[t, held[t]] = -1
    selected_mask = torch.zeros(n, dtype=torch.bool)
    selected_mask[selected_flat] = True
    unused = known[~selected_mask[known]]
    unused = unused[ids[unused].argsort()]
    return dict(fold_contexts=folds, held_nodes=held, held_labels=labels,
                held_flat=selected_flat, labels=labels.flatten(),
                selected_nodes=selected_flat, selected_labels=labels.flatten(),
                unused_nodes=unused, class_counts=counts,
                represented_classes=represented, stable_ids=ids.clone(),
                selected_stable_ids=ids[held], m=int(m), points=int(held.numel()),
                original_context_count=int(known.numel()))


def _weight_statistics(weights):
    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum()
    return dict(entropy=entropy, ess=weights.square().sum().reciprocal(),
                prior_kl=weights.new_tensor(float(weights.numel())).log() - entropy)


def score_weights(heldp, heldlabels):
    """Fixed macro log-score weights from saved raw probabilities.

    heldp[G,S,K], heldlabels[S]. Convert to float64, clip EVERY cell to
    [1e-8,1], row-normalize, then mean own-label log probability per represented
    class and mean these class means. Uniform candidate prior, unit MEAN score
    scale. This is predictive weighting, not posterior or stacking estimation.
    S=0 returns zero scores, uniform weights and empty per-class scores.
    """
    if heldp.ndim != 3 or heldp.shape[0] < 1 or heldp.shape[-1] < 2:
        raise ValueError('heldp must be [G>=1,S,K>=2]')
    g, s, k = heldp.shape
    labels = torch.as_tensor(heldlabels, device=heldp.device, dtype=torch.long)
    if labels.shape != (s,) or bool(((labels < 0) | (labels >= k)).any()):
        raise ValueError('invalid heldlabels shape/range')
    raw = heldp.to(torch.float64)
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError('held probabilities must be finite and nonnegative')
    if s and bool((raw.sum(-1) <= 0).any()):
        raise ValueError('held probability rows cannot have zero total')
    normalized = raw.clamp(1e-8, 1.)
    normalized = normalized / normalized.sum(-1, keepdim=True)
    represented = labels.unique(sorted=True)
    if s:
        own = normalized.gather(-1, labels[None, :, None].expand(g, -1, 1)).squeeze(-1).log()
        per_class = torch.stack([own[:, labels == c].mean(-1) for c in represented], -1)
        scores = per_class.mean(-1)
    else:
        per_class = raw.new_empty(g, 0)
        scores = raw.new_zeros(g)
    weights = (scores - math.log(g)).softmax(0)
    info = dict(scores=scores, per_class_scores=per_class,
                represented_classes=represented,
                represented_class_counts=torch.stack([(labels == c).sum() for c in represented])
                if represented.numel() else labels.new_empty(0),
                normalized_held=normalized, floor=1e-8,
                clipped_cells=((raw < 1e-8) | (raw > 1)).sum((1, 2)),
                zero_cells=(raw == 0).sum((1, 2)), map_idx=int(scores.argmax()))
    info.update(_weight_statistics(weights))
    return weights, info


def mixture_diagnostics(weights, a, b, candidate_probabilities=None):
    """Float64 mixture diagnostics, not uncertainty credible intervals.

    Optional probabilities are [G,N,K]. Mean pairwise total variation averages
    uniformly over g<h and over N; weighted TV-to-mixture is a distinct statistic.
    """
    w = torch.as_tensor(weights).to(torch.float64)
    if w.ndim != 1 or w.numel() < 1 or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError('invalid candidate weights')
    if not bool(w.sum() > 0):
        raise ValueError('candidate weights must have positive total')
    w = w / w.sum()
    aa = torch.as_tensor(a, device=w.device, dtype=w.dtype)
    bb = torch.as_tensor(b, device=w.device, dtype=w.dtype)
    if aa.shape != w.shape or bb.shape != w.shape:
        raise ValueError('contrast/weight shape mismatch')
    ma, mb = (w * aa).sum(), (w * bb).sum()
    result = _weight_statistics(w)
    result.update(a_mean=ma, b_mean=mb, a_variance=(w * (aa - ma).square()).sum(),
                  b_variance=(w * (bb - mb).square()).sum(),
                  ab_covariance=(w * (aa - ma) * (bb - mb)).sum())
    if candidate_probabilities is not None:
        p = candidate_probabilities.to(device=w.device, dtype=torch.float64)
        if p.ndim != 3 or p.shape[0] != w.numel():
            raise ValueError('candidate_probabilities must be [G,N,K]')
        mixture = torch.einsum('g,gnk->nk', w, p)
        if p.shape[1] and p.shape[0] > 1:
            pairs = [.5 * (p[i] - p[j]).abs().sum(-1).mean()
                     for i in range(p.shape[0]) for j in range(i + 1, p.shape[0])]
            result['mean_pairwise_total_variation'] = torch.stack(pairs).mean()
        else:
            result['mean_pairwise_total_variation'] = w.new_zeros(())
        result['weighted_mean_tv_to_mixture'] = (
            (w[:, None] * .5 * (p - mixture[None]).abs().sum(-1)).sum(0).mean()
            if p.shape[1] else w.new_zeros(()))
    return result

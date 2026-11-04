"""Independent FP03 tiny checks; no real bank, training, or final predictions.

NumPy enumeration/generic pairwise BP are independent of production scalar
gates. Exact tree assertions and loopy/nonneighbor-approximation diagnostics
are distinct. Save PASS only after full checks and unchanged source hashes.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows DLL import order.
import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def as_np(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def close(actual, expected, tol=2e-10, name='comparison'):
    a, b = as_np(actual), as_np(expected)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), name
    error = float(np.max(np.abs(a - b), initial=0))
    assert error <= tol, (name, error, tol)
    return error


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def opportunity(adjacency):
    degree = np.asarray(adjacency).sum(-1)
    return degree / math.sqrt(float(degree.sum())) if degree.sum() else np.zeros_like(degree, dtype=float)


def exact_enumeration(adjacency, context, classes, observed, likelihood, log_weights, a, b):
    """Whole finite-grid Potts score, all unordered pairs, centroid integrated."""
    n, k = len(context), int(classes)
    f, r = observed.shape[1], likelihood.shape[-1]
    lw = np.broadcast_to(log_weights, (f, r)).copy()
    lw -= logsumexp(lw, axis=-1, keepdims=True)
    theta = opportunity(adjacency)
    states, scores = [], []
    for state in itertools.product(range(k), repeat=n):
        z = np.asarray(state)
        if any(context[i] >= 0 and z[i] != context[i] for i in range(n)):
            continue
        score = -n * math.log(k)
        for i in range(n):
            for j in range(i + 1, n):
                coupling = a if adjacency[i, j] else b * theta[i] * theta[j]
                score += coupling * (z[i] == z[j])
        for c in range(k):
            for col in range(f):
                ids = np.flatnonzero(observed[:, col] & (z == c))
                score += logsumexp(lw[col] + likelihood[ids, col].sum(0))
        states.append(z)
        scores.append(score)
    states, scores = np.asarray(states), np.asarray(scores)
    logz = float(logsumexp(scores))
    weights = np.exp(scores - logz)
    p = np.zeros((n, k))
    for i in range(n):
        for c in range(k):
            p[i, c] = weights[states[:, i] == c].sum()
    return p, logz


def numpy_bp(adjacency, context, classes, observed, likelihood, log_weights, a, b,
             iterations=24, damping=.5, mode='mf'):
    """Generic pairwise BP; mode mf uses direct nonneighbor sums each step.

    Each observed-cell/class gate is an explicit KxR potential. Dense mode
    puts every nonedge in the factor graph instead; neither loopy version is
    claimed exact. Last readout retains the just-used OLD-q MF field.
    """
    n, k = len(context), int(classes)
    f, r = observed.shape[1], likelihood.shape[-1]
    lw = np.broadcast_to(log_weights, (f, r)).copy()
    lw -= logsumexp(lw, axis=-1, keepdims=True)
    theta = opportunity(adjacency)
    unary = [np.full(k, -math.log(k)) for _ in range(n)]
    for i in range(n):
        if context[i] >= 0:
            unary[i][:] = -np.inf
            unary[i][context[i]] = 0.
    for c in range(k):
        for col in range(f):
            unary.append(lw[col].copy())
    factors = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                factors.append((i, j, np.eye(k) * a))
            elif mode == 'dense':
                factors.append((i, j, np.eye(k) * (b * theta[i] * theta[j])))
    for i, col in zip(*np.nonzero(observed)):
        for c in range(k):
            matrix = np.zeros((k, r))
            matrix[c] = likelihood[i, col]
            factors.append((i, n + c * f + col, matrix))
    incoming = [[] for _ in unary]
    for ei, (i, j, _) in enumerate(factors):
        incoming[i].append((ei, 1))
        incoming[j].append((ei, 0))
    messages = [[np.zeros(len(unary[j])), np.zeros(len(unary[i]))] for i, j, _ in factors]
    p = np.stack([np.exp(u - logsumexp(u)) for u in unary[:n]])
    mf = np.zeros((n, k))
    field_q = p.copy()
    def totals(current, external):
        out = [v.copy() for v in unary]
        for i in range(n):
            out[i] += external[i]
        for v, indices in enumerate(incoming):
            for ei, direction in indices:
                out[v] += current[ei][direction]
        return out
    for _ in range(iterations):
        field_q = p.copy()
        mf[:] = 0
        if mode == 'mf':
            for i in range(n):
                for j in range(n):
                    if j != i and not adjacency[i, j]:
                        mf[i] += b * theta[i] * theta[j] * p[j]
        total = totals(messages, mf)
        new = []
        for ei, (i, j, matrix) in enumerate(factors):
            ci, cj = total[i] - messages[ei][1], total[j] - messages[ei][0]
            to_j = logsumexp(ci[:, None] + matrix, axis=0)
            to_i = logsumexp(cj[None, :] + matrix, axis=1)
            to_j -= to_j.mean()
            to_i -= to_i.mean()
            new.append([(1 - damping) * messages[ei][0] + damping * to_j,
                        (1 - damping) * messages[ei][1] + damping * to_i])
        messages = new
        total = totals(messages, mf)
        p = np.stack([np.exp(t - logsumexp(t)) for t in total[:n]])
    return p, dict(field=np.asarray(total[:n]), nonneighbor_field=mf, field_node_belief=field_q)


def tiny_case(n=7, classes=3, columns=2, grid=5, seed=74126, device='cpu', dtype=torch.float64):
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    if n >= 5:
        adjacency[0, 3] = adjacency[3, 0] = 1
        adjacency[1, 4] = adjacency[4, 1] = 1
    context = np.full(n, -1, dtype=np.int64)
    context[0] = 0
    if n > 3:
        context[1] = 1
    observed = np.zeros((n, columns), dtype=bool)
    observed[2:min(n - 1, 5)] = True
    x = rng.normal(size=(n, columns))
    points = np.stack([np.linspace(-1.1, 1.3, grid) + .1 * f for f in range(columns)])
    sigma = np.linspace(.6, .85, columns)
    ll = (-.5 * math.log(2 * math.pi) - np.log(sigma)[None, :, None]
          - .5 * ((x[:, :, None] - points[None]) / sigma[None, :, None]) ** 2)
    lw = rng.normal(0, .15, (columns, grid))
    src, dst = np.nonzero(adjacency)
    return dict(src=torch.tensor(src, device=device), dst=torch.tensor(dst, device=device), nodes=n,
                context=torch.tensor(context, device=device), classes=classes,
                observed=torch.tensor(observed, device=device),
                log_likelihood=torch.tensor(ll, device=device, dtype=dtype),
                log_weights=torch.tensor(lw, device=device, dtype=dtype))


def numpy_case(case):
    n = case['nodes']
    adjacency = np.zeros((n, n), dtype=np.int64)
    adjacency[as_np(case['src']), as_np(case['dst'])] = 1
    return dict(adjacency=adjacency, context=as_np(case['context']), classes=case['classes'],
                observed=as_np(case['observed']), likelihood=as_np(case['log_likelihood']),
                log_weights=as_np(case['log_weights']))


def independent_scores(probability, labels):
    """Independent float64 score on represented classes; candidate-major."""
    p = np.clip(as_np(probability).astype(np.float64), 1e-8, 1)
    p /= p.sum(-1, keepdims=True)
    y = as_np(labels).astype(np.int64)
    if not len(y):
        score = np.zeros(len(p))
    else:
        score = np.asarray([np.mean([np.log(row[np.arange(len(y))[y == c], y[y == c]]).mean()
                                    for c in np.unique(y)]) for row in p])
    logw = score - math.log(len(p))
    weight = np.exp(logw - logsumexp(logw))
    return score, weight


def branch(ops, case, a, b, **kwargs):
    dtype, device = case['log_likelihood'].dtype, case['log_likelihood'].device
    av = torch.as_tensor(a, dtype=dtype, device=device)
    bv = torch.as_tensor(b, dtype=dtype, device=device)
    p, info = ops.contrast_bp(**case, a=av, b=bv, details=True, **kwargs)
    context_batch = 1 if case['context'].ndim == 1 else len(case['context'])
    batch = max(av.numel(), bv.numel(), context_batch)
    assert p.shape == (batch, case['nodes'], case['classes']), p.shape
    assert torch.isfinite(p).all() and bool((p >= 0).all())
    close(p.sum(-1), np.ones((batch, case['nodes'])), 3e-7 if dtype == torch.float32 else 2e-11)
    return p, info


def oracle_checks(ops):
    errors, diagnostics = {}, []
    for classes in (2, 3):
        c = tiny_case(n=3, classes=classes, columns=1)
        c['observed'][:] = False
        c['observed'][1] = True  # K separate centroid leaves: an actual tree.
        args = numpy_case(c)
        for a in (-math.log(2), math.log(4)):
            exact, _ = exact_enumeration(**args, a=a, b=0.)
            oracle, _ = numpy_bp(**args, a=a, b=0., iterations=160)
            p, _ = branch(ops, c, a, 0., iterations=160)
            errors[f'tree_numpy_K{classes}_a{a}'] = close(oracle, exact, 2e-11)
            errors[f'tree_production_K{classes}_a{a}'] = close(p[0], exact, 3e-10)
    # The generic pairwise feature graph is independent of scalar gates.
    # MF parity is required even with centroid loops and signed nonedges.
    for a, b in ((-.6931471805599453, -2.), (0., 2.), (1.3862943611198906, -2.), (2.772588722239781, 2.)):
        c = tiny_case(n=5, classes=3, columns=2, grid=3)
        args = numpy_case(c)
        want, _ = numpy_bp(**args, a=a, b=b)
        p, info = branch(ops, c, a, b)
        key = f'a{a}_b{b}'
        errors['generic_pairwise_MF_' + key] = close(p[0], want, 4e-10)
        exact, logz = exact_enumeration(**args, a=a, b=b)
        dense, _ = numpy_bp(**args, a=a, b=b, iterations=160, mode='dense')
        diagnostics.append(dict(a=a, b=b, mf_vs_exact=float(np.max(np.abs(want - exact))),
                                dense_loopy_vs_exact=float(np.max(np.abs(dense - exact))),
                                mf_vs_dense=float(np.max(np.abs(want - dense))), exact_logz=logz,
                                scope='diagnostic: nonneighbor MF and finite loopy centroid/graph BP'))
        q = as_np(info['field_node_belief'])[0]
        theta = opportunity(args['adjacency'])
        volume, external = np.zeros_like(q), np.zeros_like(q)
        for i in range(c['nodes']):
            for j in range(c['nodes']):
                if i != j and not args['adjacency'][i, j]:
                    volume[i] += theta[j] * q[j]
                    external[i] += b * theta[i] * theta[j] * q[j]
        errors['self_neighbor_exclusion_' + key] = close(info['nonneighbor_field'][0], external, 2e-11)
        errors['volume_' + key] = close(info['nonneighbor_volume'][0], volume, 2e-11)
        errors['theta_no_second_v_division_' + key] = close(info['theta'], theta)
        if b < 0:
            assert bool((info['nonneighbor_field'] <= 1e-12).all())
        if b > 0:
            assert bool((info['nonneighbor_field'] >= -1e-12).all())
        # Reconstruct the saved last field; the nonedge term is added once.
        field = np.full((c['nodes'], c['classes']), -math.log(c['classes'])) + external
        for edge, target in enumerate(as_np(c['dst'])):
            field[target] += as_np(info['graph_messages'])[0, edge]
        for edge, node in enumerate(as_np(info['observed_nodes'])):
            field[node] += as_np(info['to_node'])[0, edge]
        # A class-constant unary/gauge difference is immaterial.
        got = as_np(info['field'])[0]
        errors['last_field_no_doublecount_' + key] = close(got - got.mean(-1, keepdims=True),
                                                           field - field.mean(-1, keepdims=True), 3e-11)
    c = tiny_case(n=4, columns=1)
    full_src, full_dst = np.nonzero(~np.eye(4, dtype=bool))
    c = dict(c, src=torch.tensor(full_src), dst=torch.tensor(full_dst))
    _, info = branch(ops, c, 0., [-2., 2.])
    errors['complete_graph_has_no_nonedge_field'] = close(info['nonneighbor_field'], np.zeros((2, 4, 3)), 3e-14)
    return dict(errors=errors, approximation_diagnostics=diagnostics)


def invariant_checks(ops, device='cpu', dtype=torch.float64):
    c = tiny_case(device=device, dtype=dtype)
    av, bv = [-math.log(2), 0., math.log(4)], [-2., 2., 0.]
    p, info = branch(ops, c, av, bv)
    tol = 4e-6 if dtype == torch.float32 else 4e-9
    errors = {}
    known = c['context'] >= 0
    one = torch.nn.functional.one_hot(c['context'][known], c['classes']).to(p)
    errors['context_clamp'] = close(p[:, known], one[None].expand(len(p), -1, -1))
    poison = c['log_likelihood'].clone()
    poison[~c['observed']] = float('nan')
    pp, _ = branch(ops, dict(c, log_likelihood=poison), av, bv)
    errors['hidden_NaN'] = close(pp, p, tol)
    poison[~c['observed']] = 1e30
    pp, _ = branch(ops, dict(c, log_likelihood=poison), av, bv)
    errors['hidden_large'] = close(pp, p, tol)
    cols = torch.tensor([1, 0], device=device)
    cp = dict(c, observed=c['observed'][:, cols], log_likelihood=c['log_likelihood'][:, cols],
              log_weights=c['log_weights'][cols])
    pp, _ = branch(ops, cp, av, bv)
    errors['column_and_prior_permutation'] = close(pp, p, tol)
    pp, _ = branch(ops, dict(c, log_weights=c['log_weights'] + c['log_weights'].new_tensor([[2.], [-3.]])), av, bv)
    errors['prior_log_gauge'] = close(pp, p, tol)
    shared = c['log_weights'][0]
    pp, _ = branch(ops, dict(c, log_weights=shared), av, bv)
    qq, _ = branch(ops, dict(c, log_weights=shared[None].expand(2, -1)), av, bv)
    errors['shared_prior_shape'] = close(pp, qq, tol)
    permutation = torch.tensor([2, 0, 1], device=device)
    context = c['context'].clone()
    context[known] = permutation[context[known]]
    pp, _ = branch(ops, dict(c, context=context), av, bv)
    errors['class_permutation'] = close(pp[:, :, permutation], p, tol)
    for label, order in (('nodes', [5, 2, 0, 6, 1, 4, 3]), ('source_rows', [0, 1, 4, 2, 3, 5, 6])):
        order = torch.tensor(order, device=device)
        inverse = torch.argsort(order)
        cp = dict(c, src=inverse[c['src']], dst=inverse[c['dst']], context=c['context'][order],
                  observed=c['observed'][order], log_likelihood=c['log_likelihood'][order])
        pp, _ = branch(ops, cp, av, bv)
        errors[label] = close(pp, p[:, order], tol)
    eo = torch.arange(len(c['src']) - 1, -1, -1, device=device)
    pp, _ = branch(ops, dict(c, src=c['src'][eo], dst=c['dst'][eo]), av, bv)
    errors['directed_edge_order'] = close(pp, p, tol)
    c0 = dict(c, observed=torch.zeros_like(c['observed']))
    p0, _ = branch(ops, c0, av, bv)
    no_columns = dict(c0, observed=c['observed'][:, :0],
                      log_likelihood=c['log_likelihood'][:, :0, :1],
                      log_weights=c['log_weights'][:0, :1])
    p_no_columns, _ = branch(ops, no_columns, av, bv)
    errors['graph_only_F0_vs_hidden_columns'] = close(p_no_columns, p0, tol)
    single = torch.zeros_like(c['observed'])
    single[3] = True
    p1, _ = branch(ops, dict(c, observed=single), av, bv)
    errors['one_source_row'] = close(p1, p0, tol)
    single[:] = False
    single[2, 0], single[4, 1] = True, True
    p1, _ = branch(ops, dict(c, observed=single), av, bv)
    errors['one_source_per_column'] = close(p1, p0, tol)
    pe, _ = branch(ops, dict(c, src=c['src'][:0], dst=c['dst'][:0]), av, bv)
    expected = np.full((3, c['nodes'], c['classes']), 1 / c['classes'])
    expected[:, as_np(known)] = as_np(one)
    errors['empty_edges'] = close(pe, expected, tol)
    singleton = dict(c0, nodes=1, src=c['src'][:0], dst=c['dst'][:0],
                     context=c['context'].new_tensor([-1]), observed=c['observed'][:1].clone().fill_(False),
                     log_likelihood=c['log_likelihood'][:1])
    ps, _ = branch(ops, singleton, av, bv)
    errors['one_node_empty'] = close(ps, np.full((3, 1, 3), 1 / 3), tol)
    off, _ = branch(ops, c, 0., 0.)
    query = ~c['observed'].any(-1) & ~known
    errors['graph_off_missing_queries_uniform'] = close(off[:, query], np.full((1, int(query.sum()), 3), 1 / 3), tol)
    unanchored = dict(c0, context=c['context'].clone().fill_(-1))
    pu, _ = branch(ops, unanchored, av, bv)
    errors['no_anchors_uniform_full_K'] = close(pu, np.full((3, c['nodes'], 3), 1 / 3), tol)
    assert set(inspect.signature(ops.contrast_bp).parameters) == {
        'src', 'dst', 'nodes', 'context', 'classes', 'observed', 'log_likelihood',
        'log_weights', 'a', 'b', 'iterations', 'damping', 'details'}
    return errors


def batch_checks(ops, dtype=torch.float64):
    c = tiny_case(dtype=dtype)
    a, b = ops.candidate_table(dtype=dtype)
    # 35 elements ensure both complete16 and partial3 chunks. Every batch
    # gets its own candidate and distinct context mask, never cached messages.
    ids = torch.arange(35) % len(a)
    context = c['context'][None].repeat(35, 1)
    for i in range(35):
        if i % 2:
            context[i, 0] = -1
        if i % 3 == 0:
            context[i, 1] = -1
    outputs = {}
    for chunk in (1, 16, 35):
        pieces = []
        for start in range(0, 35, chunk):
            stop = min(start + chunk, 35)
            pp, _ = branch(ops, dict(c, context=context[start:stop]), a[ids[start:stop]], b[ids[start:stop]])
            pieces.append(pp)
        outputs[chunk] = torch.cat(pieces)
    tol = 5e-6 if dtype == torch.float32 else 4e-10
    errors = {f'chunk1_vs_{chunk}': close(outputs[1], outputs[chunk], tol) for chunk in (16, 35)}
    labels = np.array([0, 1, 2])
    held = {}
    for chunk, prediction in outputs.items():
        # Same raw 15-candidate held records, then the same score reconstruction.
        held[chunk] = prediction[:15, [2, 4, 6]].detach()
    for chunk in (16, 35):
        want_s, want_w = independent_scores(held[1], labels)
        got_w, got_info = ops.score_weights(held[chunk], torch.tensor(labels))
        errors[f'chunk{chunk}_score'] = close(got_info['scores'], want_s, tol)
        errors[f'chunk{chunk}_weight'] = close(got_w, want_w, tol)
    return errors


def gradient_checks(ops):
    c = tiny_case()
    rng = np.random.default_rng(2917)
    direction = torch.tensor(rng.normal(size=c['log_likelihood'].shape), dtype=torch.float64)
    direction *= c['observed'][..., None]
    result = {}
    for a, b in ((math.log(4), 0.), (math.log(2), -2.), (-math.log(2), 2.)):
        ll = c['log_likelihood'].clone().detach().requires_grad_()
        def objective(table):
            p, _ = branch(ops, dict(c, log_likelihood=table), a, b)
            return -p[0, -1, 2].log()
        value = objective(ll)
        gradient = torch.autograd.grad(value, ll)[0]
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 1e-10
        hidden_error = close(gradient[~c['observed']], np.zeros(tuple(gradient[~c['observed']].shape)), 1e-15)
        analytic = float((gradient * direction).sum())
        eps = 2e-5
        finite = float((objective(ll.detach() + eps * direction) - objective(ll.detach() - eps * direction)) / (2 * eps))
        error = abs(analytic - finite)
        assert error < 3e-7 + 2e-5 * abs(finite), (a, b, analytic, finite, error)
        ll0 = ll.detach().clone().requires_grad_()
        p0, _ = branch(ops, dict(c, log_likelihood=ll0, observed=torch.zeros_like(c['observed'])), a, b)
        g0 = torch.autograd.grad(-p0[0, -1, 2].log(), ll0)[0]
        zero_error = close(g0, np.zeros(tuple(ll0.shape)), 1e-15)
        result[f'a{a}_b{b}'] = dict(autograd=analytic, central_difference=finite, error=error,
                                   gradient_absolute_sum=float(gradient.abs().sum()),
                                   hidden_gradient_error=hidden_error, source0_gradient_error=zero_error,
                                   query_own_features_missing=True)
    return result


def fold_score_checks(ops):
    errors = {}
    rng = np.random.default_rng(187451)
    counts = np.array([4, 4, 4, 3, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4])
    n, k = 68, len(counts)
    nodes = rng.permutation(n)[:counts.sum()]
    context = np.full(n, -1, dtype=np.int64)
    context[nodes] = np.repeat(np.arange(k), counts)
    stable = rng.permutation(np.arange(n) + 1000)
    def reference_folds(ctx, ids, classes):
        groups = []
        for c in sorted(set(ctx[ctx >= 0])):
            members = np.flatnonzero(ctx == c)
            members = members[np.argsort(ids[members])]
            groups.append((ids[members[0]], int(c), members))
        groups.sort(key=lambda row: row[0])
        m = min((len(members) for _, _, members in groups), default=0)
        held = np.stack([members[:m] for _, _, members in groups], axis=1) if groups else np.empty((0, 0), dtype=np.int64)
        folds = np.full((m, len(ctx)), -1, dtype=np.int64)
        for fold in range(m):
            for _, c, members in groups:
                for i in members[:m]:
                    if i not in held[fold]:
                        folds[fold, i] = c
        return held, folds, np.bincount(ctx[ctx >= 0], minlength=classes)
    split = ops.balanced_folds(torch.tensor(context), stable_ids=torch.tensor(stable), classes=k)
    held, expected_contexts, expected_counts = reference_folds(context, stable, k)
    errors['CS_like_held_ids'] = close(split['held_nodes'], held, 0)
    errors['CS_like_fold_contexts'] = close(split['fold_contexts'], expected_contexts, 0)
    errors['CS_like_counts'] = close(split['class_counts'], expected_counts, 0)
    errors['CS_like_selected_stable_ids'] = close(split['selected_stable_ids'], stable[held], 0)
    errors['CS_like_held_labels'] = close(split['held_labels'], context[held], 0)
    assert split['m'] == 3 and split['points'] == 42 and len(split['unused_nodes']) == 13
    assert np.array_equal(as_np(split['held_flat']), held.flatten())
    assert np.array_equal(as_np(split['labels']), context[held.flatten()])
    assert len(set(as_np(split['held_flat']).tolist())) == 42
    for row in as_np(split['fold_contexts']):
        cc = np.bincount(row[row >= 0], minlength=k)
        assert np.all(cc[counts > 0] == 2) and cc[9] == 0
        assert np.all(row[as_np(split['unused_nodes'])] == -1)
    order = rng.permutation(n)
    permuted = ops.balanced_folds(torch.tensor(context[order]), stable_ids=torch.tensor(stable[order]), classes=k)
    errors['stable_ID_node_permutation'] = close(stable[held], stable[order][as_np(permuted['held_nodes'])], 0)
    inverse = np.argsort(order)
    errors['node_permutation_fold_masks'] = close(permuted['fold_contexts'], expected_contexts[:, order], 0)
    class_permutation = rng.permutation(k)
    relabeled = context.copy()
    relabeled[context >= 0] = class_permutation[context[context >= 0]]
    permuted = ops.balanced_folds(torch.tensor(relabeled), stable_ids=torch.tensor(stable), classes=k)
    errors['class_permutation_same_physical_folds'] = close(permuted['held_nodes'], held, 0)
    expect_relabel = expected_contexts.copy()
    expect_relabel[expect_relabel >= 0] = class_permutation[expected_contexts[expected_contexts >= 0]]
    errors['class_permutation_fold_masks'] = close(permuted['fold_contexts'], expect_relabel, 0)
    empty = ops.balanced_folds(torch.full((9,), -1), classes=4)
    assert empty['m'] == 0 and empty['points'] == 0 and empty['fold_contexts'].shape == (0, 9)
    w0, i0 = ops.score_weights(torch.empty(15, 0, 4), torch.empty(0, dtype=torch.long))
    errors['m0_scores'] = close(i0['scores'], np.zeros(15), 0)
    errors['m0_prior_weights'] = close(w0, np.ones(15) / 15, 2e-15)
    # n_c=[2,1,2,0]: all selected and extra context must be hidden for m=1.
    single_context = torch.tensor([-1, 0, 0, 1, 2, 2, -1])
    single = ops.balanced_folds(single_context, classes=4)
    assert single['m'] == 1 and bool((single['fold_contexts'] == -1).all())
    c = tiny_case(classes=4, columns=1)
    c = dict(c, context=single['fold_contexts'], observed=torch.zeros_like(c['observed']))
    aa, bb = ops.candidate_table(dtype=torch.float64)
    p, _ = branch(ops, c, aa, bb)
    hp = p[:, as_np(single['held_flat'])]
    errors['m1_prediction_full_K'] = close(hp, np.full((15, 3, 4), .25), 3e-11)
    wm, im = ops.score_weights(hp, single['labels'])
    errors['m1_prior_weights'] = close(wm, np.ones(15) / 15, 2e-14)
    errors['m1_score_is_log_full_K'] = close(im['scores'], np.full(15, -math.log(4)), 2e-14)
    # High-degree all-unknown FP32 case: no rounding-created class cue may
    # differentiate the 15 candidate scores when m1 has removed all anchors.
    star = tiny_case(n=17, classes=15, columns=1, dtype=torch.float32)
    center = torch.zeros(16, dtype=torch.long)
    leaves = torch.arange(1, 17)
    star = dict(star, src=torch.cat((center, leaves)), dst=torch.cat((leaves, center)),
                context=torch.full((17,), -1), observed=star['observed'][:, :0],
                log_likelihood=star['log_likelihood'][:, :0, :1],
                log_weights=star['log_weights'][:0, :1])
    a32, b32 = ops.candidate_table(dtype=torch.float32)
    unanchored, _ = branch(ops, star, a32, b32)
    target = torch.full_like(unanchored, 1 / 15)
    errors['m1_FP32_star_bitwise_symmetry'] = close(unanchored, target, 0)
    star_weights, _ = ops.score_weights(unanchored[:, :4], torch.tensor([0, 2, 5, 14]))
    errors['m1_FP32_star_prior_weights'] = close(star_weights, np.ones(15) / 15, 0)
    # Freeze the split first. Change ALL held labels and ALL unused extras,
    # rebuild only the allowed clamped input, then compare full predictions.
    c = tiny_case(n=13, classes=4, columns=1)
    full = torch.tensor([0, 0, 0, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1])
    ids = torch.tensor(rng.permutation(np.arange(13) + 200))
    frozen = ops.balanced_folds(full, stable_ids=ids, classes=4)
    graph_only = dict(c, observed=torch.zeros_like(c['observed']))
    for fold in range(frozen['m']):
        keep = frozen['fold_contexts'][fold] >= 0
        original = torch.full_like(full, -1)
        original[keep] = full[keep]
        poisoned_full = full.clone()
        hidden_ids = torch.cat((frozen['held_nodes'][fold], frozen['unused_nodes']))
        poisoned_full[hidden_ids] = (poisoned_full[hidden_ids] + 1) % 4
        poisoned_input = torch.full_like(full, -1)
        poisoned_input[keep] = poisoned_full[keep]
        close(original, frozen['fold_contexts'][fold], 0)
        close(poisoned_input, original, 0)
        before, _ = branch(ops, dict(graph_only, context=original), aa, bb)
        after, _ = branch(ops, dict(graph_only, context=poisoned_input), aa, bb)
        errors[f'frozen_fold{fold}_held_and_extras_poison'] = close(after, before, 0)
    # Macro class mean must not silently become sample mean. Class3 absent.
    labels = np.array([0, 0, 0, 1, 2, 2])
    raw = rng.random((15, len(labels), 4))
    raw /= raw.sum(-1, keepdims=True)
    raw[0, 0, 0] = 0.
    raw[2, 3, 1] = 1e-45
    raw[3, 0] = [1., 0., 0., 0.]
    want_s, want_w = independent_scores(raw, labels)
    got_w, got_info = ops.score_weights(torch.tensor(raw, dtype=torch.float32), torch.tensor(labels))
    # Oracle consumes exactly the same raw float32 values as the helper.
    want_s, want_w = independent_scores(raw.astype(np.float32), labels)
    errors['float64_macro_clip_scores'] = close(got_info['scores'], want_s, 2e-13)
    errors['float64_macro_clip_weights'] = close(got_w, want_w, 2e-14)
    assert got_w.dtype == torch.float64 and got_info['scores'].dtype == torch.float64
    assert np.array_equal(as_np(got_info['represented_classes']), [0, 1, 2])
    errors['weights_entropy'] = close(got_info['entropy'], np.asarray(-(want_w * np.log(want_w)).sum()), 2e-13)
    errors['weights_ESS'] = close(got_info['ess'], np.asarray(1 / (want_w ** 2).sum()), 2e-12)
    errors['weights_KL_prior'] = close(got_info['prior_kl'], np.asarray((want_w * np.log(want_w * 15)).sum()), 2e-13)
    assert got_info['map_idx'] == int(want_s.argmax())
    tie_w, tie = ops.score_weights(torch.full((15, 3, 4), .25), torch.tensor([0, 1, 2]))
    assert tie['map_idx'] == 0
    errors['tie_prior_weights'] = close(tie_w, np.ones(15) / 15, 2e-15)
    return dict(errors=errors, balanced_CS=dict(original=55, selected=42, unused=13, m=3,
                                               represented=14, missing_class=9),
                poison_scope='frozen split; predictions before held truths enter score',
                uncertainty_scope='weight entropy/ESS are descriptive, not Bayesian guarantees')


def mixture_checks(ops, runner):
    rng = np.random.default_rng(54198)
    p = rng.random((15, 6, 4)).astype(np.float32)
    p /= p.sum(-1, keepdims=True)
    w = rng.dirichlet(np.ones(15))
    map_index = int(w.argmax())
    got = runner.derive_predictions(p, w, map_index=map_index)
    expected = dict(adaptive=np.einsum('g,gnk->nk', w, p.astype(float)),
                    prior=p.astype(float).mean(0), map=p[map_index],
                    fixed_edge=p[10], graph_off=p[4])
    assert set(got) == set(expected)
    errors = {name: close(got[name], value, 2e-7, name) for name, value in expected.items()}
    a, b = ops.candidate_table(dtype=torch.float64)
    expected_a = np.repeat([-math.log(2), 0., math.log(2), math.log(4), math.log(16)], 3)
    expected_b = np.tile([-2., 0., 2.], 5)
    errors['candidate_a_order'] = close(a, expected_a)
    errors['candidate_b_order'] = close(b, expected_b)
    info = ops.mixture_diagnostics(torch.tensor(w), a, b, torch.tensor(p))
    ma, mb = float(w @ expected_a), float(w @ expected_b)
    errors['weighted_a'] = close(info['a_mean'], np.asarray(ma))
    errors['weighted_b'] = close(info['b_mean'], np.asarray(mb))
    errors['weighted_a_variance'] = close(info['a_variance'], np.asarray(w @ ((expected_a - ma) ** 2)))
    errors['weighted_b_variance'] = close(info['b_variance'], np.asarray(w @ ((expected_b - mb) ** 2)))
    errors['weighted_covariance'] = close(info['ab_covariance'], np.asarray(w @ ((expected_a - ma) * (expected_b - mb))))
    pair_tv = np.mean([.5 * np.abs(p[i].astype(float) - p[j]).sum(-1).mean()
                       for i in range(15) for j in range(i + 1, 15)])
    errors['predictive_pairwise_TV'] = close(info['mean_pairwise_total_variation'], np.asarray(pair_tv))
    mix = expected['adaptive']
    tv = (w[:, None] * .5 * np.abs(p.astype(float) - mix).sum(-1)).sum(0).mean()
    errors['weighted_TV_to_mixture'] = close(info['weighted_mean_tv_to_mixture'], np.asarray(tv))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='tiny GPU invariance smoke only; no real bank')
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = [ROOT / name for name in ('checks.py', 'operators.py', 'run.py', 'protocol.json', 'PLAN_v1.md', 'PLAN_v2.md')]
    historical = [ROOT.parent / 'factor_nuisance_cycle' / name for name in ('operators.py', 'checks.py', 'PLAN_v2.md', 'protocol.json')]
    historical += [ROOT.parent / 'factor_projection_cycle' / name for name in ('operators.py', 'common.py')]
    paths += historical
    before = {str(path): sha(path) for path in paths}
    ops = load_module('fp03_checks_operators', ROOT / 'operators.py')
    runner = load_module('fp03_checks_runner', ROOT / 'run.py')
    start = time.perf_counter()
    result = dict(status='RUNNING', cycle='FP03', optimizer_updates=0,
                  real_bank_used=False, real_predictions_used=False, final_bank_used=False,
                  tested_sources=before, historical_sources_read_only=[str(path) for path in historical],
                  scope='independent tiny CPU oracles and optional tiny GPU; no training or real predictions')
    result['oracles'] = oracle_checks(ops)
    print('ORACLES_PASS', flush=True)
    result['invariants_float64'] = invariant_checks(ops)
    result['invariants_float32'] = invariant_checks(ops, dtype=torch.float32)
    print('INVARIANTS_PASS', flush=True)
    result['batch_float64'] = batch_checks(ops)
    result['batch_float32'] = batch_checks(ops, dtype=torch.float32)
    print('BATCH_PASS', flush=True)
    result['folds_and_scores'] = fold_score_checks(ops)
    result['mixtures'] = mixture_checks(ops, runner)
    print('FOLDS_SCORES_MIXTURES_PASS', flush=True)
    result['gradients'] = gradient_checks(ops)
    print('GRADIENT_PASS', flush=True)
    if args.gpu:
        assert torch.cuda.is_available(), '--gpu requested but CUDA unavailable'
        result['gpu'] = dict(status='PASS', tiny_float32_invariants=invariant_checks(ops, device='cuda', dtype=torch.float32))
    else:
        result['gpu'] = dict(status='NOT_RUN', reason='optional tiny smoke not requested')
    assert before == {str(path): sha(path) for path in paths}, 'Tested source changed during checks; rerun after freeze'
    result.update(status='PASS', seconds=time.perf_counter() - start)
    path = ROOT / 'results' / 'checks.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    print('CHECKS_PASS', path, flush=True)


if __name__ == '__main__':
    main()

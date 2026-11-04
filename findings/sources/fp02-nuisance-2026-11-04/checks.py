"""Independent FP02 tiny-CPU numerical checks; no bank, fitting, or real data.

The dense NumPy oracle does not import any historical or new BP helper.
All-pair dense BP is exact only on the specified tree checks. Nonneighbor
mean-field and finite loopy BP differences are diagnostics, never equality
claims. A checks.json is written atomically only after all required checks pass.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows DLL import order.
import argparse
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import digamma, logsumexp
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_close(actual, expected, tolerance=2e-10, name='comparison'):
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    if torch.is_tensor(expected):
        expected = expected.detach().cpu().numpy()
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    error = float(np.max(np.abs(actual - expected), initial=0))
    assert error <= tolerance, (name, error, tolerance)
    return error


def literal_normal(x, grid, sigma):
    """Independent NumPy literal Normal table, including its normalizer."""
    return (-.5 * np.log(2 * np.pi) - np.log(sigma)[None, :, None]
            - .5 * ((x[:, :, None] - grid[None]) / sigma[None, :, None]) ** 2)


def priors(classes):
    mean = np.full((classes, classes), .4)
    np.fill_diagonal(mean, 1 + (classes - 1) * .6)
    shape, rate = np.full_like(mean, 5.), 5 / mean
    return shape, rate, digamma(shape) - np.log(rate), shape / rate


def exact_enumeration(adjacency, context, classes, observed, likelihood,
                      log_weights, theta=None, elog_b=None, mean_b=None,
                      log_population=None, pairs=None):
    """Exact whole finite-grid working score; integrate centroids per z.

    Defaults to every unordered graph pair, including observed nonedges.
    `pairs` is only for the independent edge-only/tree oracle.
    """
    a, ctx = np.asarray(adjacency), np.asarray(context)
    n, k = len(ctx), int(classes)
    obs, ll = np.asarray(observed), np.asarray(likelihood)
    f, r = obs.shape[1], ll.shape[-1]
    lw = np.broadcast_to(np.asarray(log_weights), (f, r)).copy()
    lw -= logsumexp(lw, axis=-1, keepdims=True)
    theta = np.ones(n) if theta is None else np.asarray(theta)
    if elog_b is None:
        _, _, elog_b, mean_b = priors(k)
    lp = np.full(k, -math.log(k)) if log_population is None else np.asarray(log_population)
    density = float(a.sum() / (n * (n - 1))) if n > 1 else 0.
    pairs = list(itertools.combinations(range(n), 2)) if pairs is None else list(pairs)
    assignments, scores = [], []
    for z in itertools.product(range(k), repeat=n):
        if any(ctx[i] >= 0 and z[i] != ctx[i] for i in range(n)):
            continue
        z = np.asarray(z)
        score = float(lp[z].sum())
        for i, j in pairs:
            score += a[i, j] * elog_b[z[i], z[j]] - density * theta[i] * theta[j] * mean_b[z[i], z[j]]
        for c in range(k):
            for col in range(f):
                ids = np.flatnonzero(obs[:, col] & (z == c))
                score += logsumexp(lw[col] + ll[ids, col].sum(axis=0))
        assignments.append(z)
        scores.append(score)
    assignments, scores = np.asarray(assignments), np.asarray(scores)
    log_z = float(logsumexp(scores))
    weights = np.exp(scores - log_z)
    out = np.zeros((n, k))
    for i in range(n):
        for c in range(k):
            out[i, c] = weights[assignments[:, i] == c].sum()
    return out, log_z


def dense_bp(adjacency, context, classes, observed, likelihood, log_weights,
             iterations=180, damping=.5, theta=None, elog_b=None, mean_b=None,
             log_population=None, pairs=None):
    """Independent generic pairwise log-BP over z and all class/feature grids.

    Each observed cell adds K class-gated z--centroid pair factors. This
    deliberately uses a generic factor graph, not the production scalar gates.
    """
    a, ctx = np.asarray(adjacency), np.asarray(context)
    n, k = len(ctx), int(classes)
    obs, ll = np.asarray(observed), np.asarray(likelihood)
    f, r = obs.shape[1], ll.shape[-1]
    lw = np.broadcast_to(np.asarray(log_weights), (f, r)).copy()
    lw -= logsumexp(lw, axis=-1, keepdims=True)
    theta = np.ones(n) if theta is None else np.asarray(theta)
    if elog_b is None:
        _, _, elog_b, mean_b = priors(k)
    lp = np.full(k, -math.log(k)) if log_population is None else np.asarray(log_population)
    unaries = [lp.copy() for _ in range(n)]
    for i in range(n):
        if ctx[i] >= 0:
            unaries[i][:] = -np.inf
            unaries[i][ctx[i]] = 0.
    for c in range(k):
        for col in range(f):
            unaries.append(lw[col].copy())
    density = float(a.sum() / (n * (n - 1))) if n > 1 else 0.
    pairs = list(itertools.combinations(range(n), 2)) if pairs is None else list(pairs)
    factors = []
    for i, j in pairs:
        factors.append((i, j, a[i, j] * elog_b - density * theta[i] * theta[j] * mean_b))
    for i, col in zip(*np.nonzero(obs)):
        for c in range(k):
            potential = np.zeros((k, r))
            potential[c] = ll[i, col]
            factors.append((i, n + c * f + col, potential))
    incoming = [[] for _ in unaries]
    for ei, (i, j, _) in enumerate(factors):
        incoming[i].append((ei, 1))
        incoming[j].append((ei, 0))
    messages = [[np.zeros(len(unaries[j])), np.zeros(len(unaries[i]))] for i, j, _ in factors]
    for _ in range(iterations):
        total = [u.copy() for u in unaries]
        for v, indices in enumerate(incoming):
            for ei, direction in indices:
                total[v] += messages[ei][direction]
        nxt = []
        for ei, (i, j, potential) in enumerate(factors):
            cavity_i = total[i] - messages[ei][1]
            cavity_j = total[j] - messages[ei][0]
            to_j = logsumexp(cavity_i[:, None] + potential, axis=0)
            to_i = logsumexp(cavity_j[None, :] + potential, axis=1)
            to_j -= to_j.mean()
            to_i -= to_i.mean()
            nxt.append([(1 - damping) * messages[ei][0] + damping * to_j,
                        (1 - damping) * messages[ei][1] + damping * to_i])
        messages = nxt
    total = [u.copy() for u in unaries]
    for v, indices in enumerate(incoming):
        for ei, direction in indices:
            total[v] += messages[ei][direction]
    return np.stack([np.exp(t - logsumexp(t)) for t in total[:n]])


def tiny_case(n=7, classes=3, columns=2, grid=5, seed=75321, device='cpu'):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = 1
    if n >= 5:
        a[0, 3] = a[3, 0] = 1
        a[1, 4] = a[4, 1] = 1
    ctx = np.full(n, -1, dtype=np.int64)
    ctx[0] = 0
    if n > 3:
        ctx[1] = 1
    obs = np.zeros((n, columns), dtype=bool)
    obs[2:min(5, n), :] = True
    x = rng.normal(size=(n, columns))
    points = np.linspace(-1.2, 1.3, grid)
    centroid_grid = np.stack([points + .11 * f for f in range(columns)])
    sigma = np.linspace(.55, .8, columns)
    ll = literal_normal(x, centroid_grid, sigma)
    lw = rng.normal(0, .13, (columns, grid))
    src, dst = np.nonzero(a)
    return dict(src=torch.tensor(src, device=device), dst=torch.tensor(dst, device=device), nodes=n,
                context=torch.tensor(ctx, device=device), classes=classes,
                observed=torch.tensor(obs, device=device),
                log_likelihood=torch.tensor(ll, dtype=torch.float64, device=device),
                log_weights=torch.tensor(lw, dtype=torch.float64, device=device))


def as_numpy(case):
    n = case['nodes']
    a = np.zeros((n, n), dtype=np.int64)
    a[case['src'].cpu().numpy(), case['dst'].cpu().numpy()] = 1
    return dict(adjacency=a, context=case['context'].cpu().numpy(), classes=case['classes'],
                observed=case['observed'].cpu().numpy(),
                likelihood=case['log_likelihood'].detach().cpu().numpy(),
                log_weights=case['log_weights'].detach().cpu().numpy())


def call(ops, case, variant='dc_update', **kwargs):
    p, info = ops.nuisance_bp(**case, variant=variant, details=True, **kwargs)
    assert p.shape == (case['nodes'], case['classes'])
    assert torch.isfinite(p).all() and bool((p >= 0).all())
    assert_close(p.sum(-1), np.ones(case['nodes']), 2e-10, 'normalized prediction')
    return p, info


def oracle_checks(ops):
    errors, diagnostics = {}, []
    # Full Poisson N=2 is an actual factor tree, even with one observed row:
    # each class/column centroid is a leaf. Both literal constants are kept.
    for classes in (2, 3):
        c = tiny_case(n=2, classes=classes, columns=2)
        c['observed'][0] = True
        c['context'][:] = -1
        c['context'][1] = classes - 1
        args = as_numpy(c)
        exact, _ = exact_enumeration(**args)
        dense = dense_bp(**args)
        public, _ = call(ops, c, 'poisson_fixed', iterations=180)
        errors[f'full_poisson_tree_dense_K{classes}'] = assert_close(dense, exact, 2e-11)
        errors[f'full_poisson_tree_operator_K{classes}'] = assert_close(public, exact, 2e-9)
    # Independent edge-only tree oracle has three nodes and two full edge
    # factors. Nonedges intentionally omitted only in this oracle check.
    c = tiny_case(n=3, classes=3, columns=1)
    c['observed'][:] = False
    args = as_numpy(c)
    edge_pairs = [(0, 1), (1, 2)]
    exact, _ = exact_enumeration(**args, pairs=edge_pairs)
    dense = dense_bp(**args, pairs=edge_pairs)
    errors['edge_only_tree_dense'] = assert_close(dense, exact, 2e-11)
    for n in (3, 4, 5):
        c = tiny_case(n=n, classes=2, columns=1)
        # Multi-source creates true feature loops in addition to graph loops.
        c['observed'][:] = False
        c['observed'][1:min(n, 3)] = True
        args = as_numpy(c)
        exact, logz = exact_enumeration(**args)
        dense = dense_bp(**args, iterations=180)
        sparse, _ = call(ops, c, 'poisson_fixed', iterations=180)
        diagnostics.append(dict(nodes=n, dense_vs_exact=float(np.max(np.abs(dense - exact))),
                                sparse_vs_exact=float(np.max(np.abs(sparse.detach().numpy() - exact))),
                                sparse_vs_dense=float(np.max(np.abs(sparse.detach().numpy() - dense))),
                                exact_log_partition=logz,
                                scope='diagnostic: finite loopy BP and nonneighbor MF are approximations'))
    return dict(errors=errors, approximation_diagnostics=diagnostics)


def invariant_checks(ops, device='cpu'):
    result = {}
    for variant in ('poisson_fixed', 'sbm_update', 'dc_update'):
        c = tiny_case(device=device)
        p, _ = call(ops, c, variant)
        errs = {}
        known = c['context'] >= 0
        one = torch.nn.functional.one_hot(c['context'][known], c['classes']).to(p)
        errs['context'] = assert_close(p[known], one)
        poisoned = dict(c, log_likelihood=c['log_likelihood'].clone())
        poisoned['log_likelihood'][~c['observed']] = float('nan')
        pp, _ = call(ops, poisoned, variant)
        errs['hidden_nan'] = assert_close(pp, p, 3e-9)
        poisoned['log_likelihood'][~c['observed']] = 1e100
        pp, _ = call(ops, poisoned, variant)
        errs['hidden_large'] = assert_close(pp, p, 3e-9)
        columns = torch.tensor([1, 0], device=device)
        cp = dict(c, observed=c['observed'][:, columns],
                  log_likelihood=c['log_likelihood'][:, columns], log_weights=c['log_weights'][columns])
        pp, _ = call(ops, cp, variant)
        errs['columns_with_prior'] = assert_close(pp, p, 3e-9)
        cp = dict(c, log_weights=c['log_weights'] + c['log_weights'].new_tensor([[3.], [-2.]]))
        pp, _ = call(ops, cp, variant)
        errs['prior_normalization_gauge'] = assert_close(pp, p, 3e-9)
        shared = c['log_weights'][0]
        one_prior, _ = call(ops, dict(c, log_weights=shared), variant)
        repeated_prior, _ = call(ops, dict(c, log_weights=shared[None].expand(2, -1)), variant)
        errs['shared_prior_shape'] = assert_close(one_prior, repeated_prior, 3e-9)
        classes = torch.tensor([2, 0, 1], device=device)
        context = c['context'].clone()
        context[known] = classes[context[known]]
        pp, _ = call(ops, dict(c, context=context), variant)
        errs['classes'] = assert_close(pp[:, classes], p, 3e-9)
        order = torch.tensor([5, 2, 0, 6, 1, 4, 3], device=device)
        inverse = torch.argsort(order)
        cp = dict(c, src=inverse[c['src']], dst=inverse[c['dst']], context=c['context'][order],
                  observed=c['observed'][order], log_likelihood=c['log_likelihood'][order])
        pp, _ = call(ops, cp, variant)
        errs['nodes'] = assert_close(pp, p[order], 3e-9)
        # Explicitly permute only source rows and the corresponding graph IDs.
        order = torch.tensor([0, 1, 4, 2, 3, 5, 6], device=device)
        inverse = torch.argsort(order)
        cp = dict(c, src=inverse[c['src']], dst=inverse[c['dst']], context=c['context'][order],
                  observed=c['observed'][order], log_likelihood=c['log_likelihood'][order])
        pp, _ = call(ops, cp, variant)
        errs['source_rows'] = assert_close(pp, p[order], 3e-9)
        edge_order = torch.arange(len(c['src']) - 1, -1, -1, device=device)
        pp, _ = call(ops, dict(c, src=c['src'][edge_order], dst=c['dst'][edge_order]), variant)
        errs['directed_edge_order'] = assert_close(pp, p, 3e-9)
        c0 = dict(c, observed=torch.zeros_like(c['observed']))
        p0, _ = call(ops, c0, variant)
        c1 = dict(c, observed=torch.zeros_like(c['observed']))
        c1['observed'][3] = True
        p1, _ = call(ops, c1, variant)
        errs['singleton_vs_no_source_full_outer'] = assert_close(p1, p0, 3e-9)
        scattered = dict(c, observed=torch.zeros_like(c['observed']))
        scattered['observed'][2, 0] = True
        scattered['observed'][4, 1] = True
        psingle, _ = call(ops, scattered, variant)
        errs['one_source_per_column_different_rows'] = assert_close(psingle, p0, 3e-9)
        empty = dict(c0, src=c['src'][:0], dst=c['dst'][:0])
        pe, _ = call(ops, empty, variant)
        expected = np.full((c['nodes'], c['classes']), 1 / c['classes'])
        expected[known.cpu()] = one.detach().cpu().numpy()
        errs['empty_edges'] = assert_close(pe, expected, 3e-9)
        # A one-node graph exercises n-1=0 separately from merely E=0.
        singleton = dict(empty, nodes=1, context=c['context'].new_tensor([-1]),
                         observed=c['observed'][:1].clone().fill_(False),
                         log_likelihood=c['log_likelihood'][:1])
        ps, _ = call(ops, singleton, variant)
        errs['single_node_no_division_zero'] = assert_close(ps, np.full((1, c['classes']), 1 / c['classes']), 3e-9)
        initial, initial_info = call(ops, c, variant, iterations=0)
        expected_initial = initial_info['field'].softmax(-1)
        expected_initial[known] = one
        errs['zero_steps_field_readout_consistency'] = assert_close(initial, expected_initial, 2e-12)
        assert initial_info['nuisance_updates'] == 0 and not initial_info['update_history']
        result[variant] = errs
    signature = inspect.signature(ops.nuisance_bp)
    assert not any(name in signature.parameters for name in ('y', 'labels', 'query_labels', 'truth', 'class_sizes'))
    result['query_truth'] = dict(status='PASS', boundary='explicit input whitelist; only context labels supplied',
                                signature=str(signature))
    return result


def gradient_checks(ops):
    outputs = {}
    c = tiny_case(n=7, columns=2, grid=5)
    rng = np.random.default_rng(2874)
    direction = torch.tensor(rng.normal(size=c['log_likelihood'].shape), dtype=torch.float64)
    direction = direction * c['observed'][..., None]
    for variant in ('poisson_fixed', 'sbm_update', 'dc_update'):
        ll = c['log_likelihood'].detach().clone().requires_grad_()
        def objective(table):
            p, _ = call(ops, dict(c, log_likelihood=table), variant)
            return -p[-1, 2].log()
        value = objective(ll)
        gradient = torch.autograd.grad(value, ll)[0]
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 1e-10, (variant, 'missing query lost source gradient')
        assert_close(gradient[~c['observed']], np.zeros(tuple(gradient[~c['observed']].shape)), 1e-13)
        analytic = float((gradient * direction).sum())
        eps = 2e-5
        finite = float((objective(ll.detach() + eps * direction) - objective(ll.detach() - eps * direction)) / (2 * eps))
        error = abs(analytic - finite)
        assert error < 3e-7 + 2e-5 * abs(finite), (variant, analytic, finite, error)
        outputs[variant] = dict(autograd=analytic, central_difference=finite, error=error,
                                abs_gradient=float(gradient.abs().sum()), query_has_no_observed_features=True)
        ll0 = c['log_likelihood'].detach().clone().requires_grad_()
        p0, _ = call(ops, dict(c, observed=torch.zeros_like(c['observed']), log_likelihood=ll0), variant)
        zero_gradient = torch.autograd.grad(-p0[-1, 2].log(), ll0)[0]
        outputs[variant]['source0_zero_gradient_error'] = assert_close(zero_gradient, np.zeros(tuple(ll0.shape)), 1e-15)
    return outputs


def statistic_checks(ops):
    """Direct i<j loops independent of production matrix/volume formulas."""
    rng = np.random.default_rng(560720)
    n, k = 5, 3
    edge_ids = [(0, 1), (0, 3), (1, 2), (2, 4), (3, 4)]
    q = rng.dirichlet(np.array([.4, 1.2, .7]), size=n)
    context = np.array([0, -1, -1, 2, -1])
    q[0], q[3] = [1, 0, 0], [0, 0, 1]
    theta = np.array([.2, .7, 2.4, 1.2, 3.1])
    b = np.zeros((len(edge_ids), k, k))
    for ei, (i, j) in enumerate(edge_ids):
        pair = rng.gamma(.7, 1., (k, k))
        if context[i] >= 0:
            pair[np.arange(k) != context[i]] = 0
        if context[j] >= 0:
            pair[:, np.arange(k) != context[j]] = 0
        b[ei] = pair / pair.sum()
    rate = rng.uniform(.3, 1.5, (k, k))
    rate = (rate + rate.T) / 2
    counts, exposure, interaction = np.zeros((k, k)), np.zeros((k, k)), np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            pair = b[edge_ids.index((i, j))] if (i, j) in edge_ids else np.outer(q[i], q[j])
            membership = pair + pair.T
            np.fill_diagonal(membership, np.diag(pair))
            if (i, j) in edge_ids:
                counts += membership
            exposure += theta[i] * theta[j] * membership
            scalar = float((pair * rate).sum())
            interaction[i] += theta[j] * scalar
            interaction[j] += theta[i] * scalar
    tensor = lambda v: torch.tensor(v, dtype=torch.float64)
    src = torch.tensor([i for i, _ in edge_ids])
    dst = torch.tensor([j for _, j in edge_ids])
    got = ops.gamma_statistics(src, dst, tensor(q), tensor(b), tensor(theta), tensor(rate), torch.tensor(context))
    expected = dict(edge_count=counts, pair_exposure=exposure,
                    population_count=q[context < 0].sum(0), theta_interaction=interaction,
                    class_volume=(theta[:, None] * q).sum(0))
    errors = {name: assert_close(got[name], value, 2e-11, name) for name, value in expected.items()}
    errors['unordered_count_conservation'] = assert_close(np.triu(counts).sum(), np.asarray(float(len(edge_ids))))
    errors['unordered_exposure_conservation'] = assert_close(np.triu(exposure).sum(),
        np.asarray(sum(theta[i] * theta[j] for i in range(n) for j in range(i + 1, n))))
    errors['population_count_conservation'] = assert_close(expected['population_count'].sum(), np.asarray(3.))
    errors['degree_volume_conservation'] = assert_close(expected['class_volume'].sum(), np.asarray(theta.sum()))
    errors['theta_pair_exposure_identity'] = assert_close(np.asarray(theta @ interaction),
        np.asarray(2 * np.triu(exposure * rate).sum()), 3e-11)
    assert np.max(np.abs(expected['population_count'] - expected['class_volume'])) > .5
    # Uniform-population and high-volume nodes must remain different quantities.
    assert not np.allclose(expected['population_count'], expected['class_volume'])
    c = tiny_case()
    _, fixed = call(ops, c, 'poisson_fixed')
    s0, r0, l0, m0 = priors(c['classes'])
    errors['gamma_initial_shape'] = assert_close(fixed['affinity_shape'], s0)
    errors['gamma_initial_rate'] = assert_close(fixed['affinity_rate'], r0)
    errors['gamma_Elog_not_logE'] = assert_close(fixed['affinity_meanlog'], l0)
    errors['gamma_mean'] = assert_close(fixed['affinity_mean'], m0)
    assert np.max(np.abs(l0 - np.log(m0))) > .05
    errors['theta_initial_mean'] = assert_close(fixed['theta_mean'], np.ones(c['nodes']))
    errors['population_initial_mean'] = assert_close(fixed['population_mean'], np.full(c['classes'], 1 / c['classes']))
    # Further helper/history assertions are installed once the implementation
    # exposes its specified field-state and nuisance-update audit records.
    history = nuisance_history_checks(ops)
    return dict(brute_force_errors=errors, nuisance_history=history)


def nuisance_history_checks(ops):
    errors, histories = {}, {}
    c = tiny_case()
    n, k = c['nodes'], c['classes']
    a = as_numpy(c)['adjacency']
    degree = a.sum(-1).astype(float)
    density = a.mean() * n / (n - 1)
    s0, r0, _, _ = priors(k)
    for variant in ('poisson_fixed', 'sbm_update', 'dc_update'):
        p, info = call(ops, c, variant)
        history = info['update_history']
        expected_steps = [] if variant == 'poisson_fixed' else [6, 12, 18]
        assert [e['after_steps'] for e in history] == expected_steps
        assert info['nuisance_updates'] == len(expected_steps)
        previous = dict(affinity_shape=s0, affinity_rate=r0, theta_shape=np.ones(n),
                        theta_rate=np.ones(n), population_alpha=np.ones(k))
        records = []
        for entry in history:
            for key, value in previous.items():
                errors[f'{variant}_step{entry["after_steps"]}_before_{key}'] = assert_close(entry['before'][key], value)
            stats = entry['statistics']
            proposal = dict(affinity_shape=s0 + stats['edge_count'].detach().numpy(),
                            affinity_rate=r0 + density * stats['pair_exposure'].detach().numpy(),
                            theta_shape=1 + degree,
                            theta_rate=1 + density * stats['theta_interaction'].detach().numpy(),
                            population_alpha=1 + stats['population_count'].detach().numpy())
            for key, value in proposal.items():
                errors[f'{variant}_step{entry["after_steps"]}_proposal_{key}'] = assert_close(entry['proposal'][key], value, 3e-11)
                expected_after = (.5 * previous[key] + .5 * value)
                if variant == 'sbm_update' and key.startswith('theta_'):
                    expected_after = previous[key]
                errors[f'{variant}_step{entry["after_steps"]}_after_{key}'] = assert_close(entry['after'][key], expected_after, 3e-11)
                previous[key] = expected_after
            before_theta = (entry['before']['theta_shape'] / entry['before']['theta_rate']).detach().numpy()
            volume_total = float(stats['class_volume'].sum())
            population_total = float(stats['population_count'].sum())
            errors[f'{variant}_step{entry["after_steps"]}_nodecount'] = assert_close(np.asarray(population_total),
                                                                                           np.asarray(float((c['context'] < 0).sum())))
            errors[f'{variant}_step{entry["after_steps"]}_volume'] = assert_close(np.asarray(volume_total), np.asarray(before_theta.sum()))
            errors[f'{variant}_step{entry["after_steps"]}_edge_count'] = assert_close(stats['edge_count'].triu().sum(), np.asarray(float(a.sum() / 2)))
            pair_total = sum(before_theta[i] * before_theta[j] for i in range(n) for j in range(i + 1, n))
            errors[f'{variant}_step{entry["after_steps"]}_pair_exposure'] = assert_close(stats['pair_exposure'].triu().sum(), np.asarray(pair_total), 5e-11)
            records.append(dict(after_steps=entry['after_steps'], node_count=population_total,
                                theta_volume=volume_total, prior_readded_once=True))
        for key, value in previous.items():
            errors[f'{variant}_last_used_{key}'] = assert_close(info[key], value, 3e-11)
        if variant == 'dc_update':
            errors['theta_shape_closed_form_three_updates'] = assert_close(info['theta_shape'], 1 + (1 - .5 ** 3) * degree)
        errors[f'{variant}_gamma_mean_current'] = assert_close(info['affinity_mean'],
            info['affinity_shape'] / info['affinity_rate'])
        expected_log = digamma(info['affinity_shape'].detach().numpy()) - np.log(info['affinity_rate'].detach().numpy())
        errors[f'{variant}_gamma_meanlog_current'] = assert_close(info['affinity_meanlog'], expected_log)
        alpha = info['population_alpha'].detach().numpy()
        errors[f'{variant}_population_Elog'] = assert_close(info['population_unary'], digamma(alpha) - digamma(alpha.sum()))
        errors[f'{variant}_population_mean'] = assert_close(info['population_mean'], alpha / alpha.sum())
        errors[f'{variant}_theta_mean'] = assert_close(info['theta_mean'], info['theta_shape'] / info['theta_rate'])
        errors[f'{variant}_density'] = assert_close(info['density_offset'], np.asarray(density))
        # Independent nonneighbor sum uses precisely the q/theta state that
        # generated the saved field, not a newly recomputed final MF iterate.
        q = info['field_node_belief'].detach().numpy()
        theta = info['field_theta_mean'].detach().numpy()
        mean = info['affinity_mean'].detach().numpy()
        volume = np.zeros((n, k))
        penalty = np.zeros((n, k))
        for i in range(n):
            for j in range(n):
                if i != j and not a[i, j]:
                    volume[i] += theta[j] * q[j]
                    penalty[i] += density * theta[i] * theta[j] * (mean @ q[j])
        errors[f'{variant}_nonneighbor_volume_direct'] = assert_close(info['nonneighbor_volume'], volume, 3e-11)
        errors[f'{variant}_nonneighbor_field_direct'] = assert_close(info['nonneighbor_field'], penalty, 3e-11)
        field = np.broadcast_to(info['population_unary'].detach().numpy(), (n, k)).copy() - penalty
        messages = info['graph_messages'].detach().numpy()
        src, dst = c['src'].numpy(), c['dst'].numpy()
        for ei, j in enumerate(dst):
            field[j] += messages[ei]
        for ei, i in enumerate(info['observed_nodes'].numpy()):
            field[i] += info['to_node'][ei].detach().numpy()
        errors[f'{variant}_field_no_double_count'] = assert_close(info['field'], field, 3e-11)
        manual_node = np.exp(field - logsumexp(field, axis=-1, keepdims=True))
        known = c['context'].numpy() >= 0
        manual_node[known] = np.eye(k)[c['context'].numpy()[known]]
        errors[f'{variant}_readout_same_saved_field'] = assert_close(p, manual_node)
        reverse = info['reverse'].cpu().numpy()
        assert np.array_equal(src[reverse], dst) and np.array_equal(dst[reverse], src)
        pairs = []
        for eid in info['undirected_indices'].cpu().numpy():
            i, j = src[eid], dst[eid]
            left, right = field[i] - messages[reverse[eid]], field[j] - messages[eid]
            left, right = left - logsumexp(left), right - logsumexp(right)
            for v, local in ((i, left), (j, right)):
                if known[v]:
                    local[:] = -np.inf
                    local[c['context'][v]] = 0
            score = (left[:, None] + right[None] + expected_log
                     - density * theta[i] * theta[j] * mean)
            pairs.append(np.exp(score - logsumexp(score)))
        errors[f'{variant}_full_edge_pair_contraction'] = assert_close(info['edge_belief'], np.asarray(pairs), 3e-11)
        histories[variant] = records
    # The graph update itself is tested away from a BP fixed point. Excluding
    # the opposite directional message and including the negative edge term
    # once must agree with an independent explicit matrix contraction.
    rng = np.random.default_rng(9562)
    src = torch.tensor([0, 1, 1, 2, 2, 3])
    dst = torch.tensor([1, 0, 2, 1, 3, 2])
    rev = torch.tensor([1, 0, 3, 2, 5, 4])
    context = torch.tensor([-1, 2, -1, 0])
    old = torch.tensor(rng.normal(0, .2, (6, 3)), dtype=torch.float64)
    field = torch.tensor(rng.normal(0, .3, (4, 3)), dtype=torch.float64)
    theta = torch.tensor([.4, 1.7, 2.3, .8], dtype=torch.float64)
    _, _, lp, mean = priors(3)
    got = ops._graph_proposal(src, dst, rev, old, field, context, theta,
                              torch.tensor(lp), torch.tensor(mean), .17, .5)
    want = []
    for ei, (i, j) in enumerate(zip(src.tolist(), dst.tolist())):
        local = (field[i] - old[rev[ei]]).numpy()
        local -= logsumexp(local)
        if context[i] >= 0:
            local[:] = -np.inf
            local[context[i]] = 0
        potential = lp - .17 * float(theta[i] * theta[j]) * mean
        proposal = logsumexp(local[:, None] + potential, axis=0)
        proposal -= proposal.mean()
        want.append(.5 * old[ei].numpy() + .5 * proposal)
    errors['single_graph_cavity_full_poisson_update'] = assert_close(got, np.asarray(want), 2e-11)
    complete_src, complete_dst = np.where(~np.eye(4, dtype=bool))
    q = torch.tensor(rng.dirichlet(np.ones(3), size=4), dtype=torch.float64)
    penalty, volume, minimum = ops.nonneighbor_field(torch.tensor(complete_src), torch.tensor(complete_dst), q,
                                                    theta, torch.tensor(mean), .17)
    errors['complete_graph_no_external_nonedge_field'] = assert_close(penalty, np.zeros((4, 3)), 2e-14)
    errors['complete_graph_no_external_volume'] = assert_close(volume, np.zeros((4, 3)), 2e-14)
    assert float(minimum) >= -1e-12
    return dict(errors=errors, updates=histories,
                scope='fresh original-prior proposals; 0.5 damping; 3 updates; direct fields and edge cavities')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='optional tiny GPU invariance smoke, never real graph data')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import operators as ops
    assert Path(ops.__file__).resolve() == ROOT / 'operators.py', ops.__file__
    source_paths = [ROOT / 'checks.py', ROOT / 'operators.py', ROOT / 'PLAN_v2.md']
    historical_paths = [ROOT.parent / 'factor_projection_cycle' / name
                        for name in ('operators.py', 'common.py', 'PLAN_v2.md', 'DECISION_KO.md')]
    source_paths += historical_paths
    before = {str(p): sha(p) for p in source_paths}
    start = time.perf_counter()
    result = dict(status='RUNNING', scope='independent small CPU checks; no real bank, optimizer, training, or predictions',
                  optimizer_updates=0, real_bank_used=False, real_predictions_used=False,
                  tested_sources=before, historical_files_read_only=[str(p) for p in historical_paths])
    result['oracle'] = oracle_checks(ops)
    print('ORACLE_PASS', flush=True)
    result['statistics'] = statistic_checks(ops)
    print('STATISTICS_PASS', flush=True)
    result['invariance'] = invariant_checks(ops)
    print('INVARIANCE_PASS', flush=True)
    result['gradients'] = gradient_checks(ops)
    print('GRADIENT_PASS', flush=True)
    if args.gpu:
        assert torch.cuda.is_available(), '--gpu requested but CUDA unavailable'
        result['gpu'] = dict(status='PASS', tiny_invariance=invariant_checks(ops, 'cuda'))
    else:
        result['gpu'] = dict(status='NOT_RUN', reason='optional tiny GPU smoke not requested')
    after = {str(p): sha(p) for p in source_paths}
    assert before == after, 'Tested source changed during numerical checks; rerun after edits settle'
    result.update(status='PASS', seconds=time.perf_counter() - start)
    path = ROOT / 'results' / 'checks.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)
    print('CHECKS_PASS', path, flush=True)


if __name__ == '__main__':
    main()

"""Independent FP04 tiny checks. Root executes this CLI; no banks or fitting.

The contraction oracle uses literal Python/NumPy sums, not production
scatter/einsum. Pointwise learned modules may be shared by the differentiable
reference; their algebra is not a second independently learned model.
"""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows DLL import order, before importing torch.
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def close(actual, expected, tolerance=2e-9, name='comparison'):
    actual, expected = numpy(actual), numpy(expected)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
    error = float(np.max(np.abs(actual - expected), initial=0.))
    assert error <= tolerance, (name, error, tolerance)
    return error


def clone_batch(batch):
    return {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in batch.items()}


def load_model():
    spec = importlib.util.spec_from_file_location('fp04_independent_model', ROOT / 'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_batch(device='cpu', dtype=torch.float64, complete=False):
    rng = np.random.default_rng(47047103)
    n, f, k = 7, 5, 3
    mask = np.ones((n, f), bool) if complete else np.array([
        [1, 0, 1, 0, 1], [0, 1, 0, 0, 0], [1, 1, 0, 1, 0],
        [0, 0, 1, 1, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1]], dtype=bool)
    raw = rng.normal(size=(n, f))
    raw[0, 0] = 0.  # A real measured zero must remain an included cell.
    x = np.where(mask, raw, 0.)
    imputed = np.where(mask, raw, rng.normal(size=(n, f)))
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (4, 5)]
    src = [i for i, j in pairs] + [j for i, j in pairs]
    dst = [j for i, j in pairs] + [i for i, j in pairs]
    tensor = lambda value, dt: torch.tensor(value, dtype=dt, device=device)
    return dict(x=tensor(x, dtype), observed=tensor(mask, torch.bool),
                pcfi=tensor(imputed, dtype), src=tensor(src, torch.long),
                dst=tensor(dst, torch.long),
                context=tensor([0, -1, 1, -1, -1, -1, 0], torch.long), classes=k)


def new_model(module, device='cpu', dtype=torch.float64):
    torch.manual_seed(47047121)
    # Test the locked architecture, not a reduced width/round surrogate.
    return module.ObservedCellPFN(width=32, rank=4, rounds=4).to(device=device, dtype=dtype).eval()


def run(model, batch, arm='observed', **kwargs):
    out = model(batch, arm=arm, **kwargs)
    assert out.shape == (len(batch['context']), batch['classes']), out.shape
    assert torch.isfinite(out).all(), 'nonfinite logits'
    return out


def objective(logits):
    coefficients = torch.linspace(-.83, 1.17, logits.numel(), device=logits.device,
                                  dtype=logits.dtype).reshape_as(logits)
    return (coefficients * logits).sum() + .031 * logits.square().sum()


def value_and_grad(model, batch, arm='observed', call=None, **kwargs):
    batch = clone_batch(batch)
    x = batch['x'].detach().requires_grad_(True)
    batch['x'] = x
    if bool(batch['observed'].all()):
        # Same effective input leaf in all three fully observed arms.
        batch['pcfi'] = x
    out = run(model, batch, arm=arm, **kwargs) if call is None else call(model, batch, arm)
    params = list(model.named_parameters())
    grad = torch.autograd.grad(objective(out), [p for _, p in params] + [x], allow_unused=True)
    named = {name: (torch.zeros_like(p) if g is None else g)
             for (name, p), g in zip(params, grad[:-1])}
    named['INPUT.x'] = torch.zeros_like(x) if grad[-1] is None else grad[-1]
    for name, value in named.items():
        assert torch.isfinite(value).all(), ('nonfinite gradient', name)
    return out.detach(), {name: value.detach() for name, value in named.items()}


def gradient_close(first, second, tolerance, name):
    assert first.keys() == second.keys()
    return {key: close(first[key], second[key], tolerance, f'{name}:{key}') for key in first}


def numpy_contraction(node, basis, projection, reverse_projection, support, column_state=None):
    """Literal class/column/node/rank loops; no contraction routine used.

    Returns the pre-column-nonlinearity and reverse pre-read reductions.
    Reverse receives a deterministic stand-in column state; it tests the
    sum/count algebra independently of learned pointwise transformations.
    """
    node, basis = np.asarray(node), np.asarray(basis)
    projection, reverse_projection = np.asarray(projection), np.asarray(reverse_projection)
    support = np.asarray(support, dtype=bool)
    c, n, w = node.shape
    f, r = basis.shape[1:]
    column = np.zeros((c, f, w), dtype=node.dtype)
    for ci in range(c):
        for fi in range(f):
            count = int(support[:, fi].sum())
            for ni in range(n):
                if support[ni, fi]:
                    for ri in range(r):
                        column[ci, fi] += basis[ni, fi, ri] * (node[ci, ni] @ projection[ri].T)
            column[ci, fi] /= max(count, 1)
    read_column = column if column_state is None else np.asarray(column_state)
    reverse = np.zeros((c, n, w), dtype=node.dtype)
    for ci in range(c):
        for ni in range(n):
            count = int(support[ni].sum())
            for fi in range(f):
                if support[ni, fi]:
                    for ri in range(r):
                        reverse[ci, ni] += basis[ni, fi, ri] * (read_column[ci, fi] @ reverse_projection[ri].T)
            reverse[ci, ni] /= max(count, 1)
    return column, reverse


def numpy_oracle_selfcheck():
    rng = np.random.default_rng(47047161)
    node, basis = rng.normal(size=(3, 4, 2)), rng.normal(size=(4, 5, 4))
    p, v = rng.normal(size=(4, 2, 2)), rng.normal(size=(4, 2, 2))
    support = rng.random((4, 5)) < .45
    support[:, -1] = False
    support[-1] = False
    col, rev = numpy_contraction(node, basis, p, v, support)
    weight = basis * support[..., None]
    direct_col = np.einsum('nfr,cnw,rvw->cfv', weight, node, p)
    direct_col /= np.maximum(support.sum(0), 1)[None, :, None]
    direct_rev = np.einsum('nfr,cfw,rvw->cnv', weight, direct_col, v)
    direct_rev /= np.maximum(support.sum(1), 1)[None, :, None]
    return dict(column=close(col, direct_col, 2e-13), reverse=close(rev, direct_rev, 2e-13))


def reference_forward(model, batch, arm):
    """Differentiable reference: literal cell/edge sums, shared local modules.

    Does not call any production private helper, scatter, einsum, or matmul
    reduction. Sharing nn.Linear/MLP/LayerNorm parameters keeps the function
    and gradients comparable without duplicating learned state dictionaries.
    """
    raw, observed, context = batch['x'], batch['observed'], batch['context']
    x = torch.where(observed, raw, torch.zeros_like(raw))
    n, f = x.shape
    k = batch['classes']
    src, dst = batch['src'].tolist(), batch['dst'].tolist()
    degree = x.new_tensor([sum(j == i for j in dst) for i in range(n)])
    row_count = observed.sum(1).to(x.dtype)
    col_count = observed.sum(0).to(x.dtype)
    row_mean = torch.stack([sum((x[i, j] for j in range(f)), x.new_zeros(())) / max(int(row_count[i]), 1)
                            for i in range(n)])
    row_second = torch.stack([sum((x[i, j].square() for j in range(f)), x.new_zeros(())) / max(int(row_count[i]), 1)
                              for i in range(n)])
    col_mean = torch.stack([sum((x[i, j] for i in range(n)), x.new_zeros(())) / max(int(col_count[j]), 1)
                            for j in range(f)])
    col_second = torch.stack([sum((x[i, j].square() for i in range(n)), x.new_zeros(())) / max(int(col_count[j]), 1)
                              for j in range(f)])
    column_stats = torch.stack((col_mean.tanh(), col_second.tanh(), col_count.log1p()/np.log1p(n)), -1)
    node_base = torch.stack((row_mean.tanh(), row_second.tanh(), row_count.log1p()/np.log1p(f),
                            row_count/f, degree.log1p()/np.log1p(n)), -1)
    total_context = int((context >= 0).sum())
    h_list = []
    for c in range(k):
        count = int((context == c).sum())
        seeds = torch.tensor([0 if int(y) < 0 else 1 if int(y) == c else 2 for y in context],
                             device=x.device, dtype=torch.long)
        support = x.new_full((n, 1), count/max(total_context, 1))
        fraction = x.new_full((n, 1), total_context/n)
        h_list.append(model.seed_embedding(seeds) + model.node_mlp(torch.cat((node_base, support, fraction), -1)))
    h = torch.stack(h_list)
    values = torch.where(observed, x, batch['pcfi']) if arm == 'pcfi' else x
    support = observed if arm == 'observed' else torch.ones_like(observed)
    basis = []
    for i in range(n):
        cells = []
        for j in range(f):
            # Locked prefit amendment: all arm value-basis representations
            # saturate identically; observed-only statistics are unchanged.
            value = values[i, j].clamp(-10., 10.)
            inputs = torch.cat((torch.stack((value/10, value.abs()/10, value.square()/100,
                                            observed[i, j].to(x.dtype))), column_stats[j]))
            cells.append(model.value_mlp(inputs))
        basis.append(torch.stack(cells))
    basis = torch.stack(basis)
    for _ in range(model.rounds):
        neighbor = []
        for i in range(n):
            terms = [h[:, s] for s, d in zip(src, dst) if d == i]
            neighbor.append(sum(terms, torch.zeros_like(h[:, i])) / max(int(degree[i]), 1))
        neighbor = torch.stack(neighbor, 1)
        h = model.graph_norm(h + model.residual * model.graph_mlp(torch.cat((h, neighbor), -1)))
        projected = [layer(h) for layer in model.P]
        columns = []
        for j in range(f):
            terms = [basis[i, j, r] * projected[r][:, i]
                     for i in range(n) if bool(support[i, j]) for r in range(model.rank)]
            aggregate = sum(terms, torch.zeros_like(h[:, 0])) / max(int(support[:, j].sum()), 1)
            stats = column_stats[j].expand(k, -1)
            columns.append(model.column_norm(model.column_mlp(torch.cat((aggregate, stats), -1))))
        z = torch.stack(columns, 1)
        projected_z = [layer(z) for layer in model.V]
        reads = []
        for i in range(n):
            terms = [basis[i, j, r] * projected_z[r][:, j]
                     for j in range(f) if bool(support[i, j]) for r in range(model.rank)]
            reads.append(sum(terms, torch.zeros_like(h[:, i])) / max(int(support[i].sum()), 1))
        read = torch.stack(reads, 1)
        h = model.read_norm(h + model.residual * model.read_mlp(torch.cat((h, read), -1)))
    return model.head(h).squeeze(-1).T


def reference_checks(module):
    model, batch = new_model(module), tiny_batch()
    errors = {}
    for arm in ('observed', 'mask', 'pcfi'):
        expected, eg = value_and_grad(model, batch, arm, call=reference_forward)
        out, grad = value_and_grad(model, batch, arm, class_chunk=2, feature_chunk=2, sparse=True)
        errors[f'{arm}:full_loop_forward'] = close(out, expected, 3e-9)
        errors[f'{arm}:full_loop_gradient'] = gradient_close(grad, eg, 3e-8, f'{arm}:loop')
    # NumPy checks both production reduction strategies with arbitrary signed
    # rank weights, support holes, empty rows/columns and fixed learned P/V.
    rng = np.random.default_rng(47047201)
    h = rng.normal(size=(3, 7, model.width))
    b = rng.normal(size=(7, 5, model.rank))
    support = numpy(batch['observed']).copy()
    support[:, -1] = False
    p = np.stack([numpy(layer.weight) for layer in model.P])
    v = np.stack([numpy(layer.weight) for layer in model.V])
    stats = rng.normal(size=(5, 3))
    column, _ = numpy_contraction(h, b, p, v, support)
    stats_t = torch.tensor(stats).unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        z = model.column_norm(model.column_mlp(torch.cat((torch.tensor(column), stats_t), -1)))
    _, expected = numpy_contraction(h, b, p, v, support, numpy(z))
    rows, columns = np.nonzero(support)
    chunks = [dict(start=0, stop=5, basis=torch.tensor(b * support[..., None]))]
    sparse_chunks = [dict(start=0, stop=5, rows=torch.tensor(rows), columns=torch.tensor(columns),
                          basis=torch.tensor(b[rows, columns]))]
    for name, selected in (('dense', chunks), ('sparse', sparse_chunks)):
        got = model._column_read(torch.tensor(h), selected, torch.tensor(stats),
                                 torch.tensor(support.sum(1)), torch.tensor(support.sum(0)))
        errors[f'numpy_{name}_reduction'] = close(got, expected, 4e-12)
    return dict(errors=errors,
                independence='NumPy cell loops and full differentiable Python edge/cell sums; shared pointwise learned modules',
                limitation='not a separate implementation of GELU/LayerNorm/Linear library primitives')


def algebra_checks(module, dtype=torch.float64, device='cpu'):
    model, batch = new_model(module, device, dtype), tiny_batch(device, dtype)
    tol = 3e-9 if dtype == torch.float64 else 2e-5
    gradtol = 3e-8 if dtype == torch.float64 else 3e-4
    errors = {}
    for arm in ('observed', 'mask', 'pcfi'):
        base, bg = value_and_grad(model, batch, arm, class_chunk=1, feature_chunk=1, sparse=True)
        for name, options in {
            'dense_partial': dict(class_chunk=2, feature_chunk=2, sparse=False),
            'sparse_partial': dict(class_chunk=2, feature_chunk=3, sparse=True),
            'dense_complete': dict(class_chunk=17, feature_chunk=17, sparse=False),
        }.items():
            got, gg = value_and_grad(model, batch, arm, **options)
            errors[f'{arm}:{name}:forward'] = close(got, base, tol)
            errors[f'{arm}:{name}:gradient'] = gradient_close(gg, bg, gradtol, f'{arm}:{name}')
    full = tiny_batch(device, dtype, complete=True)
    expected, eg = value_and_grad(model, full, 'observed', class_chunk=2, feature_chunk=2)
    for arm in ('mask', 'pcfi'):
        out, grad = value_and_grad(model, full, arm, class_chunk=2, feature_chunk=2)
        errors[f'full:{arm}:forward'] = close(out, expected, tol)
        errors[f'full:{arm}:gradient'] = gradient_close(grad, eg, gradtol, f'full:{arm}')
    return errors


def invariant_checks(module, device='cpu', dtype=torch.float64):
    model, batch = new_model(module, device, dtype), tiny_batch(device, dtype)
    tolerance = 3e-9 if dtype == torch.float64 else 3e-5
    result = {}
    node_order = torch.tensor([4, 0, 6, 2, 5, 1, 3], device=device)
    inverse = torch.empty_like(node_order)
    inverse[node_order] = torch.arange(len(node_order), device=device)
    column_order = torch.tensor([3, 0, 4, 1, 2], device=device)
    class_map = torch.tensor([2, 0, 1], device=device)
    with torch.no_grad():
        for arm in ('observed', 'mask', 'pcfi'):
            base = run(model, batch, arm)
            permuted = clone_batch(batch)
            for key in ('x', 'observed', 'pcfi', 'context'):
                permuted[key] = permuted[key][node_order]
            permuted['src'], permuted['dst'] = inverse[batch['src']], inverse[batch['dst']]
            result[f'{arm}:node_permutation'] = close(run(model, permuted, arm), base[node_order], tolerance)
            permuted = clone_batch(batch)
            for key in ('x', 'observed', 'pcfi'):
                permuted[key] = permuted[key][:, column_order]
            result[f'{arm}:column_permutation'] = close(run(model, permuted, arm), base, tolerance)
            permuted = clone_batch(batch)
            selected = permuted['context'] >= 0
            permuted['context'][selected] = class_map[permuted['context'][selected]]
            result[f'{arm}:class_permutation'] = close(run(model, permuted, arm)[:, class_map], base, tolerance)
            poison = clone_batch(batch)
            poison['x'][~poison['observed']] = float('nan')
            # PCFI missing entries are legitimate visible-input derivatives;
            # they are deliberately NOT poisoned as if hidden ground truth.
            result[f'{arm}:hidden_NaN'] = close(run(model, poison, arm), base, 0.)
            poison = clone_batch(batch)
            poison['x'][~poison['observed']] = 9.123e29
            result[f'{arm}:hidden_finite_poison'] = close(run(model, poison, arm), base, 0.)
            if arm == 'pcfi':
                poison = clone_batch(batch)
                poison['pcfi'][poison['observed']] = float('nan')
                result['pcfi:observed_entries_clamped'] = close(run(model, poison, arm), base, 0.)
            poison = clone_batch(batch)
            poison['y'] = torch.arange(len(batch['context']), device=device) % batch['classes']
            poison['query_labels'] = torch.full_like(batch['context'], 927)
            result[f'{arm}:extraneous_target_fields_ignored'] = close(run(model, poison, arm), base, 0.)
            empty_context = clone_batch(batch)
            empty_context['context'].fill_(-1)
            out = run(model, empty_context, arm, class_chunk=2)
            result[f'{arm}:empty_context_symmetry'] = close(out, out[:, :1].expand_as(out), tolerance)
            result[f'{arm}:empty_context_uniform'] = close(out.softmax(-1), torch.full_like(out, 1/3), tolerance)
            absent = clone_batch(batch)
            absent['context'].fill_(-1)
            absent['context'][0] = 0
            out = run(model, absent, arm)
            result[f'{arm}:identical_unrepresented_classes'] = close(out[:, 1], out[:, 2], tolerance)
    return result


def boundary_checks(module, device='cpu', dtype=torch.float64):
    model = new_model(module, device, dtype)
    cases = {}
    for label in ('no_source', 'singleton', 'all_observed_zero', 'empty_graph', 'all_unknown'):
        batch = tiny_batch(device, dtype)
        if label in ('no_source', 'singleton'):
            batch['observed'].fill_(False)
            batch['x'].zero_()
            batch['pcfi'].zero_()
            if label == 'singleton':
                batch['observed'][2, 1] = True
                batch['x'][2, 1] = .75
                batch['pcfi'][2, 1] = .75
        elif label == 'all_observed_zero':
            batch['x'].zero_()
            batch['pcfi'].zero_()
        elif label == 'empty_graph':
            batch['src'], batch['dst'] = batch['src'][:0], batch['dst'][:0]
        else:
            batch['context'].fill_(-1)
            batch['observed'].fill_(False)
            batch['x'].zero_()
            batch['pcfi'].zero_()
            batch['src'], batch['dst'] = batch['src'][:0], batch['dst'][:0]
        cases[label] = {}
        for arm in ('observed', 'mask', 'pcfi'):
            logits, grads = value_and_grad(model, batch, arm, class_chunk=2, feature_chunk=2)
            cases[label][arm] = dict(max_abs_logit=float(logits.abs().max()),
                                    finite_parameter_gradients=True,
                                    max_abs_gradient=max(float(g.abs().max()) for g in grads.values()))
    # Measured zero changes support/count, even though its numerical value is 0.
    zero = tiny_batch(device, dtype)
    missing = clone_batch(zero)
    assert zero['observed'][0, 0] and zero['x'][0, 0] == 0
    missing['observed'][0, 0] = False
    observed_logits, missing_logits = run(model, zero), run(model, missing)
    difference = float((observed_logits - missing_logits).abs().max())
    assert difference > (1e-12 if dtype == torch.float64 else 1e-7), 'observed zero silently treated as missing'
    cases['measured_zero_versus_missing'] = dict(max_logit_difference=difference,
        input_count_difference=1, assertion='zero value is included; prediction need not be invariant')
    return cases


def representation_clip_checks(module, device='cpu', dtype=torch.float64):
    model, extreme = new_model(module, device, dtype), tiny_batch(device, dtype)
    hidden = ~extreme['observed']
    signs = torch.where(torch.arange(extreme['x'].numel(), device=device).reshape_as(hidden) % 2 == 0,
                        1., -1.).to(dtype=dtype)
    extreme['pcfi'][hidden] = (1234. * signs)[hidden]
    clipped = clone_batch(extreme)
    clipped['pcfi'] = clipped['pcfi'].clamp(-10., 10.)
    first, fg = value_and_grad(model, extreme, 'pcfi', class_chunk=2, feature_chunk=2)
    second, sg = value_and_grad(model, clipped, 'pcfi', class_chunk=2, feature_chunk=2)
    tolerance = 3e-9 if dtype == torch.float64 else 3e-5
    gradient_tolerance = 3e-8 if dtype == torch.float64 else 3e-4
    errors = dict(pcfi_extreme_clipped_forward=close(first, second, tolerance),
                  pcfi_extreme_clipped_gradient=gradient_close(fg, sg, gradient_tolerance, 'PCFI_CLIP'))
    if device == 'cpu' and dtype == torch.float64:
        reference, rg = value_and_grad(model, extreme, 'pcfi', call=reference_forward)
        errors['pcfi_extreme_reference_forward'] = close(first, reference, tolerance)
        errors['pcfi_extreme_reference_gradient'] = gradient_close(fg, rg, gradient_tolerance, 'PCFI_CLIP_LOOP')
    # Public basis helper itself must apply the same representation bound for
    # every caller/arm. This does not modify raw imputation or observed stats.
    values = torch.tensor([-1234., -10., 0., 10., 1234.], device=device, dtype=dtype)
    observed = torch.tensor([False, True, True, False, False], device=device)
    stats = torch.zeros((5, 3), device=device, dtype=dtype)
    got = model._basis_inputs(values, observed, stats)
    bounded = values.clamp(-10., 10.)
    expected = torch.cat((torch.stack((bounded/10, bounded.abs()/10, bounded.square()/100,
                                      observed.to(dtype)), -1), stats), -1)
    errors['common_basis_value_bounds'] = close(got, expected, 0.)
    return dict(errors=errors, raw_pcfi_extrema=[-1234., 1234.], represented_value_clip=10.,
                scope='representation only; input/raw cache remains unchanged')


def bridge_checks(module, device='cpu', dtype=torch.float64):
    model = new_model(module, device, dtype)
    batch = dict(x=torch.tensor([[0.], [.7], [-.8], [0.], [0.]], device=device, dtype=dtype),
                 observed=torch.tensor([[False], [True], [True], [False], [False]], device=device),
                 pcfi=torch.tensor([[0.], [.7], [-.8], [0.], [0.]], device=device, dtype=dtype),
                 src=torch.tensor([0, 1, 2, 3], device=device),
                 dst=torch.tensor([1, 0, 3, 2], device=device),
                 context=torch.tensor([0, -1, -1, -1, 1], device=device), classes=2)
    output = {}
    for disable in (False, True):
        case = clone_batch(batch)
        case['x'].requires_grad_(True)
        logits = run(model, case, disable_columns=disable)
        derivative = torch.autograd.grad(logits[3, 0] - logits[3, 1], case['x'],
                                         allow_unused=True)[0]
        derivative = torch.zeros_like(case['x']) if derivative is None else derivative
        assert torch.isfinite(derivative).all()
        cross = float(derivative[1, 0].abs())
        output['columns_disabled' if disable else 'columns_enabled'] = cross
        if disable:
            assert cross == 0., ('cross-component feature leak with columns off', cross)
        else:
            assert cross > (1e-13 if dtype == torch.float64 else 1e-9), ('bridge derivative missing', cross)
    # Query has no columns; deleting ALL graph edges removes its feature path.
    disconnected = clone_batch(batch)
    disconnected['src'] = disconnected['src'][:0]
    disconnected['dst'] = disconnected['dst'][:0]
    disconnected['x'].requires_grad_(True)
    out = run(model, disconnected)
    g = torch.autograd.grad(out[3, 0] - out[3, 1], disconnected['x'], allow_unused=True)[0]
    graph_off = 0. if g is None else float(g[1, 0].abs())
    assert graph_off == 0., ('feature reaches column-free query without graph', graph_off)
    output['all_graph_edges_removed'] = graph_off
    source_edge_cut = clone_batch(batch)
    source_edge_cut['src'] = source_edge_cut['src'][2:]
    source_edge_cut['dst'] = source_edge_cut['dst'][2:]
    source_edge_cut['x'].requires_grad_(True)
    out = run(model, source_edge_cut)
    g = torch.autograd.grad(out[3, 0] - out[3, 1], source_edge_cut['x'], allow_unused=True)[0]
    cut_derivative = 0. if g is None else float(g[1, 0].abs())
    cut_tol = 2e-13 if dtype == torch.float64 else 2e-7
    assert cut_derivative <= cut_tol, ('balanced source-edge cut breaks class symmetry', cut_derivative)
    # Known context nodes legitimately retain class-specific ternary seeds.
    # Only the unknown source/query nodes become class-symmetric after this cut.
    unknown = source_edge_cut['context'] < 0
    output['cut_unknown_class_symmetry_error'] = close(
        out[unknown, 0], out[unknown, 1],
        3e-10 if dtype == torch.float64 else 3e-6, 'balanced_cut_unknown_class_symmetry')
    output['context_to_source_edge_removed'] = cut_derivative
    output['static_context_counts'] = [1, 1]
    output['scope'] = 'implemented derivative route; not accuracy, causal identification, or novelty'
    return output


def module_gradient_checks(module):
    model, batch = new_model(module), tiny_batch()
    batch['x'].requires_grad_(True)
    logits = run(model, batch, class_chunk=2, feature_chunk=2)
    loss = torch.nn.functional.cross_entropy(logits[[3, 5]], torch.tensor([2, 1]))
    named = list(model.named_parameters())
    grads = torch.autograd.grad(loss, [parameter for _, parameter in named] + [batch['x']], allow_unused=True)
    groups = ('value_mlp', 'seed_embedding', 'node_mlp', 'graph_mlp', 'graph_norm',
              'P', 'column_mlp', 'column_norm', 'V', 'read_mlp', 'read_norm', 'head')
    norms = {}
    for group in groups:
        selected = [g for (name, _), g in zip(named, grads[:-1]) if name.split('.')[0] == group and g is not None]
        assert selected, ('no gradient tensors', group)
        assert all(torch.isfinite(g).all() for g in selected), group
        norm = float(sum(g.square().sum() for g in selected).sqrt())
        assert norm > 1e-14, ('missing module gradient in nondegenerate CE case', group, norm)
        norms[group] = norm
    xgrad = grads[-1]
    assert xgrad is not None and torch.isfinite(xgrad).all()
    assert float(xgrad[batch['observed']].abs().max()) > 1e-13
    hidden = close(xgrad[~batch['observed']], torch.zeros_like(xgrad[~batch['observed']]), 0., 'hidden_input_gradient_zero')
    return dict(ce=float(loss), parameter_group_l2=norms, hidden_gradient_max_error=hidden)


def cpu_gpu_checks(module):
    cpu, gpu = new_model(module, dtype=torch.float32), new_model(module, 'cuda', torch.float32)
    cb, gb = tiny_batch(dtype=torch.float32), tiny_batch('cuda', torch.float32)
    result = {}
    for arm in ('observed', 'mask', 'pcfi'):
        cp, cg = value_and_grad(cpu, cb, arm, class_chunk=2, feature_chunk=2)
        gp, gg = value_and_grad(gpu, gb, arm, class_chunk=2, feature_chunk=2)
        result[f'{arm}:forward'] = close(gp, cp, 8e-5)
        result[f'{arm}:gradient'] = gradient_close(gg, cg, 8e-4, f'{arm}:CPU_GPU')
    return result


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='also run tiny FP32 CUDA invariants/bridge, root only')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'checks.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = [ROOT / name for name in ('model.py', 'checks.py', 'PLAN_v2.md', 'protocol.json')]
    paths += [ROOT / name for name in ('data.py', 'run.py') if (ROOT / name).exists()]
    before = {str(path): sha(path) for path in paths}
    start = time.perf_counter()
    result = dict(status='RUNNING', cycle='FP04', optimizer_updates=0, real_bank_used=False,
                  real_predictions_used=False, final_bank_used=False, tested_sources=before,
                  scope='tiny independent contractions, gradients and invariants; no training or bank reads')
    try:
        module = load_model()
        result['numpy_oracle_selfcheck'] = numpy_oracle_selfcheck()
        result['reference'] = reference_checks(module)
        print('REFERENCE_PASS', flush=True)
        result['algebra_float64'] = algebra_checks(module)
        result['algebra_float32'] = algebra_checks(module, dtype=torch.float32)
        print('ALGEBRA_GRADIENT_PASS', flush=True)
        result['invariants_float64'] = invariant_checks(module)
        result['invariants_float32'] = invariant_checks(module, dtype=torch.float32)
        result['boundaries'] = boundary_checks(module)
        result['represented_clip_float64'] = representation_clip_checks(module)
        result['represented_clip_float32'] = representation_clip_checks(module, dtype=torch.float32)
        result['bridge_float64'] = bridge_checks(module)
        result['bridge_float32'] = bridge_checks(module, dtype=torch.float32)
        print('INVARIANTS_BOUNDARIES_BRIDGE_PASS', flush=True)
        result['module_gradients'] = module_gradient_checks(module)
        if args.gpu:
            assert torch.cuda.is_available(), '--gpu requested but CUDA unavailable'
            result['gpu'] = dict(status='PASS', invariants=invariant_checks(module, 'cuda', torch.float32),
                                 bridge=bridge_checks(module, 'cuda', torch.float32),
                                 represented_clip=representation_clip_checks(module, 'cuda', torch.float32),
                                 cpu_gpu=cpu_gpu_checks(module))
        else:
            result['gpu'] = dict(status='NOT_RUN', reason='root did not request optional tiny GPU smoke')
        after = {str(path): sha(path) for path in paths}
        assert before == after, 'source changed during checks; freeze and rerun'
        result.update(status='PASS', tested_sources_after=after, seconds=time.perf_counter()-start)
        save_json(args.output, result)
        print('CHECKS_PASS', args.output, flush=True)
    except Exception as exc:
        result.update(status='FAIL', error=str(exc), traceback=traceback.format_exc(),
                      seconds=time.perf_counter()-start,
                      tested_sources_after={str(path): sha(path) for path in paths if path.exists()})
        failure = args.output.parent / f'checks_failure_{time.time_ns()}.json'
        save_json(failure, result)
        print('CHECKS_FAIL', failure, flush=True)
        raise


if __name__ == '__main__':
    main()

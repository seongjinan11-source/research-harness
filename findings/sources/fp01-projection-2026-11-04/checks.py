"""Pre-lock FP01 numerical checks. No training, source lock or final bank.

Run with the existing research Python: checks.py [--resource].
--resource adds CPU-independent CUDA inference/backward resource checks and a
reused CiteSeer development input. Numerical results go to results/checks.json.
"""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import copy
import itertools
import math
import time
import gc
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT, OLD_BP, GraphCore, digest, dump, read_real, real_rows, tensor_input, sha
from operators import FeatureModel, centroid_bp, potts_conditional_bp


def normal_table(x, grid, sigma):
    return -.5 * math.log(2 * math.pi) - sigma.log()[None, :, None] - .5 * ((x[..., None] - grid[None]) / sigma[None, :, None]).square()


def exact_enumeration(j, valid, context, classes, observed, log_likelihood, log_weights):
    """Independent NumPy brute force of the *whole* class-gated grid model.

    Node prior is 1 (context indicator); centroid prior is normalized discrete w.
    Returns node[H,N,K], centroid[H,K,F,R], logZ[H]. Tiny problems only.
    """
    j, valid, context = [t.detach().cpu().numpy() for t in (j, valid, context)]
    obs, ll = observed.cpu().numpy(), log_likelihood.detach().cpu().numpy()
    lw = log_weights.detach().cpu().numpy()
    h, n, k, f, r = j.shape[1], j.shape[-1], int(classes), obs.shape[1], ll.shape[-1]
    if lw.ndim == 1:
        lw = np.broadcast_to(lw, (f, r))
    lw = lw - np.logaddexp.reduce(lw, axis=-1, keepdims=True)
    states = np.array(list(itertools.product(range(k), repeat=n)), dtype=np.int64)
    keep = np.all((context[0] < 0) | (states == context[0]), axis=1)
    states = states[keep]
    latent = np.array(list(itertools.product(range(r), repeat=k * f)), dtype=np.int64).reshape(-1, k, f)
    if len(states) * len(latent) > 2000000:
        raise ValueError('exact reference is intentionally tiny')
    ss = np.repeat(states, len(latent), axis=0)
    zz = np.tile(latent, (len(states), 1, 1))
    score = np.zeros((h, len(ss)))
    for c in range(k):
        for col in range(f):
            score += lw[col, zz[:, c, col]][None]
    for i in range(n):
        for q in range(i + 1, n):
            if valid[0, i, q]:
                score += j[0, :, i, q, None] * (ss[:, i] == ss[:, q])[None]
    ids = np.arange(len(ss))
    for i, col in zip(*np.nonzero(obs)):
        score += ll[i, col, zz[ids, ss[:, i], col]][None]
    logz = np.logaddexp.reduce(score, axis=-1)
    probability = np.exp(score - logz[:, None])
    node = np.zeros((h, n, k))
    centroid = np.zeros((h, k, f, r))
    for i in range(n):
        for c in range(k):
            node[:, i, c] = probability[:, ss[:, i] == c].sum(-1)
    for c in range(k):
        for col in range(f):
            for state in range(r):
                centroid[:, c, col, state] = probability[:, zz[:, c, col] == state].sum(-1)
    return node, centroid, logz


def gh(points, dtype=torch.float64):
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return torch.tensor(nodes, dtype=dtype), torch.tensor(np.log(weights / np.sqrt(np.pi)), dtype=dtype)


def tiny_inputs(n=8, k=3, f=3, h=3, device='cpu', dtype=torch.float64):
    rng = np.random.default_rng(318041)
    x = rng.normal(size=(n, f))
    obs = np.zeros((n, f), bool)
    obs[2:min(n - 1, 5)] = True
    ctx = np.full(n, -1, np.int64)
    ctx[0], ctx[1] = 0, 1
    j = np.zeros((1, h, n, n))
    for lane in range(h):
        raw = rng.normal(.12 + .15 * lane, .12, size=(n, n))
        j[0, lane] = .5 * (raw + raw.T)
        np.fill_diagonal(j[0, lane], 0)
    j = torch.tensor(j, dtype=dtype, device=device)
    valid = (~torch.eye(n, dtype=torch.bool, device=device))[None]
    context = torch.tensor(ctx[None], device=device)
    weights = torch.arange(1, h + 1, dtype=dtype, device=device)
    weights /= weights.sum()
    p0 = torch.einsum('h,hnk->nk', weights, potts_conditional_bp(j, valid, context, k))
    graph = dict(j=j, valid=valid, context=context, classes=k, weights=weights,
                 p0=p0, z=torch.tensor(rng.normal(size=(n, 24)), dtype=dtype, device=device))
    inp = dict(x=torch.tensor(x, dtype=dtype, device=device),
               observed=torch.tensor(obs, device=device), classes=k)
    return inp, graph


def check_messages_and_trees():
    errors = {}
    # Independent matrix contraction checks both class-gated message directions.
    rng = np.random.default_rng(5601)
    for k, r in [(2, 3), (3, 5)]:
        p = rng.dirichlet(np.ones(k)); q = rng.dirichlet(np.ones(r)); c = k - 1
        ll = rng.normal(-1, .8, r)
        psi = np.ones((k, r)); psi[c] = np.exp(ll)
        matrix_to_node, matrix_to_centroid = psi @ q, p @ psi
        a = np.logaddexp.reduce(np.log(q) + ll)
        scalar_node = np.ones(k); scalar_node[c] = np.exp(a)
        scalar_centroid = (1 - p[c]) + p[c] * np.exp(ll)
        errors[f'matrix_node_K{k}R{r}'] = float(np.max(np.abs(matrix_to_node - scalar_node)))
        errors[f'matrix_centroid_K{k}R{r}'] = float(np.max(np.abs(matrix_to_centroid - scalar_centroid)))
    # Actual heterogeneous factor tree: graph chain + one observed-node centroid leaves.
    k, n, r = 3, 3, 5
    nodes, lw = gh(r)
    grid = (.2 + math.sqrt(2) * .7 * nodes)[None]
    ll = normal_table(torch.tensor([[-.4], [0.], [0.]], dtype=torch.float64), grid,
                      torch.tensor([.65], dtype=torch.float64))
    obs = torch.tensor([[True], [False], [False]])
    j = torch.zeros(1, 2, n, n, dtype=torch.float64)
    for lane, sign in enumerate([1., -.7]):
        j[0, lane, 0, 1] = j[0, lane, 1, 0] = .8 * sign
        j[0, lane, 1, 2] = j[0, lane, 2, 1] = -.6 * sign
    valid = j[:, 0] != 0
    ctx = torch.tensor([[-1, -1, 1]])
    exact_node, exact_centroid, exact_logz = exact_enumeration(j, valid, ctx, k, obs, ll, lw)
    bp, info = centroid_bp(j, valid, ctx, k, obs, ll, lw, 120, .5, 1, True)
    errors['heterogeneous_tree_node'] = float(np.max(np.abs(bp.detach().numpy() - exact_node)))
    errors['heterogeneous_tree_centroid'] = float(np.max(np.abs(info['centroid'].detach().numpy() - exact_centroid)))
    _, graph_logz = OLD_BP(j, valid, ctx, k, 120, .5, chunk=1)
    singleton_constant = torch.logsumexp(lw + ll[0, 0], -1)
    errors['tree_partition_constants_prior'] = float(np.max(np.abs(graph_logz[0].numpy() + float(singleton_constant) - exact_logz)))
    constants = torch.tensor([-.73, .21], dtype=torch.float64)
    weights = (graph_logz[0] + constants).softmax(-1)
    exact_weights = torch.tensor(exact_logz, dtype=torch.float64).add(constants).softmax(-1)
    errors['tree_role_with_graph_baseline'] = float((weights - exact_weights).abs().max())
    errors['tree_pair_active_residual'] = float(info['pair_active_residual'])
    errors['tree_pair_centroid_residual'] = float(info['pair_centroid_residual'])
    assert max(errors.values()) < 2e-10, errors
    return errors


def check_loopy_grid():
    output = []
    for k, r in [(2, 3), (3, 3), (3, 5)]:
        nodes, lw = gh(r)
        grid = (math.sqrt(2) * .8 * nodes)[None]
        x = torch.tensor([[-.7], [.6], [0.]], dtype=torch.float64)
        ll = normal_table(x, grid, torch.tensor([.7], dtype=torch.float64))
        obs = torch.tensor([[True], [True], [False]])
        # Both source labels remain unknown. A separate missing context breaks
        # class symmetry without conditioning away the source/centroid cycle.
        ctx = torch.tensor([[-1, -1, 0]])
        j = torch.zeros(1, 3, 3, 3, dtype=torch.float64)
        j[0, :, 0, 1] = j[0, :, 1, 0] = torch.tensor([-.8, 0, .8], dtype=torch.float64)
        j[0, :, 0, 2] = j[0, :, 2, 0] = torch.tensor([-.5, .3, .9], dtype=torch.float64)
        j[0, :, 1, 2] = j[0, :, 2, 1] = torch.tensor([-.3, .2, .6], dtype=torch.float64)
        valid = (~torch.eye(3, dtype=torch.bool))[None]
        en, ec, ez = exact_enumeration(j, valid, ctx, k, obs, ll, lw)
        bp, info = centroid_bp(j, valid, ctx, k, obs, ll, lw, 24, .5, 1, True)
        output.append(dict(classes=k, grid=r, nodes=3, sources=2, context_source_overlap=False,
                           node_abs_error=float(np.max(np.abs(bp.detach().numpy() - en))),
                           centroid_abs_error=float(np.max(np.abs(info['centroid'].detach().numpy() - ec))),
                           pair_active_residual=float(info['pair_active_residual']),
                           pair_centroid_residual=float(info['pair_centroid_residual']),
                           exact_logz=ez.tolist(), scope='finite grid loopy approximation, not an equality assertion'))
        assert torch.isfinite(bp).all() and (bp.sum(-1) - 1).abs().max() < 1e-12
    return output


def gaussian_integral_diagnostics():
    results = []
    rng = np.random.default_rng(71331)
    # Fixed diagnostic cases, including means between quadrature nodes.
    for n in [1, 5, 20, 100]:
        x = .37 + .45 * rng.normal(size=n)
        mean, ss = float(x.mean()), float(((x - x.mean()) ** 2).sum())
        m, tau, sigma = -.13, .8, .45
        analytic = (-n / 2 * np.log(2 * np.pi) - (n - 1) / 2 * np.log(sigma ** 2)
                    - .5 * np.log(sigma ** 2 + n * tau ** 2)
                    - .5 * (ss / sigma ** 2 + n * (mean - m) ** 2 / (sigma ** 2 + n * tau ** 2)))
        variance = 1 / (1 / tau ** 2 + n / sigma ** 2)
        posterior_mean = variance * (m / tau ** 2 + x.sum() / sigma ** 2)
        for r in [11, 41]:
            nodes, weights = np.polynomial.hermite.hermgauss(r)
            grid = m + math.sqrt(2) * tau * nodes
            logs = np.log(weights / np.sqrt(np.pi)) - .5 * n * np.log(2 * np.pi) - n * np.log(sigma)
            logs -= .5 * (((x[:, None] - grid[None]) / sigma) ** 2).sum(0)
            grid_logz = np.logaddexp.reduce(logs)
            posterior = np.exp(logs - grid_logz)
            gm = float(posterior @ grid)
            gv = float(posterior @ ((grid - gm) ** 2))
            results.append(dict(sources=n, grid=r, analytic_logz=float(analytic), grid_logz=float(grid_logz),
                                logz_error=float(grid_logz - analytic), posterior_mean_error=float(gm - posterior_mean),
                                posterior_variance_error=float(gv - variance)))
    return dict(scope='diagnostic only; finite-grid model does not claim continuous fidelity', records=results)


def _permute_graph(graph, order=None, class_permutation=None):
    given = dict(graph)
    if order is not None:
        given['j'] = graph['j'][:, :, order][:, :, :, order]
        given['valid'] = graph['valid'][:, order][:, :, order]
        given['context'] = graph['context'][:, order]
        given['p0'], given['z'] = graph['p0'][order], graph['z'][order]
    if class_permutation is not None:
        ctx = given['context'].clone(); known = ctx >= 0
        ctx[known] = class_permutation[ctx[known]]
        given['context'] = ctx
        given['p0'] = given['p0'][:, torch.argsort(class_permutation)]
    return given


def check_models():
    torch.manual_seed(3108)
    inp, graph = tiny_inputs()
    models = {a: FeatureModel(a, 11).double() for a in ['A', 'B', 'C_global', 'C']}
    assert digest(models['A']) == digest(models['B'])
    errors, gradients = {}, {}
    assert [sum(p.numel() for p in models[a].parameters()) for a in models] == [721, 721, 3, 751]
    errors['centroid_initial_match'] = float((models['C'](inp, graph) - models['C_global'](inp, graph)).abs().max())
    # A/B parity with the exact original message implementation under common q.
    p, info = models['A'](inp, graph, True)
    n, f = inp['x'].shape
    j = torch.zeros(1, graph['j'].shape[1], n + f, n + f, dtype=torch.float64)
    j[:, :, :n, :n] = graph['j']
    j[0, :, :n, n:] = info['feature_j']; j[0, :, n:, :n] = info['feature_j'].T
    valid = torch.zeros(1, n + f, n + f, dtype=torch.bool)
    valid[:, :n, :n] = graph['valid']; valid[0, :n, n:] = inp['observed']; valid[0, n:, :n] = inp['observed'].T
    ctx = torch.cat((graph['context'], torch.full((1, f), -1)), -1)
    old_node, _ = OLD_BP(j, valid, ctx, inp['classes'], 24, .5, chunk=128)
    old_p = torch.einsum('h,hnk->nk', graph['weights'], old_node[0, :, :n])
    errors['old_operator_parity'] = float((p - old_p).abs().max())
    # Test an encoder away from its special zero-final-layer initialization.
    with torch.no_grad():
        models['C'].feature[-1].weight.normal_(0, .02)
        models['C'].feature[-1].bias.copy_(torch.tensor([.08, -.04, .06]))
    order = torch.tensor([6, 3, 0, 7, 2, 5, 1, 4])
    perm = torch.tensor([2, 0, 1])
    for arm, model in models.items():
        p = model(inp, graph)
        known = graph['context'][0] >= 0
        errors[f'{arm}_clamp'] = float((p[known] - F.one_hot(graph['context'][0, known], inp['classes'])).abs().max())
        poison = dict(inp, x=inp['x'].clone()); poison['x'][~inp['observed']] = float('nan')
        errors[f'{arm}_hidden_poison'] = float((model(poison, graph) - p).abs().max())
        cols = torch.tensor([2, 0, 1])
        errors[f'{arm}_column_permutation'] = float((model(dict(inp, x=inp['x'][:, cols], observed=inp['observed'][:, cols]), graph) - p).abs().max())
        node_input = dict(inp, x=inp['x'][order], observed=inp['observed'][order])
        errors[f'{arm}_node_source_order_permutation'] = float((model(node_input, _permute_graph(graph, order)) - p[order]).abs().max())
        errors[f'{arm}_class_permutation'] = float((model(inp, _permute_graph(graph, class_permutation=perm))[:, perm] - p).abs().max())
        none = dict(inp, observed=torch.zeros_like(inp['observed']))
        errors[f'{arm}_none'] = float((model(none, graph) - graph['p0']).abs().max())
        single_obs = torch.zeros_like(inp['observed']); single_obs[3] = True
        errors[f'{arm}_singleton'] = float((model(dict(inp, observed=single_obs), graph) - graph['p0']).abs().max())
        query = torch.tensor([5, 6, 7]); y = torch.tensor([0, 1, 2])
        loss = -p[query, y].log().mean()
        loss.backward()
        grads = [v.grad for v in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        norm = sum(float(g.abs().sum()) for g in grads)
        assert norm > 1e-10, (arm, norm)
        gradients[arm] = dict(absolute_sum=norm, query_feature_missing=True,
                              source_grad_crosses_graph=True,
                              parameter_count=sum(v.numel() for v in model.parameters()))
        model.zero_grad(set_to_none=True)
        model(none, graph)[query, y].log().mean().backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() and not v.grad.any() for v in model.parameters())
    # The actual low-level source0 algorithm is checked before relying on p0 shortcut.
    nodes, lw = gh(5)
    ll = normal_table(inp['x'], nodes[None].expand(f, -1), torch.ones(f, dtype=torch.float64))
    no_node, no_info = centroid_bp(graph['j'], graph['valid'], graph['context'], inp['classes'],
                                  torch.zeros_like(inp['observed']), ll, lw)
    no_p = torch.einsum('h,hnk->nk', graph['weights'], no_node)
    errors['centroid_lowlevel_none'] = float((no_p - graph['p0']).abs().max())
    errors['isolated_centroid_prior'] = float((no_info['centroid'] - lw.softmax(-1)).abs().max())
    assert max(errors.values()) < 2e-9, errors
    return errors, gradients


def check_finite_gradient():
    inp, graph = tiny_inputs(n=6, f=1, h=1)
    nodes, lw = gh(3)
    ll = normal_table(inp['x'], nodes[None], torch.ones(1, dtype=torch.float64)).detach().requires_grad_()
    direction = torch.tensor(np.random.default_rng(921).normal(size=ll.shape), dtype=torch.float64)
    def objective(table):
        b, _ = centroid_bp(graph['j'], graph['valid'], graph['context'], 3, inp['observed'], table, lw, 12, .5, 2)
        return -b[0, 5, 2].log()
    value = objective(ll)
    gradient = torch.autograd.grad(value, ll)[0]
    analytic = float((gradient * direction).sum())
    eps = 1e-5
    finite = float((objective(ll.detach() + eps * direction) - objective(ll.detach() - eps * direction)) / (2 * eps))
    error = abs(analytic - finite)
    assert error < 2e-7, (analytic, finite, error)
    return dict(autograd=analytic, central_difference=finite, abs_error=error)


def inherited_graph_check():
    rng = np.random.default_rng(32811)
    n = 12
    a = np.triu(rng.random((n, n)) < .2, 1); a |= a.T
    src, dst = np.nonzero(a)
    inp = dict(x=rng.normal(size=(n, 3)).astype('float32'), observed=np.zeros((n, 3), bool),
               src=src.astype('int64'), dst=dst.astype('int64'), context_ids=np.array([0, 1, 2]),
               labels_context=np.array([0, 1, 2]), classes=3, forward_seed=812)
    core = GraphCore(11, device='cpu', dtype=torch.float64)
    before = core.check()
    graphs = {}
    for arm in ['full', 'edge']:
        graph = core.encode(inp, arm, chunk=4)
        branch = potts_conditional_bp(graph['j'], graph['valid'], graph['context'], 3, chunk=4)
        mixed = torch.einsum('h,hnk->nk', graph['weights'].to(branch), branch)
        error = float((mixed - graph['p0']).abs().max())
        assert error < 1e-12
        graphs[arm] = dict(p0_error=error, lanes=int(graph['j'].shape[1]), active_roles=graph['active_role_ids'].tolist())
    assert before == core.check()
    return dict(status='PASS', digest=before, panels=graphs, real_gradient_updates=0)


def resource_checks():
    if not torch.cuda.is_available():
        raise RuntimeError('--resource requires CUDA; do not silently pass an unrun check')
    records = {}
    def measure(name, model, inp, graph, backward=False):
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); begin = time.perf_counter()
        if backward:
            p = model(inp, graph)
            p[-1, 0].clamp_min(1e-10).log().neg().backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        else:
            with torch.inference_mode():
                p = model(inp, graph)
        torch.cuda.synchronize()
        assert torch.isfinite(p).all() and (p.sum(-1) - 1).abs().max() < 2e-5
        records[name] = dict(seconds=time.perf_counter() - begin, peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                             nodes=len(inp['x']), classes=inp['classes'], source_nodes=int(inp['observed'].any(-1).sum()),
                             lanes=graph['j'].shape[1], grid=model.grid_points, backward=backward)
        del p
    inp, graph = tiny_inputs(n=128, k=7, f=16, h=3, device='cuda', dtype=torch.float32)
    inp['observed'][:] = False; inp['observed'][8:72] = True
    model = FeatureModel('C').cuda()
    measure('train_shape_forward_backward', model, inp, graph, True)
    del model, inp, graph
    gc.collect(); torch.cuda.empty_cache()
    # Frozen graph-only lanes may be exactly pruned; this checks a large H=1 lane.
    n, k, f = 2000, 15, 16
    j = torch.full((1, 1, n, n), -.003, device='cuda')
    j[0, 0].fill_diagonal_(0)
    valid = (~torch.eye(n, device='cuda', dtype=torch.bool))[None]
    ctx = torch.full((1, n), -1, device='cuda', dtype=torch.long); ctx[0, :k] = torch.arange(k, device='cuda')
    obs = torch.zeros(n, f, dtype=torch.bool, device='cuda'); obs[20:120] = True
    inp = dict(x=torch.randn(n, f, device='cuda'), observed=obs, classes=k)
    graph = dict(j=j, valid=valid, context=ctx, classes=k, weights=torch.ones(1, device='cuda'),
                 p0=torch.full((n, k), 1 / k, device='cuda'), z=torch.zeros(n, 24, device='cuda'))
    model = FeatureModel('C').cuda()
    measure('large_C15_source100', model, inp, graph)
    del model, inp, graph, j, valid, ctx, obs
    gc.collect(); torch.cuda.empty_cache()
    row = next(row for row in real_rows() if row['graph'] == 'CiteSeer' and row['mode'] == '95')
    raw, _, _ = read_real(row)
    core = GraphCore(11)
    graph = core.encode(raw, 'full', chunk=128)
    inp = tensor_input(raw)
    with torch.inference_mode():
        active_branches = potts_conditional_bp(graph['j'], graph['valid'], graph['context'], inp['classes'])
        active_p0 = torch.einsum('h,hnk->nk', graph['weights'].to(active_branches), active_branches)
        pruning_error = float((active_p0 - graph['p0']).abs().max())
    assert pruning_error < 2e-6, pruning_error
    model = FeatureModel('C').cuda()
    measure('reused_CiteSeer95_inference', model, inp, graph)
    records['reused_CiteSeer95_inference']['exact_zero_role_pruning_p0_error'] = pruning_error
    core.check()
    del model, inp, graph, core
    gc.collect(); torch.cuda.empty_cache()
    return dict(status='PASS', records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resource', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    tested_paths = [ROOT / name for name in ['operators.py', 'checks.py', 'common.py']]
    tested_sources = {str(path): sha(path) for path in tested_paths}
    result = dict(status='RUNNING', scope='pre-lock numerical checks; no optimizer updates or final predictions',
                  optimizer_updates=0, final_bank_used=False, tested_sources=tested_sources)
    result['tree_message_errors'] = check_messages_and_trees()
    print('TREE_AND_MESSAGES_PASS', flush=True)
    result['loopy_grid_diagnostics'] = check_loopy_grid()
    result['continuous_quadrature_diagnostics'] = gaussian_integral_diagnostics()
    result['model_errors'], result['gradients'] = check_models()
    print('MODEL_INVARIANCE_GRADIENT_PASS', flush=True)
    result['directional_gradient'] = check_finite_gradient()
    result['frozen_graph'] = inherited_graph_check()
    result['resource'] = resource_checks() if args.resource else dict(status='NOT_RUN', reason='run checks.py --resource before lock')
    assert tested_sources == {str(path): sha(path) for path in tested_paths}, 'Numerical source changed during checks'
    result['status'] = 'PASS'
    dump(ROOT / 'results/checks.json', result)
    print('CHECKS_PASS', ROOT / 'results/checks.json', flush=True)


if __name__ == '__main__':
    main()

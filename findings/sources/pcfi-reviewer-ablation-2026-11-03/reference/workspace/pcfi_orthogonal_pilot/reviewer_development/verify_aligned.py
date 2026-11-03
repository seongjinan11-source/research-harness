"""CPU forward/gradient checks and bounded timing for the equivalent frontend.

No optimizer steps, no PFN execution, no query labels, and no new accuracy claim.
All original sources, result summaries and checkpoints are checked for changes.
"""
import sys
sys.dont_write_bytecode = True
import sklearn
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, WORK, Front, geometry, tensor_extra, load_episode, sha, dump
from aligned_operator import aligned_forward, prepare_sparse, source_kernel_forward

HERE = Path(__file__).resolve().parent


def numeric_extra(extra, dtype):
    result = tensor_extra(extra, 'cpu')
    return {key: value.to(dtype) if value.is_floating_point() else value
            for key, value in result.items()}


def toy_inputs(template):
    rng = np.random.default_rng(2030110301)
    inp = copy.deepcopy(template)
    inp.observed = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1],
                             [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],
                             [0, 0, 0, 0]], dtype=bool)
    full_values = rng.normal(size=(7, 4)).astype('float32')
    edges = np.array([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)])
    inp.src = np.concatenate((edges[:, 0], edges[:, 1]))
    inp.dst = np.concatenate((edges[:, 1], edges[:, 0]))
    inp.context = np.array([0, 3])
    inp.labels_context = np.array([0, 1])
    inp.query = np.array([1, 2, 4, 5, 6])
    inp.classes = 2
    variants = []
    for case in ['cell_disconnected', 'full', 'zero_sources']:
        obj = copy.deepcopy(inp)
        if case == 'full':
            obj.observed[:] = True
        elif case == 'zero_sources':
            obj.observed[:] = False
        obj.observed_values = np.where(obj.observed, full_values, 0).astype('float32')
        obj.pcfi = np.zeros_like(full_values)
        variants.append((case, obj, geometry(obj)))
    return variants


def check_forward(front, inp, extra, *, dtype, case):
    xo = torch.as_tensor(inp.observed_values, dtype=dtype)
    observed = torch.as_tensor(inp.observed)
    tex = numeric_extra(extra, dtype)
    with torch.no_grad():
        expected, _ = front(xo, observed, tex, 'node')
        aligned = aligned_forward(front, xo, observed, tex)
        sparse = aligned_forward(front, xo, observed, tex, backend='sparse')
    difference = float((expected - aligned).abs().max())
    sparse_difference = float((expected - sparse).abs().max())
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    assert difference < tolerance, (case, difference)
    # Sparse refactorization recomputes a float32-rounded cached transition.
    assert sparse_difference < 2e-5, (case, sparse_difference)
    if observed.any():
        assert torch.equal(aligned[observed], xo[observed])
        assert torch.equal(sparse[observed], xo[observed])
    if not observed.any():
        assert torch.count_nonzero(aligned) == torch.count_nonzero(sparse) == 0
    return {'case': case, 'dtype': str(dtype), 'nodes': len(xo),
            'features': xo.shape[1], 'aligned_max_abs': difference,
            'sparse_max_abs': sparse_difference}


def gradients(front, inp, extra):
    xo = torch.as_tensor(inp.observed_values, dtype=torch.float64)
    observed = torch.as_tensor(inp.observed)
    tex = numeric_extra(extra, torch.float64)
    probe = torch.linspace(-.6, .9, xo.numel(), dtype=xo.dtype).reshape_as(xo)
    parameters = tuple(front.parameters())
    expected, _ = front(xo, observed, tex, 'node')
    g0 = torch.autograd.grad((expected * probe).sum(), parameters)
    actual = aligned_forward(front, xo, observed, tex)
    g1 = torch.autograd.grad((actual * probe).sum(), parameters)
    sparse = aligned_forward(front, xo, observed, tex, backend='sparse')
    g2 = torch.autograd.grad((sparse * probe).sum(), parameters)
    g0 = torch.cat([x.flatten() for x in g0])
    g1 = torch.cat([x.flatten() for x in g1])
    g2 = torch.cat([x.flatten() for x in g2])
    error = float((g0 - g1).abs().max())
    sparse_error = float((g0 - g2).abs().max())
    magnitude = float(g0.abs().max())
    assert magnitude > 0
    assert error < 1e-9, (error, magnitude)
    assert sparse_error < 2e-6, (sparse_error, magnitude)
    return {'maximum_parameter_gradient_error': error,
            'sparse_parameter_gradient_error': sparse_error,
            'maximum_reference_gradient': magnitude}


def timings(front, inp, extra):
    xo = torch.as_tensor(inp.observed_values)
    observed = torch.as_tensor(inp.observed)
    tex = tensor_extra(extra, 'cpu')
    functions = {
        'original_node': lambda: front(xo, observed, tex, 'node')[0],
        'aligned_index_add': lambda: aligned_forward(front, xo, observed, tex),
        # Include building the sparse adjacency and its normalization each call.
        'aligned_sparse': lambda: aligned_forward(front, xo, observed, tex,
                                                  backend='sparse'),
    }
    results = []
    with torch.no_grad():
        for function in functions.values():
            for _ in range(3):
                function()
        samples = {name: [] for name in functions}
        # Alternate execution order to reduce a systematic order effect.
        for repetition in range(10):
            names = list(functions)
            if repetition % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                functions[name]()
                samples[name].append((time.perf_counter() - start) * 1000)
    for name, sample in samples.items():
        results.append({'backend': name, 'median_ms': float(np.median(sample)),
                        'p10_ms': float(np.quantile(sample, .1)),
                        'p90_ms': float(np.quantile(sample, .9)), 'samples_ms': sample})
    return {'nodes': len(xo), 'features': xo.shape[1], 'timings': results}


def main():
    torch.set_num_threads(2)
    files = [ROOT / name for name in ['core.py', 'bank.py', 'run.py', 'protocol.json',
             'METHOD_KO.md', 'REPORT_KO.md', 'bank/manifest.json', 'results/lock.json',
             'results/statistics.json', 'results/audit.json']]
    for seed in [11, 12, 13]:
        for arm in ['scalar', 'edge', 'node']:
            files.append(ROOT / 'results' / f'{arm}_s{seed}' / 'checkpoint.pt')
        files.append(WORK / 'nsd_token_pfn/results' / f'pcfi100_s{seed}' / 'checkpoint.pt')
    before = {str(path.relative_to(WORK)): sha(path) for path in files}
    rows = json.loads((ROOT / 'bank/manifest.json').read_text(encoding='utf8'))['episodes']
    development = [row for row in rows if row['split'] == 'dev']
    first_inp, _, _ = load_episode(development[0])
    toys = toy_inputs(first_inp)
    records, gradient_records, dense_records = [], [], []
    for seed in [11, 12, 13]:
        front = Front(seed)
        state = torch.load(ROOT / 'results' / f'node_s{seed}' / 'checkpoint.pt',
                           map_location='cpu', weights_only=False)
        front.load_state_dict(state['front'])
        front.eval()
        for index, row in enumerate(development):
            inp, extra, _ = load_episode(row)
            records.append({'seed': seed, **check_forward(front, inp, extra,
                            dtype=torch.float32, case=f'dev_{index}_{row["case"]}')})
        for case, inp, extra in toys:
            records.append({'seed': seed, **check_forward(front, inp, extra,
                            dtype=torch.float32, case=case)})
        double_front = copy.deepcopy(front).double()
        for name, candidate in [('learned', double_front), ('large_angle',
                                 copy.deepcopy(double_front))]:
            if name == 'large_angle':
                torch.manual_seed(seed + 99000)
                with torch.no_grad():
                    candidate.phi[-1].weight.normal_(std=.6)
                    candidate.phi[-1].bias.normal_(std=.2)
            for case, inp, extra in toys:
                records.append({'seed': seed, 'parameter_case': name,
                                **check_forward(candidate, inp, extra,
                                  dtype=torch.float64, case=case)})
                xo = torch.as_tensor(inp.observed_values, dtype=torch.float64)
                obs = torch.as_tensor(inp.observed)
                tex = numeric_extra(extra, torch.float64)
                with torch.no_grad():
                    expected, _ = candidate(xo, obs, tex, 'node')
                    dense = source_kernel_forward(candidate, xo, obs, tex)
                error = float((expected - dense).abs().max())
                assert error < 1e-10, (case, name, error)
                dense_records.append({'seed': seed, 'parameters': name,
                                      'case': case, 'maximum_error': error})
            case, inp, extra = toys[0]
            gradient_records.append({'seed': seed, 'parameters': name,
                                     **gradients(candidate, inp, extra)})
        print('EQUIVALENCE', seed, 'PASS', flush=True)
    timing = []
    front = Front(11)
    front.load_state_dict(torch.load(ROOT / 'results/node_s11/checkpoint.pt',
                          map_location='cpu', weights_only=False)['front'])
    selected = [next(r for r in development if r['case'] == 'uniform90')]
    selected += [next(r for r in rows if r['split'] == 'real' and r['graph'] == graph
                 and r['repetition'] == 0 and r['case'] == 'uniform90')
                 for graph in ['Cora', 'texas']]
    for row in selected:
        inp, extra, _ = load_episode(row)
        timing.append({'split': row['split'], 'graph': row['graph'],
                       **timings(front, inp, extra)})
        print('TIMING', row['split'], row['graph'], 'COMPLETE', flush=True)
    after = {str(path.relative_to(WORK)): sha(path) for path in files}
    assert before == after
    output = {
        'status': 'PASS', 'date': '2026-11-03', 'device': 'cpu', 'threads': 2,
        'optimizer_updates': 0, 'pfn_execution': False,
        'data_scope': '14 existing dev episodes and fabricated structural tests; real inputs used for frontend timing only',
        'forward_checks': records, 'gradient_checks': gradient_records,
        'dense_source_kernel_checks': dense_records, 'frontend_cost': timing,
        'source_and_checkpoint_preservation': before,
        'new_source_hashes': {name: sha(HERE / name) for name in
                              ['aligned_operator.py', 'verify_aligned.py']},
        'limits': 'Exact algebra for static node/column frames, not edge-incidence maps. Sparse path refactors cached float32 P. CPU forward/gradient and frontend-only cost do not establish PFN accuracy, GPU reproducibility, or end-to-end speed. Existing test scores and gates unchanged.',
    }
    dump(HERE / 'results/node_alignment.json', output)
    print('NODE ALIGNMENT REVIEW PASS', len(records), len(gradient_records),
          len(dense_records), flush=True)


if __name__ == '__main__':
    main()

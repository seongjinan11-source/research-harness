"""Prespecified base-graph paired analysis, cost measurement and integrity audit."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, C, D, cfg, dump, sha, read, rows, lock
from operators import FrozenGraph
from reference import Reference
from run import load_model, predict, is_public

METRICS = ['accuracy', 'nll', 'brier', 'reference_kl']


def metric(p, y, ref):
    p = np.clip(p.astype(float), 1e-10, 1)
    p /= p.sum(-1, keepdims=True)
    ref = np.clip(ref.astype(float), 1e-10, 1)
    ref /= ref.sum(-1, keepdims=True)
    return [float(np.mean(p.argmax(-1) == y)), float(-np.log(p[np.arange(len(y)), y]).mean()),
            float(((p - np.eye(2)[y]) ** 2).sum(-1).mean()), float((ref * (np.log(ref) - np.log(p))).sum(-1).mean())]


def gather():
    c, test = cfg(), rows('test')
    inputs = [read(r) for r in test]
    refs = {a: [] for a in ['fixed_exact', 'fixed_bp', 'native_exact']}
    for r in test:
        with np.load(ROOT / 'references' / (Path(r['path']).stem + '.npz')) as z:
            for arm in refs:
                refs[arm].append(z[arm])
    reference = np.stack(refs['native_exact'])
    output = {}
    for arm in c['arms'] + list(refs):
        seed_predictions = []
        if arm in refs:
            seed_predictions = [np.stack(refs[arm])]
        else:
            for seed in c['seeds']:
                with np.load(ROOT / 'results' / f'{arm}_s{seed}' / 'predictions.npz') as z:
                    seed_predictions.append(z['p'])
        values = []
        for probability in seed_predictions:
            per_episode = []
            for i, (inp, q, y) in enumerate(inputs):
                missing = ~inp['observed'][q].any(-1)
                assert missing.any()
                per_episode.append([metric(probability[i, q[missing]], y[missing], reference[i, q[missing]]),
                                    metric(probability[i, q], y, reference[i, q])])
            values.append(per_episode)
        output[arm] = np.asarray(values)  # seed, episode, subset, metric
    return test, output


def groups(test, predicate):
    ids = [i for i, r in enumerate(test) if predicate(r)]
    return [[i for i in ids if test[i]['graph'] == gid] for gid in sorted({test[i]['graph'] for i in ids})]


def graph_values(value, indices, subset=0):
    return np.stack([value[:, ids, subset].mean((0, 1)) for ids in indices])


def difference(a, b):
    delta = a - b
    rng = np.random.default_rng(202611021421)
    boot = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1)
    return {m: {'delta': float(delta[:, i].mean()), 'ci95': np.quantile(boot[:, i], [.025, .975]).tolist()} for i, m in enumerate(METRICS)}


def summarize():
    lock()
    test, values = gather()
    selectors = {'primary_heldout': lambda r: r['parameter_mode'] == 'heldout', 'all': lambda r: True,
                 'matched': lambda r: r['parameter_mode'] == 'matched', 'gaussian_guard': lambda r: r['regime'] == 'gaussian'}
    for regime in ['gaussian', 'heavy', 'variance', 'mixed']:
        for mode in ['matched', 'heldout']:
            selectors[f'{regime}_{mode}'] = lambda r, rg=regime, md=mode: r['regime'] == rg and r['parameter_mode'] == md
    for k in cfg()['contexts']:
        selectors[f'heldout_k{k}'] = lambda r, kk=k: r['parameter_mode'] == 'heldout' and r['context_total'] == kk
    for prior in ['independent', 'exclusive', 'coupled']:
        selectors[f'heldout_{prior}'] = lambda r, p=prior: r['parameter_mode'] == 'heldout' and r['prior'] == p
    result = {}
    pairs = [('adaptive', a) for a in ['original', 'static', 'fixed_exact', 'fixed_bp', 'mask_pfn', 'pcfi_pfn']] + [('static', 'original')]
    for name, selector in selectors.items():
        indices = groups(test, selector)
        val = {arm: graph_values(v, indices) for arm, v in values.items()}
        means = {arm: dict(zip(METRICS, v.mean(0).tolist())) for arm, v in val.items()}
        comparisons = {a + '-' + b: difference(val[a], val[b]) for a, b in pairs}
        all_query_means = {arm: dict(zip(METRICS, graph_values(v, indices, 1).mean(0).tolist())) for arm, v in values.items()}
        result[name] = {'graphs': len(indices), 'means': means, 'differences': comparisons, 'all_query_secondary': all_query_means}
    primary, guard = result['primary_heldout'], result['gaussian_guard']
    core = all(primary['differences']['adaptive-' + a]['accuracy']['ci95'][0] > 0 for a in ['original', 'static'])
    gaussian = all(guard['differences']['adaptive-' + a]['accuracy']['ci95'][0] > -.02 for a in ['original', 'static'])
    verdict = {'adaptation_accuracy_gate': core, 'gaussian_2pp_noninferiority_guard': gaussian,
               'competitive_gate': None, 'interpretation': 'Primary and Gaussian tests are prespecified; other subgroup CIs exploratory, unadjusted. Seeds are fixed replications, not population-level seed inference.'}
    cost_path = ROOT / 'results/cost.json'
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())['single_graph']
        faster = min(['fixed_exact', 'fixed_bp'], key=lambda a: cost[a]['median_ms'])
        ci = primary['differences']['adaptive-' + faster]['accuracy']['ci95']
        ratio = cost[faster]['median_ms'] / cost['adaptive']['median_ms']
        competitive = (ci[0] > 0 and ratio >= 1) or (ci[0] > -.02 and ratio >= 2)
        verdict.update(competitive_gate=bool(core and gaussian and competitive), faster_fixed_control=faster, speed_ratio=ratio)
    dump(ROOT / 'results/statistics.json', {'subsets': result, 'verdict': verdict,
         'unit': 'Base graph: mean queries within episode, mean six episodes and three fixed seeds within graph. Paired5000 bootstrap. No node bootstrap.'})
    np.savez_compressed(ROOT / 'results/metric_arrays.npz', **values)
    print(json.dumps({'primary': primary, 'gaussian_guard': guard, 'verdict': verdict}, ensure_ascii=False), flush=True)


def costs():
    C.setup()
    torch.set_num_threads(2)
    lock()
    # One fixed case per law regime, same context2/O6, chosen without outcomes.
    test = rows('test')
    chosen = [next((i, r) for i, r in enumerate(test) if r['regime'] == law and r['parameter_mode'] == 'heldout' and r['context_total'] == 2 and r['observed_nodes'] == 6) for law in ['gaussian', 'heavy', 'variance', 'mixed']]
    raw = [(i, read(r)[0], read(r)[1]) for i, r in chosen]
    ref = Reference()
    graph = FrozenGraph().cuda().double().eval()

    def measure(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(30):
            torch.cuda.synchronize()
            started = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
        return {'median_ms': float(np.median(times)), 'p25_ms': float(np.quantile(times, .25)), 'p75_ms': float(np.quantile(times, .75)),
                'additional_peak_mib': (torch.cuda.max_memory_allocated() - baseline) / 2 ** 20, 'times_ms': times}

    single, batch = {}, {}
    for arm in cfg()['arms'] + ['fixed_exact', 'fixed_bp']:
        model = load_model(arm, 11) if arm in cfg()['arms'] else None

        def execute(items):
            # Include tensor/stat preparation and actual PCFI. No truth or bank I/O.
            if is_public(arm):
                output = []
                for index, inp, q in items:
                    given = dict(inp)
                    if arm == 'pcfi_pfn':
                        given['pcfi'] = D.pcfi(given)
                    output.append(predict(model, arm, given, q, index))
                return output
            tensor = D.tensors([v[1] for v in items], dtype=torch.float64 if arm.startswith('fixed') else torch.float32)
            with torch.inference_mode():
                if model is not None:
                    return model(tensor)
                gp = graph(tensor)
                return ref.point_exact(tensor, gp) if arm == 'fixed_exact' else ref.fixed_bp(tensor, gp)

        # Four regimes measured individually, average medians prevents regime selection.
        per = [measure(lambda item=item: execute([item])) for item in raw]
        single[arm] = {'median_ms': float(np.mean([v['median_ms'] for v in per])),
                       'additional_peak_mib': max(v['additional_peak_mib'] for v in per), 'per_regime': per}
        batch[arm] = measure(lambda: execute(raw * 4))
        print('COST', arm, single[arm]['median_ms'], batch[arm]['median_ms'], flush=True)
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    dump(ROOT / 'results/cost.json', {'single_graph': single, 'batch16': batch,
         'scope': 'RTX3090 after all our training workers; external GPU workload uncontrolled.30 repeats after5 warmups. Single is average of four prespecified law-regime median latencies. Public batch is16 sequential variable-task forwards, structured methods batched. Additional working memory, not total GPU.'})


def audit():
    C.setup()
    torch.set_num_threads(2)
    state = lock()
    bank = json.loads((ROOT / 'bank/manifest.json').read_text())
    for row in bank['episodes']:
        assert sha(ROOT / 'bank' / row['path']) == row['sha256']
        inp, q, y = read(row)
        assert not inp['observed'][inp['context'] >= 0].any() and (~inp['observed'][q].any(-1)).any()
    refs = json.loads((ROOT / 'references/manifest.json').read_text())
    for row in refs['episodes']:
        assert sha(ROOT / row['path']) == row['sha256']
    schedules, factor_frozen, replay = {}, set(), []
    test = rows('test')
    selected = [0, 385, 770]
    training = []
    for seed in cfg()['seeds']:
        for arm in cfg()['arms']:
            folder = ROOT / 'results' / f'{arm}_s{seed}'
            info = json.loads((folder / 'training.json').read_text())
            training.append(info)
            assert info['updates'] == 256 and info['examples'] == 2048
            assert schedules.setdefault(seed, info['schedule_sha256']) == info['schedule_sha256']
            if not is_public(arm):
                factor_frozen.add(info['frozen_sha256'])
            evaluation = json.loads((folder / 'evaluation.json').read_text())
            assert evaluation['lock'] == state and evaluation['prediction_sha256'] == sha(folder / 'predictions.npz')
            model = load_model(arm, seed)
            with np.load(folder / 'predictions.npz') as z:
                p = z['p']
            for i in selected:
                inp, q, _ = read(test[i])
                new = predict(model, arm, inp, q, i)
                error = float(abs(new - p[i, q]).max())
                assert error < 5e-4, (arm, seed, error)
                replay.append({'arm': arm, 'seed': seed, 'episode': i, 'max_abs_error': error})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert len(factor_frozen) == 1
    dump(ROOT / 'results/audit.json', {'all_source_config_and_bank_hashes_valid': True, 'all_masks_valid': True,
         'all_reference_hashes_valid': True, 'same_task_schedule_per_seed': True, 'same_frozen_graph_all_factor_arms_and_seeds': True,
         'new_training_models': len(training), 'successful_updates': sum(r['updates'] for r in training),
         'task_exposures': sum(r['examples'] for r in training), 'new_training_worker_seconds_sum': sum(r['seconds'] for r in training),
         'development_smoke_updates_separate': 3, 'historical_checkpoint_training_reused_not_in_new_budget': True,
         'prediction_replays': replay, 'parameter_counts': {a: next(r['trainable_parameters'] for r in training if r['arm'] == a) for a in cfg()['arms']}})
    print('AUDIT_PASSED', len(replay), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['summary', 'cost', 'audit'])
    args = ap.parse_args()
    {'summary': summarize, 'cost': costs, 'audit': audit}[args.action]()

"""Frozen-checkpoint confirmation; no training or mutation of the original study."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OLD = WORK / 'internal_pcfi_pilot'
sys.path.insert(0, str(OLD))
import pcfi_shared as S
from pcfi_model import predict
from pcfi_trial import load
from pcfi_tasks import generator_config
C, dump, sha = S.C, S.dump, S.sha
METRICS = ['accuracy', 'nll', 'brier']


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def checkpoint_files():
    return [OLD / 'results' / f'{a}_s{s}' / f for a in cfg()['arms']
            for s in cfg()['seeds'] for f in ['checkpoint.pt', 'training.json']]


def sources():
    paths = list(ROOT.glob('*.py')) + [ROOT / 'protocol.json']
    return {str(p.relative_to(WORK)): sha(p) for p in paths}


def design():
    d = json.loads((ROOT / 'results/design.json').read_text(encoding='utf8'))
    assert d['sources'] == sources(), 'Confirmation sources changed after lock'
    assert d['original_lock_sha256'] == sha(OLD / 'results/lock.json')
    assert d['original_sources'] == S.sources()
    assert d['checkpoints'] == {str(p.relative_to(WORK)): sha(p) for p in checkpoint_files()}
    S.lock()
    return d


def rows():
    d = design()
    m = ROOT / 'bank/manifest.json'
    assert json.loads(m.read_text(encoding='utf8'))['design_sha256'] == sha(ROOT / 'results/design.json')
    l = json.loads((ROOT / 'results/bank_lock.json').read_text(encoding='utf8'))
    assert l == {'design_sha256': sha(ROOT / 'results/design.json'), 'manifest_sha256': sha(m)}
    return json.loads(m.read_text(encoding='utf8'))['episodes']


def read(row):
    with np.load(ROOT / 'bank' / row['path']) as z:
        inp = C.Inputs(**{k: z[k] for k in C.Inputs.__annotations__})
        extra = {k: z[k] for k in ['confidence', 'edge_i', 'edge_j']}
        return inp, extra, z['labels_query'].astype(np.int64)


def lock_design():
    assert not (ROOT / 'results/design.json').exists()
    assert not (ROOT / 'bank').exists(), 'Lock before generating any final graph'
    C.setup(); torch.set_num_threads(2); S.lock()
    replay = []
    oldrows = S.rows('test')
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            model = load(arm, seed)
            for i in [1, 116, 230]:
                inp, extra, _ = S.read(oldrows[i])
                with np.load(OLD / 'results' / f'{arm}_s{seed}' / 'predictions' / f'e{i}.npz') as z:
                    error = float(np.abs(predict(model, inp, extra) - z['p']).max())
                assert error < .004
                replay.append({'arm': arm, 'seed': seed, 'old_episode': i, 'max_abs_error': error})
            del model; gc.collect(); torch.cuda.empty_cache()
    dump(ROOT / 'results/preflight.json', {'old_prediction_replays': replay, 'new_training_updates': 0})
    dump(ROOT / 'results/design.json', {
        'sources': sources(), 'original_sources': S.sources(),
        'original_lock_sha256': sha(OLD / 'results/lock.json'),
        'checkpoints': {str(p.relative_to(WORK)): sha(p) for p in checkpoint_files()},
        'preflight_sha256': sha(ROOT / 'results/preflight.json'),
        'locked_before_bank_generation': True, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    design(); print('DESIGN_LOCKED', flush=True)


def generate():
    design(); C.setup()
    assert not (ROOT / 'bank').exists(), 'No silent final bank restart'
    from mgpfn import prior_adapter as PA
    c = cfg(); generation = generator_config(); records = []; raw = []
    dump(ROOT / 'bank/generator.json', generation)
    old_seeds = {r['graph_seed'] for split in ['train', 'dev', 'test'] for r in S.rows(split)}
    old_requested = {S.cfg()[split + '_seed_start'] + i for split in ['train', 'dev', 'test'] for i in range(S.cfg()[split + '_graphs'])}
    assert not old_requested.intersection(range(c['graph_seed_start'], c['graph_seed_start'] + c['graphs']))
    begun = time.perf_counter()
    for gid in range(c['graphs']):
        requested = c['graph_seed_start'] + gid
        for attempt in range(20):
            actual = requested + attempt * 100000000
            assert actual not in old_seeds
            try:
                task = PA.sample_official_task(generation, actual, allowed_task_types=('binclass', 'multiclass'))
                break
            except AssertionError:
                pass
        else:
            raise RuntimeError('Generator retries exhausted')
        _, task['y'] = np.unique(task['y'], return_inverse=True)
        n, d = task['X_full'].shape; classes = len(np.unique(task['y']))
        rng = np.random.default_rng(requested + 777)
        anchors = {cl: rng.permutation(np.flatnonzero(task['y'][:task['n_train']] == cl)) for cl in range(classes)}
        assert all(len(a) > 0 for a in anchors.values())
        query = rng.permutation(np.arange(task['n_train'], n))[:c['queries']]
        assert len(query) >= 4
        rp = ROOT / 'bank/raw' / f'g{gid}.npz'; rp.parent.mkdir(parents=True, exist_ok=True)
        PA.save_task(rp, task)
        raw.append({'graph': gid, 'path': rp.relative_to(ROOT / 'bank').as_posix(), 'sha256': sha(rp)})
        for sid, (mechanism, rate, k) in enumerate(c['settings']):
            context = np.concatenate([anchors[cl][:k] for cl in range(classes)])
            mid = 2 if sid == 6 else sid
            mask_seed = c['mask_seed_start'] + gid * 100 + mid
            observed = C.BASE.mask(task, mechanism, rate, mask_seed)
            value = C.observed_scale(task['X_full'], observed)
            poison = task['X_full'].copy(); poison[~observed] = 1e12
            assert np.array_equal(value, C.observed_scale(poison, observed))
            pcfi = C.BASE.fill(np.where(observed, value, np.nan), task['src'], task['dst'], 'pcfi', mechanism).clip(-10, 10)
            if d % 2:
                value = np.pad(value, ((0, 0), (0, 1))); pcfi = np.pad(pcfi, ((0, 0), (0, 1)))
                observed = np.pad(observed, ((0, 0), (0, 1)), constant_values=True)
            inp = C.Inputs(value, observed, pcfi, task['src'], task['dst'], context, task['y'][context], query, classes, mask_seed + 10000000)
            extra = S.prepare_geometry(inp, d)
            path = ROOT / 'bank/test' / f'g{gid}_e{sid}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **vars(inp), **extra, labels_query=task['y'][query])
            missing = ~observed[query, :d].any(1)
            if sid in [1, 2, 3, 4]: assert missing.any()
            records.append({'graph': gid, 'setting': sid, 'mechanism': mechanism, 'rate': rate, 'k': k,
                'nodes': n, 'features': d, 'classes': classes, 'queries': len(query), 'context': len(context),
                'path': path.relative_to(ROOT / 'bank').as_posix(), 'sha256': sha(path),
                'observed_cells': int(observed[:, :d].sum()), 'observed_nodes': int(observed[:, :d].any(1).sum()),
                'missing_queries': int(missing.sum()), 'invalid_retries': attempt, 'requested_seed': requested,
                'graph_seed': actual, 'mask_seed': mask_seed})
        if (gid + 1) % 8 == 0: print('BANK', gid + 1, round(time.perf_counter() - begun, 1), flush=True)
    dump(ROOT / 'bank/manifest.json', {'episodes': records, 'raw': raw, 'seconds': time.perf_counter() - begun,
         'design_sha256': sha(ROOT / 'results/design.json')})
    dump(ROOT / 'results/bank_lock.json', {'design_sha256': sha(ROOT / 'results/design.json'), 'manifest_sha256': sha(ROOT / 'bank/manifest.json')})
    rows(); print('FINAL_BANK_LOCKED', len(records), flush=True)


def evaluate(seed):
    C.setup(); torch.set_num_threads(2); test = rows()
    assert seed in cfg()['seeds']
    for arm in cfg()['arms']:
        folder = ROOT / 'results' / f'{arm}_s{seed}'
        assert not folder.exists(), 'No silent partial evaluation restart'
        (folder / 'predictions').mkdir(parents=True)
        model = load(arm, seed)
        before = {'trainable': C.parameter_digest(model, trainable=True), 'frozen': C.parameter_digest(model, trainable=False)}
        records = []; begun = time.perf_counter()
        for i, row in enumerate(test):
            inp, extra, y = read(row)
            assert sha(ROOT / 'bank' / row['path']) == row['sha256']
            p = predict(model, inp, extra)
            path = folder / 'predictions' / f'e{i}.npz'; np.savez_compressed(path, p=p, y=y)
            missing = ~inp.observed[inp.query, :row['features']].any(1)
            records.append({**row, 'arm': arm, 'seed': seed, **C.metrics(p, y),
                'missing_metrics': C.metrics(p[missing], y[missing]) if missing.any() else None,
                'prediction': path.relative_to(ROOT).as_posix(), 'prediction_sha256': sha(path), 'diagnostics': model.last_diagnostics})
            if (i + 1) % 112 == 0:
                dump(folder / 'progress.json', {'episodes': i + 1, 'seconds': time.perf_counter() - begun})
                print('EVAL', arm, seed, i + 1, flush=True)
        after = {'trainable': C.parameter_digest(model, trainable=True), 'frozen': C.parameter_digest(model, trainable=False)}
        assert before == after
        dump(folder / 'evaluation.json', records)
        dump(folder / 'evaluation_info.json', {'seconds': time.perf_counter() - begun, 'before': before, 'after': after, 'new_updates': 0})
        design(); del model; gc.collect(); torch.cuda.empty_cache()


def difference(a, b):
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(cfg()['bootstrap_seed'])
    boot = d[rng.integers(len(d), size=(cfg()['bootstrap_repeats'], len(d)))].mean(1)
    return {m: {'delta': float(d[:, j].mean()), 'ci95': np.quantile(boot[:, j], [.025, .975]).tolist()} for j, m in enumerate(METRICS)}


def collect():
    test = rows(); data = {}
    for arm in cfg()['arms']:
        data[arm] = {}
        for seed in cfg()['seeds']:
            a = json.loads((ROOT / 'results' / f'{arm}_s{seed}' / 'evaluation.json').read_text(encoding='utf8'))
            assert len(a) == len(test)
            assert all((x['graph'], x['setting'], x['sha256']) == (y['graph'], y['setting'], y['sha256']) for x, y in zip(a, test))
            data[arm][seed] = a
    return test, data


def summarize():
    test, data = collect(); c = cfg(); output = {}
    selectors = {'primary': lambda r: r['setting'] in [1, 2, 3, 4], 'full': lambda r: r['setting'] == 0,
        'cell': lambda r: r['setting'] == 5, 'k1': lambda r: r['setting'] == 6}
    for sid in [1, 2, 3, 4]: selectors[f'setting_{sid}'] = lambda r, s=sid: r['setting'] == s
    for name, choose in selectors.items():
        v = {}; allv = {}; per_seed = {}
        for arm in c['arms']:
            byseed = []; allbyseed = []
            for seed in c['seeds']:
                a = []; b = []
                for gid in range(c['graphs']):
                    rec = [data[arm][seed][i] for i, r in enumerate(test) if r['graph'] == gid and choose(r)]
                    selected = [r if name == 'full' else r['missing_metrics'] for r in rec]
                    valid = [r for r in selected if r is not None]; assert valid
                    a.append(np.mean([[r[m] for m in METRICS] for r in valid], axis=0))
                    b.append(np.mean([[r[m] for m in METRICS] for r in rec], axis=0))
                byseed.append(a); allbyseed.append(b)
            byseed = np.asarray(byseed); v[arm] = byseed.mean(0); allv[arm] = np.asarray(allbyseed).mean(0)
            per_seed[arm] = {str(s): dict(zip(METRICS, byseed[j].mean(0).tolist())) for j, s in enumerate(c['seeds'])}
        output[name] = {'graphs': c['graphs'], 'means': {a: dict(zip(METRICS, x.mean(0).tolist())) for a, x in v.items()},
            'contrast': difference(v['internal_fixed'], v['external']), 'per_seed': per_seed,
            'all_query_secondary': {a: dict(zip(METRICS, x.mean(0).tolist())) for a, x in allv.items()}}
    primary = output['primary']['contrast']['accuracy']['ci95'][0] > 0
    guard = output['full']['contrast']['accuracy']['ci95'][0] > -.02
    priors = {}
    for method in ['uniform', 'context_frequency']:
        scores = []
        for gid in range(c['graphs']):
            temp = []
            for row in test:
                if row['graph'] != gid or row['setting'] not in [1, 2, 3, 4]: continue
                inp, _, y = read(row); classes = int(inp.classes)
                f = np.ones(classes) if method == 'uniform' else np.bincount(inp.labels_context, minlength=classes).astype(float)
                p = np.broadcast_to(f / f.sum(), (len(y), classes))
                missing = ~inp.observed[inp.query, :row['features']].any(1)
                temp.append([C.metrics(p[missing], y[missing])[m] for m in METRICS])
            scores.append(np.mean(temp, axis=0))
        priors[method] = dict(zip(METRICS, np.mean(scores, axis=0).tolist()))
    dump(ROOT / 'results/statistics.json', {'subsets': output, 'secondary_priors': priors,
        'primary_confirmed': bool(primary), 'full_guard': bool(guard), 'continue_to_FP_FISF': bool(primary and guard),
        'unit': '128 independent base graphs, fixed training seeds11/12/13. Development-selected candidate; confirmation bank only. No new training-seed population or schema claims.'})
    print('SUMMARY', json.dumps(output['primary']), 'CONTINUE', bool(primary and guard), flush=True)


def audit():
    C.setup(); torch.set_num_threads(2); test, data = collect()
    manifest = json.loads((ROOT / 'bank/manifest.json').read_text(encoding='utf8'))
    for row in test + manifest['raw']: assert sha(ROOT / 'bank' / row['path']) == row['sha256']
    # Compare graph/feature/label array fingerprints, independent of NPZ container timestamps.
    import hashlib
    def fingerprint(path):
        with np.load(path, allow_pickle=True) as z:
            h = hashlib.sha256()
            for key in ['X_full', 'y', 'src', 'dst']:
                a = z[key]; h.update(key.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
            return h.hexdigest()
    oldfp = {fingerprint(p) for p in (OLD / 'bank/raw').glob('*/*.npz')}
    newfp = [fingerprint(ROOT / 'bank' / r['path']) for r in manifest['raw']]
    assert len(set(newfp)) == cfg()['graphs'] and not oldfp.intersection(newfp)
    replays = []; predictions = 0
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            folder = ROOT / 'results' / f'{arm}_s{seed}'
            info = json.loads((folder / 'evaluation_info.json').read_text(encoding='utf8'))
            assert info['before'] == info['after'] and info['new_updates'] == 0
            model = load(arm, seed)
            for i, record in enumerate(data[arm][seed]):
                path = ROOT / record['prediction']; assert sha(path) == record['prediction_sha256']
                inp, extra, y = read(test[i])
                with np.load(path) as z: p = z['p']; assert np.array_equal(y, z['y'])
                assert np.isfinite(p).all() and (p >= 0).all() and np.max(abs(p.sum(1) - 1)) < 1e-5
                score = C.metrics(p, y)
                assert all(abs(score[m] - record[m]) < 1e-12 for m in METRICS)
                missing = ~inp.observed[inp.query, :test[i]['features']].any(1)
                if missing.any():
                    ms = C.metrics(p[missing], y[missing]); assert all(abs(ms[m] - record['missing_metrics'][m]) < 1e-12 for m in METRICS)
                assert record['diagnostics'] == {'observed_clamp_max': 0., 'propagation_steps': 100, 'correlations': 1, 'pfn_layers': 12}
                if i in [1, 452, 894]:
                    error = float(np.abs(predict(model, inp, extra) - p).max()); assert error < .004
                    replays.append({'arm': arm, 'seed': seed, 'episode': i, 'max_abs_error': error})
                predictions += 1
            del model; gc.collect(); torch.cuda.empty_cache()
    design()
    primary = [r for r in test if r['setting'] in [1, 2, 3, 4]]
    dump(ROOT / 'results/audit.json', {'source_bank_checkpoint_hashes_valid': True, 'new_training_updates': 0,
        'new_models': 0, 'reused_checkpoints': 6, 'old_graph_fingerprints': len(oldfp), 'new_graph_fingerprints': len(newfp),
        'graph_overlap': 0, 'predictions_rescored': predictions, 'replays': replays,
        'primary_episodes': len(primary), 'zero_observation_primary_episodes': sum(r['observed_cells'] == 0 for r in primary),
        'nodes_range': [min(r['nodes'] for r in test), max(r['nodes'] for r in test)],
        'raw_features_range': [min(r['features'] for r in test), max(r['features'] for r in test)]})
    print('AUDIT_PASSED', predictions, flush=True)


def cost():
    C.setup(); torch.set_num_threads(2); test = rows(); measured = {}
    for arm in cfg()['arms']:
        model = load(arm, 11); groups = []
        for sid in [1, 2, 3, 4]:
            row = next(r for r in test if r['graph'] == 0 and r['setting'] == sid)
            inp, _, _ = read(row)
            def execute(): return predict(model, inp, S.prepare_geometry(inp, row['features']))
            for _ in range(5): execute()
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); times = []
            for _ in range(30):
                torch.cuda.synchronize(); start = time.perf_counter(); execute(); torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            groups.append({'setting': sid, 'median_ms': float(np.median(times)), 'times_ms': times,
                'additional_peak_mib': (torch.cuda.max_memory_allocated() - base) / 2**20})
        measured[arm] = {'mean_condition_median_ms': float(np.mean([r['median_ms'] for r in groups])), 'conditions': groups}
        del model; gc.collect(); torch.cuda.empty_cache()
    dump(ROOT / 'results/cost.json', {'methods': measured, 'scope': cfg()['cost']})
    print('COST_DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['lock', 'generate', 'evaluate', 'summary', 'audit', 'cost'])
    p.add_argument('--seed', type=int); a = p.parse_args()
    if a.action == 'evaluate': evaluate(a.seed)
    else: {'lock': lock_design, 'generate': generate, 'summary': summarize, 'audit': audit, 'cost': cost}[a.action]()

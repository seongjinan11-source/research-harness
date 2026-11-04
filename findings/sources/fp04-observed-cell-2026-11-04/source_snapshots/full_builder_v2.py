"""FP04 immutable NumPy banks. Preparation is explicit; importing creates no data.

Only read()'s seven-key whitelist may enter the model. Labels, generator laws,
reconstruction identities, and support diagnostics are preparation/audit data.
"""
from __future__ import annotations
import sklearn  # Windows runtime rule: import sklearn before any Torch import.
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
INPUT_KEYS = ('x', 'observed', 'pcfi', 'src', 'dst', 'context', 'classes')


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    """Create JSON exclusively. Partial/existing artifacts are never replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def _array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _normalization(raw, observed):
    raw, observed = np.asarray(raw), np.asarray(observed, bool)
    if raw.ndim != 2 or raw.shape != observed.shape:
        raise ValueError('raw and observed must have identical [N,F] shape')
    if not np.isfinite(raw[observed]).all():
        raise ValueError('Nonfinite observed value')
    visible = np.where(observed, raw, 0).astype(np.float64)
    count = observed.sum(0)
    rms = np.sqrt(np.square(visible).sum(0) / np.maximum(count, 1))
    scale = np.where((count > 0) & (rms > 0), rms, 1.)
    out = np.clip(visible / scale, -10., 10.).astype(np.float32)
    out[~observed] = 0.
    return out, dict(column_count=count.tolist(), column_rms=rms.tolist(),
                     scale=scale.tolist(), clip=10., convention='observed-only RMS')


def normalize_visible(raw, observed):
    return _normalization(raw, observed)[0]


_PCFI = None


def _pcfi_module():
    global _PCFI
    if _PCFI is None:
        path = WORK / 'fisf_graphpfn/imputation.py'
        spec = importlib.util.spec_from_file_location('_fp04_literal_pcfi', path)
        _PCFI = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PCFI)
    return _PCFI


def actual_pcfi(x, observed, src, dst, return_info=False):
    """Literal 100-step native CPU PCFI, preserving process RNG states.

    The vendor routine sees NaN exclusively at hidden cells. CUDA RNG states are
    touched only when CUDA was already initialized; preparation never starts it.
    """
    import torch
    x = np.asarray(x, np.float32)
    observed = np.asarray(observed, bool)
    if x.ndim != 2 or observed.shape != x.shape or not np.isfinite(x[observed]).all():
        raise ValueError('Invalid visible PCFI input')
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        if not observed.any():
            out = np.zeros_like(x)
            info = dict(method='pcfi', empty_zero_shortcut=True,
                        no_source_columns=x.shape[1], original_observed_max_change=0.)
        else:
            module = _pcfi_module()
            adj = module.adjacency(np.asarray([src, dst]), len(x), self_loops=True)
            structural = bool(np.all(observed == observed[:, :1]))
            out, info = module.impute(adj, np.where(observed, x, np.nan),
                                     method='pcfi', mode='structural' if structural else 'uniform',
                                     iterations=100, alpha=.9)
            out[observed] = x[observed]
            info['empty_zero_shortcut'] = False
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite PCFI output')
        info.update(wall_seconds=time.perf_counter() - wall,
                    cpu_seconds=time.process_time() - cpu, iterations=100, alpha=.9,
                    observed_clamp=True, rng_isolation=True, native_features=x.shape[1],
                    correlation_matrix_bytes=int(4 * x.shape[1] ** 2),
                    implementation='fisf_graphpfn/imputation.py',
                    implementation_sha256=sha(WORK / 'fisf_graphpfn/imputation.py'))
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return (out.astype(np.float32), info) if return_info else out.astype(np.float32)


def _rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _edges(n, probability, rng):
    i, j = np.triu_indices(n, 1)
    p = probability[i, j] if np.ndim(probability) else probability
    take = rng.random(len(i)) < p
    src = np.concatenate([i[take], j[take]]).astype(np.int64)
    dst = np.concatenate([j[take], i[take]]).astype(np.int64)
    order = np.lexsort((dst, src))
    return src[order], dst[order]


def _synthetic(seed, n, f, k, family, repetition):
    """One task-shared rule; no class balancing or outcome rejection."""
    gr, fr, lr = _rng(seed, 2), _rng(seed, 3), _rng(seed, 4)
    degree = int(gr.choice([4, 8, 12]))
    if family == 'centroid':
        y = _rng(seed, 1).integers(k, size=n, dtype=np.int64)
        eta = float(gr.uniform(.15, .8))
        p = degree / (n - 1) * np.where(y[:, None] == y[None], 1 + (k - 1) * eta, 1 - eta)
        src, dst = _edges(n, np.minimum(p, .98), gr)
        s, noise = float(fr.uniform(.6, 1.5)), float(fr.uniform(.35, .9))
        centroids = fr.normal(0, s, size=(k, f))
        law = fr.integers(4, size=f)
        signs = fr.choice([-1., 1.], size=f)
        scales = fr.lognormal(0, .4, size=f)
        bias = fr.uniform(-3, -1, size=f)
        x = np.empty((n, f), np.float64)
        for j in range(f):
            mu = centroids[y, j]
            if law[j] == 0:
                z = mu + noise * fr.normal(size=n)
            elif law[j] == 1:
                z = mu + noise * fr.standard_t(3, size=n) / np.sqrt(3.)
            elif law[j] == 2:
                z = noise * np.exp(.4 * mu) * fr.normal(size=n)
            else:
                prob = 1 / (1 + np.exp(-(2 * mu + bias[j])))
                x[:, j] = fr.random(n) < prob
                continue
            x[:, j] = signs[j] * scales[j] * z
        params = dict(degree=degree, eta=eta, centroid_scale=s, noise=noise,
                      column_law=law.tolist(), sign=signs.tolist(), scale=scales.tolist(),
                      centroid_sha256=_array_hash(centroids), bernoulli_bias=bias.tolist())
    elif family == 'field':
        src, dst = _edges(n, degree / (n - 1), gr)
        a = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
        a.setdiag(1)
        p = sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a
        rho = float(fr.uniform(.4, .85))
        z0 = fr.normal(size=(n, f)); z = z0.copy()
        for _ in range(5):
            z = (1 - rho) * z0 + rho * (p @ z)
        law = fr.integers(3, size=f)
        x = z.copy()
        x[:, law == 1] = z[:, law == 1] ** 3 / 3
        x[:, law == 2] = np.tanh(z[:, law == 2])
        x += fr.normal(0, .1, size=x.shape)
        w1 = lr.normal(size=(2 * f, 16)) / np.sqrt(2 * f)
        b1 = lr.normal(0, .2, size=16)
        w2 = lr.normal(size=(16, k)) / np.sqrt(16.)
        b2 = lr.normal(0, .2, size=k)
        hidden = np.tanh(np.concatenate([x, p @ x], 1) @ w1 + b1)
        y = (hidden @ w2 + b2 + lr.gumbel(0, .2, size=(n, k))).argmax(1).astype(np.int64)
        params = dict(degree=degree, rho=rho, column_law=law.tolist(),
                      rule_sha256=_array_hash(w1, b1, w2, b2), hidden_activation='tanh',
                      bias_sd=.2, gumbel_scale=.2, propagation_steps=5)
    else:
        raise ValueError(family)
    cr = _rng(seed, 5)
    cap = int(cr.choice(cfg()['context_per_class']))
    context_ids = np.concatenate([cr.permutation(np.flatnonzero(y == c))[:cap] for c in range(k)])
    context_ids = cr.permutation(context_ids).astype(np.int64)
    remaining = np.setdiff1d(np.arange(n), context_ids)
    if len(remaining) < cfg()['queries']:
        raise ValueError('N insufficient for fixed query count')
    query = cr.choice(remaining, size=cfg()['queries'], replace=False).astype(np.int64)
    context = np.full(n, -1, np.int64); context[context_ids] = y[context_ids]
    return dict(raw=x.astype(np.float32), labels=y, src=src, dst=dst, context=context,
                context_ids=context_ids, query=query, classes=k, seed=int(seed),
                disjoint=bool(repetition % 2), params=params, context_cap=cap)


def _mask(base, setting):
    n, f = base['raw'].shape
    eligible = np.ones(n, bool); eligible[base['query']] = False
    if base['disjoint']:
        eligible[base['context_ids']] = False
    if setting == 'none':
        return np.zeros((n, f), bool)
    if setting == 'full':
        return np.broadcast_to(eligible[:, None], (n, f)).copy()
    rate = {'node90': .9, 'node95': .95, 'node995': .995,
            'cell95': .95, 'cell995': .995}[setting]
    if setting.startswith('node'):
        keep = _rng(base['seed'], 6).random(n) >= rate
        return np.broadcast_to((eligible & keep)[:, None], (n, f)).copy()
    return eligible[:, None] & (_rng(base['seed'], 7).random((n, f)) >= rate)


def shuffle_visible(raw, observed, seed, whole_node=True):
    """Value-only intervention; never alters mask/query/context."""
    out = np.where(observed, raw, 0).copy()
    rng = _rng(seed, 8)
    if whole_node:
        if not np.all(observed == observed[:, :1]):
            raise ValueError('Whole-node shuffle requires whole-node mask')
        ids = np.flatnonzero(observed.any(1))
        out[ids] = out[rng.permutation(ids)]
    else:
        for j in range(out.shape[1]):
            ids = np.flatnonzero(observed[:, j])
            out[ids, j] = out[rng.permutation(ids), j]
    return out


def _support(observed, context, labels, k):
    columns = observed.sum(0); source = observed.any(1)
    return dict(sources=int(source.sum()), observed_cells=int(observed.sum()),
                source_bin='0' if source.sum() == 0 else '1' if source.sum() == 1 else '2+',
                columns_support0=int((columns == 0).sum()),
                columns_support1=int((columns == 1).sum()),
                columns_support2plus=int((columns >= 2).sum()),
                source_context_overlap=int((source & (context >= 0)).sum()),
                source_class_counts=np.bincount(labels[source], minlength=k).tolist())


def _write_episode(folder, base, meta, setting, variant='standard', raw_override=None, observed=None):
    observed = _mask(base, setting) if observed is None else observed
    raw = base['raw'] if raw_override is None else raw_override
    if variant == 'shuffle':
        raw = shuffle_visible(raw, observed, base['seed'], setting.startswith('node'))
    x, norm = _normalization(raw, observed)
    pcfi, timing = actual_pcfi(x, observed, base['src'], base['dst'], return_info=True)
    query = base['query']; labels = base['labels']
    assert not observed[query].any()
    if base['disjoint']:
        assert not observed[base['context_ids']].any()
    destination = folder / f"{meta['base_id']}_{setting}_{variant}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open('xb') as f:
        np.savez_compressed(f, x=x, observed=observed, pcfi=pcfi,
                            src=base['src'], dst=base['dst'], context=base['context'],
                            classes=np.int64(base['classes']), query=query, y=labels[query],
                            context_ids=base['context_ids'], labels_context=labels[base['context_ids']])
    support = _support(observed, base['context'], labels, base['classes'])
    return dict(**meta, setting=setting, variant=variant,
                path=destination.relative_to(ROOT).as_posix(), sha256=sha(destination),
                nodes=len(x), features=x.shape[1], classes=base['classes'],
                disjoint=base['disjoint'], queries=len(query),
                context_count=len(base['context_ids']), **support,
                query_truth_sha256=_array_hash(query, labels[query]),
                context_sha256=_array_hash(base['context_ids'], labels[base['context_ids']]),
                graph_sha256=_array_hash(base['src'], base['dst']),
                visible_sha256=_array_hash(x, observed), normalization=norm, pcfi=timing)


def _design(split):
    c = cfg(); start = c[split]['start_seed']
    if split == 'train':
        for i in range(c['train']['graphs']):
            r = _rng(start + i, 0)
            yield i, int(r.choice(c['train']['nodes'])), int(r.choice(c['train']['features'])), \
                int(r.choice(c['train']['classes'])), ['centroid', 'field'][i % 2], i // 2, 'train'
    else:
        i = 0
        for shape, (n, f) in zip(['matched', 'heldout'], c['final']['shapes']):
            for k in c['final']['classes']:
                for family in c['final']['families']:
                    reps = range(c['final']['repetitions']) if split == 'final' else [i // 2]
                    for rep in reps:
                        yield i, n, f, k, family, rep, shape
                        i += 1


def prepare(split='train'):
    if split not in ('train', 'dev', 'final', 'real'):
        raise ValueError(split)
    folder = ROOT / 'bank' / split
    if folder.exists():
        raise FileExistsError(f'Refuse existing/partial split: {folder}')
    folder.mkdir(parents=True)
    if split == 'real':
        return _prepare_real(folder)
    wall, cpu = time.perf_counter(), time.process_time()
    c = cfg(); episodes = []; bases = []
    for i, n, f, k, family, rep, shape in _design(split):
        seed = c[split]['start_seed'] + i
        base = _synthetic(seed, n, f, k, family, rep)
        base_id = f'{split}_{i:04d}'
        meta = dict(base_id=base_id, split=split, stratum=f'{family}_{shape}_K{k}',
                    family=family, shape=shape, repetition=int(rep), seed=seed)
        for setting in c['settings']:
            episodes.append(_write_episode(folder, base, meta, setting))
        if split == 'final':
            for setting in c['primary_settings']:
                episodes.append(_write_episode(folder, base, meta, setting, 'shuffle'))
        bases.append(dict(**meta, nodes=n, features=f, classes=k,
                          raw_sha256=_array_hash(base['raw']), labels_sha256=_array_hash(base['labels']),
                          class_counts=np.bincount(base['labels'], minlength=k).tolist(),
                          context_cap=base['context_cap'], context_count=len(base['context_ids']),
                          disjoint=base['disjoint'], parameters=base['params']))
        if (i + 1) % 12 == 0:
            print('PREPARE', split, i + 1, len(episodes), flush=True)
    expected = c[split]['graphs'] * (9 if split == 'final' else 7)
    assert len(episodes) == expected
    manifest = dict(cycle='FP04', split=split, protocol_sha256=sha(ROOT / 'protocol.json'),
                    data_source_sha256=sha(__file__), episodes=episodes, bases=bases,
                    rng='SeedSequence([base_seed, stream]); streams 0shape 1labels 2graph 3features 4rule 5context/query 6node_mask 7cell_mask 8shuffle',
                    optimizer_updates=0, prepare_wall_seconds=time.perf_counter() - wall,
                    prepare_cpu_seconds=time.process_time() - cpu,
                    pcfi_wall_seconds=sum(r['pcfi']['wall_seconds'] for r in episodes))
    dump(folder / 'manifest.json', manifest)
    return manifest


def rows(split):
    path = ROOT / 'bank' / split / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['protocol_sha256'] != sha(ROOT / 'protocol.json'):
        raise ValueError('Bank/protocol mismatch')
    return manifest['episodes']


def read(row):
    path = ROOT / row['path']
    if sha(path) != row['sha256']:
        raise ValueError(f'Bank hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        inp = {key: (int(z[key]) if key == 'classes' else z[key].copy()) for key in INPUT_KEYS}
        query, y = z['query'].astype(np.int64), z['y'].astype(np.int64)
    return inp, query, y


def _canonical_edges(edge, n):
    edge = np.asarray(edge, np.int64)
    a = sp.coo_matrix((np.ones(edge.shape[1], np.float32), edge), shape=(n, n)).tocsr()
    a = a.maximum(a.T)
    a.setdiag(0); a.eliminate_zeros(); a.sort_indices()
    return tuple(np.asarray(v, np.int64) for v in a.nonzero())


def _read_raw(path, expected_hash):
    """Read the immutable native PyG source in the existing research environment."""
    import torch
    path = Path(str(path).replace('\\\\', '\\'))
    if sha(path) != expected_hash:
        raise ValueError(f'Native raw source changed: {path}')
    obj = torch.load(path, map_location='cpu', weights_only=False)
    raw = obj[0] if isinstance(obj, (tuple, list)) else obj
    def array(key):
        value = raw[key] if isinstance(raw, dict) else getattr(raw, key)
        return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    x = array('x').astype(np.float32)
    _, y = np.unique(array('y'), return_inverse=True)
    src, dst = _canonical_edges(array('edge_index'), len(y))
    return dict(x=x, y=y.astype(np.int64), src=src, dst=dst,
                train=array('train_mask').astype(bool), test=array('test_mask').astype(bool),
                path=str(path), sha256=expected_hash)


def _legacy_scale(full, mask):
    """Original RP16 RMS convention, solely for reconstruction identity checks."""
    safe = np.where(mask, full, 0).astype(np.float64)
    rms = np.sqrt((safe * safe).sum(0) / np.maximum(mask.sum(0), 1))
    rms = np.where(rms >= 1e-8, rms, 1)
    return np.clip(safe / rms, -10, 10).astype(np.float32)


def _real_base(oldrow, legacy_root, raw, legacy_config):
    path = legacy_root / oldrow['path']
    if sha(path) != oldrow['sha256']:
        raise ValueError(f'Legacy bank changed: {path}')
    with np.load(path, allow_pickle=False) as z:
        old = {key: z[key].copy() for key in z.files}
    name, rep, mode = oldrow['graph'], oldrow['repetition'], oldrow['mode']
    source_mask = old['observed'].any(1)
    ids = np.flatnonzero(source_mask)
    context_ids = old['context_ids'].astype(np.int64)
    query = old['query'].astype(np.int64)
    if name in ('Cora', 'CiteSeer'):
        n, f = raw['x'].shape
        y = raw['y']
        if not (np.array_equal(raw['src'], old['src']) and np.array_equal(raw['dst'], old['dst'])):
            raise ValueError('Raw/legacy graph mismatch')
        rng = np.random.default_rng(int(old['forward_seed']))
        train = raw['train'] if raw['train'].ndim == 1 else raw['train'][:, rep]
        test = raw['test'] if raw['test'].ndim == 1 else raw['test'][:, rep]
        ctx = np.concatenate([rng.permutation(np.flatnonzero(train & (y == lab)))
                              [:legacy_config['real_context_per_class']]
                              for lab in range(int(old['classes']))])
        ctx = rng.permutation(ctx).astype(np.int64)
        order = rng.permutation(np.setdiff1d(np.arange(n), ctx))
        rate = {'95': .95, '99': .99, '995': .995, 'none': 1., 'shuffle95': .95}[mode]
        replay_ids = np.sort(order[:round(n * (1 - rate))])
        source95 = order[:round(.05 * n)]
        replay_query = np.setdiff1d(np.flatnonzero(test), source95)
        if not (np.array_equal(ctx, context_ids) and np.array_equal(replay_ids, ids)
                and np.array_equal(replay_query, query)):
            raise ValueError('Context/source/query RNG replay mismatch')
        value_ids = rng.permutation(ids) if mode == 'shuffle95' else ids.copy()
        visible = np.zeros((n, f), np.float32)
        visible[ids] = raw['x'][value_ids]
        rp_rng = np.random.default_rng(20461104 + len(name))
        projection = rp_rng.choice([-1., 1.], size=(f, legacy_config['features'])).astype(np.float32) / np.sqrt(f)
        rp_input = _legacy_scale(visible @ projection, old['observed'])
        original_ids = np.arange(n, dtype=np.int64)
        original_source_ids = ids
        original_value_ids = value_ids
        panel = 'full'
    else:
        n, f = old['native'].shape
        original_ids = old['original_ids'].astype(np.int64)
        y = raw['y'][original_ids]
        original_source_ids = old['source_original_ids'].astype(np.int64)
        original_value_ids = old['value_original_ids'].astype(np.int64)
        if not np.array_equal(original_ids[ids], original_source_ids):
            raise ValueError('PubMed original source mapping mismatch')
        visible = np.zeros((n, f), np.float32)
        visible[ids] = raw['x'][original_value_ids]
        if not np.array_equal(old['native'][ids], visible[ids]):
            raise ValueError('PubMed native/value-original identity mismatch')
        native_mask = np.broadcast_to(source_mask[:, None], (n, f))
        if not np.array_equal(np.isfinite(old['native']), native_mask):
            raise ValueError('PubMed native missingness mismatch')
        # Verify the inherited induced edge set against immutable original IDs.
        lookup = np.full(len(raw['y']), -1, np.int64); lookup[original_ids] = np.arange(n)
        keep = (lookup[raw['src']] >= 0) & (lookup[raw['dst']] >= 0)
        src, dst = _canonical_edges(np.asarray([lookup[raw['src'][keep]], lookup[raw['dst'][keep]]]), n)
        if not (np.array_equal(src, old['src']) and np.array_equal(dst, old['dst'])):
            raise ValueError('PubMed induced graph mismatch')
        rp_input = _legacy_scale(visible @ old['projection'], old['observed'])
        panel = 'inherited_induced_2000'
    rp_error = float(np.max(np.abs(rp_input - old['x'])))
    if rp_error > 1e-5:
        raise ValueError(f'Native→legacy RP16 reconstruction mismatch: {rp_error}')
    k = int(old['classes'])
    if not (np.array_equal(y[context_ids], old['labels_context']) and
            np.array_equal(y[query], old['labels_query'])):
        raise ValueError('Native/legacy label identity mismatch')
    observed = np.broadcast_to(source_mask[:, None], (n, f)).copy()
    if observed[context_ids].any() or observed[query].any():
        raise ValueError('Legacy context/query unexpectedly observed')
    context = np.full(n, -1, np.int64); context[context_ids] = old['labels_context']
    base = dict(raw=visible, labels=y, src=old['src'], dst=old['dst'], context=context,
                context_ids=context_ids, query=query, classes=k,
                seed=int(old['forward_seed']), disjoint=True)
    identity = dict(legacy_bank_path=str(path), legacy_bank_sha256=oldrow['sha256'],
                    raw_source_path=raw['path'], raw_source_sha256=raw['sha256'],
                    native_features=f, legacy_rp_features=old['x'].shape[1],
                    rp_reconstruction_max_error=rp_error,
                    original_ids_sha256=_array_hash(original_ids),
                    source_original_ids_sha256=_array_hash(original_source_ids),
                    value_original_ids_sha256=_array_hash(original_value_ids),
                    query_truth_context_identity=True, native_value_identity=True,
                    graph_identity=True, panel=panel)
    return base, observed, identity


def _reference_metrics(p, y):
    p = np.asarray(p, np.float64)
    if p.ndim != 2 or len(p) != len(y) or not np.isfinite(p).all() or p.min() < -1e-8:
        raise ValueError('Invalid external probabilities')
    if not np.allclose(p.sum(1), 1, atol=3e-5):
        raise ValueError('External probabilities not normalized')
    p = np.clip(p, 1e-8, 1); p /= p.sum(1, keepdims=True)
    pred = p.argmax(1); recalls = []; f1 = []
    for c in range(p.shape[1]):
        truth, guess = y == c, pred == c
        tp = int(np.sum(truth & guess)); den = int(truth.sum() + guess.sum())
        if truth.any():
            recalls.append(float(tp / truth.sum()))
        f1.append(2 * tp / den if den else 0.)
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(np.mean(recalls)),
                macro_f1=float(np.mean(f1)), nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float((np.square(p).sum(1) - 2 * p[np.arange(len(y)), y] + 1).mean()))


def _external_for_row(row, base, identity):
    arms = ['lp_harmonic', 'lp_restart', 'public_graphpfn',
            'pcfi_public_graphpfn', 'fisf_public_graphpfn']
    records, missing = [], []
    folder = WORK / 'factor_edge_only_benchmark/results/predictions'
    for arm in arms:
        path = folder / f"{row['graph']}_r{row['repetition']}_{row['mode']}_{arm}.npz"
        sidecar = path.with_suffix('.json')
        if not path.exists() or not sidecar.exists():
            missing.append(dict(graph=row['graph'], repetition=row['repetition'], mode=row['mode'],
                                arm=arm, identity_pass=False, reason='missing frozen reference', path=str(path)))
            continue
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
        pred_sha = sha(path)
        if pred_sha != meta['prediction_sha256'] or meta['sha256'] != identity['legacy_bank_sha256']:
            raise ValueError(f'External provenance mismatch: {path}')
        if (meta['graph'], meta['repetition'], meta['mode'], meta['arm']) != \
                (row['graph'], row['repetition'], row['mode'], arm):
            raise ValueError(f'External row association mismatch: {path}')
        with np.load(path, allow_pickle=False) as z:
            if not (np.array_equal(z['query'], base['query']) and
                    np.array_equal(z['y'], base['labels'][base['query']]) and
                    np.array_equal(z['context'], base['context_ids']) and
                    np.array_equal(z['yc'], base['labels'][base['context_ids']])):
                raise ValueError(f'External query/truth/context mismatch: {path}')
            if z['p'].shape != (len(base['query']), base['classes']):
                raise ValueError('External class axis mismatch')
            metrics = _reference_metrics(z['p'], z['y'])
        records.append(dict(graph=row['graph'], repetition=row['repetition'], mode=row['mode'],
                            arm=arm, metrics=metrics, path=str(path), sha256=pred_sha,
                            sidecar_path=str(sidecar), sidecar_sha256=sha(sidecar),
                            identity_pass=True, identity_scope='graph/context/query/truth and native observed values; preprocessing differs',
                            input_representation='graph/context only' if arm.startswith('lp_') else 'legacy native raw features with public preprocessing',
                            preprocessing_matched=bool(arm.startswith('lp_')),
                            legacy_bank_path=identity['legacy_bank_path'],
                            legacy_bank_sha256=identity['legacy_bank_sha256'],
                            native_reconstruction_verified=True,
                            query_truth_sha256=row['query_truth_sha256'], context_sha256=row['context_sha256'],
                            target_optimizer_updates=0))
    return records, missing


def _prepare_real(folder):
    wall, cpu = time.perf_counter(), time.process_time()
    raw_wall = time.perf_counter()
    pilot = WORK / 'factor_real_graph_pilot'
    hom = WORK / 'factor_homophily_benchmark'
    pilot_manifest = json.loads((pilot / 'bank/manifest.json').read_text(encoding='utf-8'))
    hom_manifest = json.loads((hom / 'bank/manifest.json').read_text(encoding='utf-8'))
    pilot_config = json.loads((pilot / 'protocol.json').read_text(encoding='utf-8'))
    legacy_rows = [(r, pilot) for r in pilot_manifest['episodes']
                   if r['split'] == 'real' and r['graph'] in ('Cora', 'CiteSeer')]
    legacy_rows += [(r, hom) for r in hom_manifest['episodes'] if r['graph'] == 'PubMed']
    bykey = {(r['graph'], r['repetition'], r['mode']): (r, p) for r, p in legacy_rows}
    if len(bykey) != 45 or len(legacy_rows) != 45:
        raise ValueError('Expected exactly 45 inherited real episodes')
    raw = {}
    for graph in cfg()['real']['graphs']:
        if graph in ('Cora', 'CiteSeer'):
            rawpaths = pilot_manifest['real_sources'][graph]
            if len(rawpaths) != 1:
                raise ValueError('Unexpected native raw source count')
            path, expected = next(iter(rawpaths.items()))
        else:
            desc = hom_manifest['datasets']['PubMed']
            path, expected = desc['raw_path'], desc['raw_sha256']
        raw[graph] = _read_raw(path, expected)
    raw_seconds = time.perf_counter() - raw_wall
    episodes, references, missing_refs, identities = [], [], [], []
    for graph in cfg()['real']['graphs']:
        for rep in cfg()['real']['repetitions']:
            for mode in cfg()['real']['modes']:
                oldrow, legacy_root = bykey[(graph, rep, mode)]
                base, observed, identity = _real_base(oldrow, legacy_root, raw[graph], pilot_config)
                meta = dict(base_id=f'{graph}_r{rep}', split='real', graph=graph,
                            stratum=graph, family='real', shape=identity['panel'],
                            repetition=rep, mode=mode, seed=base['seed'])
                # Legacy shuffle is already reconstructed, so no second shuffle occurs.
                row = _write_episode(folder, base, meta, mode, 'standard', observed=observed)
                row['legacy_identity'] = identity
                episodes.append(row); identities.append(identity)
                good, missing = _external_for_row(row, base, identity)
                references.extend(good); missing_refs.extend(missing)
                print('PREPARE_REAL', graph, rep, mode, row['features'],
                      round(row['pcfi']['wall_seconds'], 3), flush=True)
    # Existing raw files are also checked after preparation, not merely at load.
    for source in raw.values():
        if sha(source['path']) != source['sha256']:
            raise ValueError('Native source changed during preparation')
    query_groups = {}
    for row in episodes:
        key = row['base_id']
        identity = row['query_truth_sha256'], row['context_sha256'], row['graph_sha256']
        if key in query_groups and query_groups[key] != identity:
            raise ValueError('Real modes changed context/query/truth/graph')
        query_groups[key] = identity
    dump(folder / 'references.json', references)
    manifest = dict(cycle='FP04', split='real', protocol_sha256=sha(ROOT / 'protocol.json'),
                    data_source_sha256=sha(__file__), episodes=episodes,
                    references_path='bank/real/references.json',
                    references_sha256=sha(folder / 'references.json'),
                    matched_reference_count=len(references), missing_references=missing_refs,
                    lineage={str(p): sha(p) for p in [pilot / 'bank/manifest.json', pilot / 'protocol.json',
                              hom / 'bank/manifest.json', WORK / 'factor_edge_only_benchmark/run.py']},
                    raw_sources={g:dict(path=r['path'], sha256=r['sha256']) for g, r in raw.items()},
                    native_raw_load_seconds=raw_seconds,
                    prepare_wall_seconds=time.perf_counter() - wall,
                    prepare_cpu_seconds=time.process_time() - cpu,
                    pcfi_wall_seconds=sum(r['pcfi']['wall_seconds'] for r in episodes),
                    optimizer_updates=0, scope='reused real development graphs; Cora/CiteSeer full, PubMed inherited induced2000',
                    preprocessing='new arms: observed-only normalized native channels, actual PCFI on normalized channels; public references: legacy native raw features and public scaling/fill/PCA pipeline; LP: graph/context only',
                    identity_scope='same native observed values, graph/context/query/truth; processed feature representations and model/training budgets differ',
                    hidden_projection=False)
    dump(folder / 'manifest.json', manifest)
    return manifest


def external_records():
    path = ROOT / 'bank/real/references.json'
    manifest = json.loads((ROOT / 'bank/real/manifest.json').read_text(encoding='utf-8'))
    if sha(path) != manifest['references_sha256']:
        raise ValueError('Reference index changed')
    return json.loads(path.read_text(encoding='utf-8'))


def selfcheck(include_pcfi=False):
    raw = np.array([[0., 2., np.nan], [0., 4., 123.], [0., 999., np.nan]])
    obs = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], bool)
    x = normalize_visible(raw, obs)
    poison = raw.copy(); poison[~obs] = -8e20
    assert np.array_equal(x, normalize_visible(poison, obs))
    assert np.all(x[~obs] == 0) and np.isfinite(x).all()
    assert np.allclose(x[:2, 1], [2, 4] / np.sqrt(10))
    for family in ['centroid', 'field']:
        base = _synthetic(91431, 48, 4, 3, family, 1)
        masks = {s: _mask(base, s) for s in cfg()['settings']}
        assert not masks['full'][base['query']].any()
        assert not masks['full'][base['context_ids']].any()
        assert np.all(masks['node995'] <= masks['node95'])
        assert np.all(masks['node95'] <= masks['node90'])
        assert np.all(masks['cell995'] <= masks['cell95'])
        assert _array_hash(base['raw']) == _array_hash(_synthetic(91431, 48, 4, 3, family, 1)['raw'])
    if include_pcfi:
        import torch
        pystate, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cx = np.array([[1., 0.], [0., 0.], [0., 2.]], np.float32)
        cm = np.array([[1, 0], [0, 0], [0, 1]], bool)
        out, info = actual_pcfi(cx, cm, [0, 1, 1, 2], [1, 0, 2, 1], True)
        assert np.array_equal(out[cm], cx[cm]) and np.isfinite(out).all()
        assert random.getstate() == pystate
        now = np.random.get_state()
        assert now[0] == npstate[0] and np.array_equal(now[1], npstate[1]) and now[2:] == npstate[2:]
        assert torch.equal(ts, torch.get_rng_state())
    return dict(status='PASS', bank_created=False, real_predictions_used=False,
                gpu_used=False, pcfi_tested=bool(include_pcfi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'selfcheck'])
    parser.add_argument('--split', choices=['train', 'dev', 'final', 'real'], default='train')
    parser.add_argument('--pcfi', action='store_true', help='CPU tiny PCFI RNG/clamp test')
    args = parser.parse_args()
    print(json.dumps(prepare(args.split) if args.action == 'prepare' else selfcheck(args.pcfi),
                     ensure_ascii=False, default=str))

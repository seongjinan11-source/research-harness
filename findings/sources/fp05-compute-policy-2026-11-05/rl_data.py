"""FP05 create-only physical observations and explicit model-input whitelist.

Preparation is an explicit root-owned action. Importing does not sample tasks,
load real arrays/checkpoints, create banks, or initialize a GPU model.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows import order, before Torch in rl_util.
import argparse
from collections import deque
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import scipy.sparse as sp
from rl_util import ROOT, WORK, MG, cfg, setup, sha, dump

INPUT_KEYS = ('values', 'observed', 'src', 'dst', 'context_ids',
              'context_labels', 'query_ids', 'classes', 'forward_seed')
SETTINGS = ('full', 'node90', 'node95', 'node995', 'cluster95', 'cluster995', 'cell95')
_FP04 = None


def _fp04():
    global _FP04
    if _FP04 is None:
        path = WORK / 'observed_cell_cycle/data.py'
        spec = importlib.util.spec_from_file_location('_fp05_readonly_fp04_data', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FP04 = module
    return _FP04


def dependencies(include_prior=False):
    """Imported local scientific dependencies, also recorded in each manifest."""
    paths = [Path(__file__), ROOT / 'rl_util.py', ROOT / 'protocol.json', ROOT / 'PLAN_v2.md',
             WORK / 'observed_cell_cycle/data.py', WORK / 'fisf_graphpfn/imputation.py',
             MG / 'mgpfn/upstream.py', MG / 'mgpfn/prior_adapter.py']
    if include_prior:
        from mgpfn import upstream as U
        paths += [U.SRC / 'paper/exp/graphpfn/pretrain/main/pretrain.toml',
                  U.E1 / 'graphpfn_windows_dev.py']
        paths += sorted((U.SRC / 'paper/lib/graphpfn/prior').rglob('*.py'))
    return {str(p.resolve()): sha(p) for p in paths}


def _rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _arrays_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode()); h.update(a.tobytes())
    return h.hexdigest()


def normalize_visible(raw, observed):
    return _fp04().normalize_visible(raw, observed)


def actual_pcfi(x, observed, src, dst, return_info=False):
    return _fp04().actual_pcfi(x, observed, src, dst, return_info=return_info)


def project_visible(raw, observed, projection):
    """Whole-row RP only. Hidden native values never enter a matrix product."""
    raw, observed = np.asarray(raw), np.asarray(observed, bool)
    projection = np.asarray(projection, np.float32)
    if raw.shape != observed.shape or raw.ndim != 2 or projection.shape[0] != raw.shape[1]:
        raise ValueError('Invalid native/RP shapes')
    if not np.all(observed == observed[:, :1]):
        raise ValueError('Partial native cells do not define observed RP coordinates')
    if not np.isfinite(raw[observed]).all() or not np.isfinite(projection).all():
        raise ValueError('Nonfinite observed native value/projection')
    visible = np.where(observed, raw, 0).astype(np.float32)
    projected = visible @ projection
    projected_mask = np.broadcast_to(observed.any(1)[:, None], projected.shape).copy()
    return projected, projected_mask


def _bfs_order(n, src, dst, seed):
    """Random BFS forest; disconnected components get independent random roots."""
    rng = _rng(seed, 7)
    adjacency = [[] for _ in range(n)]
    for i, j in zip(src, dst):
        if int(i) != int(j):
            adjacency[int(i)].append(int(j))
    roots = rng.permutation(n)
    seen = np.zeros(n, bool); result = []
    for root in roots:
        if seen[root]:
            continue
        queue = deque([int(root)]); seen[root] = True
        while queue:
            i = queue.popleft(); result.append(i)
            for j in rng.permutation(np.asarray(adjacency[i], np.int64)):
                if not seen[j]:
                    seen[j] = True; queue.append(int(j))
    return np.asarray(result, np.int64)


def make_mask(shape, src, dst, context_ids, query_ids, setting, seed, regime):
    """Mask labels never enter this API. Rates precede forced disjoint exclusion."""
    n, f = map(int, shape)
    if setting == 'full':
        observed = np.ones((n, f), bool)
    elif setting.startswith('node'):
        rate = {'node90': .9, 'node95': .95, 'node995': .995}[setting]
        keep = _rng(seed, 6).random(n) >= rate
        observed = np.broadcast_to(keep[:, None], (n, f)).copy()
    elif setting.startswith('cluster'):
        rate = {'cluster95': .95, 'cluster995': .995}[setting]
        keep = np.zeros(n, bool)
        keep[_bfs_order(n, src, dst, seed)[:int(round(n * (1 - rate)))]] = True
        observed = np.broadcast_to(keep[:, None], (n, f)).copy()
    elif setting == 'cell95':
        observed = _rng(seed, 8).random((n, f)) >= .95
    else:
        raise ValueError(setting)
    if regime == 'disjoint':
        observed[np.asarray(context_ids, np.int64)] = False
        observed[np.asarray(query_ids, np.int64)] = False
    elif regime != 'natural':
        raise ValueError(regime)
    return observed


def _intervene(raw, observed, setting, intervention, seed):
    if intervention == 'none':
        return np.zeros_like(raw, dtype=np.float32), np.zeros_like(observed)
    visible = np.where(observed, raw, 0).astype(np.float32)
    if intervention == 'standard':
        return visible, observed.copy()
    if intervention != 'shuffle':
        raise ValueError(intervention)
    rng = _rng(seed, 30 + SETTINGS.index(setting))
    if setting.startswith(('node', 'cluster')) or setting == 'full':
        if not np.all(observed == observed[:, :1]):
            raise ValueError('Row shuffle requires identical availability across columns')
        ids = np.flatnonzero(observed.any(1))
        visible[ids] = visible[rng.permutation(ids)]
    else:
        for j in range(visible.shape[1]):
            ids = np.flatnonzero(observed[:, j])
            visible[ids, j] = visible[rng.permutation(ids), j]
    return visible, observed.copy()


def _episode_arrays(base, setting, intervention='standard', standard_mask=None):
    raw = base['raw']; n, f = raw.shape
    standard = make_mask(raw.shape, base['src'], base['dst'], base['context_ids'],
                         base['query_ids'], setting, base['seed'], base['regime']) \
        if standard_mask is None else np.asarray(standard_mask, bool)
    visible, observed = _intervene(raw, standard, setting, intervention, base['seed'])
    x = normalize_visible(visible, observed)
    pcfi, timing = actual_pcfi(x, observed, base['src'], base['dst'], return_info=True)
    values = np.clip(pcfi, -cfg()['pcfi']['clip'], cfg()['pcfi']['clip']).astype(np.float32)
    q = base['query_ids']
    standard_missing = ~standard[q].any(1)
    arrays = dict(values=values, observed=observed, src=np.asarray(base['src'], np.int64),
                  dst=np.asarray(base['dst'], np.int64), context_ids=base['context_ids'],
                  context_labels=base['context_labels'], query_ids=q,
                  classes=np.int64(base['classes']), forward_seed=np.int64(base['forward_seed']),
                  y=np.asarray(base['y_query'], np.int64),
                  own_missing=standard_missing, standard_own_missing=standard_missing,
                  query_observed=~standard_missing,
                  actual_own_missing=~observed[q].any(1),
                  standard_source_nodes=np.int64(standard.any(1).sum()),
                  physical_features=np.int64(f), visible_values=x, pcfi_raw=pcfi)
    # No dummy columns are added: GraphPFNLight retains its own preprocessing.
    assert values.shape == observed.shape == (n, f)
    assert set(INPUT_KEYS).issubset(arrays) and 'y' not in INPUT_KEYS
    assert np.isfinite(values).all() and np.array_equal(pcfi[observed], x[observed])
    if base['regime'] == 'disjoint':
        assert not observed[base['context_ids']].any() and not observed[q].any()
    info = dict(source_nodes=int(observed.any(1).sum()),
                standard_source_nodes=int(standard.any(1).sum()),
                observed_cells=int(observed.sum()), standard_observed_cells=int(standard.sum()),
                standard_missing_queries=int(standard_missing.sum()),
                standard_observed_queries=int((~standard_missing).sum()),
                actual_missing_queries=int((~observed[q].any(1)).sum()),
                column_support0=int((observed.sum(0) == 0).sum()),
                column_support1=int((observed.sum(0) == 1).sum()),
                context_observed_nodes=int(observed[base['context_ids']].any(1).sum()),
                pcfi=timing, pcfi_raw_maxabs=float(np.abs(pcfi).max()) if pcfi.size else 0.,
                represented_clip_fraction=float((np.abs(pcfi) > cfg()['pcfi']['clip']).mean()))
    return arrays, info


def _save_episode(folder, base, meta, setting, intervention='standard', standard_mask=None):
    arrays, info = _episode_arrays(base, setting, intervention, standard_mask)
    row_id = f"{meta['base_id']}__{setting}__{intervention}"
    path = folder / (row_id + '.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(**meta, id=row_id, row_id=row_id, setting=setting, intervention=intervention,
                path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                nodes=len(arrays['values']), features=arrays['values'].shape[1],
                physical_features=int(arrays['physical_features']), classes=int(arrays['classes']),
                queries=len(arrays['query_ids']), context_count=len(arrays['context_ids']),
                forward_seed=int(arrays['forward_seed']),
                input_arrays_sha256=_arrays_hash(*(arrays[k] for k in INPUT_KEYS)),
                query_truth_sha256=_arrays_hash(arrays['query_ids'], arrays['y']),
                context_sha256=_arrays_hash(arrays['context_ids'], arrays['context_labels']),
                graph_sha256=_arrays_hash(arrays['src'], arrays['dst']),
                standard_membership_sha256=_arrays_hash(arrays['standard_own_missing']),
                **info)


def _design(split):
    c = cfg()
    if split == 'final':
        i = 0
        for shape in ('matched', 'heldout'):
            for k in c['context_per_class']:
                for regime in ('natural', 'disjoint'):
                    for rep in range(8):
                        seed = c['final']['seed_start'] + i
                        rng = _rng(seed, 0)
                        n = c['final']['heldout_nodes'] if shape == 'heldout' else int(rng.choice(c['train']['nodes']))
                        f = c['final']['heldout_features'] if shape == 'heldout' else int(rng.choice(c['train']['features']))
                        yield dict(index=i, seed=seed, requested_nodes=n, requested_features=f,
                                   k=int(k), regime=regime, shape=shape, repetition=rep)
                        i += 1
        assert i == c['final']['graphs']
    else:
        for i in range(c[split]['graphs']):
            seed = c[split]['seed_start'] + i
            rng = _rng(seed, 0)
            yield dict(index=i, seed=seed, requested_nodes=int(rng.choice(c['train']['nodes'])),
                       requested_features=int(rng.choice(c['train']['features'])),
                       k=int(c['context_per_class'][(i // 2) % len(c['context_per_class'])]),
                       regime=('natural', 'disjoint')[i % 2], shape='matched', repetition=i // 6)


def generator_config(nodes, features):
    from mgpfn import upstream as U
    config = deepcopy(U.prior_config())
    v = config['prior']['values'][0]
    v['graph']['n_nodes'] = int(nodes)
    v['graph']['avg_degree'] = {'_distribution_': 'uniform', 'min': 4., 'max': 12.}
    v['graph']['sampler']['values'][0]['min_first_level_n_nodes'] = 16
    v['scm']['base']['n_features'] = int(features)
    v['task']['n_classes']['min'] = 2
    v['task']['n_classes']['max'] = 4
    # Other official task/SCM parameters, including quantiles, are unchanged.
    return config


def _sample(spec, config):
    from mgpfn import prior_adapter as PA
    failures = []
    for attempt in range(50):
        actual_seed = spec['seed'] + attempt * 100000000
        try:
            task = PA.sample_official_task(config, actual_seed,
                                          allowed_task_types=('binclass', 'multiclass'))
        except AssertionError as exc:
            failures.append(dict(attempt=attempt, actual_seed=actual_seed,
                                 kind='official structural/sanity AssertionError', message=str(exc)))
            continue
        break
    else:
        raise RuntimeError(f"50 invalid structural draws for {spec['seed']}: {failures}")
    _, labels = np.unique(task['y'], return_inverse=True)
    labels = labels.astype(np.int64)
    raw = np.asarray(task['X_full'], np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('Official task has nonfinite physical features; no outcome resampling')
    n = len(raw); n_train = int(task['n_train']); classes = len(np.unique(labels))
    if not 0 < n_train < n or classes < 1 or classes > 4:
        raise ValueError('Invalid official task definition; preserve failure instead of selective rejection')
    rng = _rng(spec['seed'], 5)
    context = np.concatenate([rng.permutation(np.flatnonzero(labels[:n_train] == lab))[:spec['k']]
                              for lab in range(classes)]).astype(np.int64)
    context = rng.permutation(context)
    query = rng.permutation(np.arange(n_train, n))[:cfg()['queries']].astype(np.int64)
    assert len(query) and not np.intersect1d(context, query).size
    src, dst = np.asarray(task['src'], np.int64), np.asarray(task['dst'], np.int64)
    # Retain literal official edge records; every arm receives this same graph.
    base = dict(raw=raw, src=src, dst=dst, context_ids=context,
                context_labels=labels[context], query_ids=query, y_query=labels[query],
                classes=classes, seed=spec['seed'], regime=spec['regime'],
                forward_seed=int(_rng(spec['seed'], 90).integers(1, 2**31 - 1)))
    private = dict(X_full=raw, y_full=labels, src=src, dst=dst,
                   n_train=np.int64(n_train), task_type=np.asarray(task['task_type']),
                   context_ids=context, query_ids=query, classes=np.int64(classes))
    audit = dict(actual_seed=actual_seed, structural_retries=attempt, retry_records=failures,
                 task_type_rejections_in_successful_call=int(task.get('rejected', 0)),
                 actual_nodes=n, actual_features=raw.shape[1], actual_classes=classes,
                 requested_classes_max=4, n_train=n_train, task_type=task['task_type'],
                 self_edges=int((src == dst).sum()),
                 duplicate_directed_edges=int(len(src) - len(np.unique(np.column_stack([src, dst]), axis=0))),
                 class_counts=np.bincount(labels, minlength=classes).tolist(),
                 context_class_counts=np.bincount(labels[context], minlength=classes).tolist(),
                 raw_task_fingerprint=_arrays_hash(raw, labels, src, dst, np.int64(n_train)))
    return base, private, audit


def _prepare_synthetic(split, folder):
    setup()  # Bootstrap the pinned CPU prior; no model/checkpoint load here.
    wall, cpu = time.perf_counter(), time.process_time()
    configs = {}; episodes = []; bases = []
    raw_folder = folder / 'raw'; raw_folder.mkdir()
    for spec in _design(split):
        key = f"N{spec['requested_nodes']}_F{spec['requested_features']}"
        if key not in configs:
            configs[key] = generator_config(spec['requested_nodes'], spec['requested_features'])
        base, private, audit = _sample(spec, configs[key])
        base_id = f"{split}_g{spec['index']:04d}"
        raw_path = raw_folder / (base_id + '.npz')
        with raw_path.open('xb') as stream:
            np.savez_compressed(stream, **private)
        meta = dict(split=split, base=base_id, base_id=base_id,
                    shape=spec['shape'], k=spec['k'], regime=spec['regime'],
                    stratum=f"{spec['shape']}_k{spec['k']}_{spec['regime']}",
                    repetition=spec['repetition'], requested_seed=spec['seed'],
                    requested_nodes=spec['requested_nodes'], requested_features=spec['requested_features'])
        for setting in SETTINGS:
            episodes.append(_save_episode(folder, base, meta, setting))
        if split == 'final':
            for setting in cfg()['primary_settings']:
                for intervention in cfg()['interventions']:
                    episodes.append(_save_episode(folder, base, meta, setting, intervention))
        bases.append(dict(**meta, **audit, raw_path=raw_path.relative_to(ROOT).as_posix(),
                          raw_sha256=sha(raw_path), generator_key=key))
        if (spec['index'] + 1) % 8 == 0:
            print('FP05_PREPARE', split, spec['index'] + 1, len(episodes), flush=True)
    expected = cfg()[split]['graphs'] * (11 if split == 'final' else 7)
    assert len(episodes) == expected
    dump(folder / 'generator_configs.json', configs)
    manifest = dict(cycle='FP05', split=split, episodes=episodes, bases=bases,
                    protocol_sha256=sha(ROOT / 'protocol.json'),
                    plan_sha256=sha(ROOT / 'PLAN_v2.md'), data_source_sha256=sha(__file__),
                    dependencies=dependencies(include_prior=True),
                    generator_configs_path=(folder / 'generator_configs.json').relative_to(ROOT).as_posix(),
                    generator_configs_sha256=sha(folder / 'generator_configs.json'),
                    prepare_wall_seconds=time.perf_counter() - wall,
                    prepare_cpu_seconds=time.process_time() - cpu,
                    pcfi_wall_seconds=sum(r['pcfi']['wall_seconds'] for r in episodes),
                    rng_streams='SeedSequence([requested_seed,stream]):0shape,5context/query,6node,7clusterBFS,8cell,30+setting shuffle,90forward',
                    optimizer_updates=0, source0_or_1_filtered=False,
                    physical_padding_added=False,
                    task_quantiles='unchanged official distribution',
                    retries='only official AssertionError; task-type filter rejections counted separately')
    if split == 'final':
        manifest['design_sha256'] = sha(ROOT / 'results/design.json')
    dump(folder / 'manifest.json', manifest)
    return manifest


def _prepare_real(folder):
    wall, cpu = time.perf_counter(), time.process_time()
    old = _fp04(); old_rows = old.rows('real')
    old_manifest_path = old.ROOT / 'bank/real/manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text(encoding='utf8'))
    pilot = WORK / 'factor_real_graph_pilot'; hom = WORK / 'factor_homophily_benchmark'
    pm = json.loads((pilot / 'bank/manifest.json').read_text(encoding='utf8'))
    hm = json.loads((hom / 'bank/manifest.json').read_text(encoding='utf8'))
    pc = json.loads((pilot / 'protocol.json').read_text(encoding='utf8'))
    inherited = {(r['graph'], r['repetition'], r['mode']):(r, pilot)
                 for r in pm['episodes'] if r['split'] == 'real' and r['graph'] in ('Cora', 'CiteSeer')}
    inherited.update({(r['graph'], r['repetition'], r['mode']):(r, hom)
                      for r in hm['episodes'] if r['graph'] == 'PubMed'})
    raw = {}
    for graph in cfg()['real']['graphs']:
        source = old_manifest['raw_sources'][graph]
        raw[graph] = old._read_raw(source['path'], source['sha256'])
    episodes = []; projections = {}; reconstruction = []
    projection_folder = folder / 'projections'; projection_folder.mkdir()
    for graph_index, graph in enumerate(cfg()['real']['graphs']):
        f = raw[graph]['x'].shape[1]; d = cfg()['real']['projection_features']
        projection_seed = cfg()['real']['projection_seed'] + graph_index
        projection = np.random.default_rng(projection_seed).choice([-1., 1.], size=(f, d)).astype(np.float32) / np.sqrt(d)
        path = projection_folder / f'{graph}_RP{d}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, projection=projection, seed=np.int64(projection_seed))
        projections[graph] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
                                  seed=projection_seed, raw_features=f, projected_features=d,
                                  array_sha256=_arrays_hash(projection), scale='1/sqrt16')
        for rep in cfg()['real']['repetitions']:
            for mode in cfg()['real']['modes']:
                fp04row = next(r for r in old_rows if (r['graph'], r['repetition'], r['mode']) == (graph, rep, mode))
                if sha(old.ROOT / fp04row['path']) != fp04row['sha256']:
                    raise ValueError('FP04 real bank changed')
                inherited_row, inherited_root = inherited[graph, rep, mode]
                physical, physical_mask, identity = old._real_base(inherited_row, inherited_root, raw[graph], pc)
                # Compare actual frozen FP04 targets/context, not only manifest strings.
                with np.load(old.ROOT / fp04row['path']) as z:
                    assert np.array_equal(z['query'], physical['query'])
                    assert np.array_equal(z['y'], physical['labels'][physical['query']])
                    assert np.array_equal(z['context_ids'], physical['context_ids'])
                    assert np.array_equal(z['labels_context'], physical['labels'][physical['context_ids']])
                    assert np.array_equal(z['observed'], physical_mask)
                visible = np.where(physical_mask, physical['raw'], 0).astype(np.float32)
                projected, observed = project_visible(visible, physical_mask, projection)
                base = dict(raw=projected, src=physical['src'], dst=physical['dst'],
                            context_ids=physical['context_ids'],
                            context_labels=physical['labels'][physical['context_ids']],
                            query_ids=physical['query'], y_query=physical['labels'][physical['query']],
                            classes=physical['classes'], seed=physical['seed'],
                            forward_seed=physical['seed'], regime='disjoint')
                meta = dict(split='real', base=f'{graph}_r{rep}', base_id=f'{graph}_r{rep}',
                            graph=graph, repetition=rep, mode=mode, shape=identity['panel'],
                            k='legacy', regime='disjoint', stratum=graph,
                            requested_seed=physical['seed'], requested_nodes=len(projected),
                            requested_features=d)
                # Inherited shuffle has already permuted raw values; never shuffle twice.
                row = _save_episode(folder, base, meta, mode, 'standard', observed)
                row.update(native_features=f, projection=projections[graph],
                           native_visible_sha256=_arrays_hash(visible, physical_mask),
                           fp04_path=str(old.ROOT / fp04row['path']), fp04_sha256=fp04row['sha256'],
                           native_reconstruction=identity)
                episodes.append(row); reconstruction.append(identity)
                print('FP05_REAL', graph, rep, mode, row['source_nodes'], flush=True)
    assert len(episodes) == 45
    for graph, source in raw.items():
        assert sha(source['path']) == source['sha256'], graph
    manifest = dict(cycle='FP05', split='real', episodes=episodes,
                    protocol_sha256=sha(ROOT / 'protocol.json'), plan_sha256=sha(ROOT / 'PLAN_v2.md'),
                    data_source_sha256=sha(__file__), dependencies=dependencies(),
                    fp04_manifest_path=str(old_manifest_path), fp04_manifest_sha256=sha(old_manifest_path),
                    inherited_manifests={str(p):sha(p) for p in [pilot / 'bank/manifest.json',
                                             hom / 'bank/manifest.json', pilot / 'protocol.json']},
                    raw_sources={g:dict(path=r['path'], sha256=r['sha256']) for g, r in raw.items()},
                    projections=projections,
                    historical_references_path=str(old.ROOT / 'bank/real/references.json'),
                    historical_references_sha256=sha(old.ROOT / 'bank/real/references.json'),
                    historical_scope='same physical observations/graph/context/query/truth; different representation/preprocessing/training budget',
                    pipeline='raw visible rows -> fixed RP16 /sqrt16 -> observed RMS -> PCFI100 alpha.9 -> clip10',
                    prepare_wall_seconds=time.perf_counter() - wall,
                    prepare_cpu_seconds=time.process_time() - cpu,
                    pcfi_wall_seconds=sum(r['pcfi']['wall_seconds'] for r in episodes),
                    optimizer_updates=0, target_updates=0,
                    scope='reused real development: Cora/CiteSeer full, PubMed inherited induced2000')
    dump(folder / 'manifest.json', manifest)
    return manifest


def prepare(split):
    if split not in ('train', 'dev', 'final', 'real'):
        raise ValueError(split)
    if split == 'final' and not (ROOT / 'results/design.json').is_file():
        raise RuntimeError('Final generation requires root-created results/design.json')
    folder = ROOT / 'bank' / split
    if folder.exists():
        raise FileExistsError(f'Existing/partial bank is preserved: {folder}')
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir()
    return _prepare_real(folder) if split == 'real' else _prepare_synthetic(split, folder)


def rows(split):
    path = ROOT / 'bank' / split / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf8'))
    if manifest['protocol_sha256'] != sha(ROOT / 'protocol.json'):
        raise ValueError('Bank protocol mismatch')
    return manifest['episodes']


def _checked_path(row):
    path = ROOT / row['path']
    if sha(path) != row['sha256']:
        raise ValueError(f'Episode changed: {path}')
    return path


def read(row):
    with np.load(_checked_path(row), allow_pickle=False) as z:
        inp = {key:(int(z[key]) if key in ('classes', 'forward_seed') else z[key].copy())
               for key in INPUT_KEYS}
        y = z['y'].astype(np.int64)
    assert set(inp) == set(INPUT_KEYS)
    return inp, y


def read_extra(row):
    with np.load(_checked_path(row), allow_pickle=False) as z:
        out = {key:z[key].copy() for key in ('own_missing', 'standard_own_missing',
                'query_observed', 'actual_own_missing', 'standard_source_nodes',
                'physical_features', 'visible_values', 'pcfi_raw')}
        out['standard_source_nodes'] = int(out['standard_source_nodes'])
        out['physical_features'] = int(out['physical_features'])
        out['source_nodes'] = int(z['observed'].any(1).sum())
    return out


extra = read_extra


def simple_reference(inp, arm):
    """Target-free Laplace context prior or clamped restart-LP40 reference."""
    n, k = len(inp['values']), int(inp['classes'])
    ids = np.asarray(inp['context_ids'], np.int64)
    labels = np.asarray(inp['context_labels'], np.int64)
    query = np.asarray(inp['query_ids'], np.int64)
    prior = (np.bincount(labels, minlength=k) + 1).astype(np.float64) / (len(labels) + k)
    if arm == 'prior':
        return np.broadcast_to(prior, (len(query), k)).copy()
    if arm != 'lp':
        raise ValueError(arm)
    src, dst = np.asarray(inp['src'], np.int64), np.asarray(inp['dst'], np.int64)
    adjacency = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(1); adjacency.data[:] = 1; adjacency.sort_indices()
    transition = sp.diags(1 / np.asarray(adjacency.sum(1)).ravel()) @ adjacency
    anchors = np.eye(k)[labels]
    probabilities = np.broadcast_to(prior, (n, k)).copy()
    probabilities[ids] = anchors
    for _ in range(40):
        probabilities = .9 * (transition @ probabilities) + .1 * prior
        probabilities[ids] = anchors
    p = probabilities[query]
    p = .99 * p / p.sum(1, keepdims=True) + .01 * prior
    assert p.shape == (len(query), k) and np.isfinite(p).all() and np.allclose(p.sum(1), 1)
    return p


def historical_references():
    """Frozen FP04 identity-checked refs; not representation-matched predictions."""
    manifest = json.loads((ROOT / 'bank/real/manifest.json').read_text(encoding='utf8'))
    path = Path(manifest['historical_references_path'])
    if sha(path) != manifest['historical_references_sha256']:
        raise ValueError('Historical reference index changed')
    return _fp04().external_records()


def selfcheck():
    """CPU toy arrays only: no official-task/real/bank/checkpoint generation."""
    raw = np.array([[0., 2., np.nan], [1., 3., -9.], [8., 100., 4.]], np.float32)
    mask = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], bool)
    poison = raw.copy(); poison[~mask] = 8e25
    assert np.array_equal(normalize_visible(raw, mask), normalize_visible(poison, mask))
    assert not np.any(normalize_visible(np.zeros((3, 2)), np.zeros((3, 2), bool)))
    src = np.array([0, 1, 1, 2, 2, 3], np.int64); dst = np.array([1, 0, 2, 1, 3, 2], np.int64)
    n, f = 32, 3
    base = dict(raw=np.arange(n*f, dtype=np.float32).reshape(n, f) / 7,
                src=src, dst=dst, context_ids=np.array([0, 1], np.int64),
                context_labels=np.array([0, 1], np.int64), query_ids=np.array([2, 3, 4], np.int64),
                y_query=np.array([0, 1, 0], np.int64), classes=2, seed=18954,
                forward_seed=77, regime='natural')
    for regime in ('natural', 'disjoint'):
        masks = {s:make_mask((n, f), src, dst, base['context_ids'], base['query_ids'], s, 916, regime)
                 for s in SETTINGS}
        assert np.all(masks['node995'] <= masks['node95']) and np.all(masks['node95'] <= masks['node90'])
        assert np.all(masks['cluster995'] <= masks['cluster95'])
        if regime == 'disjoint':
            assert all(not m[base['query_ids']].any() and not m[base['context_ids']].any() for m in masks.values())
    chosen = np.zeros((n, f), bool); chosen[[2, 7, 9]] = True
    standard, _ = _episode_arrays(base, 'node95', standard_mask=chosen)
    none, _ = _episode_arrays(base, 'node95', 'none', chosen)
    shuffle, _ = _episode_arrays(base, 'node95', 'shuffle', chosen)
    assert np.array_equal(standard['own_missing'], [False, True, True])
    for other in (none, shuffle):
        assert np.array_equal(standard['own_missing'], other['own_missing'])
        assert np.array_equal(standard['query_ids'], other['query_ids'])
        assert np.array_equal(standard['context_ids'], other['context_ids'])
        assert np.array_equal(standard['y'], other['y'])
    assert not none['observed'].any() and not none['values'].any()
    assert np.array_equal(shuffle['observed'], chosen)
    assert np.array_equal(np.sort(base['raw'][[2, 7, 9]], axis=0),
                          np.sort(_intervene(base['raw'], chosen, 'node95', 'shuffle', base['seed'])[0][[2, 7, 9]], axis=0))
    projection = np.array([[1., -1.], [1., 1.], [-1., 1.]], np.float32)
    projected, projected_mask = project_visible(base['raw'], chosen, projection)
    poisoned = base['raw'].copy(); poisoned[~chosen] = np.nan
    assert np.array_equal(projected, project_visible(poisoned, chosen, projection)[0])
    assert np.array_equal(projected_mask.any(1), chosen.any(1))
    toy_inp = {k:(int(standard[k]) if k in ('classes', 'forward_seed') else standard[k]) for k in INPUT_KEYS}
    for arm in ('prior', 'lp'):
        p = simple_reference(toy_inp, arm)
        assert p.shape == (3, 2) and np.isfinite(p).all() and np.allclose(p.sum(1), 1.)
    assert len(list(_design('final'))) == 96
    assert set(INPUT_KEYS) == {'values', 'observed', 'src', 'dst', 'context_ids',
                              'context_labels', 'query_ids', 'classes', 'forward_seed'}
    return dict(status='PASS', gpu_used=False, bank_created=False,
                official_tasks_sampled=0, real_inputs_loaded=False, optimizer_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'selfcheck'])
    parser.add_argument('--split', choices=['train', 'dev', 'final', 'real'])
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.split is None:
            parser.error('--split is required for prepare')
        result = prepare(args.split)
        print(json.dumps(dict(split=args.split, rows=len(result['episodes']),
                              manifest_sha256=sha(ROOT / 'bank' / args.split / 'manifest.json'))), flush=True)
    else:
        print(json.dumps(selfcheck()), flush=True)

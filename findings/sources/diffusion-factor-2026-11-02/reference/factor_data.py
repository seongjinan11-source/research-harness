"""Fresh disjoint/paired-overlap tasks. Query truth is not a model input."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('disjoint_native_reference', WORK / 'direct_marginal_pilot/data.py')
NATIVE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = NATIVE
spec.loader.exec_module(NATIVE)
spec = importlib.util.spec_from_file_location('disjoint_public_common', WORK / 'joint_imputation_pfn/common.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
INPUT_KEYS = NATIVE.INPUT_KEYS + ['pcfi']


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcfi(inp):
    union = inp['adj'].sum(0) > 0
    src, dst = np.nonzero(union)
    raw = np.where(inp['observed'], inp['x'], np.nan)
    result = C.BASE.fill(raw, src, dst, 'pcfi', 'uniform_nodes')
    result = np.nan_to_num(result).astype(np.float32)
    assert np.isfinite(result).all()
    # Official imputation is permitted to transform values internally; original
    # observed values are retained exactly at the public predictor interface.
    result[inp['observed']] = inp['x'][inp['observed']]
    return result


def observation(master, k, count, coverage):
    n, f = master['full'].shape
    ids = master['context_order'][:k]
    outside = np.setdiff1d(np.arange(n), ids)
    rng = np.random.default_rng(master['seed'] + 310019 * k + count)
    order = rng.permutation(outside)
    observed_ids = order[:count].copy()
    if coverage == 'paired_overlap':
        observed_ids[-1] = ids[int(rng.integers(k))]
    else:
        assert coverage == 'disjoint'
    obs = np.zeros((n, f), bool)
    obs[observed_ids] = True
    context = np.full(n, -1, np.int64)
    context[ids] = master['latent'][ids]
    query = np.setdiff1d(np.arange(cfg()['target_nodes']), ids)
    inp = {key: master[key] for key in ['incidence', 'adj', 'pair_valid', 'type', 'parent', 'order']}
    inp.update(x=np.where(obs, master['full'], 0).astype(np.float32), observed=obs, context=context,
               prior=np.eye(3, dtype=np.float32)[NATIVE.NAMES.index(master['prior'])])
    inp['pcfi'] = pcfi(inp)
    assert int(obs.any(-1).sum()) == count
    assert bool(obs[ids].any()) == (coverage == 'paired_overlap')
    return inp, query, master['latent'][query]


def rows(split):
    return [r for r in json.loads((ROOT / 'bank/manifest.json').read_text(encoding='utf-8'))['episodes'] if r['split'] == split]


def read(path):
    with np.load(ROOT / 'bank' / path) as z:
        return {k: z[k] for k in INPUT_KEYS}, z['query'], z['labels_query']


def tensors(items, device='cuda', dtype=torch.float32):
    return NATIVE.tensors(items, device, dtype)


def intervention(inp, mode, seed):
    given = {k: v.copy() for k, v in inp.items()}
    if mode == 'no_observed':
        given['observed'][:] = False
        given['x'][:] = 0
    elif mode == 'shuffled':
        for f in range(given['x'].shape[1]):
            ids = np.flatnonzero(given['observed'][:, f])
            perm = np.random.default_rng(seed + f).permutation(ids)
            given['x'][ids, f] = given['x'][perm, f]
    elif mode == 'graph_permuted':
        rng = np.random.default_rng(seed)
        order = np.arange(len(given['x']))
        for typ in np.unique(given['type']):
            ids = np.flatnonzero(given['type'] == typ)
            order[ids] = rng.permutation(ids)
        given['adj'] = given['adj'][:, order][:, :, order]
        given['incidence'] = given['adj'][:, :cfg()['target_nodes'], cfg()['target_nodes']:]
        given['parent'], given['order'] = NATIVE.tree_for(given['adj'], seed)
    elif mode != 'standard':
        raise ValueError(mode)
    if mode != 'standard':
        given['pcfi'] = pcfi(given)
    return given


def public_input(inp, query, seed):
    n = len(inp['x'])
    a = inp['adj']
    deg = a.sum(-1)
    norm = a / np.maximum(deg[:, :, None], 1)
    known = inp['context'] >= 0
    seed_labels = np.eye(2)[inp['context'].clip(0)] * known[:, None]
    hop = np.einsum('rij,jc->ric', norm, seed_labels)
    hop2 = np.einsum('rij,rjc->ric', norm, hop)
    extra = np.concatenate((deg.T / n, np.tile(inp['prior'], (n, 1)), inp['type'][:, None], hop.transpose(1, 0, 2).reshape(n, -1), hop2.transpose(1, 0, 2).reshape(n, -1)), 1).astype(np.float32)
    values = np.concatenate((inp['x'], extra), 1)
    filled = np.concatenate((inp['pcfi'], extra), 1)
    mask = np.concatenate((inp['observed'], np.ones_like(extra, bool)), 1)
    if values.shape[1] % 2:
        values = np.pad(values, ((0, 0), (0, 1)))
        filled = np.pad(filled, ((0, 0), (0, 1)))
        mask = np.pad(mask, ((0, 0), (0, 1)), constant_values=True)
    src, dst = np.nonzero((a.sum(0) > 0) | np.eye(n, dtype=bool))
    ids = np.flatnonzero(known)
    return C.Inputs(values, mask, filled, src, dst, ids, inp['context'][ids], query, 2, int(seed))


def feature_only(inp):
    x, obs, c = inp['x'], inp['observed'], inp['context']
    f = x.shape[-1]
    orientation = 2 * ((np.arange(2 ** f)[:, None] >> np.arange(f)) & 1) - 1
    spin = np.array([-1, 1])
    logp = -.5 * ((x[None, :, :, None] - .8 * orientation[:, None, :, None] * spin) / .6) ** 2 - np.log(.6 * np.sqrt(2 * np.pi))
    leaf = (logp * obs[None, :, :, None]).sum(2) - np.log(2)
    for i in np.flatnonzero(c >= 0):
        leaf[:, i, 1 - c[i]] = -np.inf
    from scipy.special import logsumexp
    local = logsumexp(leaf, -1)
    weight = np.exp(local.sum(-1) - logsumexp(local.sum(-1)))
    pred = np.exp(leaf - local[..., None])
    return np.einsum('h,hnc->nc', weight, pred)


def prepare():
    c = cfg()
    assert not (ROOT / 'bank/manifest.json').exists()
    C.setup()
    records = []
    for folder in ['train', 'test']:
        (ROOT / 'bank' / folder).mkdir(parents=True, exist_ok=True)
    def save(m, split, gid, k, count, cov):
        inp, q, y = observation(m, k, count, cov)
        name = f'{split}/g{gid}_k{k}_o{count}_{cov}.npz'
        np.savez_compressed(ROOT / 'bank' / name, **inp, query=q, labels_query=y)
        records.append({'path': name, 'split': split, 'graph': gid, 'prior': m['prior'], 'case': m['case'], 'context_total': k,
                        'observed_nodes': count, 'coverage': cov, 'sha256': sha(ROOT / 'bank' / name)})
    for gid in range(c['train_episodes']):
        m = NATIVE.master(c['train_seed_start'] + gid, NATIVE.NAMES[gid % 3])
        save(m, 'train', gid, c['contexts'][(gid // 3) % 3], c['observed_nodes'][(gid // 9) % 2], ['disjoint', 'paired_overlap'][(gid // 18) % 2])
        if (gid + 1) % 512 == 0:
            print('PREPARE_TRAIN', gid + 1, flush=True)
    for pid, prior in enumerate(NATIVE.NAMES):
        for local in range(c['test_graphs_per_prior']):
            gid = pid * c['test_graphs_per_prior'] + local
            case = ['matched', 'matched', 'density_low', 'signal_high'][local % 4]
            m = NATIVE.master(c['test_seed_start'] + gid, prior, case)
            for k in c['contexts']:
                for count in c['observed_nodes']:
                    for cov in ['disjoint', 'paired_overlap']:
                        save(m, 'test', gid, k, count, cov)
    dump(ROOT / 'bank/manifest.json', {'episodes': records, 'protocol_sha256': sha(ROOT / 'protocol.json'), 'native_data_sha256': sha(WORK / 'direct_marginal_pilot/data.py')})
    engine = NATIVE.GridReference('cuda')
    test = rows('test')
    (ROOT / 'references').mkdir(exist_ok=True)
    for start in range(0, len(test), 16):
        chunk = test[start:start + 16]
        with torch.inference_mode():
            out = engine.infer([read(r['path'])[0] for r in chunk])
        for i, r in enumerate(chunk):
            np.savez_compressed(ROOT / 'references' / (Path(r['path']).stem + '.npz'), **{key: value[i] for key, value in out.items()})
        if start % 192 == 0:
            print('PREPARE_REFERENCE', start, len(test), flush=True)
    dump(ROOT / 'references/completed.json', {'episodes': len(test), 'scope': 'Same historical six-support prior and exact observed Gaussian orientation marginalization; no actual test parameters supplied.'})
    print('PREPARED', len(records), flush=True)

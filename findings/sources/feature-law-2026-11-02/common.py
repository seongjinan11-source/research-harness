"""Shared native generator/public infrastructure; hidden variables remain outside inputs."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OLD = WORK / 'disjoint_factor_pilot'
sys.path.insert(0, str(OLD))
import data as D
import model as OLD_MODEL
import experiment as OLD_E
C, NATIVE = D.C, D.NATIVE
LAWS = ['gaussian', 'heavy', 'variance']
REGIMES = LAWS + ['mixed']
LEVELS = {'gaussian': [[.8, 1.2], [.6, .9]], 'heavy': [[.8, 1.2], [.6, .9]], 'variance': [[.35, .5], [1., 1.5]]}


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def dump(path, value):
    D.dump(path, value)


def sha(path):
    return D.sha(path)


def make_master(seed, prior, parameter_mode='matched', regime=None):
    master = NATIVE.master(seed, prior)
    rng = np.random.default_rng(seed + 0x514541)
    if regime is None:
        law = rng.integers(0, 3, 2)
        regime = LAWS[int(law[0])] if law[0] == law[1] else 'mixed'
    elif regime == 'mixed':
        law = rng.choice(3, 2, replace=False)
    else:
        law = np.full(2, LAWS.index(regime))
    orientation = rng.choice([-1, 1], 2)
    values = np.empty((12, 2), np.float32)
    parameters = []
    spin = 2 * master['latent'] - 1
    for f in range(2):
        name = LAWS[int(law[f])]
        level = int(rng.integers(2))
        second = level if parameter_mode == 'matched' else 1 - level
        a, b = LEVELS[name][0][level], LEVELS[name][1][second]
        u = spin * orientation[f]
        if name == 'gaussian':
            values[:, f] = a * u + b * rng.standard_normal(12)
        elif name == 'heavy':
            values[:, f] = a * u + b / np.sqrt(3) * rng.standard_t(3, 12)
        else:
            values[:, f] = np.where(u > 0, b, a) * rng.standard_normal(12)
        parameters.append([a, b])
    master.update(full=values, feature_law=law, feature_parameters=np.asarray(parameters), feature_orientation=orientation,
                  feature_regime=regime, parameter_mode=parameter_mode)
    return master


def observation(master, k, count):
    context_ids = master['context_order'][:k]
    reserve = int(master['context_order'][k])
    allowed = np.setdiff1d(np.arange(12), np.r_[context_ids, reserve])
    rng = np.random.default_rng(master['seed'] + k * 310019 + count)
    observed_ids = rng.permutation(allowed)[:count]
    observed = np.zeros((12, 2), bool)
    observed[observed_ids] = True
    context = np.full(12, -1, np.int64)
    context[context_ids] = master['latent'][context_ids]
    query = np.setdiff1d(np.arange(8), context_ids)
    inp = {key: master[key] for key in ['incidence', 'adj', 'pair_valid', 'type', 'parent', 'order']}
    inp.update(x=np.where(observed, master['full'], 0).astype(np.float32), observed=observed, context=context,
               prior=np.eye(3, dtype=np.float32)[NATIVE.NAMES.index(master['prior'])])
    inp['pcfi'] = D.pcfi(inp)
    assert not observed[context_ids].any() and not observed[reserve].any()
    assert (~observed[query].any(-1)).any() and int(observed.any(-1).sum()) == count
    return inp, query, master['latent'][query]


def rows(split):
    path = ROOT / 'bank/manifest.json'
    if not path.exists():
        path = ROOT / 'bank/development_manifest.json'
    return [r for r in json.loads(path.read_text())['episodes'] if r['split'] == split]


def read(row):
    with np.load(ROOT / 'bank' / row['path']) as z:
        return {key: z[key] for key in D.INPUT_KEYS}, z['query'], z['labels_query']


def sources():
    files = {p.name: sha(p) for p in ROOT.glob('*.py')}
    files['protocol.json'] = sha(ROOT / 'protocol.json')
    files['old_lock'] = sha(OLD / 'results/lock.json')
    files['graph_checkpoint'] = sha(WORK / cfg()['graph_checkpoint'])
    return files


def lock():
    OLD_E.lock()
    design = json.loads((ROOT / 'results/design_lock.json').read_text())
    assert design['sources'] == sources(), 'Implementation changed after development lock'
    value = {'sources': sources(), 'bank_manifest': sha(ROOT / 'bank/manifest.json'),
             'references': sha(ROOT / 'references/manifest.json'), 'preflight': sha(ROOT / 'results/preflight.json'),
             'design_lock': sha(ROOT / 'results/design_lock.json')}
    path = ROOT / 'results/lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == value, 'Source/bank/checkpoint changed after lock'
    else:
        dump(path, value)
    return value

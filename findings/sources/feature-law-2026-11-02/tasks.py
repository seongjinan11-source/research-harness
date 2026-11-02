"""Prepare train/development first; final bank only after the design/source lock."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import ROOT, C, D, NATIVE, REGIMES, cfg, dump, sha, make_master, observation, read, rows, sources, lock
from operators import FrozenGraph
from reference import Reference


def save(master, split, gid, k, count):
    inp, q, y = observation(master, k, count)
    name = f'{split}/g{gid}_k{k}_o{count}.npz'
    path = ROOT / 'bank' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **inp, query=q, labels_query=y)
    return {'path': name, 'split': split, 'graph': gid, 'seed': master['seed'], 'prior': master['prior'],
            'regime': master['feature_regime'], 'parameter_mode': master['parameter_mode'], 'context_total': k,
            'observed_nodes': count, 'query_without_features': int((~inp['observed'][q].any(-1)).sum()), 'sha256': sha(path)}


def truth(master, split, gid):
    path = ROOT / 'bank/truth' / split / f'g{gid}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: master[key] for key in ['latent', 'full', 'feature_law', 'feature_parameters', 'feature_orientation', 'parameters', 'role']})


def prepare():
    assert not (ROOT / 'bank/development_manifest.json').exists()
    C.setup()
    c, records = cfg(), []
    begun = time.perf_counter()
    for gid in range(c['train_episodes']):
        m = make_master(c['train_seed_start'] + gid, NATIVE.NAMES[gid % 3])
        truth(m, 'train', gid)
        records.append(save(m, 'train', gid, c['contexts'][(gid // 3) % 3], c['observed_nodes'][(gid // 9) % 2]))
        if (gid + 1) % 512 == 0:
            print('TRAIN_BANK', gid + 1, flush=True)
    for pid, prior in enumerate(NATIVE.NAMES):
        for rid, regime in enumerate(REGIMES):
            gid = pid * 4 + rid
            m = make_master(c['dev_seed_start'] + gid, prior, 'matched', regime)
            truth(m, 'dev', gid)
            for k in c['contexts']:
                for count in c['observed_nodes']:
                    records.append(save(m, 'dev', gid, k, count))
    dump(ROOT / 'bank/development_manifest.json', {'episodes': records, 'seconds': time.perf_counter() - begun,
         'scope': 'Fresh train+separate development graphs; no final graph generated yet'})
    print('PREPARED_TRAIN_DEV', len(records), flush=True)


def finalize():
    assert not (ROOT / 'bank/manifest.json').exists()
    design = json.loads((ROOT / 'results/design_lock.json').read_text())
    assert design['sources'] == sources()
    C.setup()
    torch.set_num_threads(2)
    c = cfg()
    development = json.loads((ROOT / 'bank/development_manifest.json').read_text())
    assert sha(ROOT / 'bank/development_manifest.json') == design['development_manifest']
    records = list(development['episodes'])
    begun = time.perf_counter()
    gid = 0
    for prior in NATIVE.NAMES:
        for regime in REGIMES:
            for mode in ['matched', 'heldout']:
                for _ in range(c['test_graphs_per_cell']):
                    m = make_master(c['test_seed_start'] + gid, prior, mode, regime)
                    truth(m, 'test', gid)
                    for k in c['contexts']:
                        for count in c['observed_nodes']:
                            records.append(save(m, 'test', gid, k, count))
                    gid += 1
        print('FINAL_BANK', prior, gid, flush=True)
    dump(ROOT / 'bank/manifest.json', {'episodes': records, 'final_graphs': gid, 'final_preparation_seconds': time.perf_counter() - begun,
         'design_lock': sha(ROOT / 'results/design_lock.json'), 'scope': 'Final independent graph seeds generated after design lock'})
    model = FrozenGraph().cuda().double().eval()
    reference = Reference()
    references = []
    test = rows('test')
    begun = time.perf_counter()
    for start in range(0, len(test), 4):
        chunk = test[start:start + 4]
        b = D.tensors([read(r)[0] for r in chunk], dtype=torch.float64)
        with torch.inference_mode():
            graph = model(b)
            values = {'fixed_exact': reference.point_exact(b, graph).cpu().numpy(),
                      'fixed_bp': reference.fixed_bp(b, graph).cpu().numpy(),
                      'native_exact': reference.native_exact(b).cpu().numpy()}
        for i, row in enumerate(chunk):
            path = ROOT / 'references' / (Path(row['path']).stem + '.npz')
            path.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(path, **{key: value[i] for key, value in values.items()})
            references.append({'path': path.relative_to(ROOT).as_posix(), 'input_sha256': row['sha256'], 'sha256': sha(path)})
        if (start + len(chunk)) % 96 == 0:
            print('REFERENCES', start + len(chunk), flush=True)
    dump(ROOT / 'references/manifest.json', {'episodes': references, 'seconds': time.perf_counter() - begun,
         'scope': 'Unknown actual feature law/tuple/orientation. Broad fixed12-candidate feature law prior in all cells. Same frozen graph exact/BP plus normalized native six-graph-parameter reference.'})
    lock()
    print('FINAL_BANK_AND_REFERENCES_LOCKED', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'finalize'])
    args = ap.parse_args()
    prepare() if args.action == 'prepare' else finalize()

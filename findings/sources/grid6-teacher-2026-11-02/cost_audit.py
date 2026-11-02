"""Post-worker cost and independent integrity checks, not additional fitting."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from pilot import ROOT, OLD, D, E, B, C, NATIVE, NEW_ARMS, cfg, dump, lock, read_test, test_rows, eval_spec, load_model


def benchmark():
    C.setup()
    torch.set_num_threads(2)
    lock()
    row = next(r for r in test_rows() if r['case'] == 'matched' and r['coverage'] == 'disjoint' and r['context_total'] == 2 and r['observed_nodes'] == 6)
    inp, q, y = read_test(row)
    values = []
    for arm in NEW_ARMS:
        for seed in cfg()['seeds']:
            model = load_model(arm, seed, True)
            values.append({'arm': arm + '_distill', 'seed': seed, **B.measure(lambda: E.predict(model, arm, inp, q, 111))})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    engine = NATIVE.GridReference('cuda')
    with torch.inference_mode():
        values.append({'arm': 'grid6_bp', 'seed': 0, **B.measure(lambda: B.bp_only(engine, [inp]))})
        values.append({'arm': 'grid6_exact', 'seed': 0, **B.measure(lambda: engine.infer([inp], with_bp=False))})
    dump(ROOT / 'results/benchmark.json', {'measurements': values, 'device': torch.cuda.get_device_name(),
         'mask_distill_median_ms': float(np.median([r['median_ms'] for r in values if r['arm'] == 'mask_pfn_distill'])),
         'factor_distill_median_ms': float(np.median([r['median_ms'] for r in values if r['arm'] == 'factor_pfn_distill'])),
         'grid6_bp_median_ms': next(r['median_ms'] for r in values if r['arm'] == 'grid6_bp'),
         'scope': 'After our training/evaluation workers finish; visible inputs to probability,30 repeats after5 warmups. Bank I/O and model construction excluded. Grid6BP skips unused exact query moments with preflight parity. External GPU load uncontrolled.'})
    print('BENCHMARKED', flush=True)


def audit():
    C.setup()
    torch.set_num_threads(2)
    state = lock()
    train, test, spec = D.rows('train'), test_rows(), eval_spec()
    for r in train:
        assert D.sha(OLD / 'bank' / r['path']) == r['sha256']
    for r in test:
        assert D.sha(ROOT / 'bank' / r['path']) == r['sha256']
        inp, q, y = read_test(r)
        known = inp['context'] >= 0
        assert not np.intersect1d(q, np.flatnonzero(known)).size
        assert int(inp['observed'].any(-1).sum()) == r['observed_nodes']
        assert bool(inp['observed'][known].any()) == (r['coverage'] == 'paired_overlap')
        assert np.all(inp['x'][~inp['observed']] == 0)
    teacher = json.loads((ROOT / 'teachers/manifest.json').read_text())
    for r in teacher['episodes']:
        assert D.sha(ROOT / r['path']) == r['sha256']
        assert r['input_sha256'] == train[r['index']]['sha256']
        with np.load(ROOT / r['path']) as z:
            p = z['p']
            assert np.isfinite(p).all() and np.max(abs(p.sum(-1) - 1)) < 1e-10
    for r in json.loads((ROOT / 'references/manifest.json').read_text())['episodes']:
        assert D.sha(ROOT / r['path']) == r['sha256']
    replay = []
    combinations = [('mask_pfn', False), ('pcfi_pfn', False), ('factor_pfn', False), ('mask_pfn', True), ('factor_pfn', True)]
    for arm, distilled in combinations:
        for seed in cfg()['seeds']:
            name = arm + ('_distill' if distilled else '_ce')
            model = load_model(arm, seed, distilled)
            folder = ROOT / 'evaluation' / f'{name}_s{seed}'
            with np.load(folder / 'predictions.npz') as z:
                p, ids, modes = z['p'], z['episode'], z['mode']
            assert list(zip(ids.tolist(), modes.tolist())) == spec
            for j in [0, len(spec) // 2, len(spec) - 1]:
                i, mode = spec[j]
                inp, q, y = read_test(test[i])
                given = inp if mode == 'standard' else D.intervention(inp, mode, 88000000 + i)
                actual = E.predict(model, arm, given, q, 88000000 + i)
                err = float(abs(actual - p[j, q]).max())
                assert err < 2e-5, (name, seed, j, err)
                replay.append({'arm': name, 'seed': seed, 'index': j, 'max_error': err})
            src = ROOT / f'results/{arm}_distill_s{seed}' if distilled else OLD / f'results/{arm}_s{seed}'
            saved = torch.load(src / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
            if distilled:
                assert saved['metadata'] == state
                logs = json.loads((src / 'train_log.json').read_text())
                assert [r['step'] for r in logs] == list(range(1, cfg()['updates'] + 1))
                assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in logs)
            del model, saved
            gc.collect()
            torch.cuda.empty_cache()
    engine = NATIVE.GridReference('cuda')
    parity = []
    for i in [0, 255, 767, 1023, 1535, 2047]:
        inp = D.read(train[i]['path'])[0]
        with torch.inference_mode():
            p = engine.infer([inp], with_bp=False)['grid6_exact'][0]
        with np.load(ROOT / teacher['episodes'][i]['path']) as z:
            err = float(abs(p - z['p']).max())
        assert err < 1e-10
        parity.append({'training_episode': i, 'error': err})
    decision = json.loads((ROOT / 'results/decision.json').read_text())
    assert decision['competitive_compression_gate'] is not None
    dump(ROOT / 'results/final_integrity.json', {'source_lock_unchanged': True, 'train_bank_hashes_checked': len(train),
         'fresh_test_bank_hashes_checked': len(test), 'teacher_hashes_checked': len(teacher['episodes']),
         'reference_hashes_checked': len(test), 'teacher_normalized': True, 'original_masks_and_query_disjointness_checked': True,
         'prediction_reexecution': replay, 'teacher_reexecution': parity, 'all_checkpoints_finite': True,
         'actual_new_models': 6, 'successful_new_updates': 6 * cfg()['updates'], 'decision': decision})
    dump(ROOT / 'results/completed.json', decision)
    print('FINAL_INTEGRITY_OK', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['benchmark', 'audit'])
    args = ap.parse_args()
    benchmark() if args.action == 'benchmark' else audit()

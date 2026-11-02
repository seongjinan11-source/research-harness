"""Final source/bank/checkpoint/probability integrity checks; no scientific tuning."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import json
import numpy as np
import torch
from data import ROOT, cfg, rows, read, dump, sha, C
from experiment import lock, make_model, predict


def main():
    state = lock()
    c = cfg()
    manifest = json.loads((ROOT / 'bank/manifest.json').read_text())
    for r in manifest['episodes']:
        assert sha(ROOT / 'bank' / r['path']) == r['sha256']
        inp, q, y = read(r['path'])
        known = inp['context'] >= 0
        assert not np.intersect1d(q, np.flatnonzero(known)).size
        assert int(inp['observed'].any(-1).sum()) == r['observed_nodes']
        assert bool(inp['observed'][known].any()) == (r['coverage'] == 'paired_overlap')
        assert np.all(inp['x'][~inp['observed']] == 0)
    C.setup()
    torch.set_num_threads(2)
    predictions_checked = []
    for arm in c['arms']:
        for seed in c['seeds']:
            folder = ROOT / 'results' / f'{arm}_s{seed}'
            saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert saved['metadata'] == state
            assert all(torch.isfinite(p).all() for p in saved['state_dict'].values())
            model = make_model(arm, seed)
            meta = json.loads((folder / 'training.json').read_text())
            assert C.parameter_digest(model, trainable=False) == meta['frozen_sha256']
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.copy_(saved['state_dict'][name].to(p.device))
            assert C.parameter_digest(model, trainable=True) == meta['final_sha256']
            for index in [0, len(rows('test')) // 2, len(rows('test')) - 1]:
                row = rows('test')[index]
                inp, q, y = read(row['path'])
                actual = predict(model, arm, inp, q, 88000000 + index)
                with np.load(folder / 'predictions' / f'{index}_standard.npz') as z:
                    error = float(np.max(np.abs(actual - z['p'])))
                    assert error < 2e-5, (arm, seed, index, error)
                predictions_checked.append({'arm': arm, 'seed': seed, 'index': index, 'max_error': error})
            del model, saved
            torch.cuda.empty_cache()
    decision = json.loads((ROOT / 'results/decision.json').read_text())
    assert decision['competitive_gate'] is not None
    dump(ROOT / 'results/final_integrity.json', {'source_lock_unchanged': True, 'all_bank_sha256_checked': len(manifest['episodes']),
        'no_context_query_leak': True, 'coverage_and_observation_budget_checked': True, 'all_checkpoints_finite': True,
        'frozen_weights_unchanged': True, 'sampled_probability_reexecution': predictions_checked, 'decision': decision})
    dump(ROOT / 'results/completed.json', decision)
    print('FINAL_INTEGRITY_OK', flush=True)


if __name__ == '__main__':
    main()

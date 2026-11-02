"""No gate changes: CIs for prespecified no-feature queries and context diagnostic."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from collections import defaultdict
from data import ROOT, cfg, dump, read
from analyze import primary, ci


def main():
    records = json.loads((ROOT / 'results/baselines.json').read_text())
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            records += json.loads((ROOT / f'results/{arm}_s{seed}/evaluation.json').read_text())
    coverage = {}
    for r in records:
        if primary(r) and r['path'] not in coverage:
            inp, _, _ = read(r['path'])
            known = inp['context'] >= 0
            coverage[r['path']] = len(np.unique(inp['context'][known])) == 2
    def vector(arm, metric, subset):
        group = defaultdict(list)
        for r in records:
            if r['arm'] != arm or r['mode'] != 'standard' or not primary(r):
                continue
            inp, q, y = read(r['path'])
            if subset == 'both_context_classes' and not coverage[r['path']]:
                continue
            with np.load(ROOT / r['prediction']) as z:
                p = z['p'] if arm in cfg()['arms'] else z[arm]
            if subset == 'query_without_features':
                take = ~inp['observed'][q].any(-1)
                if not take.any():
                    continue
                p, y = p[take], y[take]
            from data import C
            group[r['graph']].append(C.metrics(p, y)[metric])
        return {g: float(np.mean(v)) for g, v in group.items()}
    out = {'scope': 'Query-without-features is prespecified secondary; both-context-classes is a post-lock diagnostic of context support. Neither changes the primary/gates. Same bank, not independent confirmation or multiplicity adjustment.', 'results': []}
    for subset in ['query_without_features', 'both_context_classes']:
        for arm in ['mask_pfn', 'pcfi_pfn', 'constant_factor', 'grid6_bp']:
            for metric in ['accuracy', 'nll', 'brier']:
                a, b = vector('factor_pfn', metric, subset), vector(arm, metric, subset)
                assert set(a) == set(b)
                out['results'].append({'subset': subset, 'baseline': arm, 'metric': metric,
                    'factor_mean': float(np.mean(list(a.values()))), 'baseline_mean': float(np.mean(list(b.values()))), **ci([a[g] - b[g] for g in a])})
    dump(ROOT / 'results/secondary_partitions.json', out)
    print('SECONDARY_PARTITIONS_DONE', flush=True)


if __name__ == '__main__':
    main()

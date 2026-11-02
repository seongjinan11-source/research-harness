"""Post-lock reanalysis of saved predictions; no training or gate changes."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'direct_marginal_pilot'
METRICS = ['accuracy', 'nll', 'brier']


def summarize(pairs):
    groups = defaultdict(list)
    for graph, full, none in pairs:
        groups[graph].append((full, none))
    result = {'graphs': len(groups), 'paired_episode_seed_records': len(pairs)}
    for metric in METRICS:
        full = np.array([np.mean([x[0][metric] for x in rr]) for _, rr in sorted(groups.items())])
        none = np.array([np.mean([x[1][metric] for x in rr]) for _, rr in sorted(groups.items())])
        delta = full - none
        rng = np.random.default_rng(20261102)
        boot = delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1)
        result[metric] = {'observed_graph_mean': float(full.mean()), 'no_features_graph_mean': float(none.mean()),
                          'observed_minus_none': float(delta.mean()), 'exploratory_graph_bootstrap95': np.quantile(boot, [.025, .975]).tolist()}
    return result


def metrics(prob, labels):
    return {'accuracy': float((prob.argmax(-1) == labels).mean()),
            'nll': float(-np.log(np.maximum(prob[np.arange(len(labels)), labels], 1e-12)).mean()),
            'brier': float(((prob - np.eye(2)[labels]) ** 2).sum(-1).mean())}


def main():
    manifest = json.loads((ROOT / 'bank/manifest.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT / 'results/baselines.json').read_text(encoding='utf-8'))
    records = list(baseline)
    for arm in ['circuit', 'mask_pfn']:
        for seed in [11, 12, 13]:
            records += json.loads((ROOT / f'results/{arm}_s{seed}/evaluation.json').read_text(encoding='utf-8'))
    lookup = {(r['arm'], r['seed'], r['graph'], r['context_total'], r['setting']): r for r in records if r['mode'] == 'standard'}
    selected = []
    query_free_pairs = defaultdict(list)
    for r in manifest['episodes']:
        if r['split'] != 'test' or r['case'] != 'matched' or r['mechanism'] not in ('uniform_nodes', 'clustered_nodes') or r['rate'] not in (.5, .9):
            continue
        with np.load(ROOT / 'bank' / r['path']) as z:
            known, obs = z['context'] >= 0, z['observed']
            if obs[known].any() or not obs[~known].any():
                continue
            selected.append(r)
            none_record = lookup['grid6_exact', 0, r['graph'], r['context_total'], 1]
            with np.load(ROOT / 'bank' / none_record['path']) as zn:
                for field in ['context', 'adj', 'pair_valid', 'prior', 'query', 'labels_query']:
                    assert np.array_equal(z[field], zn[field]), (r['path'], field)
                assert not zn['observed'].any()
            q, y = z['query'], z['labels_query']
            free = ~obs[q].any(-1)
            if free.any():
                with np.load(ROOT / 'references' / (Path(r['path']).stem + '.npz')) as a, np.load(ROOT / 'references' / (Path(none_record['path']).stem + '.npz')) as b:
                    for arm in ['grid6_exact', 'grid6_bp']:
                        query_free_pairs[arm].append((r['graph'], metrics(a[arm][q[free]], y[free]), metrics(b[arm][q[free]], y[free])))
    results = {}
    for arm in ['grid6_exact', 'grid6_bp', 'lp', 'circuit', 'mask_pfn']:
        pairs = []
        for r in selected:
            for seed in ([11, 12, 13] if arm in ['circuit', 'mask_pfn'] else [0]):
                a = lookup[arm, seed, r['graph'], r['context_total'], r['setting']]
                b = lookup[arm, seed, r['graph'], r['context_total'], 1]
                pairs.append((r['graph'], a, b))
        results[arm] = summarize(pairs)
    report = {
        'scope': 'Exploratory post-lock analysis of saved predictions on the existing matched bank. No new learning, primary gate, independent confirmation, or multiplicity-adjusted claim.',
        'selection': 'Original primary masks, observed feature nodes disjoint from context, at least one observed cell outside context. Selection uses visible inputs only. Query labels used for scoring only.',
        'pairing': 'Same graph/context/query/truth; remove all physical features using existing setting1. Assert graph, context, query and truth equality. Average selected episodes and fixed training seeds within graph; 5000 graph bootstrap resamples.',
        'limits': 'Grid6 is a known-family inference reference, not a learned PFN or true-parameter oracle. Query-feature-free partition is an additional exploratory analysis. Model-family assumptions are not established for real graphs.',
        'selected_episodes': len(selected), 'results': results,
        'query_without_features': {arm: summarize(rr) for arm, rr in query_free_pairs.items()},
    }
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

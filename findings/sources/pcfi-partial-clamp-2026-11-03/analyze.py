import sys
sys.dont_write_bytecode = True
import sklearn
import json
from collections import defaultdict
import numpy as np
from core import ROOT,B,cfg,dump,sha

METRICS = ['accuracy','nll','brier']
PAIRS = [('partial_trained','node'),('partial_trained','pcfi'),('partial_frozen','node'),
    ('partial_trained','partial_frozen'),('partial_trained','partial_initial'),
    ('partial_trained','partial_random'),('partial_random','full_random'),
    ('partial_trained','signed_smooth'),('signed_smooth','signed'),('signed_smooth','pcfi')]


def average(rows,field='missing'):
    values = [r[field] for r in rows if r[field] is not None]
    assert values
    return {m:float(np.mean([v[m] for v in values])) for m in METRICS}


def bootstrap(values):
    values = np.asarray(values,dtype='float64')
    rng = np.random.default_rng(2041411300)
    sampled = values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(sampled,[.025,.975]).tolist(),'graphs':len(values)}


def contrasts(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['arm'],r['graph'])].append(r)
    scores = {key:average(items) for key,items in groups.items()}
    result = {}
    for a,b in PAIRS:
        graphs = sorted(g for arm,g in scores if arm==a and (b,g) in scores)
        result[f'{a}_minus_{b}'] = {m:bootstrap([scores[(a,g)][m]-scores[(b,g)][m] for g in graphs]) for m in METRICS}
    return result


def summarize(rows,field='missing'):
    result = {}
    for arm in cfg()['evaluation_arms']:
        groups = defaultdict(list)
        for row in rows:
            if row['arm']==arm and row[field] is not None:
                groups[row['graph']].append(row)
        scores = [average(items,field) for items in groups.values()]
        assert scores
        result[arm] = {m:float(np.mean([s[m] for s in scores])) for m in METRICS}
    return result


def main():
    records,hashes = [],{}
    for seed in cfg()['seeds']:
        for arm in cfg()['evaluation_arms']:
            for split,count in [('development',448),('real',60)]:
                path = ROOT/'results'/f'{arm}_s{seed}'/f'{split}_evaluation.json'
                rows = json.loads(path.read_text())
                assert len(rows)==count
                records.extend(rows)
                hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    primary = [r for r in records if r['case'] in cfg()['primary_cases'] and r['missing'] is not None]
    dev = [r for r in primary if r['split']=='development']
    by_source = {}
    for name,predicate in [('zero',lambda r:r['observed_nodes']==0),('nonzero',lambda r:r['observed_nodes']>0),
                           ('one',lambda r:r['observed_nodes']==1),('two_or_more',lambda r:r['observed_nodes']>=2)]:
        rows = [r for r in dev if predicate(r)]
        by_source[name] = {'episodes_per_arm_seed':len(rows)//(len(cfg()['evaluation_arms'])*len(cfg()['seeds'])),
                           'scores':summarize(rows),'comparisons':contrasts(rows)}
    real = {}
    previous = json.loads((B.ROOT/'results/statistics.json').read_text())
    for graph in ['Cora','CiteSeer','texas','wisconsin']:
        real[graph] = summarize([r for r in primary if r['split']=='real' and r['graph']==graph])
        for arm in ['native_pcfi','negative_lp','mask_reference']:
            real[graph][arm] = previous['real'][graph][arm]
    diagnostics = {}
    for split in ['development','real']:
        diagnostics[split] = {}
        for arm in cfg()['evaluation_arms']:
            values = [r['diagnostics'] for r in primary if r['split']==split and r['arm']==arm]
            keys = set.intersection(*(set(v) for v in values))
            diagnostics[split][arm] = {key:{'mean':float(np.mean([v[key] for v in values])),
                'max':float(np.max([v[key] for v in values])),
                'median':float(np.median([v[key] for v in values])),
                'min':float(np.min([v[key] for v in values]))} for key in sorted(keys)}
    by_cell = {}
    for label in ['homo','hetero']:
        for feature in ['smooth','rough']:
            rows = [r for r in dev if r['label_mode']==label and r['feature_mode']==feature]
            by_cell[f'{label}_{feature}'] = {'scores':summarize(rows),'comparisons':contrasts(rows)}
    result = {
        'development':summarize(dev),'comparisons':contrasts(dev),'by_source':by_source,'real':real,
        'by_cell':by_cell,
        'by_context':{str(k):{'scores':summarize([r for r in dev if r['context_per_class']==k]),
            'comparisons':contrasts([r for r in dev if r['context_per_class']==k])} for k in cfg()['context_per_class']},
        'by_case':{case:summarize([r for r in dev if r['case']==case]) for case in cfg()['primary_cases']},
        'by_seed':{str(seed):{'scores':summarize([r for r in dev if r['seed']==seed]),
            'comparisons':contrasts([r for r in dev if r['seed']==seed])} for seed in cfg()['seeds']},
        'secondary':{case:summarize([r for r in records if r['split']=='development' and r['case']==case],
            'all' if case=='full' else 'missing') for case in ['full','sources2','sources8']},
        'diagnostics':diagnostics,'evaluation_hashes':hashes,'predictions':len(records),
        'inference':'Known-family new development graphs; real graphs reused development inputs. Equal base-graph averaging after averaging available fixed seeds/masks/contexts within each stratum; primary balanced so equivalent to episode mean. Graph-paired5000 bootstrap. Multiple comparisons unadjusted; numerical replay uncertainty not included.'}
    dump(ROOT/'results/records.json',records)
    dump(ROOT/'results/statistics.json',result)
    print(json.dumps({'development':result['development'],
        'accuracy_differences':{k:v['accuracy'] for k,v in result['comparisons'].items()},'real':real},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

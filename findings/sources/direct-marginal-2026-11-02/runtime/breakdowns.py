"""Descriptive context/class-coverage tables; never alters the primary gates."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2]


def main():
    folder=WORK/'direct_marginal_pilot';manifest=json.loads((folder/'bank/manifest.json').read_text(encoding='utf8'))
    coverage={}
    for r in manifest['episodes']:
        if r['split']!='test':continue
        with np.load(folder/'bank'/r['path']) as z:
            c=z['context'];obs=z['observed'];known=c>=0
            inside=bool(obs[known].any());outside=bool(obs[~known].any())
            coverage[r['path']]={'both_context_classes':len(np.unique(c[known]))==2,'nonzero_physical_observations':bool(obs.any()),'observed_physical_cells':int(obs.sum()),
                'feature_source':('both' if inside and outside else 'context_only' if inside else 'outside_context_only' if outside else 'none')}
    allrows=[]
    for arm in ['circuit','direct_graph','mask_pfn']:
        for seed in [11,12,13]:allrows+=json.loads((folder/'results'/f'{arm}_s{seed}'/'evaluation.json').read_text())
    allrows+=json.loads((folder/'results/baselines.json').read_text())
    allrows+=json.loads((folder/'results/feature_controls.json').read_text())['rows']
    summary=defaultdict(list)
    for r in allrows:
        if r['mode']!='standard':continue
        cov=coverage[r['path']]
        keys=[('condition',r['arm'],r['case'],r['context_total'],r['mechanism'],r['rate'])]
        is_primary=r['case']=='matched' and r['mechanism'] in ('uniform_nodes','clustered_nodes') and r['rate'] in (.5,.9)
        if is_primary:
            keys += [('primary_context',r['arm'],r['context_total']),('primary_context_coverage',r['arm'],cov['both_context_classes']),('primary_feature_coverage',r['arm'],cov['nonzero_physical_observations']),('primary_feature_source',r['arm'],cov['feature_source'])]
        for key in keys:summary[key].append(r)
    output=[]
    for key,rr in summary.items():
        output.append({'group':list(key),'episodes':len(rr),'graphs':len({r['graph'] for r in rr}),**{m:float(np.mean([r[m] for r in rr])) for m in ['accuracy','nll','brier']}})
    (folder/'results/descriptive_breakdowns.json').write_text(json.dumps({'scope':'Descriptive secondary conditions and post-lock visible-context/observation-coverage partitions; no new primary gate or independent confirmation. No query truth used to define coverage.','tables':output},ensure_ascii=False,indent=2),encoding='utf8')
    print('BREAKDOWNS_DONE',flush=True)
if __name__=='__main__':main()

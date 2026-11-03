import sys
sys.dont_write_bytecode=True
import sklearn
import json
from collections import defaultdict
import numpy as np
from core import ROOT,B,cfg,dump,sha

METRICS=['accuracy','nll','brier']


def average(rows):
    assert rows
    return {metric:float(np.mean([r['missing'][metric] for r in rows])) for metric in METRICS}


def bootstrap(values):
    values=np.asarray(values,dtype='float64')
    rng=np.random.default_rng(2040611300)
    samples=values[rng.integers(0,len(values),(5000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'graphs':len(values)}


def contrasts(records,selected=None):
    selected=[r for r in records if r['split']=='development' and r['case'] in cfg()['primary_cases'] and r['missing'] is not None and (selected is None or selected(r))]
    group=defaultdict(list)
    for row in selected:
        group[(row['arm'],row['graph'])].append(row)
    values={key:average(rows) for key,rows in group.items()}
    pairs=[('node','pcfi'),('node','node_nolabel'),('node','signed'),('node','node_geometry_shuffle'),
           ('node_uniform','uniform'),('node','node_uniform'),('pcfi','uniform'),
           ('signed','pcfi'),('node_nolabel','pcfi')]
    output={}
    graphs=sorted({r['graph'] for r in selected})
    for a,b in pairs:
        common=[g for g in graphs if (a,g) in values and (b,g) in values]
        output[f'{a}_minus_{b}']={m:bootstrap([values[(a,g)][m]-values[(b,g)][m] for g in common]) for m in METRICS}
    output['spatial_confidence_by_node_interaction']={m:bootstrap([
        values[('node',g)][m]-values[('pcfi',g)][m]-values[('node_uniform',g)][m]+values[('uniform',g)][m]
        for g in graphs]) for m in METRICS}
    return output


def main():
    records=[]
    evaluation_hashes={}
    for seed in cfg()['seeds']:
        for arm in cfg()['evaluation_arms']:
            for split in ['development','real']:
                path=ROOT/'results'/f'{arm}_s{seed}'/f'{split}_evaluation.json'
                rows=json.loads(path.read_text())
                assert len(rows)==(448 if split=='development' else 60)
                records+=rows
                evaluation_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    primary=[r for r in records if r['case'] in cfg()['primary_cases'] and r['missing'] is not None]
    dev=[r for r in primary if r['split']=='development']
    summary={arm:average([r for r in dev if r['arm']==arm]) for arm in cfg()['evaluation_arms']}
    by_k={str(k):{arm:average([r for r in dev if r['context_per_class']==k and r['arm']==arm])
                        for arm in cfg()['evaluation_arms']} for k in cfg()['context_per_class']}
    by_cell={}
    for label in ['homo','hetero']:
        for feature in ['smooth','rough']:
            chosen=[r for r in dev if r['label_mode']==label and r['feature_mode']==feature]
            by_cell[f'{label}_{feature}']={arm:average([r for r in chosen if r['arm']==arm]) for arm in cfg()['evaluation_arms']}
    by_source={}
    for key,predicate in [('zero',lambda r:r['observed_nodes']==0),('nonzero',lambda r:r['observed_nodes']>0)]:
        chosen=[r for r in dev if predicate(r)]
        by_source[key]={'episodes_per_arm_seed':len(chosen)//(len(cfg()['evaluation_arms'])*len(cfg()['seeds'])),
                        'scores':{arm:average([r for r in chosen if r['arm']==arm]) for arm in cfg()['evaluation_arms']}}
    secondary={case:{arm:{m:float(np.mean([r['all'][m] if case=='full' else r['missing'][m]
        for r in records if r['split']=='development' and r['case']==case and r['arm']==arm])) for m in METRICS}
        for arm in cfg()['evaluation_arms']} for case in ['full','sources2','sources8']}
    real={}
    for graph in ['Cora','CiteSeer','texas','wisconsin']:
        real[graph]={arm:average([r for r in primary if r['split']=='real' and r['graph']==graph and r['arm']==arm])
                    for arm in cfg()['evaluation_arms']}
    # These cached floors have identical real visible inputs, but a different native-space pipeline.
    previous=json.loads((B.ROOT/'results/statistics.json').read_text())
    for graph in real:
        for arm in ['native_pcfi','lp','negative_lp','mask_reference']:
            real[graph][arm]=previous['real'][graph][arm]
    geometry_effect={}
    for k in cfg()['context_per_class']:
        geometry_effect[str(k)]=contrasts(records,lambda r:r['context_per_class']==k)['node_minus_node_geometry_shuffle']
    diagnostics={}
    for split in ['development','real']:
        diagnostics[split]={}
        for arm in cfg()['evaluation_arms']:
            chosen=[r['diagnostics'] for r in primary if r['split']==split and r['arm']==arm]
            keys=chosen[0].keys()
            diagnostics[split][arm]={key:float(np.mean([r[key] for r in chosen])) for key in keys}
    result={'development':summary,'comparisons':contrasts(records),'by_context':by_k,'by_cell':by_cell,
            'by_source':by_source,'secondary':secondary,'real':real,
            'geometry_shuffle_by_context':geometry_effect,'diagnostics':diagnostics,
            'evaluation_hashes':evaluation_hashes,'predictions':len(records),
            'inference':'Development only; graphs are known-family new seeds. CI graph-paired/5000 draws, conditional stored predictions and three fixed training seeds; multiple contrasts unadjusted. Real floors native/LP/mask are previous identical-input caches, not newly trained arms.'}
    dump(ROOT/'results/records.json',records)
    dump(ROOT/'results/statistics.json',result)
    print(json.dumps({'development':summary,'accuracy_differences':{k:v['accuracy'] for k,v in result['comparisons'].items()},'real':real},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

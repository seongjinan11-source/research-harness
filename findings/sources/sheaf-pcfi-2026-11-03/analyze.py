import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from core import ROOT,cfg,dump,C

METRICS=['accuracy','nll','brier','macro_f1']
def ci(x):
    x=np.array(x);r=np.random.default_rng(20361103);boot=x[r.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'n_graphs':len(x)}
def records():
    result=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            folder=ROOT/'results'/f'{arm}_s{seed}'
            for split in ['test','real']:
                rows=json.loads((folder/f'{split}_evaluation.json').read_text())
                for row in rows:
                    if row['case']=='full':continue
                    z=np.load(ROOT/row['prediction']);m=z['missing'];p=z['p'][m];y=z['y'][m]
                    assert len(y)>0
                    met=C.metrics(p,y);met['macro_f1']=float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0))
                    assert all(abs(met[k]-row['missing'][k])<1e-12 for k in ['accuracy','nll','brier'])
                    result.append(dict(split=split,arm=arm,seed=seed,graph=row['graph'],case=row['case'],repetition=row.get('repetition',0),
                         label_mode=row['label_mode'],feature_mode=row['feature_mode'],observed_nodes=row['observed_nodes'],**met))
                    if split=='real' and arm=='pcfi100':
                        zn=np.load(ROOT/row['native_prediction']);pn=zn['p'][m]
                        mm=C.metrics(pn,y);mm['macro_f1']=float(f1_score(y,pn.argmax(1),labels=np.arange(pn.shape[1]),average='macro',zero_division=0))
                        assert all(abs(mm[k]-row['native_missing'][k])<1e-12 for k in ['accuracy','nll','brier'])
                        result.append(dict(split=split,arm='native_pcfi100',seed=seed,graph=row['graph'],case=row['case'],repetition=row.get('repetition',0),
                            label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**mm))
    for row in json.loads((ROOT/'results/public_baselines.json').read_text()):
        if row['case']=='full':continue
        z=np.load(ROOT/row['prediction']);m=z['missing'];p=z['p'][m];y=z['y'][m]
        mm=C.metrics(p,y);mm['macro_f1']=float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0))
        result.append(dict(split='real',arm=row['arm'],seed=11,graph=row['graph'],case=row['case'],repetition=row['repetition'],
            label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**mm))
    return result

def main():
    rows=records();syn=[x for x in rows if x['split']=='test'];real=[x for x in rows if x['split']=='real']
    graph={}
    for arm in cfg()['arms']:
        graph[arm]={i:{k:float(np.mean([x[k] for x in syn if x['arm']==arm and x['graph']==i])) for k in METRICS} for i in range(cfg()['test_graphs'])}
    synth={arm:{k:float(np.mean([graph[arm][i][k] for i in graph[arm]])) for k in METRICS} for arm in graph}
    comparisons={}
    for other in ['mask','pcfi100','pcfi200','weighted','signed']:
        comparisons['sheaf - '+other]={k:ci([graph['sheaf'][i][k]-graph[other][i][k] for i in graph['sheaf']]) for k in METRICS}
    modes={}
    for label in ['homo','hetero']:
        modes[label]={arm:{k:float(np.mean([x[k] for x in syn if x['arm']==arm and x['label_mode']==label])) for k in METRICS} for arm in cfg()['arms']}
    homo_ids=sorted({x['graph'] for x in syn if x['label_mode']=='homo'})
    guard=ci([graph['sheaf'][i]['accuracy']-graph['pcfi100'][i]['accuracy'] for i in homo_ids])
    sg=all(comparisons['sheaf - '+a]['accuracy']['ci95'][0]>0 for a in ['pcfi100','pcfi200','weighted','signed']) and guard['ci95'][0]>-.02
    real_table={};real_cases={};real_arms=cfg()['arms']+['native_pcfi100','public_mask','public_pcfi100','public_native_pcfi100','lp','negative_lp','context_prior']
    for ds in cfg()['real_datasets']:
        real_table[ds]={arm:{k:float(np.mean([x[k] for x in real if x['graph']==ds and x['arm']==arm])) for k in METRICS}
                        for arm in real_arms}
        real_cases[ds]={case:{arm:{k:float(np.mean([x[k] for x in real if x['graph']==ds and x['case']==case and x['arm']==arm])) for k in METRICS}
                        for arm in real_arms} for case in cfg()['masks']}
    real_groups={}
    for mode,datasets in [('homo',['Cora','CiteSeer']),('hetero',['texas','wisconsin'])]:
        real_groups[mode]={arm:{k:float(np.mean([real_table[d][arm][k] for d in datasets])) for k in METRICS} for arm in real_arms}
    rd={a:real_groups['hetero']['sheaf']['accuracy']-real_groups['hetero'][a]['accuracy'] for a in ['pcfi100','pcfi200','weighted','signed','native_pcfi100']}
    hd=real_groups['homo']['sheaf']['accuracy']-real_groups['homo']['pcfi100']['accuracy']
    rg=rd['pcfi100']>0 and rd['signed']>0 and hd>=-.02
    native_floor=rd['native_pcfi100']>0
    training=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:training.append(json.loads((ROOT/'results'/f'{arm}_s{seed}/training.json').read_text()))
    counts={ds:[x['observed_nodes'] for x in real if x['graph']==ds and x['arm']=='pcfi100' and x['seed']==11] for ds in cfg()['real_datasets']}
    controls=json.loads((ROOT/'results/controls.json').read_text());control_summary={}
    for ds in cfg()['real_datasets']:
        control_summary[ds]={a:{control:{k:float(np.mean([r['original_missing'][k] for r in controls if r['graph']==ds and r['arm']==a and r['control']==control])) for k in ['accuracy','nll','brier']}
             for control in ['no_features','shuffled_sources']} for a in ['pcfi100','sheaf']}
    out={'synthetic':synth,'synthetic_by_label_mode':modes,'synthetic_comparisons':comparisons,'synthetic_homo_guard':guard,
         'real':real_table,'real_by_case':real_cases,'real_groups':real_groups,'real_hetero_differences':rd,'real_homo_difference':hd,
         'gates':{'synthetic':bool(sg),'real_development':bool(rg),'real_native_floor':bool(native_floor),'expand':bool(sg and rg and native_floor)},
         'observed_node_counts_real':counts,'training':{'valid_checkpoints':len(training),'valid_updates':sum(x['updates'] for x in training),
             'valid_training_seconds':sum(x['seconds'] for x in training),'discarded_design_updates':1152,'real_updates':0},
         'scope':'Missing-query primary. Synthetic graph-paired CI conditions on3training seeds. Real4datasets development pilot, no node/dataset-population bootstrap. No post-test method/loss/step rescue.',
         'real_validation_labels_used':False,'controls_seed11':control_summary}
    dump(ROOT/'results/statistics.json',out);dump(ROOT/'results/rescored_records.json',rows)
    print('SYNTHETIC',synth,'REAL',real_table,'GATES',out['gates'],flush=True)

if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import sklearn
import json
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from core import ROOT,WORK,C,cfg,dump,sha
KEYS=['accuracy','nll','brier','macro_f1']
def metrics(p,y):
    result=C.metrics(p,y);result['macro_f1']=float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0));return result
def ci(values):
    x=np.array(values);r=np.random.default_rng(20371103);b=x[r.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n_base_graphs':len(x)}
def average(rows,field=None):return {k:float(np.mean([(r[field] if field else r)[k] for r in rows])) for k in KEYS}
def main():
    records=[];full=[];diag={};training=[]
    for arm in cfg()['arms']:
        ds=[]
        for seed in cfg()['seeds']:
            folder=ROOT/'results'/f'{arm}_s{seed}';training.append(json.loads((folder/'training.json').read_text()))
            for split in ['test','real']:
                rows=json.loads((folder/f'{split}_evaluation.json').read_text())
                for row in rows:
                    z=np.load(ROOT/row['prediction']);p=z['p'];y=z['y'];m=z['missing']
                    key=dict(split=split,arm=arm,seed=seed,graph=row['graph'],case=row['case'],repetition=row.get('repetition',0),label_mode=row['label_mode'],feature_mode=row['feature_mode'],observed_nodes=row['observed_nodes'])
                    if row['case']=='full':full.append({**key,**metrics(p,y)});continue
                    met=metrics(p[m],y[m]);assert all(abs(met[k]-row['missing'][k])<1e-12 for k in ['accuracy','nll','brier'])
                    records.append({**key,**met})
                    if arm=='pcfi100' and split=='real':
                        zn=np.load(ROOT/row['native_prediction']);met=metrics(zn['p'][m],y[m]);assert all(abs(met[k]-row['native_missing'][k])<1e-12 for k in ['accuracy','nll','brier'])
                        records.append({**key,'arm':'native_pcfi100',**met})
                    if row['diagnostics']['stages']:ds.append({'split':split,'graph':row['graph'],'stages':row['diagnostics']['stages']})
        if ds:diag[arm]={split:{'negative_cos_fraction_by_stage':[float(np.mean([r['stages'][l]['negative_cos_fraction'] for r in ds if r['split']==split])) for l in range(4)],
            'refresh_difference_from_first_by_stage':[float(np.mean([r['stages'][l]['map_refresh_difference'] for r in ds if r['split']==split])) for l in range(4)],
            'mean_abs_update_by_stage':[float(np.mean([r['stages'][l]['mean_abs_update'] for r in ds if r['split']==split])) for l in range(4)],
            'max_anchor_change':max(d['anchor_change'] for r in ds if r['split']==split for d in r['stages'])} for split in ['test','real']}
    # Same real input arrays and fixed baselines, no recomputation/selection by labels.
    public=json.loads((WORK/'sheaf_pcfi_pilot/results/public_baselines.json').read_text())
    for row in public:
        if row['case']=='full':continue
        assert sha(WORK/'sheaf_pcfi_pilot'/row['prediction'])==row['prediction_sha256']
        z=np.load(WORK/'sheaf_pcfi_pilot'/row['prediction']);m=z['missing'];met=metrics(z['p'][m],z['y'][m])
        records.append(dict(split='real',arm=row['arm'],seed=11,graph=row['graph'],case=row['case'],repetition=row['repetition'],label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**met))
    syn=[r for r in records if r['split']=='test' and r['case'] in cfg()['primary_cases']];graph={}
    for arm in cfg()['arms']:graph[arm]={i:average([r for r in syn if r['arm']==arm and r['graph']==i]) for i in range(64)}
    summary={arm:average(list(g.values())) for arm,g in graph.items()}
    comp={other:{k:ci([graph['dynamic'][i][k]-graph[other][i][k] for i in range(64)]) for k in KEYS} for other in ['mask','pcfi100','identity','static']}
    homo=sorted({r['graph'] for r in syn if r['label_mode']=='homo'});guard=ci([graph['dynamic'][i]['accuracy']-graph['pcfi100'][i]['accuracy'] for i in homo])
    sg=all(comp[a]['accuracy']['ci95'][0]>0 for a in ['pcfi100','identity','static']) and guard['ci95'][0]>-.02
    secondary={case:{a:average([r for r in records if r['split']=='test' and r['case']==case and r['arm']==a]) for a in cfg()['arms']} for case in cfg()['test_cases'] if case!='full'}
    realarms=cfg()['arms']+['native_pcfi100','public_mask','public_pcfi100','public_native_pcfi100','lp','negative_lp','context_prior'];realtable={};realcase={}
    for ds in cfg()['real_datasets']:
        realtable[ds]={a:average([r for r in records if r['split']=='real' and r['graph']==ds and r['arm']==a]) for a in realarms}
        realcase[ds]={case:{a:average([r for r in records if r['split']=='real' and r['graph']==ds and r['case']==case and r['arm']==a]) for a in realarms} for case in cfg()['primary_cases']}
    groups={mode:{a:average([realtable[d][a] for d in ds]) for a in realarms} for mode,ds in [('homo',['Cora','CiteSeer']),('hetero',['texas','wisconsin'])]}
    rh={a:groups['hetero']['dynamic']['accuracy']-groups['hetero'][a]['accuracy'] for a in ['mask','pcfi100','identity','static','native_pcfi100','negative_lp']}
    hg=groups['homo']['dynamic']['accuracy']-groups['homo']['pcfi100']['accuracy'];rg=all(rh[a]>0 for a in ['pcfi100','identity','static','native_pcfi100']) and hg>=-.02
    fulls={split:{g:{a:average([r for r in full if r['split']==split and (g=='all' or r['graph']==g) and r['arm']==a]) for a in cfg()['arms']} for g in (['all'] if split=='test' else cfg()['real_datasets'])} for split in ['test','real']}
    no_source={label:{a:average([r for r in syn if r['arm']==a and ((r['observed_nodes']==0) if label=='zero' else (r['observed_nodes']>0))]) for a in cfg()['arms']} for label in ['zero','nonzero']}
    seedtable={str(seed):{a:average([r for r in syn if r['arm']==a and r['seed']==seed]) for a in cfg()['arms']} for seed in cfg()['seeds']}
    out={'synthetic':summary,'comparisons_dynamic_minus':comp,'synthetic_homo_guard':guard,'synthetic_by_case':secondary,'synthetic_by_seed':seedtable,'zero_source_partition':no_source,
        'real':realtable,'real_by_case':realcase,'real_groups':groups,'real_hetero_differences':rh,'real_homo_difference':hg,'full_feature':fulls,'geometry':diag,
        'gates':{'synthetic':bool(sg),'real_input':bool(rg),'negative_lp_floor':bool(rh['negative_lp']>=0),'mask_floor':bool(summary['dynamic']['accuracy']>=summary['mask']['accuracy']),
                 'expand':bool(sg and rg and rh['negative_lp']>=0 and summary['dynamic']['accuracy']>=summary['mask']['accuracy'])},
        'training':{'checkpoints':len(training),'valid_updates':sum(r['updates'] for r in training),'training_seconds_sum':sum(r['seconds'] for r in training),'real_updates':0,'discarded_smoke_steps':5},
        'scope':'New64syntheticgraphs butsameknownfamily;3fixedtrainingseeds. Real4datasets reuseddevelopmentinputs. CI conditional onstoredforecasts;no nodebootstrap orrealvalidationselection.'}
    dump(ROOT/'results/statistics.json',out);dump(ROOT/'results/records.json',records);print('ANALYSIS COMPLETE',out['gates'],flush=True)
if __name__=='__main__':main()

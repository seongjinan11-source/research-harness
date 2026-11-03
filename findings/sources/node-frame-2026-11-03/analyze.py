import sys
sys.dont_write_bytecode=True
import sklearn
import json
import numpy as np
from sklearn.metrics import f1_score
from core import ROOT,WORK,C,cfg,dump,sha

KEYS=['accuracy','nll','brier','macro_f1']
def metrics(p,y):
    m=C.metrics(p,y);m['macro_f1']=float(f1_score(y,p.argmax(1),labels=np.arange(p.shape[1]),average='macro',zero_division=0));return m
def avg(rows):return {k:float(np.mean([r[k] for r in rows])) for k in KEYS} if rows else None
def ci(x):
    x=np.array(x);r=np.random.default_rng(20381103);b=x[r.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n_base_graphs':len(x)}

def main():
    arms=['node_frame']+cfg()['comparison_arms'];records=[];full=[];geometry={}
    for arm in arms:
        ds=[]
        for seed in cfg()['seeds']:
            for split in ['test','real']:
                base=ROOT if split=='test' or arm=='node_frame' else WORK/'nsd_token_pfn'
                path=base/'results'/f'{arm}_s{seed}'/f'{split}_evaluation.json'
                rows=json.loads(path.read_text())
                if base!=ROOT:
                    reuse=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'real_reuse.json').read_text());assert reuse['sha256']==sha(path)
                for row in rows:
                    dest=base/row['prediction'];assert sha(dest)==row['prediction_sha256']
                    z=np.load(dest);p,y,m=z['p'],z['y'],z['missing']
                    k=dict(arm=arm,seed=seed,split=split,graph=row['graph'],case=row['case'],repetition=row.get('repetition',0),
                           label_mode=row['label_mode'],feature_mode=row['feature_mode'],observed_nodes=row['observed_nodes'])
                    if row['case']=='full':full.append({**k,**metrics(p,y)});continue
                    met=metrics(p[m],y[m]);assert all(abs(met[key]-row['missing'][key])<1e-12 for key in ['accuracy','nll','brier'])
                    records.append({**k,**met})
                    if arm=='pcfi100' and split=='real':
                        nz=np.load(base/row['native_prediction']);assert sha(base/row['native_prediction'])==row['native_sha256']
                        records.append({**k,'arm':'native_pcfi100',**metrics(nz['p'][m],y[m])})
                    if row['diagnostics']['stages']:ds.append({'split':split,'stages':row['diagnostics']['stages']})
        if ds:
            geometry[arm]={split:{'negative_cos_fraction_by_stage':[float(np.mean([d['stages'][l]['negative_cos_fraction'] for d in ds if d['split']==split])) for l in range(4)],
                                'refresh_from_first_by_stage':[float(np.mean([d['stages'][l]['map_refresh_difference'] for d in ds if d['split']==split])) for l in range(4)],
                                'mean_abs_update_by_stage':[float(np.mean([d['stages'][l]['mean_abs_update'] for d in ds if d['split']==split])) for l in range(4)],
                                'max_anchor_change':max(s['anchor_change'] for d in ds if d['split']==split for s in d['stages'])} for split in ['test','real']}
    # Same archived graph/context/mask inputs, retain strong fixed real floors.
    for row in json.loads((WORK/'sheaf_pcfi_pilot/results/public_baselines.json').read_text()):
        if row['case']=='full':continue
        p=WORK/'sheaf_pcfi_pilot'/row['prediction'];assert sha(p)==row['prediction_sha256'];z=np.load(p);m=z['missing']
        records.append(dict(arm=row['arm'],seed=11,split='real',graph=row['graph'],case=row['case'],repetition=row['repetition'],
                            label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**metrics(z['p'][m],z['y'][m])))
    primary=[r for r in records if r['split']=='test' and r['case'] in cfg()['primary_cases']]
    ng=cfg()['test_graphs'];graphs={a:{i:avg([r for r in primary if r['arm']==a and r['graph']==i]) for i in range(ng)} for a in arms}
    comp={a:{k:ci([graphs['node_frame'][i][k]-graphs[a][i][k] for i in range(ng)]) for k in KEYS} for a in cfg()['comparison_arms']}
    homo=sorted({r['graph'] for r in primary if r['label_mode']=='homo'})
    guard=ci([graphs['node_frame'][i]['accuracy']-graphs['pcfi100'][i]['accuracy'] for i in homo])
    secondary={case:{a:avg([r for r in records if r['split']=='test' and r['case']==case and r['arm']==a]) for a in arms} for case in cfg()['test_cases'] if case!='full'}
    groups={}
    for label in ['homo','hetero']:
        ids=sorted({r['graph'] for r in primary if r['label_mode']==label})
        groups[label]={a:avg([graphs[a][i] for i in ids]) for a in arms}
    realarms=arms+['native_pcfi100','public_mask','public_pcfi100','public_native_pcfi100','lp','negative_lp','context_prior']
    real={d:{a:avg([r for r in records if r['split']=='real' and r['graph']==d and r['arm']==a]) for a in realarms} for d in cfg()['real_datasets']}
    realgroups={mode:{a:avg([real[d][a] for d in ds]) for a in realarms} for mode,ds in [('homo',['Cora','CiteSeer']),('hetero',['texas','wisconsin'])]}
    rd={a:realgroups['hetero']['node_frame']['accuracy']-realgroups['hetero'][a]['accuracy'] for a in realarms if a!='node_frame'}
    rh=realgroups['homo']['node_frame']['accuracy']-realgroups['homo']['pcfi100']['accuracy']
    sg=all(comp[a]['accuracy']['ci95'][0]>0 for a in ['identity','dynamic']) and comp['pcfi100']['accuracy']['mean']>0 and guard['ci95'][0]>-.02
    rg=all(rd[a]>0 for a in ['identity','dynamic','pcfi100','native_pcfi100']) and rh>=-.02
    train=[json.loads((ROOT/'results'/f'node_frame_s{s}'/'training.json').read_text()) for s in cfg()['seeds']]
    fulls={split:{g:{a:avg([r for r in full if r['split']==split and r['arm']==a and (g=='all' or r['graph']==g)]) for a in arms} for g in (['all'] if split=='test' else cfg()['real_datasets'])} for split in ['test','real']}
    out={'synthetic':{a:avg(list(g.values())) for a,g in graphs.items()},'comparisons_node_minus':comp,'synthetic_label_groups':groups,
         'homo_guard':guard,'synthetic_by_case':secondary,'synthetic_by_seed':{str(s):{a:avg([r for r in primary if r['seed']==s and r['arm']==a]) for a in arms} for s in cfg()['seeds']},
         'zero_source_partition':{part:{a:avg([r for r in primary if r['arm']==a and ((r['observed_nodes']==0) if part=='zero' else r['observed_nodes']>0)]) for a in arms} for part in ['zero','nonzero']},
         'real':real,'real_groups':realgroups,'real_hetero_differences':rd,'real_homo_difference':rh,'full_feature':fulls,'geometry':geometry,
         'gates':{'synthetic':bool(sg),'real_input':bool(rg),'negative_lp_floor':bool(rd['negative_lp']>=0),'expand':bool(sg and rg and rd['negative_lp']>=0)},
         'training':{'new_checkpoints':3,'valid_updates':sum(r['updates'] for r in train),'training_seconds_sum':sum(r['seconds'] for r in train),'real_updates':0,'discarded_smoke_steps':1},
         'scope':'Fresh64syntheticgraphs/sameknownfamily and3fixedtrainseeds; oldtrainbank/schedule reused; all baselines reevaluated onfreshsyntheticinputs. Existingrealdevelopmentbanks/predictions reused forcomparisons. CI conditional onstoredforecasts;numericreplayseparate.'}
    dump(ROOT/'results/statistics.json',out);dump(ROOT/'results/records.json',records)
    print('ANALYSIS COMPLETE',out['gates'],flush=True)

if __name__=='__main__':main()

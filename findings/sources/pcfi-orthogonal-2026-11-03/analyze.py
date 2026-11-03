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
    x=np.array(x);r=np.random.default_rng(20391103);b=x[r.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n_base_graphs':len(x)}

def main():
    records=[];full=[];geometry={};arms=cfg()['arms']
    assert len(list((ROOT/'results').glob('*_s*/test_evaluation.json')))==12
    assert len(list((ROOT/'results').glob('*_s*/real_evaluation.json')))==12
    for arm in arms:
        ds=[]
        for seed in cfg()['seeds']:
            for split in ['test','real']:
                rows=json.loads((ROOT/'results'/f'{arm}_s{seed}'/f'{split}_evaluation.json').read_text())
                for row in rows:
                    pth=ROOT/row['prediction'];assert sha(pth)==row['prediction_sha256'];z=np.load(pth);p,y,m=z['p'],z['y'],z['missing']
                    key=dict(arm=arm,seed=seed,split=split,graph=row['graph'],case=row['case'],repetition=row.get('repetition',0),label_mode=row['label_mode'],feature_mode=row['feature_mode'],observed_nodes=row['observed_nodes'])
                    if row['case']=='full':full.append({**key,**metrics(p,y)});continue
                    met=metrics(p[m],y[m]);assert all(abs(met[k]-row['missing'][k])<1e-12 for k in ['accuracy','nll','brier'])
                    records.append({**key,**met});ds.append({'split':split,'diagnostics':row['diagnostics']})
        geometry[arm]={split:{k:float(np.mean([r['diagnostics'][k] for r in ds if r['split']==split])) for k in ['negative_cos_fraction','mean_abs_sin','second_coordinate_mean_abs','observed_change','spatial_steps','channel_corrections','warmup_spatial_steps']} for split in ['test','real']}
    # Frozen downstream checkpoint identical to old pcfi100; retain nativeinput floor.
    for seed in cfg()['seeds']:
        for row in json.loads((WORK/f'nsd_token_pfn/results/pcfi100_s{seed}/real_evaluation.json').read_text()):
            if row['case']=='full':continue
            pth=WORK/'nsd_token_pfn'/row['prediction'];assert sha(pth)==row['prediction_sha256'];original=np.load(pth);om=original['missing']
            records.append(dict(arm='pcfi_numpy',seed=seed,split='real',graph=row['graph'],case=row['case'],repetition=row['repetition'],label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**metrics(original['p'][om],original['y'][om])))
            pth=WORK/'nsd_token_pfn'/row['native_prediction'];assert sha(pth)==row['native_sha256'];z=np.load(pth);m=z['missing']
            records.append(dict(arm='native_pcfi',seed=seed,split='real',graph=row['graph'],case=row['case'],repetition=row['repetition'],label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**metrics(z['p'][m],z['y'][m])))
        for row in json.loads((WORK/f'nsd_token_pfn/results/mask_s{seed}/real_evaluation.json').read_text()):
            if row['case']=='full':continue
            pth=WORK/'nsd_token_pfn'/row['prediction'];assert sha(pth)==row['prediction_sha256'];z=np.load(pth);m=z['missing']
            records.append(dict(arm='mask_reference',seed=seed,split='real',graph=row['graph'],case=row['case'],repetition=row['repetition'],label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**metrics(z['p'][m],z['y'][m])))
    for row in json.loads((WORK/'sheaf_pcfi_pilot/results/public_baselines.json').read_text()):
        if row['case']=='full' or row['arm'] not in ['lp','negative_lp','context_prior']:continue
        pth=WORK/'sheaf_pcfi_pilot'/row['prediction'];assert sha(pth)==row['prediction_sha256'];z=np.load(pth);m=z['missing']
        records.append(dict(arm=row['arm'],seed=11,split='real',graph=row['graph'],case=row['case'],repetition=row['repetition'],label_mode=row['label_mode'],feature_mode='real',observed_nodes=row['observed_nodes'],**metrics(z['p'][m],z['y'][m])))
    primary=[r for r in records if r['split']=='test' and r['case'] in cfg()['primary_cases']];ng=cfg()['test_graphs']
    graph={a:{i:avg([r for r in primary if r['arm']==a and r['graph']==i]) for i in range(ng)} for a in arms}
    comp={candidate:{other:{k:ci([graph[candidate][i][k]-graph[other][i][k] for i in range(ng)]) for k in KEYS} for other in arms if other!=candidate} for candidate in ['scalar','edge','node']}
    homo=sorted({r['graph'] for r in primary if r['label_mode']=='homo'})
    guard={a:ci([graph[a][i]['accuracy']-graph['pcfi'][i]['accuracy'] for i in homo]) for a in ['edge','node']}
    secondary={case:{a:avg([r for r in records if r['split']=='test' and r['case']==case and r['arm']==a]) for a in arms} for case in cfg()['test_cases'] if case!='full'}
    realarms=arms+['pcfi_numpy','native_pcfi','mask_reference','lp','negative_lp','context_prior']
    real={d:{a:avg([r for r in records if r['split']=='real' and r['graph']==d and r['arm']==a]) for a in realarms} for d in cfg()['real_datasets']}
    groups={mode:{a:avg([real[d][a] for d in ds]) for a in realarms} for mode,ds in [('homo',['Cora','CiteSeer']),('hetero',['texas','wisconsin'])]}
    rg={a:{other:groups['hetero'][a]['accuracy']-groups['hetero'][other]['accuracy'] for other in realarms if other!=a} for a in ['edge','node']}
    hg={a:groups['homo'][a]['accuracy']-groups['homo']['pcfi']['accuracy'] for a in ['edge','node']}
    gates={a:{'synthetic':bool(all(comp[a][other]['accuracy']['ci95'][0]>0 for other in ['pcfi','scalar']) and guard[a]['ci95'][0]>-.02),
              'real_input':bool(all(rg[a][other]>0 for other in ['pcfi','scalar','native_pcfi']) and hg[a]>=-.02),
              'negative_lp_floor':bool(rg[a]['negative_lp']>=0)} for a in ['edge','node']}
    for a in gates:gates[a]['expand']=all(gates[a].values())
    train=[json.loads((ROOT/'results'/f'{arm}_s{seed}'/'training.json').read_text()) for arm in cfg()['learned_arms'] for seed in cfg()['seeds']]
    out={'synthetic':{a:avg(list(g.values())) for a,g in graph.items()},'comparisons':comp,'homo_guard':guard,'synthetic_by_case':secondary,
         'synthetic_by_seed':{str(seed):{a:avg([r for r in primary if r['seed']==seed and r['arm']==a]) for a in arms} for seed in cfg()['seeds']},
         'zero_source_partition':{part:{a:avg([r for r in primary if r['arm']==a and ((r['observed_nodes']==0) if part=='zero' else r['observed_nodes']>0)]) for a in arms} for part in ['zero','nonzero']},
         'real':real,'real_groups':groups,'real_hetero_differences':rg,'real_homo_difference':hg,'geometry':geometry,
         'full_feature':{split:{g:{a:avg([r for r in full if r['split']==split and r['arm']==a and (g=='all' or r['graph']==g)]) for a in arms} for g in (['all'] if split=='test' else cfg()['real_datasets'])} for split in ['test','real']},
         'gates':gates,'training':{'new_front_checkpoints':9,'valid_updates':sum(r['updates'] for r in train),'pfn_updates':0,'real_updates':0,'seconds_sum':sum(r['seconds'] for r in train),'discarded_smoke_steps':3},
         'scope':'SamePFNcheckpointperseedALLarms,frozenallphases;737paramfrontendsonlytraining.64freshsame-familytestgraphs;real4olddevelopmentgraphs.3fixedtrainingseeds;5000graphbootstrapconditionalstoredforecasts,unadjustedmultiplecomparisons. Bothprespecifiedcandidatesreportedwithoutpost-bankselection.'}
    dump(ROOT/'results/statistics.json',out);dump(ROOT/'results/records.json',records);print('ANALYSIS COMPLETE',gates,flush=True)

if __name__=='__main__':main()

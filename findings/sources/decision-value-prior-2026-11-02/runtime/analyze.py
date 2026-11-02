"""Post-lock analysis; never selects checkpoints or changes training inputs."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent
sys.path.insert(0,str(ROOT))
import experiment as E
import family as FA
import numpy as np
import json,hashlib,copy,time
from collections import Counter

def diff(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(17630000)
    vals=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
    return {'mean':float(d.mean()),'ci95':np.quantile(vals,[.025,.975]).tolist(),'graphs':len(d)}
def isprimary(r):return r['condition']=='uniform' and r['context']==8 and r['budget']==4 and r.get('mode','standard')=='standard'
def cells(r):return f"{r['condition']}_c{r['context']}_o{r['budget']}_{r.get('mode','standard')}"
def main():
    E.C.setup();locked=E.lock();c=E.cfg();arms={};metrics=['accuracy','nll','brier'];audit=[];train=[]
    import platform,scipy,torch,dgl
    E.write(ROOT/'results/environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'torch':torch.__version__,'dgl':dgl.__version__,'gpu':torch.cuda.get_device_name(0),'torch_threads':torch.get_num_threads(),
        'public_upstream_commit':locked['upstream_commit'],'timing_scope':'3 seed workers overlap; no isolated latency claim'})
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());banksha=E.C.sha(ROOT/'bank/manifest.json')
    for r in manifest['episodes']:assert E.C.sha(ROOT/'bank'/r['path'])==r['sha256']
    for interface in c['interfaces']:
        for prior in c['priors']:
            name=f'{prior}_{interface}';runs=[]
            for seed in c['seeds']:
                out=ROOT/'results'/f'{name}_s{seed}';rows=json.loads((out/'evaluation.json').read_text());info=json.loads((out/'training.json').read_text())
                assert info['checkpoint_sha']==E.C.sha(out/'trainable.pt') and info['bank_sha']==banksha
                steps=c['cost_updates'] if prior=='random_cost' else c['updates'];assert info['updates']==steps
                assert info['schedule_sha']==hashlib.sha256(json.dumps(E.schedule(seed,steps)).encode()).hexdigest()
                logs=json.loads((out/'train_log.json').read_text());assert len(logs)==steps
                assert Counter((r['design'][1],r['design'][2]) for r in logs)=={(k,b):steps//6 for k in c['contexts'] for b in c['budgets']}
                assert info['initial_sha']!=info['final_sha'] and info['frozen_unchanged']
                train.append(info)
                with np.load(out/'predictions.npz') as d:
                    assert d['p'].shape==(640,24,2) and d['y'].shape==(640,24)
                    for i,r in enumerate(rows):
                        assert np.max(np.abs(d['p'][i].sum(1)-1))<.002
                        mm=E.C.metrics(d['p'][i],d['y'][i])
                        for key in metrics:assert abs(mm[key]-r[key])<1e-12
                # One exact replay per final checkpoint, including reconstructed ablation input.
                model=E.load(prior,interface,seed);r=next(r for r in rows if isprimary(r));idx=rows.index(r);inp,y=E.read(r)
                pp=E.C.predict(model,inp)
                with np.load(out/'predictions.npz') as d:error=float(np.abs(pp-d['p'][idx]).max())
                assert error==0;assert E.C.parameter_digest(model,trainable=False)==info['frozen_sha']
                assert E.C.parameter_digest(model,trainable=True)==info['final_sha']
                import torch,gc
                del model;gc.collect();torch.cuda.empty_cache()
                audit.append({'arm':name,'seed':seed,'replay_max_error':error});runs.append(rows)
                print('AUDIT',name,seed,flush=True)
            assert all([(r['path'],r['mode']) for r in run]==[(r['path'],r['mode']) for r in runs[0]] for run in runs)
            ag=[]
            for idx,row in enumerate(runs[0]):
                ag.append({**{k:v for k,v in row.items() if k not in metrics},**{key:float(np.mean([run[idx][key] for run in runs])) for key in metrics}})
            arms[name]=ag
    for seed in c['seeds']:
        assert len({r['initial_sha'] for r in train if r['seed']==seed})==1
        assert len({r['schedule_sha'] for r in train if r['seed']==seed and r['prior']!='random_cost'})==1
    primary={name:{m:np.array([r[m] for r in rows if isprimary(r)]) for m in metrics} for name,rows in arms.items()}
    primary_means={name:{m:float(x.mean()) for m,x in val.items()} for name,val in primary.items()}
    comparisons={}
    for interface in c['interfaces']:
        for prior in ['entropy','decision']:
            name=f'{prior}_{interface}';comp={}
            for control in ['random','geometry','random_cost']:
                other=f'{control}_{interface}';comp[other]={m:diff(primary[name][m],primary[other][m]) for m in metrics}
            comparisons[name]=comp
    strong={name:{other:{m:diff(primary[name][m],primary[other][m]) for m in metrics}
                 for other in primary if other.endswith('_pcfi')} for name in ['entropy_mask','decision_mask']}
    intervention={}
    allcells={}
    for name,rows in arms.items():
        allcells[name]={cell:{m:float(np.mean([r[m] for r in rows if cells(r)==cell])) for m in metrics} for cell in sorted({cells(r) for r in rows})}
        p=primary[name];intervention[name]={}
        for mode in ['no_features','shuffled']:
            rr=[r for r in rows if r['condition']=='uniform' and r['context']==8 and r['budget']==4 and r['mode']==mode]
            intervention[name][mode]={m:diff(p[m],[r[m] for r in rr]) for m in metrics}
    refs=json.loads((ROOT/'results/reference.json').read_text());reference={}
    with np.load(ROOT/'results/reference_predictions.npz') as d:
        assert len(refs)==len(d['p'])==1536
        for i,r in enumerate(refs):
            for key,value in E.C.metrics(d['p'][i],d['y'][i]).items():assert abs(value-r[key])<1e-12
    # Reconstruct one reference from graph seed, without stored reference parameters or query labels.
    rr=next(r for r in refs if r['arm']=='exact_family' and isprimary(r));inp,y=E.read(rr)
    g=FA.graph(c['test_seed_start']+rr['graph']);o=np.flatnonzero(inp.observed.any(1))
    pr=FA.predict(g,inp.context,inp.labels_context,inp.query,o,inp.observed_values)[0]
    with np.load(ROOT/'results/reference_predictions.npz') as d:
        assert np.max(np.abs(np.stack([1-pr,pr],-1)-d['p'][refs.index(rr)]))<1e-12
    for name in ['exact_family','exact_no_features','context_frequency']:
        rr=[r for r in refs if r['arm']==name and isprimary(r)];reference[name]={m:float(np.mean([r[m] for r in rr])) for m in metrics}
        reference[name]['expected_accuracy']=float(np.mean([r['expected_accuracy'] for r in rr]))
    exact_feature_gain={m:diff([r[m] for r in refs if r['arm']=='exact_family' and isprimary(r)],
                             [r[m] for r in refs if r['arm']=='exact_no_features' and isprimary(r)]) for m in metrics}
    ds=json.loads((ROOT/'bank/designs.json').read_text());design_summary={}
    for k in c['contexts']:
        for b in c['budgets']:
            dd=[r for r in ds if r['context']==k and r['budget']==b]
            design_summary[f'c{k}_o{b}']={
                'same_entropy_decision_mask_fraction':float(np.mean([r['selected']['entropy']==r['selected']['decision'] for r in dd])),
                'score_rank_correlation':float(np.mean([np.corrcoef(np.argsort(np.argsort(r['J'])),np.argsort(np.argsort(r['V'])))[0,1] for r in dd])),
                'selected_scores':{a:{s:float(np.mean([r[s][r['selected'][a]] for r in dd])) for s in ['J','V','geometry']} for a in ['random','geometry','entropy','decision']}}
    scored_seconds=manifest['cost']['train']['cold_scoring_seconds'];cost={}
    for interface in c['interfaces']:
        extra=np.mean([r['seconds'] for r in train if r['interface']==interface and r['prior']=='random_cost'])-np.mean([r['seconds'] for r in train if r['interface']==interface and r['prior']=='random'])
        # Conservative charge entire shared cold bank scoring to one arm; no amortization discount.
        cost[interface]={'cold_all_scores_seconds':scored_seconds,'cold_entire_train_bank_seconds':manifest['cost']['train']['total_seconds'],
                         'mean_extra_random_training_seconds':float(extra),'extra_random_covers_cold_scoring':bool(extra>=scored_seconds),
                         'extra_random_covers_entire_bank_generation':bool(extra>=manifest['cost']['train']['total_seconds'])}
    gates={p:all(comparisons[f'{p}_mask'][f'{control}_mask']['accuracy']['ci95'][0]>0 for control in ['random','geometry','random_cost']) for p in ['entropy','decision']}
    seed_means={f"{r['prior']}_{r['interface']}_s{r['seed']}":{m:float(np.mean([z[m] for z in json.loads((ROOT/'results'/f"{r['prior']}_{r['interface']}_s{r['seed']}"/'evaluation.json').read_text()) if isprimary(z)])) for m in metrics} for r in train}
    direct_j_vs_v={it:{m:diff(primary[f'entropy_{it}'][m],primary[f'decision_{it}'][m]) for m in metrics} for it in c['interfaces']}
    stats=dict(primary_means=primary_means,comparisons=comparisons,strong_pipeline_floor=strong,gates=gates,direct_J_minus_V=direct_j_vs_v,
               interventions=intervention,reference_primary=reference,exact_feature_gain=exact_feature_gain,all_cells=allcells,
               seed_primary=seed_means,design_summary=design_summary,cost=cost,training=train,
               models=len(train),updates=sum(r['updates'] for r in train),prediction_arrays=len(audit)*640,
               scope='64 finite-family graphs;3 fixed seeds;4source/96nodes=95.833% whole-node missing;8 context labels TOTAL; query-source-context disjoint; pointwise graph-paired intervals.')
    E.write(ROOT/'results/statistics.json',stats)
    E.write(ROOT/'results/audit.json',{'status':'PASS','source_lock_verified':True,'bank_hashes_verified':True,'replays':audit,
        'identical_initializations':True,'matched_normal_schedules':True,'updates_verified':stats['updates'],
        'all_metrics_rescored':True,'primary_query_cohort_identical_for_interventions':True,'analysis_sha':E.C.sha(Path(__file__))})
    print('STATISTICS',json.dumps({'primary':primary_means,'gates':gates,'cost':cost}),flush=True)
if __name__=='__main__':main()

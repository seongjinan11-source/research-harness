"""Prespecified graph bootstrap, independent re-score/replay and CPU-output costs."""
import sys
sys.dont_write_bytecode = True
import sklearn
import argparse
import gc
import json
import time
import numpy as np
import torch
from pcfi_shared import ROOT,C,cfg,dump,sha,sources,rows,read,lock,prepare_geometry
from pcfi_model import predict
from pcfi_trial import load
METRICS=['accuracy','nll','brier']


def collect():
    data={};test=rows('test')
    for arm in cfg()['arms']:
        data[arm]={}
        for seed in cfg()['seeds']:
            record=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'evaluation.json').read_text(encoding='utf8'))
            data[arm][seed]=record
            assert len(record)==len(test)
    return test,data


def difference(a,b):
    d=a-b;rng=np.random.default_rng(202611021600)
    boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
    return {m:{'delta':float(d[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,m in enumerate(METRICS)}


def summarize():
    lock();test,data=collect()
    selectors={'primary':lambda r:r['mechanism'] in ['uniform_nodes','clustered_nodes'] and r['k']==4,
               'full':lambda r:r['mechanism']=='full','uniform':lambda r:r['mechanism']=='uniform_nodes' and r['k']==4,
               'clustered':lambda r:r['mechanism']=='clustered_nodes','cell':lambda r:r['mechanism']=='uniform_cells','k1':lambda r:r['k']==1}
    for mode in ['uniform_nodes','clustered_nodes']:
        for rate in [.9,.995]:selectors[f'{mode}_{rate}']=lambda r,m=mode,p=rate:r['mechanism']==m and r['rate']==p and r['k']==4
    output={}
    for name,selector in selectors.items():
        ids=[i for i,r in enumerate(test) if selector(r)]
        graphs=sorted({test[i]['graph'] for i in ids});v={};allv={}
        for arm in cfg()['arms']:
            a=[];b=[]
            for gid in graphs:
                episodes=[i for i in ids if test[i]['graph']==gid]
                targets=[];alltargets=[]
                for seed in cfg()['seeds']:
                    for i in episodes:
                        record=data[arm][seed][i]
                        target=record['missing_metrics'] if name!='full' else record
                        if target is not None:targets.append([target[m] for m in METRICS])
                        alltargets.append([record[m] for m in METRICS])
                assert targets,'No missing query in selected primary base graph'
                a.append(np.mean(targets,axis=0));b.append(np.mean(alltargets,axis=0))
            v[arm]=np.array(a);allv[arm]=np.array(b)
        output[name]={'graphs':len(graphs),'means':{a:dict(zip(METRICS,x.mean(0).tolist())) for a,x in v.items()},
             'all_query_secondary':{a:dict(zip(METRICS,x.mean(0).tolist())) for a,x in allv.items()},
             'contrasts':{'context-'+a:difference(v['context'],v[a]) for a in ['external','internal_fixed','label_blind']}}
    gate=all(d['accuracy']['ci95'][0]>0 for d in output['primary']['contrasts'].values())
    guard=output['full']['contrasts']['context-external']['accuracy']['ci95'][0]>-.02
    dump(ROOT/'results/statistics.json',{'subsets':output,'primary_gate':bool(gate),'full_guard':bool(guard),'expand':bool(gate and guard),
         'unit':'64 independent base graphs. Episode query mean, four primary conditions and fixed3 seeds averaged before5000 paired graph bootstrap. Subgroups exploratory/unadjusted; not seed-population CIs.'})
    print('SUMMARY',output['primary'],gate,guard,flush=True)


def audit():
    C.setup();torch.set_num_threads(2);state=lock()
    import pathlib
    bank=json.loads((ROOT/'bank/manifest.json').read_text(encoding='utf8'))
    for row in bank['episodes']:assert sha(ROOT/'bank'/row['path'])==row['sha256']
    test,data=collect();replay=[];schedules={};initial={};frozen=set();train=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            folder=ROOT/'results'/f'{arm}_s{seed}'
            info=json.loads((folder/'training.json').read_text(encoding='utf8'));train.append(info)
            assert info['updates']==256 and info['exposures']==256 and info['frozen_unchanged']
            assert schedules.setdefault(seed,info['schedule_sha256'])==info['schedule_sha256']
            assert initial.setdefault(seed,info['common_initial_sha256'])==info['common_initial_sha256']
            frozen.add(info['frozen_sha256'])
            model=load(arm,seed)
            for i,row in enumerate(data[arm][seed]):
                path=ROOT/row['prediction'];assert sha(path)==row['prediction_sha256']
                inp,extra,y=read(test[i])
                with np.load(path) as z:
                    p=z['p'];assert np.array_equal(y,z['y'])
                score=C.metrics(p,y)
                assert all(abs(score[m]-row[m])<1e-12 for m in METRICS)
                assert row['diagnostics']=={'observed_clamp_max':0.,'propagation_steps':100,'correlations':1,'pfn_layers':12}
                if i in [1,116,230]:
                    error=float(abs(predict(model,inp,extra)-p).max());assert error<.004
                    replay.append({'arm':arm,'seed':seed,'episode':i,'max_abs_error':error})
            del model;gc.collect();torch.cuda.empty_cache()
    assert len(frozen)==1
    primary=[r for r in test if r['mechanism'] in ['uniform_nodes','clustered_nodes'] and r['k']==4]
    assert len(primary)==256 and all(r['missing_queries']>0 for r in primary)
    dump(ROOT/'results/audit.json',{'source_config_bank_and_checkpoint_hashes_valid':True,'all_saved_predictions_recomputed':True,
         'same_schedules_and_common_initialization':True,'same_frozen_backbone':True,'exact_operations_and_clamps_all_evaluations':True,
         'new_models':len(train),'successful_updates':sum(r['updates'] for r in train),'exposures':sum(r['exposures'] for r in train),
         'development_smoke_updates_separate':4,'training_seconds_sum':sum(r['seconds'] for r in train),'replays':replay,
         'counts':{a:next(r['trainable_parameters'] for r in train if r['arm']==a) for a in cfg()['arms']},
         'zero_observation_primary_episodes':sum(r['observed_cells']==0 for r in primary),'primary_episodes':len(primary)})
    print('AUDIT_PASSED',len(replay),flush=True)


def cost():
    C.setup();torch.set_num_threads(2);lock()
    test=rows('test')
    chosen=[next(r for r in test if r['mechanism']==m and r['rate']==p and r['k']==4) for m in ['uniform_nodes','clustered_nodes'] for p in [.9,.995]]
    measured={}
    for arm in cfg()['arms']:
        model=load(arm,11);groups=[]
        for row in chosen:
            inp,_,_=read(row)
            def execute():
                geometry=prepare_geometry(inp,row['features'])
                return predict(model,inp,geometry)
            for _ in range(5):execute()
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(30):
                torch.cuda.synchronize();start=time.perf_counter();execute();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            groups.append({'setting':[row['mechanism'],row['rate']],'median_ms':float(np.median(times)),
                 'p25_ms':float(np.quantile(times,.25)),'p75_ms':float(np.quantile(times,.75)),
                 'additional_peak_mib':(torch.cuda.max_memory_allocated()-base)/2**20,'times_ms':times})
        measured[arm]={'mean_of_condition_medians_ms':float(np.mean([g['median_ms'] for g in groups])),'conditions':groups}
        print('COST',arm,measured[arm]['mean_of_condition_medians_ms'],flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/cost.json',{'methods':measured,'scope':'SameRTX3090 after our training workers; external workload uncontrolled. Visible CPU input->CPU probabilities includes graph/distance/confidence preprocessing and actual100 PCFI steps/correlation.5warmups+30repeats, same four prespecified cases, exclude model loading and bank I/O. Memory is additional work, not total GPU.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['summary','audit','cost']);a=p.parse_args()
    {'summary':summarize,'audit':audit,'cost':cost}[a.action]()

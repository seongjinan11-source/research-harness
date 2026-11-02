import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pilot as P
import sampler as S
import numpy as np
import torch
import json,time,gc,hashlib

def diff(a,b):
    x=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(18401000)
    samples=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),graphs=len(x))
def kl(p,q):
    p=np.clip(p,1e-8,1);p/=p.sum(-1,keepdims=True);q=np.clip(q,1e-8,1);q/=q.sum(-1,keepdims=True)
    return float((p*(np.log(p)-np.log(q))).sum(-1).mean())
def probs(p1):return np.stack((1-p1,p1),-1)
def collect():
    rows=P.test_rows();allrows=[];diagnostics={}
    with np.load(ROOT/'results/grid_reference.npz') as z:grid=z['grid6_exact'];gridbp=z['grid6_bp']
    constant=np.load(ROOT/'results/constant_bp.npz')['p']
    for seed in P.cfg()['seeds']:
        cond,extras=P.read_conditions(ROOT/'bank'/f'test_s{seed}.npz');exact=probs(extras['p1'])
        bp=np.load(ROOT/'results'/f'factor_bp_s{seed}.npz')['p']
        arrays={'factor_exact':exact,'factor_bp':bp,'constant_bp':constant,'grid6_exact':grid,'grid6_bp':gridbp}
        for arm in ['mask_pfn','pcfi_pfn']:arrays[arm]=np.load(ROOT/'results'/f'{arm}_s{seed}.npz')['p']
        for arm in P.cfg()['arms']:
            with np.load(ROOT/'results'/f'denoiser_s{seed}'/f'{arm}.npz') as z:
                arrays[arm]=probs(z['p1']);arrays[arm+'_frequency']=probs(z['raw'])
                diagnostics[f'{arm}_s{seed}']={'mean_final_ess':float(z['ess'].mean()),'mean_acceptance':float(z['acceptance'].mean()),
                    'mean_forced_final_beta_jump':float(z['final_jump'].mean()),'role_kl':kl(extras['role'],z['role']),
                    'beta_schedule_mean':z['beta'].mean(0).tolist()}
        for i,row in enumerate(rows):
            inp,q,y=P.read_test(row);missing=~inp['observed'][q].any(-1)
            for arm,array in arrays.items():
                p=array[i,q];metrics=P.C.metrics(p,y)
                out={**row,'seed':seed,'arm':arm,**metrics,'factor_kl':kl(exact[i,q],p),'grid6_kl':kl(grid[i,q],p)}
                if missing.any():out['missing_query']={**P.C.metrics(p[missing],y[missing]),'factor_kl':kl(exact[i,q][missing],p[missing]),'grid6_kl':kl(grid[i,q][missing],p[missing])}
                allrows.append(out)
    return allrows,diagnostics
def aggregate(rows,condition=lambda r:r['case']=='matched',missing=False):
    arms=sorted({r['arm'] for r in rows});graphs=sorted({r['graph'] for r in rows if condition(r)})
    result={}
    for arm in arms:
        result[arm]={m:[] for m in ['accuracy','nll','brier','factor_kl','grid6_kl']}
        for gid in graphs:
            rr=[r for r in rows if r['arm']==arm and r['graph']==gid and condition(r) and (not missing or 'missing_query'in r)]
            if not rr:continue
            for m in result[arm]:result[arm][m].append(float(np.mean([(r['missing_query'] if missing else r)[m] for r in rr])))
        result[arm]={m:np.array(v) for m,v in result[arm].items()}
    return result
def audit():
    P.lock();c=P.cfg();checks=[];diversity={}
    import importlib.util
    spec=importlib.util.spec_from_file_location('historical_factor_audit',P.OLD/'experiment.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.lock()
    for row in P.D.rows('train')[:c['train_tasks']]:assert P.D.sha(P.OLD/'bank'/row['path'])==row['sha256']
    u=P.C.setup();previous=json.loads((P.WORK/'decision_value_prior_pilot/results/lock.json').read_text())['public_weights']
    assert previous=={p.name:P.D.sha(p) for p in [u.ADAPTERS,u.LIMIX]}
    for name,digest in json.loads((ROOT/'results/bank_hashes.json').read_text()).items():assert P.D.sha(ROOT/name)==digest
    for seed in c['seeds']:
        out=ROOT/'results'/f'denoiser_s{seed}';info=json.loads((out/'training.json').read_text());assert info['updates']==c['updates'] and info['parameters']==8476
        assert P.D.sha(out/'checkpoint.pt')==info['checkpoint_sha']
        logs=json.loads((out/'train_log.json').read_text());assert len(logs)==c['updates'] and all(np.isfinite(r['loss']) for r in logs)
        model=S.Denoiser(c['width']).cuda().eval();model.load_state_dict(torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=True))
        assert P.C.parameter_digest(model)==info['final_sha'] and info['initial_sha']!=info['final_sha']
        cond,ref=P.read_conditions(ROOT/'bank'/f'test_s{seed}.npz')
        for arm in c['arms']:
            # Observe final particles at the existing RB function; no source/kernel/RNG changes.
            captured=[];original=S.rb
            def observer(cc,ss,hh,ww):
                captured.append((ss.detach().cpu().numpy().copy(),hh.detach().cpu().numpy().copy(),ww.detach().cpu().numpy().copy()))
                return original(cc,ss,hh,ww)
            S.rb=observer;replayed=[];unique=[];context_error=0
            try:
                for off in range(0,len(cond['j']),16):
                    cc=P.subset(cond,slice(off,off+16));p,raw,diag=S.sample(model,cc,arm,seed*100000+off,particles=c['particles'],stages=c['stages'],
                        sweeps=c['gibbs_cost_sweeps'] if arm=='smc_gibbs_cost' else 1,cess=c['cess_fraction'])
                    replayed.append(p.cpu().numpy());ss,hh,ww=captured.pop();clamp=cc['context'].cpu().numpy()
                    for b in range(len(ss)):
                        unique.append(len(np.unique(np.c_[ss[b],hh[b]],axis=0)))
                        known=clamp[b]>=0
                        if known.any():context_error=max(context_error,float(np.abs(ss[b,:,known]-(2*clamp[b,known]-1)[:,None]).max()))
            finally:S.rb=original
            with np.load(out/f'{arm}.npz') as z:error=float(np.abs(np.concatenate(replayed)-z['p1']).max())
            assert error==0 and context_error==0
            checks.append({'seed':seed,'arm':arm,'replay_max_error':error,'context_clamp_error':context_error})
            diversity[f'{arm}_s{seed}']={'unique_final_states_mean':float(np.mean(unique)),'unique_final_states_min':int(min(unique))}
            print('AUDIT',seed,arm,'replay',error,flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    # Original model's exact context and teacher-draw support.
    for seed in c['seeds']:
        cond,ref=P.read_conditions(ROOT/'bank'/f'train_s{seed}.npz');s=ref['draw_s'];h=ref['draw_h'];ctx=cond['context'].cpu().numpy()
        for i in range(len(s)):
            known=ctx[i]>=0
            assert np.all(s[i,:,known]==(2*ctx[i,known]-1)[:,None])
            assert np.all(cond['mass'][i].cpu().numpy()[h[i]]>0)
    public_replays=[]
    inp,q,y=P.read_test(P.test_rows()[0])
    for seed in c['seeds']:
        for arm in ['mask_pfn','pcfi_pfn']:
            model=P.C.Model('mask_plain' if arm=='mask_pfn' else 'pcfi_plain',seed)
            state=torch.load(P.OLD/'results'/f'{arm}_s{seed}'/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if p.requires_grad:p.copy_(state[name].to(p.device))
            prob=P.C.predict(model,P.D.public_input(inp,q,18700000))
            with np.load(ROOT/'results'/f'{arm}_s{seed}.npz') as z:err=float(np.abs(prob-z['p'][0,q]).max())
            assert err==0;public_replays.append({'arm':arm,'seed':seed,'max_error':err})
            del model;gc.collect();torch.cuda.empty_cache()
    P.dump(ROOT/'results/audit.json',{'status':'PASS','actual_updates':3*c['updates'],'new_models':3,'replays':checks,
        'replayed_episode_predictions':len(P.test_rows())*len(checks),'immutable_sources_and_frozen_checkpoints':True,
        'bank_hashes_verified':True,'context_clamps_verified':True,'teacher_joint_draw_support_verified':True,
        'diversity':diversity,'public_replays':public_replays,'analysis_sha':P.D.sha(Path(__file__))})
    return diversity
def main():
    P.C.setup();torch.set_num_threads(2);diversity=audit();rows,diag=collect();agg=aggregate(rows)
    comparisons={}
    pairs=[('diffusion_smc_fixed','diffusion_only'),('diffusion_smc_fixed','smc_gibbs_cost'),('diffusion_smc_fixed','factor_bp'),
           ('diffusion_smc_ess','diffusion_smc_fixed'),('diffusion_smc_ess','smc_gibbs_cost'),('diffusion_only','smc_gibbs_cost'),
           ('factor_exact','factor_bp'),('diffusion_smc_fixed','constant_bp'),('diffusion_smc_fixed','grid6_bp')]
    for a,b in pairs:comparisons[f'{a} - {b}']={m:diff(agg[a][m],agg[b][m]) for m in agg[a]}
    gates={'inference':all(comparisons[f'diffusion_smc_fixed - {b}']['factor_kl']['ci95'][1]<0 for b in ['diffusion_only','smc_gibbs_cost']),
           'ess_increment':comparisons['diffusion_smc_ess - diffusion_smc_fixed']['factor_kl']['ci95'][1]<0,
           'classification':all(comparisons[f'diffusion_smc_fixed - {b}']['accuracy']['ci95'][0]>0 for b in ['smc_gibbs_cost','factor_bp'])}
    means=lambda aa:{a:{m:float(v.mean()) for m,v in vv.items()} for a,vv in aa.items()}
    secondary={case:means(aggregate(rows,lambda r:r['case']==case)) for case in ['density_low','signal_high']}
    missing=means(aggregate(rows,missing=True))
    seeds={str(seed):means(aggregate([r for r in rows if r['seed']==seed])) for seed in P.cfg()['seeds']}
    settings={f'k{k}_o{o}':means(aggregate(rows,lambda r:r['case']=='matched' and r['context_total']==k and r['observed_nodes']==o)) for k in P.cfg()['contexts'] for o in P.cfg()['observed_nodes']}
    times={}
    for arm in P.cfg()['arms']:
        times[arm]=float(np.mean([json.loads((ROOT/'results'/f'denoiser_s{seed}'/f'{arm}_time.json').read_text())['seconds'] for seed in P.cfg()['seeds']]))
    data={'primary':means(agg),'comparisons':comparisons,'gates':gates,'diagnostics':diag,'diversity':diversity,'secondary_missing_query':missing,
          'secondary_ood':secondary,'seed_primary':seeds,'settings':settings,'batch16_cached_inference_total_seconds_per576':times,
          'new_models':3,'new_updates':3072,'factor_checkpoints_frozen':3,'public_checkpoints_frozen':6,
          'sampler_episode_predictions':3*5*576,'scope':'Primary48matchedgraphs,6disjointsettings,3fixedfactor/denoiser seeds. ExactFactor-KL is inference fit,Grid6-KL is a known-family reference;OOD Grid6 is not true Bayes.'}
    P.dump(ROOT/'results/statistics.json',data);P.dump(ROOT/'results/metrics.json',rows)
    print('RESULT',json.dumps({'primary':{k:v for k,v in data['primary'].items() if not k.endswith('frequency')},'gates':gates,'times':times}),flush=True)
if __name__=='__main__':main()

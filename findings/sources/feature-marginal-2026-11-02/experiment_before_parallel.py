"""Frozen conditional feature-law marginalization, with covariance and compute controls."""
import sys
sys.dont_write_bytecode = True
import sklearn  # Required Windows torch/DGL import order.
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.special import logsumexp
import torch

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent/'joint_imputation_pfn'
spec = importlib.util.spec_from_file_location('marginal_reference_common', REFERENCE/'common.py')
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def config():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    C.write_json(path, obj)


def sources():
    return {str(p.relative_to(ROOT)):C.sha(p) for p in ROOT.glob('*.py')} | {
        'protocol.json':C.sha(ROOT/'protocol.json'), 'reference_common':C.sha(REFERENCE/'common.py')}


def prepare():
    """Fresh graphs; only observed values and held-out label targets are retained."""
    from mgpfn import prior_adapter as PA, upstream as U
    c=config(); bank=ROOT/'bank'
    if (bank/'manifest.json').exists():
        raise RuntimeError('Preserve existing bank; do not redraw after observing results.')
    bank.mkdir(exist_ok=True)
    cfg=U.prior_config(); v=cfg['prior']['values'][0]
    v['graph']['n_nodes']=c['nodes']
    v['graph']['avg_degree']={'_distribution_':'uniform','min':4.,'max':12.}
    v['graph']['sampler']['values'][0]['min_first_level_n_nodes']=16
    v['scm']['base']['n_features']['min']=8; v['scm']['base']['n_features']['max']=16
    v['task']['n_classes']['max']=4
    v['task']['quantile']['min']=.3; v['task']['quantile']['max']=.7
    write(bank/'generator.json',cfg)
    rows=[]
    for gid in range(c['graphs']):
        requested=c['graph_seed_start']+gid
        for attempt in range(20):
            try:
                task=PA.sample_official_task(cfg,requested+attempt*100000000,
                    allowed_task_types=('binclass','multiclass'))
                break
            except AssertionError:
                pass
        else:
            raise RuntimeError('Invalid generator retries exhausted')
        _,task['y']=np.unique(task['y'],return_inverse=True)
        n,d=task['X_full'].shape; classes=len(np.unique(task['y']))
        rng=np.random.default_rng(requested+777)
        anchors={cl:rng.permutation(np.flatnonzero(task['y'][:task['n_train']]==cl)) for cl in range(classes)}
        assert all(len(a)>0 for a in anchors.values())
        query=rng.permutation(np.arange(task['n_train'],n))[:c['queries']]
        for sid,(mechanism,rate,k) in enumerate(c['settings']):
            context=np.concatenate([anchors[cl][:k] for cl in range(classes)])
            mask_seed=c['mask_seed_start']+gid*100+(2 if sid==5 else sid)
            observed=C.BASE.mask(task,mechanism,rate,mask_seed)
            values=C.observed_scale(task['X_full'],observed)
            poisoned=task['X_full'].copy(); poisoned[~observed]=1e12
            assert np.array_equal(values,C.observed_scale(poisoned,observed))
            pcfi=C.BASE.fill(np.where(observed,values,np.nan),task['src'],task['dst'],'pcfi',mechanism).clip(-10,10)
            if d%2:
                values=np.pad(values,((0,0),(0,1)))
                pcfi=np.pad(pcfi,((0,0),(0,1)))
                observed=np.pad(observed,((0,0),(0,1)),constant_values=True)
            path=bank/f'e{gid}_{sid}.npz'
            np.savez_compressed(path,observed_values=values,observed=observed,pcfi=pcfi,
                src=task['src'],dst=task['dst'],context=context,labels_context=task['y'][context],
                query=query,labels_query=task['y'][query],classes=classes,forward_seed=mask_seed+10000000)
            rows.append({'graph':gid,'setting':sid,'mechanism':mechanism,'rate':rate,'k':k,
                'path':path.name,'features':d,'nodes':n,'classes':classes,'context':len(context),
                'observed_cells':int(observed[:,:d].sum()),'observed_rows':int(observed[:,:d].any(1).sum()),
                'graph_seed':requested+attempt*100000000,'invalid_retries':attempt,'sha256':C.sha(path)})
        if (gid+1)%8==0:
            print('BANK',gid+1,flush=True)
    write(bank/'manifest.json',{'episodes':rows,'sources':sources(),'hidden_value_poisoning_invariance':True})


def read(row):
    with np.load(ROOT/'bank'/row['path']) as d:
        return C.Inputs(**{k:d[k] for k in C.Inputs.__annotations__}), d['labels_query'].astype(np.int64)


def kernels(inp):
    c=config(); n=len(inp.observed)
    a=np.zeros((n,n),np.float64); a[inp.src,inp.dst]=1
    a=np.maximum(a,a.T); np.fill_diagonal(a,0.)
    inv=np.maximum(a.sum(1),1.)**-.5
    s=a*inv[:,None]*inv[None,:]
    ev,vec=la.eigh(s,check_finite=False)
    result=[]
    for rho in c['rho_grid']:
        cov=(vec/(1-rho*ev)[None,:])@vec.T
        scale=np.sqrt(np.diag(cov)); cov=cov/scale[:,None]/scale[None,:]
        frac=c['global_covariance_fraction']
        cov=frac*np.ones((n,n))+(1-frac)*cov
        result.append((cov+cov.T)*.5)
    return result


def conditional(inp, dimensions, covariance):
    """Exact Gaussian conditioning under the stated surrogate, not SCM Bayes."""
    n,d=inp.observed_values.shape
    means=np.zeros((n,d),np.float64); variance=np.zeros_like(means)
    blocks=[]; evidence=0.; min_eigen=0.
    # Whole-node masks share conditioning geometry; cache repeated patterns.
    cache={}
    for f in range(dimensions):
        obs=np.flatnonzero(inp.observed[:,f]); mis=np.flatnonzero(~inp.observed[:,f])
        means[obs,f]=inp.observed_values[obs,f]
        key=tuple(obs)
        if key not in cache:
            if len(obs):
                chol=la.cholesky(covariance[np.ix_(obs,obs)],lower=True,check_finite=False)
                projection=la.cho_solve((chol,True),covariance[np.ix_(obs,mis)],check_finite=False).T
                cc=covariance[np.ix_(mis,mis)]-projection@covariance[np.ix_(obs,mis)]
            else:
                chol=np.empty((0,0)); projection=np.empty((len(mis),0)); cc=covariance.copy()
            cc=(cc+cc.T)*.5
            if len(mis):
                ev,vec=la.eigh(cc,check_finite=False)
                assert ev.min()>-1e-8,ev.min()
                root=vec*np.sqrt(np.maximum(ev,0))[None,:]
                min_eigen=min(min_eigen,float(ev.min()))
            else:
                root=np.empty((0,0))
            cache[key]=(chol,projection,cc,root)
        chol,projection,cc,root=cache[key]
        if len(obs):
            x=inp.observed_values[obs,f].astype(np.float64)
            means[mis,f]=projection@x
            z=la.solve_triangular(chol,x,lower=True,check_finite=False)
            evidence+=-.5*(len(obs)*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+z@z)
        variance[mis,f]=np.diag(cc)
        blocks.append((mis,root))
    return {'mean':means,'variance':variance,'blocks':blocks,'evidence':float(evidence),'minimum_covariance_eigenvalue':min_eigen}


def completions(inp, dimensions, covs, seed):
    c=config(); components=[conditional(inp,dimensions,k) for k in covs]
    loge=np.array([p['evidence'] for p in components])
    weights=np.exp(loge-logsumexp(loge))
    mean=sum(w*p['mean'] for w,p in zip(weights,components))
    # Padding and all observed entries are exact clamps.
    mean=np.where(inp.observed,inp.observed_values,mean)
    rng=np.random.default_rng(seed)
    regimes=rng.choice(len(covs),size=c['views'],p=weights)
    joint=[]; diagonal=[]; clipped=0; total=0
    for regime in regimes:
        p=components[int(regime)]; x=p['mean'].copy(); xd=x.copy()
        for f,(mis,root) in enumerate(p['blocks']):
            z=rng.standard_normal(len(mis))
            x[mis,f]+=root@z
            xd[mis,f]+=np.sqrt(np.maximum(p['variance'][mis,f],0))*z
        for a in (x,xd):
            clipped+=int((np.abs(a[~inp.observed])>10).sum()); total+=int((~inp.observed).sum())
        joint.append(np.where(inp.observed,inp.observed_values,np.clip(x,-10,10)).astype(np.float32))
        diagonal.append(np.where(inp.observed,inp.observed_values,np.clip(xd,-10,10)).astype(np.float32))
    arms={'zero':[inp.observed_values]*c['views'],'pcfi':[inp.pcfi]*c['views'],
        'native_nan':[np.where(inp.observed,inp.observed_values,np.nan)]*c['views'],
        'posterior_mean':[np.clip(mean,-10,10).astype(np.float32)]*c['views'],
        'joint_samples':joint,'diagonal_samples':diagonal}
    for vals in arms.values():
        assert all(np.array_equal(a[inp.observed],inp.observed_values[inp.observed]) for a in vals)
    return arms,{'regime_weights':weights.tolist(),'regime_draws':regimes.tolist(),
        'maximum_clamp_error':0.,'sample_clipping_fraction':clipped/max(total,1),
        'conditional_min_eigenvalue':min(p['minimum_covariance_eigenvalue'] for p in components)}


def load_model(seed):
    # pcfi_plain is only the existing wrapper's switch to read inp.pcfi.
    # Actual weights always come from mask_plain, for every candidate and control.
    model=C.Model('pcfi_plain',seed)
    path=REFERENCE/'results'/f'mask_plain_s{seed}'/'trainable.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['arm']=='mask_plain' and ck['seed']==seed
    state=model.state_dict(); state.update(ck['state_dict']); model.load_state_dict(state,strict=True)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    return model,path


def forward_views(model, inp, arrays, view_count=None):
    probabilities=[]
    for view,values in enumerate(arrays[:view_count]):
        altered=copy.copy(inp); altered.pcfi=values
        altered.forward_seed=int(inp.forward_seed)+view*100003
        probabilities.append(C.predict(model,altered))
    return np.stack(probabilities)


def preflight():
    rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    inp,y=read(next(r for r in rows if r['mechanism']=='uniform_nodes' and r['rate']==.9))
    row=next(r for r in rows if r['mechanism']=='uniform_nodes' and r['rate']==.9)
    covs=kernels(inp); arms,diag=completions(inp,row['features'],covs,9182)
    assert not hasattr(inp,'labels_query') and not hasattr(inp,'hidden_values')
    # Completion law is label-independent; labels enter PFN only through context.
    changed=copy.deepcopy(inp); changed.labels_context=changed.labels_context[::-1].copy()
    other,_=completions(changed,row['features'],covs,9182)
    assert all(np.array_equal(arms[a][0],other[a][0],equal_nan=True) for a in arms)
    # Node permutation equivariance of exact moments, not sample-root eigenvectors.
    order=np.random.default_rng(9).permutation(len(inp.observed)); inv=np.argsort(order)
    ren=copy.deepcopy(inp)
    for key in ('observed_values','observed','pcfi'): setattr(ren,key,getattr(inp,key)[order])
    ren.src=inv[inp.src]; ren.dst=inv[inp.dst]
    permuted=kernels(ren); maximum=0.
    for k,kp in zip(covs,permuted):
        p=conditional(inp,row['features'],k); pp=conditional(ren,row['features'],kp)
        maximum=max(maximum,float(np.abs(p['mean'][order]-pp['mean']).max()),float(np.abs(p['variance'][order]-pp['variance']).max()))
    assert maximum<1e-7,maximum
    # Conditional sample covariance sanity, including an actual nonzero off diagonal.
    p=conditional(inp,row['features'],covs[-1]); mis,root=p['blocks'][0]
    z=np.random.default_rng(512).standard_normal((len(mis),12000))
    actual=np.cov(root@z,bias=True); expected=root@root.T
    relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert relative<.15,relative
    model,path=load_model(config()['checkpoint_seeds'][0]); before=C.parameter_digest(model)
    outputs={a:forward_views(model,inp,vals,1)[0] for a,vals in arms.items()}
    assert all(np.isfinite(p).all() for p in outputs.values())
    repeat=forward_views(model,inp,arms['joint_samples'],1)[0]
    assert np.array_equal(outputs['joint_samples'],repeat)
    fullrow=next(r for r in rows if r['mechanism']=='full'); full,_=read(fullrow)
    complete,fd=completions(full,fullrow['features'],kernels(full),42)
    assert all(np.array_equal(complete['zero'][0],vals[0],equal_nan=True) for vals in complete.values())
    assert C.parameter_digest(model)==before
    write(ROOT/'results/preflight.json',{'finite_all_arms':True,'input_excludes_query_labels':True,
        'completion_label_invariance':True,'moment_node_permutation_max_error':maximum,
        'sample_covariance_relative_frobenius_error_12000':relative,'full_inputs_identical':True,
        'repeat_identical':True,'weights_unchanged':True,'checkpoint_sha256':C.sha(path),
        'diagnostics':diag,'sources':sources()})
    del model;gc.collect();torch.cuda.empty_cache()
    print('PREFLIGHT_OK',flush=True)


def run():
    c=config();rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    lock={'sources':sources(),'bank_manifest_sha256':C.sha(ROOT/'bank/manifest.json'),
        'checkpoint_sha256':{str(s):C.sha(REFERENCE/'results'/f'mask_plain_s{s}'/'trainable.pt') for s in c['checkpoint_seeds']}}
    path=ROOT/'results/lock.json'
    if path.exists(): assert json.loads(path.read_text())==lock
    else: write(path,lock)
    generated={}
    for row in rows:
        inp,_=read(row)
        generated[(row['graph'],row['setting'])]=completions(inp,row['features'],kernels(inp),row['graph_seed']*11+row['setting'])
    for seed in c['checkpoint_seeds']:
        out=ROOT/'results'/f's{seed}'
        if (out/'evaluation.json').exists(): continue
        out.mkdir(parents=True,exist_ok=True)
        model,_=load_model(seed); before=C.parameter_digest(model); records=[];start=time.perf_counter()
        for i,row in enumerate(rows):
            inp,y=read(row);arms,diagnostics=generated[(row['graph'],row['setting'])]
            for arm,arrays in arms.items():
                pred=out/f'{row["graph"]}_{row["setting"]}_{arm}.npz'
                if pred.exists():
                    with np.load(pred) as d: views=d['views']; assert np.array_equal(d['y'],y)
                    seconds=None
                else:
                    torch.cuda.synchronize();t=time.perf_counter()
                    views=forward_views(model,inp,arrays)
                    torch.cuda.synchronize();seconds=time.perf_counter()-t
                    np.savez_compressed(pred,views=views,y=y)
                probability=views.astype(np.float64).mean(0)
                records.append({**row,'arm':arm,'seed':seed,**C.metrics(probability,y),
                    'single_view_metrics':C.metrics(views[0],y),'prediction':pred.relative_to(ROOT).as_posix(),
                    'seconds':seconds,'diagnostics':diagnostics})
            if (i+1)%12==0:
                progress={'seed':seed,'episodes':i+1,'total':len(rows),'seconds':time.perf_counter()-start}
                write(ROOT/'results/progress.json',progress);print('EVAL',progress,flush=True)
        assert C.parameter_digest(model)==before
        write(out/'evaluation.json',records)
        write(out/'audit.json',{'weights_unchanged':True,'seconds':time.perf_counter()-start,'episodes':len(rows),'probability_files':len(records)})
        del model;gc.collect();torch.cuda.empty_cache()


def subset(rows, group):
    if group in ('uniform','clustered'): return [r for r in rows if r['mechanism']==group+'_nodes' and r['k']==4]
    if group=='full': return [r for r in rows if r['mechanism']=='full']
    if group=='k1': return [r for r in rows if r['k']==1]
    return [r for r in rows if f'{r["mechanism"]}:{r["rate"]}:{r["k"]}'==group]


def analyze():
    c=config();rows=[]
    for seed in c['checkpoint_seeds']:rows+=json.loads((ROOT/'results'/f's{seed}'/'evaluation.json').read_text())
    for r in rows:
        with np.load(ROOT/r['prediction']) as d: m=C.metrics(d['views'].astype(np.float64).mean(0),d['y'])
        assert all(abs(m[k]-r[k])<1e-10 for k in m)
    groups=['uniform','clustered','full','k1']+[f'{m}:{r}:{k}' for m,r,k in c['settings']]
    summary=[];pairs=[];boot=np.random.default_rng(20261102).integers(c['graphs'],size=(5000,c['graphs']))
    for group in groups:
        selected=subset(rows,group)
        for arm in c['arms']:
            v=[r for r in selected if r['arm']==arm]
            summary.append({'group':group,'arm':arm,**{k:float(np.mean([r[k] for r in v])) for k in ('nll','accuracy','brier')},
                'per_seed_nll':{str(s):float(np.mean([r['nll'] for r in v if r['seed']==s])) for s in c['checkpoint_seeds']},
                'single_view_nll':float(np.mean([r['single_view_metrics']['nll'] for r in v]))})
        for base in (a for a in c['arms'] if a!='joint_samples'):
            for metric in ('nll','accuracy','brier'):
                def graphvalues(arm):
                    return np.array([np.mean([r[metric] for r in selected if r['arm']==arm and r['graph']==g]) for g in range(c['graphs'])])
                diff=graphvalues('joint_samples')-graphvalues(base)
                lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
                pairs.append({'group':group,'contrast':'joint_samples - '+base,'metric':metric,'delta':float(diff.mean()),'low':float(lo),'high':float(hi)})
    passed=all(next(r for r in pairs if r['group']==g and r['contrast']=='joint_samples - '+a and r['metric']=='nll')['high']<0
        for g in ('uniform','clustered') for a in ('posterior_mean','diagonal_samples','zero','pcfi'))
    priors=[]
    manifest=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    for r in manifest:
        inp,y=read(r);freq=np.bincount(inp.labels_context,minlength=int(inp.classes)).astype(float)
        p=np.broadcast_to((freq+1)/(freq.sum()+int(inp.classes)),(len(y),int(inp.classes)))
        priors.append({**r,'context_prior':C.metrics(p,y),'uniform_prior':C.metrics(np.ones_like(p)/int(inp.classes),y)})
    prior_summary={g:{kind:{metric:float(np.mean([r[kind][metric] for r in subset(priors,g)])) for metric in ('nll','accuracy','brier')}
        for kind in ('context_prior','uniform_prior')} for g in ('uniform','clustered','full','k1')}
    full=[r for r in rows if r['mechanism']=='full']
    full_max=0.
    for seed in c['checkpoint_seeds']:
        for gid in range(c['graphs']):
            rr=[r for r in full if r['seed']==seed and r['graph']==gid]
            values=[]
            for r in rr:
                with np.load(ROOT/r['prediction']) as d:values.append(d['views'])
            full_max=max(full_max,max(float(np.abs(values[0]-v).max()) for v in values))
    assert full_max==0.,full_max
    write(ROOT/'results/summary.json',summary);write(ROOT/'results/paired.json',pairs)
    write(ROOT/'results/priors.json',prior_summary)
    write(ROOT/'results/audit.json',{'saved_probability_files':len(rows),'scored_probabilities':len(rows)*c['views'],
        'full_observed_max_arm_probability_delta':full_max,'all_saved_metrics_independently_recomputed':True,
        'hidden_feature_values_not_retained_in_bank':True,'sources':sources()})
    write(ROOT/'results/decision.json',{'primary_passed':passed,'decision':'Conditional Go, frozen pilot only' if passed else 'No-Go for scaling this frozen surrogate completion operator; trained alternatives remain untested.'})
    print('COMPLETED','primary_passed',passed,flush=True)
    for row in summary:
        if row['group'] in ('uniform','clustered','k1'): print(row['group'],row['arm'],round(row['nll'],6),round(row['accuracy'],6),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','preflight','run','analyze','all']);args=parser.parse_args()
    C.setup()
    if args.command=='all':
        if not (ROOT/'bank/manifest.json').exists():prepare()
        if not (ROOT/'results/preflight.json').exists():preflight()
        run();analyze()
    else:globals()[args.command]()


if __name__=='__main__': main()

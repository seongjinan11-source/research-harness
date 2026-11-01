"""Run the prespecified random-fill controls without changing previous experiments."""
import sys
sys.dont_write_bytecode=True
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'feature_marginal_pfn'
spec=importlib.util.spec_from_file_location('random_control_reference',REFERENCE/'experiment.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
import argparse
import copy
import gc
import json
import time
import numpy as np
import torch


def config():return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def write(path,obj):E.write(path,obj)


def sources():
    return {'experiment.py':E.C.sha(ROOT/'experiment.py'),'protocol.json':E.C.sha(ROOT/'protocol.json'),
        'reference_experiment':E.C.sha(REFERENCE/'experiment.py'),'reference_common':E.C.sha(E.REFERENCE/'common.py')}


def rows():return json.loads((REFERENCE/'bank/manifest.json').read_text())['episodes']


def generate(inp,row):
    covs=E.kernels(inp);dims=row['features'];n,d=inp.observed.shape
    original,diagnostics=E.completions(inp,dims,covs,row['graph_seed']*11+row['setting'])
    components=[E.conditional(inp,dims,c) for c in covs]
    weights=np.array(diagnostics['regime_weights'])
    mixture_mean=sum(w*c['mean'] for w,c in zip(weights,components))
    variance=sum(w*(c['variance']+(c['mean']-mixture_mean)**2) for w,c in zip(weights,components))
    scale=np.sqrt(np.maximum(variance,0))
    out={a:[] for a in config()['arms']};out['conditional_joint']=original['joint_samples']
    # Independent controls use a fixed schedule and the same random normals among nulls.
    rng=np.random.default_rng(row['graph_seed']*11+row['setting']+170000000)
    prior_regimes=rng.integers(len(covs),size=config()['views'])
    roots={};clipped={a:0 for a in out};total_missing=int((~inp.observed).sum())*config()['views']
    for view in range(config()['views']):
        chosen=int(diagnostics['regime_draws'][view]);prior_r=int(prior_regimes[view])
        iid=np.zeros((n,d),np.float64);unit=iid.copy();graph=iid.copy()
        for f in range(dims):
            missing=np.flatnonzero(~inp.observed[:,f]);z=rng.standard_normal(len(missing))
            iid[missing,f]=scale[missing,f]*z;unit[missing,f]=z
            key=(prior_r,tuple(missing))
            if key not in roots:
                cc=covs[prior_r][np.ix_(missing,missing)]
                if len(missing):
                    ev,vec=E.la.eigh(cc,check_finite=False)
                    assert ev.min()>-1e-8
                    roots[key]=vec*np.sqrt(np.maximum(ev,0))[None,:]
                else:roots[key]=np.empty((0,0))
            graph[missing,f]=scale[missing,f]*(roots[key]@z)
        wrong=original['joint_samples'][view].astype(np.float64)-2*components[chosen]['mean']
        for name,array in [('iid_variance_matched',iid),('graph_prior_variance_matched',graph),
                ('iid_unit',unit),('wrong_sign_mean',wrong)]:
            clipped[name]+=int((np.abs(array[~inp.observed])>10).sum())
            array=np.where(inp.observed,inp.observed_values,np.clip(array,-10,10)).astype(np.float32)
            assert np.array_equal(array[inp.observed],inp.observed_values[inp.observed])
            out[name].append(array)
    return {a:np.stack(v) for a,v in out.items()}, {
        'conditional_regimes':diagnostics['regime_draws'],'prior_regimes':prior_regimes.tolist(),
        'max_variance':float(variance.max()),'variance_minimum':float(variance.min()),
        'clip_fraction':{a:v/max(total_missing,1) for a,v in clipped.items()},'observed_clamp_max_error':0.}


def prepare():
    bank=ROOT/'bank';bank.mkdir(exist_ok=True);manifest=[]
    if (bank/'manifest.json').exists():raise RuntimeError('Preserve fixed completion bank')
    for index,row in enumerate(rows()):
        inp,_=E.read(row);arrays,diag=generate(inp,row)
        path=bank/f'{row["graph"]}_{row["setting"]}.npz'
        np.savez_compressed(path,**arrays)
        manifest.append({**row,'completion_path':path.name,'completion_sha256':E.C.sha(path),'diagnostics':diag})
        if (index+1)%24==0:print('PREPARED',index+1,flush=True)
    write(bank/'manifest.json',{'episodes':manifest,'sources':sources(),
        'reference_manifest_sha256':E.C.sha(REFERENCE/'bank/manifest.json')})


def preflight():
    manifest=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row=next(r for r in manifest if r['mechanism']=='uniform_nodes' and r['rate']==.9)
    inp,_=E.read(row);arrays,diag=generate(inp,row)
    with np.load(ROOT/'bank'/row['completion_path']) as data:
        assert all(np.array_equal(arrays[a],data[a]) for a in arrays)
    changed=copy.deepcopy(inp);changed.labels_context=changed.labels_context[::-1].copy()
    renamed,_=generate(changed,row)
    assert all(np.array_equal(arrays[a],renamed[a]) for a in arrays)
    assert not hasattr(inp,'labels_query') and not hasattr(inp,'hidden_values')
    model,path=E.load_model(config()['checkpoint_seeds'][0]);before=E.C.parameter_digest(model)
    output={a:E.forward_views(model,inp,v,1) for a,v in arrays.items()}
    assert all(np.isfinite(p).all() for p in output.values())
    assert E.C.parameter_digest(model)==before
    fullrow=next(r for r in manifest if r['mechanism']=='full')
    with np.load(ROOT/'bank'/fullrow['completion_path']) as data:
        assert all(np.array_equal(data[a],data['conditional_joint']) for a in config()['arms'])
    write(ROOT/'results/preflight.json',{'finite_predictions':True,'deterministic_completions':True,
        'completion_label_invariance':True,'input_excludes_query_labels':True,'weights_unchanged':True,
        'full_observed_inputs_identical':True,'checkpoint_sha256':E.C.sha(path),'sources':sources()})
    del model;gc.collect();torch.cuda.empty_cache();print('PREFLIGHT_OK',flush=True)


def lock():
    state={'sources':sources(),'bank_manifest_sha256':E.C.sha(ROOT/'bank/manifest.json'),
        'reference_lock_sha256':E.C.sha(REFERENCE/'results/lock.json'),
        'checkpoints':{str(s):E.C.sha(E.REFERENCE/'results'/f'mask_plain_s{s}'/'trainable.pt') for s in config()['checkpoint_seeds']}}
    path=ROOT/'results/lock.json'
    if path.exists():assert json.loads(path.read_text())==state
    else:write(path,state)


def run(seed):
    lock();manifest=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    out=ROOT/'results'/f's{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists():return
    model,_=E.load_model(seed);before=E.C.parameter_digest(model);records=[];start=time.perf_counter()
    for index,row in enumerate(manifest):
        inp,y=E.read(row);assert E.C.sha(REFERENCE/'bank'/row['path'])==row['sha256']
        assert E.C.sha(ROOT/'bank'/row['completion_path'])==row['completion_sha256']
        with np.load(ROOT/'bank'/row['completion_path']) as data:arrays={a:data[a] for a in config()['arms']}
        full=None
        for arm in config()['arms']:
            path=out/f'{row["graph"]}_{row["setting"]}_{arm}.npz'
            if path.exists():
                with np.load(path) as data:views=data['views'];assert np.array_equal(y,data['y'])
            elif row['mechanism']=='full' and full is not None:
                views=full;np.savez_compressed(path,views=views,y=y)
            else:
                views=E.forward_views(model,inp,arrays[arm]);np.savez_compressed(path,views=views,y=y)
            if row['mechanism']=='full':full=views
            records.append({**row,'seed':seed,'arm':arm,'prediction':path.relative_to(ROOT).as_posix(),
                **E.C.metrics(views.astype(np.float64).mean(0),y)})
        if (index+1)%24==0:
            p={'seed':seed,'episodes':index+1,'total':len(manifest),'seconds':time.perf_counter()-start}
            write(out/'progress.json',p);print('EVAL',p,flush=True)
    assert E.C.parameter_digest(model)==before
    write(out/'evaluation.json',records);write(out/'audit.json',{'weights_unchanged':True,'seconds':time.perf_counter()-start})
    print('FINISHED',seed,flush=True)


def analyze():
    lock();c=config();allrows=[];original=[]
    for seed in c['checkpoint_seeds']:
        allrows+=json.loads((ROOT/'results'/f's{seed}'/'evaluation.json').read_text())
        original+=json.loads((REFERENCE/'results'/f's{seed}'/'evaluation.json').read_text())
    replay=[]
    for row in allrows:
        with np.load(ROOT/row['prediction']) as data:
            p=data['views'];measured=E.C.metrics(p.astype(np.float64).mean(0),data['y'])
        assert all(abs(measured[k]-row[k])<1e-10 for k in measured)
        if row['arm']=='conditional_joint':
            old=next(r for r in original if r['seed']==row['seed'] and r['graph']==row['graph'] and r['setting']==row['setting'] and r['arm']=='joint_samples')
            with np.load(REFERENCE/old['prediction']) as data:delta=float(np.abs(p-data['views']).max())
            replay.append({'seed':row['seed'],'graph':row['graph'],'setting':row['setting'],
                'max_probability_delta':delta,'nll_delta':row['nll']-old['nll']})
    groups=['uniform','clustered','full','k1']+[f'{m}:{r}:{k}' for m,r,k in E.config()['settings']]
    summary=[];paired=[];boot=np.random.default_rng(2026110202).integers(c['graphs'],size=(5000,c['graphs']))
    for group in groups:
        subset=E.subset(allrows,group)
        for arm in c['arms']:
            rr=[r for r in subset if r['arm']==arm]
            summary.append({'group':group,'arm':arm,**{m:float(np.mean([r[m] for r in rr])) for m in ('nll','accuracy','brier')},
                'per_seed_nll':{str(s):float(np.mean([r['nll'] for r in rr if r['seed']==s])) for s in c['checkpoint_seeds']}})
        for other in c['arms'][1:]:
            for metric in ('nll','accuracy','brier'):
                def gv(arm):return np.array([np.mean([r[metric] for r in subset if r['arm']==arm and r['graph']==g]) for g in range(c['graphs'])])
                diff=gv('conditional_joint')-gv(other);lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
                paired.append({'group':group,'contrast':'conditional_joint - '+other,'metric':metric,'delta':float(diff.mean()),'low':float(lo),'high':float(hi)})
    passed=all(next(r for r in paired if r['group']==g and r['contrast']=='conditional_joint - '+a and r['metric']=='nll')['high']<0
        for g in ('uniform','clustered') for a in ('iid_variance_matched','graph_prior_variance_matched'))
    write(ROOT/'results/summary.json',summary);write(ROOT/'results/paired.json',paired)
    write(ROOT/'results/replay.json',{'episodes':replay,'maximum_probability_delta':max(r['max_probability_delta'] for r in replay),
        'mean_nll_delta':float(np.mean([r['nll_delta'] for r in replay]))})
    write(ROOT/'results/decision.json',{'primary_passed':passed,
        'decision':'Conditional Go for label-conditioned law pilot; no learned method established' if passed else 'Frozen Gaussian conditional advantage over random-fill nulls not established; no automatic scaling of this operator',
        'does_not_close':'Context-conditioned feature inference, shared learned encoders, coherence learning and real graph transfer remain untested.'})
    write(ROOT/'results/audit.json',{'probability_files':len(allrows),'scored_views':len(allrows)*c['views'],
        'all_saved_metrics_independently_recomputed':True,'parameter_updates':0,
        'full_observed_shared_identical_inputs':True,'sources':sources()})
    print('COMPLETE','primary_passed',passed,flush=True)
    for r in summary:
        if r['group'] in ('uniform','clustered','k1'):print(r['group'],r['arm'],round(r['nll'],6),round(r['accuracy'],6),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','preflight','run','analyze'])
    ap.add_argument('--seed',type=int,choices=config()['checkpoint_seeds']);args=ap.parse_args()
    E.C.setup()
    if args.command=='run':assert args.seed is not None;run(args.seed)
    else:globals()[args.command]()


if __name__=='__main__':main()

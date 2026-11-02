"""Frozen paired sign-transport diagnostic. No optimizer or new training."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import time
from collections import deque
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
OLD = WORK / 'joint_imputation_pfn'
sys.path.insert(0, str(WORK / 'internal_pcfi_pilot'))
import pcfi_shared as S
from imputation import adjacency, impute
C, dump, sha = S.C, S.dump, S.sha
METRICS = ['accuracy', 'nll', 'brier']


def cfg(): return json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def files():
    return [OLD / 'results' / f'{a}_s{s}' / p for a in cfg()['arms'] for s in cfg()['seeds'] for p in ['trainable.pt', 'training.json']]


def sources():
    paths = list(ROOT.glob('*.py')) + [ROOT / 'protocol.json']
    paths += [WORK / p for p in ['internal_pcfi_pilot/pcfi_shared.py', 'joint_imputation_pfn/common.py',
        'joint_imputation_pfn/protocol.json', 'joint_imputation_pfn/experiment.py', 'joint_imputation_pfn/run_manifest.json',
        'joint_imputation_pfn/runtime/numerical_repair.json', 'regret_prior_pilot/core.py', 'fisf_graphpfn/imputation.py']]
    return {str(p.relative_to(WORK)): sha(p) for p in paths}


def load(arm, seed):
    from mgpfn import upstream as U
    folder = OLD / 'results' / f'{arm}_s{seed}'
    info = json.loads((folder / 'training.json').read_text(encoding='utf8'))
    path = folder / 'trainable.pt'; assert sha(path) == info['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['arm'] == arm and state['seed'] == seed
    assert state['metadata'] == info['metadata'] == json.loads((OLD / 'run_manifest.json').read_text())
    for name in ['common.py', 'protocol.json']:
        assert sha(OLD / name) == state['metadata']['sources'][name]
    assert sha(OLD / 'bank/manifest.json') == state['metadata']['bank_manifest_sha256']
    assert {p.name: sha(p) for p in [U.ADAPTERS, U.LIMIX]} == state['metadata']['public_weights']
    model = C.Model(arm, seed)
    assert C.parameter_digest(model, trainable=True) == info['initial_trainable_sha256']
    assert C.parameter_digest(model, trainable=False) == info['frozen_sha256']
    params = {n: p for n,p in model.named_parameters() if p.requires_grad}
    assert set(params) == set(state['state_dict'])
    with torch.no_grad():
        for n,p in params.items(): p.copy_(state['state_dict'][n].to(p.device))
    return model.eval()


def gauge(a):
    graph = a.copy(); graph.setdiag(0); graph.eliminate_zeros()
    b = np.zeros(graph.shape[0], np.float32); b[0] = 1; queue = deque([0])
    while queue:
        i = queue.popleft()
        for j in graph.indices[graph.indptr[i]:graph.indptr[i+1]]:
            if b[j] == 0: b[j] = -b[i]; queue.append(int(j))
            else: assert b[j] == -b[i], 'Not bipartite'
    assert (b != 0).all(), 'Not connected'
    return b


def master(seed):
    c = cfg(); n = c['nodes']; f = c['features']; rng = np.random.default_rng(seed)
    a = np.zeros((n,n), bool)
    for i in range(n): a[i,(i+1)%n] = True; a[(i+1)%n,i] = True
    for _ in range(64):
        i = int(rng.integers(n//2))*2; j = int(rng.integers(n//2))*2+1
        a[i,j] = a[j,i] = True
    permutation = rng.permutation(n); a = a[permutation][:,permutation]
    src,dst = np.nonzero(a)
    graph = adjacency((src,dst),n); b = gauge(graph)
    p = graph.toarray(); p /= p.sum(1,keepdims=True)
    signal = np.linalg.solve(np.eye(n)-.92*p, rng.normal(size=n))
    signal = (signal-signal.mean())/signal.std()
    orientation = rng.choice([-1.,1.],size=f); scale = np.exp(rng.normal(0,.25,f))
    x = ((signal[:,None]+.15*rng.normal(size=(n,f)))*orientation*scale).astype(np.float32)
    class_swap = int(rng.integers(2))
    return {'x':x,'signal':signal,'b':b,'src':src.astype(np.int64),'dst':dst.astype(np.int64),'class_swap':class_swap,'seed':seed}


def make_inputs(x, observed, src, dst, context, labels, query, forward_seed):
    value = C.observed_scale(x, observed)
    poison = x.copy(); poison[~observed] = 1e12
    assert np.array_equal(value, C.observed_scale(poison, observed))
    graph = adjacency((src,dst),len(x))
    filled = impute(graph,np.where(observed,value,np.nan),'pcfi','structural')[0].clip(-10,10)
    filled = np.where(observed,value,filled)
    return C.Inputs(value,observed,filled,src,dst,context,labels,query,2,forward_seed)


def signed_fill(inp, feature_law):
    if feature_law == 0: return inp.pcfi.copy()
    graph = adjacency((inp.src,inp.dst),len(inp.observed)); b = gauge(graph)
    canonical = np.where(inp.observed, inp.observed_values*b[:,None], np.nan)
    filled = impute(graph,canonical,'pcfi','structural')[0]*b[:,None]
    return np.where(inp.observed,inp.observed_values,filled.clip(-10,10)).astype(np.float32)


def lp(inp, signed=False):
    graph = adjacency((inp.src,inp.dst),len(inp.observed)); p = sp.diags(1/np.asarray(graph.sum(1)).ravel())@graph
    b = gauge(graph) if signed else np.ones(len(inp.observed))
    field = np.zeros(len(b)); seed = (2*inp.labels_context-1)*b[inp.context]
    field[inp.context] = seed
    for _ in range(100): field = p@field; field[inp.context] = seed
    probability = np.clip((1+field[inp.query]*b[inp.query])/2,0,1)
    return np.column_stack((1-probability,probability))


def read(row):
    with np.load(ROOT/'bank'/row['path']) as z:
        inp = C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
        return inp,z['labels_query'].astype(np.int64),z['signed_fill']


def prepare_graph(split,gid):
    c = cfg(); seed = c['dev_seed_start' if split=='dev' else 'test_seed_start']+gid
    m = master(seed); graph = adjacency((m['src'],m['dst']),c['nodes']); edge = graph.copy(); edge.setdiag(0); edge.eliminate_zeros()
    records = []; rawpath = ROOT/'bank/raw'/split/f'g{gid}.npz';rawpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(rawpath,**m)
    for t in [0,1]:
        target = m['signal']*(m['b'] if t else 1)
        y = (target>np.median(target)).astype(np.int64)^m['class_swap']
        rng = np.random.default_rng([seed,t,811])
        anchors = {cl:rng.permutation(np.flatnonzero(y==cl)) for cl in [0,1]}
        homophily = float((y[m['src']]==y[m['dst']]).mean())
        for k in c['context_per_class']:
            context = np.concatenate([anchors[cl][:k] for cl in [0,1]])
            rest = np.setdiff1d(np.arange(c['nodes']),context)
            query = np.random.default_rng([seed,t,k,912]).permutation(rest)[:c['queries']]
            sources_order = np.random.default_rng([seed,t,k,913]).permutation(rest)
            for budget in c['observation_budgets']:
                observed = np.zeros((c['nodes'],c['features']),bool)
                if budget=='full': observed[:]=True
                else: observed[sources_order[:budget]]=True
                for g in [0,1]:
                    x = m['x']*(m['b'][:,None] if g else 1)
                    inp = make_inputs(x,observed,m['src'],m['dst'],context,y[context],query,seed*10+t*100+k)
                    signed = signed_fill(inp,g)
                    assert np.array_equal(signed[observed],inp.observed_values[observed])
                    count=observed.sum(0);safe=np.where(observed,x,0).astype(float)
                    rms=np.sqrt(np.divide((safe*safe).sum(0),count,out=np.ones(c['features']),where=count>0))
                    rms=np.where(rms>=1e-8,rms,1.)
                    truth_scaled=np.clip(x/rms,-10,10)
                    miss=~observed[query].any(1)
                    path=ROOT/'bank'/split/f'g{gid}_t{t}_k{k}_o{budget}_f{g}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**vars(inp),labels_query=y[query],signed_fill=signed)
                    dif = x[m['src']]-x[m['dst']]
                    energy=float((dif*dif).sum()/max(float((x*x).sum())*len(m['src'])/len(x),1e-20))
                    records.append({'graph':gid,'seed':seed,'split':split,'label_law':t,'feature_law':g,'k':k,'sources':budget,
                        'path':path.relative_to(ROOT/'bank').as_posix(),'sha256':sha(path),'label_homophily':homophily,'feature_energy':energy,
                        'missing_queries':int(miss.sum()),'pcfi_mse':float(np.mean((inp.pcfi[query[miss]]-truth_scaled[query[miss]])**2)) if miss.any() else None,
                        'signed_mse':float(np.mean((signed[query[miss]]-truth_scaled[query[miss]])**2)) if miss.any() else None})
    return records,{'graph':gid,'path':rawpath.relative_to(ROOT/'bank').as_posix(),'sha256':sha(rawpath)}


def identity_checks(records):
    for row in records:
        other=next(r for r in records if all(r[k]==row[k] for k in ['graph','label_law','k','sources']) and r['feature_law']==1-row['feature_law'])
        inp,y,_=read(row);paired,yp,_=read(other)
        for key in ['src','dst','context','labels_context','query','observed']: assert np.array_equal(getattr(inp,key),getattr(paired,key))
        assert np.array_equal(y,yp)
        if row['sources']==0:
            for key in ['observed_values','pcfi']: assert np.array_equal(getattr(inp,key),getattr(paired,key))


def preflight():
    assert not (ROOT/'results/design.json').exists(); C.setup();torch.set_num_threads(2)
    assert not (ROOT/'bank').exists()
    dev=[];raw=[]
    for i in range(cfg()['dev_graphs']):
        r,rr=prepare_graph('dev',i);dev+=r;raw.append(rr)
    dump(ROOT/'bank/development.json',{'episodes':dev,'raw':raw})
    identity_checks(dev)
    # Scalar positive-only counterexample, independently of node classification.
    src=np.arange(7);dst=src+1;graph=adjacency((np.r_[src,dst],np.r_[dst,src]),8)
    values=np.full((8,1),np.nan,np.float32);values[0]=1
    out=impute(graph,values,'pcfi','structural')[0]
    b=gauge(graph);signed=impute(graph,values*b[:,None],'pcfi','structural')[0]*b[:,None]
    assert (out>=0).all() and signed[1,0]<0
    oldbank=json.loads((OLD/'bank/manifest.json').read_text())['episodes'];oldrows=[r for r in oldbank if r['split']=='test']
    replay=[];inits={};schedules={};checks=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            info=json.loads((OLD/'results'/f'{arm}_s{seed}'/'training.json').read_text())
            assert schedules.setdefault(seed,info['schedule_sha256'])==info['schedule_sha256']
            model=load(arm,seed)
            for i in [1,82,153]:
                row=oldrows[i];inp,y=C.read_episode(row)
                path=OLD/'results/predictions'/f'{arm}_s{seed}_g{row["graph"]}_e{row["setting"]}_standard.npz'
                with np.load(path) as z: error=float(abs(C.predict(model,inp)-z['p']).max())
                assert error<.004;replay.append({'arm':arm,'seed':seed,'old_episode':i,'max_abs_error':error})
            if seed==11:
                row=next(r for r in dev if r['graph']==0 and r['label_law']==1 and r['feature_law']==1 and r['k']==4 and r['sources']==8)
                inp,y,_=read(row);prob=C.predict(model,inp)
                repeat=C.predict(model,copy.deepcopy(inp));assert np.array_equal(prob,repeat)
                order=np.random.default_rng(39).permutation(len(inp.observed));inv=np.argsort(order);changed=copy.deepcopy(inp)
                for key in ['observed_values','observed','pcfi']:setattr(changed,key,getattr(inp,key)[order])
                changed.src=inv[inp.src];changed.dst=inv[inp.dst];changed.context=inv[inp.context];changed.query=inv[inp.query]
                error=float(abs(C.predict(model,changed)-prob).max());assert error<.006
                checks.append({'arm':arm,'node_permutation_max_error':error})
            del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/preflight.json',{'paired_factorial_inputs_valid':True,'hidden_value_poison_invariance':True,'old_replays':replay,'model_checks':checks,
        'scalar_counterexample':{'positive_output':out.ravel().tolist(),'signed_output':signed.ravel().tolist()},'new_updates':0})
    dump(ROOT/'results/design.json',{'sources':sources(),'checkpoints':{str(p.relative_to(WORK)):sha(p) for p in files()},
        'development_sha256':sha(ROOT/'bank/development.json'),'preflight_sha256':sha(ROOT/'results/preflight.json'),
        'locked_before_test_generation':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    lock();print('PREFLIGHT_LOCKED',flush=True)


def lock():
    d=json.loads((ROOT/'results/design.json').read_text(encoding='utf8'))
    assert d['sources']==sources()
    assert d['checkpoints']=={str(p.relative_to(WORK)):sha(p) for p in files()}
    assert d['development_sha256']==sha(ROOT/'bank/development.json')
    return d


def generate():
    lock();assert not (ROOT/'bank/manifest.json').exists();assert not (ROOT/'bank/test').exists()
    records=[];raw=[];begun=time.perf_counter()
    for gid in range(cfg()['graphs']):
        r,rr=prepare_graph('test',gid);records+=r;raw.append(rr)
        if (gid+1)%8==0:print('BANK',gid+1,round(time.perf_counter()-begun,1),flush=True)
    identity_checks(records)
    dump(ROOT/'bank/manifest.json',{'episodes':records,'raw':raw,'seconds':time.perf_counter()-begun,'design_sha256':sha(ROOT/'results/design.json')})
    dump(ROOT/'results/bank_lock.json',{'manifest_sha256':sha(ROOT/'bank/manifest.json'),'design_sha256':sha(ROOT/'results/design.json')})
    print('BANK_LOCKED',len(records),flush=True)


def rows():
    lock();m=ROOT/'bank/manifest.json'
    expected={'manifest_sha256':sha(m),'design_sha256':sha(ROOT/'results/design.json')}
    assert json.loads((ROOT/'results/bank_lock.json').read_text())==expected
    return json.loads(m.read_text())['episodes']


def evaluate(seed):
    C.setup();torch.set_num_threads(2);test=rows()
    for arm in cfg()['arms']:
        folder=ROOT/'results'/f'{arm}_s{seed}';assert not folder.exists();(folder/'predictions').mkdir(parents=True)
        model=load(arm,seed);before=C.parameter_digest(model);records=[];begun=time.perf_counter()
        for i,row in enumerate(test):
            inp,y,signed=read(row);assert sha(ROOT/'bank'/row['path'])==row['sha256']
            ordinary=C.predict(model,inp);variants={'ordinary':ordinary}
            if arm=='pcfi_plain':
                if np.array_equal(signed,inp.pcfi):variants['known_signed']=ordinary.copy()
                else:
                    changed=copy.deepcopy(inp);changed.pcfi=signed
                    variants['known_signed']=C.predict(model,changed)
            missing=~inp.observed[inp.query].any(1)
            for variant,p in variants.items():
                path=folder/'predictions'/f'e{i}_{variant}.npz';np.savez_compressed(path,p=p,y=y)
                records.append({**row,'arm':arm,'seed_model':seed,'variant':variant,'episode':i,**C.metrics(p,y),
                    'missing_metrics':C.metrics(p[missing],y[missing]) if missing.any() else None,
                    'prediction':path.relative_to(ROOT).as_posix(),'prediction_sha256':sha(path)})
            if (i+1)%128==0:print('EVAL',arm,seed,i+1,round(time.perf_counter()-begun,1),flush=True)
        after=C.parameter_digest(model);assert before==after
        dump(folder/'evaluation.json',records);dump(folder/'evaluation_info.json',{'before':before,'after':after,'new_updates':0,'seconds':time.perf_counter()-begun})
        del model;gc.collect();torch.cuda.empty_cache()
    lock()


def controls():
    test=rows();records=[]
    for i,row in enumerate(test):
        inp,y,_=read(row);missing=~inp.observed[inp.query].any(1)
        variants={'positive_lp':lp(inp),'signed_lp':lp(inp,True),'uniform':np.full((len(y),2),.5)}
        for name,p in variants.items():records.append({**row,'variant':name,'episode':i,**C.metrics(p,y),'missing_metrics':C.metrics(p[missing],y[missing]) if missing.any() else None})
    dump(ROOT/'results/controls.json',records);print('CONTROLS_DONE',flush=True)


def difference(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(202611022330)
    boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
    return {m:{'delta':float(d[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,m in enumerate(METRICS)}


def collect():
    data={}
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            records=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'evaluation.json').read_text())
            for r in records:
                name='mask' if arm=='mask_plain' else ('pcfi' if r['variant']=='ordinary' else 'known_signed')
                data.setdefault(name,[]).append(r)
    for r in json.loads((ROOT/'results/controls.json').read_text()):data.setdefault(r['variant'],[]).append(r)
    return data


def summarize():
    test=rows();data=collect();output={};vectors={}
    for t in [0,1]:
        for g in [0,1]:
            for k in cfg()['context_per_class']:
                for budget in cfg()['observation_budgets']:
                    key=f't{t}_g{g}_k{k}_o{budget}';v={};perseed={}
                    for method,records in data.items():
                        a=[]
                        for gid in range(cfg()['graphs']):
                            rec=[r for r in records if r['graph']==gid and r['label_law']==t and r['feature_law']==g and r['k']==k and r['sources']==budget]
                            selected=[r if budget=='full' else r['missing_metrics'] for r in rec];assert all(r is not None for r in selected)
                            a.append(np.mean([[r[m] for m in METRICS] for r in selected],axis=0))
                        v[method]=np.asarray(a)
                        if method in ['mask','pcfi','known_signed']:
                            perseed[method]={}
                            for seed in cfg()['seeds']:
                                rec=[r for r in records if r['seed_model']==seed and r['label_law']==t and r['feature_law']==g and r['k']==k and r['sources']==budget]
                                selected=[r if budget=='full' else r['missing_metrics'] for r in rec]
                                perseed[method][str(seed)]={m:float(np.mean([r[m] for r in selected])) for m in METRICS}
                    base=[r for r in test if r['label_law']==t and r['feature_law']==g and r['k']==k and r['sources']==budget]
                    output[key]={'means':{a:dict(zip(METRICS,x.mean(0).tolist())) for a,x in v.items()},'per_seed':perseed,
                        'mean_label_homophily':float(np.mean([r['label_homophily'] for r in base])),
                        'mean_feature_energy':float(np.mean([r['feature_energy'] for r in base])),
                        'mean_pcfi_mse':float(np.mean([r['pcfi_mse'] for r in base])) if budget!='full' else None,
                        'mean_signed_mse':float(np.mean([r['signed_mse'] for r in base])) if budget!='full' else None,
                        'contrasts':{'known_signed-'+a:difference(v['known_signed'],v[a]) for a in ['pcfi','mask','positive_lp','signed_lp']}}
                    vectors[key]=v
    primary='t1_g1_k4_o8';v=vectors[primary];c=output[primary]['contrasts']
    nofeature=difference(v['known_signed'],vectors['t1_g1_k4_o0']['pcfi'])
    full=difference(vectors['t1_g1_k4_ofull']['pcfi'],v['pcfi'])
    did=difference(v['pcfi']-v['mask'],vectors['t1_g0_k4_o8']['pcfi']-vectors['t1_g0_k4_o8']['mask'])
    gate=c['known_signed-pcfi']['accuracy']['ci95'][0]>0
    floor=all(d['accuracy']['ci95'][0]>0 for d in c.values()) and nofeature['accuracy']['ci95'][0]>0
    dump(ROOT/'results/statistics.json',{'cells':output,'primary_cell':primary,'primary':output[primary],
        'signed_minus_zero_source':nofeature,'pcfi_full_minus_missing':full,'feature_flip_difference_in_differences':did,
        'primary_headroom':bool(gate),'strong_floor':bool(floor),'new_training_updates':0,
        'interpretation':'Privileged known-law diagnostic only. Frozen out-of-prior models, not a new learned method or proof of architecture failure.'})
    print('SUMMARY',json.dumps(output[primary]),'HEADROOM',bool(gate),'STRONG_FLOOR',bool(floor),flush=True)


def audit():
    C.setup();torch.set_num_threads(2);test=rows();data=collect();identity_checks(test)
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());fingerprints=[]
    for row in test+manifest['raw']:assert sha(ROOT/'bank'/row['path'])==row['sha256']
    for r in manifest['raw']:
        with np.load(ROOT/'bank'/r['path']) as z:
            h=hashlib.sha256()
            for key in ['x','signal','src','dst']:h.update(z[key].tobytes())
            fingerprints.append(h.hexdigest())
    assert len(set(fingerprints))==cfg()['graphs']
    count=0;replay=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            folder=ROOT/'results'/f'{arm}_s{seed}';info=json.loads((folder/'evaluation_info.json').read_text());assert info['before']==info['after']
            records=json.loads((folder/'evaluation.json').read_text());model=load(arm,seed)
            for r in records:
                path=ROOT/r['prediction'];assert sha(path)==r['prediction_sha256']
                inp,y,signed=read(test[r['episode']])
                if r['variant']=='known_signed':inp.pcfi=signed
                with np.load(path) as z:p=z['p'];assert np.array_equal(y,z['y'])
                assert np.isfinite(p).all() and (p>=0).all() and np.max(abs(p.sum(1)-1))<1e-5
                score=C.metrics(p,y);assert all(abs(score[m]-r[m])<1e-12 for m in METRICS)
                miss=~inp.observed[inp.query].any(1)
                if miss.any():
                    score=C.metrics(p[miss],y[miss]);assert all(abs(score[m]-r['missing_metrics'][m])<1e-12 for m in METRICS)
                if r['episode'] in [37,515,999] and r['variant']=='ordinary':
                    error=float(abs(C.predict(model,inp)-p).max());assert error<.004
                    replay.append({'arm':arm,'seed':seed,'episode':r['episode'],'max_abs_error':error})
                count+=1
            del model;gc.collect();torch.cuda.empty_cache()
    # All no-source pairs are bit-identical in model input and output.
    for method,records in data.items():
        if method not in ['mask','pcfi','known_signed']:continue
        for r in records:
            if r['sources']!=0 or r['feature_law']!=0:continue
            other=next(q for q in records if all(q[k]==r[k] for k in ['graph','label_law','k','sources','seed_model']) and q['feature_law']==1)
            with np.load(ROOT/r['prediction']) as z:pa=z['p']
            with np.load(ROOT/other['prediction']) as z:pb=z['p']
            assert np.array_equal(pa,pb)
    lock();dump(ROOT/'results/audit.json',{'all_hashes_valid':True,'paired_factorial_inputs_valid':True,'zero_source_output_invariance':True,
        'predictions_rescored':count,'replays':replay,'new_updates':0,'new_models':0,'reused_checkpoints':6,
        'test_base_graphs':len(fingerprints),'test_episodes':len(test),'primary_unit':'32 paired base graphs,3 fixed existing training seeds'})
    print('AUDIT_PASSED',count,flush=True)


def cost():
    C.setup();torch.set_num_threads(2);test=rows();row=next(r for r in test if r['graph']==0 and r['label_law']==1 and r['feature_law']==1 and r['k']==4 and r['sources']==8)
    inp,_,_=read(row);measured={}
    for method in ['mask','pcfi','known_signed']:
        model=load('mask_plain' if method=='mask' else 'pcfi_plain',11)
        def execute():
            changed=copy.deepcopy(inp)
            if method!='mask':
                graph=adjacency((inp.src,inp.dst),len(inp.observed))
                if method=='pcfi':changed.pcfi=impute(graph,np.where(inp.observed,inp.observed_values,np.nan),'pcfi','structural')[0].clip(-10,10)
                else:changed.pcfi=signed_fill(inp,1)
            return C.predict(model,changed)
        for _ in range(5):execute()
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            torch.cuda.synchronize();started=time.perf_counter();execute();torch.cuda.synchronize();times.append((time.perf_counter()-started)*1000)
        measured[method]={'median_ms':float(np.median(times)),'times_ms':times,'additional_peak_mib':(torch.cuda.max_memory_allocated()-base)/2**20}
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/cost.json',{'methods':measured,'scope':cfg()['cost']});print('COST_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','generate','evaluate','controls','summary','audit','cost']);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.action=='evaluate':evaluate(a.seed)
    else:{'preflight':preflight,'generate':generate,'controls':controls,'summary':summarize,'audit':audit,'cost':cost}[a.action]()

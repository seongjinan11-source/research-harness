"""Unknown-law adaptive transport around six fixed public GraphPFN checkpoints."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(WORK/'sign_transport_probe'))
import probe as P
import working_model as W
C,dump,sha=P.C,P.dump,P.sha
METRICS=['accuracy','nll','brier']
PCFI_METHODS=['identity','parity','uniform_two','feature_adapt','joint_adapt','known_g_reference']
CONTROL_METHODS=['positive_lp','signed_lp','adaptive_lp','gaussian_joint','gaussian_label_only','known_t_lp_reference','uniform']


def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def compact(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf8')
def sources():
    files=list(ROOT.glob('*.py'))+[ROOT/'protocol.json']+[WORK/p for p in P.sources()]
    return {str(p.relative_to(WORK)):sha(p) for p in sorted(set(files))}
def lock():
    d=json.loads((ROOT/'results/design.json').read_text(encoding='utf8'))
    assert d['sources']==sources()
    assert d['checkpoints']=={str(p.relative_to(WORK)):sha(p) for p in P.files()}
    assert d['development']==sha(ROOT/'bank/development.json')
    P.lock()
    return d


def read(row):
    with np.load(ROOT/'bank'/row['path']) as z:
        inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
        return inp,z['labels_query'].astype(np.int64),z['parity_fill']


def prepare_graph(split,gid):
    seed=cfg()['dev_seed_start' if split=='dev' else 'test_seed_start']+gid
    master=P.master(seed);records=[]
    path=ROOT/'bank/raw'/split/f'g{gid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**master)
    for t in [0,1]:
        target=master['signal']*(master['b'] if t else 1)
        y=(target>np.median(target)).astype(np.int64)^master['class_swap']
        rng=np.random.default_rng([seed,t,811]);anchors={cl:rng.permutation(np.flatnonzero(y==cl)) for cl in [0,1]}
        homophily=float((y[master['src']]==y[master['dst']]).mean())
        for k in cfg()['context_per_class']:
            context=np.concatenate([anchors[cl][:k] for cl in [0,1]])
            rest=np.setdiff1d(np.arange(cfg()['nodes']),context)
            query=np.random.default_rng([seed,t,k,912]).permutation(rest)[:cfg()['queries']]
            order=np.random.default_rng([seed,t,k,913]).permutation(rest)
            for budget in cfg()['observation_budgets']:
                observed=np.zeros((cfg()['nodes'],cfg()['features']),bool)
                if budget=='full':observed[:]=True
                else:observed[order[:budget]]=True
                for g in [0,1]:
                    x=master['x']*(master['b'][:,None] if g else 1)
                    inp=P.make_inputs(x,observed,master['src'],master['dst'],context,y[context],query,seed*10+t*100+k)
                    parity=P.signed_fill(inp,1) # Both candidates computed for every task, never from true g.
                    assert np.array_equal(parity[observed],inp.observed_values[observed])
                    ep=ROOT/'bank'/split/f'g{gid}_t{t}_k{k}_o{budget}_f{g}.npz';ep.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(ep,**vars(inp),labels_query=y[query],parity_fill=parity)
                    records.append({'graph':gid,'split':split,'label_law':t,'feature_law':g,'k':k,'sources':budget,
                        'seed':seed,'path':ep.relative_to(ROOT/'bank').as_posix(),'sha256':sha(ep),
                        'missing_queries':int((~observed[query].any(1)).sum()),'label_homophily':homophily})
    return records,{'graph':gid,'path':path.relative_to(ROOT/'bank').as_posix(),'sha256':sha(path)}


def analytic_checks():
    u=W.sobol(1)
    mean=np.zeros(2);cov=np.array([[1.,.6],[.6,1.]])
    logz,prob,ess=W.orthant_predict(mean,cov,np.array([0]),np.array([1.]),np.array([1]),np.array([1.]),u)
    expected=.5+np.arcsin(.6)/np.pi
    error=abs(float(prob[0])-expected)
    assert abs(logz-np.log(.5))<1e-12 and error<.002
    total=0.
    for a in [-1.,1.]:
        for b in [-1.,1.]:
            l,_,_=W.orthant_predict(np.zeros(3),np.eye(3),np.array([0,1]),np.array([a,b]),np.array([2]),np.array([1.]),W.sobol(2))
            total+=np.exp(l)
    assert abs(total-1)<1e-12
    return {'one_context_conditional_probability_error':error,'two_independent_context_probability_sum_error':abs(total-1),'particles':2048}


def input_checks(records):
    index={(r['graph'],r['label_law'],r['feature_law'],r['k'],str(r['sources'])):r for r in records}
    for r in records:
        q=index[(r['graph'],r['label_law'],1-r['feature_law'],r['k'],str(r['sources']))]
        a,y,_=read(r);b,z,_=read(q)
        for k in ['src','dst','context','labels_context','query','observed']:assert np.array_equal(getattr(a,k),getattr(b,k))
        assert np.array_equal(y,z)
    return True


def preflight():
    assert not (ROOT/'bank').exists();C.setup();torch.set_num_threads(2)
    checks=analytic_checks();dev=[];raw=[]
    for gid in range(cfg()['dev_graphs']):
        r,rr=prepare_graph('dev',gid);dev+=r;raw.append(rr)
    dump(ROOT/'bank/development.json',{'episodes':dev,'raw':raw})
    input_checks(dev);invariance=[];resolution=[]
    for gid in range(cfg()['dev_graphs']):
        selected=[r for r in dev if r['graph']==gid and r['label_law']==1 and r['feature_law']==1 and r['k']==4]
        for row in selected:
            inp,_,_=read(row);graph=P.adjacency((inp.src,inp.dst),len(inp.observed));k=W.kernel(graph);b=P.gauge(graph)
            result=W.infer(inp,k,b)
            fg=W.feature_infer(inp,k,b);assert np.max(abs(fg-result['feature_g']))<1e-12
            blind=copy.deepcopy(inp);blind.labels_context=1-blind.labels_context
            assert np.array_equal(fg,W.feature_infer(blind,k,b))
            if row['sources'] in [0,1]:assert np.max(abs(result['joint_g']-.5))<1e-8
            renamed=copy.deepcopy(inp);renamed.labels_context=1-renamed.labels_context
            other=W.infer(renamed,k,b)
            class_error=float(abs(result['probability']-other['probability'][:,::-1]).max())
            assert class_error<1e-8
            other=W.infer(inp,k,-b)
            gauge_error=float(abs(result['probability']-other['probability']).max());assert gauge_error<1e-8
            if row['sources']==8:
                finer=W.infer(inp,k,b,bits=11)
                error=float(abs(result['probability']-finer['probability']).max());assert error<.03
                resolution.append({'graph':gid,'2048_vs4096_probability_max_error':error})
                order=np.random.default_rng(771).permutation(len(inp.observed));inv=np.argsort(order);renamed=copy.deepcopy(inp)
                for key in ['observed_values','observed','pcfi']:setattr(renamed,key,getattr(inp,key)[order])
                renamed.src=inv[inp.src];renamed.dst=inv[inp.dst];renamed.context=inv[inp.context];renamed.query=inv[inp.query]
                kg=k[order][:,order];other=W.infer(renamed,kg,b[order])
                node_error=float(abs(result['probability']-other['probability']).max());assert node_error<.03
                invariance.append({'graph':gid,'class_error':class_error,'gauge_error':gauge_error,'node_permutation_error':node_error})
    # Reuse original frozen loader and replay one old sign-probe prediction per checkpoint.
    replay=[];oldrows=P.rows()
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            model=P.load(arm,seed);r=oldrows[37];inp,_,_=P.read(r)
            name='ordinary';path=P.ROOT/'results'/f'{arm}_s{seed}'/'predictions'/f'e37_{name}.npz'
            with np.load(path) as z:error=float(abs(C.predict(model,inp)-z['p']).max())
            assert error<.004;replay.append({'arm':arm,'seed':seed,'max_abs_error':error})
            del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/preflight.json',{'analytic':checks,'invariance':invariance,'resolution':resolution,'old_prediction_replays':replay,
        'law_or_query_truth_not_an_inference_argument':True,'new_updates':0})
    dump(ROOT/'results/design.json',{'sources':sources(),'checkpoints':{str(p.relative_to(WORK)):sha(p) for p in P.files()},
        'development':sha(ROOT/'bank/development.json'),'preflight':sha(ROOT/'results/preflight.json'),
        'locked_before_test_generation':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    lock();print('PREFLIGHT_LOCKED',flush=True)


def generate():
    lock();assert not (ROOT/'bank/test').exists();records=[];raw=[];started=time.perf_counter()
    for gid in range(cfg()['graphs']):
        r,rr=prepare_graph('test',gid);records+=r;raw.append(rr)
        if (gid+1)%8==0:print('BANK',gid+1,round(time.perf_counter()-started,1),flush=True)
    input_checks(records)
    dump(ROOT/'bank/manifest.json',{'episodes':records,'raw':raw,'seconds':time.perf_counter()-started,'design':sha(ROOT/'results/design.json')})
    dump(ROOT/'results/bank_lock.json',{'manifest':sha(ROOT/'bank/manifest.json'),'design':sha(ROOT/'results/design.json')})
    print('BANK_LOCKED',len(records),flush=True)


def rows():
    lock();assert json.loads((ROOT/'results/bank_lock.json').read_text())=={'manifest':sha(ROOT/'bank/manifest.json'),'design':sha(ROOT/'results/design.json')}
    return json.loads((ROOT/'bank/manifest.json').read_text())['episodes']


def inference(part):
    test=rows();folder=ROOT/'results/inference';folder.mkdir(parents=True,exist_ok=True);records=[]
    kernels={};label_cache={};started=time.perf_counter()
    for i,row in enumerate(test):
        if i%3!=part:continue
        inp,y,_=read(row);gid=row['graph']
        if gid not in kernels:
            graph=P.adjacency((inp.src,inp.dst),len(inp.observed));kernels[gid]=(W.kernel(graph),P.gauge(graph))
        k,b=kernels[gid]
        result=W.infer(inp,k,b)
        key=(gid,tuple(inp.context.tolist()),tuple(inp.labels_context.tolist()),tuple(inp.query.tolist()))
        if key not in label_cache:label_cache[key]=W.infer(inp,k,b,without_features=True)
        label=label_cache[key]
        if row['sources'] in [0,1]:assert np.max(abs(result['joint_g']-.5))<1e-8
        ppos=P.lp(inp);pneg=P.lp(inp,True)
        # True t is used ONLY to form the explicitly privileged reference below.
        probabilities=np.stack((ppos,pneg,label['joint_t'][0]*ppos+label['joint_t'][1]*pneg,
            result['probability'],label['probability'],pneg if row['label_law'] else ppos,np.full_like(ppos,.5)))
        path=folder/f'e{i}.npz'
        np.savez_compressed(path,probabilities=probabilities,joint_g=result['joint_g'],feature_g=result['feature_g'],
            joint_t=result['joint_t'],label_t=label['joint_t'],joint_weights=result['joint_weights'],
            weighted_ess=result['weighted_ess'],min_branch_ess=result['min_branch_ess'])
        records.append({'episode':i,'input_sha256':row['sha256'],'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
        if len(records)%128==0:print('INFER',part,len(records),round(time.perf_counter()-started,1),flush=True)
    compact(ROOT/'results'/f'inference_part{part}.json',records)
    print('INFERRED',part,len(records),flush=True)


def inference_index():
    rowsout=[]
    for part in range(3):rowsout+=json.loads((ROOT/'results'/f'inference_part{part}.json').read_text())
    out={r['episode']:r for r in rowsout};assert len(out)==len(rows())
    return out


def evaluate(seed):
    C.setup();torch.set_num_threads(2);test=rows();index=inference_index()
    for arm in cfg()['arms']:
        folder=ROOT/'results'/f'{arm}_s{seed}';assert not folder.exists();(folder/'predictions').mkdir(parents=True)
        model=P.load(arm,seed);before=C.parameter_digest(model);records=[];started=time.perf_counter()
        for i,row in enumerate(test):
            inp,y,parity=read(row);inferrow=index[i]
            assert inferrow['input_sha256']==row['sha256'] and sha(ROOT/inferrow['path'])==inferrow['sha256']
            p0=C.predict(model,inp)
            if arm=='mask_plain':names=['mask'];pred=np.stack([p0])
            else:
                if np.array_equal(inp.pcfi,parity):p1=p0.copy()
                else:
                    changed=copy.deepcopy(inp);changed.pcfi=parity;p1=C.predict(model,changed)
                with np.load(ROOT/inferrow['path']) as z:fg=z['feature_g'];jg=z['joint_g']
                pred=np.stack((p0,p1,.5*(p0+p1),fg[0]*p0+fg[1]*p1,jg[0]*p0+jg[1]*p1,p1 if row['feature_law'] else p0))
                names=PCFI_METHODS
            missing=~inp.observed[inp.query].any(1)
            path=folder/'predictions'/f'e{i}.npz';np.savez_compressed(path,p=pred,y=y)
            metrics={name:{'all':C.metrics(pred[j],y),'missing':C.metrics(pred[j,missing],y[missing]) if missing.any() else None} for j,name in enumerate(names)}
            records.append({'episode':i,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'metrics':metrics})
            if (i+1)%256==0:print('EVAL',arm,seed,i+1,round(time.perf_counter()-started,1),flush=True)
        after=C.parameter_digest(model);assert before==after
        compact(folder/'evaluation.json',records);dump(folder/'evaluation_info.json',{'before':before,'after':after,'new_updates':0,'seconds':time.perf_counter()-started})
        del model;gc.collect();torch.cuda.empty_cache()
    lock()


def difference(a,b):
    d=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(202611030101)
    boot=d[rng.integers(len(d),size=(5000,len(d)))].mean(1)
    return {m:{'delta':float(d[:,j].mean()),'ci95':np.quantile(boot[:,j],[.025,.975]).tolist()} for j,m in enumerate(METRICS)}


def collect():
    test=rows();index=inference_index();data={};lookup={}
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            records=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'evaluation.json').read_text());assert len(records)==len(test)
            for r in records:
                i=r['episode']
                for name,m in r['metrics'].items():data.setdefault(name,[]).append({**test[i],'seed_model':seed,'episode':i,**m})
                lookup[(arm,seed,i)]=r
    for i,row in enumerate(test):
        with np.load(ROOT/index[i]['path']) as z:prob=z['probabilities']
        inp,y,_=read(row);missing=~inp.observed[inp.query].any(1)
        for j,name in enumerate(CONTROL_METHODS):data.setdefault(name,[]).append({**row,'episode':i,'all':C.metrics(prob[j],y),
            'missing':C.metrics(prob[j,missing],y[missing]) if missing.any() else None})
    return test,index,data,lookup


def summarize():
    test,index,data,lookup=collect();cells={};vectors={}
    selectors={'primary':lambda r:r['sources']==8 and r['k']==4}
    for t in [0,1]:
        for g in [0,1]:
            for k in cfg()['context_per_class']:
                for budget in cfg()['observation_budgets']:
                    selectors[f't{t}_g{g}_k{k}_o{budget}']=lambda r,t=t,g=g,k=k,o=budget:r['label_law']==t and r['feature_law']==g and r['k']==k and r['sources']==o
    for name,choose in selectors.items():
        v={}
        for method,records in data.items():
            a=[]
            for gid in range(cfg()['graphs']):
                selected=[r for r in records if r['graph']==gid and choose(r)]
                target=[r['all'] if r['sources']=='full' else r['missing'] for r in selected];assert all(r is not None for r in target)
                a.append(np.mean([[r[m] for m in METRICS] for r in target],axis=0))
            v[method]=np.asarray(a)
        cells[name]={'means':{a:dict(zip(METRICS,x.mean(0).tolist())) for a,x in v.items()},
            'contrasts':{'joint_adapt-'+a:difference(v['joint_adapt'],v[a]) for a in v if a!='joint_adapt'}}
        vectors[name]=v
    # Match exact query cohort for source-removal interventions.
    zero_lookup={(r['graph'],r['label_law'],r['feature_law'],r['k']):i for i,r in enumerate(test) if r['sources']==0}
    zero_vectors=[];gpzero_vectors=[]
    for gid in range(cfg()['graphs']):
        episode_metrics=[];gp_metrics=[]
        for i,row in enumerate(test):
            if row['graph']!=gid or row['sources']!=8 or row['k']!=4:continue
            zi=zero_lookup[(gid,row['label_law'],row['feature_law'],row['k'])]
            inp,y,_=read(row);zinp,zy,_=read(test[zi]);assert np.array_equal(inp.query,zinp.query) and np.array_equal(y,zy)
            mask=~inp.observed[inp.query].any(1)
            for seed in cfg()['seeds']:
                rr=lookup[('pcfi_plain',seed,zi)]
                with np.load(ROOT/rr['path']) as z:prob=z['p'][PCFI_METHODS.index('joint_adapt')]
                score=C.metrics(prob[mask],y[mask]);episode_metrics.append([score[m] for m in METRICS])
            with np.load(ROOT/index[zi]['path']) as z:prob=z['probabilities'][CONTROL_METHODS.index('gaussian_joint')]
            score=C.metrics(prob[mask],y[mask]);gp_metrics.append([score[m] for m in METRICS])
        zero_vectors.append(np.mean(episode_metrics,axis=0));gpzero_vectors.append(np.mean(gp_metrics,axis=0))
    contrasts=cells['primary']['contrasts'];primary=all(contrasts['joint_adapt-'+a]['accuracy']['ci95'][0]>0 for a in ['uniform_two','feature_adapt'])
    nofeature=difference(vectors['primary']['joint_adapt'],zero_vectors)
    gpfeature=difference(vectors['primary']['gaussian_joint'],gpzero_vectors)
    floor=primary and all(contrasts['joint_adapt-'+a]['accuracy']['ci95'][0]>0 for a in ['mask','identity','parity','adaptive_lp','gaussian_joint']) and nofeature['accuracy']['ci95'][0]>0
    weights=[]
    for i,row in enumerate(test):
        with np.load(ROOT/index[i]['path']) as z:
            weights.append({**{k:row[k] for k in ['graph','label_law','feature_law','k','sources']},
                'feature_g1':float(z['feature_g'][1]),'joint_g1':float(z['joint_g'][1]),'label_t1':float(z['label_t'][1]),
                'joint_t1':float(z['joint_t'][1]),'weighted_ess':float(z['weighted_ess']),'min_branch_ess':float(z['min_branch_ess'])})
    weight_summary={}
    for k in cfg()['context_per_class']:
        for o in cfg()['observation_budgets']:
            rr=[r for r in weights if r['k']==k and r['sources']==o]
            item={}
            for field,target in [('feature_g1','feature_law'),('joint_g1','feature_law'),('label_t1','label_law'),('joint_t1','label_law')]:
                q=np.array([r[field] for r in rr]);y=np.array([r[target] for r in rr])
                item[field]={'accuracy':float(((q>.5)==y).mean()),'brier':float(((q-y)**2).mean()),
                    'entropy':float(-(q*np.log(np.maximum(q,1e-300))+(1-q)*np.log(np.maximum(1-q,1e-300))).mean()),
                    'max_abs_from_half':float(abs(q-.5).max())}
            item['weighted_ess_mean']=float(np.mean([r['weighted_ess'] for r in rr]));item['min_branch_ess']=float(min(r['min_branch_ess'] for r in rr))
            weight_summary[f'k{k}_o{o}']=item
    dump(ROOT/'results/statistics.json',{'primary':cells['primary'],'cells':cells,'primary_gate':bool(primary),'strong_floor':bool(floor),
        'joint_adapt_minus_zero_same_queries':nofeature,'gaussian_joint_minus_zero_same_queries':gpfeature,
        'zero_source_primary_cohort_means':dict(zip(METRICS,np.mean(zero_vectors,0).tolist())),
        'gaussian_zero_source_primary_cohort_means':dict(zip(METRICS,np.mean(gpzero_vectors,0).tolist())),
        'weight_summary':weight_summary,'new_updates':0,'unit':'64 paired base graphs;3 fixed historical training seeds; all4 laws averaged for primary.'})
    compact(ROOT/'results/weights.json',weights)
    print('SUMMARY',json.dumps(cells['primary']),'PRIMARY',bool(primary),'FLOOR',bool(floor),flush=True)


def audit():
    C.setup();torch.set_num_threads(2);test,index,data,lookup=collect();input_checks(test)
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());newfp=[];oldfp=set()
    def fingerprint(path):
        with np.load(path) as z:
            h=hashlib.sha256()
            for key in ['x','signal','src','dst']:h.update(z[key].tobytes())
            return h.hexdigest()
    for p in (P.ROOT/'bank/raw').glob('*/*.npz'):oldfp.add(fingerprint(p))
    for r in test+manifest['raw']:assert sha(ROOT/'bank'/r['path'])==r['sha256']
    newfp=[fingerprint(ROOT/'bank'/r['path']) for r in manifest['raw']]
    assert len(set(newfp))==cfg()['graphs'] and not oldfp.intersection(newfp)
    inferences=[];count=0;replays=[]
    for i,row in enumerate(test):
        rr=index[i];assert sha(ROOT/rr['path'])==rr['sha256'] and row['sha256']==rr['input_sha256']
        with np.load(ROOT/rr['path']) as z:
            assert np.max(abs(z['joint_weights'].sum()-1))<1e-10
            if row['sources'] in [0,1]:assert np.max(abs(z['joint_g']-.5))<1e-8
        if i in [26,793,1663]:
            inp,_,_=read(row);a=P.adjacency((inp.src,inp.dst),len(inp.observed));r=W.infer(inp,W.kernel(a),P.gauge(a))
            with np.load(ROOT/rr['path']) as z:error=float(abs(r['probability']-z['probabilities'][CONTROL_METHODS.index('gaussian_joint')]).max())
            assert error<1e-10;inferences.append({'episode':i,'replay_probability_error':error})
    for arm in cfg()['arms']:
        names=['mask'] if arm=='mask_plain' else PCFI_METHODS
        for seed in cfg()['seeds']:
            folder=ROOT/'results'/f'{arm}_s{seed}';info=json.loads((folder/'evaluation_info.json').read_text());assert info['before']==info['after'] and info['new_updates']==0
            model=P.load(arm,seed)
            for i,row in enumerate(test):
                rr=lookup[(arm,seed,i)];assert sha(ROOT/rr['path'])==rr['sha256']
                inp,y,parity=read(row);mask=~inp.observed[inp.query].any(1)
                with np.load(ROOT/rr['path']) as z:prob=z['p'];assert np.array_equal(y,z['y'])
                assert np.isfinite(prob).all() and (prob>=0).all() and np.max(abs(prob.sum(-1)-1))<1e-5
                for j,name in enumerate(names):
                    score=C.metrics(prob[j],y);assert all(abs(score[m]-rr['metrics'][name]['all'][m])<1e-12 for m in METRICS)
                    if mask.any():
                        score=C.metrics(prob[j,mask],y[mask]);assert all(abs(score[m]-rr['metrics'][name]['missing'][m])<1e-12 for m in METRICS)
                    count+=1
                if i in [26,793,1663]:
                    error=float(abs(C.predict(model,inp)-prob[0]).max());assert error<.004
                    replays.append({'arm':arm,'seed':seed,'episode':i,'error':error})
            del model;gc.collect();torch.cuda.empty_cache()
    lock();dump(ROOT/'results/audit.json',{'all_hashes_valid':True,'paired_factorial_inputs_valid':True,'old_graph_overlap':0,
        'new_graphs':len(newfp),'new_episodes':len(test),'neural_variant_probabilities_rescored':count,'inference_replays':inferences,
        'neural_replays':replays,'new_updates':0,'new_models':0,'reused_checkpoints':6,'law_metadata_not_in_inference_signature':True})
    print('AUDIT_PASSED',count,flush=True)


def cost():
    C.setup();torch.set_num_threads(2);test=rows();row=next(r for r in test if r['graph']==0 and r['label_law']==1 and r['feature_law']==1 and r['k']==4 and r['sources']==8)
    inp,_,_=read(row);measured={}
    methods=['mask','identity','uniform_two','feature_adapt','joint_adapt','adaptive_lp','gaussian_joint']
    for method in methods:
        model=P.load('mask_plain' if method=='mask' else 'pcfi_plain',11) if method not in ['adaptive_lp','gaussian_joint'] else None
        def execute():
            graph=P.adjacency((inp.src,inp.dst),len(inp.observed))
            if method=='mask':return C.predict(model,inp)
            r=None
            if method=='feature_adapt':
                r={'feature_g':W.feature_infer(inp,W.kernel(graph),P.gauge(graph))}
            if method in ['joint_adapt','adaptive_lp','gaussian_joint']:
                k=W.kernel(graph);b=P.gauge(graph)
                r=W.infer(inp,k,b,without_features=method=='adaptive_lp')
            if method=='gaussian_joint':return r['probability']
            if method=='adaptive_lp':return r['joint_t'][0]*P.lp(inp)+r['joint_t'][1]*P.lp(inp,True)
            x=copy.deepcopy(inp);x.pcfi=P.impute(graph,np.where(inp.observed,inp.observed_values,np.nan),'pcfi','structural')[0].clip(-10,10)
            p0=C.predict(model,x)
            if method=='identity':return p0
            x.pcfi=P.signed_fill(inp,1);p1=C.predict(model,x)
            w=.5 if method=='uniform_two' else float(r['feature_g' if method=='feature_adapt' else 'joint_g'][1])
            return (1-w)*p0+w*p1
        for _ in range(5):execute()
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            torch.cuda.synchronize();start=time.perf_counter();execute();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        measured[method]={'median_ms':float(np.median(times)),'times_ms':times,'additional_peak_mib':(torch.cuda.max_memory_allocated()-base)/2**20}
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/cost.json',{'methods':measured,'scope':cfg()['cost']});print('COST_DONE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','generate','infer','evaluate','summary','audit','cost']);parser.add_argument('--seed',type=int);parser.add_argument('--part',type=int);args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action=='infer':inference(args.part)
        elif args.action=='evaluate':evaluate(args.seed)
        else:{'preflight':preflight,'generate':generate,'summary':summarize,'audit':audit,'cost':cost}[args.action]()

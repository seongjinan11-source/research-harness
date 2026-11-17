"""Run the locked CPU pilot and audit saved arrays before calculating gates."""
import sklearn
import itertools
import json
from collections import defaultdict
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import torch
from model import ARMS, RULES, QUERY_BLOCK, Head, prepare, preproc, probabilities
from data import ROOT, SEEDS, UPDATES, array_hash, episode, lock, real, sha, specs, write


def tensors(pack):
    return tuple(torch.from_numpy(v) for v in pack)


@torch.inference_mode()
def scores(model, pack, k, chunk=QUERY_BLOCK):
    z,g,n=tensors(pack)
    return torch.cat([model(zz,g,n,k) for zz in z.split(chunk)]).softmax(-1).numpy()


def balanced_ce(logits, y):
    counts=torch.bincount(y,minlength=logits.size(-1))
    ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
    return torch.sum(ce/counts[y])/(counts>0).sum()


def train_bank():
    caches={arm:[] for arm in ARMS}; exposures=[]
    t=perf_counter()
    for i,params in enumerate(specs('train')):
        inp,y,meta=episode(*params)
        shared=preproc(inp)
        for arm in ARMS:
            pack,_=prepare(inp,arm,shared)
            caches[arm].append((tensors(pack),torch.from_numpy(y),len(meta['counts'])))
        exposures.append(dict(**meta,inputs=input_hash(inp),truth=array_hash(y)))
        if (i+1)%80==0: print('train-cache',i+1,flush=True)
    return caches,exposures,perf_counter()-t


def fit(cache, arm, seed, digest):
    t0=perf_counter();torch.manual_seed(seed)
    model=Head()
    initial=array_hash(*[v.detach().numpy() for v in model.parameters()])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    order=np.random.default_rng(17986000+seed).permutation(UPDATES)
    steps=[]
    for i in order:
        (p,g,n),y,k=cache[i]
        opt.zero_grad(set_to_none=True)
        loss=balanced_ce(model(p,g,n,k),y)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        assert torch.isfinite(loss)
        steps.append([float(loss.detach()),float(norm)])
    model.eval()
    params={a:b.detach().clone() for a,b in model.state_dict().items()}
    cp=f'results/checkpoints/{arm}-{seed}.pt'
    (ROOT/cp).parent.mkdir(exist_ok=True)
    with (ROOT/cp).open('xb') as f:torch.save(dict(parameters=params, arm=arm,seed=seed,lock=digest),f)
    restored=Head().eval(); restored.load_state_dict(torch.load(ROOT/cp,weights_only=True)['parameters'])
    t, _, k=cache[order[0]]
    with torch.no_grad():assert torch.equal(restored(*t,k),model(*t,k))
    row=dict(arm=arm,seed=seed,checkpoint=cp,sha256=sha(ROOT/cp),initial=initial,
             updates=len(order),order=order.tolist(),optimizer_log=steps,seconds=perf_counter()-t0,
             parameters=sum(p.numel() for p in model.parameters()))
    print('fit',arm,seed,len(order),round(row['seconds'],2),flush=True)
    return model,row


def input_hash(inp):
    return array_hash(inp.x,inp.edge,inp.context,inp.yc,inp.query)


def save_predictions(split, inp, truth, meta, models):
    truth=np.asarray(truth); probs={}; timings={}; k=len(meta['counts'])
    t0=perf_counter(); shared=preproc(inp); shared_seconds=perf_counter()-t0
    t=perf_counter(); probs.update(probabilities(inp,shared)); rules_seconds=perf_counter()-t
    for arm in ARMS:
        pack,tm=prepare(inp,arm,shared)
        tm['preproc']=shared_seconds
        tm['score_by_seed']={}
        for seed in SEEDS:
            t=perf_counter(); probs[f'{arm}_{seed}']=scores(models[arm,seed],pack,k)
            tm['score_by_seed'][str(seed)]=perf_counter()-t
        timings[arm]=tm
    if split=='real' and meta['draw']==0:
        for arm in ARMS:
            def timed():
                t0=perf_counter(); pack,tm=prepare(inp,arm); t1=perf_counter()
                p=scores(models[arm,SEEDS[0]], pack,k)
                return p,dict(**tm,score=perf_counter()-t1,total=perf_counter()-t0)
            timed()  # fixed warm-up; no scores used for selection
            timings[arm]['profile']=[timed()[1] for _ in range(3)]
    path=f"results/predictions/{split}/{meta['id']}.npz"
    (ROOT/path).parent.mkdir(parents=True,exist_ok=True)
    with (ROOT/path).open('xb') as f:
        np.savez_compressed(f,y=truth,q=inp.query,c=inp.context,yc=inp.yc,**probs)
    return dict(**meta,path=path,sha256=sha(ROOT/path),input_hash=input_hash(inp), timings=timings,rules_seconds=rules_seconds)


def metrics(p,y,counts):
    k=len(counts)
    p=np.asarray(p,dtype=np.float64); a=p.argmax(1)
    per=np.array([(a[y==c]==c).mean() for c in range(k)])
    return dict(accuracy=float((a==y).mean()), bacc=float(per.mean()),
                tail=float(per[np.array(counts)<=4].mean()), head=float(per[np.array(counts)>4].mean()),
                nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
                brier=float(np.square(p-np.eye(k)[y]).sum(1).mean()))


def mean_dict(rows):
    return {key:float(np.mean([r[key] for r in rows])) for key in rows[0]}


def audit(split, rows):
    kset=set(RULES)|{f'{a}_{s}' for a,s in itertools.product(ARMS,SEEDS)}
    iterator=(episode(*s) for s in specs(split)) if split!='real' else real()
    checked=[]
    for row,(inp,y,meta) in zip(rows,iterator,strict=True):
        assert row['id']==meta['id'] and row['input_hash']==input_hash(inp) and row['sha256']==sha(ROOT/row['path'])
        assert row['counts']==np.bincount(inp.yc).tolist() and (np.bincount(y)==(8 if split!='real' else 32)).all()
        assert all(row['timings'][arm]['tokens']==row['timings']['tail']['tokens'] for arm in ('proportional','top'))
        assert row['timings']['top']['counts']==row['timings']['tail']['counts']
        assert row['timings']['full']['tokens']==len(inp.context)
        with np.load(ROOT/row['path'],allow_pickle=False) as z:
            assert set(z.files)==kset|{'y','q','c','yc'}
            assert all(np.array_equal(z[key], x) for key,x in zip(('y','q','c','yc'),(y,inp.query,inp.context,inp.yc)))
            results={}
            for a in kset:
                p=z[a]
                assert p.shape==(len(y),len(row['counts'])) and np.all(p>=0) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
                results[a]=metrics(p,y,row['counts'])
            for arm in ARMS:
                results[arm]=mean_dict([results[f'{arm}_{seed}'] for seed in SEEDS])
        checked.append(results)
    return checked


def audit_fits(fits, digest):
    assert {(r['arm'],r['seed']) for r in fits}==set(itertools.product(ARMS,SEEDS)) and len(fits)==12
    for r in fits:
        d=torch.load(ROOT/r['checkpoint'],weights_only=True)
        assert d['lock']==digest and d['arm']==r['arm'] and d['seed']==r['seed']
        assert r['updates']==UPDATES and r['sha256']==sha(ROOT/r['checkpoint'])
        assert r['order']==np.random.default_rng(17986000+r['seed']).permutation(UPDATES).tolist()
        torch.manual_seed(r['seed']);model=Head(); assert r['initial']==array_hash(*[v.detach().numpy() for v in model.parameters()])
        model.load_state_dict(d['parameters'],strict=True)
        assert all(torch.isfinite(v).all() for v in model.parameters())


def contrast(rows, vals, a, b, key, rng_seed, threshold):
    delta=np.array([v[a][key]-v[b][key] for v in vals]); strata=defaultdict(list)
    for i,r in enumerate(rows):strata[tuple(r['stratum'])].append(i)
    assert sorted(map(len,strata.values()))==[12]*8
    rng=np.random.default_rng(rng_seed)
    reps=np.concatenate([rng.choice(i,(5000,len(i)),replace=True) for i in strata.values()],axis=1)
    ci=np.quantile(delta[reps].mean(1),[.0125,.9875]).tolist()
    return dict(diff=float(delta.mean()),ci=ci,pass_gate=ci[0]>threshold)


def analyze(records, fits, digest):
    audit_fits(fits,digest)
    syn=audit('final',records['final']); real_rows=audit('real',records['real'])
    assert len(syn)==96 and len(real_rows)==15
    rows=records['final']; large=[r for r in rows if r['stratum'][0]]
    token_ratio=np.mean([r['timings']['full']['tokens'] for r in large])/np.mean([r['timings']['tail']['tokens'] for r in large])
    averages={a:mean_dict([x[a] for x in syn]) for a in ARMS+RULES}
    gates={
        'tail_vs_proportional':contrast(rows,syn,'tail','proportional','tail',17987100,0.),
        'tail_vs_top':contrast(rows,syn,'tail','top','tail',17987101,0.),
        'overall_vs_full_guard':contrast(rows,syn,'tail','full','bacc',17987102,-.02),
        'holdout_token_ratio':dict(ratio=float(token_ratio),pass_gate=bool(token_ratio>=2)),
        'rule_floor':dict(pass_gate=averages['tail']['tail']>=max(averages[a]['tail'] for a in RULES))}
    real_avg={}
    for g in ('Cora','CiteSeer','PubMed'):
        sub=[x for r,x in zip(records['real'],real_rows) if r['graph']==g]; assert len(sub)==5
        real_avg[g]={a:mean_dict([x[a] for x in sub]) for a in ARMS+RULES}
    report=dict(status='PASS' if all(x['pass_gate'] for x in gates.values()) else 'FAIL',
                synthetic=averages,gates=gates,real_descriptive=real_avg)
    write('results/audit.json',dict(status='PASS',lock=digest,paired_graphs=96,real_draws=15,
          probabilities_recomputed=111*(3*len(ARMS)+len(RULES)),target_optimizer_steps=0))
    write('results/summary.json',report)
    return report


def main():
    t0=perf_counter(); digest=lock(); caches,exposures,prep_seconds=train_bank()
    models={};fits=[]
    for a,s in itertools.product(ARMS, SEEDS):
        model,row=fit(caches[a],a,s,digest); models[a,s]=model; fits.append(row)
    del caches
    write('results/training.json',dict(fits=fits,episodes=exposures,prepare_seconds=prep_seconds))
    records={'final':[],'real':[]}
    for split in records:
        iterator=(episode(*s) for s in specs(split)) if split!='real' else real()
        for i,(inp,y,meta) in enumerate(iterator):
            records[split].append(save_predictions(split,inp,y,meta,models))
            if (i+1)%12==0:print('predictions',split,i+1,flush=True)
    lock()
    write('results/artifact_manifest.json',dict(records=records,checkpoints={r['checkpoint']:r['sha256'] for r in fits},
           lock=digest, seconds_before_audit=perf_counter()-t0))
    print(json.dumps(analyze(records,fits,digest)),flush=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()

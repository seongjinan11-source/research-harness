"""FP05 post-result CPU saved-artifact diagnostic; no model imports/forward.

Creates results/routing_diagnostic.json once. Keeps scientific sources, banks,
locks, checkpoints and predictions unchanged. Training entropy is not a claim
about the unrecorded final conditional policy distribution on evaluation tasks.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'pfn_rl_cycle'
TRACES = tuple(itertools.product((0, 1), repeat=3))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),minimum=float(a.min()),maximum=float(a.max()))


def probabilities(record):
    path=ROOT/record['prediction']
    assert sha(path)==record['prediction_sha256'],path
    with np.load(path,allow_pickle=False) as z:
        p=z['p'].copy(); identity={k:z[k].copy() for k in ('y','query','context','yc')}
    assert np.isfinite(p).all()
    return p,identity


def paired(main,replay):
    assert set(main)==set(replay)
    answer=dict(pairs=len(main),exact_arrays=0,allclose_numpy_default=0,
                allclose_abs1e_7=0,allclose_abs1e_6=0,maximum_abs_probability_difference=0.,
                query_rows=0,argmax_changed_rows=0,mean_record_TV=0.)
    tv=[]
    for key in sorted(main):
        p,identity=probabilities(main[key]);q,other=probabilities(replay[key])
        assert p.shape==q.shape and all(np.array_equal(identity[k],other[k]) for k in identity)
        difference=np.abs(p.astype(float)-q.astype(float))
        answer['exact_arrays']+=int(np.array_equal(p,q))
        answer['allclose_numpy_default']+=int(np.allclose(p,q))
        answer['allclose_abs1e_7']+=int(np.allclose(p,q,atol=1e-7,rtol=0))
        answer['allclose_abs1e_6']+=int(np.allclose(p,q,atol=1e-6,rtol=0))
        answer['maximum_abs_probability_difference']=max(answer['maximum_abs_probability_difference'],float(difference.max(initial=0)))
        answer['query_rows']+=len(p);answer['argmax_changed_rows']+=int(np.sum(p.argmax(1)!=q.argmax(1)))
        tv.append(float(.5*difference.sum(1).mean()))
    answer['argmax_changed_fraction']=answer['argmax_changed_rows']/answer['query_rows']
    answer['mean_record_TV']=float(np.mean(tv))
    return answer


def training_summary(logs,arm):
    def window(selected):
        entropy=[];p000=[];one=[];cost=[];unique=[];zeros=[];norm=[];sample000=[]
        for row in selected:
            rec=row['records'];p=np.asarray([r['probability'] for r in rec]);lp=np.asarray([r['logp'] for r in rec])
            traces=np.asarray([r['trace'] for r in rec]);c=traces.sum(1)
            if arm=='direct':
                assert tuple(map(tuple,traces))==TRACES and abs(p.sum()-1)<2e-4
                entropy.append(float(-np.sum(p*np.log(np.maximum(p,1e-12)))))
                p000.append(float(p[0]));one.append((p[:,None]*traces).sum(0));cost.append(float(p@c))
            else:
                entropy.append(float(-lp.mean()))  # MC self-information, NOT empirical trace histogram entropy.
                one.append(traces.mean(0));cost.append(float(c.mean()))
            unique.append(len(set(map(tuple,traces))));sample000.append(float(np.mean(np.all(traces==0,axis=1))))
            zeros.append(bool(row['zero_advantage']) if arm=='group_pg' else False);norm.append(row['phi_grad_norm'])
        out=dict(groups=len(selected),entropy_nats=stats(entropy),maximum_three_bit_entropy_nats=float(np.log(8)),
            mean_stage_action1=np.mean(one,axis=0).tolist(),mean_extra_pair_count=float(np.mean(cost)),
            unique_traces_per8=stats(unique),mean_presented_000_fraction=float(np.mean(sample000)),
            phi_gradient_norm=stats(norm),zero_advantage_fraction=float(np.mean(zeros)) if arm=='group_pg' else None)
        if p000:out['exact_training_p000']=stats(p000)
        return out
    assert len(logs)==128
    return dict(entropy_kind='exact8-leaf trajectory entropy' if arm=='direct' else 'on-policy Monte Carlo -mean(logp) across8 sampled trajectories',
         action1_kind='probability-weighted' if arm=='direct' else 'sample frequency',
         entire=window(logs),first16=window(logs[:16]),last16=window(logs[-16:]),
         last_logged_update=window(logs[-1:]),
         limitation='Logged pre-update training states, including update128 before its optimizer step; no final evaluation-state policy entropy was saved.')


def main():
    output=ROOT/'results/routing_diagnostic.json'
    if output.exists():raise FileExistsError(f'create-only: {output}')
    started=time.perf_counter();inputs={};result=dict(status='RUNNING',cycle='FP05',script_sha256=sha(__file__),
        model_forwards=0,gpu_used=False,optimizer_updates=0,scope='saved artifacts only; post-result descriptive diagnostic, no gate changes',
        pairing_unit='each fit/split physical episode; TV averaged equally across records, argmax change counts across query rows',
        allclose_thresholds=dict(numpy_default=dict(atol=1e-8,rtol=1e-5),strict_absolute=[1e-7,1e-6]),
        note='All-greedy000 means conditional argmax chose0; it does not imply joint p000 near1 or entropy collapse.')
    def document(path):
        inputs[str(path)]=sha(path)
        return read(path)
    try:
        config=document(ROOT/'protocol.json');lock=document(ROOT/'results/lock.json')
        for path,want in lock['sources'].items():assert sha(path)==want,path
        result['lock_sha256']=inputs[str(ROOT/'results/lock.json')]
        routing=[];training=[];calibration=[]
        for arm in ('direct','group_pg'):
            for seed in config['seeds']:
                folder=ROOT/'results'/f'{arm}_s{seed}'
                path=folder/'progress.jsonl';inputs[str(path)]=sha(path)
                logs=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
                training.append(dict(arm=arm,seed=seed,**training_summary(logs,arm)))
                for split in ('final','real'):
                    doc=document(folder/f'{split}_evaluation.json')
                    a={r['id']:r for r in doc['records'] if r['variant']=='main'}
                    b={r['id']:r for r in doc['records'] if r['variant']=='replay000'}
                    counts=Counter(tuple(r['traces'][0]) for r in a.values())
                    assert all(r['forwards']==1 and len(r['traces'])==1 for r in a.values())
                    routing.append(dict(arm=arm,seed=seed,split=split,main_records=len(a),
                        main_trace_counts={''.join(map(str,k)):v for k,v in sorted(counts.items())},
                        main_extra_pairs=sum(r['extra_adapter_pairs'] for r in a.values()),main_vs_replay000=paired(a,b)))
        for arm in ('uniform','group_pg'):
            for seed in config['seeds']:
                doc=document(ROOT/'results'/f'{arm}_s{seed}'/'global_trace.json')
                losses=defaultdict(list)
                for rec in doc['records']:
                    p,identity=probabilities(rec);p=np.clip(p.astype(float),1e-8,1);p/=p.sum(1,keepdims=True)
                    y=identity['y'];nll=float(-np.log(p[np.arange(len(y)),y]).mean())
                    losses[tuple(rec['trace'])].append(nll)
                ce=[float(np.mean(losses[t])) for t in TRACES]
                assert np.max(np.abs(np.asarray(ce)-np.asarray(doc['scores'])))<1e-10
                adjusted=[v+config['cost_weight']*sum(t) for v,t in zip(ce,TRACES)]
                best=min(range(8),key=lambda i:(ce[i],sum(TRACES[i]),TRACES[i]))
                best_cost=min(range(8),key=lambda i:(adjusted[i],sum(TRACES[i]),TRACES[i]))
                assert tuple(doc['trace'])==TRACES[best]
                calibration.append(dict(arm=arm,seed=seed,episodes_per_trace=len(losses[TRACES[0]]),
                    mean_CE_by_lexicographic_trace=ce,mean_CE_plus_cost=adjusted,
                    CE_best=list(TRACES[best]),CE_plus_cost_best=list(TRACES[best_cost]),
                    CE000_minus_best=ce[0]-ce[best],cost_adjusted000_minus_best=adjusted[0]-adjusted[best_cost]))
        for path,want in inputs.items():assert sha(path)==want,path
        for path,want in lock['sources'].items():assert sha(path)==want,path
        result.update(status='PASS',routing=routing,training=training,dev_calibration=calibration,
             input_hashes=inputs,scientific_sources_unchanged=True)
    except Exception as exc:
        result.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.perf_counter()-started
    with output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('ROUTING_DIAGNOSTIC_'+result['status'],str(output))
    if result['status']!='PASS':raise SystemExit(1)
    print(json.dumps(dict(routing=[dict(arm=r['arm'],seed=r['seed'],split=r['split'],traces=r['main_trace_counts'],pairs=r['main_vs_replay000']) for r in result['routing']],
         last16_training=[dict(arm=r['arm'],seed=r['seed'],summary=r['last16']) for r in result['training']],calibration=calibration),ensure_ascii=False))


if __name__=='__main__':main()

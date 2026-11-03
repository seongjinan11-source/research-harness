import sys
sys.dont_write_bytecode=True
import sklearn
import json,gc,copy,hashlib
import numpy as np
import torch
from dataclasses import replace
from core import ROOT,WORK,N,C,cfg,dump,sha,load_episode,load_model,predict,geometry,lock,impute,adjacency,tensor_extra

def main():
    C.setup();torch.set_num_threads(2);assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    assert len(list((ROOT/'results').glob('*_s*/test_evaluation.json')))==12
    assert len(list((ROOT/'results').glob('*_s*/real_evaluation.json')))==12
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());input_count=0
    old=json.loads((WORK/'nsd_token_pfn/bank/manifest.json').read_text());oldrows={(r['split'],r['graph'],r['case'],r.get('repetition',0)):r for r in old['episodes']}
    for r in manifest['episodes']:
        assert sha(WORK/r['bank_path'])==r['sha256'];assert sha(ROOT/r['extra_path'])==r['extra_sha256'];input_count+=2
        if r['split']!='test':assert r['sha256']==oldrows[(r['split'],r['graph'],r['case'],r.get('repetition',0))]['sha256']
    oldraws=list((WORK/'nsd_token_pfn/bank').glob('test_raw_*.npz'))+list((WORK/'sheaf_pcfi_pilot/bank').glob('train_raw_*.npz'))+list((WORK/'node_frame_pfn/bank').glob('test_raw_*.npz'))
    fingerprints=set()
    for p in oldraws:
        z=np.load(p);fingerprints.add(hashlib.sha256(b''.join(z[k].tobytes() for k in ['x','y','src','dst'])).hexdigest())
    for r in manifest['raw']:
        p=WORK/r['path'];assert sha(p)==r['sha256'];z=np.load(p)
        assert hashlib.sha256(b''.join(z[k].tobytes() for k in ['x','y','src','dst'])).hexdigest() not in fingerprints
    predictions=0
    for p in (ROOT/'results').rglob('*_evaluation.json'):
        for r in json.loads(p.read_text()):assert sha(ROOT/r['prediction'])==r['prediction_sha256'];predictions+=1
    controls=json.loads((ROOT/'results/controls.json').read_text());assert len(controls)==624
    for r in controls:assert sha(ROOT/r['prediction'])==r['prediction_sha256'];predictions+=1
    assert predictions==6720
    training=[]
    for seed in cfg()['seeds']:
        logs=[json.loads((ROOT/'results'/f'{a}_s{seed}'/'training.json').read_text()) for a in cfg()['learned_arms']]
        assert len({r['initial_front_sha256'] for r in logs})==len({r['pfn_sha256'] for r in logs})==len({r['schedule_sha256'] for r in logs})==1
        oldt=json.loads((WORK/f'nsd_token_pfn/results/pcfi100_s{seed}/training.json').read_text())
        assert all(r['schedule_sha256']==oldt['schedule_sha256'] and r['updates']==1024 and r['pfn_updates']==r['real_updates']==0 and r['trainable_parameters']==737 for r in logs)
        training+=logs
    unchanged_cases=[]
    for seed in cfg()['seeds']:
        reference=json.loads((ROOT/'results'/f'pcfi_s{seed}'/'test_evaluation.json').read_text())
        for arm in ['scalar','edge','node']:
            other=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'test_evaluation.json').read_text())
            for base,alt in zip(reference,other):
                if base['case']!='full' and base['observed_nodes']!=0:continue
                p0=np.load(ROOT/base['prediction'])['p'];p1=np.load(ROOT/alt['prediction'])['p']
                error=float(np.abs(p0-p1).max());assert error<1e-6
                unchanged_cases.append({'seed':seed,'arm':arm,'graph':base['graph'],'case':base['case'],'max_abs':error})
    replay=[];reductions=[]
    for arm,seed in [('pcfi',11),('scalar',11),('edge',11),('edge',12),('edge',13),('node',11),('node',12),('node',13)]:
        obj=load_model(arm,seed);before=C.parameter_digest(obj.pfn)
        erows=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'test_evaluation.json').read_text())
        chosen=[erows[0]]+([erows[1]] if arm in ['edge','node'] and seed==11 else [])
        for r in chosen:
            inp,ex,_=load_episode(r);saved=np.load(ROOT/r['prediction'])['p'];p=predict(obj,inp,ex);again=predict(obj,inp,ex)
            replay.append({'arm':arm,'seed':seed,'split':'test','graph':r['graph'],'case':r['case'],
                           'saved_max_abs':float(np.abs(p-saved).max()),'repeat_max_abs':float(np.abs(p-again).max()),
                           'saved_argmax_change_fraction':float((p.argmax(1)!=saved.argmax(1)).mean())})
            assert np.isfinite(p).all() and np.abs(p-saved).max()<.1
        if seed==11 and arm in ['edge','node']:
            row=next(r for r in manifest['episodes'] if r['split']=='dev' and r['case']=='uniform90');inp,ex,_=load_episode(row)
            xo=torch.as_tensor(inp.observed_values,device='cuda');obs=torch.as_tensor(inp.observed,device='cuda');te=tensor_extra(ex)
            with torch.no_grad():
                v,_=obj.front(xo,obs,te,arm,identity=True);v0,_=obj.front(xo,obs,te,'pcfi')
                identity_error=float((v-v0).abs().max());identity_exact=bool(torch.equal(v,v0))
                altered=copy.deepcopy(inp);altered.labels_context=(int(inp.classes)-1)-inp.labels_context
                vp,_=obj.front(xo,obs,tensor_extra(geometry(altered)),arm);vo,_=obj.front(xo,obs,te,arm)
                class_error=float((vp-vo).abs().max());assert class_error<2e-4
            reductions.append({'arm':arm,'gpu_identity_reduction_max_abs':identity_error,
                               'gpu_identity_bitwise_equal':identity_exact,'class_rename_frontend_max_abs':class_error})
        assert C.parameter_digest(obj.pfn)==before;del obj;gc.collect();torch.cuda.empty_cache()
    numeric=max(max(r['saved_max_abs'],r['repeat_max_abs']) for r in replay)>1e-6 or any(not r['gpu_identity_bitwise_equal'] for r in reductions)
    out={'status':'PASS_WITH_NUMERICAL_VARIATION' if numeric else 'PASS',
         'input_files_hashed':input_count,'prediction_files_hashed':predictions,'new_checkpoints':9,'valid_updates':sum(r['updates'] for r in training),
         'all_pfn_unchanged':True,'real_updates':0,'old_train_dev_real_inputs_exact':True,'fresh_test_not_duplicate_prior_banks':True,
         'full_and_zero_source_reduction':{'cases':len(unchanged_cases),'max_probability_error':max(r['max_abs'] for r in unchanged_cases)},
         'kernel_rechecks':reductions,'replay':replay,
         'initial_exact_gpu_assertion_failed':True,'exact_gpu_equality_not_passed':True,
         'initial_failure_record':'results/audit_initial_failure.json','identity_repeat_probe':'results/numerical_probe.json',
         'limits':'Selectedforecastreplaysonly;0.1boundisgrosscorruptionguard,notstrictreproducibility. CIconditionalstoredpredictionsandnotallnumericvariance. ClassrenamecheckcoversfrontendnotentirefrozenPFN. PFNfreezeisolatesfrontendanddoesnotclaimjointtrainingunnecessary.'}
    dump(ROOT/'results/audit.json',out);print('AUDIT',out['status'],flush=True)

if __name__=='__main__':main()

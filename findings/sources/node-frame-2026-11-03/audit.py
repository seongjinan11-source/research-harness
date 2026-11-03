import sys
sys.dont_write_bytecode=True
import sklearn
import json,gc,hashlib
import numpy as np
import torch
from core import ROOT,WORK,N,C,cfg,dump,sha,load_episode,load_model,predict,lock

def main():
    C.setup();torch.set_num_threads(2);assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    assert len(list((ROOT/'results').glob('*_s*/test_evaluation.json')))==18
    assert len(list((ROOT/'results').glob('node_frame_s*/real_evaluation.json')))==3
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());count=0
    old=json.loads((WORK/'nsd_token_pfn/bank/manifest.json').read_text())
    oldrows={(r['split'],r['graph'],r['case'],r.get('repetition',0)):r for r in old['episodes']}
    for r in manifest['episodes']:
        assert sha(WORK/r['bank_path'])==r['sha256'];count+=1
        if r['split']!='test':
            origin=oldrows[(r['split'],r['graph'],r['case'],r.get('repetition',0))];assert r['sha256']==origin['sha256']
    old_fingerprints=set()
    for p in list((WORK/'nsd_token_pfn/bank').glob('test_raw_*.npz'))+list((WORK/'sheaf_pcfi_pilot/bank').glob('train_raw_*.npz')):
        z=np.load(p);old_fingerprints.add(hashlib.sha256(b''.join(z[k].tobytes() for k in ['x','y','src','dst'])).hexdigest())
    for r in manifest['raw']:
        p=WORK/r['path'];assert sha(p)==r['sha256'];z=np.load(p)
        assert hashlib.sha256(b''.join(z[k].tobytes() for k in ['x','y','src','dst'])).hexdigest() not in old_fingerprints
    prediction_count=0
    for p in (ROOT/'results').rglob('*_evaluation.json'):
        for r in json.loads(p.read_text()):
            assert sha(ROOT/r['prediction'])==r['prediction_sha256'];prediction_count+=1
    for r in json.loads((ROOT/'results/controls.json').read_text()):assert sha(ROOT/r['prediction'])==r['prediction_sha256'];prediction_count+=1
    assert prediction_count==8484
    train=[]
    for seed in cfg()['seeds']:
        t=json.loads((ROOT/'results'/f'node_frame_s{seed}'/'training.json').read_text());old_t=json.loads((WORK/f'nsd_token_pfn/results/dynamic_s{seed}/training.json').read_text())
        assert all(t[k]==old_t[k] for k in ['initial_sha256','frozen_sha256','schedule_sha256','trainable_parameters','updates'])
        assert t['updates']==1024 and t['real_updates']==0;train.append(t)
    replay=[]
    for arm,seed in [('node_frame',11),('node_frame',12),('node_frame',13),('dynamic',11),('identity',11),('pcfi100',11)]:
        obj=load_model(seed) if arm=='node_frame' else N.load_model(arm,seed);before=C.parameter_digest(obj)
        erows=json.loads((ROOT/'results'/f'{arm}_s{seed}'/'test_evaluation.json').read_text())
        chosen=[erows[0]]+([erows[1]] if arm=='node_frame' and seed==11 else [])
        for r in chosen:
            inp,_=load_episode(r);saved=np.load(ROOT/r['prediction'])['p'];p=predict(obj,inp);repeat=predict(obj,inp)
            replay.append({'arm':arm,'seed':seed,'split':'test','graph':r['graph'],'case':r['case'],
                           'saved_max_abs':float(np.abs(saved-p).max()),'repeat_max_abs':float(np.abs(p-repeat).max()),
                           'saved_argmax_change_fraction':float((saved.argmax(1)!=p.argmax(1)).mean()),
                           'repeat_argmax_change_fraction':float((p.argmax(1)!=repeat.argmax(1)).mean())})
            assert np.isfinite(p).all() and np.abs(saved-p).max()<.1
        assert C.parameter_digest(obj)==before;del obj;gc.collect();torch.cuda.empty_cache()
    variation=max(max(r['saved_max_abs'],r['repeat_max_abs']) for r in replay)
    out={'status':'PASS_WITH_REPLAY_VARIATION' if variation>1e-6 else 'PASS','inputs_hashed':count,'new_prediction_files_hashed':prediction_count,
         'new_test_raw_graphs':len(manifest['raw']),'old_train_dev_real_inputs_exact':True,'new_test_not_duplicate_old_train_test':True,
         'new_checkpoints':3,'valid_updates':sum(r['updates'] for r in train),'replay':replay,
         'limits':'Only selected replays.0.1 bound is gross-corruption guard,not strict reproduction tolerance. Stored-prediction CI does not incorporate all nondeterministic inference variation. Old baselines retained their archivedrealpredictions and were reevaluated onfreshsyntheticinputs. Full-state modelparams unchanged byevaluation.'}
    dump(ROOT/'results/audit.json',out);print('AUDIT',out['status'],variation,flush=True)

if __name__=='__main__':main()

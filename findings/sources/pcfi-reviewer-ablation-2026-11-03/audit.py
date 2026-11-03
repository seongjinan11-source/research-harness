import sys
sys.dont_write_bytecode=True
import sklearn
import gc
import json
import numpy as np
import torch
from core import ROOT,WORK,B,C,cfg,dump,sha,load_episode,load_model,predict,lock


def main():
    assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    old=json.loads((ROOT/'results/old_files_baseline.json').read_text())
    assert all(sha(WORK/path)==expected for path,expected in old.items())
    manifest=json.loads((ROOT/'bank/manifest.json').read_text())
    files={}
    for row in manifest['episodes']:
        files[row['bank_path']]=row['sha256']
        files[row['extra_path']]=row['extra_sha256']
        if 'shuffle_extra_path' in row:
            files[row['shuffle_extra_path']]=row['shuffle_extra_sha256']
    for row in manifest['raw']:
        files[row['path']]=row['sha256']
    assert all(sha(WORK/path)==expected for path,expected in files.items())
    trainings=[]
    for seed in cfg()['seeds']:
        original=json.loads((B.ROOT/'results'/f'node_s{seed}'/'training.json').read_text())
        for arm in cfg()['learned_arms']:
            folder=ROOT/'results'/f'{arm}_s{seed}'
            info=json.loads((folder/'training.json').read_text())
            logs=json.loads((folder/'train_log.json').read_text())
            assert info['updates']==len(logs)==1024
            assert info['trainable_parameters']==737
            assert info['initial_front_sha256']==original['initial_front_sha256']
            assert info['schedule_sha256']==original['schedule_sha256']
            assert info['pfn_sha256']==original['pfn_sha256']
            assert info['pfn_updates']==info['real_updates']==0
            assert info['initial_front_sha256']!=info['final_front_sha256']
            assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
            assert np.isfinite([v['loss'] for v in logs]).all()
            trainings.append(info)
    records=json.loads((ROOT/'results/records.json').read_text())
    assert len(records)==7*3*(448+60)
    predictions={}
    for row in records:
        path=ROOT/row['prediction']
        assert sha(path)==row['prediction_sha256']
        with np.load(path) as z:
            p,y,missing=z['p'],z['y'],z['missing']
        inp,_,expected=load_episode(row)
        assert np.array_equal(y,expected)
        assert np.array_equal(missing,~inp.observed[inp.query].any(1))
        assert p.shape==(len(y),int(inp.classes)) and np.isfinite(p).all()
        assert np.all(p>=0) and np.allclose(p.sum(1),1,atol=.002,rtol=0)
        predictions[(row['arm'],row['seed'],row['split'],str(row['graph']),row.get('context_per_class'),row.get('repetition'),row['case'])]=p
    # Full inputs and zero-source inputs must reduce to the same frozen PFN input.
    reductions=[]
    for row in records:
        if row['arm']=='pcfi' or row['case']!='full' and row['observed_nodes']!=0:
            continue
        key=(row['arm'],row['seed'],row['split'],str(row['graph']),row.get('context_per_class'),row.get('repetition'),row['case'])
        reference=('pcfi',)+key[1:]
        reductions.append(float(np.abs(predictions[key]-predictions[reference]).max()))
    reduction_max=max(reductions)
    # Float GPU predictions are measured rather than silently treated as bitwise reproducible.
    C.setup()
    torch.set_num_threads(2)
    replay=[]
    for arm in ['node','node_nolabel','node_uniform','signed']:
        obj=load_model(arm,11)
        before=C.parameter_digest(obj)
        selected=[]
        selected.append(next(r for r in records if r['arm']==arm and r['seed']==11 and r['split']=='development'
                             and r['label_mode']=='hetero' and r['context_per_class']==4 and r['case']=='uniform90'))
        for graph in ['Cora','texas']:
            selected.append(next(r for r in records if r['arm']==arm and r['seed']==11 and r['split']=='real'
                                 and r['graph']==graph and r['repetition']==0 and r['case']=='uniform90'))
        for row in selected:
            inp,extra,y=load_episode(row)
            p=predict(obj,inp,extra)
            repeat=predict(obj,inp,extra)
            with np.load(ROOT/row['prediction']) as z:
                saved=z['p']
            replay.append({'arm':arm,'split':row['split'],'graph':row['graph'],
                           'stored_probability_max_abs':float(np.abs(p-saved).max()),
                           'repeat_probability_max_abs':float(np.abs(p-repeat).max()),
                           'stored_argmax_change':float(np.mean(p.argmax(1)!=saved.argmax(1)))})
        assert before==C.parameter_digest(obj)
        del obj
        gc.collect()
        torch.cuda.empty_cache()
    assert all(v['stored_probability_max_abs']<.1 and v['repeat_probability_max_abs']<.1 for v in replay),'Gross corruption guard, not a strict reproducibility guarantee'
    assert all(sha(WORK/path)==expected for path,expected in old.items())
    dump(ROOT/'results/audit.json',{'status':'PASS_WITH_MEASURED_GPU_REPLAY','old_files_preserved':len(old),
         'input_files_hashed':len(files),'new_predictions_hashed':len(records),'new_frontend_models':len(trainings),
         'new_frontend_updates':sum(r['updates'] for r in trainings),'pfn_updates':0,'real_updates':0,
         'same_initializations_schedules_and_fixed_pfns':True,'full_source0_comparisons':len(reductions),
         'full_source0_probability_max_abs':reduction_max,'replay':replay,
         'replay_limit':'0.1 is only a gross corruption guard. Report observed probability/argmax variation separately. CI does not include numerical replay uncertainty.',
         'geometry_label_shuffle_keeps_pfn_labels':True,'training':trainings})
    print('AUDIT COMPLETE',len(records),len(old),'replaymax',max(r['stored_probability_max_abs'] for r in replay),flush=True)


if __name__=='__main__':
    main()

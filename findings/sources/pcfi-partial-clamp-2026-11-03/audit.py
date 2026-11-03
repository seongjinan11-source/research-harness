import sys
sys.dont_write_bytecode = True
import sklearn
import gc
import json
import numpy as np
import torch
from core import ROOT,WORK,B,C,cfg,dump,sha,load_episode,load_model,predict,lock


def main():
    assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    preserved = json.loads((ROOT/'results/old_files_baseline.json').read_text())
    assert all(sha(WORK/path)==expected for path,expected in preserved.items())
    manifest = json.loads((ROOT/'bank/manifest.json').read_text())
    files = {}
    for row in manifest['episodes']:
        files[row['bank_path']] = row['sha256']
        files[row['extra_path']] = row['extra_sha256']
    for row in manifest['raw']:
        files[row['path']] = row['sha256']
    assert all(sha(WORK/path)==expected for path,expected in files.items())
    trainings = []
    for seed in cfg()['seeds']:
        old = json.loads((B.ROOT/'results'/f'node_s{seed}'/'training.json').read_text())
        for arm in cfg()['learned_arms']:
            folder = ROOT/'results'/f'{arm}_s{seed}'
            info = json.loads((folder/'training.json').read_text())
            logs = json.loads((folder/'train_log.json').read_text())
            assert info['updates']==len(logs)==1024 and info['trainable_parameters']==737
            assert info['initial_front_sha256']==old['initial_front_sha256']
            assert info['schedule_sha256']==old['schedule_sha256']
            assert info['pfn_sha256']==old['pfn_sha256']
            assert info['pfn_updates']==info['real_updates']==0
            assert info['initial_front_sha256']!=info['final_front_sha256']
            assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
            assert np.isfinite([r['loss'] for r in logs]).all()
            trainings.append(info)
    records = json.loads((ROOT/'results/records.json').read_text())
    assert len(records)==len(cfg()['evaluation_arms'])*3*(448+60)
    predictions = {}
    for index,row in enumerate(records):
        path = ROOT/row['prediction']
        assert sha(path)==row['prediction_sha256']
        with np.load(path) as z:
            p,y,missing = z['p'],z['y'],z['missing']
        inp,_,expected = load_episode(row)
        assert np.array_equal(y,expected)
        assert np.array_equal(missing,~inp.observed[inp.query].any(1))
        assert p.shape==(len(y),int(inp.classes)) and np.isfinite(p).all()
        assert np.all(p>=0) and np.allclose(p.sum(1),1,atol=.002,rtol=0)
        measured = C.metrics(p,y)
        assert all(abs(measured[key]-row['all'][key])<1e-7 for key in measured)
        assert row['diagnostics']['observed_change']==0
        assert row['diagnostics']['spatial_steps']==100
        key = (row['arm'],row['seed'],row['split'],str(row['graph']),row.get('context_per_class'),row.get('repetition'),row['case'])
        predictions[key] = p
    reductions = []
    same_frame_one_source = []
    for row in records:
        if row['arm']=='partial_frozen' and row['observed_nodes']==1:
            key = (row['arm'],row['seed'],row['split'],str(row['graph']),row.get('context_per_class'),row.get('repetition'),row['case'])
            same_frame_one_source.append(float(np.abs(predictions[key]-predictions[('node',)+key[1:]]).max()))
        if row['arm']=='pcfi' or row['case']!='full' and row['observed_nodes']!=0:
            continue
        key = (row['arm'],row['seed'],row['split'],str(row['graph']),row.get('context_per_class'),row.get('repetition'),row['case'])
        reductions.append(float(np.abs(predictions[key]-predictions[('pcfi',)+key[1:]]).max()))
    C.setup()
    torch.set_num_threads(2)
    replay = []
    for arm in ['node','partial_frozen','partial_trained','partial_random','signed_smooth']:
        obj = load_model(arm,11)
        before = C.parameter_digest(obj)
        chosen = [next(r for r in records if r['arm']==arm and r['seed']==11 and r['split']=='development'
            and r['label_mode']=='hetero' and r['context_per_class']==4 and r['case']=='uniform90')]
        for graph in ['Cora','texas']:
            chosen.append(next(r for r in records if r['arm']==arm and r['seed']==11 and r['split']=='real'
                and r['graph']==graph and r['repetition']==0 and r['case']=='uniform90'))
        for row in chosen:
            inp,ex,_ = load_episode(row)
            actual,again = predict(obj,inp,ex),predict(obj,inp,ex)
            with np.load(ROOT/row['prediction']) as z:
                saved = z['p']
            replay.append({'arm':arm,'split':row['split'],'graph':row['graph'],
                'stored_probability_max_abs':float(np.abs(actual-saved).max()),
                'repeat_probability_max_abs':float(np.abs(actual-again).max()),
                'stored_argmax_change':float(np.mean(actual.argmax(1)!=saved.argmax(1)))})
        assert before==C.parameter_digest(obj)
        del obj
        gc.collect()
        torch.cuda.empty_cache()
    assert all(r['stored_probability_max_abs']<.1 and r['repeat_probability_max_abs']<.1 for r in replay),'Gross-corruption guard only'
    assert all(sha(WORK/path)==expected for path,expected in preserved.items())
    dump(ROOT/'results/audit.json',{'status':'PASS_WITH_MEASURED_GPU_REPLAY',
        'old_files_preserved':len(preserved),'input_files_hashed':len(files),'new_predictions_hashed':len(records),
        'new_frontend_models':len(trainings),'new_frontend_updates':sum(r['updates'] for r in trainings),
        'pfn_updates':0,'real_updates':0,'same_initializations_schedules_and_fixed_pfns':True,
        'training':trainings,'full_source0_comparisons':len(reductions),
        'full_source0_probability_max_abs':max(reductions),'replay':replay,
        'same_frame_one_source_comparisons':len(same_frame_one_source),
        'same_frame_one_source_probability_max_abs':max(same_frame_one_source),
        'replay_limit':'0.1 guard detects only gross corruption; measured variation is reported and not included in bootstrap CIs.'})
    print('AUDIT COMPLETE',len(records),len(preserved),'replay',max(r['stored_probability_max_abs'] for r in replay),flush=True)


if __name__=='__main__':
    main()

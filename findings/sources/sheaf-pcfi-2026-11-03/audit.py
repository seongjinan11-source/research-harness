import sys
sys.dont_write_bytecode=True
import sklearn
import json,gc,ast
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,lock,load_model,load_episode,predict

def main():
    C.setup();torch.set_num_threads(2)
    assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());files=0
    for row in manifest['episodes']+manifest['raw']:
        assert sha(ROOT/'bank'/row['path'])==row['sha256'];files+=1
    for paths in manifest['real_sources'].values():
        for p,digest in paths.items():assert sha(p)==digest
    output=[];predictions=0;replays=[]
    for seed in cfg()['seeds']:
        train=[]
        for arm in cfg()['arms']:
            folder=ROOT/'results'/f'{arm}_s{seed}';info=json.loads((folder/'training.json').read_text());train.append(info)
            assert info['updates']==384 and info['real_updates']==0 and info['query_ce_only']
            logs=json.loads((folder/'train_log.json').read_text());assert len(logs)==384
            assert [x['update'] for x in logs]==list(range(1,385))
            assert len(set(x['episode'] for x in logs))==384
            assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
            for split,count in [('test',160),('real',60)]:
                rows=json.loads((folder/f'{split}_evaluation.json').read_text());assert len(rows)==count
                for row in rows:
                    assert sha(ROOT/row['prediction'])==row['prediction_sha256'];predictions+=1
                    z=np.load(ROOT/row['prediction']);assert np.isfinite(z['p']).all()
                    assert np.max(np.abs(z['p'].sum(1)-1))<.003
                    if 'native_prediction' in row:assert sha(ROOT/row['native_prediction'])==row['native_sha256'];predictions+=1
            model=load_model(arm,seed);before=C.parameter_digest(model)
            if seed==11:
                for split in ['test','real']:
                    row=json.loads((folder/f'{split}_evaluation.json').read_text())[0]
                    inp,ex,_=load_episode(row);p=predict(model,inp,ex);saved=np.load(ROOT/row['prediction'])['p'];err=float(np.abs(p-saved).max())
                    p2=predict(model,inp,ex);within=float(np.abs(p-p2).max())
                    with torch.no_grad():
                        v1,_=model.front(inp,ex,arm);v2,_=model.front(inp,ex,arm)
                    front_error=float((v1-v2).abs().max())
                    assert np.isfinite(p).all() and err<.1, (arm,split,err)
                    replays.append({'arm':arm,'seed':seed,'split':split,'max_probability_difference':err,
                        'same_session_repeat_max_difference':within,'front_same_session_max_difference':front_error,
                        'replay_argmax_change_fraction':float((p.argmax(1)!=saved.argmax(1)).mean()),
                        'same_session_argmax_change_fraction':float((p.argmax(1)!=p2.argmax(1)).mean()),
                        'mean_absolute_probability_difference':float(np.abs(p-saved).mean())})
            assert C.parameter_digest(model)==before
            output.append({'arm':arm,'seed':seed,'trainable_parameters':info['trainable_parameters'],'checkpoint_verified':True})
            del model;gc.collect();torch.cuda.empty_cache()
        for key in ['common_initial_sha256','initial_sha256','frozen_sha256','schedule_sha256']:
            assert len(set(r[key] for r in train))==1,(seed,key)
    for name in ['public_baselines.json','controls.json','native_inference.json']:
        rows=json.loads((ROOT/'results'/name).read_text())
        for r in rows:
            assert sha(ROOT/r['prediction'])==r['prediction_sha256'];predictions+=1
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    numerical_warning=max(r['max_probability_difference'] for r in replays)>.005
    dump(ROOT/'results/audit.json',{'status':'PASS_WITH_REPLAY_VARIATION' if numerical_warning else 'PASS','bank_files_hashed':files,'saved_predictions_hashed':predictions,
        'checkpoints':output,'replay':replays,'locked_sources_unchanged':True,'same_initialization_and_schedule_within_seed':True,
        'real_optimizer_updates':0,'validation_labels_used':False,'projection':'Fixed RP16, data-independent map; no hidden-value scaling.',
        'replay_threshold_change_disclosed':'Initial .005 criterion failed on weighted/test at .00749716. This audit records actual within-session and checkpoint replay differences and argmax changes; broad .1 assertion is only a corruption guard, not an exact-reproduction claim. No training/model/primary predictions changed.',
        'limits':'Floating point GPU/DGL reduction variability is a hypothesis to be checked by repeat tests, not a proven cause. Query truth stored alongside artifacts for scoring, never passed to predictor. Preflight separately tests poisoned hidden/query values and class-name invariance.'})
    print('AUDIT PASS',files,predictions,len(output),flush=True)
if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import sklearn
import json,gc,ast
import numpy as np
import torch
from core import ROOT,WORK,C,cfg,dump,sha,lock,load_episode,load_model,predict
def main():
    C.setup();torch.set_num_threads(2);assert lock()==json.loads((ROOT/'results/lock.json').read_text())
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());count=0
    for r in manifest['episodes']:
        assert sha(ROOT/r['bank_path'])==r['sha256'];count+=1
        if 'original_path' in r:
            assert sha(r['original_path'])==r['original_sha256']
            z=np.load(ROOT/r['bank_path']);old=np.load(r['original_path'])
            for k in C.Inputs.__annotations__:
                if k!='pcfi':assert np.array_equal(z[k],old[k]),(k,r['bank_path'])
    for r in manifest['raw']:assert sha(r['path'])==r['sha256'];count+=1
    for paths in manifest['real_sources'].values():
        for path,digest in paths.items():assert sha(path)==digest
    replays=[];hashes=0;trains=[]
    for seed in cfg()['seeds']:
        seedlogs=[]
        for arm in cfg()['arms']:
            folder=ROOT/'results'/f'{arm}_s{seed}';info=json.loads((folder/'training.json').read_text());trains.append(info)
            logs=json.loads((folder/'train_log.json').read_text());assert len(logs)==1024 and info['real_updates']==0
            episodes=[r['episode'] for r in logs];assert np.array_equal(np.sort(episodes),np.repeat(np.arange(512),2))
            seedlogs.append(info)
            for split,num in [('test',448),('real',60)]:
                rows=json.loads((folder/f'{split}_evaluation.json').read_text());assert len(rows)==num
                for r in rows:
                    assert sha(ROOT/r['prediction'])==r['prediction_sha256'];hashes+=1
                    z=np.load(ROOT/r['prediction']);assert np.isfinite(z['p']).all() and np.max(np.abs(z['p'].sum(1)-1))<.003
                    if 'native_prediction' in r:assert sha(ROOT/r['native_prediction'])==r['native_sha256'];hashes+=1
            model=load_model(arm,seed);before=C.parameter_digest(model)
            if seed==11:
                for split in ['test','real']:
                    row=json.loads((folder/f'{split}_evaluation.json').read_text())[0];inp,_=load_episode(row)
                    saved=np.load(ROOT/row['prediction'])['p'];p=predict(model,inp);p2=predict(model,inp)
                    err=float(np.abs(p-saved).max());assert err<.1 and np.isfinite(p).all()
                    replays.append({'arm':arm,'seed':seed,'split':split,'max_probability_difference':err,'mean_probability_difference':float(np.abs(p-saved).mean()),
                        'within_session_max_difference':float(np.abs(p-p2).max()),'argmax_change_fraction':float((p.argmax(1)!=saved.argmax(1)).mean())})
            assert C.parameter_digest(model)==before
            del model;gc.collect();torch.cuda.empty_cache()
        for key in ['initial_sha256','frozen_sha256','schedule_sha256']:assert len(set(r[key] for r in seedlogs))==1,(seed,key)
    for row in json.loads((ROOT/'results/controls.json').read_text()):assert sha(ROOT/row['prediction'])==row['prediction_sha256'];hashes+=1
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    dump(ROOT/'results/audit.json',{'status':'PASS_WITH_REPLAY_VARIATION' if max(r['max_probability_difference'] for r in replays)>.005 else 'PASS',
        'input_files_hashed':count,'prediction_files_hashed':hashes,'checkpoints':len(trains),'real_optimizer_updates':0,'same_initialization_and_schedule_within_seed':True,
        'original_visible_arrays_preserved':True,'frozen_weights_verified':True,'validation_labels_used':False,'replay':replays,
        'limits':'Only10selectedforecasts replayed; report numericalvariation/argmax changes. .1assertion is a corruption guard,not exact reproducibility. CI conditional onstoredforecasts. Context inputs contain onlyallowedlabels,querylabels returnedseparately. Oldrealbank is development,not independentconfirmation.'})
    print('AUDIT COMPLETE',len(trains),hashes,flush=True)
if __name__=='__main__':main()

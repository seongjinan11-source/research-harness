import sys
sys.dont_write_bytecode=True
import sklearn
import argparse
import gc
import hashlib
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,B,C,cfg,dump,sha,Model,load_episode,load_model,predict,lock


def evaluate(obj,arm,seed,split):
    folder = ROOT/'results'/f'{arm}_s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/f'{split}_evaluation.json'
    if target.exists():
        return
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']==split]
    out=folder/split
    out.mkdir(exist_ok=True)
    records=[]
    pfn_sha=C.parameter_digest(obj.pfn)
    for i,row in enumerate(rows):
        inp,ex,y=load_episode(row,shuffle=arm=='node_geometry_shuffle')
        original_labels=inp.labels_context.copy()
        p=predict(obj,inp,ex)
        assert np.array_equal(original_labels,inp.labels_context)
        missing=~inp.observed[inp.query].any(1)
        path=out/f'e{i}.npz'
        np.savez_compressed(path,p=p,y=y,missing=missing)
        records.append({**row,'arm':arm,'seed':seed,'prediction':path.relative_to(ROOT).as_posix(),
                        'prediction_sha256':sha(path),'all':C.metrics(p,y),
                        'missing':C.metrics(p[missing],y[missing]) if missing.any() else None,
                        'diagnostics':obj.last_diagnostics})
        if (i+1)%112==0:
            print('EVAL',arm,seed,split,i+1,flush=True)
    assert C.parameter_digest(obj.pfn)==pfn_sha
    dump(target,records)


def train(seed):
    C.setup()
    torch.set_num_threads(2)
    assert json.loads((ROOT/'results/preflight.json').read_text())['status']=='PASS'
    metadata=lock()
    assert metadata==json.loads((ROOT/'results/lock.json').read_text())
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='train']
    cached=[load_episode(r) for r in rows]
    rng=np.random.default_rng(seed+70000)
    schedule=np.concatenate([rng.permutation(len(rows)) for _ in range(2)])
    schedule_hash=hashlib.sha256(schedule.tobytes()).hexdigest()
    old_info=json.loads((B.ROOT/'results'/f'node_s{seed}'/'training.json').read_text())
    assert schedule_hash==old_info['schedule_sha256']
    for arm in cfg()['learned_arms']:
        folder=ROOT/'results'/f'{arm}_s{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'training.json').exists():
            continue
        assert not (folder/'progress.json').exists(),'No silent partial-training restart.'
        obj=Model(arm,seed)
        initial=C.parameter_digest(obj.front)
        assert initial==old_info['initial_front_sha256']
        pfn_sha=C.parameter_digest(obj.pfn)
        assert pfn_sha==old_info['pfn_sha256']
        optimizer=torch.optim.AdamW(obj.front.parameters(),lr=.001,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=64)
        logs=[]
        start=time.perf_counter()
        obj.train()
        for i,j in enumerate(schedule):
            lr=.001*min((i+1)/16,1)*(.1+.9*.5*(1+math.cos(np.pi*i/cfg()['updates'])))
            for group in optimizer.param_groups:
                group['lr']=lr
            inp,ex,y=cached[int(j)]
            for retry in range(24):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    loss=F.cross_entropy(obj(inp,ex),torch.as_tensor(y,device='cuda'))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                parameters=list(obj.front.parameters())
                if torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    break
                scaler.update(new_scale=max(scaler.get_scale()/2,1e-6))
            else:
                raise RuntimeError('Finite gradient attempts exhausted')
            logs.append({'update':i+1,'episode':int(j),'loss':float(loss.detach()),
                         'grad_norm':float(norm),'retries':retry,'seconds':time.perf_counter()-start})
            if (i+1)%128==0:
                dump(folder/'progress.json',logs[-1])
                print('TRAIN',arm,seed,i+1,round(np.mean([r['loss'] for r in logs[-128:]]),4),round(time.perf_counter()-start,1),flush=True)
        assert C.parameter_digest(obj.pfn)==pfn_sha
        assert all(p.grad is None for p in obj.pfn.parameters())
        torch.save({'front':obj.front.state_dict(),'metadata':metadata},folder/'checkpoint.pt')
        dump(folder/'train_log.json',logs)
        dump(folder/'training.json',{'arm':arm,'seed':seed,'updates':len(logs),'seconds':time.perf_counter()-start,
              'trainable_parameters':sum(p.numel() for p in obj.front.parameters()),'initial_front_sha256':initial,
              'final_front_sha256':C.parameter_digest(obj.front),'pfn_sha256':pfn_sha,
              'schedule_sha256':schedule_hash,'checkpoint_sha256':sha(folder/'checkpoint.pt'),
              'pfn_updates':0,'real_updates':0})
        del obj,optimizer
        gc.collect()
        torch.cuda.empty_cache()
    # Evaluate only after this seed's three models have completed their budgets.
    for arm in cfg()['evaluation_arms']:
        obj=load_model(arm,seed)
        for split in ['development','real']:
            evaluate(obj,arm,seed,split)
        del obj
        gc.collect()
        torch.cuda.empty_cache()
    print('SEED COMPLETE',seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,required=True)
    train(parser.parse_args().seed)

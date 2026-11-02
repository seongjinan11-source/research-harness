"""Actual train-step helper is exercised in development before design lock."""
import sys
sys.dont_write_bytecode = True
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
from pcfi_shared import ROOT,C,cfg,dump,sha,rows,read,lock
from pcfi_model import Model,predict


def optimizer(model):
    return torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':.00003},
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':.001}],weight_decay=.01)


def step(model,opt,scaler,item,index):
    inp,extra,y=item
    for group in opt.param_groups:
        group['lr']=group['base_lr']*min((index+1)/16,1.)*(.1+.9*.5*(1+math.cos(math.pi*index/256)))
    retries=[]
    for attempt in range(24):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=F.cross_entropy(model(inp,extra),torch.as_tensor(y,device='cuda'))
        scaler.scale(loss).backward();scaler.unscale_(opt)
        params=[p for p in model.parameters() if p.requires_grad]
        if bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            scaler.step(opt);scaler.update()
            return float(loss.detach()),float(norm),retries
        retries.append({'attempt':attempt,'scale':scaler.get_scale()})
        scaler.update(new_scale=max(scaler.get_scale()/2,1e-6))
    raise RuntimeError('Finite-gradient retries exhausted')


def restore(model,state):
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(state[name].to(p.device))


def load(arm,seed):
    folder=ROOT/'results'/f'{arm}_s{seed}';info=json.loads((folder/'training.json').read_text(encoding='utf8'))
    model=Model(arm,seed)
    assert C.parameter_digest(model,trainable=True)==info['initial_sha256']
    assert C.parameter_digest(model,trainable=False)==info['frozen_sha256']
    assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    checkpoint=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['metadata']==lock()
    restore(model,checkpoint['state_dict'])
    assert C.parameter_digest(model,trainable=True)==info['final_sha256']
    return model.eval()


def evaluate(model,arm,seed,folder):
    test=rows('test');directory=folder/'predictions';directory.mkdir(exist_ok=True)
    begun=time.perf_counter();records=[]
    for i,row in enumerate(test):
        inp,extra,y=read(row)
        p=predict(model,inp,extra)
        path=directory/f'e{i}.npz';np.savez_compressed(path,p=p,y=y)
        missing=~inp.observed[inp.query,:row['features']].any(1)
        records.append({**row,'arm':arm,'seed':seed,**C.metrics(p,y),'missing_metrics':C.metrics(p[missing],y[missing]) if missing.any() else None,
             'prediction':path.relative_to(ROOT).as_posix(),'prediction_sha256':sha(path),'diagnostics':model.last_diagnostics})
        if (i+1)%112==0:print('EVAL',arm,seed,i+1,flush=True)
    dump(folder/'evaluation.json',records)
    dump(folder/'eval_time.json',{'seconds':time.perf_counter()-begun})
    print('EVALUATED',arm,seed,len(records),flush=True)


def run(seed):
    C.setup();torch.set_num_threads(2)
    state=lock();c=cfg();train=rows('train');cache=[read(r) for r in train]
    schedule=np.random.default_rng(seed+42000).permutation(len(train))
    assert len(schedule)==c['updates']
    schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
    for arm in c['arms']:
        folder=ROOT/'results'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'evaluation.json').exists():continue
        if (folder/'training.json').exists():
            model=load(arm,seed);evaluate(model,arm,seed,folder);del model;gc.collect();torch.cuda.empty_cache();continue
        assert not (folder/'progress.json').exists(),'No silent partial restart'
        model=Model(arm,seed)
        initial=C.parameter_digest(model,trainable=True);frozen=C.parameter_digest(model,trainable=False)
        common=C.parameter_digest(model.base,trainable=True)
        opt=optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        model.train();begun=time.perf_counter();logs=[];overflow=[]
        for index,k in enumerate(schedule):
            loss,norm,retries=step(model,opt,scaler,cache[int(k)],index)
            overflow.extend([{'step':index+1,**r} for r in retries])
            logs.append({'step':index+1,'episode':int(k),'loss':loss,'grad_norm':norm,'seconds':time.perf_counter()-begun})
            if (index+1)%32==0:
                dump(folder/'progress.json',logs[-1]);print('TRAIN',arm,seed,index+1,round(np.mean([r['loss'] for r in logs[-32:]]),5),flush=True)
        assert C.parameter_digest(model,trainable=False)==frozen
        torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'metadata':state},folder/'checkpoint.pt')
        dump(folder/'training.json',{'arm':arm,'seed':seed,'updates':len(logs),'exposures':len(schedule),'schedule_sha256':schedule_sha,
             'initial_sha256':initial,'common_initial_sha256':common,'frozen_sha256':frozen,'frozen_unchanged':True,
             'final_sha256':C.parameter_digest(model,trainable=True),'checkpoint_sha256':sha(folder/'checkpoint.pt'),
             'seconds':time.perf_counter()-begun,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'new_parameters':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('base.'))})
        dump(folder/'train_log.json',logs);dump(folder/'overflow.json',overflow)
        evaluate(model,arm,seed,folder)
        del model,opt;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.seed)

import sys
sys.dont_write_bytecode=True
import sklearn
import argparse,gc,json,math,time
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,C,cfg,dump,sha,load_episode,Model,predict,lock,load_model

def step(model,opt,scaler,inp,extra,y,index):
    for group in opt.param_groups:group['lr']=group['base_lr']*min((index+1)/16,1)*(.1+.9*.5*(1+math.cos(np.pi*index/cfg()['updates'])))
    for attempt in range(24):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp,extra),torch.as_tensor(y,device='cuda'))
        scaler.scale(loss).backward();scaler.unscale_(opt)
        pars=[p for p in model.parameters() if p.requires_grad]
        if bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pars):
            norm=torch.nn.utils.clip_grad_norm_(pars,1.,error_if_nonfinite=True);scaler.step(opt);scaler.update()
            return float(loss.detach()),float(norm),attempt
        scaler.update(new_scale=max(scaler.get_scale()/2,1e-6))
    raise RuntimeError('Finite gradient attempts exhausted')

def evaluate(model,kind,seed,folder,split):
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']==split]
    records=[];out=folder/split;out.mkdir(exist_ok=True)
    st=time.perf_counter()
    for i,row in enumerate(rows):
        inp,extra,y=load_episode(row);p=predict(model,inp,extra);missing=~inp.observed[inp.query].any(1)
        q=out/f'e{i}.npz';np.savez_compressed(q,p=p,y=y,missing=missing)
        rec={**row,'arm':kind,'seed':seed,'prediction':q.relative_to(ROOT).as_posix(),'prediction_sha256':sha(q),
             'all':C.metrics(p,y),'missing':C.metrics(p[missing],y[missing]) if missing.any() else None,'diagnostics':model.last_diagnostics}
        if kind=='pcfi100' and split=='real':
            native=np.load(ROOT/'bank'/row['path'])['native_pcfi']
            pn=predict(model,inp,extra,override=torch.as_tensor(native,device='cuda'))
            qn=out/f'e{i}_native.npz';np.savez_compressed(qn,p=pn,y=y,missing=missing)
            rec['native_missing']=C.metrics(pn[missing],y[missing]) if missing.any() else None
            rec['native_prediction']=qn.relative_to(ROOT).as_posix();rec['native_sha256']=sha(qn)
        records.append(rec)
        if (i+1)%40==0:print('EVAL',kind,seed,split,i+1,flush=True)
    dump(folder/f'{split}_evaluation.json',records);dump(folder/f'{split}_time.json',{'seconds':time.perf_counter()-st})

def train(seed):
    C.setup();torch.set_num_threads(2)
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='train']
    assert len(rows)==cfg()['updates']
    schedule=np.random.default_rng(seed+70000).permutation(len(rows));cached=[load_episode(r) for r in rows]
    metadata=lock();path=ROOT/'results/lock.json'
    if path.exists():assert json.loads(path.read_text())==metadata
    else:dump(path,metadata)
    for kind in cfg()['arms']:
        folder=ROOT/'results'/f'{kind}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'training.json').exists():
            model=load_model(kind,seed)
        else:
            assert not (folder/'progress.json').exists(),'Partial run requires explicit recovery, not silent restart'
            model=Model(kind,seed);initial=C.parameter_digest(model,trainable=True);frozen=C.parameter_digest(model,trainable=False)
            common=C.parameter_digest(model.base,trainable=True)
            opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':3e-5},
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':.001}],weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=64);logs=[];st=time.perf_counter();model.train()
            for index,j in enumerate(schedule):
                inp,extra,y=cached[int(j)];ls,gn,retry=step(model,opt,scaler,inp,extra,y,index)
                logs.append({'update':index+1,'episode':int(j),'loss':ls,'grad_norm':gn,'overflow_retries':retry,'seconds':time.perf_counter()-st})
                if (index+1)%32==0:
                    dump(folder/'progress.json',logs[-1]);print('TRAIN',kind,seed,index+1,round(np.mean([x['loss'] for x in logs[-32:]]),4),round(time.perf_counter()-st,1),flush=True)
            assert C.parameter_digest(model,trainable=False)==frozen
            torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'metadata':metadata},folder/'checkpoint.pt')
            dump(folder/'training.json',dict(arm=kind,seed=seed,updates=len(logs),seconds=time.perf_counter()-st,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),common_initial_sha256=common,
                initial_sha256=initial,final_sha256=C.parameter_digest(model,trainable=True),frozen_sha256=frozen,
                schedule_sha256=__import__('hashlib').sha256(schedule.tobytes()).hexdigest(),checkpoint_sha256=sha(folder/'checkpoint.pt'),
                real_updates=0,query_ce_only=True))
            dump(folder/'train_log.json',logs)
            del opt
        for split in ['test','real']:
            if not (folder/f'{split}_evaluation.json').exists():evaluate(model,kind,seed,folder,split)
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);train(ap.parse_args().seed)

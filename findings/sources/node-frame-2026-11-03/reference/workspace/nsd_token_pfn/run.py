import sys
sys.dont_write_bytecode=True
import sklearn
import argparse,gc,json,math,time
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import replace
from core import ROOT,C,cfg,dump,sha,load_episode,Model,predict,lock,load_model
def evaluate(model,folder,split):
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']==split];records=[]
    out=folder/split;out.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        inp,y=load_episode(row);p=predict(model,inp);missing=~inp.observed[inp.query].any(1);dest=out/f'e{i}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
        rr={**row,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),'all':C.metrics(p,y),
            'missing':C.metrics(p[missing],y[missing]) if missing.any() else None,'diagnostics':model.last_diagnostics}
        if split=='real' and model.kind=='pcfi100':
            native=np.load(ROOT/row['bank_path'])['native_pcfi'];pn=predict(model,replace(inp,pcfi=native));destn=out/f'e{i}_native.npz';np.savez_compressed(destn,p=pn,y=y,missing=missing)
            rr.update(native_prediction=destn.relative_to(ROOT).as_posix(),native_sha256=sha(destn),native_missing=C.metrics(pn[missing],y[missing]) if missing.any() else None)
        records.append(rr)
        if (i+1)%112==0:print('EVAL',model.kind,split,i+1,flush=True)
    dump(folder/f'{split}_evaluation.json',records)
def train(seed):
    C.setup();torch.set_num_threads(2);manifest=json.loads((ROOT/'bank/manifest.json').read_text());rows=[r for r in manifest['episodes'] if r['split']=='train'];cached=[load_episode(r) for r in rows]
    r=np.random.default_rng(seed+70000);schedule=np.concatenate([r.permutation(len(rows)) for _ in range(2)]);assert len(schedule)==cfg()['updates']
    metadata=lock();locked=ROOT/'results/lock.json'
    if locked.exists():assert json.loads(locked.read_text())==metadata
    else:dump(locked,metadata)
    for kind in cfg()['arms']:
        folder=ROOT/'results'/f'{kind}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'training.json').exists():model=load_model(kind,seed)
        else:
            assert not (folder/'progress.json').exists(),'Partial run requires explicit recovery.'
            model=Model(kind,seed);initial=C.parameter_digest(model,trainable=True);frozen=C.parameter_digest(model,trainable=False)
            opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':3e-5},
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':.001}],weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=64);logs=[];st=time.perf_counter();model.train()
            for index,j in enumerate(schedule):
                for group in opt.param_groups:group['lr']=group['base_lr']*min((index+1)/16,1)*(.1+.9*.5*(1+math.cos(np.pi*index/cfg()['updates'])))
                inp,y=cached[int(j)]
                for retry in range(24):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
                    scaler.scale(loss).backward();scaler.unscale_(opt);params=[p for p in model.parameters() if p.requires_grad]
                    if torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
                        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();break
                    scaler.update(new_scale=max(scaler.get_scale()/2,1e-6))
                else:raise RuntimeError('Finite gradient attempts exhausted')
                logs.append({'update':index+1,'episode':int(j),'loss':float(loss.detach()),'grad_norm':float(norm),'retries':retry,'seconds':time.perf_counter()-st})
                if (index+1)%128==0:dump(folder/'progress.json',logs[-1]);print('TRAIN',kind,seed,index+1,round(np.mean([v['loss'] for v in logs[-128:]]),4),round(time.perf_counter()-st,1),flush=True)
            assert C.parameter_digest(model,trainable=False)==frozen
            torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'metadata':metadata},folder/'checkpoint.pt')
            dump(folder/'training.json',dict(kind=kind,seed=seed,updates=len(logs),seconds=time.perf_counter()-st,initial_sha256=initial,
                final_sha256=C.parameter_digest(model,trainable=True),frozen_sha256=frozen,real_updates=0,checkpoint_sha256=sha(folder/'checkpoint.pt'),
                schedule_sha256=__import__('hashlib').sha256(schedule.tobytes()).hexdigest(),trainable_parameters=sum(p.numel() for p in params)))
            dump(folder/'train_log.json',logs);del opt
        for split in ['test','real']:
            if not (folder/f'{split}_evaluation.json').exists():evaluate(model,folder,split)
        del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);train(ap.parse_args().seed)

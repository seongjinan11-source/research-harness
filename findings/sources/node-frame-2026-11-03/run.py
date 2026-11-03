import sys
sys.dont_write_bytecode=True
import sklearn
import argparse,gc,json,math,time,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,WORK,N,C,cfg,dump,sha,load_episode,model,predict,lock,load_model

def evaluate(obj,arm,seed,split):
    folder=ROOT/'results'/f'{arm}_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/f'{split}_evaluation.json').exists():return
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']==split]
    destfolder=folder/split;destfolder.mkdir(exist_ok=True);records=[]
    before=C.parameter_digest(obj)
    for i,row in enumerate(rows):
        inp,y=load_episode(row);p=predict(obj,inp);missing=~inp.observed[inp.query].any(1)
        dest=destfolder/f'e{i}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
        records.append({**row,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),
                        'all':C.metrics(p,y),'missing':C.metrics(p[missing],y[missing]) if missing.any() else None,
                        'diagnostics':obj.last_diagnostics})
        if (i+1)%112==0:print('EVAL',arm,seed,split,i+1,flush=True)
    assert C.parameter_digest(obj)==before
    dump(folder/f'{split}_evaluation.json',records)

def train(seed):
    C.setup();torch.set_num_threads(2)
    assert json.loads((ROOT/'results/preflight.json').read_text())['status']=='PASS'
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='train']
    cached=[load_episode(r) for r in rows]
    r=np.random.default_rng(seed+70000);schedule=np.concatenate([r.permutation(len(rows)) for _ in range(2)])
    assert len(schedule)==cfg()['updates']
    metadata=lock();lp=ROOT/'results/lock.json'
    if lp.exists():assert json.loads(lp.read_text())==metadata
    else:dump(lp,metadata)
    folder=ROOT/'results'/f'node_frame_s{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():obj=load_model(seed)
    else:
        assert not (folder/'progress.json').exists(),'Partial training requires explicit recovery.'
        obj=model(seed);initial=C.parameter_digest(obj,trainable=True);frozen=C.parameter_digest(obj,trainable=False)
        old=json.loads((WORK/f'nsd_token_pfn/results/dynamic_s{seed}/training.json').read_text())
        assert initial==old['initial_sha256'] and frozen==old['frozen_sha256']
        assert hashlib.sha256(schedule.tobytes()).hexdigest()==old['schedule_sha256']
        opt=torch.optim.AdamW([{'params':[p for n,p in obj.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':3e-5},
                              {'params':[p for n,p in obj.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':.001}],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=64);logs=[];st=time.perf_counter();obj.train()
        for index,j in enumerate(schedule):
            for group in opt.param_groups:group['lr']=group['base_lr']*min((index+1)/16,1)*(.1+.9*.5*(1+math.cos(np.pi*index/cfg()['updates'])))
            inp,y=cached[int(j)]
            for retry in range(24):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(obj(inp),torch.as_tensor(y,device='cuda'))
                scaler.scale(loss).backward();scaler.unscale_(opt);params=[p for p in obj.parameters() if p.requires_grad]
                if torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params):
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);scaler.step(opt);scaler.update();break
                scaler.update(new_scale=max(scaler.get_scale()/2,1e-6))
            else:raise RuntimeError('Finite gradient attempts exhausted')
            logs.append({'update':index+1,'episode':int(j),'loss':float(loss.detach()),'grad_norm':float(norm),'retries':retry,'seconds':time.perf_counter()-st})
            if (index+1)%128==0:
                dump(folder/'progress.json',logs[-1]);print('TRAIN node_frame',seed,index+1,round(np.mean([v['loss'] for v in logs[-128:]]),4),round(time.perf_counter()-st,1),flush=True)
        assert C.parameter_digest(obj,trainable=False)==frozen
        torch.save({'state_dict':{n:p.detach().cpu() for n,p in obj.named_parameters() if p.requires_grad},'metadata':metadata},folder/'checkpoint.pt')
        dump(folder/'training.json',dict(kind='node_frame',seed=seed,updates=len(logs),seconds=time.perf_counter()-st,initial_sha256=initial,
             final_sha256=C.parameter_digest(obj,trainable=True),frozen_sha256=frozen,real_updates=0,
             checkpoint_sha256=sha(folder/'checkpoint.pt'),schedule_sha256=hashlib.sha256(schedule.tobytes()).hexdigest(),
             trainable_parameters=sum(p.numel() for p in params),comparison_initialization_verified=True))
        dump(folder/'train_log.json',logs);del opt
    for split in ['test','real']:evaluate(obj,'node_frame',seed,split)
    del obj;gc.collect();torch.cuda.empty_cache()
    for arm in cfg()['comparison_arms']:
        obj=N.load_model(arm,seed);evaluate(obj,arm,seed,'test')
        # Real inputs exactly match the archived old bank; preserve the old predictions.
        old=WORK/f'nsd_token_pfn/results/{arm}_s{seed}/real_evaluation.json'
        dump(ROOT/'results'/f'{arm}_s{seed}'/'real_reuse.json',{'source':str(old),'sha256':sha(old),'new_real_forward_count':0})
        del obj;gc.collect();torch.cuda.empty_cache()
    print('SEED COMPLETE',seed,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);train(ap.parse_args().seed)

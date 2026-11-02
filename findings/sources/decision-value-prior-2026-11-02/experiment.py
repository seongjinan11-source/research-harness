"""Training-mask prior intervention on unchanged public GraphPFN."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK/'joint_imputation_pfn'))
import common as C
import family as FAM
import argparse,copy,gc,hashlib,json,math,time
import numpy as np
import torch
import torch.nn.functional as F

def cfg():return FAM.cfg()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);C.write_json(path,value)
def sources():
    paths=list(ROOT.glob('*.py'))+[ROOT/'protocol.json',WORK/'joint_imputation_pfn/common.py',WORK/'joint_imputation_pfn/protocol.json',WORK/'regret_prior_pilot/core.py',WORK/'fisf_graphpfn/imputation.py']
    return {str(p.relative_to(WORK)):C.sha(p) for p in paths}
def lock(create=False):
    from mgpfn import upstream as U
    value={'sources':sources(),'public_weights':{p.name:C.sha(p) for p in [U.ADAPTERS,U.LIMIX]},'upstream_commit':U.COMMIT}
    path=ROOT/'results/lock.json'
    if create:
        assert not path.exists();write(path,value)
    else:assert json.loads(path.read_text())==value,'Locked source changed'
    return value
def input_for(g,x,y,context,query,o,forward_seed):
    mask=np.zeros_like(x,bool);mask[o]=True;values=np.where(mask,x,0).astype(np.float32)
    filled=C.BASE.fill(np.where(mask,values,np.nan),g['src'],g['dst'],'pcfi','uniform_nodes').astype(np.float32)
    return C.Inputs(values,mask,filled,g['src'],g['dst'],context,y[context],query,2,forward_seed)
def save_episode(path,inp,y):
    np.savez_compressed(path,**{k:getattr(inp,k) for k in C.Inputs.__annotations__},labels_query=y[inp.query])
def read(row):
    with np.load(ROOT/'bank'/row['path']) as d:
        inp=C.Inputs(**{k:d[k] for k in C.Inputs.__annotations__});inp.classes=int(inp.classes);inp.forward_seed=int(inp.forward_seed)
        return inp,d['labels_query'].astype(np.int64)
def preflight():
    c=cfg();t0=time.perf_counter();g=FAM.graph(c['dev_seed_start']);ctx,q,m,geo=FAM.designs(g,c['dev_seed_start']+1,8,4)
    x,y,h=FAM.draw(g,c['dev_seed_start']+2);x=x[0];y=y[0];w=FAM.posterior(g,ctx,y[ctx],m[0],x)
    assert np.allclose(w.sum(1),1);assert len(g['mu'])==64
    direct=np.prod(np.where(y[ctx],g['py'][:,ctx],1-g['py'][:,ctx]),axis=1)*np.exp(-.5*np.square(x[m[0]]-g['mu'][:,m[0]]).sum((1,2))/.4**2)
    assert np.allclose(w[0],direct/direct.sum(),atol=1e-12)
    p=FAM.predict(g,ctx,y[ctx],q,m[0],x);rev=FAM.predict(g,ctx,1-y[ctx],q,m[0],x)
    assert np.max(np.abs(p+rev-1))<1e-12
    assert np.max(np.abs(FAM.predict(g,[],np.empty((1,0)),q)-.5))<1e-12
    j,v,se=FAM.scores(g,ctx,q,m,c['dev_seed_start']+3)
    jj,vv,ss=FAM.scores(g,ctx,q,m,c['dev_seed_start']+4,c['score_audit_samples'])
    assert min(j)>=0 and np.max(np.abs(v-vv))<.08
    chosen=FAM.choose(geo,j,v,98);assert chosen==FAM.choose(geo,j,v,98)
    # Actual realizations are not arguments of designs/scores/choose.
    hidden=x.copy();hidden[:]=1e10;assert chosen==FAM.choose(geo,j,v,98)
    inp=input_for(g,x,y,ctx,q,m[0],999);assert not set(ctx)&set(q) and not set(m[0])&(set(ctx)|set(q))
    assert not hasattr(inp,'labels_query') and not hasattr(inp,'hidden_values')
    info={'posterior_direct_error':float(np.abs(w[0]-direct/direct.sum()).max()),'class_swap_error':float(np.abs(p+rev-1).max()),
          'latent_states':len(g['mu']),'score_samples':c['score_samples'],'audit_samples':c['score_audit_samples'],
          'J':j.tolist(),'V':v.tolist(),'V_se':se.tolist(),'audit_J':jj.tolist(),'audit_V':vv.tolist(),
          'J_rank_correlation':float(np.corrcoef(np.argsort(np.argsort(j)),np.argsort(np.argsort(jj)))[0,1]),
          'V_rank_correlation':float(np.corrcoef(np.argsort(np.argsort(v)),np.argsort(np.argsort(vv)))[0,1]),
          'J_V_rank_correlation':float(np.corrcoef(np.argsort(np.argsort(jj)),np.argsort(np.argsort(vv)))[0,1]),'models':{}}
    digests=[]
    for interface in c['interfaces']:
        model=C.Model('pcfi_plain' if interface=='pcfi' else 'mask_plain',11);digests.append(C.parameter_digest(model,trainable=True))
        a=C.predict(model,inp);assert np.array_equal(a,C.predict(model,copy.deepcopy(inp)))
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y[q],device='cuda'))
        loss.backward();grad=sum(float(z.grad.abs().sum()) for z in model.parameters() if z.requires_grad and z.grad is not None)
        assert math.isfinite(grad) and grad>0
        info['models'][interface]={'trainable':sum(z.numel() for z in model.parameters() if z.requires_grad),'gradient_l1':grad}
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(set(digests))==1;info['common_initial_sha']=digests[0];info['seconds']=time.perf_counter()-t0
    write(ROOT/'results/preflight.json',info);lock(create=True);print('PREFLIGHT_OK',json.dumps(info),flush=True)
def prepare():
    lock();assert not (ROOT/'bank/manifest.json').exists();c=cfg();rows=[];design_rows=[];cost={};t0=time.perf_counter()
    for split in ['train','test']:
        (ROOT/'bank'/split).mkdir(parents=True,exist_ok=True);start=time.perf_counter();scoring=0.;pcfi_time=0.
        for gid in range(c[split+'_graphs']):
            seed=c[split+'_seed_start']+gid;g=FAM.graph(seed)
            # Independent from all mask-value simulation streams, shared by every arm/cell.
            xx,yy,hh=FAM.draw(g,seed+80000000);x,y=xx[0],yy[0]
            for k in c['contexts']:
                for b in c['budgets']:
                    ds=seed+1000000+k*10000+b*100;ctx,q,m,geo=FAM.designs(g,ds,k,b)
                    if split=='train':
                        ts=time.perf_counter();j,v,se=FAM.scores(g,ctx,q,m,ds+20000000);scoring+=time.perf_counter()-ts
                        selection=FAM.choose(geo,j,v,ds+30000000)
                        design_rows.append({'graph':gid,'context':k,'budget':b,'J':j.tolist(),'V':v.tolist(),'V_se':se.tolist(),
                            'geometry':geo.tolist(),'masks':m.tolist(),'selected':selection,'score_seed':ds+20000000,'actual_seed':seed+80000000})
                    else:selection={'test':0}
                    for prior,index in selection.items():
                        if prior=='random_cost':continue
                        ts=time.perf_counter();inp=input_for(g,x,y,ctx,q,m[index],ds+40000000);pcfi_time+=time.perf_counter()-ts
                        name=f'{split}/g{gid}_c{k}_o{b}_{prior}.npz';save_episode(ROOT/'bank'/name,inp,y)
                        rows.append(dict(path=name,split=split,graph=gid,context=k,budget=b,prior=prior,condition='uniform',sha256=C.sha(ROOT/'bank'/name)))
            if split=='test':
                ctx,q,m,_=FAM.designs(g,seed+1000000+8*10000+4*100,8,4)
                pool=np.setdiff1d(np.arange(c['nodes']),np.r_[ctx,q]);rng=np.random.default_rng(seed+90000000);root=int(rng.choice(pool))
                order=np.lexsort((rng.random(len(pool)),g['distance'][root,pool]));o=pool[order[:4]]
                for condition,o in [('clustered',o),('full',np.arange(c['nodes']))]:
                    inp=input_for(g,x,y,ctx,q,o,seed+70000000);name=f'test/g{gid}_{condition}.npz';save_episode(ROOT/'bank'/name,inp,y)
                    rows.append(dict(path=name,split=split,graph=gid,context=8,budget=len(o),prior='test',condition=condition,sha256=C.sha(ROOT/'bank'/name)))
                np.savez_compressed(ROOT/'bank/test'/f'g{gid}_reference.npz',mu=g['mu'],py=g['py'],x=x,y=y)
            if (gid+1)%16==0:print('PREPARE',split,gid+1,'scoring_s',round(scoring,2),'elapsed',round(time.perf_counter()-start,1),flush=True)
        cost[split]={'total_seconds':time.perf_counter()-start,'cold_scoring_seconds':scoring,'pcfi_seconds':pcfi_time}
    write(ROOT/'bank/designs.json',design_rows);write(ROOT/'bank/manifest.json',{'episodes':rows,'cost':cost,'sources':sources()})
    print('PREPARED',len(rows),'seconds',round(time.perf_counter()-t0,1),flush=True)
def rowset(prior=None):
    rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    return [r for r in rows if r['split']==('train' if prior else 'test') and (not prior or r['prior']==('random' if prior=='random_cost' else prior))]
def schedule(seed,steps):
    # Same graph/condition schedule for all priors/interfaces; longer control extends it.
    rng=np.random.default_rng(seed+40000);cells=[(k,b) for k in cfg()['contexts'] for b in cfg()['budgets']];out=[]
    for block in range(steps//len(cells)):
        for cell in rng.permutation(len(cells)):
            k,b=cells[cell];out.append((int(rng.integers(cfg()['train_graphs'])),k,b))
    return out
def intervene(inp,mode):
    out=copy.deepcopy(inp)
    if mode=='no_features':out.observed[:]=False;out.observed_values[:]=0;out.pcfi[:]=0
    elif mode=='shuffled':
        o=np.flatnonzero(out.observed.any(1));perm=np.random.default_rng(inp.forward_seed+331).permutation(o)
        out.observed_values[o]=out.observed_values[perm]
        out.pcfi=C.BASE.fill(np.where(out.observed,out.observed_values,np.nan),out.src,out.dst,'pcfi','uniform_nodes').astype(np.float32)
    return out
def load(prior,interface,seed):
    model=C.Model('pcfi_plain' if interface=='pcfi' else 'mask_plain',seed)
    state=torch.load(ROOT/'results'/f'{prior}_{interface}_s{seed}'/'trainable.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:p.copy_(state['state_dict'][n].to(p.device))
    return model
def evaluate(model,prior,interface,seed,out):
    rows=rowset();records=[];ps=[];ys=[];t0=time.perf_counter()
    for row in rows:
        inp,y=read(row);modes=['standard']
        if row['condition']=='uniform' and row['context']==8 and row['budget']==4:modes+=['no_features','shuffled']
        for mode in modes:
            p=C.predict(model,intervene(inp,mode));ps.append(p);ys.append(y)
            records.append({**row,'mode':mode,**C.metrics(p,y)})
    np.savez_compressed(out/'predictions.npz',p=np.array(ps),y=np.array(ys))
    write(out/'evaluation.json',records);print('EVALUATED',prior,interface,seed,len(records),round(time.perf_counter()-t0,1),flush=True)
def run(seed):
    c=cfg();meta=lock();bank_sha=C.sha(ROOT/'bank/manifest.json')
    for interface in c['interfaces']:
        for prior in c['priors']:
            out=ROOT/'results'/f'{prior}_{interface}_s{seed}';out.mkdir(exist_ok=True)
            if (out/'evaluation.json').exists():continue
            if (out/'training.json').exists():
                model=load(prior,interface,seed);evaluate(model,prior,interface,seed,out);del model;gc.collect();torch.cuda.empty_cache();continue
            assert not (out/'progress.json').exists(),'Partial training: preserve and audit explicitly'
            model=C.Model('pcfi_plain' if interface=='pcfi' else 'mask_plain',seed)
            rows=rowset(prior);cache={(r['graph'],r['context'],r['budget']):read(r) for r in rows}
            steps=c['cost_updates'] if prior=='random_cost' else c['updates'];sched=schedule(seed,steps)
            initial=C.parameter_digest(model,trainable=True);frozen=C.parameter_digest(model,trainable=False)
            optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':c['base_lr']},
                {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':c['new_lr']}],weight_decay=c['weight_decay'])
            scaler=torch.amp.GradScaler('cuda',init_scale=128.);logs=[];overflows=[];started=time.perf_counter();model.train()
            for step,key in enumerate(sched):
                inp,y=cache[key];mult=min((step+1)/16,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
                for group in optimizer.param_groups:group['lr']=group['base_lr']*mult
                for attempt in range(24):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
                    assert torch.isfinite(loss);scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    params=[p for p in model.parameters() if p.requires_grad]
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                        overflows.append({'step':step,'attempt':attempt,'scale':scale});scaler.update(new_scale=scale/2);continue
                    norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True);scaler.step(optimizer);scaler.update();break
                else:raise RuntimeError('Finite gradient retries exhausted')
                logs.append({'step':step+1,'design':key,'loss':float(loss.detach()),'gradient_norm':float(norm),'scale':scaler.get_scale(),'seconds':time.perf_counter()-started})
                if (step+1)%48==0:
                    write(out/'progress.json',logs[-1]);print('TRAIN',prior,interface,seed,step+1,'loss',round(float(np.mean([z['loss'] for z in logs[-48:]])),5),'s',round(time.perf_counter()-started,1),flush=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            assert C.parameter_digest(model,trainable=False)==frozen and C.parameter_digest(model,trainable=True)!=initial
            torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'prior':prior,'interface':interface,'seed':seed,'metadata':meta,'bank_sha':bank_sha},out/'trainable.pt')
            write(out/'train_log.json',logs);write(out/'overflow_retries.json',overflows)
            write(out/'training.json',dict(prior=prior,interface=interface,seed=seed,updates=steps,seconds=seconds,
                schedule_sha=hashlib.sha256(json.dumps(sched).encode()).hexdigest(),initial_sha=initial,frozen_sha=frozen,
                frozen_unchanged=True,trainable=sum(p.numel() for p in params),final_sha=C.parameter_digest(model,trainable=True),
                checkpoint_sha=C.sha(out/'trainable.pt'),bank_sha=bank_sha,overflow_retries=len(overflows)))
            evaluate(model,prior,interface,seed,out);del model,optimizer,cache;gc.collect();torch.cuda.empty_cache()
def reference():
    lock();records=[];pp=[];yy=[]
    for row in rowset():
        inp,y=read(row)
        with np.load(ROOT/'bank/test'/f'g{row["graph"]}_reference.npz') as d:g={'mu':d['mu'],'py':d['py']}
        o=np.flatnonzero(inp.observed.any(1));x=inp.observed_values
        for name,use in [('exact_family',True),('exact_no_features',False),('context_frequency',False)]:
            if name=='context_frequency':p=np.full(len(y),(inp.labels_context.sum()+1)/(len(inp.labels_context)+2))
            else:p=FAM.predict(g,inp.context,inp.labels_context,inp.query,o if use else None,x if use else None)[0]
            p=np.stack([1-p,p],-1);pp.append(p);yy.append(y)
            records.append({**row,'arm':name,**C.metrics(p,y),'expected_accuracy':float(p.max(1).mean())})
    np.savez_compressed(ROOT/'results/reference_predictions.npz',p=np.array(pp),y=np.array(yy));write(ROOT/'results/reference.json',records)
    print('REFERENCE_OK',len(records),flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','prepare','run','reference']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    C.setup();torch.set_num_threads(4)
    if a.action=='run':assert a.seed in cfg()['seeds'];run(a.seed)
    else:globals()[a.action]()
if __name__=='__main__':main()

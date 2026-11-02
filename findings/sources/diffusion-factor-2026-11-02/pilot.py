"""Frozen Factor target; small conditional diffusion/SMC inference pilot."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;OLD=WORK/'disjoint_factor_pilot'
sys.path.insert(0,str(OLD))
import data as D
from model import FactorPFN
sys.path.insert(0,str(ROOT))
from sampler import Denoiser,energy,proposal,base_draw,mh_diff,sample
import argparse,copy,gc,hashlib,itertools,json,math,time
import numpy as np
import torch
from torch.nn import functional as F
C=D.C

def cfg():return json.loads((ROOT/'protocol.json').read_text())
def dump(path,obj):D.dump(path,obj)
def source_hashes():
    paths=[ROOT/'sampler.py',ROOT/'pilot.py',ROOT/'protocol.json']+[OLD/p for p in ['data.py','model.py','protocol.json']]+[WORK/'direct_marginal_pilot/data.py']
    return {p.relative_to(WORK).as_posix():D.sha(p) for p in paths}
def checkpoint_files():
    return [OLD/'results'/f'{arm}_s{s}'/name for arm in ['factor_pfn','mask_pfn','pcfi_pfn'] for s in cfg()['seeds'] for name in ['checkpoint.pt','training.json']]
def lock(create=False):
    state={'sources':source_hashes(),'checkpoints':{p.relative_to(WORK).as_posix():D.sha(p) for p in checkpoint_files()},
           'historical_train_manifest':D.sha(OLD/'bank/manifest.json')}
    path=ROOT/'results/lock.json'
    if create:assert not path.exists();dump(path,state)
    else:assert json.loads(path.read_text())==state,'Frozen source/checkpoint changed'
    return state
def load_factor(seed,constant=False):
    model=FactorPFN(constant=constant).cuda()
    if not constant:
        out=OLD/'results'/f'factor_pfn_s{seed}';info=json.loads((out/'training.json').read_text());assert D.sha(out/'checkpoint.pt')==info['checkpoint_sha256']
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
        with torch.no_grad():
            for n,p in model.named_parameters():p.copy_(saved[n].to(p.device))
        assert C.parameter_digest(model,trainable=True)==info['final_sha256']
    for p in model.parameters():p.requires_grad=False
    return model.eval()
def condition(model,items):
    batch=D.tensors(items)
    with torch.inference_mode():j,valid,context,const=model.potentials(batch)
    pri=j.new_tensor([[.25,.25,.25,.25],[0,.5,.5,0],[.5,0,0,.5]])
    mass=batch['prior']@pri;n=batch['x'].shape[1];f=batch['x'].shape[-1]
    types=torch.cat((batch['type'].long(),torch.full((len(items),f),2,device=j.device,dtype=torch.long)),1)
    return {'j':j,'const':const,'mass':mass,'context':context,'types':types}
def spin_table(l,device='cuda',dtype=torch.float64):
    return (2*((torch.arange(2**l,device=device)[:,None]>>torch.arange(l,device=device))&1)-1).to(dtype)
def exact(c,draws=0,seed=0):
    j=c['j'].double();const=c['const'].double();context=c['context'];states=spin_table(j.shape[-1])
    score=.5*torch.einsum('si,brij,sj->brs',states,j,states)+const[:,:,None]+c['mass'].double().log()[:,:,None]
    okay=((context[:,None]<0)|(context[:,None]==((states+1)/2)[None])).all(-1)
    score=score.masked_fill(~okay[:,None],-torch.inf);w=score.flatten(1).softmax(-1)
    wstate=w.reshape_as(score).sum(1);p1=wstate@((states+1)/2)
    out={'p1':p1.float(),'role':w.reshape_as(score).sum(-1).float()}
    if draws:
        gen=torch.Generator(device=j.device).manual_seed(seed);ids=torch.multinomial(w,draws,replacement=True,generator=gen)
        out['draw_h']=(ids//len(states)).long();out['draw_s']=states[ids%len(states)].float()
    return out
def subset(c,ids):return {k:v[ids] for k,v in c.items()}
def save_conditions(path,c,teacher):
    np.savez_compressed(path,**{k:v.cpu().numpy() for k,v in c.items()},**{k:v.cpu().numpy() for k,v in teacher.items()})
def read_conditions(path):
    with np.load(path) as d:
        c={k:torch.as_tensor(d[k],device='cuda',dtype=torch.long if k in ['context','types'] else torch.float32) for k in ['j','const','mass','context','types']}
        extra={k:d[k] for k in d.files if k not in c}
    return c,extra
def read_test(row):
    with np.load(ROOT/'bank'/row['path']) as z:return {k:z[k] for k in D.INPUT_KEYS},z['query'],z['labels_query']
def test_rows():return json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
def dev_condition():
    master=D.NATIVE.master(18300001,'exclusive');inp,q,y=D.observation(master,2,6,'disjoint')
    return inp,q,y,condition(load_factor(11),[inp])

def preflight():
    c=cfg();inp,q,y,cond=dev_condition();torch.manual_seed(991);model=Denoiser(c['width']).cuda().eval()
    # Exact4-spin/2-role toy, full proposal transition and detailed balance matrix.
    j=cond['j'][:,:,:4,:4].clone();j[:]=0
    for i,k,v in [(0,1,.4),(1,2,-.3),(2,3,.8)]:j[:,:,i,k]=j[:,:,k,i]=v
    toy={'j':j,'const':j.new_tensor([[.1,-.2,.3,-.1]]),'mass':j.new_tensor([[.5,0,0,.5]]),
         'context':torch.full((1,4),-1,device='cuda',dtype=torch.long),'types':torch.zeros((1,4),device='cuda',dtype=torch.long)}
    spins=spin_table(4,dtype=torch.float32);ss=spins.repeat(2,1)[None];hh=torch.cat((torch.zeros(16),torch.full((16,),3))).long().cuda()[None]
    t=j.new_full((1,32),4);ps,ph=proposal(model,toy,ss,hh,t,8)
    target=exact(toy)['p1'];assert torch.isfinite(target).all()
    Q=[]
    for k in range(32):
        prob=ps[0].gather(-1,((ss[0,k]+1)/2).long()[None,:,None].expand(32,-1,-1)).squeeze(-1).prod(1)*ph[0,:,hh[0,k]]
        Q.append(prob)
    Q=torch.stack(Q,1).double();assert float((Q.sum(1)-1).abs().max())<2e-6
    ee=energy(toy['j'],toy['const'],ss,hh)[0].double();lp=.6*ee+toy['mass'][0,hh[0]].double().log();pi=lp.softmax(0)
    acc=(pi[:,None]*Q).T/(pi[:,None]*Q).clamp_min(1e-30);acc=acc.clamp_max(1)
    P=Q*acc;P=P+torch.diag(1-P.sum(1));balance=float((pi[:,None]*P-(pi[:,None]*P).T).abs().max());assert balance<1e-10
    # Independent direct tensor energy computation includes full role constants.
    manual=[]
    for state,hyp in zip(ss[0],hh[0]):manual.append(.5*state@toy['j'][0,hyp]@state+toy['const'][0,hyp])
    assert float((ee-torch.stack(manual).double()).abs().max())<1e-6
    teacher=exact(cond,16,333);s=teacher['draw_s'];h=teacher['draw_h'];t=cond['j'].new_full(h.shape,4)
    p1,ph=model(cond,s,h,t,8);flip={**cond,'context':torch.where(cond['context']>=0,1-cond['context'],cond['context'])}
    fp,fh=model(flip,-s,h,t,8);class_error=float(torch.where(cond['context'][:,None]<0,(p1+fp-1).abs(),0).max());assert class_error<2e-6
    assert float((ph-fh).abs().max())<2e-6
    perm=torch.randperm(s.shape[-1],device='cuda');pc={**cond,'j':cond['j'][:,:,perm][:,:,:,perm],'context':cond['context'][:,perm],'types':cond['types'][:,perm]}
    pp,pr=model(pc,s[:,:,perm],h,t,8);perm_error=float((pp-p1[:,:,perm]).abs().max());assert perm_error<2e-6
    model.train();before=C.parameter_digest(model)
    loss=F.binary_cross_entropy(p1.clamp(1e-6,1-1e-6),((s+1)/2))+F.nll_loss(ph.flatten(0,1).clamp_min(1e-20).log(),h.flatten())
    opt=torch.optim.AdamW(model.parameters(),lr=.001);opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();assert C.parameter_digest(model)!=before
    # Time-only choice; no quality values inspected for selecting a cost control.
    model.eval();timings={}
    for arm,sweeps in [('diffusion_smc_fixed',1)]+[('smc_gibbs_cost',k) for k in range(1,17)]:
        times=[]
        for repeat in range(3):
            torch.cuda.synchronize();start=time.perf_counter();sample(model,cond,arm,11+repeat,particles=c['particles'],stages=c['stages'],sweeps=sweeps)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        timings[f'{arm}_{sweeps}']=float(np.median(times))
    neural=timings['diffusion_smc_fixed_1'];chosen=min(range(1,17),key=lambda k:abs(timings[f'smc_gibbs_cost_{k}']-neural))
    assert c['gibbs_cost_sweeps'] is None;c['gibbs_cost_sweeps']=chosen;dump(ROOT/'protocol.json',c)
    dump(ROOT/'results/preflight.json',{'detailed_balance_error':balance,'proposal_row_sum_error':float((Q.sum(1)-1).abs().max()),
        'class_flip_error':class_error,'node_permutation_error':perm_error,'finite_gradients_and_actual_update':True,'timings_seconds':timings,
        'chosen_cost_sweeps':chosen,'denoiser_parameters':sum(p.numel() for p in model.parameters())})
    lock(create=True);print('PREFLIGHT_OK','sweeps',chosen,'parameters',sum(p.numel() for p in model.parameters()),'timings',timings,flush=True)

def prepare():
    lock();c=cfg();assert not (ROOT/'bank/manifest.json').exists();rows=[];start=time.perf_counter();(ROOT/'bank/test').mkdir(parents=True,exist_ok=True)
    for pid,prior in enumerate(D.NATIVE.NAMES):
        for local in range(c['test_graphs_per_prior']):
            gid=pid*c['test_graphs_per_prior']+local;seed=c['test_seed_start']+gid;case=['matched','matched','density_low','signal_high'][local%4]
            master=D.NATIVE.master(seed,prior,case)
            for k in c['contexts']:
                for count in c['observed_nodes']:
                    inp,q,y=D.observation(master,k,count,'disjoint');path=f'test/g{gid}_k{k}_o{count}.npz'
                    np.savez_compressed(ROOT/'bank'/path,**inp,query=q,labels_query=y)
                    rows.append(dict(path=path,graph=gid,seed=seed,prior=prior,case=case,context_total=k,observed_nodes=count,sha256=D.sha(ROOT/'bank'/path)))
    dump(ROOT/'bank/manifest.json',{'episodes':rows,'sources':source_hashes(),'prepare_seconds':time.perf_counter()-start})
    inputs=[read_test(r)[0] for r in rows];train=[D.read(r['path'])[0] for r in D.rows('train')[:c['train_tasks']]]
    for seed in c['seeds']:
        frozen=load_factor(seed);allc=[];allteach=[];start=time.perf_counter()
        for offset in range(0,len(train),16):
            cond=condition(frozen,train[offset:offset+16]);teacher=exact(cond,c['teacher_draws'],seed*100000+offset)
            allc.append(cond);allteach.append(teacher)
        cond={key:torch.cat([x[key] for x in allc]) for key in allc[0]};teacher={key:torch.cat([x[key] for x in allteach]) for key in allteach[0]}
        save_conditions(ROOT/'bank'/f'train_s{seed}.npz',cond,teacher);elapsed=time.perf_counter()-start
        allc=[];allteach=[];bp=[]
        for offset in range(0,len(inputs),16):
            items=inputs[offset:offset+16];cond=condition(frozen,items);teacher=exact(cond)
            allc.append(cond);allteach.append(teacher)
            with torch.inference_mode():bp.append(frozen(D.tensors(items)).cpu().numpy())
        cond={key:torch.cat([x[key] for x in allc]) for key in allc[0]};teacher={key:torch.cat([x[key] for x in allteach]) for key in allteach[0]}
        save_conditions(ROOT/'bank'/f'test_s{seed}.npz',cond,teacher);np.savez_compressed(ROOT/'results'/f'factor_bp_s{seed}.npz',p=np.concatenate(bp))
        dump(ROOT/'results'/f'preparation_s{seed}.json',{'teacher_train_seconds':elapsed,'teacher_tasks':len(train),'draws_per_task':c['teacher_draws'],'frozen_checkpoint_sha':D.sha(OLD/'results'/f'factor_pfn_s{seed}'/'checkpoint.pt')})
        print('TARGET_PREPARED',seed,flush=True)
    reference=D.NATIVE.GridReference('cuda');records=[]
    for off in range(0,len(inputs),16):
        with torch.inference_mode():records.append(reference.infer(inputs[off:off+16]))
    np.savez_compressed(ROOT/'results/grid_reference.npz',**{key:np.concatenate([x[key] for x in records]) for key in records[0]})
    dump(ROOT/'results/bank_hashes.json',{p.relative_to(ROOT).as_posix():D.sha(p) for p in (ROOT/'bank').rglob('*.npz')})
    print('PREPARED',len(rows),'seconds',round(time.perf_counter()-start,2),flush=True)

def train(seed):
    c=cfg();lock();out=ROOT/'results'/f'denoiser_s{seed}';out.mkdir(exist_ok=True)
    assert not (out/'checkpoint.pt').exists() and not (out/'progress.json').exists(),'Preserve completed/partial training'
    cond,extra=read_conditions(ROOT/'bank'/f'train_s{seed}.npz');clean=torch.as_tensor(extra['draw_s'],device='cuda');roles=torch.as_tensor(extra['draw_h'],device='cuda',dtype=torch.long)
    torch.manual_seed(seed);model=Denoiser(c['width']).cuda();initial=C.parameter_digest(model)
    opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);gen=torch.Generator(device='cuda').manual_seed(seed+59000)
    logs=[];start=time.perf_counter()
    for step in range(c['updates']):
        idx=torch.randint(len(clean),(c['batch_size'],),device='cuda',generator=gen);mc=torch.randint(c['teacher_draws'],(c['batch_size'],),device='cuda',generator=gen)
        given=subset(cond,idx);s0=clean[idx,mc][:,None];h0=roles[idx,mc][:,None]
        t=torch.randint(1,c['stages']+1,(c['batch_size'],1),device='cuda',generator=gen).float();retain=1-t/c['stages']
        sn,hn=base_draw(given,1,gen);noisy=torch.where(torch.rand(s0.shape,device='cuda',generator=gen)<retain[...,None],s0,sn)
        nh=torch.where(torch.rand(h0.shape,device='cuda',generator=gen)<retain,h0,hn)
        p1,ph=model(given,noisy,nh,t,c['stages']);unknown=given['context'][:,None]<0
        spin_loss=F.binary_cross_entropy(p1.clamp(1e-7,1-1e-7),(s0+1)/2,reduction='none')
        loss=(spin_loss*unknown).sum()/unknown.sum()+F.nll_loss(ph.flatten(0,1).clamp_min(1e-30).log(),h0.flatten())
        opt.zero_grad();loss.backward();assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        logs.append({'step':step+1,'loss':float(loss.detach()),'seconds':time.perf_counter()-start})
        if (step+1)%256==0:dump(out/'progress.json',logs[-1]);print('TRAIN',seed,step+1,'loss',round(np.mean([x['loss'] for x in logs[-256:]]),4),'s',round(time.perf_counter()-start,1),flush=True)
    final=C.parameter_digest(model);assert initial!=final
    torch.save(model.state_dict(),out/'checkpoint.pt');dump(out/'training.json',{'seed':seed,'updates':len(logs),'initial_sha':initial,'final_sha':final,
        'checkpoint_sha':D.sha(out/'checkpoint.pt'),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start})
    dump(out/'train_log.json',logs)

def evaluate(seed):
    c=cfg();lock();out=ROOT/'results'/f'denoiser_s{seed}';model=Denoiser(c['width']).cuda().eval()
    model.load_state_dict(torch.load(out/'checkpoint.pt',map_location='cuda',weights_only=True));cond,extra=read_conditions(ROOT/'bank'/f'test_s{seed}.npz')
    for arm in c['arms']:
        path=out/f'{arm}.npz'
        if path.exists():continue
        pp=[];raw=[];infos={};seconds=0.
        for off in range(0,len(cond['j']),16):
            cc=subset(cond,slice(off,off+16));torch.cuda.synchronize();start=time.perf_counter()
            p,pr,diag=sample(model,cc,arm,seed*100000+off,particles=c['particles'],stages=c['stages'],
                           sweeps=c['gibbs_cost_sweeps'] if arm=='smc_gibbs_cost' else 1,cess=c['cess_fraction'])
            torch.cuda.synchronize();seconds+=time.perf_counter()-start;pp.append(p.cpu().numpy());raw.append(pr.cpu().numpy())
            for key,value in diag.items():infos.setdefault(key,[]).append(value.cpu().numpy())
        np.savez_compressed(path,p1=np.concatenate(pp),raw=np.concatenate(raw),**{key:np.concatenate(value) for key,value in infos.items()})
        dump(out/f'{arm}_time.json',{'seconds':seconds,'episodes':len(cond['j']),'batch_size':16,'includes':'SMC/proposal/MH/RB;cached frozen potentials;factor extraction reported separately'})
        print('EVALUATED',seed,arm,'s',round(seconds,2),flush=True)

def public_references():
    lock();rows=test_rows();items=[read_test(r) for r in rows]
    for seed in cfg()['seeds']:
        for arm in ['mask_pfn','pcfi_pfn']:
            path=ROOT/'results'/f'{arm}_s{seed}.npz'
            if path.exists():continue
            model=C.Model('mask_plain' if arm=='mask_pfn' else 'pcfi_plain',seed)
            saved=torch.load(OLD/'results'/f'{arm}_s{seed}'/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if p.requires_grad:p.copy_(saved[name].to(p.device))
            probs=[]
            for i,(inp,q,y) in enumerate(items):
                p=C.predict(model,D.public_input(inp,q,18700000+i));padded=np.full((8,2),np.nan);padded[q]=p;probs.append(padded)
            np.savez_compressed(path,p=np.array(probs));del model;gc.collect();torch.cuda.empty_cache();print('PUBLIC',arm,seed,flush=True)
    constant=load_factor(11,True);pp=[]
    for off in range(0,len(items),16):
        with torch.inference_mode():pp.append(constant(D.tensors([x[0] for x in items[off:off+16]])).cpu().numpy())
    np.savez_compressed(ROOT/'results/constant_bp.npz',p=np.concatenate(pp))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','prepare','train','evaluate','public_references','all']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    C.setup();torch.set_num_threads(2)
    if a.action in ['train','evaluate']:globals()[a.action](a.seed)
    elif a.action=='all':
        prepare()
        for seed in cfg()['seeds']:train(seed);evaluate(seed)
        public_references()
    else:globals()[a.action]()
if __name__=='__main__':main()

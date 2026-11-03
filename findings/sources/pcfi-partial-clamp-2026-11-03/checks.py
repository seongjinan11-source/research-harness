import sys
sys.dont_write_bytecode = True
import sklearn
import gc
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,WORK,B,R,C,cfg,dump,sha,Model,load_episode,node_theta,transport_forward,smooth_gain,lock


def aligned_reference(x,obs,ex,theta,steps):
    c,s = theta.cos(),theta.sin()
    a,b = x*c,x*s
    ii,jj,w = ex['ii'],ex['jj'],ex['weights']
    for _ in range(steps):
        m0 = torch.zeros_like(a).index_add(0,ii,w*a[jj])
        m1 = torch.zeros_like(b).index_add(0,ii,w*b[jj])
        residual = torch.where(obs,x-c*m0-s*m1,0.)
        a,b = m0+c*residual,m1+s*residual
    return torch.where(obs,x,B.correlation(c*a+s*b,ex['pc']))


def cpu_checks(inp,extra):
    torch.set_num_threads(2)
    ex = {k:v.double() if v.is_floating_point() else v for k,v in B.tensor_extra(extra,'cpu').items()}
    x,obs = torch.as_tensor(inp.observed_values,dtype=torch.float64),torch.as_tensor(inp.observed)
    front = B.Front(11).double()
    theta = node_theta(front,ex)
    full,_,_ = transport_forward(front,x,obs,ex,partial=False,return_state=True)
    old,_ = front(x,obs,ex,'node')
    full_error = float((full-old).detach().abs().max())
    assert full_error<1e-12
    partial,diag = transport_forward(front,x,obs,ex,partial=True)
    aligned = aligned_reference(x,obs,ex,theta,100)
    error = float((partial-aligned).detach().abs().max())
    assert error<1e-10,error
    probe = torch.linspace(-.7,.9,x.numel(),dtype=x.dtype).reshape_as(x)
    params = tuple(front.parameters())
    g0 = torch.autograd.grad((partial*probe).sum(),params)
    g1 = torch.autograd.grad((aligned*probe).sum(),params)
    gradient_error = max(float((a-b).abs().max()) for a,b in zip(g0,g1))
    assert gradient_error<1e-9,gradient_error
    assert sum(float(g.abs().sum()) for g in g0)>0
    for mask in [obs,torch.ones_like(obs),torch.zeros_like(obs)]:
        values = torch.where(mask,x,0.)
        for partial_mode in [False,True]:
            out,info = transport_forward(front,values,mask,ex,partial=partial_mode)
            assert torch.isfinite(out).all()
            assert torch.equal(out[mask],values[mask])
            if not mask.any():
                assert torch.count_nonzero(out)==0
    identity,_ = transport_forward(front,x,obs,ex,partial=True,theta=torch.zeros_like(x))
    pcfi,_ = front(x,obs,ex,'pcfi')
    assert torch.equal(identity,pcfi)
    gain = smooth_gain(front,ex)
    assert gain.abs().max()<=1
    assert torch.equal(gain[ex['ii']==ex['jj']],torch.ones_like(gain[ex['ii']==ex['jj']]))
    # Independent tiny complete graph: release of partial observations changes the estimator.
    tx = torch.tensor([[1.],[1.],[0.]],dtype=torch.float64)
    to = torch.tensor([[True],[True],[False]])
    ti,tj = torch.meshgrid(torch.arange(3),torch.arange(3),indexing='ij')
    te = {'ii':ti.flatten(),'jj':tj.flatten(),'weights':torch.full((9,1),1/3,dtype=tx.dtype),'pc':torch.ones_like(tx)}
    tt = torch.tensor([[0.],[math.pi/2],[math.pi/4]],dtype=tx.dtype)
    full,_,_ = transport_forward(None,tx,to,te,partial=False,theta=tt,steps=1000,return_state=True)
    part,td,(h0,h1) = transport_forward(None,tx,to,te,partial=True,theta=tt,steps=1000,return_state=True)
    assert abs(float(full[2])-1/math.sqrt(2))<1e-10
    assert abs(float(part[2])-math.sqrt(2))<1e-9
    assert float(h1[to].abs().min())>.99
    return {'full_matches_old_max_abs':full_error,'partial_aligned_max_abs':error,
        'partial_aligned_gradient_max_abs':gradient_error,'identity_is_pcfi_exact':True,
        'toy_full_query':float(full[2]),'toy_partial_query':float(part[2]),
        'toy_observed_second_mean_abs':td['observed_second_mean_abs'],
        'toy_note':'Analytic construction demonstrates a different function; not node-classification improvement.'}


def main():
    C.setup()
    torch.set_num_threads(2)
    rows = json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row = next(r for r in rows if r['split']=='development' and r['case']=='uniform90' and r['context_per_class']==4 and r['label_mode']=='hetero')
    inp,extra,y = load_episode(row)
    checks = cpu_checks(inp,extra)
    print('CPU CHECKS PASS',checks,flush=True)
    smoke = []
    for seed in cfg()['seeds']:
        original = json.loads((B.ROOT/'results'/f'node_s{seed}'/'training.json').read_text())
        for arm in cfg()['learned_arms']:
            obj = Model(arm,seed)
            initial,pfn = C.parameter_digest(obj.front),C.parameter_digest(obj.pfn)
            assert initial==original['initial_front_sha256']
            assert pfn==original['pfn_sha256']
            assert sum(p.numel() for p in obj.parameters() if p.requires_grad)==737
            obj.train()
            optimizer = torch.optim.AdamW(obj.front.parameters(),lr=.001,weight_decay=.01)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = F.cross_entropy(obj(inp,extra),torch.as_tensor(y,device='cuda'))
            loss.backward()
            grads = [p.grad for p in obj.front.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads)
            magnitude = sum(float(g.abs().sum()) for g in grads)
            assert magnitude>0 and all(p.grad is None for p in obj.pfn.parameters())
            torch.nn.utils.clip_grad_norm_(obj.front.parameters(),1.)
            optimizer.step()
            assert C.parameter_digest(obj.front)!=initial
            assert C.parameter_digest(obj.pfn)==pfn
            smoke.append({'arm':arm,'seed':seed,'loss':float(loss.detach()),'gradient_abs_sum':magnitude,
                           'discarded_updates':1,'pfn_unchanged':True})
            print('SMOKE',arm,seed,'PASS',flush=True)
            del obj,optimizer
            gc.collect()
            torch.cuda.empty_cache()
    preserved = {}
    for folder in [B.ROOT,R.ROOT]:
        for path in folder.rglob('*'):
            if path.is_file():
                preserved[path.relative_to(WORK).as_posix()] = sha(path)
    for seed in cfg()['seeds']:
        path = WORK/f'nsd_token_pfn/results/pcfi100_s{seed}/checkpoint.pt'
        preserved[path.relative_to(WORK).as_posix()] = sha(path)
    dump(ROOT/'results/old_files_baseline.json',preserved)
    dump(ROOT/'results/preflight.json',{'status':'PASS','cpu_checks':checks,'smoke':smoke,
          'discarded_updates':len(smoke),'old_files_preserved_baseline':len(preserved)})
    dump(ROOT/'results/lock.json',lock())
    print('PREFLIGHT PASS',len(preserved),flush=True)


if __name__=='__main__':
    main()

import sys
sys.dont_write_bytecode=True
import sklearn
import copy,json,gc
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,Front,Model,predict,load_episode,tensor_extra,geometry,lock,impute,adjacency

def kernel_checks(inp,ex):
    xo=torch.as_tensor(inp.observed_values);obs=torch.as_tensor(inp.observed);te=tensor_extra(ex,'cpu');front=Front(11)
    expected=impute(adjacency((inp.src,inp.dst),len(xo)),np.where(inp.observed,inp.observed_values,np.nan),'pcfi','structural',iterations=100)[0]
    got,_=front(xo,obs,te,'pcfi');error=float(np.max(np.abs(got.detach().numpy()-expected)));assert error<2e-4,error
    reductions={}
    for arm in ['scalar','edge','node']:
        out,info=front(xo,obs,te,arm,identity=True);reductions[arm]=float((out-got).abs().max());assert reductions[arm]==0
        assert info['spatial_steps']==100 and info['channel_corrections']==1 and info['warmup_spatial_steps']==0
        out,info=front(xo,obs,te,arm);assert info['observed_change']==0
    torch.manual_seed(203);perm=torch.randperm(len(xo)).numpy();inverse=np.argsort(perm)
    altered=copy.deepcopy(inp);altered.observed_values=inp.observed_values[perm];altered.observed=inp.observed[perm]
    altered.src=inverse[inp.src];altered.dst=inverse[inp.dst];altered.context=inverse[inp.context];altered.query=inverse[inp.query]
    pe=tensor_extra(geometry(altered),'cpu');errs={}
    for arm in ['edge','node']:
        orig,_=front(xo,obs,te,arm);changed,_=front(xo[perm],obs[perm],pe,arm)
        errs[arm]=float((changed-orig[perm]).abs().max());assert errs[arm]<2e-4
    absent=copy.deepcopy(inp);absent.observed[:]=False;absent.observed_values[:]=0
    ae=tensor_extra(geometry(absent),'cpu')
    for arm in ['pcfi','scalar','edge','node']:
        out,_=front(torch.zeros_like(xo),torch.zeros_like(obs),ae,arm);assert out.abs().max()==0
    # Inspect transport on a complete three-node graph, using only fabricated visible inputs.
    tiny=copy.deepcopy(inp);tiny.observed_values=np.array([[1.,-.7],[.2,.6],[-.4,1.2]],'float32');tiny.observed=np.ones((3,2),bool)
    tiny.src=np.array([0,1,0,2,1,2]);tiny.dst=np.array([1,0,2,0,2,1]);tiny.context=np.array([0,1]);tiny.labels_context=np.array([0,1]);tiny.classes=2
    tiny.query=np.array([2]);tx=tensor_extra(geometry(tiny),'cpu');angles=front.angles(tx,'node');ii,jj=tx['ii'].numpy(),tx['jj'].numpy()
    get=lambda i,j:angles[np.flatnonzero((ii==i)&(jj==j))[0]]
    cycle=get(0,1)+get(1,2)+get(2,0);cycle_error=float(torch.maximum((cycle.cos()-1).abs(),cycle.sin().abs()).max());assert cycle_error<1e-6
    reverse={arm:float(max((front.angles(tx,arm)[np.flatnonzero((ii==i)&(jj==j))[0]]+front.angles(tx,arm)[np.flatnonzero((ii==j)&(jj==i))[0]]).abs().max() for i,j in [(0,1),(1,2),(2,0)])) for arm in ['edge','node']}
    assert max(reverse.values())<1e-6
    return {'pcfi_numpy_max_abs':error,'identity_reduction_max_abs':reductions,'node_permutation_max_abs':errs,
            'no_source_exact_zero':True,'node_cycle_error':cycle_error,'reverse_angle_error':reverse,'observed_clamp_exact':True}

def main():
    C.setup();torch.set_num_threads(2);rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row=next(r for r in rows if r['split']=='dev' and r['case']=='uniform90');inp,ex,y=load_episode(row)
    out=kernel_checks(inp,ex);smoke=[]
    for arm in ['scalar','edge','node']:
        obj=Model(arm,11);pfn_before=C.parameter_digest(obj.pfn);front_before=C.parameter_digest(obj.front)
        assert sum(p.numel() for p in obj.parameters() if p.requires_grad)==737
        obj.train();assert not obj.pfn.training and all(not p.requires_grad for p in obj.pfn.parameters())
        opt=torch.optim.AdamW(obj.front.parameters(),lr=.001)
        with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(obj(inp,ex),torch.as_tensor(y,device='cuda'))
        loss.backward();grad=sum(float(p.grad.abs().sum()) for p in obj.front.parameters() if p.grad is not None)
        assert grad>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.front.parameters())
        assert all(p.grad is None for p in obj.pfn.parameters())
        torch.nn.utils.clip_grad_norm_(obj.front.parameters(),1.);opt.step()
        assert C.parameter_digest(obj.front)!=front_before and C.parameter_digest(obj.pfn)==pfn_before
        p=predict(obj,inp,ex);r=predict(obj,inp,ex)
        smoke.append({'arm':arm,'discarded_optimizer_steps':1,'loss':float(loss.detach()),'frontend_gradient_abs_sum':grad,
                      'pfn_unchanged':True,'repeat_probability_max_abs':float(np.abs(p-r).max())})
        del obj,opt;gc.collect();torch.cuda.empty_cache()
    out.update(status='PASS',smoke_discarded=smoke,protocol_sha256=sha(ROOT/'protocol.json'),all_pfn_weights_fixed=True)
    dump(ROOT/'results/preflight.json',out);dump(ROOT/'results/lock.json',lock());print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()

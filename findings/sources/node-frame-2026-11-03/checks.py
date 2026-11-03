import sys
sys.dont_write_bytecode=True
import sklearn
import copy,gc,json,importlib.util
import numpy as np
import torch
from core import ROOT,WORK,N,C,cfg,dump,sha,rotation,laplacian,NodeFrames,model,predict,load_episode,lock

def operators():
    torch.manual_seed(434);n=7;t=2;c=4
    ii=torch.tensor([0,1,0,2,3,4,4]);jj=torch.tensor([1,2,2,3,4,5,6])
    angle=torch.randn(n,dtype=torch.float64,requires_grad=True);q=rotation(angle)
    x=torch.randn(n,t,2,c,dtype=torch.float64);w=torch.rand(len(ii),dtype=torch.float64)+.2
    got=laplacian(x,ii,jj,w,q[ii],q[jj])
    aligned=torch.einsum('nab,ntbc->ntac',q,x)
    eye=torch.eye(2,dtype=torch.float64).expand(len(ii),2,2)
    expected=torch.einsum('nba,ntbc->ntac',q,laplacian(aligned,ii,jj,w,eye,eye))
    align_error=float((got-expected).abs().max());assert align_error<1e-10
    T01=q[0].T@q[1];T12=q[1].T@q[2];T20=q[2].T@q[0]
    cycle_error=float((T01@T12@T20-torch.eye(2,dtype=torch.float64)).abs().max());assert cycle_error<1e-10
    ortho_error=float((q.transpose(-1,-2)@q-torch.eye(2,dtype=torch.float64)).abs().max());assert ortho_error<1e-10
    z=got.square().sum();gr=torch.autograd.grad(z,angle)[0][0];eps=1e-6
    with torch.no_grad():
        ap=angle.detach().clone();am=ap.clone();ap[0]+=eps;am[0]-=eps
        qp,qm=rotation(ap),rotation(am)
        fd=(laplacian(x,ii,jj,w,qp[ii],qp[jj]).square().sum()-laplacian(x,ii,jj,w,qm[ii],qm[jj]).square().sum())/(2*eps)
    graderror=float((gr-fd).abs());assert graderror<1e-5
    learner=NodeFrames(11).double();h=torch.randn(n,4,192,dtype=torch.float64);meta=torch.randn(n,4,dtype=torch.float64)
    fi,fj=learner.maps(h,ii,jj,meta)
    perm=torch.randperm(n);inv=torch.argsort(perm)
    fp,fjp=learner.maps(h[perm],inv[ii],inv[jj],meta[perm])
    perr=max(float((fp-fi).abs().max()),float((fjp-fj).abs().max()));assert perr<1e-10
    # A single node's frame must be identical across all of its incidences.
    incidence={}
    for e,(i,j) in enumerate(zip(ii.tolist(),jj.tolist())):
        incidence.setdefault(i,[]).append(fi[e]);incidence.setdefault(j,[]).append(fj[e])
    sharing=max(float((v[0]-v[-1]).abs().max()) for v in incidence.values());assert sharing==0
    learned_cycle=float(((fi[0].T@fj[0])@(fi[1].T@fj[1])@(fj[2].T@fi[2])-torch.eye(2,dtype=torch.float64)).abs().max());assert learned_cycle<1e-10
    return {'aligned_linear_operator_error':align_error,'orthogonality_error':ortho_error,'triangle_cycle_error':cycle_error,
            'learned_triangle_cycle_error':learned_cycle,'node_permutation_map_error':perr,'shared_incidence_error':sharing,
            'frame_gradient_finite_difference_error':graderror,'scope':'Fixed-frame linear identities, not nonlinear PFN gauge equivariance.'}

def main():
    C.setup();torch.set_num_threads(2);out=operators()
    rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row=next(r for r in rows if r['split']=='dev' and r['case']=='uniform90');inp,y=load_episode(row)
    obj=model(11);initial=C.parameter_digest(obj,trainable=True)
    info=json.loads((WORK/'nsd_token_pfn/results/dynamic_s11/training.json').read_text())
    assert initial==info['initial_sha256']
    count=sum(p.numel() for p in obj.parameters() if p.requires_grad);assert count==info['trainable_parameters']
    p=predict(obj,inp);diagnostics=copy.deepcopy(obj.last_diagnostics)
    assert max(s['anchor_change'] for s in diagnostics['stages'])==0
    assert max(s['map_refresh_difference'] for s in diagnostics['stages'])>1e-6
    repeat=predict(obj,inp);repeat_error=float(np.abs(p-repeat).max())
    poison=copy.deepcopy(inp);poison.pcfi[:]=12345.;pp=predict(obj,poison)
    poisonerror=float(np.abs(p-pp).max());assert poisonerror<.02
    absent=copy.deepcopy(inp);absent.observed[:]=False;absent.observed_values[:]=0
    pn=predict(obj,absent);assert np.isfinite(pn).all()
    obj.train();opt=torch.optim.AdamW([q for q in obj.parameters() if q.requires_grad],lr=1e-4)
    with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(obj(inp),torch.as_tensor(y,device='cuda'))
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.parameters())
    grad=sum(float(p.grad.abs().sum()) for name,p in obj.nsd.named_parameters() if name.startswith(('project','phi')) and p.grad is not None)
    assert grad>0
    torch.nn.utils.clip_grad_norm_([p for p in obj.parameters() if p.requires_grad],1.);opt.step()
    assert C.parameter_digest(obj,trainable=True)!=initial
    out.update(status='PASS',named_initialization_matches_edge_dynamic=True,trainable_parameters=count,discarded_smoke_steps=1,
               smoke_loss=float(loss.detach()),geometry_gradient_abs_sum=grad,repeat_probability_error=repeat_error,
               unused_pcfi_poison_error=poisonerror,diagnostics=diagnostics,protocol_sha256=sha(ROOT/'protocol.json'))
    dump(ROOT/'results/preflight.json',out)
    dump(ROOT/'results/lock.json',lock());print('PREFLIGHT PASS',flush=True)

if __name__=='__main__':main()

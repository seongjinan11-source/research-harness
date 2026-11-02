import sys
sys.dont_write_bytecode=True
import sklearn
import copy,gc,json
import numpy as np
import torch
from core import ROOT,C,cfg,dump,rotation,laplacian,load_episode,Model,predict
def operator_checks():
    torch.manual_seed(313);n=6;t=3;c=4
    ii=torch.tensor([0,0,1,2,2,4]);jj=torch.tensor([1,2,3,3,4,5]);w=torch.rand(len(ii),dtype=torch.float64)+.2
    a=torch.randn(len(ii),dtype=torch.float64,requires_grad=True);b=torch.randn(len(ii),dtype=torch.float64)
    fi,fj=rotation(a),rotation(b);x=torch.randn(n,t,2,c,dtype=torch.float64)
    z=laplacian(x,ii,jj,w,fi,fj);B=torch.zeros(len(ii)*2,n*2,dtype=torch.float64)
    for e,(i,j) in enumerate(zip(ii,jj)):
        B[2*e:2*e+2,2*i:2*i+2]=w[e].sqrt()*fi[e];B[2*e:2*e+2,2*j:2*j+2]=-w[e].sqrt()*fj[e]
    deg=torch.zeros(n,dtype=torch.float64).index_add(0,ii,w).index_add(0,jj,w)
    K=torch.diag((deg+1).rsqrt().repeat_interleave(2));L=K@B.T@B@K
    vec=x.permute(0,2,1,3).reshape(2*n,-1);dense=(L@vec).reshape(n,2,t,c).permute(0,2,1,3)
    error=float((dense-z).abs().max());assert error<1e-10
    ev=torch.linalg.eigvalsh(L);assert ev.min()>-1e-10 and ev.max()<2+1e-10
    p=torch.randperm(n);inv=torch.argsort(p);zp=laplacian(x[p],inv[ii],inv[jj],w,fi,fj);permutation=float((zp-z[p]).abs().max());assert permutation<1e-10
    gauge=rotation(torch.randn(n,dtype=torch.float64));xg=torch.einsum('nab,ntbc->ntac',gauge,x)
    zg=laplacian(xg,ii,jj,w,fi@gauge[ii].transpose(-1,-2),fj@gauge[jj].transpose(-1,-2))
    gauge_error=float((zg-torch.einsum('nab,ntbc->ntac',gauge,z)).abs().max());assert gauge_error<1e-10
    loss=z.square().sum();gr=torch.autograd.grad(loss,a)[0][0];eps=1e-6
    with torch.no_grad():
        ap=a.detach().clone();am=ap.clone();ap[0]+=eps;am[0]-=eps
        fd=(laplacian(x,ii,jj,w,rotation(ap),fj).square().sum()-laplacian(x,ii,jj,w,rotation(am),fj).square().sum())/(2*eps)
    gradient_error=float((gr-fd).abs());assert gradient_error<1e-5
    eye=rotation(torch.zeros(len(ii),dtype=torch.float64));neg=rotation(torch.full((len(ii),),np.pi,dtype=torch.float64));assert torch.all((eye.transpose(-1,-2)@neg)[:,0,0]<0)
    return {'explicit_incidence_error':error,'minimum_eigenvalue':float(ev.min()),'maximum_eigenvalue':float(ev.max()),'node_permutation_error':permutation,
        'laplacian_gauge_equivariance_error':gauge_error,'map_gradient_finite_difference_error':gradient_error,'negative_transport_representable':True,
        'gauge_scope':'Operator identity under supplied transformed maps; not a claim that the learned map encoder is gauge-equivariant.'}
def main():
    C.setup();torch.set_num_threads(2);out=operator_checks();rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row=next(r for r in rows if r['split']=='dev' and r['case']=='uniform90');inp,y=load_episode(row);smoke=[]
    for kind in cfg()['arms']:
        model=Model(kind,11);model.eval();p=predict(model,inp);original=copy.deepcopy(model.last_diagnostics)
        if kind in ['identity','static','dynamic']:
            assert max(d['anchor_change'] for d in original['stages'])==0
            changes=[d['map_refresh_difference'] for d in original['stages']]
            if kind in ['identity','static']:assert max(changes)==0
            if kind=='dynamic':assert max(changes)>1e-6
            altered=copy.deepcopy(inp);altered.pcfi=np.full_like(inp.pcfi,12345.)
            pp=predict(model,altered);assert np.max(np.abs(p-pp))<.005
            altered=copy.deepcopy(inp);altered.labels_context=(inp.labels_context+1)%int(inp.classes)
            pctx=predict(model,altered)
            no=copy.deepcopy(inp);no.observed[:]=False;no.observed_values[:]=0.;pn=predict(model,no);assert np.isfinite(pn).all()
        else:changes=[]
        before={name:q.detach().clone() for name,q in model.nsd.named_parameters()}
        model.train();opt=torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],lr=1e-4)
        with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
        loss.backward();assert all(q.grad is None or torch.isfinite(q.grad).all() for q in model.parameters())
        phi_grad=sum(float(q.grad.abs().sum()) for name,q in model.nsd.named_parameters() if name.startswith(('phi','project')) and q.grad is not None)
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.);opt.step()
        movement=max(float((q-before[name]).abs().max()) for name,q in model.nsd.named_parameters())
        if kind in ['static','dynamic']:assert phi_grad>0 and movement>0
        smoke.append({'arm':kind,'discarded_optimizer_steps':1,'loss':float(loss.detach()),'geometry_gradient_abs_sum':phi_grad,'nsd_parameter_update_max':movement,'layer_map_change':changes,'diagnostics':original})
        del model,opt;gc.collect();torch.cuda.empty_cache()
    out.update(status='PASS',smoke_discarded=smoke,protocol_sha256=__import__('hashlib').sha256((ROOT/'protocol.json').read_bytes()).hexdigest())
    dump(ROOT/'results/preflight.json',out);print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()

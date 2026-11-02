import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import importlib.util, json, hashlib, copy
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('sheaf_reference_common',WORK/'joint_imputation_pfn/common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
from imputation import adjacency, distances, diffuse, impute

def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def correlation(out,confidence):
    h=out-out.mean(0);norm=h.square().sum(0).clamp_min(1e-12).sqrt()
    unit=h/norm;cor=(unit.T@unit).clamp(-1,1)
    cor=cor-torch.diag_embed(torch.diagonal(cor))
    return out+(1-confidence)*((confidence*h)@cor)

def geometry(src,dst,observed,x,context,labels,classes):
    n,f=x.shape;a=adjacency((src,dst),n)
    dist=distances(a,observed,True);pc=np.power(np.float32(.9),dist)
    warm=diffuse(a,x,observed,pc,100)
    ii,jj=a.nonzero();pcden=np.maximum(np.asarray(a@pc),1e-30)
    weights=pc[jj]/pcden[ii]
    deg=np.asarray(a.sum(1)).ravel().astype('float32')
    # Context evidence: positive LP is merely an input descriptor, never an output offset.
    yy=np.zeros((n,classes),'float32');yy[context]=np.eye(classes,dtype='float32')[labels]
    p=sp.diags(1/deg)@a;belief=yy.copy()
    for _ in range(20):
        belief=p@belief;belief[context]=yy[context]
    norm=np.maximum(belief.sum(1,keepdims=True),1e-12);belief=belief/norm
    ent=-(belief*np.log(np.maximum(belief,1e-12))).sum(1)/max(np.log(classes),1)
    affinity=(belief[ii]*belief[jj]).sum(1)
    # Only provided context labels enter; no validation/test labels or hidden feature values.
    cc=np.zeros(n,bool);cc[context]=True;valid=cc[ii]&cc[jj]&(ii!=jj)
    yc=np.full(n,-1);yc[context]=labels
    alignment=float((yc[ii[valid]]==yc[jj[valid]]).mean()) if valid.any() else 1/classes
    support=float(valid.sum()/max(len(context)**2,1))
    a0,b0=warm[ii],warm[jj]
    rep=lambda v:np.broadcast_to(v[:,None],(len(ii),f))
    descriptors=np.stack([a0+b0,np.abs(a0-b0),a0*b0,np.minimum(np.abs(a0),np.abs(b0)),np.maximum(np.abs(a0),np.abs(b0)),
        observed[ii].astype('float32')+observed[jj].astype('float32'),np.minimum(pc[ii],pc[jj]),np.maximum(pc[ii],pc[jj]),
        rep(np.minimum(np.log1p(deg[ii]),np.log1p(deg[jj]))),rep(np.maximum(np.log1p(deg[ii]),np.log1p(deg[jj]))),
        rep(affinity),rep(ent[ii]+ent[jj]),np.full_like(a0,alignment),np.full_like(a0,support),
        np.full_like(a0,observed[:,0].mean())],-1).astype('float32')
    return dict(confidence=pc,warm=warm,edge_i=ii.astype('int64'),edge_j=jj.astype('int64'),weights=weights.astype('float32'),descriptors=descriptors)

def load_episode(row):
    z=np.load(ROOT/'bank'/row['path'])
    names=list(C.Inputs.__annotations__)
    inp=C.Inputs(**{k:z[k] for k in names})
    extra={k:z[k] for k in ['confidence','warm','edge_i','edge_j','weights','descriptors']}
    return inp,extra,z['labels_query'].astype('int64')

class Front(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed+30000)
        self.net=nn.Sequential(nn.Linear(15,24),nn.SiLU(),nn.Linear(24,3))
        nn.init.normal_(self.net[-1].weight,std=.01)
        with torch.no_grad():self.net[-1].bias.copy_(torch.tensor([.05,1.5,0.]))
    def forward(self,inp,extra,kind):
        dev='cuda';xo=torch.as_tensor(inp.observed_values,device=dev);obs=torch.as_tensor(inp.observed,device=dev)
        pc=torch.as_tensor(extra['confidence'],device=dev);warm=torch.as_tensor(extra['warm'],device=dev)
        ii=torch.as_tensor(extra['edge_i'],device=dev);jj=torch.as_tensor(extra['edge_j'],device=dev)
        weights=torch.as_tensor(extra['weights'],device=dev);n,f=xo.shape
        if kind=='mask':return xo,{}
        if kind=='pcfi100':return torch.where(obs,xo,correlation(warm,pc)),{}
        if kind=='pcfi200':
            h=warm
            for _ in range(100):h=torch.where(obs,xo,torch.zeros_like(h).index_add(0,ii,weights*h[jj]))
            return torch.where(obs,xo,correlation(h,pc)),{}
        raw=self.net(torch.as_tensor(extra['descriptors'],device=dev))
        theta=np.pi*torch.tanh(raw[...,0]);gamma=torch.sigmoid(raw[...,1]);cos=theta.cos();sin=theta.sin()
        reweight=torch.exp(.5*torch.tanh(raw[...,2]));ww=weights*reweight
        ww=ww/torch.zeros((n,f),device=dev).index_add(0,ii,ww).clamp_min(1e-12)[ii]
        a=1-gamma+gamma*cos;b=gamma*sin;d=1-gamma-gamma*cos
        if kind=='weighted':a=torch.ones_like(a);b=torch.zeros_like(b);d=a
        elif kind=='signed':b=torch.zeros_like(b);d=a
        assert kind in ['weighted','signed','sheaf']
        selfedge=(ii==jj)[:,None];a=torch.where(selfedge,1.,a);b=torch.where(selfedge,0.,b);d=torch.where(selfedge,1.,d)
        h0=warm;h1=torch.zeros_like(h0)
        for _ in range(100):
            q0=ww*(a*h0[jj]+b*h1[jj]);q1=ww*(b*h0[jj]+d*h1[jj])
            h0=torch.where(obs,xo,torch.zeros_like(h0).index_add(0,ii,q0))
            h1=torch.where(obs,0.,torch.zeros_like(h1).index_add(0,ii,q1))
        out=torch.where(obs,xo,correlation(h0,pc))
        info={'negative_transport_fraction':float((a.detach()<0).float().mean()),'mean_abs_cross_stalk':float(b.detach().abs().mean()),
              'observed_change':float((out.detach()[obs]-xo[obs]).abs().max()) if obs.any() else 0.}
        return out,info

class Model(C.Model):
    def __init__(self,kind,seed):
        super().__init__('pcfi_plain',seed);self.kind=kind;self.front=Front(seed).to('cuda')
    def forward(self,inp,extra,*,override=None):
        with torch.autocast('cuda',enabled=False):values,info=self.front(inp,extra,self.kind)
        if override is not None:values=override
        result=super().forward(replace(inp,pcfi=values))
        self.last_diagnostics.update(info);return result

def predict(model,inp,extra,**kw):
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p=model(inp,extra,**kw).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all();return p

def lock():
    from mgpfn import upstream as U
    return {'sources':{p.name:sha(p) for p in [ROOT/'core.py',ROOT/'data.py',ROOT/'trial.py',ROOT/'protocol.json']},
            'reference_sources':{str(p.relative_to(WORK)):sha(p) for p in [WORK/'joint_imputation_pfn/common.py',WORK/'regret_prior_pilot/core.py',WORK/'fisf_graphpfn/imputation.py']},
            'manifest':sha(ROOT/'bank/manifest.json'),'public':{p.name:sha(p) for p in [U.ADAPTERS,U.LIMIX]},'upstream':U.COMMIT}

def load_model(kind,seed):
    folder=ROOT/'results'/f'{kind}_s{seed}';info=json.loads((folder/'training.json').read_text())
    model=Model(kind,seed);state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['metadata']==lock()
    assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(state['state_dict'][name].to(p.device))
    assert C.parameter_digest(model,trainable=False)==info['frozen_sha256']
    assert C.parameter_digest(model,trainable=True)==info['final_sha256']
    return model.eval()

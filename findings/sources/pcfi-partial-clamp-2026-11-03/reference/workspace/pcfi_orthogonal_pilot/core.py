import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import importlib.util,json
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('pcfi_orth_reference',WORK/'nsd_token_pfn/core.py')
N=importlib.util.module_from_spec(spec);sys.modules[spec.name]=N;spec.loader.exec_module(N)
C=N.C;dump=N.dump;sha=N.sha;adjacency=N.adjacency;impute=N.impute
from imputation import distances

def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))

def correlation(out,pc):
    h=out-out.mean(0);norm=h.square().sum(0).clamp_min(1e-30).sqrt()
    unit=h/norm;cor=(unit.T@unit).clamp(-1,1);cor=cor-torch.diag_embed(torch.diagonal(cor))
    return out+(1-pc)*((pc*h)@cor)

def geometry(inp):
    x=inp.observed_values;obs=inp.observed;n,f=x.shape;a=adjacency((inp.src,inp.dst),n)
    pc=np.power(np.float32(.9),distances(a,obs,True));ii,jj=a.nonzero()
    den=np.maximum(np.asarray(a@pc),1e-30);weights=(pc[jj]/den[ii]).astype('float32')
    deg=np.asarray(a.sum(1)).ravel().astype('float32');p=sp.diags(1/deg)@a
    yy=np.zeros((n,int(inp.classes)),'float32');yy[inp.context]=np.eye(int(inp.classes),dtype='float32')[inp.labels_context]
    belief=yy.copy()
    for _ in range(20):belief=p@belief;belief[inp.context]=yy[inp.context]
    belief=belief/np.maximum(belief.sum(1,keepdims=True),1e-12)
    entropy=-(belief*np.log(np.maximum(belief,1e-12))).sum(1)/max(np.log(int(inp.classes)),1)
    support=np.asarray(a@obs.astype('float32'));near=np.asarray(a@x)/np.maximum(support,1)
    context=np.zeros(n,'float32');context[inp.context]=1
    node_label=np.full(n,-1);node_label[inp.context]=inp.labels_context
    valid=(node_label[ii]>=0)&(node_label[jj]>=0)&(ii!=jj)
    alignment=float((node_label[ii[valid]]==node_label[jj[valid]]).mean()) if valid.any() else 1/int(inp.classes)
    rep=lambda v:np.broadcast_to(v[:,None],(n,f))
    descriptor=np.stack([x,np.abs(x),near,obs.astype('float32'),pc,rep(np.log1p(deg)/10),rep(context),
                         rep(entropy),rep(belief.max(1)),np.full_like(x,alignment)],-1).astype('float32')
    # Geometry sees observed values/masks/context only. No imputed warm-up or hidden X.
    return {'pc':pc,'ii':ii.astype('int64'),'jj':jj.astype('int64'),'weights':weights,'descriptor':descriptor,
            'neighbor':np.stack([p@descriptor[...,k] for k in range(10)],-1).astype('float32'),
            'affinity':(belief[ii]*belief[jj]).sum(1).astype('float32'),
            'node_affinity':np.asarray(a.multiply(belief@belief.T).sum(1)).ravel().astype('float32')/deg}

def tensor_extra(extra,device='cuda'):return {k:torch.as_tensor(v,device=device) for k,v in extra.items()}

class Front(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed+50000)
        self.phi=nn.Sequential(nn.Linear(21,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.normal_(self.phi[-1].weight,std=.01);nn.init.zeros_(self.phi[-1].bias)

    def angles(self,ex,kind):
        u=ex['descriptor'];ii,jj=ex['ii'],ex['jj'];f=u.shape[1]
        if kind=='node':
            affinity=ex['node_affinity'][:,None,None].expand(-1,f,1)
            theta=np.pi*torch.tanh(self.phi(torch.cat((u,ex['neighbor'],affinity),-1)).squeeze(-1))
            delta=theta[jj]-theta[ii]
        else:
            affinity=ex['affinity'][:,None,None].expand(-1,f,1)
            theta_i=np.pi*torch.tanh(self.phi(torch.cat((u[ii],u[jj],affinity),-1)).squeeze(-1))
            theta_j=np.pi*torch.tanh(self.phi(torch.cat((u[jj],u[ii],affinity),-1)).squeeze(-1))
            delta=theta_j-theta_i
        return torch.where((ii==jj)[:,None],0.,delta)

    def forward(self,xo,obs,ex,kind,*,identity=False):
        ii,jj,w,pc=ex['ii'],ex['jj'],ex['weights'],ex['pc'];n,f=xo.shape
        delta=xo.new_zeros((len(ii),f)) if identity or kind=='pcfi' else self.angles(ex,kind)
        cos,sin=delta.cos(),delta.sin()
        if kind in ['pcfi','scalar'] or identity:sin=torch.zeros_like(sin)
        h0=xo;h1=torch.zeros_like(xo)
        for _ in range(100):
            q0=w*(cos*h0[jj]-sin*h1[jj]);q1=w*(sin*h0[jj]+cos*h1[jj])
            h0=torch.where(obs,xo,torch.zeros_like(h0).index_add(0,ii,q0))
            h1=torch.where(obs,0.,torch.zeros_like(h1).index_add(0,ii,q1))
        out=torch.where(obs,xo,correlation(h0,pc))
        nonself=ii!=jj
        info={'negative_cos_fraction':float((cos.detach()[nonself]<0).float().mean()) if nonself.any() else 0.,
              'mean_abs_sin':float(sin.detach()[nonself].abs().mean()) if nonself.any() else 0.,
              'second_coordinate_mean_abs':float(h1.detach().abs().mean()),
              'observed_change':float((out.detach()[obs]-xo[obs]).abs().max()) if obs.any() else 0.,
              'spatial_steps':100,'channel_corrections':1,'warmup_spatial_steps':0}
        return out,info

class Model(nn.Module):
    def __init__(self,kind,seed):
        super().__init__();self.kind=kind;self.front=Front(seed).to('cuda')
        self.pfn=N.load_model('pcfi100',seed)
        for p in self.pfn.parameters():p.requires_grad_(False)
        self.pfn.eval();self.last_diagnostics={}
    def train(self,mode=True):
        super().train(mode);self.pfn.eval();return self
    def forward(self,inp,extra,*,identity=False):
        with torch.autocast('cuda',enabled=False):
            xo=torch.as_tensor(inp.observed_values,device='cuda');obs=torch.as_tensor(inp.observed,device='cuda')
            ex=tensor_extra(extra) if isinstance(extra['pc'],np.ndarray) else extra
            values,info=self.front(xo,obs,ex,self.kind,identity=identity)
        logits=self.pfn(replace(inp,pcfi=values));self.last_diagnostics=info;return logits

def predict(model,inp,extra,**kw):
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p=model(inp,extra,**kw).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all();return p

def load_episode(row):
    z=np.load(WORK/row['bank_path']);inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
    ex=np.load(ROOT/row['extra_path']);return inp,{k:ex[k] for k in ex.files},z['labels_query'].astype('int64')

def lock():
    return {'sources':{p.name:sha(p) for p in [ROOT/'core.py',ROOT/'bank.py',ROOT/'run.py',ROOT/'protocol.json']},
            'manifest':sha(ROOT/'bank/manifest.json'),'old_lock':N.lock(),
            'pfn_checkpoints':{str(s):sha(WORK/f'nsd_token_pfn/results/pcfi100_s{s}/checkpoint.pt') for s in cfg()['seeds']}}

def load_model(kind,seed):
    obj=Model(kind,seed)
    if kind=='pcfi':return obj.eval()
    folder=ROOT/'results'/f'{kind}_s{seed}';info=json.loads((folder/'training.json').read_text());assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert state['metadata']==lock()
    obj.front.load_state_dict(state['front']);assert C.parameter_digest(obj.pfn)==info['pfn_sha256']
    assert C.parameter_digest(obj.front)==info['final_front_sha256'];return obj.eval()

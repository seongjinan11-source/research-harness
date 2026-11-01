"""Normalized graph-conditioned tree circuits and a direct graph predictor.

Each graph has ONE latent state/label variable per original node and ONE
Gaussian leaf per feature coordinate. A spanning tree induces disjoint subtree
scopes. Its transitions are row-normalized; its root and mixture are normalized.
Parameters depend on G and a known family descriptor, never on X_obs/M/Y_C.
Thus missing features integrate to one and context labels are leaf evidence.
This is exact inference in a learned conditional tree model, not exact Bayes
inference for the original cyclic SBM, nor a reproduction of GSPN or MaM.
"""
import sys
sys.dont_write_bytecode=True
import sklearn
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphEncoder(nn.Module):
    def __init__(self,width=48,extra=0):
        super().__init__();self.width=width
        self.stem=nn.Linear(7,width)
        self.extra=nn.Linear(extra,width,bias=False) if extra else None
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(5*width,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) for _ in range(3)])
    def forward(self,batch,signals=None):
        a=batch['adj'];deg=a.sum(-1);n=a.shape[-1]
        typ=batch['type'];prior=batch['prior'][:,None].expand(-1,n,-1)
        x=torch.cat((typ[...,None],deg.transpose(1,2)/n,prior,torch.ones_like(typ[...,None])),dim=-1)
        x=self.stem(x)
        if signals is not None:x=x+self.extra(signals)
        valid=batch['pair_valid']
        mats=torch.cat((a,(valid-a).clamp_min(0)),dim=1)
        mats=mats/mats.sum(-1,keepdim=True).clamp_min(1)
        for block in self.blocks:
            messages=torch.einsum('brij,bjd->brid',mats,x).transpose(1,2).flatten(2)
            x=x+.2*block(torch.cat((x,messages),-1))
        return x


class DirectGraph(nn.Module):
    def __init__(self,width=48,features=2):
        super().__init__();self.encoder=GraphEncoder(width,extra=2*features+2)
        self.read=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,2))
    def forward(self,batch):
        c=batch['context'];known=c>=0
        labels=F.one_hot(c.clamp_min(0).long(),2).to(batch['x'].dtype)*known[...,None]
        x=self.encoder(batch,torch.cat((batch['x'],batch['observed'].to(batch['x'].dtype),labels),-1))
        return self.read(torch.cat((x,x.mean(1,keepdim=True).expand_as(x)),-1)).softmax(-1)


class TreeCircuit(nn.Module):
    def __init__(self,width=48,components=16,features=2):
        super().__init__();self.encoder=GraphEncoder(width)
        self.components=components;self.features=features
        self.hypothesis=nn.Parameter(torch.randn(components,16)*.15)
        self.transition=nn.Sequential(nn.Linear(3*width+18,64),nn.SiLU(),nn.Linear(64,1))
        self.mix=nn.Sequential(nn.Linear(width,48),nn.SiLU(),nn.Linear(48,components))
        self.mean=nn.Parameter(torch.randn(components,features)*.7)
        self.raw_scale=nn.Parameter(torch.full((components,features),-.2))

    def parameters_for(self,batch):
        z=self.encoder(batch);b,n,d=z.shape;parent=batch['parent'];ar=torch.arange(b,device=z.device)[:,None]
        zp=z[ar,parent.clamp_min(0)];pool=z.mean(1)
        edge=batch['adj'][torch.arange(b,device=z.device)[:,None,None],torch.arange(2,device=z.device)[None,:,None],torch.arange(n,device=z.device)[None,None,:],parent.clamp_min(0)[:,None,:]].transpose(1,2)
        h=self.hypothesis[None,:,None,:].expand(b,-1,n,-1)
        inp=torch.cat(((z+zp)[:,None].expand(-1,self.components,-1,-1),
            (z-zp).abs()[:,None].expand(-1,self.components,-1,-1),pool[:,None,None].expand(-1,self.components,n,-1),
            edge[:,None].expand(-1,self.components,-1,-1),h),-1)
        j=4*torch.tanh(self.transition(inp).squeeze(-1))
        # Paired feature orientations enforce global label-flip symmetry.
        j=j.repeat(1,2,1)
        logmix=F.log_softmax(self.mix(pool),-1).repeat(1,2)-math.log(2)
        mean=3*torch.tanh(self.mean/3);mean=torch.cat((mean,-mean),0)
        scale=(F.softplus(self.raw_scale)+.1).repeat(2,1)
        return dict(j=j,logmix=logmix,mean=mean,scale=scale)

    def emissions(self,batch,parameters):
        p=parameters;x=batch['x'];obs=batch['observed'];spin=x.new_tensor([-1,1])
        mu=p['mean'][None,:,None,:,None]*spin[None,None,None,None,:]
        sd=p['scale'][None,:,None,:,None]
        logp=-.5*((x[:,None,:,:,None]-mu)/sd)**2-torch.log(sd)-.5*math.log(2*math.pi)
        leaf=(logp*obs[:,None,:,:,None].to(logp.dtype)).sum(-2)
        c=batch['context']
        allowed=(c[:,None,:,None]<0)|(c[:,None,:,None]==x.new_tensor([0,1])[None,None,None,:])
        return leaf.masked_fill(~allowed,-torch.inf)

    @staticmethod
    def sum_product(batch,p,leaf,need_marginals=True):
        b,h,n,_=leaf.shape;ar=torch.arange(b,device=leaf.device);order=batch['order'];parent=batch['parent']
        eye=torch.eye(2,device=leaf.device,dtype=torch.bool)[None,None]
        j=p['j'];total=leaf;up=leaf.new_zeros((b,h,n,2))
        for t in range(n-1,0,-1):
            child=order[:,t];pa=parent[ar,child]
            sub=total[ar,:,child];jc=j[ar,:,child]
            trans=torch.where(eye,F.logsigmoid(jc)[...,None,None],F.logsigmoid(-jc)[...,None,None])
            message=torch.logsumexp(trans+sub[:,:,None,:],-1)
            up=up+F.one_hot(child,n)[:,None,:,None]*message[:,:,None,:]
            total=total+F.one_hot(pa,n)[:,None,:,None]*message[:,:,None,:]
        root=order[:,0];root_score=total[ar,:,root]-math.log(2)
        logz=torch.logsumexp(root_score,-1)
        log_evidence=torch.logsumexp(p['logmix']+logz,-1)
        if not need_marginals:return log_evidence
        down=leaf.new_zeros((b,h,n,2))
        down=down-F.one_hot(root,n)[:,None,:,None]*math.log(2)
        for t in range(1,n):
            child=order[:,t];pa=parent[ar,child]
            cavity=total[ar,:,pa]+down[ar,:,pa]-up[ar,:,child]
            jc=j[ar,:,child]
            trans=torch.where(eye,F.logsigmoid(jc)[...,None,None],F.logsigmoid(-jc)[...,None,None])
            message=torch.logsumexp(cavity[:,:,:,None]+trans,-2)
            down=down+F.one_hot(child,n)[:,None,:,None]*message[:,:,None,:]
        conditional=(total+down).softmax(-1)
        weight=(p['logmix']+logz).softmax(-1)
        predictive=torch.einsum('bh,bhnc->bnc',weight,conditional)
        return predictive,log_evidence,weight,conditional

    def forward(self,batch,details=False):
        p=self.parameters_for(batch);leaf=self.emissions(batch,p)
        result=self.sum_product(batch,p,leaf)
        return (result,p) if details else result[0]

    def evidence(self,batch,parameters=None):
        p=self.parameters_for(batch) if parameters is None else parameters
        return self.sum_product(batch,p,self.emissions(batch,p),False)


def enumerate_circuit(model,batch):
    """Independent finite-label evaluator; missing Gaussian coordinates integrate analytically."""
    assert len(batch['x'])==1
    p=model.parameters_for(batch);leaf=model.emissions(batch,p);n=leaf.shape[2]
    states=((torch.arange(2**n,device=leaf.device)[:,None]>>torch.arange(n,device=leaf.device))&1).long()
    out=[];parent=batch['parent'][0];root=batch['order'][0,0]
    for hh in range(leaf.shape[1]):
        score=leaf[0,hh,torch.arange(n,device=leaf.device)[None],states].sum(-1)-math.log(2)
        for node in range(n):
            if node==root:continue
            jc=p['j'][0,hh,node];same=states[:,node]==states[:,parent[node]]
            score=score+torch.where(same,F.logsigmoid(jc),F.logsigmoid(-jc))
        out.append(score+p['logmix'][0,hh])
    joint=torch.stack(out);z=torch.logsumexp(joint.flatten(),0)
    pred=torch.einsum('hs,sn->n',torch.exp(joint-z),states.to(joint.dtype))
    return z,torch.stack((1-pred,pred),-1)

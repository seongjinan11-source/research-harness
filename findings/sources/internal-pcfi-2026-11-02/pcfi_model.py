"""Literal PCFI kernels and an interleaved public GraphPFN."""
import sys
sys.dont_write_bytecode = True
import sklearn
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pcfi_shared import C


def weights(confidence,i,j,n,log_adjust=None):
    value=confidence[j]
    if log_adjust is not None:value=value*log_adjust.exp()
    # Unique edge entries, then deterministic row reductions; no atomic float sums.
    matrix=value.new_zeros((value.shape[-1],n,n))
    matrix[:,i,j]=value.T
    return matrix/matrix.sum(-1,keepdim=True).clamp_min(torch.finfo(value.dtype).tiny)


def diffuse(value,x,observed,p,i,j,steps):
    for _ in range(steps):
        message=torch.bmm(p,value.T.unsqueeze(-1)).squeeze(-1).T
        value=torch.where(observed,x,message)
    return value


def correlate(value,confidence):
    h=value-value.mean(0)
    # clamp squared norm before sqrt: finite zero-variance backward.
    normalized=h/h.square().sum(0).clamp_min(1e-20).sqrt()
    r=(normalized.T@normalized).clamp(-1,1)
    r=r-torch.diag_embed(r.diagonal())
    return value+(1-confidence)*((confidence*h)@r)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,value,observed,confidence,i,j,block,hidden,labels,blind=False):
        n,d=value.shape
        degree=torch.bincount(i,minlength=n).to(value.dtype)/n
        degree=degree[:,None].expand(-1,d)
        count=observed.float().mean(0)[None].expand(len(i),-1)
        structural=[]
        for t in [value,confidence,observed.float(),degree]:
            structural.extend([t[i]+t[j],(t[i]-t[j]).abs()])
        structural.extend([value.new_full((len(i),d),block/3),count])
        fields=value.new_zeros((len(i),d,4))
        if not blind:
            # No query label. Class names disappear through symmetric prototype statistics.
            groups=d//2
            h=F.normalize((hidden[0,:,:groups]+hidden[0,:,-1:]).float(),dim=-1)
            prototypes=torch.stack([h[:len(labels)][labels==c].mean(0) for c in labels.unique(sorted=True)])
            prototypes=F.normalize(prototypes,dim=-1)
            scores=3*torch.einsum('ngh,cgh->ngc',h,prototypes)
            prob=scores.softmax(-1)
            ent=-(prob*prob.clamp_min(1e-10).log()).sum(-1)/max(math.log(prob.shape[-1]),1e-6)
            top=prob.max(-1).values
            affinity=(prob[i]*prob[j]).sum(-1)
            fields=torch.stack((affinity,top[i]*top[j],.5*(ent[i]+ent[j]),(ent[i]-ent[j]).abs()),-1).repeat_interleave(2,dim=1)
        raw=self.net(torch.cat((torch.stack(structural,-1),fields),-1)).squeeze(-1)
        return math.log(2)*torch.tanh(raw)


class Model(C.Model):
    def __init__(self,arm,seed):
        super().__init__('mask_plain',seed)
        self.arm=arm
        if arm in ['context','label_blind']:
            torch.manual_seed(seed+30000)
            self.router=Router().cuda()

    def forward(self,inp,extra,*,blank_router_context=False):
        import dgl
        torch.manual_seed(int(inp.forward_seed))
        ids=np.concatenate((inp.context,np.setdiff1d(np.arange(len(inp.observed)),inp.context)))
        inv=np.argsort(ids)
        src=torch.as_tensor(inv[inp.src],device='cuda',dtype=torch.long)
        dst=torch.as_tensor(inv[inp.dst],device='cuda',dtype=torch.long)
        graph=dgl.graph((src,dst),num_nodes=len(ids),device='cuda')
        observed=torch.as_tensor(inp.observed[ids],device='cuda',dtype=torch.bool)
        xo=torch.as_tensor(np.where(inp.observed,inp.observed_values,0)[ids],device='cuda')
        conf=torch.as_tensor(extra['confidence'][ids],device='cuda')
        i=torch.as_tensor(inv[extra['edge_i']],device='cuda',dtype=torch.long)
        j=torch.as_tensor(inv[extra['edge_j']],device='cuda',dtype=torch.long)
        labels=torch.as_tensor(inp.labels_context,device='cuda',dtype=torch.long)
        value=xo
        fixed=weights(conf,i,j,len(ids))
        clamp_error=0.
        if self.arm=='external':
            with torch.autocast('cuda',enabled=False):
                value=correlate(diffuse(value,xo,observed,fixed,i,j,100),conf).clamp(-10,10)
                value=torch.where(observed,xo,value)
        t=self.base.tfm.module
        encoded=self.numeric_embedding(value)
        missing=(~observed).float().reshape(1,len(ids),-1,2)
        features=t.add_embeddings(encoded+self.mask_embedding(missing).to(encoded.dtype))
        y=torch.full((1,len(ids),1),float('nan'),device='cuda')
        y[0,:len(labels),0]=labels.float()
        y_emb=t.mixed_y_embedding({'data':y},y_type=torch.zeros_like(y),eval_pos=len(labels))
        h=torch.cat((features,y_emb.unsqueeze(2)),2)
        self.base._graph_holder.graph=graph
        block=0
        try:
            for index,layer in enumerate(t.transformer_encoder.layers):
                if index in [0,3,6,9]:
                    if self.arm!='external':
                        with torch.autocast('cuda',enabled=False):
                            adjust=self.router(value,observed,conf,i,j,block,h,labels,blind=self.arm=='label_blind' or blank_router_context) if hasattr(self,'router') else None
                            p=weights(conf,i,j,len(ids),adjust)
                            updated=diffuse(value,xo,observed,p,i,j,25)
                            if block==3:updated=correlate(updated,conf).clamp(-10,10)
                            updated=torch.where(observed,xo,updated)
                        delta=self.numeric_embedding(updated)-self.numeric_embedding(value)
                        h=torch.cat((h[:,:,:-1]+delta,h[:,:,-1:]),2)
                        value=updated
                        with torch.no_grad():
                            if observed.any():clamp_error=max(clamp_error,float((value[observed]-xo[observed]).abs().max()))
                    h=self.extra(h)
                    block+=1
                h,_,_=layer(h,feature_atten_mask=None,eval_pos=len(labels),layer_idx=index)
            h=t.encoder_out_norm(h)
            logits,_=t.y_decoder(h[:,len(labels):,-1],torch.zeros((1,len(ids)-len(labels),1),device='cuda'))
        finally:self.base._graph_holder.graph=None
        self.last_diagnostics={'observed_clamp_max':clamp_error,'propagation_steps':100,'correlations':1,'pfn_layers':12}
        q=torch.as_tensor(inv[inp.query]-len(labels),device='cuda')
        return logits[0,q,:int(inp.classes)].float()


def predict(model,inp,extra,**kwargs):
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        p=model(inp,extra,**kwargs).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all()
    return p

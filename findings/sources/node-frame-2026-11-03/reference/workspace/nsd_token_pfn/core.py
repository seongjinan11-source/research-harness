import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('nsd_reference_common',WORK/'joint_imputation_pfn/common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
from imputation import adjacency,impute
def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_episode(row):
    z=np.load(ROOT/row['bank_path']);inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
    return inp,z['labels_query'].astype('int64')
def confidence(src,dst,observed):
    n=len(observed);a=adjacency((src,dst),n);sources=np.flatnonzero(observed.any(1))
    if len(sources):
        ds=dijkstra(a,directed=False,unweighted=True,indices=sources,min_only=True)
        return np.where(np.isfinite(ds),.9**ds,0.).astype('float32')
    return np.zeros(n,'float32')
def rotation(theta):
    c,s=theta.cos(),theta.sin();return torch.stack([c,-s,s,c],-1).reshape(-1,2,2)
def laplacian(x,ii,jj,weight,fi,fj):
    # Unique undirected edges. E=sum_e w||F_i h_i-F_j h_j||².
    n=len(x);degree=x.new_zeros(n).index_add(0,ii,weight).index_add(0,jj,weight)
    inv=(degree+1).rsqrt();scaled=x*inv[:,None,None,None]
    delta=torch.einsum('eab,etbc->etac',fi,scaled[ii])-torch.einsum('eab,etbc->etac',fj,scaled[jj])
    left=torch.einsum('eba,etbc->etac',fi,delta)*weight[:,None,None,None]
    right=-torch.einsum('eba,etbc->etac',fj,delta)*weight[:,None,None,None]
    return x.new_zeros(x.shape).index_add(0,ii,left).index_add(0,jj,right)*inv[:,None,None,None]

class TokenNSD(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed+40000)
        self.project=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,32),nn.Tanh())
        self.phi=nn.Sequential(nn.Linear(72,32),nn.SiLU(),nn.Linear(32,1,bias=False))
        nn.init.normal_(self.phi[-1].weight,std=.05)
        self.left=nn.Parameter(torch.eye(2).repeat(4,1,1))
        self.right=nn.Parameter(torch.eye(96).repeat(4,1,1))
        self.step=nn.Parameter(torch.full((4,),float(np.log(.2/.8))))
    def maps(self,h,ii,jj,meta):
        summary=torch.cat((h[:,:-1].mean(1),h[:,-1]),-1)
        u=torch.cat((self.project(summary),meta),-1)
        fi=rotation(np.pi*torch.tanh(self.phi(torch.cat((u[ii],u[jj]),-1)).squeeze(-1)))
        fj=rotation(np.pi*torch.tanh(self.phi(torch.cat((u[jj],u[ii]),-1)).squeeze(-1)))
        return fi,fj
    def forward(self,h,ii,jj,weight,meta,anchor,stage,kind,cached=None):
        with torch.autocast('cuda',enabled=False):
            z=h.float();n,t,_=z.shape
            if kind=='identity':
                fi=torch.eye(2,device=h.device).expand(len(ii),2,2);fj=fi
            elif kind=='static' and cached is not None:fi,fj=cached
            else:fi,fj=self.maps(z,ii,jj,meta)
            x=z.reshape(n,t,2,96)
            x=torch.einsum('ab,ntbc->ntac',self.left[stage],x)@self.right[stage]
            update=F.elu(laplacian(x,ii,jj,weight,fi,fj)).reshape(n,t,192)
            out=z-.5*torch.sigmoid(self.step[stage])*update
            out=torch.where(anchor[:,:,None],z,out)
            transport=fi.transpose(-1,-2)@fj
            info={'negative_cos_fraction':float((transport.detach()[:,0,0]<0).float().mean()) if len(ii) else 0.,
                  'anchor_change':float((out.detach()-z.detach())[anchor].abs().max()) if anchor.any() else 0.,
                  'map_refresh_difference':float((fi.detach()-cached[0].detach()).abs().mean()) if cached is not None and len(ii) else 0.,
                  'mean_abs_update':float((out.detach()-z.detach()).abs().mean())}
        return out.to(h.dtype),(fi,fj),info

class Model(C.Model):
    def __init__(self,kind,seed):
        super().__init__('pcfi_plain',seed);self.kind=kind;self.nsd=TokenNSD(seed).to('cuda')
    def forward(self,inp):
        import dgl
        torch.manual_seed(int(inp.forward_seed))
        ids=np.concatenate((inp.context,np.setdiff1d(np.arange(len(inp.observed)),inp.context)));inverse=np.argsort(ids);n=len(ids)
        src=torch.as_tensor(inverse[inp.src],device='cuda',dtype=torch.long);dst=torch.as_tensor(inverse[inp.dst],device='cuda',dtype=torch.long)
        graph=dgl.graph((src,dst),num_nodes=n,device='cuda')
        observed=torch.as_tensor(inp.observed[ids],device='cuda',dtype=torch.bool)
        xo=torch.as_tensor(inp.observed_values[ids],device='cuda')
        values=torch.as_tensor(inp.pcfi[ids],device='cuda') if self.kind=='pcfi100' else xo
        context=torch.arange(n,device='cuda')<len(inp.context);t=self.base.tfm.module
        features=t.add_embeddings(self.numeric_embedding(values)+self.mask_embedding((~observed).float().reshape(1,n,-1,2)))
        y=torch.full((1,n,1),float('nan'),device='cuda');y[0,:len(inp.context),0]=torch.as_tensor(inp.labels_context,device='cuda',dtype=torch.float32)
        y_emb=t.mixed_y_embedding({'data':y},y_type=torch.zeros_like(y),eval_pos=len(inp.context));h=torch.cat((features,y_emb.unsqueeze(2)),2)
        # These quantities depend only on observed input/context and graph, never query labels.
        use=self.kind in ['identity','static','dynamic'];trace=[];cached=None
        if use:
            idx=src<dst;ii,jj=src[idx],dst[idx]
            c=torch.as_tensor(confidence(inp.src,inp.dst,inp.observed)[ids],device='cuda')
            degree=torch.bincount(src,minlength=n).float()
            meta=torch.stack((observed.float().mean(1),context.float(),c,torch.log1p(degree)/10),-1)
            weight=.25+.75*(c[ii]*c[jj]).sqrt()
            anchor=torch.cat((observed.reshape(n,-1,2).all(2),context[:,None]),1)
        self.base._graph_holder.graph=graph
        try:
            for index,layer in enumerate(t.transformer_encoder.layers):
                if index in cfg()['layers']:
                    h=self.extra(h)
                    if use:
                        stage=cfg()['layers'].index(index)
                        h_step,cached_new,info=self.nsd(h[0],ii,jj,weight,meta,anchor,stage,self.kind,cached)
                        if cached is None:cached=cached_new
                        h=h_step.unsqueeze(0);trace.append(info)
                h,_,_=layer(h,feature_atten_mask=None,eval_pos=len(inp.context),layer_idx=index)
            h=t.encoder_out_norm(h)
            logits,_=t.y_decoder(h[:,len(inp.context):,-1],torch.zeros((1,n-len(inp.context),1),device='cuda'))
        finally:self.base._graph_holder.graph=None
        q=inverse[inp.query]-len(inp.context)
        self.last_diagnostics={'stages':trace,'input_observed_nodes':int(inp.observed.any(1).sum()),'geometry_updates':4 if self.kind=='dynamic' else 1 if self.kind=='static' else 0}
        return logits[0,torch.as_tensor(q,device='cuda'),:int(inp.classes)].float()
def predict(model,inp):
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):p=model(inp).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all();return p
def lock():
    from mgpfn import upstream as U
    return {'sources':{p.name:sha(p) for p in [ROOT/'core.py',ROOT/'bank.py',ROOT/'run.py',ROOT/'protocol.json']},
      'references':{str(p.relative_to(WORK)):sha(p) for p in [WORK/'joint_imputation_pfn/common.py',WORK/'sheaf_pcfi_pilot/data.py',WORK/'fisf_graphpfn/imputation.py']},
      'manifest':sha(ROOT/'bank/manifest.json'),'public':{p.name:sha(p) for p in [U.ADAPTERS,U.LIMIX]}}
def load_model(kind,seed):
    folder=ROOT/'results'/f'{kind}_s{seed}';info=json.loads((folder/'training.json').read_text());assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    model=Model(kind,seed);st=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);assert st['metadata']==lock()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(st['state_dict'][name].to(p.device))
    assert C.parameter_digest(model,trainable=False)==info['frozen_sha256'];assert C.parameter_digest(model,trainable=True)==info['final_sha256']
    return model.eval()

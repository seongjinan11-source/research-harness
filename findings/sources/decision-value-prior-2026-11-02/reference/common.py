"""Public GraphPFN pilot infrastructure; input objects never contain query labels."""
import sys
sys.dont_write_bytecode = True
import sklearn  # Windows import order required by the existing DGL environment.
import gc
import hashlib
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('joint_imputation_reference_core', WORK/'regret_prior_pilot/core.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)


def config():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sources():
    return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))} | {'protocol.json':sha(ROOT/'protocol.json')}


def setup():
    return BASE.setup()


@dataclass
class Inputs:
    observed_values: np.ndarray
    observed: np.ndarray
    pcfi: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    context: np.ndarray
    labels_context: np.ndarray
    query: np.ndarray
    classes: int
    forward_seed: int


def observed_scale(x, mask):
    safe = np.where(mask, x, 0.).astype(np.float64)
    count = mask.sum(0)
    rms = np.sqrt(np.divide((safe*safe).sum(0), count, out=np.ones(x.shape[1]), where=count>0))
    rms = np.where(rms>=1e-8, rms, 1.)
    return np.where(mask, np.clip(safe/rms, -10, 10), 0).astype(np.float32)


def read_episode(row):
    with np.load(ROOT/'bank'/row['path']) as d:
        inp = Inputs(**{k:d[k] for k in Inputs.__annotations__})
        return inp, d['labels_query'].astype(np.int64)


def metrics(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-8, 1)
    p /= p.sum(1, keepdims=True)
    return {'accuracy':float((p.argmax(1)==y).mean()),
        'nll':float(-np.log(p[np.arange(len(y)),y]).mean()),
        'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean())}


def parameter_digest(model, *, trainable=None):
    h = hashlib.sha256()
    for name,p in model.named_parameters():
        if trainable is not None and p.requires_grad != trainable:
            continue
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def mean_neighbors(x, src, dst, n):
    total = x.new_zeros((n,)+x.shape[1:]).index_add(0,dst,x[src])
    degree = torch.bincount(dst,minlength=n).clamp_min(1).to(x.dtype)
    return total/degree.reshape((n,)+(1,)*(x.ndim-1))


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(192,16)
        self.label = nn.Linear(192,16)
        self.net = nn.Sequential(nn.Linear(74,64),nn.SiLU(),nn.Linear(64,4))
        nn.init.normal_(self.net[-1].weight,std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values, observed_values, observed, tokens, context, src, dst):
        n,d = values.shape; groups=d//2
        v = values.reshape(n,groups,2)
        m = observed.reshape(n,groups,2).to(v.dtype)
        fx = torch.tanh(self.feature(tokens[0,:,:groups].float()))
        fy = torch.tanh(self.label(tokens[0,:,-1].float()))[:,None].expand(-1,groups,-1)
        c = context.float()[:,None,None].expand(n,groups,1)
        rate = c.mean().expand_as(c)
        parts = [v, mean_neighbors(v,src,dst,n), m, mean_neighbors(m,src,dst,n),
            fx, mean_neighbors(fx,src,dst,n),fy,mean_neighbors(fy,src,dst,n),c,rate]
        raw = self.net(torch.cat(parts,-1))
        gate = torch.sigmoid(raw[...,2:])
        proposal = (1-gate)*v + gate*mean_neighbors(v,src,dst,n) + torch.tanh(raw[...,:2])
        proposal = proposal.reshape(n,d).clamp(-10,10)
        return torch.where(observed, observed_values, proposal)


class TokenControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,24),nn.SiLU(),nn.Linear(24,192))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,tokens):
        return torch.cat((tokens[:,:,:-1]+.1*self.net(tokens[:,:,:-1]),tokens[:,:,-1:]),2)


class Model(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        from mgpfn import upstream as U
        from mgpfn.train import trainable_policy
        torch.manual_seed(seed)
        self.base, keys=U.load_light('cuda',always_checkpoint=False)
        self.groups=trainable_policy(self.base,keys)
        self.arm=arm
        # Reset independently so all arms share exactly the same common initialization.
        torch.manual_seed(seed+10000)
        self.mask_embedding=nn.Linear(2,192,bias=False)
        nn.init.zeros_(self.mask_embedding.weight)
        torch.manual_seed(seed+20000)
        self.extra=Refiner() if arm in ('frontend','interleaved') else TokenControl()
        self.last_diagnostics={}
        self.to('cuda')

    def numeric_embedding(self, values):
        t=self.base.tfm.module
        grouped=values.reshape(1,len(values),-1,2)
        return t.encoder_x({'data':grouped,'nan_encoding':torch.zeros_like(grouped)})['data']

    def forward(self, inp, *, disable_feedback=False, disable_refinement=False):
        import dgl
        torch.manual_seed(int(inp.forward_seed))
        # Explicit context-first order; unlabelled original context labels stay withheld.
        ids=np.concatenate((inp.context,np.setdiff1d(np.arange(len(inp.observed)),inp.context)))
        inverse=np.argsort(ids)
        src=torch.as_tensor(inverse[inp.src],device='cuda',dtype=torch.long)
        dst=torch.as_tensor(inverse[inp.dst],device='cuda',dtype=torch.long)
        graph=dgl.graph((src,dst),num_nodes=len(ids),device='cuda')
        observed=torch.as_tensor(inp.observed[ids],device='cuda',dtype=torch.bool)
        xo=torch.as_tensor(inp.observed_values[ids],device='cuda')
        values=torch.as_tensor(inp.pcfi[ids],device='cuda') if self.arm=='pcfi_plain' else xo
        context=torch.arange(len(ids),device='cuda')<len(inp.context)
        t=self.base.tfm.module
        assert t.features_per_group==2 and values.shape[1]%2==0
        encoded=self.numeric_embedding(values)
        missing=(~observed).float().reshape(1,len(ids),-1,2)
        features=t.add_embeddings(encoded+self.mask_embedding(missing).to(encoded.dtype))
        y=torch.full((1,len(ids),1),float('nan'),device='cuda')
        y[0,:len(inp.context),0]=torch.as_tensor(inp.labels_context,device='cuda',dtype=torch.float32)
        y_emb=t.mixed_y_embedding({'data':y},y_type=torch.zeros_like(y),eval_pos=len(inp.context))
        h=torch.cat((features,y_emb.unsqueeze(2)),2)
        h0=h
        states=[]
        def refine(current,v):
            with torch.autocast('cuda',enabled=False):
                updated=self.extra(v.float(),xo,observed,current.float(),context,src,dst)
            delta=self.numeric_embedding(updated)-self.numeric_embedding(v)
            new=torch.cat((current[:,:,:-1]+delta,current[:,:,-1:]),2)
            states.append(updated)
            return new,updated
        if self.arm=='frontend' and not disable_refinement:
            for _ in config()['refinement_layers']:
                h,values=refine(h,values)
        self.base._graph_holder.graph=graph
        try:
            for index, layer in enumerate(t.transformer_encoder.layers):
                if index in config()['refinement_layers']:
                    if self.arm=='interleaved' and not disable_refinement:
                        if disable_feedback:
                            # Numeric increments still reach the running PFN, but refiner reads only input tokens.
                            old_embedding=self.numeric_embedding(values)
                            read=torch.cat((h0[:,:,:-1]+old_embedding-encoded,h0[:,:,-1:]),2)
                            with torch.autocast('cuda',enabled=False):
                                updated=self.extra(values.float(),xo,observed,read.float(),context,src,dst)
                            h=torch.cat((h[:,:,:-1]+self.numeric_embedding(updated)-old_embedding,h[:,:,-1:]),2)
                            values=updated; states.append(updated)
                        else:
                            h,values=refine(h,values)
                    elif self.arm in ('mask_plain','pcfi_plain'):
                        h=self.extra(h)
                h,_,_=layer(h,feature_atten_mask=None,eval_pos=len(inp.context),layer_idx=index)
            h=t.encoder_out_norm(h)
            logits,_=t.y_decoder(h[:,len(inp.context):,-1],torch.zeros((1,len(ids)-len(inp.context),1),device='cuda'))
        finally:
            self.base._graph_holder.graph=None
        q=inverse[inp.query]-len(inp.context)
        result=logits[0,torch.as_tensor(q,device='cuda'),:int(inp.classes)].float()
        with torch.no_grad():
            self.last_diagnostics={'observed_clamp_max':max([float((z[observed]-xo[observed]).abs().max()) if observed.any() else 0. for z in states] or [0.]),
                'refinement_steps':len(states),'missing_state_variance':float(values[~observed].var(unbiased=False)) if (~observed).any() else 0.}
        return result


def predict(model, inp, **kwargs):
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        p=model(inp,**kwargs).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all()
    return p

"""Shared frozen graph core and immutable-source utilities for FP01."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
PYTHON='C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe'
spec=importlib.util.spec_from_file_location('fp01_readonly_edge',WORK/'factor_edge_only_benchmark/run.py')
E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E)
OLD_BP=E.BP


def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()
def setup():E.H.setup();torch.set_num_threads(4)
def real_rows():return E.rows()
def read_real(row):return E.read(row)
def tensor_input(inp,device='cuda',dtype=torch.float32):
    return dict(x=torch.as_tensor(inp['x'],device=device,dtype=dtype),
                observed=torch.as_tensor(inp['observed'],device=device,dtype=torch.bool),classes=int(inp['classes']))
def graph_identity(inp):
    h=hashlib.sha256()
    h.update(str((len(inp['x']),int(inp['classes']))).encode())
    for key in ['src','dst','context_ids','labels_context']:
        a=np.ascontiguousarray(inp[key]);h.update(key.encode());h.update(a.tobytes())
    return h.hexdigest()
def to_device(graph,device='cuda',dtype=None):
    return {k:v.to(device=device,dtype=dtype if v.is_floating_point() and dtype is not None else v.dtype)
            if isinstance(v,torch.Tensor) else v for k,v in graph.items()}


class GraphCore:
    def __init__(self,seed,device='cuda',dtype=torch.float32):
        self.seed=int(seed);self.device=torch.device(device);self.dtype=dtype
        self.base=E.H.R.load_model('factor',seed).to(device=self.device,dtype=dtype).eval()
        for p in self.base.parameters():p.requires_grad_(False)
        self.initial=digest(self.base)
        self.checkpoint=WORK/f'factor_real_graph_pilot/results/factor_s{seed}/checkpoint.pt'
        self.checkpoint_sha=sha(self.checkpoint)

    def check(self):
        assert digest(self.base)==self.initial,'Inherited graph checkpoint mutated'
        assert sha(self.checkpoint)==self.checkpoint_sha
        return self.initial

    def encode(self,inp,graph_arm='full',chunk=128):
        assert graph_arm in ['full','edge']
        t=E.D.tensor(inp,device=str(self.device),dtype=self.dtype)
        # Erase every feature before extracting graph-only role weights.
        t['x']=torch.zeros_like(t['x']);t['observed']=torch.zeros_like(t['observed'])
        n=len(t['x']);a=t['adj'];c=int(t['classes'])
        with torch.no_grad():
            j,valid,context,constant,edge_constant,error=E.factors(self.base,t)
            j=j[:,:,:n,:n].contiguous();valid=valid[:,:n,:n].contiguous();context=context[:,:n].contiguous()
            if graph_arm=='edge':
                edge=a>0;j=j*edge[None,None];valid=edge[None];constant=edge_constant
            node,logz=OLD_BP(j,valid,context,c,iterations=24,damping=.5,chunk=chunk)
            weights=(logz[0]+constant).softmax(-1)
            p0=torch.einsum('h,hnc->nc',weights.to(node.dtype),node[0])
            active=torch.nonzero(weights.to(j.dtype)!=0,as_tuple=False).flatten()
            degree=a.sum(-1);density=(a.sum()/max(n*(n-1),1)).clamp(1e-6,.95)
            features=torch.stack((degree/n,torch.log1p(degree)/np.log1p(n),density.expand(n),
                a.new_full((n,),1/c),a.new_full((n,),1/n),torch.ones_like(degree)),-1)
            z=self.base.stem(features)
            for block in self.base.blocks:
                neighbor=(a@z)/degree[:,None].clamp_min(1)
                other=(z.sum(0)-z-a@z)/(n-1-degree)[:,None].clamp_min(1)
                z=z+.2*block(torch.cat((z,neighbor,other),-1))
            out=dict(j=j[:,active],valid=valid,context=context,weights=weights[active],p0=p0,z=z,
                     classes=c,graph_arm=graph_arm,constant=constant,logz=logz[0],
                     graph_identity=graph_identity(inp),checkpoint_sha=self.checkpoint_sha,
                     full_role_weights=weights,active_role_ids=active)
        assert all(not v.requires_grad for v in out.values() if isinstance(v,torch.Tensor))
        return out


def source_lock(create=False):
    inherited=E.lock()
    paths=[ROOT/n for n in ['common.py','operators.py','checks.py','bank.py','experiment.py','analysis.py','protocol.json','PLAN_v2.md',
                            'IMPLEMENTATION_API.md','IMPLEMENTATION_NOTES.md']]
    paths += [ROOT/'bank/train_manifest.json',ROOT/'bank/dev_manifest.json',
              WORK/'factor_edge_only_benchmark/results/lock.json',WORK/'factor_homophily_benchmark/results/lock.json']
    state=dict(files={str(p):sha(p) for p in paths},inherited=inherited,
               graph_checkpoints={str(WORK/f'factor_real_graph_pilot/results/factor_s{s}/checkpoint.pt'):
                                  sha(WORK/f'factor_real_graph_pilot/results/factor_s{s}/checkpoint.pt') for s in [11,12,13]})
    path=ROOT/'results/lock.json'
    if not path.exists():
        assert create
        checks=json.loads((ROOT/'results/checks.json').read_text())
        assert checks['status']=='PASS' and checks['resource']['status']=='PASS'
        for tested,value in checks['tested_sources'].items():
            assert sha(tested)==value,('Numerical code changed after checks',tested)
        dump(path,state)
    assert json.loads(path.read_text())==state,'FP01 locked source/bank changed'
    return state

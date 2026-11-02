import sys
sys.dont_write_bytecode=True
import sklearn
import json,time
import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import replace
from core import ROOT,C,cfg,dump,sha,load_episode,Front,adjacency

def label_prop(inp,sign=1):
    n=len(inp.observed);a=adjacency((inp.src,inp.dst),n);p=sp.diags(1/np.asarray(a.sum(1)).ravel())@a
    yy=np.eye(int(inp.classes))[inp.labels_context];prior=(yy.sum(0)+1)/(len(yy)+int(inp.classes))
    f=np.broadcast_to(prior,(n,int(inp.classes))).copy();f[inp.context]=yy
    for _ in range(100):
        f=.1*prior+.9*(prior+sign*(p@(f-prior)));f[inp.context]=yy
    f=np.maximum(f,1e-8);f/=f.sum(1,keepdims=True)
    return f[inp.query],np.broadcast_to(prior,(len(inp.query),len(prior))).copy()

def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real']
    model=C.Model('pcfi_plain',11).eval();before=C.parameter_digest(model);front=Front(11).to('cuda')
    records=[];out=ROOT/'results/public_baselines';out.mkdir(exist_ok=True);st=time.perf_counter()
    for i,row in enumerate(rows):
        inp,ex,y=load_episode(row);missing=~inp.observed[inp.query].any(1)
        with torch.no_grad():comp,_=front(inp,ex,'pcfi100')
        native=np.load(ROOT/'bank'/row['path'])['native_pcfi']
        lp,prior=label_prop(inp,1);negative,_=label_prop(inp,-1)
        for name,values in [('public_mask',inp.observed_values),('public_pcfi100',comp),('public_native_pcfi100',native),('lp',None),('negative_lp',None),('context_prior',None)]:
            if values is not None:p=C.predict(model,replace(inp,pcfi=values))
            else:p={'lp':lp,'negative_lp':negative,'context_prior':prior}[name]
            dest=out/f'{name}_{i}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
            records.append({**row,'arm':name,'seed':11,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),
                 'all':C.metrics(p,y),'missing':C.metrics(p[missing],y[missing]) if missing.any() else None})
        if (i+1)%10==0:print('BASELINES',i+1,len(rows),flush=True)
    assert C.parameter_digest(model)==before
    dump(ROOT/'results/public_baselines.json',records);dump(ROOT/'results/public_baseline_audit.json',{'unchanged_sha256':before,'real_updates':0,'seconds':time.perf_counter()-st,
         'scope':'Shared RP16/context wrapper, one public initialization/forward. Not official full preprocessing/10-member ensemble; LP/negativeLP are fixed controls, not CPGNN/CLP reproduction.'})

if __name__=='__main__':main()

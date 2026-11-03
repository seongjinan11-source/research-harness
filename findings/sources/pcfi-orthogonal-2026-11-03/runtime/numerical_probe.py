import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sklearn
import json,gc
import numpy as np
import torch
from core import ROOT,C,dump,load_episode,load_model,tensor_extra,geometry

C.setup();torch.set_num_threads(2)
rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes'];out=[]
for arm in ['edge','node']:
    obj=load_model(arm,11);before=C.parameter_digest(obj)
    for split,ds in [('dev',None),('real','Cora')]:
        row=next(r for r in rows if r['split']==split and r['case']=='uniform90' and (ds is None or (r['graph']==ds and r.get('repetition',0)==0)))
        inp,ex,_=load_episode(row);xo=torch.as_tensor(inp.observed_values,device='cuda');obs=torch.as_tensor(inp.observed,device='cuda');te=tensor_extra(ex)
        with torch.no_grad():
            v,_=obj.front(xo,obs,te,arm,identity=True);base,_=obj.front(xo,obs,te,'pcfi')
            probe=[]
            for _ in range(5):
                repeat,_=obj.front(xo,obs,te,arm,identity=True);probe.append(float((repeat-v).abs().max()))
            oncpu={k:t.cpu() for k,t in te.items()};cpu_front=__import__('core').Front(11);vc,_=cpu_front(xo.cpu(),obs.cpu(),oncpu,arm,identity=True);vb,_=cpu_front(xo.cpu(),obs.cpu(),oncpu,'pcfi')
        out.append({'arm':arm,'split':split,'graph':row['graph'],'gpu_identity_pcfi_max_abs':float((v-base).abs().max()),
                    'gpu_identity_repeats_max_abs':probe,'cpu_identity_exact':bool(torch.equal(vc,vb)),
                    'cpu_gpu_identity_max_abs':float((vc-v.cpu()).abs().max())})
    assert C.parameter_digest(obj)==before;del obj;gc.collect();torch.cuda.empty_cache()
dump(ROOT/'results/numerical_probe.json',out);print(json.dumps(out,indent=2),flush=True)

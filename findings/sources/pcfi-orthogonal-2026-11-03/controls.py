import sys
sys.dont_write_bytecode=True
import sklearn
import json,copy,gc,time
import numpy as np
import torch
from core import ROOT,C,dump,sha,load_episode,load_model,predict,geometry

def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['case']!='full']
    out=[];folder=ROOT/'results/controls';folder.mkdir(exist_ok=True)
    for arm in ['pcfi','edge','node']:
        while arm!='pcfi' and not (ROOT/'results'/f'{arm}_s11'/'training.json').exists():
            print('WAIT FOR CHECKPOINT',arm,flush=True);time.sleep(5)
        obj=load_model(arm,11);before=C.parameter_digest(obj)
        for i,row in enumerate(rows):
            inp,ex,y=load_episode(row);missing=~inp.observed[inp.query].any(1)
            for control in ['no_features','shuffle_features','shuffle_context']+(['identity_transport','drop_cross_stalk'] if arm!='pcfi' else []):
                z=copy.deepcopy(inp);rng=np.random.default_rng(2037110300+i);identity=False
                if control=='no_features':z.observed[:]=False;z.observed_values[:]=0;z.pcfi[:]=0;ex0=geometry(z)
                elif control=='shuffle_features':
                    sources=np.flatnonzero(z.observed.any(1));z.observed_values[sources]=z.observed_values[rng.permutation(sources)];ex0=geometry(z)
                elif control=='shuffle_context':z.labels_context=rng.permutation(z.labels_context);ex0=geometry(z)
                else:ex0=ex
                if control=='identity_transport':identity=True
                if control=='drop_cross_stalk':
                    # Keep this checkpoint's learned angle function; only zero its sin.
                    saved=obj.front.angles
                    if arm=='node':
                        obj.front.angles=lambda e,k,fn=saved:fn(e,'node')
                    obj.kind='scalar'
                p=predict(obj,z,ex0,identity=identity)
                if control=='drop_cross_stalk':obj.kind=arm;obj.front.angles=saved
                dest=folder/f'{arm}_{i}_{control}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
                out.append({**row,'arm':arm,'seed':11,'control':control,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),'original_missing':C.metrics(p[missing],y[missing])})
            if (i+1)%12==0:print('CONTROL',arm,i+1,flush=True)
        assert C.parameter_digest(obj)==before;del obj;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/controls.json',out);print('CONTROLS COMPLETE',len(out),flush=True)

if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import sklearn
import copy,json,gc
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,load_episode,load_model,predict,geometry,Front
from data import read_real

def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['case']!='full']
    outputs=[];out=ROOT/'results/controls';out.mkdir(exist_ok=True)
    for kind in ['pcfi100','sheaf']:
        model=load_model(kind,11);before=C.parameter_digest(model)
        for i,row in enumerate(rows):
            inp,ex,y=load_episode(row);missing=~inp.observed[inp.query].any(1)
            for name in ['no_features','shuffled_sources']:
                z=copy.deepcopy(inp)
                if name=='no_features':z.observed[:]=False;z.observed_values[:]=0
                else:
                    sources=np.flatnonzero(z.observed[:,0]);r=np.random.default_rng(2036110300+i)
                    z.observed_values[sources]=z.observed_values[r.permutation(sources)]
                zz=geometry(z.src,z.dst,z.observed,z.observed_values,z.context,z.labels_context,z.classes)
                p=predict(model,z,zz);dest=out/f'{kind}_{i}_{name}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
                outputs.append({**row,'arm':kind,'control':name,'seed':11,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),
                                'original_missing':C.metrics(p[missing],y[missing]),'observed_sources_before':int(inp.observed[:,0].sum())})
            if (i+1)%16==0:print('CONTROLS',kind,i+1,flush=True)
        assert C.parameter_digest(model)==before
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/controls.json',outputs)
    # Numeric reconstruction on the compressed RP16 interface. Hidden truth is evaluation-only.
    full={name:read_real(name)['x'] for name in cfg()['real_datasets']};reconstruction=[]
    for kind in cfg()['arms']:
        front=Front(11).to('cuda');st=torch.load(ROOT/'results'/f'{kind}_s11/checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
        front.load_state_dict({k.removeprefix('front.'):v for k,v in st.items() if k.startswith('front.')})
        for row in rows:
            inp,ex,_=load_episode(row);truth=full[row['graph']]
            safe=np.where(inp.observed,truth,0.);count=inp.observed.sum(0);rms=np.sqrt(np.divide((safe*safe).sum(0),count,out=np.ones(truth.shape[1]),where=count>0));rms=np.where(rms>=1e-8,rms,1.)
            target=truth/rms
            with torch.no_grad():values,info=front(inp,ex,kind)
            values=values.cpu().numpy();mask=~inp.observed
            reconstruction.append({**row,'arm':kind,'seed':11,'missing_projected_mse':float(((values-target)[mask]**2).mean()),
                'observed_max_change':float(np.max(np.abs(values[inp.observed]-inp.observed_values[inp.observed]))) if inp.observed.any() else 0.,
                'scope':'RP16 numeric reconstruction in observed-only RMS coordinates; hidden target used after prediction only; not raw original feature-space MSE.'})
    dump(ROOT/'results/reconstruction.json',reconstruction)
    print('CONTROLS COMPLETE',len(outputs),len(reconstruction),flush=True)

if __name__=='__main__':main()

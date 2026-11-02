import sys
sys.dont_write_bytecode=True
import sklearn
import json,copy,gc
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,load_episode,load_model,predict,adjacency,impute
def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['case']!='full'];output=[]
    out=ROOT/'results/controls';out.mkdir(exist_ok=True)
    for arm in ['pcfi100','dynamic']:
        model=load_model(arm,11);before=C.parameter_digest(model)
        for index,row in enumerate(rows):
            inp,y=load_episode(row);missing=~inp.observed[inp.query].any(1)
            for control in ['no_features','shuffle_features','shuffle_context']+(['freeze_geometry'] if arm=='dynamic' else []):
                z=copy.deepcopy(inp);rng=np.random.default_rng(2037110300+index)
                if control=='no_features':z.observed[:]=False;z.observed_values[:]=0;z.pcfi[:]=0
                elif control=='shuffle_features':
                    sources=np.flatnonzero(z.observed.any(1));z.observed_values[sources]=z.observed_values[rng.permutation(sources)]
                    if arm=='pcfi100':z.pcfi=impute(adjacency((z.src,z.dst),len(z.observed)),np.where(z.observed,z.observed_values,np.nan),'pcfi','structural',iterations=100)[0].astype('float32')
                elif control=='shuffle_context':z.labels_context=rng.permutation(z.labels_context)
                if control=='freeze_geometry':model.kind='static'
                p=predict(model,z);model.kind=arm;dest=out/f'{arm}_{index}_{control}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
                output.append({**row,'arm':arm,'seed':11,'control':control,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),'original_missing':C.metrics(p[missing],y[missing])})
            if (index+1)%12==0:print('CONTROL',arm,index+1,flush=True)
        assert C.parameter_digest(model)==before;del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/controls.json',output);print('CONTROLS COMPLETE',len(output),flush=True)
if __name__=='__main__':main()

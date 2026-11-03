import sys
sys.dont_write_bytecode=True
import sklearn
import copy,json
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,load_episode,load_model,predict

def main():
    C.setup();torch.set_num_threads(2);obj=load_model(11);before=C.parameter_digest(obj)
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['case']!='full']
    folder=ROOT/'results/controls';folder.mkdir(exist_ok=True);out=[]
    for i,row in enumerate(rows):
        inp,y=load_episode(row);missing=~inp.observed[inp.query].any(1)
        for control in ['no_features','shuffle_features','shuffle_context','freeze_frames','identity_frames']:
            z=copy.deepcopy(inp);rng=np.random.default_rng(2037110300+i)
            if control=='no_features':z.observed[:]=False;z.observed_values[:]=0;z.pcfi[:]=0
            elif control=='shuffle_features':
                sources=np.flatnonzero(z.observed.any(1));z.observed_values[sources]=z.observed_values[rng.permutation(sources)]
            elif control=='shuffle_context':z.labels_context=rng.permutation(z.labels_context)
            obj.kind='static' if control=='freeze_frames' else 'identity' if control=='identity_frames' else 'dynamic'
            p=predict(obj,z);obj.kind='dynamic';dest=folder/f'{i}_{control}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
            out.append({**row,'arm':'node_frame','seed':11,'control':control,'prediction':dest.relative_to(ROOT).as_posix(),
                        'prediction_sha256':sha(dest),'original_missing':C.metrics(p[missing],y[missing])})
        if (i+1)%12==0:print('CONTROL node_frame',i+1,flush=True)
    assert C.parameter_digest(obj)==before
    dump(ROOT/'results/controls.json',out);print('CONTROLS COMPLETE',len(out),flush=True)

if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import sklearn
import json,importlib.util
import numpy as np
from core import ROOT,WORK,C,cfg,dump,sha,impute,adjacency
spec=importlib.util.spec_from_file_location('nsd_previous_core',WORK/'sheaf_pcfi_pilot/core.py')
OC=importlib.util.module_from_spec(spec);sys.modules[spec.name]=OC;spec.loader.exec_module(OC)
spec=importlib.util.spec_from_file_location('nsd_previous_data',WORK/'sheaf_pcfi_pilot/data.py')
OLD=importlib.util.module_from_spec(spec);sys.modules[spec.name]=OLD
saved_core=sys.modules['core'];sys.modules['core']=OC
try:spec.loader.exec_module(OLD)
finally:sys.modules['core']=saved_core
def save(g,observed,context,query,seed,dest):
    xo=C.observed_scale(g['x'],observed);a=adjacency((g['src'],g['dst']),len(xo))
    pcfi=impute(a,np.where(observed,xo,np.nan),'pcfi','structural',iterations=100)[0].astype('float32')
    inp=C.Inputs(xo,observed,pcfi,g['src'],g['dst'],context,g['y'][context],query,g['classes'],seed)
    np.savez_compressed(dest,**{k:np.asarray(getattr(inp,k)) for k in C.Inputs.__annotations__},labels_query=g['y'][query])
def prepare():
    path=ROOT/'bank/manifest.json';assert not path.exists();folder=path.parent;folder.mkdir(exist_ok=True)
    old=json.loads((WORK/'sheaf_pcfi_pilot/bank/manifest.json').read_text());rows=[];raws=[]
    # Copy original visible input/context/query arrays; complete the PCFI cache with the original channel correction.
    for row in old['episodes']:
        if row['split'] in ['train','dev','real']:
            p=WORK/'sheaf_pcfi_pilot/bank'/row['path'];assert sha(p)==row['sha256']
            z=np.load(p);data={k:z[k] for k in z.files if k in C.Inputs.__annotations__ or k in ['labels_query','native_pcfi']}
            aa=adjacency((data['src'],data['dst']),len(data['observed']))
            data['pcfi']=impute(aa,np.where(data['observed'],data['observed_values'],np.nan),'pcfi','structural',iterations=100)[0].astype('float32')
            dest=folder/('reuse_'+p.name);np.savez_compressed(dest,**data)
            rows.append({**row,'bank_path':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest),'original_path':str(p),'original_sha256':sha(p)})
    print('BANK reused visible inputs',len(rows),flush=True)
    for split,count,start in [('train',64,None),('dev',2,None),('test',64,2037111300)]:
        for i in range(count):
            if split!='test':
                p=WORK/f'sheaf_pcfi_pilot/bank/{split}_raw_{i}.npz';z=np.load(p)
                template=next(r for r in old['episodes'] if r['split']==split and r['graph']==i)
                g={k:z[k] for k in z.files};g['classes']=int(g['classes']);g['label_mode']=template['label_mode'];g['feature_mode']=template['feature_mode']
                cases=['sources2','sources8'];raws.append({'split':split,'graph':i,'path':str(p),'sha256':sha(p)})
            else:
                g=OLD.synthetic(start+i,i);p=folder/f'test_raw_{i}.npz';np.savez_compressed(p,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']});raws.append({'split':split,'graph':i,'path':str(p),'sha256':sha(p)});cases=cfg()['test_cases']
            for ci,case in enumerate(cases):
                seed=2037211300+10000*['train','dev','test'].index(split)+100*i+ci
                context=OLD.contexts(g,seed+999,[1,4][(i+ci)%2]);query=np.setdiff1d(np.arange(len(g['y'])),context)
                if case.startswith('sources'):
                    sources=np.random.default_rng(seed).permutation(len(query)+len(context))[:int(case.removeprefix('sources'))]
                    obs=np.broadcast_to(np.isin(np.arange(len(g['y'])),sources)[:,None],g['x'].shape).copy()
                else:obs=OLD.mask(g,case,seed)
                dest=folder/f'{split}_{i}_{case}.npz';save(g,obs,context,query,seed,dest)
                rows.append({'split':split,'graph':i,'case':case,'bank_path':str(dest.relative_to(ROOT)).replace('\\','/'),'sha256':sha(dest),
                    'label_mode':g['label_mode'],'feature_mode':g['feature_mode'],'classes':g['classes'],'observed_nodes':int(obs[:,0].sum()),
                    'context_feature_overlap':int(obs[context,0].sum()),'label_agreement':float(g['label_agreement']),'feature_energy':float(g['feature_energy'])})
        print('BANK',split,flush=True)
    assert len([r for r in rows if r['split']=='train'])==512
    dump(path,{'episodes':rows,'raw':raws,'real_sources':old['real_sources'],'previous_manifest_sha256':sha(WORK/'sheaf_pcfi_pilot/bank/manifest.json')})
if __name__=='__main__':prepare()

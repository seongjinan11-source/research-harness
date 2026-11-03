import sys
sys.dont_write_bytecode=True
import sklearn
import json,importlib.util
import numpy as np
from core import ROOT,WORK,N,C,cfg,dump,sha

def prepare():
    folder=ROOT/'bank';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'manifest.json').exists()
    old=json.loads((WORK/'nsd_token_pfn/bank/manifest.json').read_text())
    rows=[]
    for r in old['episodes']:
        if r['split'] in ['train','dev','real']:
            p=WORK/'nsd_token_pfn'/r['bank_path'];assert sha(p)==r['sha256']
            rows.append({**r,'bank_path':p.relative_to(WORK).as_posix()})
    path=WORK/'nsd_token_pfn/bank.py'
    sp=importlib.util.spec_from_file_location('node_frame_reference_bank',path)
    module=importlib.util.module_from_spec(sp);saved=sys.modules['core'];sys.modules['core']=N
    try:sp.loader.exec_module(module)
    finally:sys.modules['core']=saved
    raws=[]
    for i in range(cfg()['test_graphs']):
        g=module.OLD.synthetic(cfg()['test_seed_start']+i,i)
        p=folder/f'test_raw_{i}.npz';np.savez_compressed(p,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']})
        raws.append({'graph':i,'path':p.relative_to(WORK).as_posix(),'sha256':sha(p)})
        for ci,case in enumerate(cfg()['test_cases']):
            seed=2038211300+100*i+ci
            context=module.OLD.contexts(g,seed+999,[1,4][(i+ci)%2]);query=np.setdiff1d(np.arange(len(g['y'])),context)
            if case.startswith('sources'):
                sources=np.random.default_rng(seed).permutation(len(g['y']))[:int(case.removeprefix('sources'))]
                obs=np.broadcast_to(np.isin(np.arange(len(g['y'])),sources)[:,None],g['x'].shape).copy()
            else:obs=module.OLD.mask(g,case,seed)
            dest=folder/f'test_{i}_{case}.npz';module.save(g,obs,context,query,seed,dest)
            rows.append({'split':'test','graph':i,'case':case,'bank_path':dest.relative_to(WORK).as_posix(),'sha256':sha(dest),
                         'label_mode':g['label_mode'],'feature_mode':g['feature_mode'],'classes':g['classes'],
                         'observed_nodes':int(obs[:,0].sum()),'label_agreement':g['label_agreement'],'feature_energy':g['feature_energy']})
    assert len([r for r in rows if r['split']=='train'])==512
    dump(folder/'manifest.json',{'episodes':rows,'raw':raws,'old_manifest_sha256':sha(WORK/'nsd_token_pfn/bank/manifest.json')})
    print('BANK COMPLETE',len(rows),flush=True)

if __name__=='__main__':prepare()

import sys
sys.dont_write_bytecode=True
import sklearn
import importlib.util,json
import numpy as np
from core import ROOT,WORK,N,C,cfg,dump,sha,geometry

def prepare():
    folder=ROOT/'bank';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'manifest.json').exists()
    old=json.loads((WORK/'nsd_token_pfn/bank/manifest.json').read_text());rows=[];raw=[]
    for i,r in enumerate(old['episodes']):
        if r['split'] not in ['train','dev','real']:continue
        p=WORK/'nsd_token_pfn'/r['bank_path'];assert sha(p)==r['sha256'];z=np.load(p)
        inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__});ex=geometry(inp)
        dest=folder/f'ex_reuse_{i}.npz';np.savez_compressed(dest,**ex)
        rows.append({**r,'bank_path':p.relative_to(WORK).as_posix(),'extra_path':dest.relative_to(ROOT).as_posix(),'extra_sha256':sha(dest)})
    sp=importlib.util.spec_from_file_location('pcfi_orth_reference_bank',WORK/'nsd_token_pfn/bank.py');module=importlib.util.module_from_spec(sp)
    saved=sys.modules['core'];sys.modules['core']=N
    try:sp.loader.exec_module(module)
    finally:sys.modules['core']=saved
    for i in range(cfg()['test_graphs']):
        g=module.OLD.synthetic(cfg()['test_seed_start']+i,i)
        p=folder/f'test_raw_{i}.npz';np.savez_compressed(p,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']})
        raw.append({'graph':i,'path':p.relative_to(WORK).as_posix(),'sha256':sha(p)})
        for ci,case in enumerate(cfg()['test_cases']):
            seed=2039211300+100*i+ci;context=module.OLD.contexts(g,seed+999,[1,4][(i+ci)%2]);query=np.setdiff1d(np.arange(len(g['y'])),context)
            if case.startswith('sources'):
                sources=np.random.default_rng(seed).permutation(len(g['y']))[:int(case.removeprefix('sources'))]
                obs=np.broadcast_to(np.isin(np.arange(len(g['y'])),sources)[:,None],g['x'].shape).copy()
            else:obs=module.OLD.mask(g,case,seed)
            dest=folder/f'test_{i}_{case}.npz';module.save(g,obs,context,query,seed,dest)
            z=np.load(dest);inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__});ex=geometry(inp)
            exdest=folder/f'ex_test_{i}_{case}.npz';np.savez_compressed(exdest,**ex)
            rows.append({'split':'test','graph':i,'case':case,'bank_path':dest.relative_to(WORK).as_posix(),'sha256':sha(dest),
                         'extra_path':exdest.relative_to(ROOT).as_posix(),'extra_sha256':sha(exdest),'label_mode':g['label_mode'],
                         'feature_mode':g['feature_mode'],'classes':g['classes'],'observed_nodes':int(obs[:,0].sum()),
                         'label_agreement':g['label_agreement'],'feature_energy':g['feature_energy']})
        if (i+1)%16==0:print('BANK TEST',i+1,flush=True)
    assert len([r for r in rows if r['split']=='train'])==512
    dump(folder/'manifest.json',{'episodes':rows,'raw':raw,'old_manifest_sha256':sha(WORK/'nsd_token_pfn/bank/manifest.json')})
    print('BANK COMPLETE',len(rows),flush=True)

if __name__=='__main__':prepare()

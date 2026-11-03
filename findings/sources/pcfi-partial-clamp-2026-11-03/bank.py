import sys
sys.dont_write_bytecode = True
import sklearn
import importlib.util
import json
from collections import Counter
import numpy as np
from core import ROOT,WORK,B,R,C,cfg,dump,sha


def prepare():
    target = ROOT/'bank/manifest.json'
    assert not target.exists(),'Do not overwrite bank'
    target.parent.mkdir(parents=True,exist_ok=True)
    old = json.loads((R.ROOT/'bank/manifest.json').read_text())
    rows,raws = [],[]
    for row in old['episodes']:
        if row['split'] in ['train','real']:
            assert sha(WORK/row['bank_path'])==row['sha256']
            assert sha(WORK/row['extra_path'])==row['extra_sha256']
            rows.append(dict(row))
    spec = importlib.util.spec_from_file_location('partial_reference_bank',WORK/'nsd_token_pfn/bank.py')
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules['core']
    sys.modules['core'] = B.N
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules['core'] = saved
    for index in range(cfg()['development_graphs']):
        g = module.OLD.synthetic(cfg()['development_seed_start']+index,index)
        raw = ROOT/'bank'/f'dev_raw_{index}.npz'
        np.savez_compressed(raw,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']})
        raws.append({'graph':index,'path':raw.relative_to(WORK).as_posix(),'sha256':sha(raw),
                     'label_mode':g['label_mode'],'feature_mode':g['feature_mode']})
        for k in cfg()['context_per_class']:
            context = module.OLD.contexts(g,2041211300+100*index,k)
            query = np.setdiff1d(np.arange(len(g['y'])),context)
            for ci,case in enumerate(cfg()['cases']):
                seed = 2041311300+100*index+ci
                if case.startswith('sources'):
                    sources = np.random.default_rng(seed).permutation(len(g['y']))[:int(case[7:])]
                    observed = np.broadcast_to(np.isin(np.arange(len(g['y'])),sources)[:,None],g['x'].shape).copy()
                else:
                    observed = module.OLD.mask(g,case,seed)
                path = ROOT/'bank'/f'dev_{index}_k{k}_{case}.npz'
                module.save(g,observed,context,query,seed,path)
                with np.load(path) as z:
                    inp = C.Inputs(**{key:z[key] for key in C.Inputs.__annotations__})
                ex = ROOT/'bank'/f'dev_{index}_k{k}_{case}_extra.npz'
                np.savez_compressed(ex,**B.geometry(inp))
                rows.append({'split':'development','graph':index,'context_per_class':k,'case':case,
                    'bank_path':path.relative_to(WORK).as_posix(),'sha256':sha(path),
                    'extra_path':ex.relative_to(WORK).as_posix(),'extra_sha256':sha(ex),
                    'label_mode':g['label_mode'],'feature_mode':g['feature_mode'],'classes':g['classes'],
                    'observed_nodes':int(observed.any(1).sum()),'context_feature_overlap':int(observed[context].any(1).sum()),
                    'label_agreement':g['label_agreement'],'feature_energy':g['feature_energy']})
        if (index+1)%8==0:
            print('BANK',index+1,flush=True)
    counts = Counter((r['label_mode'],r['feature_mode']) for r in raws)
    assert len(counts)==4 and set(counts.values())=={8}
    assert Counter(r['split'] for r in rows)=={'train':512,'development':448,'real':60}
    dump(target,{'episodes':rows,'raw':raws,'old_manifest_sha256':sha(R.ROOT/'bank/manifest.json'),
                 'development_cells':{str(k):v for k,v in counts.items()},'status':'new_known_family_development_bank'})
    print('BANK COMPLETE',Counter(r['split'] for r in rows),flush=True)


if __name__=='__main__':
    prepare()

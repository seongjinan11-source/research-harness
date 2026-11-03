import sys
sys.dont_write_bytecode = True
import sklearn
import copy
import importlib.util
import json
from collections import Counter
import numpy as np
from core import ROOT,WORK,B,C,cfg,dump,sha


def previous_bank_module():
    spec = importlib.util.spec_from_file_location('review_reference_bank',WORK/'nsd_token_pfn/bank.py')
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules['core']
    sys.modules['core'] = B.N
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules['core'] = saved
    return module


def write_extra(inp, stem, permutation_seed):
    ex = B.geometry(inp)
    altered = copy.deepcopy(inp)
    altered.labels_context = np.random.default_rng(permutation_seed).permutation(inp.labels_context)
    shuffled = B.geometry(altered)
    for field in ['observed_values','observed','context','query']:
        assert np.array_equal(getattr(inp,field),getattr(altered,field))
    paths = {}
    for key,values in [('extra',ex),('shuffle_extra',shuffled)]:
        path = ROOT/'bank'/f'{stem}_{key}.npz'
        np.savez_compressed(path,**values)
        paths[f'{key}_path'] = path.relative_to(WORK).as_posix()
        paths[f'{key}_sha256'] = sha(path)
    paths['geometry_shuffle_label_sha256'] = __import__('hashlib').sha256(altered.labels_context.tobytes()).hexdigest()
    paths['geometry_shuffle_descriptor_max_abs'] = float(np.abs(ex['descriptor']-shuffled['descriptor']).max())
    return paths


def prepare():
    manifest = ROOT/'bank/manifest.json'
    assert not manifest.exists(), 'Do not overwrite an existing bank.'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    old = json.loads((B.ROOT/'bank/manifest.json').read_text())
    rows,raws = [],[]
    for i,row in enumerate(old['episodes']):
        if row['split'] not in ['train','real']:
            continue
        path = WORK/row['bank_path']
        assert sha(path) == row['sha256']
        extra = B.ROOT/row['extra_path']
        assert sha(extra) == row['extra_sha256']
        rewritten = {**row,'extra_path':extra.relative_to(WORK).as_posix()}
        if row['split']=='real':
            inp,_,_ = B.load_episode(row)
            rewritten.update(write_extra(inp,f'real_{i}',2040311300+i))
        rows.append(rewritten)
    module = previous_bank_module()
    for index in range(cfg()['development_graphs']):
        g = module.OLD.synthetic(cfg()['development_seed_start']+index,index)
        raw = ROOT/'bank'/f'dev_raw_{index}.npz'
        np.savez_compressed(raw,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']})
        raws.append({'graph':index,'path':raw.relative_to(WORK).as_posix(),'sha256':sha(raw),
                     'label_mode':g['label_mode'],'feature_mode':g['feature_mode']})
        for k in cfg()['context_per_class']:
            context = module.OLD.contexts(g,2040211300+100*index,k)
            query = np.setdiff1d(np.arange(len(g['y'])),context)
            for ci,case in enumerate(cfg()['cases']):
                seed = 2040411300+100*index+ci
                if case.startswith('sources'):
                    sources = np.random.default_rng(seed).permutation(len(g['y']))[:int(case.removeprefix('sources'))]
                    observed = np.broadcast_to(np.isin(np.arange(len(g['y'])),sources)[:,None],g['x'].shape).copy()
                else:
                    observed = module.OLD.mask(g,case,seed)
                path = ROOT/'bank'/f'dev_{index}_k{k}_{case}.npz'
                module.save(g,observed,context,query,seed,path)
                with np.load(path) as z:
                    inp = C.Inputs(**{key:z[key] for key in C.Inputs.__annotations__})
                extras = write_extra(inp,f'dev_{index}_k{k}_{case}',2040511300+100*index+k)
                rows.append({'split':'development','graph':index,'context_per_class':k,'case':case,
                             'bank_path':path.relative_to(WORK).as_posix(),'sha256':sha(path),**extras,
                             'label_mode':g['label_mode'],'feature_mode':g['feature_mode'],
                             'classes':g['classes'],'observed_nodes':int(observed.any(1).sum()),
                             'context_feature_overlap':int(observed[context].any(1).sum()),
                             'label_agreement':g['label_agreement'],'feature_energy':g['feature_energy']})
        if (index+1)%8==0:
            print('BANK',index+1,flush=True)
    cells = Counter((r['label_mode'],r['feature_mode']) for r in raws)
    assert len(cells)==4 and len(set(cells.values()))==1
    assert len([r for r in rows if r['split']=='train'])==512
    dump(manifest,{'episodes':rows,'raw':raws,'old_manifest_sha256':sha(B.ROOT/'bank/manifest.json'),
                   'development_cells':{str(k):v for k,v in cells.items()},
                   'status':'development_only_no_independent_real_confirmation'})
    print('BANK COMPLETE',Counter(r['split'] for r in rows),flush=True)


if __name__=='__main__':
    prepare()

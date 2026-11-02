"""Audit correction: compare interventions on the primary missing-query cohort."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import probe as P

primary={};zero={};full={}
for seed in [11,12,13]:
    records=json.loads((root/'results'/f'pcfi_plain_s{seed}'/'evaluation.json').read_text())
    for r in records:
        if r['label_law']!=1 or r['feature_law']!=1 or r['k']!=4:continue
        if r['sources']==8:primary[(r['graph'],seed,r['variant'])]=r
        if r['sources']==0 and r['variant']=='ordinary':zero[(r['graph'],seed)]=r
        if r['sources']=='full' and r['variant']=='ordinary':full[(r['graph'],seed)]=r
vectors={name:[] for name in ['pcfi','known_signed','zero_source','full']}
for gid in range(32):
    perseed={name:[] for name in vectors}
    for seed in [11,12,13]:
        r=primary[(gid,seed,'ordinary')]
        inp,y,_=P.read(r);cohort=~inp.observed[inp.query].any(1)
        for name,row in [('pcfi',r),('known_signed',primary[(gid,seed,'known_signed')]),('zero_source',zero[(gid,seed)]),('full',full[(gid,seed)])]:
            other,other_y,_=P.read(row)
            assert np.array_equal(inp.query,other.query) and np.array_equal(y,other_y)
            with np.load(root/row['prediction']) as z:
                assert np.array_equal(z['y'],y)
                m=P.C.metrics(z['p'][cohort],y[cohort])
            perseed[name].append([m[k] for k in P.METRICS])
    for name in vectors:vectors[name].append(np.mean(perseed[name],axis=0))
vectors={k:np.asarray(v) for k,v in vectors.items()}
contrasts={a+'-'+b:P.difference(vectors[a],vectors[b]) for a,b in [('known_signed','zero_source'),('pcfi','zero_source'),('full','pcfi')]}
old=json.loads((root/'results/statistics.json').read_text())
floor=all(d['accuracy']['ci95'][0]>0 for d in old['primary']['contrasts'].values()) and contrasts['known_signed-zero_source']['accuracy']['ci95'][0]>0
P.dump(root/'results/common_query_interventions.json',{
    'reason':'Original supplementary cross-budget differences used each budget\'s own missing-query subset (full used all queries). This confounds cohort changes with intervention. Restrict both predictions to the prespecified primary8-source missing-query set. No model, bank, saved probabilities or original statistics file was changed.',
    'original_statistics_sha256':P.sha(root/'results/statistics.json'),
    'original_primary_and_within_cell_comparisons_unchanged':True,
    'means':{k:dict(zip(P.METRICS,v.mean(0).tolist())) for k,v in vectors.items()},
    'contrasts':contrasts,'strong_floor_corrected':bool(floor),
    'full_reference_note':'Full input also observes context features and uses its own observed-only RMS. This is a full-input reference, not an isolated feature-value-only intervention.'})
print('COMMON_QUERY_CORRECTION',json.dumps(contrasts),flush=True)

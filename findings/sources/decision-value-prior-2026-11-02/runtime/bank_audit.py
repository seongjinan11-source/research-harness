import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiment as E
import family as FA
import numpy as np,json,time
from collections import defaultdict

def main():
    c=E.cfg();m=json.loads((ROOT/'bank/manifest.json').read_text());groups=defaultdict(list)
    for r in m['episodes']:
        inp,y=E.read(r);o=np.flatnonzero(inp.observed.any(1))
        assert len(o)==r['budget'] and inp.observed.sum()==2*r['budget']
        assert not set(inp.context)&set(inp.query)
        assert np.all(inp.observed_values[~inp.observed]==0)
        assert np.array_equal(inp.pcfi[inp.observed],inp.observed_values[inp.observed])
        assert np.isfinite(inp.observed_values).all() and np.isfinite(inp.pcfi).all()
        if r['condition']!='full':assert not set(o)&(set(inp.context)|set(inp.query))
        if r['split']=='train':groups[r['graph'],r['context'],r['budget']].append((r,inp,y))
    for _,rs in groups.items():
        assert len(rs)==4
        for _,inp,y in rs[1:]:
            first=rs[0][1];assert np.array_equal(inp.context,first.context) and np.array_equal(inp.query,first.query)
            assert np.array_equal(inp.labels_context,first.labels_context) and np.array_equal(y,rs[0][2])
    designs=json.loads((ROOT/'bank/designs.json').read_text());audit=[];start=time.perf_counter()
    # Fixed first16 train graphs; independent4096-draw values, descriptive only.
    for r in [r for r in designs if r['context']==8 and r['budget']==4 and r['graph']<16]:
        seed=c['train_seed_start']+r['graph'];g=FA.graph(seed);ds=seed+1000000+8*10000+4*100
        ctx,q,masks,geo=FA.designs(g,ds,8,4)
        assert np.array_equal(masks,r['masks'])
        j,v,se=FA.scores(g,ctx,q,masks,ds+50000000,c['score_audit_samples'])
        audit.append({'graph':r['graph'],'selected':{a:{'J':float(j[i]),'V':float(v[i]),'V_se':float(se[i])} for a,i in r['selected'].items()},
            'J_V_rank_correlation':float(np.corrcoef(np.argsort(np.argsort(j)),np.argsort(np.argsort(v)))[0,1])})
    means={a:{s:float(np.mean([r['selected'][a][s] for r in audit])) for s in ['J','V']} for a in ['random','geometry','entropy','decision']}
    E.write(ROOT/'results/bank_audit.json',{'status':'PASS','episodes':len(m['episodes']),'matched_training_groups':len(groups),
        'no_hidden_values_in_inputs':True,'observed_pcfi_clamp':True,'source_context_query_disjoint_except_full':True,
        'same_realization_across_priors':True,'independent_score_audit':audit,'independent_score_means':means,
        'audit_seconds':time.perf_counter()-start,'scope':'First16 train graphs primary design; independent Monte Carlo descriptive audit, no tuning or mask changes.'})
    print('BANK_AUDIT_OK',json.dumps(means),flush=True)
if __name__=='__main__':main()

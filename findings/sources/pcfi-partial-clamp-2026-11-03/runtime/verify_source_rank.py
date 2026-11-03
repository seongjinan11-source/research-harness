"""Post-protocol algebra check; no PFN execution, optimizer or model selection."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sklearn
import json
import numpy as np
import torch
from core import ROOT,transport_forward,dump


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(2041511300)
    checks = []
    for trial in range(24):
        n,f = 8,3
        adjacent = np.ones((n,n))
        disconnected = trial%2==0
        if disconnected:
            adjacent[:4,4:] = 0
            adjacent[4:,:4] = 0
        i,j = np.nonzero(adjacent)
        confidence = rng.uniform(.2,1,(n,f))
        denominator = adjacent@confidence
        ex = {'ii':torch.as_tensor(i),'jj':torch.as_tensor(j),
            'weights':torch.as_tensor(confidence[j]/denominator[i],dtype=torch.float64),
            'pc':torch.as_tensor(confidence,dtype=torch.float64)}
        obs = np.zeros((n,f),bool)
        for column in range(f):
            obs[rng.integers(0,4 if disconnected else n),column] = True
            if disconnected:
                obs[rng.integers(4,n),column] = True
        observed = torch.as_tensor(obs)
        x = torch.as_tensor(np.where(obs,rng.normal(size=(n,f)),0),dtype=torch.float64)
        theta = torch.as_tensor(rng.uniform(-np.pi,np.pi,(n,f)),dtype=torch.float64)
        full,_ = transport_forward(None,x,observed,ex,partial=False,theta=theta)
        partial,_ = transport_forward(None,x,observed,ex,partial=True,theta=theta)
        error = float((full-partial).abs().max())
        assert error<1e-12,error
        checks.append({'trial':trial,'components':2 if disconnected else 1,
                       'one_source_per_column_per_component':True,'max_abs':error})
    rows = json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    primary = [r for r in rows if r['split']=='development' and r['case'] in ['uniform90','uniform995','cluster90','cluster995']]
    counts = {'zero':sum(r['observed_nodes']==0 for r in primary),
              'one':sum(r['observed_nodes']==1 for r in primary),
              'two_or_more':sum(r['observed_nodes']>=2 for r in primary)}
    result = {'status':'PASS','post_protocol_mechanism_analysis':True,
        'no_pfn_or_optimizer':True,'checks':checks,'primary_episode_counts':counts,
        'scope':'Same fixed frames, at most one observed source per feature per component, zero initial second coordinate, scalar P shared across coordinates. All aligned states stay in the span of the source direction; releasing its orthogonal coordinate changes nothing. This is not equivalence between separately retrained frame networks or all multiple-source tasks.'}
    dump(ROOT/'results/source_rank_checks.json',result)
    print('SOURCE RANK CHECK PASS',max(c['max_abs'] for c in checks),counts)


if __name__=='__main__':
    main()

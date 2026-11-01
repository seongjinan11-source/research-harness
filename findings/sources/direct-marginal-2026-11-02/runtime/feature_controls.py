"""Additional descriptive known-family controls; primary locked gates stay unchanged.

Added after the primary design was locked, before reading the test summaries.
These controls can limit an adaptive-encoder claim; they cannot promote a new
success criterion or choose/retrain the learned model.
"""
import sys
sys.dont_write_bytecode=True
import sklearn  # Initialize Windows OpenMP before torch, as in the main workers.
from pathlib import Path
WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'direct_marginal_pilot'))
import json,math,time
import numpy as np
import torch
from experiment import ROOT,C,rows,read
from data import GridReference,tensors,dump
from model import TreeCircuit
from analyze import ci,kl,primary


def fixed_tree(items,engine):
    batch=tensors(items,'cuda',torch.float64);b,n,_=batch['x'].shape;h=len(engine.orient)
    parent=batch['parent'].clamp_min(0)
    edge=batch['adj'][torch.arange(b,device='cuda')[:,None,None],torch.arange(2,device='cuda')[None,:,None],torch.arange(n,device='cuda')[None,None,:],parent[:,None,:]].sum(1)>0
    parameters={'j':edge[:,None].expand(-1,h,-1).double()*math.log((1+.72)/(1-.72)),
        'logmix':batch['x'].new_full((b,h),-math.log(h)),'mean':engine.orient*.8,'scale':batch['x'].new_full((h,2),.6)}
    leaf=TreeCircuit.emissions(None,batch,parameters)
    return TreeCircuit.sum_product(batch,parameters,leaf)[0].cpu().numpy()


def main():
    torch.set_num_threads(2);test=rows('test');engine=GridReference('cuda');records=[]
    for start in range(0,len(test),32):
        rr=test[start:start+32];values=[read(r['path']) for r in rr];items=[v[0] for v in values]
        with torch.inference_mode():
            feature=engine.infer(items,np.tile([.25,0.],(len(items),1)),False)['grid6_exact']
            tree=fixed_tree(items,engine)
        for j,r in enumerate(rr):
            inp,q,y=values[j]
            with np.load(ROOT/'references'/(Path(r['path']).stem+'.npz')) as z:target=z['grid6_exact'][q]
            for name,p in [('known_feature_only',feature[j,q]),('known_constant_tree',tree[j,q])]:
                records.append({**r,'arm':name,'seed':0,'mode':'standard',**C.metrics(p,y),'grid6_kl':kl(target,p)})
        if start%1024==0:print('FEATURE_CONTROLS',start,flush=True)
    learned=[]
    for seed in [11,12,13]:learned+=json.loads((ROOT/'results'/f'circuit_s{seed}'/'evaluation.json').read_text())
    report={'status':'post-lock descriptive controls, not preregistered primary gates','knowledge':'Fixed Gaussian mean magnitude .8/sd .6 from the declared training family; constant-tree coupling uses the modal training signal .72. No test tuning or learned posterior head. Feature-only ignores G; constant tree uses G only to form its fixed tree.',
        'primary':[],'paired':{},'rows':records}
    for name in ['known_feature_only','known_constant_tree']:
        rr=[r for r in records if r['arm']==name and primary(r)]
        report['primary'].append({'arm':name,**{m:float(np.mean([r[m] for r in rr])) for m in ['accuracy','nll','brier','grid6_kl']}})
        for metric in ['accuracy','nll','brier']:
            dif=[]
            for g in sorted({r['graph'] for r in rr}):
                a=np.mean([r[metric] for r in learned if r['graph']==g and r['mode']=='standard' and primary(r)])
                b=np.mean([r[metric] for r in rr if r['graph']==g]);dif.append(a-b)
            report['paired'][f'circuit-{name}/{metric}']=ci(dif)
    dump(ROOT/'results/feature_controls.json',report)
    print('FEATURE_CONTROLS_DONE',flush=True)
if __name__=='__main__':main()

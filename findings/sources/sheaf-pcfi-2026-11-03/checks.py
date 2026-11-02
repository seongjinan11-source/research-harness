import sys
sys.dont_write_bytecode=True
import sklearn
import gc,json,copy,time
import numpy as np
import torch
from core import ROOT,C,cfg,dump,sha,Front,Model,load_episode,predict,geometry,adjacency,impute
from data import synthetic,episode,contexts
from trial import step

def main():
    print('CHECK imports ready',flush=True)
    C.setup();torch.set_num_threads(2)
    print('CHECK backend ready',flush=True)
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='dev']
    inp,ex,y=load_episode(rows[0]);front=Front(11).to('cuda')
    a=adjacency((inp.src,inp.dst),len(inp.observed));nan=np.where(inp.observed,inp.observed_values,np.nan)
    checks={}
    with torch.no_grad():
        for kind,iterations in [('pcfi100',100),('pcfi200',200)]:
            actual,_=front(inp,ex,kind);reference=impute(a,nan,'pcfi','structural',iterations=iterations)[0]
            checks[kind+'_scipy_max_abs']=float(np.max(np.abs(actual.cpu().numpy()-reference)))
            assert checks[kind+'_scipy_max_abs']<2e-4
        # Zero angle makes physical output exactly PCFI200; source second stalk stays zero.
        front.net[-1].weight.zero_();front.net[-1].bias.copy_(torch.tensor([0.,1.5,0.],device='cuda'))
        ref,_=front(inp,ex,'pcfi200')
        for kind in ['weighted','signed','sheaf']:
            actual,_=front(inp,ex,kind);err=float((actual-ref).abs().max());checks[kind+'_identity_error']=err;assert err<2e-4
        # No observed sources => zero numeric output, even if distances follow author's0 convention.
        z=copy.deepcopy(inp);z.observed[:]=False;z.observed_values[:]=0
        ez=geometry(z.src,z.dst,z.observed,z.observed_values,z.context,z.labels_context,z.classes)
        actual,_=front(z,ez,'sheaf');assert float(actual.abs().max())==0.;checks['zero_source_numeric_max']=0.
    g=synthetic(2036110300,0);con=contexts(g,2036119999,4);query=np.setdiff1d(np.arange(len(g['y'])),con)
    z1=episode(g,'uniform90',2036119990,con,query);poison=copy.deepcopy(g);mask=z1['observed']
    poison['x'][~mask]=12345.;poison['y'][query]=(poison['y'][query]+1)%g['classes']
    z2=episode(poison,'uniform90',2036119990,con,query)
    for key in C.Inputs.__annotations__:assert np.array_equal(z1[key],z2[key]),key
    for key in ['confidence','warm','weights','descriptors']:assert np.array_equal(z1[key],z2[key]),key
    checks['hidden_feature_query_label_poison']='PASS'
    renamed=copy.deepcopy(inp);perm=np.arange(int(inp.classes))[::-1];renamed.labels_context=perm[renamed.labels_context]
    ee=geometry(inp.src,inp.dst,inp.observed,inp.observed_values,inp.context,renamed.labels_context,inp.classes)
    checks['router_class_name_permutation_max']=float(np.max(np.abs(ee['descriptors']-ex['descriptors'])))
    assert checks['router_class_name_permutation_max']<1e-5
    # Actual public-PFN backward: verifies numeric imputer gradient reaches the router.
    smoke=[]
    for kind in ['pcfi100','weighted','signed','sheaf']:
        print('CHECK optimizer smoke',kind,flush=True)
        model=Model(kind,11);frozen=C.parameter_digest(model,trainable=False)
        opt=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':3e-5},
              {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':.001}],weight_decay=.01)
        sc=torch.amp.GradScaler('cuda',init_scale=64);model.train();before=[p.detach().clone() for p in model.front.parameters()]
        st=time.perf_counter();ls,gn,retry=step(model,opt,sc,inp,ex,y,0)
        delta=max(float((p-q).abs().max()) for p,q in zip(model.front.parameters(),before))
        assert C.parameter_digest(model,trainable=False)==frozen
        if kind!='pcfi100':assert delta>0
        pr=predict(model,inp,ex);assert np.isfinite(pr).all()
        smoke.append(dict(arm=kind,loss=ls,grad_norm=gn,retries=retry,router_update_max=delta,seconds=time.perf_counter()-st))
        del model,opt;gc.collect();torch.cuda.empty_cache()
    checks['smoke_discarded']=smoke;checks['status']='PASS';checks['protocol_sha256']=sha(ROOT/'protocol.json')
    dump(ROOT/'results/preflight.json',checks);print('PREFLIGHT',checks,flush=True)

if __name__=='__main__':main()

"""Numerical and contract checks, plus a dev-only discarded update per arm."""
import sklearn
from dataclasses import fields, replace
from collections import Counter
import numpy as np
from threadpoolctl import threadpool_limits
import torch
from data import episode, lock, specs, write
from model import ARMS, Input, Head, diffuse, moments, preproc, prepare, quotas, singleton_masses, transition
from run import balanced_ce, scores, tensors


def checks():
    edge=np.array([[0,1,1,2,4,4],[1,2,3,4,5,5]])
    n=8
    p=transition(n,edge,np.float64)
    P=p.toarray(); np.testing.assert_allclose(P.sum(1),1); assert P[7,7]==1 and P[1,0]!=P[0,1]
    D=sum(.15*.85**i*np.linalg.matrix_power(P,i) for i in range(8))+.85**8*np.linalg.matrix_power(P,8)
    g=np.random.default_rng(12345)
    seed=g.normal(size=(n,5));np.testing.assert_allclose(diffuse(p, seed),D@seed,atol=1e-13)
    h=g.normal(size=(n,3,2)).astype(np.float32); q=np.array([2,6,7]); groups=[np.array([0,1]),np.array([4]),np.array([5])]
    mass,local,mu,cnt,maxwidth,_=moments(p.astype(np.float32),h,q,groups)
    dense_mass=np.stack([D[q][:,a].sum(1) for a in groups],axis=1)
    dense_local=mu[None].repeat(len(q),0).astype(np.float64)
    for j,a in enumerate(groups):
        s=D[q][:,a]; m=s.sum(1); u=s@h[a].reshape(len(a),-1)
        dense_local[m>0,j]=(u[m>0]/m[m>0,None]).reshape(-1,3,2)
    np.testing.assert_allclose(mass,dense_mass,atol=1e-7)
    np.testing.assert_allclose(local,dense_local,atol=5e-7)
    ids=np.concatenate(groups); sm=singleton_masses(p.astype(np.float32),q,ids)
    np.testing.assert_allclose(sm.sum(1),mass.sum(1),atol=2e-7)
    assert np.array_equal(cnt,[2,1,1]) and maxwidth==3+6
    assert np.array_equal(local[1],mu) and np.array_equal(local[2],mu)
    sizes=(1,4,5,13)
    classes=np.split(np.arange(sum(sizes)),np.cumsum(sizes)[:-1]); tail=quotas(classes,'tail'); prop=quotas(classes,'proportional')
    assert np.array_equal(tail,[1,4,2,2]) and prop.sum()==tail.sum() and (prop<=sizes).all() and (prop>=1).all()
    assert set(f.name for f in fields(Input))=={'x','edge','context','yc','query'}
    assert not set(s[0] for s in specs('train')) & set(s[0] for s in specs('final'))
    inp,y,m=episode(*specs('dev')[0]); k=len(np.unique(inp.yc)); torch.manual_seed(0); net=Head()
    qp=g.permutation(len(y)); cp=g.permutation(len(inp.yc)); labelperm=g.permutation(k); check_history={}
    for arm in ARMS:
        original,tm=prepare(inp,arm)
        z=scores(net,original,k)
        shifted=replace(inp,context=inp.context[cp],yc=labelperm[inp.yc[cp]],query=inp.query[qp])
        qpack,shiftm=prepare(shifted,arm)
        zp=scores(net,qpack,k,chunk=5)
        err=float(np.abs(z[qp]-zp[:,labelperm]).max()); assert err<5e-7,(arm,err)
        assert np.array_equal(tm['counts'],np.array(shiftm['counts'])[labelperm])
        # Balanced CE and gradients must equal the class-wise nonempty average.
        t=tensors(original); target=torch.tensor(y)
        logits=net(*t,k)
        manual=torch.stack([torch.nn.functional.cross_entropy(logits[target==c],target[target==c]) for c in range(k)]).mean()
        loss=balanced_ce(logits,target); assert abs(float(loss-manual))<1e-6
        gp=torch.autograd.grad(loss,net.parameters(),retain_graph=True)
        gm=torch.autograd.grad(manual,net.parameters())
        assert all(torch.allclose(a,b,atol=3e-7,rtol=3e-6) for a,b in zip(gp,gm))
        assert all(torch.isfinite(a).all() for a in gp) and any(float(a.norm())>0 for a in gp)
        # Changing truth cannot affect preprocessed arrays or predictions.
        poisoned=labelperm[y]; assert np.array_equal(z,scores(net,prepare(inp,arm)[0],k))
        assert len(poisoned)==len(y)
        # A real, discarded optimizer update uses only dev labels and a throw-away copy.
        check_net=Head(); check_net.load_state_dict(net.state_dict()); opt=torch.optim.AdamW(check_net.parameters(),lr=.001)
        balanced_ce(check_net(*t,k),target).backward(); opt.step()
        assert not np.array_equal(scores(net,original,k),scores(check_net,original,k))
        check_history[arm]={'permutation_chunk_max_prob_error':err,'finite_head_grads':True,'discarded_updates':1}
    tiny=Input(h[:,0],edge,np.array([0,4]),np.array([0,1]),q)
    full=prepare(tiny,'full')[0]
    tail=prepare(tiny,'tail')[0]
    assert np.array_equal(full[0],tail[0])
    assert np.isfinite(scores(net,tail,2)).all()
    for split, expected in [('train',320),('dev',12),('final',96)]:assert len(specs(split))==expected
    return dict(status='PASS',dense_oracle=True,singleton_parity=True,zero_reach=True,quota=True,
                paired_indices=True,checks=check_history)


if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result=checks()
        write('results/checks.json',result)
        print(result,flush=True)
        print('design-lock',lock(create=True),flush=True)

"""Consistent finite latent family. No sampled query labels enter mask scoring."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.special import expit, logsumexp, xlogy
from scipy.sparse.csgraph import shortest_path

ROOT=Path(__file__).resolve().parent
def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))

def graph(seed):
    c=cfg();rng=np.random.default_rng(seed);n=c['nodes'];a=np.zeros((n,n),bool)
    for i in range(n):a[i,(i+1)%n]=a[(i+1)%n,i]=True
    for _ in range(32):
        i=int(rng.integers(n//2))*2;j=int(rng.integers(n//2))*2+1;a[i,j]=a[j,i]=True
    perm=rng.permutation(n);a=a[perm][:,perm];b=np.where(perm%2, -1.,1.)
    d=a.sum(1)+1;s=(a+np.eye(n))/np.sqrt(d[:,None]*d[None,:])
    _,u=np.linalg.eigh(s);basis=u[:,-4:-1];basis/=np.sqrt(np.mean(basis*basis,axis=0))
    states=[];mus=[];probs=[]
    for z in itertools.product([-1.,1.],repeat=3):
        v=basis@z/np.sqrt(3)
        for g,t,sigma in itertools.product([0,1],[0,1],[-1,1]):
            f=v*b**g;mus.append(np.stack([f,.6*f],-1));probs.append(expit(2*sigma*v*b**t));states.append([*z,g,t,sigma])
    src,dst=np.nonzero(a)
    return dict(a=a,b=b,basis=basis,mu=np.array(mus),py=np.array(probs),states=np.array(states),
                src=src.astype(np.int64),dst=dst.astype(np.int64),distance=shortest_path(a,directed=False,unweighted=True))

def draw(g,seed,samples=1):
    rng=np.random.default_rng(seed);h=rng.integers(len(g['mu']),size=samples)
    x=g['mu'][h]+.4*rng.standard_normal((samples,cfg()['nodes'],2))
    y=(rng.random((samples,cfg()['nodes']))<g['py'][h]).astype(np.int64)
    return x,y,h

def posterior(g,context,yc,observed=None,x=None):
    yc=np.atleast_2d(yc);p=g['py'][:,context]
    ll=yc@np.log(p).T+(1-yc)@np.log1p(-p).T
    if observed is not None and len(observed):
        xx=np.asarray(x)
        if xx.ndim==2:xx=xx[None]
        # Gaussian constants are common across hypotheses with the same mask.
        delta=xx[:,None,observed,:]-g['mu'][None,:,observed,:]
        ll-=.5*np.square(delta).sum((2,3))/.4**2
    return np.exp(ll-logsumexp(ll,axis=1,keepdims=True))

def predict(g,context,yc,query,observed=None,x=None):
    return posterior(g,context,yc,observed,x)@g['py'][:,query]

def designs(g,seed,k,budget):
    rng=np.random.default_rng(seed);order=rng.permutation(cfg()['nodes']);context=order[:k];query=order[k:k+cfg()['queries']]
    pool=order[k+cfg()['queries']:]
    masks=np.stack([rng.choice(pool,budget,replace=False) for _ in range(cfg()['candidate_masks'])])
    geometry=np.array([.5*np.exp(-g['distance'][np.ix_(context,o)].min(1)).mean()+
                       .5*np.exp(-g['distance'][np.ix_(query,o)].min(1)).mean() for o in masks])
    return context,query,masks,geometry

def scores(g,context,query,masks,seed,samples=None):
    samples=samples or cfg()['score_samples'];x,y,_=draw(g,seed,samples)
    p0=predict(g,context,y[:,context],query);p0=np.clip(p0,1e-12,1-1e-12)
    j=[];v=[];vse=[]
    for o in masks:
        px=predict(g,context,y[:,context],query,o,x);px=np.clip(px,1e-12,1-1e-12)
        kl=xlogy(px,px/p0)+xlogy(1-px,(1-px)/(1-p0))
        gain=(np.maximum(px,1-px)-np.maximum(p0,1-p0)).mean(1)
        j.append(kl.mean());v.append(gain.mean());vse.append(gain.std(ddof=1)/np.sqrt(samples))
    return np.array(j),np.array(v),np.array(vse)

def choose(geometry,j,v,seed):
    rng=np.random.default_rng(seed);random=int(rng.integers(len(j)));uniform=rng.random()<.5;rank=int(rng.integers(3))
    out={'random':random,'random_cost':random}
    for name,s in [('geometry',geometry),('entropy',j),('decision',v)]:
        out[name]=random if uniform else int(np.argsort(s,kind='stable')[-3:][rank])
    return out

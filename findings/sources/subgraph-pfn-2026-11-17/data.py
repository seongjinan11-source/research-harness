"""Synthetic prior and immutable Planetoid receipts for SPFN01."""
import sklearn
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from model import Input, transition

ROOT = Path(__file__).absolute().parent
WORK = ROOT.parent
SEEDS = (11, 12, 13)
UPDATES = 320
ORIGINALS = 'observed_cell_cycle/bank/real/manifest.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    path = ROOT / path
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as file:
        json.dump(record,file,indent=2,ensure_ascii=False,allow_nan=False)
        file.write('\n')


def observed(rng, y, k, sizes, c_pool, q_pool, qcount, c_ver=None):
    """Only the episode builder sees y. Outputs C,y_C,Q separately from y_Q."""
    c, q = [], []
    for j, nc in enumerate(sizes):
        a = rng.permutation(np.flatnonzero((y==j)&c_pool))
        if c_ver is None:
            assert len(a) >= nc + qcount
            b = a[nc:]
        else:
            assert not np.any(q_pool & c_pool)
            b = rng.permutation(np.flatnonzero((y==j)&q_pool))
            assert len(a)>=nc and len(b)>=qcount
        c.extend(a[:nc]); q.extend(b[:qcount])
    c = rng.permutation(np.array(c,np.int64));q = rng.permutation(np.array(q,np.int64))
    assert len(np.intersect1d(c,q)) == 0 and len(np.unique(y[q])) == k
    assert np.array_equal(np.bincount(y[c],minlength=k),sizes)
    return c,y[c].copy(),q,y[q].copy()


def episode(seed, shape, relation, feature):
    rng=np.random.default_rng(seed)
    k=int(rng.integers(3,6))
    n=int(rng.choice([160,224,288] if not shape else [512,640])); f=int(rng.choice([8,12,16]) if not shape else 24)
    headmax=24 if not shape else 40
    floor=headmax+8
    counts=floor+rng.multinomial(n-k*floor,rng.dirichlet(np.repeat(.9,k)))
    y=rng.permutation(np.repeat(np.arange(k),counts)).astype(np.int64)
    a,b=np.triu_indices(n,1)
    d=np.clip(np.exp(rng.normal(0,.60,n)),.4,3.)
    interaction=np.where(y[a]==y[b], 4.5 if relation else .35, 1.)
    prob=interaction*d[a]*d[b]
    prob=np.minimum(prob*(rng.uniform(5,11)*n/2)/prob.sum(),.5)
    keep=rng.random(len(a))<prob
    edge=np.stack([a[keep],b[keep]])
    p=transition(n,edge)
    # Graph-conditioned nonlinear feature SCM: independent exogenous noise and per-class latent features.
    z=rng.normal(size=(n,f)).astype(np.float32)
    c=rng.normal(size=(k,f)).astype(np.float32)
    w=rng.normal(0,1/np.sqrt(f),size=(f,f)).astype(np.float32)
    feat=(1.40 if feature else .25)*(c[y]+.35*(p@c[y])) + np.tanh((z+.5*(p@z))@w)+rng.normal(0,.55,(n,f))
    tail=rng.permutation(k)[:(k+1)//2]
    sizes=rng.integers(12 if not shape else 24,headmax+1,size=k)
    sizes[tail]=rng.integers(1,5,size=len(tail))
    pool=np.ones(n,bool)
    cids,yc,q,truth=observed(rng,y,k,sizes,pool,pool,8)
    meta=dict(id=f'syn-{seed}', seed=seed,stratum=[int(shape),int(relation),int(feature)],n=n, f=f,
              population=counts.tolist(),counts=sizes.tolist(),edge_assort=float((y[edge[0]]==y[edge[1]]).mean()))
    return Input(feat.astype(np.float32),edge,cids,yc,q), truth,meta


def specs(split):
    base={'train':17981000,'dev':17982000,'final':17983000}[split]
    if split=='train':
        return [(base+i,0,(i//2)%2,i%2) for i in range(UPDATES)]
    if split=='dev':
        return [(base+i,i//8,(i//2)%2,i%2) for i in range(12)]
    return [(base+shape*48+relation*24+feat*12+i,shape,relation,feat)
            for shape in (0,1) for relation in (0,1) for feat in (0,1) for i in range(12)]


def real_specs():
    desc=json.loads((WORK / ORIGINALS).read_text(encoding='utf8'))['raw_sources']
    for name, row in desc.items():
        path=Path(row['path'])
        if sha(path)!=row['sha256']:
            raise ValueError('raw real dataset hash mismatch: '+name)
    return desc


def real():
    import torch
    desc=real_specs()
    for gi,name in enumerate(('Cora','CiteSeer','PubMed')):
        obj=torch.load(desc[name]['path'],map_location='cpu',weights_only=False)
        v=obj[0] if isinstance(obj,(tuple,list)) else obj
        def get(k): return np.asarray(v[k] if isinstance(v,dict) else getattr(v,k))
        native=get('x').astype(np.float32); _,y=np.unique(get('y'),return_inverse=True); y=y.astype(np.int64)
        e=get('edge_index').astype(np.int64)
        tr,te=get('train_mask'),get('test_mask')
        tr=tr if tr.ndim==1 else tr[:,0]; te=te if te.ndim==1 else te[:,0]
        k=int(y.max()+1)
        rng=np.random.default_rng(17985000+gi)
        rp=rng.choice(np.array([-1.,1.],np.float32),(native.shape[1],24))/np.float32(np.sqrt(24))
        t0=perf_counter(); x=native@rp; projection_seconds=perf_counter()-t0
        for draw in range(5):
            seed=17984000+gi*100+draw; rng=np.random.default_rng(seed)
            sizes=np.full(k,16,np.int64); sizes[rng.permutation(k)[:(k+1)//2]]=2
            c,yc,q,truth=observed(rng,y,k,sizes,tr,te,32,c_ver=True)
            meta=dict(id=f'{name}-{draw}', graph=name, draw=draw,seed=seed,n=len(y),f=24,
                      native_width=native.shape[1],counts=sizes.tolist(),source_sha256=desc[name]['sha256'],
                      projection_seconds=projection_seconds)
            yield Input(x,e,c,yc,q),truth,meta


def array_hash(*a):
    h=hashlib.sha256()
    for v in a:
        v=np.ascontiguousarray(v);h.update(str((v.shape,v.dtype.str)).encode());h.update(v.tobytes())
    return h.hexdigest()


def lock_body():
    files=['PLAN_v1.md','PLAN_v2.md','RESPONSE_v1.md','model.py','data.py','run.py','checks.py']
    return dict(sources={p:sha(ROOT/p) for p in files}, raw_sources=real_specs(),
                split_specs={s:[list(v) for v in specs(s)] for s in ('train','dev','final')}, seeds=list(SEEDS), updates=UPDATES)


def lock(create=False):
    p=ROOT/'results/lock.json'; expected=lock_body()
    if create:
        assert json.loads((ROOT/'results/checks.json').read_text())['status']=='PASS'
        write('results/lock.json',expected)
    assert json.loads(p.read_text())==expected, 'locked source or data changed'
    return sha(p)

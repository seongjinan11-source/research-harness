import sys
sys.dont_write_bytecode=True
import sklearn
import json, argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import torch
from core import ROOT,WORK,C,cfg,dump,sha,geometry,adjacency,impute

def synthetic(seed,index):
    r=np.random.default_rng(seed);c=cfg();n=c['nodes'];f=c['features'];k=int(r.choice(c['classes']))
    y=r.permutation(np.arange(n)%k).astype('int64');mode=(index//2)%2;flaw=index%2
    pref=6. if mode==0 else .15
    base=8/(n*(pref/k+1-1/k));pr=base*np.where(y[:,None]==y[None,:],pref,1.)
    aa=np.triu(r.random((n,n))<pr,1);aa=aa|aa.T;src,dst=np.nonzero(aa)
    a=adjacency((src,dst),n);p=sp.diags(1/np.asarray(a.sum(1)).ravel())@a
    z=r.normal(size=(k,f))[y]+.35*r.normal(size=(n,f));lam=.65 if flaw==0 else -.65
    x=np.linalg.solve(np.eye(n)-lam*p.toarray(),z).astype('float32')
    x*=r.choice([-1.,1.],f)*r.uniform(.5,1.5,f)
    edges=float((y[src]==y[dst]).mean()) if len(src) else 0.
    energy=float(((x[src]-x[dst])**2).mean()/(np.var(x)+1e-8)) if len(src) else 0.
    return dict(x=x,y=y,src=src.astype('int64'),dst=dst.astype('int64'),classes=k,
                label_mode='homo' if mode==0 else 'hetero',feature_mode='smooth' if flaw==0 else 'rough',label_agreement=edges,feature_energy=energy)

def mask(g,case,seed):
    r=np.random.default_rng(seed);n=len(g['y']);f=g['x'].shape[1]
    if case=='full':rows=np.ones(n,bool)
    elif case.startswith('uniform'):
        rate={'uniform50':.5,'uniform90':.9,'uniform995':.995}[case];rows=r.random(n)>=rate
    else:
        rate=.995 if case.endswith('995') else .9
        a=adjacency((g['src'],g['dst']),n)
        d=shortest_path(a,directed=False,unweighted=True,indices=int(r.integers(n)))
        order=np.lexsort((r.random(n),d));rows=np.zeros(n,bool);rows[order[:round(n*(1-rate))]]=True
    return np.broadcast_to(rows[:,None],(n,f)).copy()

def episode(g,case,seed,context,query,*,native=None):
    obs=mask(g,case,seed);xo=C.observed_scale(g['x'],obs)
    ex=geometry(g['src'],g['dst'],obs,xo,context,g['y'][context],g['classes'])
    inp=C.Inputs(xo,obs,ex['warm'],g['src'],g['dst'],context,g['y'][context],query,g['classes'],seed)
    result={k:np.asarray(getattr(inp,k)) for k in C.Inputs.__annotations__}
    result.update(ex);result['labels_query']=g['y'][query]
    if native is not None:result['native_pcfi']=native(obs,xo)
    return result

def contexts(g,seed,k,allowed=None):
    r=np.random.default_rng(seed);c=[]
    for label in range(g['classes']):
        ids=np.flatnonzero(g['y']==label)
        if allowed is not None:ids=ids[allowed[ids]]
        c.extend(r.permutation(ids)[:k].tolist())
    return np.array(r.permutation(c),dtype='int64')

def read_real(name):
    if name in ['Cora','CiteSeer']:
        p=WORK/f'ctxedit/data/planetoid_pyg/{name}/processed/data.pt'
        raw=torch.load(p,map_location='cpu',weights_only=False)[0]
        x=raw.x.numpy().astype('float32');y=raw.y.numpy().astype('int64');src,dst=raw.edge_index.numpy()
        train=[raw.train_mask.numpy()]*3;test=[raw.test_mask.numpy()]*3;paths=[p]
    else:
        folder=Path('C:/Users/seongjin/Desktop/code/data')/name/'raw'
        rows=[line.strip().split('\t') for line in (folder/'out1_node_feature_label.txt').read_text().splitlines()[1:] if line.strip()]
        rows=sorted(rows,key=lambda row:int(row[0]));x=np.array([[float(v) for v in row[1].split(',')] for row in rows],dtype='float32');y=np.array([int(row[2]) for row in rows])
        edges=np.array([[int(v) for v in line.split()] for line in (folder/'out1_graph_edges.txt').read_text().splitlines()[1:] if line.strip()]);src,dst=edges.T
        splits=[np.load(folder/f'{name}_split_0.6_0.2_{j}.npz') for j in range(3)]
        train=[s['train_mask'] for s in splits];test=[s['test_mask'] for s in splits]
        paths=[folder/'out1_node_feature_label.txt',folder/'out1_graph_edges.txt']+[folder/f'{name}_split_0.6_0.2_{j}.npz' for j in range(3)]
    a=adjacency((src,dst),len(y),False);src,dst=np.array(a.nonzero());_,y=np.unique(y,return_inverse=True)
    f=x.shape[1];r=np.random.default_rng(19300+len(name));rp=r.choice([-1.,1.],size=(f,cfg()['features'])).astype('float32')/np.sqrt(f)
    # RP is fixed independently of all data values. Inference only uses known rows.
    projected=x@rp
    return dict(x=projected,y=y,src=src.astype('int64'),dst=dst.astype('int64'),classes=int(y.max()+1),native=x,rp=rp,train=train,test=test,
                paths={str(p):sha(p) for p in paths},label_agreement=float((y[src]==y[dst]).mean()))

def prepare(split):
    bank=ROOT/'bank';bank.mkdir(exist_ok=True)
    manifest=bank/'manifest.json'
    state=json.loads(manifest.read_text()) if manifest.exists() else {'episodes':[],'raw':[],'real_sources':{}}
    assert not any(r['split']==split for r in state['episodes']),split
    cases=cfg()['masks']+(['full','uniform50'] if split=='train' else ['full'])
    if split in ['dev','train','test']:
        count=cfg()[split+'_graphs'];start={'dev':2036110300,'train':2036111300,'test':2036121300}[split]
        for i in range(count):
            g=synthetic(start+i,i);raw=bank/f'{split}_raw_{i}.npz';np.savez_compressed(raw,**{k:v for k,v in g.items() if k not in ['label_mode','feature_mode']})
            state['raw'].append({'split':split,'graph':i,'path':raw.name,'sha256':sha(raw)})
            for ci,case in enumerate(cases):
                seed=start+i*100+ci+100000;k=cfg()['context_per_class'][(i+ci)%2];con=contexts(g,seed+100,k)
                query=np.setdiff1d(np.arange(len(g['y'])),con)
                z=episode(g,case,seed,con,query);dest=bank/f'{split}_{i}_{case}.npz';np.savez_compressed(dest,**z)
                state['episodes'].append(dict(split=split,graph=i,case=case,path=dest.name,sha256=sha(dest),label_mode=g['label_mode'],feature_mode=g['feature_mode'],
                    classes=g['classes'],observed_nodes=int(z['observed'][:,0].sum()),context_feature_overlap=int(z['observed'][con,0].sum()),label_agreement=g['label_agreement'],feature_energy=g['feature_energy']))
    else:
        assert split=='real'
        for name in cfg()['real_datasets']:
            g=read_real(name);state['real_sources'][name]=g['paths']
            for rep in range(3):
                seed=2036131300+10000*cfg()['real_datasets'].index(name)+rep*100
                con=contexts(g,seed+999,cfg()['real_context_per_class'],g['train'][rep]);query=np.flatnonzero(g['test'][rep]);query=np.setdiff1d(query,con)
                for ci,case in enumerate(cfg()['masks']+['full']):
                    def native(obs,xo):
                        an=adjacency((g['src'],g['dst']),len(obs))
                        # Scale projection on actual observed rows, matching the common interface.
                        projected_obs=np.where(obs,g['x'],0.)
                        rms=np.sqrt(np.sum(projected_obs**2,0)/max(int(obs[:,0].sum()),1));rms=np.where(rms>=1e-8,rms,1.)
                        # Native original PCFI runs before projection; no hidden values are used.
                        raw_nan=np.where(obs[:,0,None],g['native'],np.nan)
                        fill=impute(an,raw_nan,'pcfi','structural',iterations=100)[0]@g['rp']
                        return np.where(obs,xo,np.clip(fill/rms,-10,10)).astype('float32')
                    z=episode(g,case,seed+ci,con,query,native=native);dest=bank/f'real_{name}_{rep}_{case}.npz';np.savez_compressed(dest,**z)
                    state['episodes'].append(dict(split='real',graph=name,repetition=rep,case=case,path=dest.name,sha256=sha(dest),
                        label_mode='homo' if name in ['Cora','CiteSeer'] else 'hetero',feature_mode='real',classes=g['classes'],
                        observed_nodes=int(z['observed'][:,0].sum()),context_feature_overlap=int(z['observed'][con,0].sum()),label_agreement=g['label_agreement'],native_features=g['native'].shape[1]))
                print('PREPARED REAL',name,rep,flush=True)
    dump(manifest,state);print('PREPARED',split,len([r for r in state['episodes'] if r['split']==split]),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('split',choices=['dev','train','test','real']);prepare(ap.parse_args().split)

"""Fresh small native SBM tasks with Gaussian attributes; hidden truth stays outside inputs."""
import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import json, math, hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
ROLES=np.asarray([[0,0],[1,0],[0,1],[1,1]],dtype=np.float64)
PRIORS={'independent':np.ones(4)/4,'exclusive':np.asarray([0,.5,.5,0]),'coupled':np.asarray([.5,0,0,.5])}
PARAMETERS=np.asarray([[.25,.72],[.18,.45],[.18,.72],[.25,.45],[.32,.45],[.32,.72]])
MASS=np.asarray([.5,.1,.1,.1,.1,.1])
NAMES=list(PRIORS)
SETTINGS=[('full',0.),('none',1.),('uniform_nodes',.5),('uniform_nodes',.9),('uniform_nodes',.995),('clustered_nodes',.5),('clustered_nodes',.9),('uniform_cells',.9)]


def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def tree_for(adj,seed):
    # One variable per original node. Independent tie-breaking travels with nodes.
    n=adj.shape[-1];union=adj.sum(0);rng=np.random.default_rng(seed+0x4d5354)
    priority=rng.random(n);pair=priority[:,None]+priority[None,:]+priority[:,None]*priority[None,:]
    affinity=100*union+union@union+.001*pair
    root=int(priority.argmin());chosen=[root];parent=np.full(n,-1,dtype=np.int64)
    while len(chosen)<n:
        rest=np.setdiff1d(np.arange(n),chosen);scores=affinity[np.ix_(chosen,rest)]
        p,k=np.unravel_index(scores.argmax(),scores.shape);node=int(rest[k]);parent[node]=chosen[p];chosen.append(node)
    return parent,np.asarray(chosen,dtype=np.int64)


def master(seed,prior,case='matched'):
    c=cfg();rng=np.random.default_rng(seed);nt=c['target_nodes'];nh=c['helper_nodes'];n=nt+nh
    latent=rng.integers(0,2,n);role=int(rng.choice(4,p=PRIORS[prior]));spins=2*latent-1
    if case=='matched':xi=PARAMETERS[rng.choice(len(MASS),p=MASS)]
    elif case=='density_low':xi=np.asarray([.12,.72])
    elif case=='density_high':xi=np.asarray([.38,.72])
    elif case=='signal_high':xi=np.asarray([.25,.95])
    else:raise ValueError(case)
    prob=xi[0]*(1+xi[1]*ROLES[role,:,None,None]*spins[:nt][None,:,None]*spins[nt:][None,None,:])
    incidence=(rng.random((2,nt,nh))<prob).astype(np.float32)
    adj=np.zeros((2,n,n),np.float32);adj[:,:nt,nt:]=incidence;adj[:,nt:,:nt]=incidence.transpose(0,2,1)
    valid=np.zeros_like(adj);valid[:,:nt,nt:]=1;valid[:,nt:,:nt]=1
    orient=rng.choice([-1,1],size=c['features'])
    full=c['feature_signal']*spins[:,None]*orient[None]+c['feature_noise']*rng.standard_normal((n,c['features']))
    parent,order=tree_for(adj,seed)
    return dict(seed=seed,prior=prior,case=case,latent=latent,role=role,parameters=xi,full=full.astype(np.float32),incidence=incidence,adj=adj,pair_valid=valid,
        type=np.r_[np.zeros(nt),np.ones(nh)].astype(np.float32),parent=parent,order=order,
        context_order=np.random.default_rng(seed+0x435458).permutation(nt))


def observation(m,k,setting):
    c=cfg();mech,rate=SETTINGS[setting];n=len(m['latent']);f=c['features']
    rng=np.random.default_rng(m['seed']+setting*100003+0x4d41534b)
    if mech=='full':obs=np.ones((n,f),bool)
    elif mech=='none':obs=np.zeros((n,f),bool)
    elif mech=='uniform_nodes':obs=np.repeat((rng.random(n)>=rate)[:,None],f,1)
    elif mech=='uniform_cells':obs=rng.random((n,f))>=rate
    else:
        start=int(rng.integers(n));a=m['adj'].sum(0);distance=np.full(n,1e6);distance[start]=0;queue=[start]
        for v in queue:
            for j in np.flatnonzero(a[v]):
                if distance[j]>distance[v]+1:distance[j]=distance[v]+1;queue.append(int(j))
        keep=np.lexsort((rng.random(n),distance))[:round(n*(1-rate))]
        obs=np.zeros((n,f),bool);obs[keep]=True
    context=np.full(n,-1,np.int64);ids=m['context_order'][:k];context[ids]=m['latent'][ids]
    query=np.setdiff1d(np.arange(c['target_nodes']),ids)
    x=np.where(obs,m['full'],0).astype(np.float32)
    inp={key:m[key] for key in ['incidence','adj','pair_valid','type','parent','order']}
    inp.update(x=x,observed=obs,context=context,prior=np.eye(3,dtype=np.float32)[NAMES.index(m['prior'])])
    return inp,query,m['latent'][query]


INPUT_KEYS=['incidence','adj','pair_valid','type','parent','order','x','observed','context','prior']
def read(path):
    with np.load(ROOT/'bank'/path) as z:
        inp={k:z[k] for k in INPUT_KEYS};return inp,z['query'],z['labels_query']
def tensors(inputs,device='cuda',dtype=torch.float32):
    return {k:torch.as_tensor(np.stack([x[k] for x in inputs]),device=device,dtype=(torch.long if k in ['context','parent','order'] else torch.bool if k=='observed' else dtype)) for k in INPUT_KEYS}


class GridReference:
    """Declared six-support Bayesian model; exact latent sum, optional native loopy BP."""
    def __init__(self,device='cuda',nt=8,nh=4,features=2):
        self.device=device;self.nt=nt;self.nh=nh;self.n=nt+nh;self.f=features
        ids=torch.arange(2**self.n,device=device);self.labels=((ids[:,None]>>torch.arange(self.n,device=device))&1).double();self.spin=2*self.labels-1
        self.products=(self.spin[:,:nt,None]*self.spin[:,None,nt:]).flatten(1)
        self.roles=torch.tensor(ROLES,device=device,dtype=torch.float64)
        self.orient=2*((torch.arange(2**features,device=device)[:,None]>>torch.arange(features,device=device))&1).double()-1
    def infer(self,items,true_parameters=None,with_bp=True):
        b=tensors(items,self.device,torch.float64);e=b['incidence'].flatten(2);ne=1-e;batch=len(e)
        pri=torch.as_tensor(np.stack([PRIORS[NAMES[int(x['prior'].argmax())]] for x in items]),device=self.device,dtype=torch.float64)
        if true_parameters is None:
            xi=torch.tensor(PARAMETERS,device=self.device)[None].expand(batch,-1,-1);mass=torch.tensor(MASS,device=self.device)[None].expand(batch,-1)
        else:
            xi=torch.as_tensor(true_parameters,device=self.device,dtype=torch.float64)[:,None];mass=torch.ones((batch,1),device=self.device,dtype=torch.float64)
        d,s=xi.unbind(-1);plus=d*(1+s);minus=d*(1-s)
        ce=.5*(plus.log()+minus.log())-d.log();cn=.5*(torch.log1p(-plus)+torch.log1p(-minus))-torch.log1p(-d)
        je=.5*(plus.log()-minus.log());jn=.5*(torch.log1p(-plus)-torch.log1p(-minus))
        ea=e.sum(-1)@self.roles.T;na=ne.sum(-1)@self.roles.T
        es=torch.einsum('bre,se,hr->bhs',e,self.products,self.roles)
        ns=torch.einsum('bre,se,hr->bhs',ne,self.products,self.roles)
        constant=e.sum((1,2))[:,None]*d.log()+ne.sum((1,2))[:,None]*torch.log1p(-d)
        logg=constant[:,:,None,None]+ea[:,None,:,None]*ce[:,:,None,None]+na[:,None,:,None]*cn[:,:,None,None]+es[:,None]*je[:,:,None,None]+ns[:,None]*jn[:,:,None,None]
        mu=cfg()['feature_signal'];sd=cfg()['feature_noise'];obs=b['observed'].double();x=b['x']
        field=mu/(sd*sd)*torch.einsum('bnf,af->ban',obs*x,self.orient)
        fx=torch.einsum('ban,sn->bas',field,self.spin)
        fx=fx+(-.5*((x*x+mu*mu)/(sd*sd)+math.log(2*math.pi*sd*sd))*obs).sum((1,2))[:,None,None]
        c=b['context'];allowed=((c[:,None]<0)|(c[:,None]==self.labels[None])).all(-1)
        score=logg[:,:,:,None,:]+fx[:,None,None]+mass.log()[:,:,None,None,None]+pri.log()[:,None,:,None,None]-math.log(len(self.orient))
        score=score.masked_fill(~allowed[:,None,None,None],-torch.inf)
        z=torch.logsumexp(score,-1);weight=z.flatten(1).softmax(-1)
        assignment=torch.softmax(score.flatten(1),-1).reshape_as(score).sum((1,2,3))
        p1=assignment@self.labels;exact=torch.stack((1-p1,p1),-1)
        out={'grid6_exact':exact.cpu().numpy()}
        if with_bp:
            # Identical 64-step damped native Ising BP, extended only by observed-feature fields.
            jrel=e[:,None]*je[:,:,None,None]+ne[:,None]*jn[:,:,None,None]
            couplings=torch.einsum('bgre,hr->bghe',jrel,self.roles).reshape(batch,len(xi[0]),4,self.nt,self.nh)
            j=logg.new_zeros(batch,len(xi[0]),4,self.n,self.n)
            j[:,:,:,:self.nt,self.nt:]=couplings;j[:,:,:,self.nt:,:self.nt]=couplings.transpose(-1,-2)
            a=j[:,:,:,None].expand(-1,-1,-1,len(self.orient),-1,-1).flatten(1,3)
            u=torch.zeros_like(a);t=a.tanh();unary=field[:,None,None].expand(-1,len(xi[0]),4,-1,-1).flatten(1,3)
            known=(c>=0)[:,None,:,None];spin=(2*c-1)[:,None,:,None]
            for _ in range(64):
                cavity=(unary[...,None]+u.sum(-2)[...,:,None]-u.transpose(-1,-2)).tanh()
                cavity=torch.where(known,spin,cavity)
                u=.5*u+.5*torch.atanh((t*cavity).clamp(-1+1e-15,1-1e-15))
            means=(unary+u.sum(-2)).tanh();means=torch.where((c>=0)[:,None],(2*c-1)[:,None],means)
            p=(means+1)/2;bp=torch.stack((1-p,p),-1)
            out['grid6_bp']=torch.einsum('bh,bhnc->bnc',weight,bp).cpu().numpy()
        return out


def prepare():
    c=cfg();bank=ROOT/'bank';assert not (bank/'manifest.json').exists()
    for name in ['train','test','latent']:(bank/name).mkdir(parents=True,exist_ok=True)
    rows=[];latent=[]
    def save(m,split,gid,k,setting):
        inp,q,y=observation(m,k,setting);name=f'{split}/g{gid}_k{k}_e{setting}.npz';np.savez_compressed(bank/name,**inp,query=q,labels_query=y)
        rows.append(dict(path=name,split=split,graph=gid,prior=m['prior'],case=m['case'],context_total=k,setting=setting,
            mechanism=SETTINGS[setting][0],rate=SETTINGS[setting][1],observed_cells=int(inp['observed'].sum()),sha256=sha(bank/name)))
        return inp
    for gid in range(c['train_episodes']):
        pr=NAMES[gid%3];m=master(c['train_seed_start']+gid,pr);save(m,'train',gid,c['contexts'][(gid//3)%3],(gid//9)%len(SETTINGS))
    masters={}
    for pid,pr in enumerate(NAMES):
        for local in range(c['test_graphs_per_prior']):
            gid=pid*c['test_graphs_per_prior']+local;case=c['cases'][local%len(c['cases'])]
            m=master(c['test_seed_start']+gid,pr,case);masters[gid]=m
            np.savez_compressed(bank/'latent'/f'g{gid}.npz',latent=m['latent'],parameters=m['parameters'],role=m['role'],full=m['full'])
            for k in c['contexts']:
                for setting in range(len(SETTINGS)):save(m,'test',gid,k,setting)
    dump(bank/'manifest.json',{'episodes':rows,'sources':{p.name:sha(p) for p in ROOT.glob('*.py')},'protocol_sha256':sha(ROOT/'protocol.json')})
    print('BANK',len(rows),flush=True)
    engine=GridReference(device='cuda');test=[r for r in rows if r['split']=='test'];refdir=ROOT/'references';refdir.mkdir(exist_ok=True)
    for start in range(0,len(test),16):
        rr=test[start:start+16];items=[read(r['path'])[0] for r in rr];truth=np.stack([masters[r['graph']]['parameters'] for r in rr])
        with torch.inference_mode():
            ref=engine.infer(items);oracle=engine.infer(items,truth,False)['grid6_exact']
        for j,r in enumerate(rr):np.savez_compressed(refdir/(Path(r['path']).stem+'.npz'),grid6_exact=ref['grid6_exact'][j],grid6_bp=ref['grid6_bp'][j],oracle_true_parameters=oracle[j])
        if start%384==0:print('REFERENCE',start,'/',len(test),flush=True)
    dump(ROOT/'references/completed.json',{'episodes':len(test),'reference':'same known-family Grid6 prior plus exact marginalization of independent feature orientations; BP uses same weights and feature unary fields','oracle':'true density/signal known, role and feature orientations still marginalized; privileged diagnostic only'})
    print('PREPARE_DONE',flush=True)

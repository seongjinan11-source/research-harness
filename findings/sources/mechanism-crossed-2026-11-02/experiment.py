"""Controlled prior-coupling intervention with the unchanged public Mask Plain PFN."""
import sys
sys.dont_write_bytecode=True
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT.parent/'joint_imputation_pfn'
spec=importlib.util.spec_from_file_location('crossed_reference',REFERENCE/'common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
import argparse
import copy
import gc
import hashlib
import json
import math
import time
from collections import Counter
import numpy as np
import scipy.linalg as la
from scipy.special import expit
import torch
import torch.nn.functional as F


def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True);C.write_json(Path(path),obj)
def sources():
    return {p.name:C.sha(p) for p in ROOT.glob('*.py')}|{'protocol.json':C.sha(ROOT/'protocol.json'),
        'reference_common':C.sha(REFERENCE/'common.py'),'reference_core':C.sha(C.WORK/'regret_prior_pilot/core.py')}
def manifest():return json.loads((ROOT/'bank/manifest.json').read_text(encoding='utf8'))


def official_config():
    config=json.loads((REFERENCE/'bank/generator.json').read_text(encoding='utf8'))
    v=config['prior']['values'][0];v['task']['_type_']='binclass'
    return config


def operators(task):
    n=len(task['X_full']);a=C.BASE.adjacency(task['src'],task['dst'],n).toarray().astype(np.float64)
    degree=a.sum(1);p=a/degree[:,None];s=a/np.sqrt(degree[:,None]*degree[None,:])
    ident=np.eye(n)
    normalize=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return s,[ident,normalize(p),ident,normalize(p@p-.5*ident)]


def feature_styles(task,seed):
    rng=np.random.default_rng(seed);n,d=task['X_full'].shape;s,hs=operators(task)
    rho=float(rng.uniform(.5,.9));r=la.solve(np.eye(n)-rho*s,np.eye(n),assume_a='pos',check_finite=False)
    r/=np.sqrt(np.diag(r)[:,None]*np.diag(r)[None,:]);k=.35+.65*r
    z=la.cholesky(k,lower=True,check_finite=False)@rng.standard_normal((n,d))
    styles=[z,(np.exp(np.clip(z,-4,4))-np.exp(.5))/np.sqrt(np.e*(np.e-1)),
        (np.sign(z)+.15*z)/np.sqrt(1+.3*np.sqrt(2/np.pi)+.15**2),
        (z+.35*z**3)/np.sqrt(1+2*.35*3+.35**2*15)]
    return [x.astype(np.float32) for x in styles],k,hs,rho


def rule_parameters(d,seed):
    rng=np.random.default_rng(seed);w=rng.standard_normal(d);w/=np.linalg.norm(w)
    u=rng.standard_normal(d);u/=np.linalg.norm(u)
    return {'w':w,'u':u,'bias':float(rng.normal(0,.2)),'flip':int(rng.choice([-1,1]))}


def label_probability(x,hs,rule,parameters):
    # Deliberately no feature-style ID, feature distribution statistics, or sample quantiles.
    x=np.asarray(x,np.float64);w=parameters['w'];u=parameters['u']
    score=(x@w)*(x@u) if rule==2 else hs[rule]@x@w
    return expit(2*parameters['flip']*(score+parameters['bias']))


def draw_labels(x,hs,rule,parameters,noise):
    return (noise<label_probability(x,hs,rule,parameters)).astype(np.int64)


def oracle_information(x,observed,k,h,parameters,query):
    """Known-rule Gaussian subfamily: only signs retained by the RMS input."""
    n,d=x.shape;hq=h[query]
    prior_var=np.einsum('qi,ij,qj->q',hq,k,hq)*np.square(parameters['w']).sum()
    variance=prior_var.copy();coefficients=np.zeros((len(query),d))
    for f in range(d):
        o=np.flatnonzero(observed[:,f])
        if len(o):
            covariance=hq@k[:,o[0]]
            coefficients[:,f]=parameters['w'][f]*covariance*np.sign(x[o[0],f])
            variance-=parameters['w'][f]**2*covariance**2
    magnitudes=np.abs(np.random.default_rng(2026110203).standard_normal((2048,d)))
    mu=magnitudes@coefficients.T+parameters['bias']
    nodes,weights=np.polynomial.hermite.hermgauss(24)
    integrate=lambda m,v:(expit(2*parameters['flip']*(m[...,None]+np.sqrt(2*np.maximum(v,0))[...,None]*nodes))*weights).sum(-1)/np.sqrt(np.pi)
    p=np.clip(integrate(mu,variance).mean(0),1e-10,1-1e-10)
    prior=np.clip(integrate(np.full(len(query),parameters['bias']),prior_var),1e-10,1-1e-10)
    kl=p*np.log(p/prior)+(1-p)*np.log((1-p)/(1-prior))
    return {'kl_from_no_observed':float(kl.mean()),'mean_probability_change':float(np.abs(p-prior).mean()),
        'observed_cells':int(observed.sum()),'observed_rows':int(observed.any(1).sum())}


def make_episode(task,x,y,classes,split,gid,style,rule,sid,feature_cache,rows):
    c=cfg();mechanism,rate=(c['train_settings'] if split=='train' else c['test_settings'])[sid]
    family='official' if style<0 else 'controlled';key=(family,style,sid)
    n,d=x.shape;mask_seed=c['mask_seed_start']+(0 if split=='train' else 100000)+gid*100+sid
    if key not in feature_cache:
        observed=C.BASE.mask({**task,'X_full':x},mechanism,rate,mask_seed)
        values=C.observed_scale(x,observed)
        poison=x.copy();poison[~observed]=1e12
        assert np.array_equal(values,C.observed_scale(poison,observed))
        pcfi=C.BASE.fill(np.where(observed,values,np.nan),task['src'],task['dst'],'pcfi',mechanism).clip(-10,10)
        if d%2:
            values=np.pad(values,((0,0),(0,1)));pcfi=np.pad(pcfi,((0,0),(0,1)))
            observed=np.pad(observed,((0,0),(0,1)),constant_values=True)
        feature_cache[key]=(values,observed,pcfi)
    values,observed,pcfi=feature_cache[key]
    rng=np.random.default_rng(c[split+'_seed_start']+gid+777)
    anchors=[rng.permutation(np.flatnonzero(y[:task['n_train']]==cl))[:c['context_per_class']] for cl in range(classes)]
    context=np.concatenate(anchors)
    query=np.random.default_rng(c[split+'_seed_start']+gid+999).permutation(np.arange(task['n_train'],n))[:c['queries']]
    assert len(context)>0 and len(query)==c['queries'] and not set(context)&set(query)
    filename=f'{split}/g{gid}_v{style}_r{rule}_e{sid}.npz';path=ROOT/'bank'/filename
    if path.exists():return filename
    np.savez_compressed(path,observed_values=values,observed=observed,pcfi=pcfi,src=task['src'],dst=task['dst'],
        context=context,labels_context=y[context],query=query,labels_query=y[query],classes=classes,forward_seed=mask_seed+10000000)
    rows[filename]={'path':filename,'split':split,'graph':gid,'family':family,'style':style,'rule':rule,'setting':sid,
        'mechanism':mechanism,'rate':rate,'features':d,'classes':classes,'nodes':n,'context':len(context),
        'observed_rows':int(observed[:,:d].any(1).sum()),'observed_cells':int(observed[:,:d].sum()),
        'pair':'official' if style<0 else ('diagonal' if style==rule else 'offdiagonal'),'sha256':C.sha(path)}
    return filename


def prepare():
    from mgpfn import prior_adapter as PA
    c=cfg();bank=ROOT/'bank'
    assert not (bank/'manifest.json').exists(),'Preserve existing bank'
    for name in ['train','test','latent']:(bank/name).mkdir(parents=True,exist_ok=True)
    config=official_config();write(bank/'official_generator.json',config)
    rows={};arm_paths={a:[] for a in c['arms']};test_paths=[];information=[];exposure=[];start=time.perf_counter()
    for split in ['train','test']:
        for gid in range(c[split+'_graphs']):
            requested=c[split+'_seed_start']+gid
            for attempt in range(20):
                try:
                    task=PA.sample_official_task(config,requested+attempt*100000000,allowed_task_types=('binclass',));break
                except AssertionError:pass
            else:raise RuntimeError(('Invalid official draws',split,gid))
            _,task['y']=np.unique(task['y'],return_inverse=True);styles,k,hs,rho=feature_styles(task,requested+20000000)
            n,d=task['X_full'].shape;params=[rule_parameters(d,requested+30000000+r*100000) for r in range(4)]
            noises=[np.random.default_rng(requested+40000000+r*100000).random(n) for r in range(4)]
            PA.save_task(bank/'latent'/f'{split}_official{gid}.npz',task)
            np.savez_compressed(bank/'latent'/f'{split}_controlled{gid}.npz',features=np.stack(styles),covariance=k,
                w=np.stack([p['w'] for p in params]),u=np.stack([p['u'] for p in params]),
                bias=[p['bias'] for p in params],flip=[p['flip'] for p in params],noise=np.stack(noises),rho=rho)
            cache={}
            def episode(style,rule,sid):
                x=task['X_full'] if style<0 else styles[style]
                y=task['y'] if style<0 else draw_labels(x,hs,rule,params[rule],noises[rule])
                return make_episode(task,x,y,2,split,gid,style,rule,sid,cache,rows)
            if split=='train':
                shift=(gid//4)%4
                for style in range(4):
                    sid=(style+gid%4)%4;rule=(style+shift)%4
                    arm_paths['official_mask'].append(episode(-1,-1,sid))
                    coupled=episode(style,style,sid)
                    arm_paths['coupled_mask'].append(coupled);arm_paths['coupled_pcfi'].append(coupled)
                    arm_paths['crossed_mask'].append(episode(style,rule,sid))
                    exposure.append({'graph':gid,'style':style,'coupled_rule':style,'crossed_rule':rule,'mask':sid})
            else:
                style=gid%4;off=(style+1+(gid//4)%3)%4
                for sid in range(len(c['test_settings'])):
                    test_paths.append(episode(-1,-1,sid))
                    for rule in [style,off]:test_paths.append(episode(style,rule,sid))
                    if style==0:
                        for rule in [0,1]:
                            observed=C.BASE.mask({**task,'X_full':styles[0]},*c['test_settings'][sid],c['mask_seed_start']+100000+gid*100+sid)
                            query=np.load(bank/test_paths[-1])['query']
                            information.append({'graph':gid,'rule':rule,'mechanism':c['test_settings'][sid][0],
                                'rate':c['test_settings'][sid][1],**oracle_information(styles[0],observed,k,hs[rule],params[rule],query)})
            if (gid+1)%8==0:print('PREPARE',split,gid+1,'episodes',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
    # Exact finite-bank marginals, including the concrete rule-parameter instance indexed by graph/rule.
    assert Counter((e['graph'],e['coupled_rule']) for e in exposure)==Counter((e['graph'],e['crossed_rule']) for e in exposure)
    assert len({(e['style'],e['crossed_rule'],e['mask']) for e in exposure})==64
    assert len(set(Counter((e['style'],e['crossed_rule'],e['mask']) for e in exposure).values()))==1
    assert all(len(paths)==c['updates'] for paths in arm_paths.values())
    write(bank/'exposure.json',{'records':exposure,'rule_instance_marginals_identical':True,'crossed_cartesian_triples_balanced':True,
        'note':'One graph/update, no pairing objective. All style/rule types seen in training; test graph/function seeds are new.'})
    write(bank/'oracle_information.json',information)
    write(bank/'manifest.json',{'episodes':list(rows.values()),'train':arm_paths,'test':test_paths,'sources':sources(),
        'official_invalid_retry_policy':'20 deterministic offsets; no feature-information filtering','seconds':time.perf_counter()-start})
    print('PREPARED',len(rows),'episodes',flush=True)


def read(row):
    assert C.sha(ROOT/'bank'/row['path'])==row['sha256']
    with np.load(ROOT/'bank'/row['path']) as z:
        inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__});y=z['labels_query'].astype(np.int64)
    return inp,y


def rowsets(arm=None):
    m=manifest();index={r['path']:r for r in m['episodes']}
    return [index[p] for p in m['train'][arm] if arm] if arm else [index[p] for p in m['test']]


def intervene(inp,row,mode):
    out=copy.deepcopy(inp);d=row['features']
    if mode=='no_observed':
        out.observed[:,:d]=False;out.observed_values[:,:d]=0;out.pcfi[:,:d]=0
    elif mode=='shuffled':
        observed_rows=np.flatnonzero(out.observed[:,:d].any(1))
        order=np.random.default_rng(int(inp.forward_seed)+555).permutation(observed_rows)
        out.observed_values[observed_rows,:d]=out.observed_values[order,:d]
        raw=np.where(out.observed[:,:d],out.observed_values[:,:d],np.nan)
        out.pcfi[:,:d]=C.BASE.fill(raw,out.src,out.dst,'pcfi',row['mechanism']).clip(-10,10)
    return out


def preflight():
    c=cfg();m=manifest();idx={r['path']:r for r in m['episodes']};checks={};digests=[]
    row=next(r for r in m['episodes'] if r['split']=='train' and r['family']=='controlled' and r['rate']==.9)
    inp,y=read(row)
    assert not hasattr(inp,'labels_query') and not hasattr(inp,'hidden_values')
    assert np.isfinite(inp.observed_values).all() and np.all(inp.observed_values[~inp.observed]==0)
    empty=C.BASE.fill(np.full((len(inp.observed),row['features']),np.nan),inp.src,inp.dst,'pcfi',row['mechanism'])
    assert np.all(empty==0),'No-observed PCFI must equal its actual all-missing recomputation'
    for arm in ['official_mask','coupled_pcfi']:
        model=C.Model('pcfi_plain' if arm.endswith('pcfi') else 'mask_plain',11)
        digests.append(C.parameter_digest(model,trainable=True));before=C.parameter_digest(model,trainable=False)
        p=C.predict(model,inp);assert np.array_equal(p,C.predict(model,copy.deepcopy(inp)))
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
        loss.backward();gradient=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert gradient>0 and math.isfinite(gradient)
        assert C.parameter_digest(model,trainable=False)==before
        checks[arm]={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'gradient_l1':gradient,'finite':bool(np.isfinite(p).all())}
        if arm=='official_mask':
            order=np.random.default_rng(777).permutation(len(inp.observed));inverse=np.argsort(order);renamed=copy.deepcopy(inp)
            for key in ['observed_values','observed','pcfi']:setattr(renamed,key,getattr(inp,key)[order])
            renamed.src=inverse[inp.src];renamed.dst=inverse[inp.dst];renamed.context=inverse[inp.context];renamed.query=inverse[inp.query]
            error=float(np.abs(p-C.predict(model,renamed)).max());assert error<.004;checks['node_permutation_error']=error
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(set(digests))==1
    checks.update({'identical_initialization':True,'no_query_labels_in_inputs':True,'sources':sources(),
        'exposure':json.loads((ROOT/'bank/exposure.json').read_text())|{'records':'stored in bank/exposure.json'}})
    write(ROOT/'results/preflight.json',checks);print('PREFLIGHT_OK',flush=True)


def lock():
    from mgpfn import upstream as U
    state={'sources':sources(),'bank_sha256':C.sha(ROOT/'bank/manifest.json'),'exposure_sha256':C.sha(ROOT/'bank/exposure.json'),
        'public_weights':{p.name:C.sha(p) for p in [U.ADAPTERS,U.LIMIX]},'upstream_commit':U.COMMIT}
    p=ROOT/'results/lock.json'
    if p.exists():assert json.loads(p.read_text())==state
    else:write(p,state)
    return state


def load_checkpoint(arm,seed):
    model=C.Model('pcfi_plain' if arm.endswith('pcfi') else 'mask_plain',seed)
    state=torch.load(ROOT/'results'/f'{arm}_s{seed}'/'trainable.pt',map_location='cpu',weights_only=False)['state_dict']
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.requires_grad:p.copy_(state[name].to(p.device))
    return model


def sampled_views(model,inp,row,kind):
    probabilities=[];d=row['features'];rng=np.random.default_rng(int(inp.forward_seed)+170000000)
    for view in range(4):
        given=copy.deepcopy(inp);given.forward_seed=int(inp.forward_seed)+view*100003
        if kind=='random4':
            noise=rng.standard_normal((len(inp.observed),d)).astype(np.float32)
            given.observed_values[:,:d]=np.where(inp.observed[:,:d],inp.observed_values[:,:d],noise)
        probabilities.append(C.predict(model,given))
    return np.mean(np.asarray(probabilities,dtype=np.float64),axis=0)


def evaluate(model,arm,seed,out):
    directory=out/'predictions';directory.mkdir(exist_ok=True);records=[]
    for ri,row in enumerate(rowsets()):
        inp,y=read(row);modes=['standard']
        if row['mechanism'] in ['uniform_nodes','clustered_nodes'] and row['rate']==.9:modes+=['no_observed','shuffled']
        if arm in ['official_mask','crossed_mask'] and row['family']=='official' and row['mechanism'] in ['uniform_nodes','clustered_nodes'] and row['rate']==.995:modes+=['zero4','random4']
        for mode in modes:
            path=directory/f'{ri}_{mode}.npz'
            if path.exists():
                with np.load(path) as z:p=z['p'];assert np.array_equal(y,z['y'])
            else:
                given=intervene(inp,row,mode) if mode in ['no_observed','shuffled'] else inp
                p=sampled_views(model,given,row,mode) if mode in ['zero4','random4'] else C.predict(model,given)
                np.savez_compressed(path,p=p,y=y)
            records.append({**row,'arm':arm,'seed':seed,'mode':mode,**C.metrics(p,y),'prediction':path.relative_to(ROOT).as_posix()})
        if (ri+1)%96==0:
            print('EVAL',arm,seed,ri+1,'/',len(rowsets()),flush=True);write(out/'evaluation_progress.json',{'episodes':ri+1,'total':len(rowsets())})
    write(out/'evaluation.json',records);print('EVALUATED',arm,seed,len(records),flush=True)


def run(seed,only_arm=None):
    c=cfg();state=lock();assert (ROOT/'results/preflight.json').exists()
    schedule=np.random.default_rng(seed+40000).permutation(c['updates']);schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
    for arm in ([only_arm] if only_arm else c['arms']):
        out=ROOT/'results'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'evaluation.json').exists():continue
        if (out/'training.json').exists():model=load_checkpoint(arm,seed);evaluate(model,arm,seed,out);del model;gc.collect();torch.cuda.empty_cache();continue
        assert not (out/'progress.json').exists(),'Partial training preserved; do not silently restart'
        model=C.Model('pcfi_plain' if arm.endswith('pcfi') else 'mask_plain',seed);rows=rowsets(arm);cache=[read(r) for r in rows]
        initial=C.parameter_digest(model,trainable=True);frozen=C.parameter_digest(model,trainable=False)
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'lr':c['base_lr'],'base_lr':c['base_lr']},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'lr':c['new_lr'],'base_lr':c['new_lr']}],weight_decay=c['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',init_scale=128.);logs=[];retries=[];started=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
        for step,idx in enumerate(schedule):
            inp,y=cache[int(idx)];mult=min((step+1)/16,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/c['updates'])))
            for g in optimizer.param_groups:g['lr']=g['base_lr']*mult
            for attempt in range(24):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
                assert torch.isfinite(loss);scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                params=[p for p in model.parameters() if p.requires_grad]
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    retries.append({'step':step+1,'attempt':attempt,'scale':scale});scaler.update(new_scale=scale/2);continue
                norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
                scaler.step(optimizer);scaler.update();break
            else:raise RuntimeError('Finite gradient retries exhausted')
            logs.append({'step':step+1,'logical_episode':int(idx),'path':rows[int(idx)]['path'],'loss':float(loss.detach()),
                'gradient_norm':float(norm),'scale':scaler.get_scale(),'seconds':time.perf_counter()-started})
            if (step+1)%32==0:
                write(out/'progress.json',logs[-1]);print('TRAIN',arm,seed,step+1,'loss',round(np.mean([r['loss'] for r in logs[-32:]]),5),'seconds',round(time.perf_counter()-started,1),flush=True)
        assert C.parameter_digest(model,trainable=False)==frozen and C.parameter_digest(model,trainable=True)!=initial
        torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'arm':arm,'seed':seed,'metadata':state},out/'trainable.pt')
        write(out/'train_log.json',logs);write(out/'overflow_retries.json',retries)
        write(out/'training.json',{'arm':arm,'seed':seed,'updates':len(logs),'schedule_sha256':schedule_sha,'initial_trainable_sha256':initial,
            'frozen_sha256':frozen,'frozen_weights_unchanged':True,'seconds':time.perf_counter()-started,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'overflow_retries':len(retries),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'checkpoint_sha256':C.sha(out/'trainable.pt')})
        print('TRAINED',arm,seed,flush=True);evaluate(model,arm,seed,out)
        del model,optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','run'])
    ap.add_argument('--seed',type=int,choices=cfg()['seeds']);ap.add_argument('--arm',choices=cfg()['arms']);args=ap.parse_args()
    C.setup()
    if args.action=='run':assert args.seed is not None;run(args.seed,args.arm)
    else:globals()[args.action]()


if __name__=='__main__':main()

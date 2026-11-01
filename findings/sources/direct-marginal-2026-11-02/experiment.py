"""Train and evaluate a valid local circuit against direct predictors."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('direct_marginal_public',ROOT.parent/'joint_imputation_pfn/common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
import argparse,copy,gc,hashlib,json,math,time
import numpy as np
import torch
from torch.nn import functional as F
from data import cfg,dump,sha,read,tensors,prepare,tree_for,GridReference,PRIORS,PARAMETERS,MASS,ROLES,NAMES
from model import TreeCircuit,DirectGraph,enumerate_circuit


def sources():return {p.name:sha(p) for p in ROOT.glob('*.py')}|{'protocol':sha(ROOT/'protocol.json'),'public_common':sha(ROOT.parent/'joint_imputation_pfn/common.py')}
def manifest():return json.loads((ROOT/'bank/manifest.json').read_text(encoding='utf8'))
def rows(split):return [r for r in manifest()['episodes'] if r['split']==split]
def lock():
    state={'sources':sources(),'bank_manifest':sha(ROOT/'bank/manifest.json'),'references':sha(ROOT/'references/completed.json')}
    p=ROOT/'results/lock.json'
    if p.exists():assert json.loads(p.read_text())==state
    else:dump(p,state)
    return state


def public_input(inp,query,seed):
    n=len(inp['x']);a=inp['adj'];deg=a.sum(-1);norm=a/np.maximum(deg[:,:,None],1)
    known=inp['context']>=0;seed_labels=np.eye(2)[inp['context'].clip(0)]*known[:,None]
    hop=np.einsum('rij,jc->ric',norm,seed_labels);hop2=np.einsum('rij,rjc->ric',norm,hop)
    features=np.concatenate((inp['x'],deg.T/n,np.tile(inp['prior'],(n,1)),inp['type'][:,None],hop.transpose(1,0,2).reshape(n,-1),hop2.transpose(1,0,2).reshape(n,-1)),1).astype(np.float32)
    mask=np.concatenate((inp['observed'],np.ones((n,features.shape[1]-inp['x'].shape[1]),bool)),1)
    if features.shape[1]%2:features=np.pad(features,((0,0),(0,1)));mask=np.pad(mask,((0,0),(0,1)),constant_values=True)
    union=(a.sum(0)>0)|np.eye(n,dtype=bool);src,dst=np.nonzero(union)
    context=np.flatnonzero(known)
    return C.Inputs(features,mask,features.copy(),src,dst,context,inp['context'][context],query,2,int(seed))


def make_model(arm,seed):
    torch.manual_seed(seed)
    if arm=='mask_pfn':return C.Model('mask_plain',seed)
    if arm=='circuit':return TreeCircuit().to('cuda')
    shared=TreeCircuit().encoder.state_dict()
    torch.manual_seed(seed);net=DirectGraph()
    net.encoder.load_state_dict(shared,strict=False)
    return net.to('cuda')


def preflight():
    torch.set_num_threads(2);torch.manual_seed(101)
    inp,q,y=read(rows('train')[0]['path']);idx=np.asarray([0,1,2,8,9]);small=copy.deepcopy(inp)
    for key in ['x','observed','context','type']:small[key]=small[key][idx]
    small['adj']=small['adj'][:,idx][:,:,idx];small['pair_valid']=small['pair_valid'][:,idx][:,:,idx]
    small['incidence']=small['adj'][:,:3,3:];small['context'][:]=-1;small['context'][0]=0
    small['parent'],small['order']=tree_for(small['adj'],9929991)
    b=tensors([small],'cpu',torch.float64);model=TreeCircuit().double()
    p=model.parameters_for(b);(pred,z,weight,cond),_=model(b,True)
    ze,pe=enumerate_circuit(model,b)
    errors={'enumerated_evidence':float(abs(z[0]-ze)),'enumerated_marginal':float((pred[0]-pe).abs().max())}
    assert max(errors.values())<1e-10,errors
    empty=copy.deepcopy(b);empty['observed'][:]=False;empty['context'][:]=-1
    norm=float(model.evidence(empty).abs().max());assert norm<1e-10,norm
    errors['normalization']=norm
    a=copy.deepcopy(b);a['context'][:,1]=0;d=copy.deepcopy(b);d['context'][:,1]=1
    marginal=torch.logsumexp(torch.stack((model.evidence(a,p),model.evidence(d,p))),0)
    errors['label_sum_rule']=float((model.evidence(b,p)-marginal).abs().max());assert errors['label_sum_rule']<1e-10
    # Same circuit parameters across evidence refinements, not a changing conditional imputer.
    pp=model.parameters_for(empty)
    assert all(torch.equal(p[k],pp[k]) for k in p)
    poison=copy.deepcopy(b);poison['x'][~poison['observed']]=10000
    assert torch.equal(model(b),model(poison))
    # Numerically integrate one revealed Gaussian coordinate and recover the parent evidence.
    from scipy.integrate import quad
    integration=copy.deepcopy(empty);integration['context'][0,0]=0
    parameters=model.parameters_for(integration)
    with torch.no_grad():
        original=float(model.evidence(integration,parameters).exp())
        def density(value):
            refined=copy.deepcopy(integration);refined['observed'][0,1,0]=True;refined['x'][0,1,0]=value
            return float(model.evidence(refined,parameters).exp())
        integrated,quadrature_error=quad(density,-np.inf,np.inf,epsabs=1e-9)
    errors['continuous_feature_sum_rule']=abs(original-integrated);assert errors['continuous_feature_sum_rule']<1e-8
    signed=copy.deepcopy(empty);root=int(signed['order'][0,0]);child=int(signed['order'][0,1]);signed['context'][0,root]=0
    signed_p=model.parameters_for(signed);signed_p['j']=torch.full_like(signed_p['j'],-3.)
    signed_pred=model.sum_product(signed,signed_p,model.emissions(signed,signed_p))[0]
    opposite_probability=float(signed_pred[0,child,1]);assert opposite_probability>.9
    # Actual training path has finite gradients through evidence and conditional prediction.
    model.zero_grad();prob=model(b);loss=-prob[0,2,1].log();loss.backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    assert sum(float(v.grad.abs().sum()) for v in model.parameters() if v.grad is not None)>0
    # Direct conditional ratio agrees with the all-node marginal returned by sum-product.
    e0=copy.deepcopy(b);e1=copy.deepcopy(b);e0['context'][0,2]=0;e1['context'][0,2]=1
    ratio=torch.stack((model.evidence(e0,p),model.evidence(e1,p)),-1).softmax(-1)
    errors['conditional_ratio']=float((ratio-pred[:,2]).abs().max());assert errors['conditional_ratio']<1e-10
    # Independent scalar known-family reference on the smaller native bipartite graph.
    engine=GridReference('cpu',3,2,2)
    with torch.no_grad():ref=engine.infer([small],with_bp=False)['grid6_exact'][0]
    labels=((np.arange(2**5)[:,None]>>np.arange(5))&1);sp=2*labels-1;mass=np.zeros(2**5)
    for gi,(density,signal) in enumerate(PARAMETERS):
        for hh,h in enumerate(ROLES):
            prob=density*(1+signal*h[:,None,None,None]*sp[None,:,:3,None]*sp[None,:,None,3:])
            evidence=np.where(small['incidence'][:,None].astype(bool),prob,1-prob).prod((0,2,3))
            feature=np.ones(len(labels))
            for f in range(2):
                terms=[]
                for sign in [-1,1]:
                    logs=-.5*((small['x'][:,f][None]-.8*sign*sp)/.6)**2-math.log(.6*math.sqrt(2*math.pi))
                    terms.append(np.exp((logs*small['observed'][:,f][None]).sum(-1)))
                feature*=np.mean(terms,0)
            valid=((small['context'][None]<0)|(labels==small['context'][None])).all(-1)
            mass+=MASS[gi]*PRIORS[NAMES[int(small['prior'].argmax())]][hh]*evidence*feature*valid
    mass/=mass.sum();p1=mass@labels;slow=np.stack((1-p1,p1),-1)
    errors['grid6_scalar_reference']=float(np.max(np.abs(ref-slow)));assert errors['grid6_scalar_reference']<1e-10
    # Check tree scopes and ordering, independent of numerical tests.
    assert len(set(small['order']))==len(idx)
    for node in small['order'][1:]:assert list(small['order']).index(small['parent'][node])<list(small['order']).index(node)
    # Public PFN uses only explicitly visible values and derived G/context metadata.
    public=public_input(inp,q,123);assert not hasattr(public,'labels_query')
    C.setup();net=C.Model('mask_plain',11);output=C.predict(net,public);assert np.isfinite(output).all()
    dump(ROOT/'results/preflight.json',{'pass_all':True,'errors':errors,'quadrature_reported_error':quadrature_error,
        'opposite_class_probability_with_only_class0_context':opposite_probability,
        'one_scope_per_original_variable':True,'parameters_independent_of_feature_label_evidence':True,
        'latent_and_query_truth_not_input':True,'public_mask_pfn_trainable_parameters':sum(v.numel() for v in net.parameters() if v.requires_grad),
        'circuit_parameters':sum(v.numel() for v in model.parameters()),'sources':sources()})
    print('PREFLIGHT_OK',json.dumps(errors),flush=True)


def probabilities(model,arm,inp,query,forward_seed):
    if arm=='mask_pfn':return C.predict(model,public_input(inp,query,forward_seed))
    model.eval()
    with torch.no_grad():p=model(tensors([inp]))[0,query].cpu().numpy()
    assert np.isfinite(p).all();return p


def evaluate(model,arm,seed,out):
    pred=out/'predictions';pred.mkdir(exist_ok=True);records=[];began=time.perf_counter()
    for i,row in enumerate(rows('test')):
        inp,q,y=read(row['path']);modes=['standard']
        if row['context_total']==2 and row['mechanism']=='clustered_nodes' and row['rate']==.5:modes+=['no_observed','shuffled']
        for mode in modes:
            path=pred/f'{i}_{mode}.npz'
            if path.exists():
                with np.load(path) as z:p=z['p'];assert np.array_equal(y,z['y'])
            else:
                given=copy.deepcopy(inp)
                if mode=='no_observed':given['observed'][:]=False;given['x'][:]=0
                if mode=='shuffled':
                    for f in range(given['x'].shape[1]):
                        ids=np.flatnonzero(given['observed'][:,f]);order=np.random.default_rng(700000+i).permutation(ids)
                        given['x'][ids,f]=given['x'][order,f]
                p=probabilities(model,arm,given,q,19000000+i);np.savez_compressed(path,p=p,y=y)
            records.append({**row,'arm':arm,'seed':seed,'mode':mode,**C.metrics(p,y),'prediction':path.relative_to(ROOT).as_posix()})
        if (i+1)%768==0:print('EVAL',arm,seed,i+1,flush=True)
    inp,q,y=read(rows('test')[0]['path'])
    for _ in range(5):probabilities(model,arm,inp,q,1)
    torch.cuda.synchronize();latency=[];torch.cuda.reset_peak_memory_stats()
    for _ in range(30):
        t=time.perf_counter();probabilities(model,arm,inp,q,1);torch.cuda.synchronize();latency.append(1000*(time.perf_counter()-t))
    dump(out/'timing.json',{'median_ms':float(np.median(latency)),'samples_ms':latency,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
        'scope':'Same GPU, CPU visible inputs to query probabilities; includes public typed-support preprocessing, excludes bank I/O. Concurrent jobs may affect wall-clock measurements; no unqualified speed claim.'})
    dump(out/'evaluation.json',records);dump(out/'evaluation_time.json',{'seconds':time.perf_counter()-began});print('EVALUATED',arm,seed,len(records),flush=True)


def run(seed,only_arm=None):
    c=cfg();state=lock();assert (ROOT/'results/preflight.json').exists();C.setup();torch.set_num_threads(2)
    schedule=np.random.default_rng(seed+42000).permutation(c['train_episodes']);schedulesha=hashlib.sha256(schedule.tobytes()).hexdigest();rr=rows('train');cache=[read(r['path']) for r in rr]
    for arm in ([only_arm] if only_arm else c['arms']):
        out=ROOT/'results'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'evaluation.json').exists():continue
        model=make_model(arm,seed)
        if (out/'training.json').exists():
            saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if p.requires_grad:p.copy_(saved[n].to(p.device))
            evaluate(model,arm,seed,out);del model;gc.collect();torch.cuda.empty_cache();continue
        assert not (out/'progress.json').exists(),'No silent partial restart'
        frozen=C.parameter_digest(model,trainable=False);initial=C.parameter_digest(model,trainable=True)
        if arm=='mask_pfn':
            groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':c['public_base_lr']},
                    {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':c['public_new_lr']}]
        else:groups=[{'params':list(model.parameters()),'base_lr':c['lr']}]
        optimizer=torch.optim.AdamW(groups,weight_decay=c['weight_decay']);scaler=torch.amp.GradScaler('cuda',init_scale=64.,enabled=arm=='mask_pfn')
        logs=[];retries=[];began=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
        for step in range(c['updates']):
            ids=schedule[step*c['batch_size']:(step+1)*c['batch_size']]
            for group in optimizer.param_groups:group['lr']=group['base_lr']*min((step+1)/16,1)*(.1+.9*.5*(1+math.cos(math.pi*step/c['updates'])))
            for attempt in range(24):
                optimizer.zero_grad(set_to_none=True)
                if arm=='mask_pfn':
                    losses=[]
                    for i in ids:
                        inp,q,y=cache[int(i)];given=public_input(inp,q,int(i)+24000000+seed)
                        with torch.autocast('cuda',dtype=torch.float16):one=F.cross_entropy(model(given),torch.as_tensor(y,device='cuda'))/len(ids)
                        scaler.scale(one).backward();losses.append(float(one.detach()))
                    value=sum(losses);scaler.unscale_(optimizer)
                else:
                    items=[cache[int(i)][0] for i in ids];batch=tensors(items);p=model(batch);losses=[]
                    for j,i in enumerate(ids):
                        _,q,y=cache[int(i)];losses.append(-p[j,q,torch.as_tensor(y,device='cuda')].clamp_min(1e-10).log().mean())
                    loss=torch.stack(losses).mean();assert torch.isfinite(loss);loss.backward();value=float(loss.detach())
                params=[p for p in model.parameters() if p.requires_grad]
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    if arm!='mask_pfn':raise RuntimeError('Nonfinite circuit/direct gradient')
                    scale=scaler.get_scale();retries.append({'step':step+1,'attempt':attempt,'scale':scale});scaler.update(new_scale=scale/2);continue
                norm=torch.nn.utils.clip_grad_norm_(params,c['clip'],error_if_nonfinite=True)
                if arm=='mask_pfn':scaler.step(optimizer);scaler.update()
                else:optimizer.step()
                break
            else:raise RuntimeError('Finite-gradient retries exhausted')
            logs.append({'step':step+1,'loss':value,'gradient_norm':float(norm),'seconds':time.perf_counter()-began})
            if (step+1)%32==0:dump(out/'progress.json',logs[-1]);print('TRAIN',arm,seed,step+1,'loss',round(np.mean([r['loss'] for r in logs[-32:]]),5),flush=True)
        assert C.parameter_digest(model,trainable=False)==frozen
        torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'metadata':state},out/'checkpoint.pt')
        dump(out/'train_log.json',logs);dump(out/'overflow_retries.json',retries)
        dump(out/'training.json',{'arm':arm,'seed':seed,'updates':len(logs),'examples':c['train_episodes'],'schedule_sha256':schedulesha,'initial_sha256':initial,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'total_parameters':sum(p.numel() for p in model.parameters()),
            'frozen_weights_unchanged':True,'seconds':time.perf_counter()-began,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'checkpoint_sha256':sha(out/'checkpoint.pt')})
        evaluate(model,arm,seed,out);del model,optimizer;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','run']);ap.add_argument('--seed',type=int);ap.add_argument('--arm');a=ap.parse_args()
    torch.set_num_threads(2)
    if a.action=='prepare':prepare()
    elif a.action=='preflight':preflight()
    else:run(a.seed,a.arm)
if __name__=='__main__':main()

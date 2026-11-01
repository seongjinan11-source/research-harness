"""Deployment-only Grid6-BP timing: do not also compute exact query marginals.

Training/model/primary-analysis sources remain unchanged. The preparatory
reference engine returns exact and BP outputs together, which is convenient for
teacher generation but not fair when measuring BP-only deployment. This entry
point removes that extra output computation and verifies numerical equivalence
before measuring anything. It is not a parameter or prediction change.
"""
import sys
sys.dont_write_bytecode=True
import sklearn  # Initialize Windows OpenMP before torch, as in the main workers.
from pathlib import Path
WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'direct_marginal_pilot'))
import gc,hashlib,json,math
from datetime import datetime,timezone
import numpy as np
import torch
from experiment import C,ROOT,cfg,make_model,rows,read,probabilities
from data import GridReference,tensors,dump,PRIORS,NAMES,PARAMETERS,MASS
from benchmark import measure


def bp_only(engine,items):
    b=tensors(items,engine.device,torch.float64);e=b['incidence'].flatten(2);ne=1-e;count=len(e)
    pri=torch.as_tensor(np.stack([PRIORS[NAMES[int(x['prior'].argmax())]] for x in items]),device=engine.device,dtype=torch.float64)
    xi=torch.tensor(PARAMETERS,device=engine.device)[None].expand(count,-1,-1);mass=torch.tensor(MASS,device=engine.device)[None].expand(count,-1)
    d,s=xi.unbind(-1);plus=d*(1+s);minus=d*(1-s)
    ce=.5*(plus.log()+minus.log())-d.log();cn=.5*(torch.log1p(-plus)+torch.log1p(-minus))-torch.log1p(-d)
    je=.5*(plus.log()-minus.log());jn=.5*(torch.log1p(plus.neg())-torch.log1p(minus.neg()))
    ea=e.sum(-1)@engine.roles.T;na=ne.sum(-1)@engine.roles.T
    es=torch.einsum('bre,se,hr->bhs',e,engine.products,engine.roles);ns=torch.einsum('bre,se,hr->bhs',ne,engine.products,engine.roles)
    constant=e.sum((1,2))[:,None]*d.log()+ne.sum((1,2))[:,None]*torch.log1p(-d)
    logg=constant[:,:,None,None]+ea[:,None,:,None]*ce[:,:,None,None]+na[:,None,:,None]*cn[:,:,None,None]+es[:,None]*je[:,:,None,None]+ns[:,None]*jn[:,:,None,None]
    mu=cfg()['feature_signal'];sd=cfg()['feature_noise'];obs=b['observed'].double();x=b['x']
    field=mu/(sd*sd)*torch.einsum('bnf,af->ban',obs*x,engine.orient)
    fx=torch.einsum('ban,sn->bas',field,engine.spin)+(-.5*((x*x+mu*mu)/(sd*sd)+math.log(2*math.pi*sd*sd))*obs).sum((1,2))[:,None,None]
    c=b['context'];allowed=((c[:,None]<0)|(c[:,None]==engine.labels[None])).all(-1)
    score=logg[:,:,:,None,:]+fx[:,None,None]+mass.log()[:,:,None,None,None]+pri.log()[:,None,:,None,None]-math.log(len(engine.orient))
    score=score.masked_fill(~allowed[:,None,None,None],-torch.inf)
    weight=torch.logsumexp(score,-1).flatten(1).softmax(-1)
    # No all-assignment softmax or exact label-moment multiplication here.
    jrel=e[:,None]*je[:,:,None,None]+ne[:,None]*jn[:,:,None,None]
    couplings=torch.einsum('bgre,hr->bghe',jrel,engine.roles).reshape(count,6,4,engine.nt,engine.nh)
    j=logg.new_zeros(count,6,4,engine.n,engine.n);j[:,:,:,:engine.nt,engine.nt:]=couplings;j[:,:,:,engine.nt:,:engine.nt]=couplings.transpose(-1,-2)
    a=j[:,:,:,None].expand(-1,-1,-1,len(engine.orient),-1,-1).flatten(1,3)
    u=torch.zeros_like(a);t=a.tanh();unary=field[:,None,None].expand(-1,6,4,-1,-1).flatten(1,3)
    known=(c>=0)[:,None,:,None];spin=(2*c-1)[:,None,:,None]
    for _ in range(64):
        cavity=(unary[...,None]+u.sum(-2)[...,:,None]-u.transpose(-1,-2)).tanh();cavity=torch.where(known,spin,cavity)
        u=.5*u+.5*torch.atanh((t*cavity).clamp(-1+1e-15,1-1e-15))
    means=(unary+u.sum(-2)).tanh();means=torch.where((c>=0)[:,None],(2*c-1)[:,None],means)
    p=(means+1)/2;bp=torch.stack((1-p,p),-1)
    return torch.einsum('bh,bhnc->bnc',weight,bp).cpu().numpy()


def main():
    C.setup();torch.set_num_threads(2)
    proof={'reason':'Remove exact predictive-moment calculation from BP-only deployment timing. No training, weights, inputs, predictions, priors or primary accuracy/NLL gates changed.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'locked_before_timing_utc':datetime.now(timezone.utc).isoformat(),
        'original_reference_engine':hashlib.sha256((ROOT/'data.py').read_bytes()).hexdigest()}
    dump(ROOT/'runtime/measurement_lock.json',proof)
    engine=GridReference('cuda');samples=rows('test')[::193][:16];inputs=[read(r['path'])[0] for r in samples]
    with torch.inference_mode():
        expected=engine.infer(inputs,with_bp=True)['grid6_bp'];actual=bp_only(engine,inputs)
    error=float(np.max(np.abs(expected-actual)));assert error<1e-10,error
    row=next(r for r in rows('test') if r['case']=='matched' and r['context_total']==2 and r['mechanism']=='full')
    inp,q,y=read(row['path']);result=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            out=ROOT/'results'/f'{arm}_s{seed}';model=make_model(arm,seed);state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if p.requires_grad:p.copy_(state[n].to(p.device))
            result.append({'arm':arm,'seed':seed,**measure(lambda:probabilities(model,arm,inp,q,111))})
            del model,state;gc.collect();torch.cuda.empty_cache()
    with torch.inference_mode():
        result.append({'arm':'grid6_bp','seed':0,**measure(lambda:bp_only(engine,[inp]))})
        result.append({'arm':'grid6_exact','seed':0,**measure(lambda:engine.infer([inp],with_bp=False))})
    dump(ROOT/'results/benchmark.json',{'device':torch.cuda.get_device_name(),'row':row,'measurements':result,'bp_only_equivalence_max_error':error,'measurement_source':proof,
        'scope':'Visible input to probabilities, same GPU, after our model workers finish. BP reference excludes unnecessary exact query marginal output. Model construction/bank I/O excluded. External GPU activity uncontrolled; repetitions retained.'})
    print('BENCHMARK_CORRECTED_DONE',error,flush=True)
if __name__=='__main__':main()

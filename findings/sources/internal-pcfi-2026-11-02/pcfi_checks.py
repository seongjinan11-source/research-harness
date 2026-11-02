"""Development checks, discarded actual optimizer steps, then source lock."""
import sys
sys.dont_write_bytecode = True
import sklearn
import copy
import gc
import numpy as np
import torch
from pcfi_shared import ROOT,C,cfg,dump,sha,sources,read,rows,prepare_geometry,adjacency,impute
from pcfi_model import Model,Router,predict,weights,diffuse,correlate
from pcfi_trial import optimizer,step,restore


def main():
    assert not (ROOT/'results/design.json').exists()
    C.setup();torch.set_num_threads(2)
    errors={};meta=[];initial={};common=set();frozen=set()
    # Independently computed SciPy literal PCFI, including structural and cell masks.
    rng=np.random.default_rng(777)
    for cell in [False,True]:
        n,d=9,4
        src=np.r_[np.arange(n),np.arange(n-1),np.arange(1,n)]
        dst=np.r_[np.arange(n),np.arange(1,n),np.arange(n-1)]
        full=rng.normal(size=(n,d)).astype(np.float32)
        mask=rng.random((n,d))>.65 if cell else np.broadcast_to((np.arange(n)%3==0)[:,None],(n,d)).copy()
        mask[:,3]=False # explicit no-source column, with cell-wise geometry
        x=np.where(mask,full,0)
        inp=C.Inputs(x,mask,x,src,dst,np.array([0]),np.array([0]),np.array([1]),2,1)
        geo=prepare_geometry(inp,d)
        c=torch.tensor(geo['confidence'],dtype=torch.float64)
        i,j=torch.tensor(geo['edge_i']),torch.tensor(geo['edge_j'])
        tx,tm=torch.tensor(x,dtype=torch.float64),torch.tensor(mask)
        out=correlate(diffuse(tx,tx,tm,weights(c,i,j,n),i,j,100),c).numpy()
        reference=impute(adjacency((src,dst),n),np.where(mask,full,np.nan),'pcfi','uniform',iterations=100)[0]
        errors['pcfi_scipy_'+str(cell)]=float(abs(out-reference).max())
        assert errors['pcfi_scipy_'+str(cell)]<2e-5
        assert np.array_equal(out[mask],x[mask]) and not out[:,3].any()
    dev=rows('dev')
    row=next(r for r in dev if r['mechanism']=='uniform_nodes' and r['rate']==.9 and r['observed_nodes']>=2)
    item=read(row);inp,extra,y=item
    for arm in cfg()['arms']:
        model=Model(arm,101)
        common.add(C.parameter_digest(model.base,trainable=True));frozen.add(C.parameter_digest(model,trainable=False))
        initial[arm]=predict(model,inp,extra)
        poison=copy.deepcopy(inp);poison.observed_values[~poison.observed]=123456.
        errors[arm+'_poison']=float(abs(predict(model,poison,extra)-initial[arm]).max())
        assert errors[arm+'_poison']==0
        model.train();opt=optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        frozen_before=C.parameter_digest(model,trainable=False)
        loss,norm,retries=step(model,opt,scaler,item,0)
        assert model.last_diagnostics['observed_clamp_max']==0
        if hasattr(model,'router'):
            grad=sum(float(p.grad.abs().sum()) for p in model.router.parameters() if p.grad is not None)
            assert np.isfinite(grad) and grad>0,(arm,grad)
            if arm=='label_blind':
                # Strict label blindness of routing even with arbitrary poisoned PFN states/labels.
                n,d=inp.observed.shape
                val=torch.tensor(inp.observed_values,device='cuda');obs=torch.tensor(inp.observed,device='cuda')
                c=torch.tensor(extra['confidence'],device='cuda');i=torch.tensor(extra['edge_i'],device='cuda');j=torch.tensor(extra['edge_j'],device='cuda')
                h=torch.randn(1,n,d//2+1,192,device='cuda')
                labels=torch.tensor(inp.labels_context,device='cuda')
                a=model.router(val,obs,c,i,j,1,h,labels,True)
                b=model.router(val,obs,c,i,j,1,h*100,labels.flip(0),True)
                errors['strict_label_blind_router']=float(abs(a-b).max());assert errors['strict_label_blind_router']==0
            else:
                n,d=inp.observed.shape
                val=torch.tensor(inp.observed_values,device='cuda');obs=torch.tensor(inp.observed,device='cuda')
                c=torch.tensor(extra['confidence'],device='cuda');i=torch.tensor(extra['edge_i'],device='cuda');j=torch.tensor(extra['edge_j'],device='cuda')
                h=torch.randn(1,n,d//2+1,192,device='cuda',requires_grad=True)
                labels=torch.tensor(inp.labels_context,device='cuda')
                a=model.router(val,obs,c,i,j,1,h,labels)
                b=model.router(val,obs,c,i,j,1,h,labels.max()-labels)
                errors['router_class_name_permutation']=float(abs(a-b).max());assert errors['router_class_name_permutation']<1e-6
                hgrad=torch.autograd.grad(a.square().sum(),h)[0]
                assert torch.isfinite(hgrad).all() and hgrad.abs().sum()>0
        assert C.parameter_digest(model,trainable=False)==frozen_before
        model.eval();p=predict(model,inp,extra)
        order=np.random.default_rng(17).permutation(len(inp.observed));inv=np.argsort(order)
        renamed=copy.deepcopy(inp)
        for k in ['observed_values','observed','pcfi']:setattr(renamed,k,getattr(inp,k)[order])
        renamed.src=inv[inp.src];renamed.dst=inv[inp.dst];renamed.context=inv[inp.context];renamed.query=inv[inp.query]
        geo={'confidence':extra['confidence'][order],'edge_i':inv[extra['edge_i']],'edge_j':inv[extra['edge_j']]}
        errors[arm+'_node_permutation']=float(abs(predict(model,renamed,geo)-p).max())
        assert errors[arm+'_node_permutation']<.006
        full=read(next(r for r in dev if r['mechanism']=='full'))
        pf=predict(model,full[0],full[1]);assert model.last_diagnostics['observed_clamp_max']==0
        if arm=='context':
            errors['full_router_blank']=float(abs(predict(model,full[0],full[1],blank_router_context=True)-pf).max())
            assert errors['full_router_blank']==0
        # Save/restore the same trainable-only state format/helper used in actual training.
        state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        path=ROOT/'results'/f'dev_discard_{arm}.pt';path.parent.mkdir(exist_ok=True,parents=True);torch.save(state,path)
        copied=Model(arm,101);restore(copied,torch.load(path,map_location='cpu',weights_only=False))
        errors[arm+'_checkpoint_reload']=float(abs(predict(copied,inp,extra)-p).max())
        assert errors[arm+'_checkpoint_reload']==0
        meta.append({'arm':arm,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                     'new':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('base.')),'smoke_loss':loss,'grad_norm':norm,'retries':retries})
        del model,copied,opt;gc.collect();torch.cuda.empty_cache()
    assert len(common)==1 and len(frozen)==1
    for arm in ['context','label_blind']:
        errors[arm+'_initial_vs_internal_fixed']=float(abs(initial[arm]-initial['internal_fixed']).max())
        assert errors[arm+'_initial_vs_internal_fixed']==0
    for r in rows('train')+rows('dev'):
        assert sha(ROOT/'bank'/r['path'])==r['sha256']
    dump(ROOT/'results/preflight.json',{'errors':errors,'models':meta,'actual_optimizer_smoke_updates_discarded':4,
         'same_common_initialization':True,'same_frozen_weights':True,'scope':'Dev only; initial models/optimizer weights discarded. Numeric PCFI parity and routing path checks do not establish predictive improvement.'})
    dump(ROOT/'results/design.json',{'sources':sources(),'development_manifest':sha(ROOT/'bank/development.json'),
         'preflight':sha(ROOT/'results/preflight.json'),'scope':'Fixed before generating final graph seeds'})
    print('DEV_PASSED_DESIGN_LOCKED',errors,meta,flush=True)


if __name__=='__main__':main()

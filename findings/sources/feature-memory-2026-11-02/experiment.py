"""Matched public GraphPFN feature-memory routing pilot. No upstream edits."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('memory_prior_reference', ROOT.parent/'mechanism_crossed_prior/experiment.py')
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M; spec.loader.exec_module(M)
M.ROOT = ROOT
C = M.C
import argparse, copy, functools, gc, hashlib, json, math, time
import numpy as np
import torch
import torch.nn.functional as F


def cfg(): return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); C.write_json(path, obj)
def source_hashes():
    return {p.name:C.sha(p) for p in ROOT.glob('*.py')} | {'protocol':C.sha(ROOT/'protocol.json'),
        'common':C.sha(ROOT.parent/'joint_imputation_pfn/common.py'),
        'prior':C.sha(ROOT.parent/'mechanism_crossed_prior/experiment.py')}
def rows(split='test'):
    m=M.manifest(); index={r['path']:r for r in m['episodes']}
    if split=='test': return [index[p] for p in m['test']]
    # Identical alternating official and crossed stream in every arm.
    return [index[p] for pair in zip(m['train']['official_mask'],m['train']['crossed_mask']) for p in pair]
def read(row):
    inp,y=M.read(row); inp.real_features=int(row['features']); return inp,y


class MemoryModel(C.Model):
    def __init__(self, routing, seed):
        super().__init__('pcfi_plain' if routing=='pcfi' else 'mask_plain',seed)
        self.routing=routing; self._memory_mask=None; self.route_calls=0
        if routing not in ('context','pcfi'):
            for wrapper in self.base.tfm.module.transformer_encoder.layers:
                layer=wrapper.base
                for i, step in enumerate(layer.layer_steps):
                    if isinstance(step,functools.partial) and step.func.__name__=='call_sequence_attention':
                        layer.layer_steps[i]=functools.partial(self.routed_sequence,layer,index=step.keywords.get('index',0))

    def forward(self, inp, **kwargs):
        ids=np.concatenate((inp.context,np.setdiff1d(np.arange(len(inp.observed)),inp.context)))
        obs=np.asarray(inp.observed[ids]).copy()
        obs[:,int(inp.real_features):]=False  # Padding is never an actual observation.
        groups=obs.reshape(len(obs),-1,2).any(-1)
        if self.routing=='all_nodes': groups[:]=True
        labels=np.arange(len(obs))<len(inp.context)
        mask=np.concatenate((groups,labels[:,None]),axis=1).T
        self._memory_mask=torch.as_tensor(mask,device='cuda',dtype=torch.bool)
        try: return super().forward(inp,**kwargs)
        finally: self._memory_mask=None

    def routed_sequence(self, layer, x, feature_atten_mask, eval_pos, index=0, **kwargs):
        # Original projections, norms, residuals and graph adapters are shared.
        # Only the sample-axis key/value memory changes for feature tokens.
        a=index*2 if layer.seq_attn_isolated else index
        b=a+1 if layer.seq_attn_isolated else a
        allowed=self._memory_mask
        assert allowed is not None and allowed.shape==(x.shape[2],x.shape[1])
        null=~allowed.any(-1,keepdim=True)
        assert x.shape[0]==1
        allowed=torch.cat((allowed,null),-1)[:,None,None,:]
        def attend(module,q,kv,share):
            kv=kv.transpose(1,2)
            kv=torch.cat((kv,kv.new_zeros((kv.shape[0],kv.shape[1],1,kv.shape[-1]))),2)
            return module(x=q.transpose(1,2),x_kv=kv,copy_first_head_kv=share,attn_mask=allowed)[0].transpose(1,2)
        train=attend(layer.sequence_attentions[a],x[:,:eval_pos],x,layer.self_share_all_kv_heads)
        kv=torch.cat((train,x[:,eval_pos:]),1) if layer.seq_attn_serial else x
        test=attend(layer.sequence_attentions[b],x[:,eval_pos:],kv,layer.cross_share_all_kv_heads)
        self.route_calls+=1
        return torch.cat((train,test),1),None,None


def preflight():
    sample=next(r for r in rows('train') if r['rate']==.9)
    inp,y=read(sample); out={}; initial=[]
    for arm in cfg()['routing_arms']:
        model=MemoryModel(arm,11); initial.append(C.parameter_digest(model,trainable=True))
        before=C.parameter_digest(model,trainable=False)
        p=C.predict(model,inp); assert np.array_equal(p,C.predict(model,copy.deepcopy(inp)))
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
        scaler=torch.amp.GradScaler('cuda',init_scale=8.); scaler.scale(loss).backward()
        grads=[v.grad for v in model.parameters() if v.requires_grad and v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert C.parameter_digest(model,trainable=False)==before
        empty=copy.deepcopy(inp); empty.observed[:,:empty.real_features]=False; empty.observed_values[:]=0; empty.pcfi[:]=0
        assert np.isfinite(C.predict(model,empty)).all()
        order=np.random.default_rng(991).permutation(len(inp.observed)); inv=np.argsort(order); renamed=copy.deepcopy(inp)
        for key in ['observed_values','observed','pcfi']:setattr(renamed,key,getattr(inp,key)[order])
        renamed.src=inv[inp.src]; renamed.dst=inv[inp.dst]; renamed.context=inv[inp.context]; renamed.query=inv[inp.query]
        error=float(np.abs(p-C.predict(model,renamed)).max()); assert error<.005
        out[arm]={'trainable':sum(v.numel() for v in model.parameters() if v.requires_grad),'gradient_l1':sum(float(g.abs().sum()) for g in grads),
            'node_permutation_error':error,'empty_memory_finite':True,'route_calls':model.route_calls}
        if arm=='context':
            reference=C.Model('mask_plain',11); ref=C.predict(reference,inp)
            assert np.array_equal(p,ref); out['context_reference_exactly_identical']=True
            del reference
        del model; gc.collect(); torch.cuda.empty_cache()
    assert len(set(initial))==1
    write(ROOT/'results/preflight.json',{'arms':out,'same_initialization':True,'query_labels_not_in_inputs':True,
        'feature_group_scope':'Two-coordinate native feature groups; original two-bit missing mask distinguishes partial observations.',
        'sources':source_hashes()})
    print('PREFLIGHT_OK',flush=True)


def lock():
    state={'sources':source_hashes(),'bank_manifest':C.sha(ROOT/'bank/manifest.json')}
    p=ROOT/'results/lock.json'
    if p.exists():assert json.loads(p.read_text())==state
    else:write(p,state)
    return state


def load(arm,seed):
    model=MemoryModel(arm,seed)
    s=torch.load(ROOT/'results'/f'{arm}_s{seed}'/'trainable.pt',map_location='cpu',weights_only=False)['state_dict']
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:p.copy_(s[n].to(p.device))
    return model


def evaluate(model,arm,seed,out):
    pred=out/'predictions'; pred.mkdir(exist_ok=True); records=[]; began=time.perf_counter()
    for i,row in enumerate(rows()):
        inp,y=read(row); modes=['standard']
        if row['rate']==.9 and row['pair'] in ('official','offdiagonal') and row['mechanism']=='clustered_nodes':
            modes+=['no_observed','shuffled']
        for mode in modes:
            path=pred/f'{i}_{mode}.npz'
            if path.exists():
                with np.load(path) as z:p=z['p']; assert np.array_equal(y,z['y'])
            else:
                given=M.intervene(inp,row,mode) if mode!='standard' else inp
                p=C.predict(model,given); np.savez_compressed(path,p=p,y=y)
            records.append({**row,'arm':arm,'seed':seed,'mode':mode,**C.metrics(p,y),'prediction':path.relative_to(ROOT).as_posix()})
        if (i+1)%192==0: print('EVAL',arm,seed,i+1,flush=True)
    inp,y=read(rows()[0]); model.eval()
    for _ in range(5): C.predict(model,inp)
    torch.cuda.synchronize(); timings=[]; torch.cuda.reset_peak_memory_stats()
    for _ in range(20):
        t=time.perf_counter(); C.predict(model,inp); torch.cuda.synchronize(); timings.append(1000*(time.perf_counter()-t))
    write(out/'timing.json',{'median_model_forward_ms':float(np.median(timings)),'samples_ms':timings,
        'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'scope':'CPU input to probabilities, excludes graph generation/PCFI preprocessing; 20 repetitions on first fixed test graph.'})
    write(out/'evaluation.json',records); write(out/'evaluation_time.json',{'seconds':time.perf_counter()-began})
    print('EVALUATED',arm,seed,len(records),flush=True)


def run(seed):
    c=cfg(); state=lock(); assert (ROOT/'results/preflight.json').exists()
    schedule=np.random.default_rng(seed+40000).permutation(c['training_updates']); training=rows('train')
    assert len(training)==len(schedule)
    schedule_sha=hashlib.sha256(schedule.tobytes()).hexdigest()
    for arm in c['routing_arms']:
        out=ROOT/'results'/f'{arm}_s{seed}'; out.mkdir(parents=True,exist_ok=True)
        if (out/'evaluation.json').exists():continue
        if (out/'training.json').exists():
            model=load(arm,seed); evaluate(model,arm,seed,out); del model; gc.collect(); torch.cuda.empty_cache(); continue
        assert not (out/'progress.json').exists(),'Partial run retained; no silent restart.'
        model=MemoryModel(arm,seed); cache=[read(r) for r in training]
        initial=C.parameter_digest(model,trainable=True); frozen=C.parameter_digest(model,trainable=False)
        optimizer=torch.optim.AdamW([
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],'base_lr':c['base_lr']},
            {'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('base.')],'base_lr':c['new_lr']}],weight_decay=c['weight_decay'])
        scaler=torch.amp.GradScaler('cuda',init_scale=128.); logs=[]; retries=[]; began=time.perf_counter(); model.train(); torch.cuda.reset_peak_memory_stats()
        for step,idx in enumerate(schedule):
            inp,y=cache[int(idx)]; mult=min((step+1)/16,1.)*(.1+.9*.5*(1+math.cos(math.pi*step/len(schedule))))
            for group in optimizer.param_groups:group['lr']=group['base_lr']*mult
            for attempt in range(24):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=F.cross_entropy(model(inp),torch.as_tensor(y,device='cuda'))
                assert torch.isfinite(loss); scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                params=[p for p in model.parameters() if p.requires_grad]
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    retries.append({'step':step+1,'attempt':attempt,'scale':scale}); scaler.update(new_scale=scale/2);continue
                norm=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip'],error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); break
            else:raise RuntimeError('Finite gradient retries exhausted')
            logs.append({'step':step+1,'logical_episode':int(idx),'path':training[int(idx)]['path'],'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.perf_counter()-began})
            if (step+1)%64==0:
                write(out/'progress.json',logs[-1]); print('TRAIN',arm,seed,step+1,'loss',round(np.mean([r['loss'] for r in logs[-64:]]),5),flush=True)
        assert C.parameter_digest(model,trainable=False)==frozen
        torch.save({'state_dict':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'metadata':state},out/'trainable.pt')
        write(out/'train_log.json',logs); write(out/'overflow_retries.json',retries)
        write(out/'training.json',{'arm':arm,'seed':seed,'updates':len(logs),'schedule_sha256':schedule_sha,'initial_trainable_sha256':initial,
            'frozen_sha256':frozen,'frozen_weights_unchanged':True,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'seconds':time.perf_counter()-began,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'overflow_retries':len(retries),'checkpoint_sha256':C.sha(out/'trainable.pt')})
        evaluate(model,arm,seed,out); del model,optimizer; gc.collect(); torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','run']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    C.setup();torch.set_num_threads(2)
    if a.action=='prepare':M.prepare()
    elif a.action=='preflight':preflight()
    else:run(a.seed)
if __name__=='__main__':main()

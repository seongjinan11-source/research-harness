import sys
sys.dont_write_bytecode=True
import sklearn
import copy
import gc
import json
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT,WORK,B,C,cfg,dump,sha,Model,load_episode,frontend,transform_extra,signed_gain,lock


def main():
    C.setup()
    torch.set_num_threads(2)
    rows=json.loads((ROOT/'bank/manifest.json').read_text())['episodes']
    row=next(r for r in rows if r['split']=='development' and r['context_per_class']==4 and r['case']=='uniform90' and r['label_mode']=='hetero')
    inp,extra,y=load_episode(row)
    xo=torch.as_tensor(inp.observed_values)
    observed=torch.as_tensor(inp.observed)
    tex=B.tensor_extra(extra,'cpu')
    front=B.Front(11)
    removed=transform_extra(tex,'node_nolabel')
    for key in ['descriptor','neighbor']:
        assert torch.equal(removed[key][...,:7],tex[key][...,:7])
        assert torch.count_nonzero(removed[key][...,7:])==0
    for key in ['pc','weights','ii','jj']:
        assert torch.equal(removed[key],tex[key])
    uniform=transform_extra(tex,'node_uniform')
    summed=torch.zeros_like(tex['pc']).index_add(0,tex['ii'],uniform['weights'])
    assert torch.allclose(summed,torch.ones_like(summed),atol=1e-6,rtol=0)
    assert torch.equal(uniform['pc'],tex['pc'])
    for key in ['descriptor','neighbor','affinity','node_affinity']:
        assert torch.equal(uniform[key],tex[key])
    gains=signed_gain(front,tex)
    assert float(gains.abs().max())<=1
    assert torch.equal(gains[tex['ii']==tex['jj']],torch.ones_like(gains[tex['ii']==tex['jj']]))
    edge_index={(int(i),int(j)):k for k,(i,j) in enumerate(zip(tex['ii'],tex['jj']))}
    reverse=torch.tensor([edge_index[(int(j),int(i))] for i,j in zip(tex['ii'],tex['jj'])])
    assert float((gains-gains[reverse]).abs().max())==0
    for arm in cfg()['evaluation_arms']:
        out,info=frontend(front,xo,observed,tex,arm)
        assert torch.isfinite(out).all()
        assert torch.equal(out[observed],xo[observed])
        assert info['spatial_steps']==100 and info['channel_corrections']==1
        no_source,_=frontend(front,torch.zeros_like(xo),torch.zeros_like(observed),tex,arm)
        assert torch.count_nonzero(no_source)==0
    _,shuffled,_=load_episode(row,shuffle=True)
    inp2,_,_=load_episode(row,shuffle=True)
    assert np.array_equal(inp.labels_context,inp2.labels_context)
    for key in ['pc','weights','ii','jj']:
        assert np.array_equal(extra[key],shuffled[key])
    k1=[r for r in rows if r['split']=='development' and r['context_per_class']==1]
    rename_error=max(r['geometry_shuffle_descriptor_max_abs'] for r in k1)
    assert rename_error<2e-5,rename_error
    smoke=[]
    for seed in cfg()['seeds']:
        old=json.loads((B.ROOT/'results'/f'node_s{seed}'/'training.json').read_text())
        for arm in cfg()['learned_arms']:
            obj=Model(arm,seed)
            initial=C.parameter_digest(obj.front)
            pfn_sha=C.parameter_digest(obj.pfn)
            assert initial==old['initial_front_sha256']
            assert pfn_sha==old['pfn_sha256']
            assert sum(p.numel() for p in obj.parameters() if p.requires_grad)==737
            obj.train()
            assert not obj.pfn.training and all(not p.requires_grad for p in obj.pfn.parameters())
            optimizer=torch.optim.AdamW(obj.front.parameters(),lr=.001,weight_decay=.01)
            with torch.autocast('cuda',dtype=torch.float16):
                loss=F.cross_entropy(obj(inp,extra),torch.as_tensor(y,device='cuda'))
            loss.backward()
            grad=sum(float(p.grad.abs().sum()) for p in obj.front.parameters() if p.grad is not None)
            assert grad>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.front.parameters())
            assert all(p.grad is None for p in obj.pfn.parameters())
            torch.nn.utils.clip_grad_norm_(obj.front.parameters(),1.)
            optimizer.step()
            assert C.parameter_digest(obj.front)!=initial
            assert C.parameter_digest(obj.pfn)==pfn_sha
            smoke.append({'arm':arm,'seed':seed,'discarded_updates':1,'gradient_abs_sum':grad,'pfn_unchanged':True})
            print('SMOKE',arm,seed,'PASS',flush=True)
            del obj,optimizer
            gc.collect()
            torch.cuda.empty_cache()
    immutable=[p for p in B.ROOT.rglob('*') if p.is_file() and 'reviewer_development' not in p.parts and 'runtime' not in p.parts]
    immutable += [WORK/f'nsd_token_pfn/results/pcfi100_s{s}/checkpoint.pt' for s in cfg()['seeds']]
    preserved={p.relative_to(WORK).as_posix():sha(p) for p in immutable}
    dump(ROOT/'results/old_files_baseline.json',preserved)
    dump(ROOT/'results/preflight.json',{'status':'PASS','smoke':smoke,'discarded_updates':len(smoke),
        'same_initial_front_states':True,'same_frozen_pfn':True,'uniform_rows_normalized':True,
        'confidence_retained_outside_spatial_weights':True,'geometry_label_channels_removed_only':True,
        'query_and_pfn_context_unchanged_by_geometry_shuffle':True,'k1_class_rename_descriptor_max_abs':rename_error,
        'signed_gain_bound_and_reverse_symmetry':True,'old_files_preserved_baseline':len(preserved)})
    dump(ROOT/'results/lock.json',lock())
    print('PREFLIGHT PASS',len(preserved),flush=True)


if __name__=='__main__':
    main()

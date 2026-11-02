import sys
sys.dont_write_bytecode=True
import sklearn
import gc,json,copy
import numpy as np
import torch
from dataclasses import replace
from core import ROOT,C,cfg,dump,sha,load_episode,load_model,geometry,correlation,Front
from data import read_real

def spatial(front,inp,ex,kind):
    xo=torch.as_tensor(inp.observed_values,device='cuda');obs=torch.as_tensor(inp.observed,device='cuda')
    h0=torch.as_tensor(ex['warm'],device='cuda');h1=torch.zeros_like(h0)
    ii=torch.as_tensor(ex['edge_i'],device='cuda');jj=torch.as_tensor(ex['edge_j'],device='cuda');w=torch.as_tensor(ex['weights'],device='cuda')
    if kind in ['pcfi100','pcfi200']:
        if kind=='pcfi200':
            for _ in range(100):h0=torch.where(obs,xo,torch.zeros_like(h0).index_add(0,ii,w*h0[jj]))
        return h0
    raw=front.net(torch.as_tensor(ex['descriptors'],device='cuda'))
    theta=np.pi*torch.tanh(raw[...,0]);gamma=torch.sigmoid(raw[...,1]);cs=theta.cos();sn=theta.sin()
    w=w*torch.exp(.5*torch.tanh(raw[...,2]));n,f=xo.shape;w=w/torch.zeros((n,f),device='cuda').index_add(0,ii,w).clamp_min(1e-12)[ii]
    a=1-gamma+gamma*cs;b=gamma*sn;d=1-gamma-gamma*cs
    if kind=='weighted':a=torch.ones_like(a);b=torch.zeros_like(b);d=a
    if kind=='signed':b=torch.zeros_like(b);d=a
    selfedge=(ii==jj)[:,None];a=torch.where(selfedge,1.,a);b=torch.where(selfedge,0.,b);d=torch.where(selfedge,1.,d)
    for _ in range(100):
        q0=w*(a*h0[jj]+b*h1[jj]);q1=w*(b*h0[jj]+d*h1[jj])
        h0=torch.where(obs,xo,torch.zeros_like(h0).index_add(0,ii,q0));h1=torch.where(obs,0.,torch.zeros_like(h1).index_add(0,ii,q1))
    return h0

def validate_chunking():
    row=next(r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='dev')
    inp,ex,_=load_episode(row);front=Front(11).to('cuda');checks={}
    with torch.no_grad():
        for kind in ['pcfi100','pcfi200','weighted','signed','sheaf']:
            a,_=front(inp,ex,kind);pieces=[]
            for start in range(0,inp.observed.shape[1],5):
                end=start+5;ix=copy.deepcopy(inp);ix.observed=inp.observed[:,start:end];ix.observed_values=inp.observed_values[:,start:end]
                xx={k:(v[:,start:end] if k in ['warm','confidence','weights','descriptors'] else v) for k,v in ex.items()}
                pieces.append(spatial(front,ix,xx,kind))
            z=correlation(torch.cat(pieces,1),torch.as_tensor(ex['confidence'],device='cuda'));err=float((a-z).abs().max());checks[kind]=err
            assert err<2e-5
    dump(ROOT/'results/native_chunking_audit.json',dict(status='PASS',errors=checks,meaning='Spatial propagation chunked; original inter-channel correction performed globally after concatenation, never separately per chunk.'))

def main():
    C.setup();torch.set_num_threads(2);validate_chunking()
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['repetition']==0 and r['case']!='full']
    output=[];out=ROOT/'results/native_inference';out.mkdir(exist_ok=True)
    for kind in ['pcfi200','weighted','signed','sheaf']:
        model=load_model(kind,11);before=C.parameter_digest(model)
        for name in cfg()['real_datasets']:
            g=read_real(name)
            for ri,row in enumerate(r for r in rows if r['graph']==name):
                inp,_,y=load_episode(row);m=inp.observed[:,0];raw=np.where(m[:,None],g['native'],0.).astype('float32');nf=raw.shape[1];parts=[];confidence=None
                with torch.no_grad():
                    for start in range(0,nf,128):
                        end=min(start+128,nf);xx=raw[:,start:end];obs=np.broadcast_to(m[:,None],xx.shape).copy()
                        ni=replace(inp,observed_values=xx,observed=obs)
                        ex=geometry(inp.src,inp.dst,obs,xx,inp.context,inp.labels_context,int(inp.classes))
                        confidence=ex['confidence'][:,0]
                        parts.append(spatial(model.front,ni,ex,kind))
                    numeric=correlation(torch.cat(parts,1),torch.as_tensor(confidence[:,None],device='cuda'))
                    projected=numeric@torch.as_tensor(g['rp'],device='cuda')
                    xp=raw@g['rp'];rms=np.sqrt((xp*xp).sum(0)/max(int(m.sum()),1));rms=np.where(rms>=1e-8,rms,1.)
                    projected=projected/torch.as_tensor(rms,device='cuda');projected=projected.clamp(-10,10)
                    projected=torch.where(torch.as_tensor(inp.observed,device='cuda'),torch.as_tensor(inp.observed_values,device='cuda'),projected)
                    model.eval()
                    with torch.autocast('cuda',dtype=torch.float16):p=C.Model.forward(model,replace(inp,pcfi=projected)).softmax(-1).cpu().numpy()
                missing=~inp.observed[inp.query].any(1);dest=out/f'{kind}_{name}_{ri}.npz';np.savez_compressed(dest,p=p,y=y,missing=missing)
                output.append({**row,'arm':kind,'seed':11,'prediction':dest.relative_to(ROOT).as_posix(),'prediction_sha256':sha(dest),
                    'missing':C.metrics(p[missing],y[missing]),'native_channels':nf,'spatial_block':128,
                    'scope':'Additional first-repetition native-feature inference diagnostic, fixed synthetic-trained parameters; global original inter-channel correction before RP16; not primary replacement.'})
                print('NATIVE',kind,name,row['case'],flush=True)
                dump(ROOT/'results/native_inference_partial.json',output)
        assert C.parameter_digest(model)==before
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/native_inference.json',output)

if __name__=='__main__':main()

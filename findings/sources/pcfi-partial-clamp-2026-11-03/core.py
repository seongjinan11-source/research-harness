import sys
sys.dont_write_bytecode = True
import sklearn  # Must precede torch on this Windows installation.
from pathlib import Path
from dataclasses import replace
import importlib.util
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('partial_clamp_previous', WORK/'pcfi_reviewer_ablation/core.py')
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
B, C, dump, sha = R.B, R.C, R.dump, R.sha


def cfg():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))


def load_episode(row, **unused):
    return R.load_episode(row)


def node_theta(front, extra):
    u = extra['descriptor']
    affinity = extra['node_affinity'][:,None,None].expand(-1,u.shape[1],1)
    return math.pi*torch.tanh(front.phi(torch.cat((u,extra['neighbor'],affinity),-1)).squeeze(-1))


def transport_forward(front, xo, observed, extra, *, partial, theta=None,
                      steps=100, return_state=False):
    ii,jj,w = extra['ii'],extra['jj'],extra['weights']
    theta = node_theta(front,extra) if theta is None else theta
    delta = torch.where((ii==jj)[:,None],0.,theta[jj]-theta[ii])
    cosine,sine = delta.cos(),delta.sin()
    h0,h1 = xo,torch.zeros_like(xo)
    for _ in range(steps):
        q0 = w*(cosine*h0[jj]-sine*h1[jj])
        q1 = w*(sine*h0[jj]+cosine*h1[jj])
        next0 = torch.where(observed,xo,torch.zeros_like(h0).index_add(0,ii,q0))
        next1 = torch.zeros_like(h1).index_add(0,ii,q1)
        if not partial:
            next1 = torch.where(observed,0.,next1)
        change = torch.maximum((next0-h0).abs().max(),(next1-h1).abs().max())
        h0,h1 = next0,next1
    out = torch.where(observed,xo,B.correlation(h0,extra['pc']))
    ns = ii!=jj
    diagnostics = {
        'negative_cos_fraction':float((cosine.detach()[ns]<0).float().mean()) if ns.any() else 0.,
        'second_coordinate_mean_abs':float(h1.detach().abs().mean()),
        'second_coordinate_max_abs':float(h1.detach().abs().max()),
        'observed_second_mean_abs':float(h1.detach()[observed].abs().mean()) if observed.any() else 0.,
        'spatial_first_max_abs':float(h0.detach().abs().max()),
        'output_max_abs':float(out.detach().abs().max()),
        'last_step_max_change':float(change.detach()),
        'observed_change':float((out.detach()[observed]-xo[observed]).abs().max()) if observed.any() else 0.,
        'spatial_steps':steps,'channel_corrections':1,'partial_clamp':int(partial),
    }
    if return_state:
        return out,diagnostics,(h0,h1)
    return out,diagnostics


def smooth_gain(front, extra):
    u,ii,jj = extra['descriptor'],extra['ii'],extra['jj']
    affinity = extra['affinity'][:,None,None].expand(-1,u.shape[1],1)
    a = front.phi(torch.cat((u[ii],u[jj],affinity),-1)).squeeze(-1)
    b = front.phi(torch.cat((u[jj],u[ii],affinity),-1)).squeeze(-1)
    gain = torch.tanh(math.atanh(.99)+.5*(a+b))
    return torch.where((ii==jj)[:,None],1.,gain)


def smooth_forward(front,xo,observed,extra):
    ii,jj,w = extra['ii'],extra['jj'],extra['weights']
    gain = smooth_gain(front,extra)
    h = xo
    for _ in range(100):
        next_h = torch.where(observed,xo,torch.zeros_like(h).index_add(0,ii,w*gain*h[jj]))
        change = (next_h-h).abs().max()
        h = next_h
    out = torch.where(observed,xo,B.correlation(h,extra['pc']))
    g = gain.detach()[ii!=jj]
    return out,{
        'negative_gain_fraction':float((g<0).float().mean()),
        'gain_mean':float(g.mean()),'gain_saturation_fraction':float((g.abs()>.999).float().mean()),
        'gain_derivative_mean':float((1-g.square()).mean()),
        'gain_min':float(g.min()),'gain_max':float(g.max()),
        'observed_change':float((out.detach()[observed]-xo[observed]).abs().max()) if observed.any() else 0.,
        'spatial_first_max_abs':float(h.detach().abs().max()),
        'output_max_abs':float(out.detach().abs().max()),
        'last_step_max_change':float(change.detach()),
        'spatial_steps':100,'channel_corrections':1,
    }


class Model(B.Model):
    def __init__(self,arm,seed):
        super().__init__('node',seed)
        self.arm,self.seed = arm,seed
        if arm not in cfg()['learned_arms']:
            for p in self.front.parameters():
                p.requires_grad_(False)

    def forward(self,inp,extra):
        with torch.autocast('cuda',enabled=False):
            xo = torch.as_tensor(inp.observed_values,device='cuda')
            observed = torch.as_tensor(inp.observed,device='cuda')
            ex = B.tensor_extra(extra) if isinstance(extra['pc'],np.ndarray) else extra
            if self.arm=='signed_smooth':
                values,info = smooth_forward(self.front,xo,observed,ex)
            else:
                theta = None
                if self.arm in ['full_random','partial_random']:
                    # Private RNG: PFN global RNG/forward_seed remains unchanged.
                    rng = np.random.default_rng(int(inp.forward_seed)+1000003*self.seed+20411103)
                    theta = torch.as_tensor(rng.uniform(-math.pi,math.pi,xo.shape),device=xo.device,dtype=xo.dtype)
                values,info = transport_forward(self.front,xo,observed,ex,
                    partial=self.arm!='full_random',theta=theta)
        self.last_diagnostics = info
        return self.pfn(replace(inp,pcfi=values))


def predict(obj,inp,extra):
    obj.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        p = obj(inp,extra).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all()
    return p


def lock():
    return {'sources':{p:sha(ROOT/p) for p in ['core.py','bank.py','checks.py','run.py','protocol.json']},
            'bank_manifest':sha(ROOT/'bank/manifest.json'),'previous':R.lock()}


def load_model(arm,seed):
    if arm in ['pcfi','node','signed']:
        return R.load_model(arm,seed)
    obj = Model(arm,seed)
    if arm in ['partial_initial','full_random','partial_random']:
        return obj.eval()
    folder = B.ROOT/'results'/f'node_s{seed}' if arm=='partial_frozen' else ROOT/'results'/f'{arm}_s{seed}'
    info = json.loads((folder/'training.json').read_text())
    assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    state = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['metadata']==(B.lock() if arm=='partial_frozen' else lock())
    obj.front.load_state_dict(state['front'])
    assert C.parameter_digest(obj.front)==info['final_front_sha256']
    assert C.parameter_digest(obj.pfn)==info['pfn_sha256']
    return obj.eval()

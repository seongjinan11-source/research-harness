import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
from dataclasses import replace
import copy
import importlib.util
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec = importlib.util.spec_from_file_location('review_pcfi_reference', WORK/'pcfi_orthogonal_pilot/core.py')
B = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = B
spec.loader.exec_module(B)
C, dump, sha = B.C, B.dump, B.sha


def cfg():
    return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))


def load_episode(row, *, shuffle=False):
    with np.load(WORK/row['bank_path']) as z:
        inp = C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
        y = z['labels_query'].astype('int64')
    key = 'shuffle_extra_path' if shuffle else 'extra_path'
    with np.load(WORK/row[key]) as z:
        extra = {k:z[k] for k in z.files}
    return inp, extra, y


def transform_extra(extra, arm):
    result = dict(extra)
    if arm == 'node_nolabel':
        for key in ['descriptor', 'neighbor']:
            result[key] = result[key].clone()
            result[key][..., 7:10] = 0
        result['affinity'] = torch.zeros_like(result['affinity'])
        result['node_affinity'] = torch.zeros_like(result['node_affinity'])
    if arm in ['uniform', 'node_uniform']:
        ii = result['ii']
        degree = torch.bincount(ii, minlength=len(result['pc'])).to(result['pc'].dtype)
        result['weights'] = (1/degree[ii])[:, None].expand_as(result['weights'])
    return result


def signed_gain(front, extra):
    u, ii, jj = extra['descriptor'], extra['ii'], extra['jj']
    affinity = extra['affinity'][:,None,None].expand(-1,u.shape[1],1)
    forward = front.phi(torch.cat((u[ii],u[jj],affinity),-1)).squeeze(-1)
    reverse = front.phi(torch.cat((u[jj],u[ii],affinity),-1)).squeeze(-1)
    gain = (.99 + .5*(forward+reverse)).clamp(-1,1)
    return torch.where((ii==jj)[:,None], 1., gain)


def signed_forward(front, xo, observed, extra):
    ii,jj,weight = extra['ii'],extra['jj'],extra['weights']
    gain = signed_gain(front, extra)
    h = xo
    for _ in range(100):
        h = torch.where(observed, xo,
                        torch.zeros_like(h).index_add(0,ii,weight*gain*h[jj]))
    out = torch.where(observed, xo, B.correlation(h,extra['pc']))
    ns = ii != jj
    return out, {'negative_gain_fraction':float((gain.detach()[ns]<0).float().mean()),
                 'gain_mean':float(gain.detach()[ns].mean()),
                 'gain_clip_positive_fraction':float((gain.detach()[ns]==1).float().mean()),
                 'gain_abs_max':float(gain.detach().abs().max()),
                 'observed_change':float((out.detach()[observed]-xo[observed]).abs().max()) if observed.any() else 0.,
                 'spatial_steps':100,'channel_corrections':1}


def frontend(front, xo, observed, extra, arm):
    extra = transform_extra(extra, arm)
    if arm == 'signed':
        return signed_forward(front, xo, observed, extra)
    kind = 'pcfi' if arm in ['pcfi','uniform'] else 'node'
    return front(xo,observed,extra,kind)


class Model(B.Model):
    def __init__(self, arm, seed):
        super().__init__('node',seed)
        self.arm = arm

    def forward(self, inp, extra):
        with torch.autocast('cuda',enabled=False):
            xo = torch.as_tensor(inp.observed_values,device='cuda')
            observed = torch.as_tensor(inp.observed,device='cuda')
            tex = B.tensor_extra(extra) if isinstance(extra['pc'],np.ndarray) else extra
            values,info = frontend(self.front,xo,observed,tex,self.arm)
        self.last_diagnostics = info
        return self.pfn(replace(inp,pcfi=values))


def predict(obj, inp, extra):
    obj.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        probabilities = obj(inp,extra).softmax(-1).cpu().numpy()
    assert np.isfinite(probabilities).all()
    return probabilities


def lock():
    sources = ['core.py','bank.py','checks.py','run.py','protocol.json']
    return {'sources':{p:sha(ROOT/p) for p in sources},
            'bank_manifest':sha(ROOT/'bank/manifest.json'), 'previous':B.lock()}


def load_model(arm, seed):
    obj = Model(arm,seed)
    if arm in ['pcfi','uniform']:
        return obj.eval()
    old = arm in ['node','node_geometry_shuffle']
    folder = B.ROOT/'results'/f'node_s{seed}' if old else ROOT/'results'/f'{arm}_s{seed}'
    info = json.loads((folder/'training.json').read_text())
    assert sha(folder/'checkpoint.pt') == info['checkpoint_sha256']
    state = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['metadata'] == (B.lock() if old else lock())
    obj.front.load_state_dict(state['front'])
    assert C.parameter_digest(obj.front) == info['final_front_sha256']
    assert C.parameter_digest(obj.pfn) == info['pfn_sha256']
    return obj.eval()

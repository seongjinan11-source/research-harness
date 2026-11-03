import sys
sys.dont_write_bytecode=True
import sklearn
from pathlib import Path
import importlib.util,json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
spec=importlib.util.spec_from_file_location('node_frame_reference',WORK/'nsd_token_pfn/core.py')
N=importlib.util.module_from_spec(spec);sys.modules[spec.name]=N;spec.loader.exec_module(N)
C=N.C;dump=N.dump;sha=N.sha;predict=N.predict;rotation=N.rotation;laplacian=N.laplacian

def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))

class NodeFrames(N.TokenNSD):
    def maps(self,h,ii,jj,meta):
        summary=torch.cat((h[:,:-1].mean(1),h[:,-1]),-1)
        u=torch.cat((self.project(summary),meta),-1)
        total=u.clone().index_add(0,ii,u[jj]).index_add(0,jj,u[ii])
        count=u.new_ones(len(u)).index_add(0,ii,u.new_ones(len(ii))).index_add(0,jj,u.new_ones(len(jj)))
        # One frame per node, with the same 72->32->1 Phi and parameter count.
        neighbor=total/count[:,None]
        q=rotation(np.pi*torch.tanh(self.phi(torch.cat((u,neighbor),-1)).squeeze(-1)))
        return q[ii],q[jj]

def model(seed):
    result=N.Model('dynamic',seed)
    result.nsd=NodeFrames(seed).to('cuda')
    return result

def load_episode(row):
    z=np.load(WORK/row['bank_path'])
    return C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__}),z['labels_query'].astype('int64')

def lock():
    return {'sources':{p.name:sha(p) for p in [ROOT/'core.py',ROOT/'bank.py',ROOT/'run.py',ROOT/'protocol.json']},
            'original_lock':N.lock(),'manifest_sha256':sha(ROOT/'bank/manifest.json')}

def load_model(seed):
    folder=ROOT/'results'/f'node_frame_s{seed}';info=json.loads((folder/'training.json').read_text())
    assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
    result=model(seed);st=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert st['metadata']==lock()
    with torch.no_grad():
        for name,p in result.named_parameters():
            if p.requires_grad:p.copy_(st['state_dict'][name].to(p.device))
    assert C.parameter_digest(result,trainable=False)==info['frozen_sha256']
    assert C.parameter_digest(result,trainable=True)==info['final_sha256']
    return result.eval()

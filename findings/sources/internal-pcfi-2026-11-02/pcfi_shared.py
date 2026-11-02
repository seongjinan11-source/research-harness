"""Explicit module names prevent collisions with historical common.py/run.py."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
spec=importlib.util.spec_from_file_location('literal_pcfi_old_common', WORK/'joint_imputation_pfn/common.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
from imputation import adjacency, distances, impute

def cfg():return json.loads((ROOT/'protocol.json').read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def rows(split):
    path=ROOT/'bank/manifest.json'
    if not path.exists():path=ROOT/'bank/development.json'
    return [r for r in json.loads(path.read_text(encoding='utf8'))['episodes'] if r['split']==split]
def read(row):
    with np.load(ROOT/'bank'/row['path']) as z:
        inp=C.Inputs(**{k:z[k] for k in C.Inputs.__annotations__})
        extra={k:z[k] for k in ['confidence','edge_i','edge_j']}
        return inp,extra,z['labels_query'].astype(np.int64)
def prepare_geometry(inp,physical_features):
    a=adjacency((inp.src,inp.dst),len(inp.observed))
    # Per-column distances also handle mixed cell masks; whole-node columns coincide.
    dist=distances(a,inp.observed[:,:physical_features],False)
    confidence=np.power(np.float32(.9),dist)
    if physical_features%2:confidence=np.pad(confidence,((0,0),(0,1)),constant_values=1)
    i,j=a.nonzero()
    return {'confidence':confidence.astype(np.float32),'edge_i':i.astype(np.int64),'edge_j':j.astype(np.int64)}
def sources():
    paths=list(ROOT.glob('*.py'))+[ROOT/'protocol.json']
    paths += [WORK/p for p in ['joint_imputation_pfn/common.py','regret_prior_pilot/core.py','fisf_graphpfn/imputation.py']]
    return {str(p.relative_to(WORK)):sha(p) for p in paths}
def design():
    d=json.loads((ROOT/'results/design.json').read_text(encoding='utf8'))
    assert d['sources']==sources(),'Source/config changed after design lock'
    assert d['development_manifest']==sha(ROOT/'bank/development.json')
    return d
def lock():
    from mgpfn import upstream as U
    design()
    value={'sources':sources(),'manifest':sha(ROOT/'bank/manifest.json'),'design':sha(ROOT/'results/design.json'),
           'public':{p.name:sha(p) for p in [U.ADAPTERS,U.LIMIX]},'upstream_commit':U.COMMIT}
    path=ROOT/'results/lock.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==value
    else:dump(path,value)
    return value

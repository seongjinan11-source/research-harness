"""FP05 shared bookkeeping; imports never create data or start a GPU job."""
import sys
sys.dont_write_bytecode = True
import sklearn
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PYTHON = 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe'
MG = Path('C:/Users/seongjin/Desktop/code/mgpfn/src')
if str(MG) not in sys.path:
    sys.path.insert(0, str(MG))


def cfg():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def setup():
    from mgpfn import upstream as U
    U.import_graphpfn()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return U


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def digest(model, trainable=None):
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if trainable is not None and param.requires_grad != trainable:
            continue
        h.update(name.encode()); h.update(str(param.dtype).encode())
        h.update(str(tuple(param.shape)).encode())
        h.update(param.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def metrics(p, y):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    p = np.clip(np.asarray(p, np.float64), 1e-8, 1.)
    p /= p.sum(1, keepdims=True)
    y = np.asarray(y, np.int64); pred = p.argmax(-1)
    return dict(accuracy=float((pred == y).mean()),
                nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float(np.square(p - np.eye(p.shape[1])[y]).sum(-1).mean()),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=np.arange(p.shape[1]), average='macro', zero_division=0)))


def sources():
    U = setup()
    paths = list(ROOT.glob('rl_*.py')) + [ROOT / 'protocol.json', ROOT / 'PLAN_v2.md']
    paths += [MG / 'mgpfn/upstream.py', MG / 'mgpfn/prior_adapter.py',
              WORK / 'fisf_graphpfn/imputation.py', WORK / 'observed_cell_cycle/data.py',
              U.SRC / 'src/graphpfn/model/light.py', U.SRC / 'src/graphpfn/model/layers.py']
    paths += sorted((U.SRC / 'src/graphpfn').rglob('*.py'))
    paths += sorted((U.SRC / 'paper/lib/graphpfn/prior').rglob('*.py'))
    paths += [U.SRC / 'paper/exp/graphpfn/pretrain/main/pretrain.toml',
              U.E1 / 'graphpfn_windows_dev.py', MG / 'mgpfn/missing.py']
    return {str(p.resolve()): sha(p) for p in sorted(set(paths))}


def design(create=False):
    U = setup()
    state = dict(sources=sources(), upstream_commit=U.COMMIT,
                 checkpoints={str(p): sha(p) for p in [U.ADAPTERS, U.LIMIX]},
                 development={str(ROOT / 'bank' / split / 'manifest.json'): sha(ROOT / 'bank' / split / 'manifest.json')
                              for split in ['train', 'dev']},
                 checks_sha256=sha(ROOT / 'results/checks.json'))
    path = ROOT / 'results/design.json'
    if not path.exists():
        assert create
        assert json.loads((ROOT / 'results/checks.json').read_text(encoding='utf8'))['status'] == 'PASS'
        dump(path, state)
    assert json.loads(path.read_text(encoding='utf8')) == state, 'Pre-final design changed'
    return state


def lock(create=False):
    U = setup()
    design()
    state = dict(sources=sources(), upstream_commit=U.COMMIT,
                 checkpoints={str(p): sha(p) for p in [U.ADAPTERS, U.LIMIX]},
                 bank={str(ROOT / 'bank' / split / 'manifest.json'): sha(ROOT / 'bank' / split / 'manifest.json')
                       for split in ['train', 'dev', 'final', 'real']},
                 checks_sha256=sha(ROOT / 'results/checks.json'))
    path = ROOT / 'results/lock.json'
    if not path.exists():
        assert create
        assert json.loads((ROOT / 'results/checks.json').read_text(encoding='utf8'))['status'] == 'PASS'
        dump(path, state)
    assert json.loads(path.read_text(encoding='utf8')) == state, 'Scientific source/bank/checkpoint changed'
    return state

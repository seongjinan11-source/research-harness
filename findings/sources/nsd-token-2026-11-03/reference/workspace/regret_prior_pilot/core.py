"""Shared inputs and unchanged GraphPFN wrapper. No query labels in Inputs."""
import sys
sys.dont_write_bytecode = True
import sklearn
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, r'C:\Users\seongjin\Desktop\code\mgpfn\src')
sys.path.insert(0, str(WORK / 'fisf_graphpfn'))


def config():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    from mgpfn import upstream as U
    U.import_graphpfn()
    return U


def adjacency(src, dst, n):
    a = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    a = a.maximum(a.T)
    a.setdiag(1)
    a.data[:] = 1
    a.sort_indices()
    return a


def mask(task, mechanism, rate, seed):
    x = task['X_full']
    rng = np.random.default_rng(seed)
    if mechanism == 'full':
        return np.ones_like(x, bool)
    if mechanism == 'uniform_cells':
        return rng.random(x.shape) >= rate
    if mechanism == 'uniform_nodes':
        rows = rng.random(len(x)) >= rate
    else:
        assert mechanism == 'clustered_nodes'
        a = adjacency(task['src'], task['dst'], len(x))
        root = int(rng.integers(len(x)))
        dist = dijkstra(a, indices=root, unweighted=True, directed=False)
        order = np.lexsort((rng.random(len(x)), dist))
        rows = np.zeros(len(x), bool)
        rows[order[:int(round(len(x) * (1 - rate)))]] = True
    return np.broadcast_to(rows[:, None], x.shape).copy()


def observed_input(task, observed):
    x = task['X_full']
    count = observed.sum(0)
    mean = np.divide(np.where(observed, x, 0).sum(0), count,
                     out=np.zeros(x.shape[1]), where=count > 0)
    var = np.divide(np.where(observed, (x - mean)**2, 0).sum(0), count,
                    out=np.ones(x.shape[1]), where=count > 0)
    scale = np.where(var >= 1e-8, np.sqrt(var), 1.)
    return np.where(observed, np.clip((x - mean) / scale, -10, 10), np.nan).astype(np.float32)


def fill(x, src, dst, method, mechanism):
    from imputation import impute
    a = adjacency(src, dst, len(x))
    mode = 'uniform' if mechanism == 'uniform_cells' else 'structural'
    return impute(a, x, method, mode, iterations=100)[0]


@dataclass
class Inputs:
    values: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    context: np.ndarray
    labels_context: np.ndarray
    classes: int
    forward_seed: int


def load_episode(row, kind='pcfi'):
    d = np.load(ROOT / 'bank' / row['path'])
    n = len(d['values'])
    context = np.arange(n) < int(d['n_train'])
    values = d['values'] if kind == 'pcfi' else fill(
        d['observed_values'], d['src'], d['dst'], kind, row['mechanism'])
    inputs = Inputs(values, d['src'], d['dst'], context,
                    d['labels_context'].astype(np.int64), int(d['classes']), row['forward_seed'])
    return inputs, d['labels_query'].astype(np.int64)


def load_model():
    from mgpfn import upstream as U
    from mgpfn.train import trainable_policy
    torch.manual_seed(config()['initialization_seed'])
    np.random.seed(config()['initialization_seed'])
    model, keys = U.load_light('cuda', always_checkpoint=True)
    groups = trainable_policy(model, keys)
    return model, groups


def forward(model, inputs):
    import dgl
    torch.manual_seed(inputs.forward_seed)
    np.random.seed(inputs.forward_seed)
    graph = dgl.graph((torch.as_tensor(inputs.src), torch.as_tensor(inputs.dst)),
                      num_nodes=len(inputs.context)).to('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(graph=graph, features=torch.as_tensor(inputs.values, device='cuda'),
            y_train=torch.as_tensor(inputs.labels_context, device='cuda', dtype=torch.float32),
            train_mask=torch.as_tensor(inputs.context, device='cuda'),
            task_type='multiclass', n_random_features=0)
    return logits[torch.as_tensor(~inputs.context, device='cuda'), :inputs.classes].float()


def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        p = forward(model, inputs).softmax(-1).cpu().numpy()
    assert np.isfinite(p).all()
    return p


def lp(inputs):
    a = adjacency(inputs.src, inputs.dst, len(inputs.context))
    t = sp.diags(1 / np.asarray(a.sum(1)).ravel()) @ a
    y = np.eye(inputs.classes)[inputs.labels_context]
    prior = (y.sum(0) + 1) / (len(y) + inputs.classes)
    f = np.broadcast_to(prior, (len(inputs.context), inputs.classes)).copy()
    f[inputs.context] = y
    for _ in range(40):
        f = .9 * (t @ f) + .1 * prior
        f[inputs.context] = y
    p = f[~inputs.context]
    p = .99 * p / p.sum(1, keepdims=True) + .01 * prior
    return p


def metrics(p, y):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    p = np.clip(np.asarray(p, np.float64), 1e-8, 1)
    p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)
    return {'accuracy': float(np.mean(pred == y)),
            'nll': float(-np.log(p[np.arange(len(y)), y]).mean()),
            'brier': float(np.mean(np.sum((p - np.eye(p.shape[1])[y])**2, axis=1))),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0))}


def digest_model(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_manifest():
    from mgpfn import upstream as U
    local = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    local['protocol.json'] = sha(ROOT / 'protocol.json')
    return {'local': local, 'upstream_commit': U.COMMIT,
            'adapter_sha256': sha(U.ADAPTERS), 'limix_sha256': sha(U.LIMIX),
            'imputation_sha256': sha(WORK / 'fisf_graphpfn/imputation.py')}

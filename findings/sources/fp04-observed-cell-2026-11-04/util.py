"""FP04 immutable files, metrics and source provenance; no model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def config():
    return json.loads((ROOT / 'protocol.json').read_text(encoding='utf8'))


def sources():
    files = list(ROOT.glob('*.py')) + list((ROOT/'source_snapshots').glob('*.py')) + [ROOT / 'protocol.json', ROOT / 'PLAN_v2.md',
                                      ROOT.parent / 'fisf_graphpfn/imputation.py']
    return {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sorted(files)}


def metrics(p, y):
    p = np.asarray(p, np.float64)
    y = np.asarray(y, np.int64)
    assert p.ndim == 2 and len(p) == len(y) and np.isfinite(p).all()
    assert p.min() >= -1e-7 and np.max(np.abs(p.sum(1) - 1)) < 1e-4
    p = np.clip(p, 1e-8, 1.)
    p /= p.sum(1, keepdims=True)
    k = p.shape[1]
    pred = p.argmax(1)
    support = np.bincount(y, minlength=k)
    correct = np.bincount(y[pred == y], minlength=k)
    predicted = np.bincount(pred, minlength=k)
    recall = np.divide(correct, support, out=np.zeros(k), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(k), where=(support + predicted) > 0)
    return dict(accuracy=float((pred == y).mean()),
                balanced_accuracy=float(recall[support > 0].mean()),
                macro_f1=float(f1.mean()), nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
                brier=float(((p - np.eye(k)[y]) ** 2).sum(1).mean()))


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()

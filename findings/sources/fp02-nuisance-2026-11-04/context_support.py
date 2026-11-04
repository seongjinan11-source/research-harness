"""Read-only context-only support diagnostic; no query labels are accessed."""
import sklearn
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'bank/dev_manifest.json').read_text())
out = []
for case in manifest['rows']:
    if case['row']['mode'] != 'none':
        continue
    with np.load(case['bank_path'], allow_pickle=False) as z:
        ids, yc = z['context_ids'], z['labels_context']
        src, dst = z['src'], z['dst']
        labels = np.full(case['nodes'], -1)
        labels[ids] = yc
        keep = (src < dst) & (labels[src] >= 0) & (labels[dst] >= 0)
        a, b = labels[src[keep]], labels[dst[keep]]
        out.append(dict(graph=case['row']['graph'], context=len(ids),
                        labeled_edges=len(a), same=int((a == b).sum()),
                        different=int((a != b).sum()),
                        edge_homophily=float((a == b).mean()) if len(a) else None))
print(json.dumps(dict(status='READ_ONLY', query_labels_accessed=False,
                      input_manifest_sha256=__import__('hashlib').sha256((ROOT / 'bank/dev_manifest.json').read_bytes()).hexdigest(),
                      rows=out), indent=2))

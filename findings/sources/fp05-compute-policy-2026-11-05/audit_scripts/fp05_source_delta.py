"""Audit preparation receipts against the later design lock; never rewrite them."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent / 'pfn_rl_cycle'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifests = {}
for split in ['train', 'dev', 'final', 'real']:
    path = ROOT / 'bank' / split / 'manifest.json'
    obj = json.loads(path.read_text(encoding='utf8'))
    mismatches = []
    for name, recorded in obj['dependencies'].items():
        actual = sha(Path(name))
        if actual != recorded:
            mismatches.append(dict(path=name, recorded_sha256=recorded, current_sha256=actual))
    assert all(Path(row['path']).name == 'rl_util.py' for row in mismatches), mismatches
    manifests[split] = dict(path=str(path), sha256=sha(path), dependency_mismatches=mismatches)

result = dict(status='DOCUMENTED_PRELOCK_DELTA', manifests=manifests,
              current_util_sha256=sha(ROOT / 'rl_util.py'),
              scope='Train/dev/real were prepared before the expanded design/source bookkeeping. Final was prepared under the design lock. Data implementation and every other recorded dependency match the current lock.',
              limitation='The exact earlier rl_util.py bytes were not snapshotted. Removing the known source-enumeration addition did not reconstruct its recorded hash, so a byte-exact claim that this was the only change is unsupported. No old source snapshot is fabricated.',
              mitigation='Preserve original preparation receipts and banks. Independent CPU audits recompute graph/mask/query/context/input identities and real projected preprocessing. Scientific training and final preparation use the explicit current design/full lock.',
              action='No input, source, manifest, checkpoint or prediction modified by this audit.')
out = ROOT / 'PRELOCK_SOURCE_DELTA.json'
encoded = (json.dumps(result, indent=2) + '\n').encode()
if out.exists():
    assert out.read_bytes() == encoded
else:
    out.write_bytes(encoded)
print(json.dumps(result, indent=2))

"""Hash the training/calibration receipts omitted by the main prediction manifest."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent / 'pfn_rl_cycle'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


cfg = json.loads((ROOT / 'protocol.json').read_text())
for name in ['audit.json', 'deployment_audit.json', 'statistics_audit.json']:
    assert json.loads((ROOT / 'results' / name).read_text())['status'] == 'PASS'
paths = []
for seed in cfg['seeds']:
    for arm in cfg['arms']:
        folder = ROOT / 'results' / f'{arm}_s{seed}'
        paths += [folder / name for name in ['training.json', 'schedule.json', 'progress.jsonl',
                  'checkpoint.pt', 'final_evaluation.json', 'real_evaluation.json']]
        if arm in ['uniform', 'group_pg']:
            paths.append(folder / 'global_trace.json')
paths += [ROOT / 'results' / name for name in ['final_references.json', 'real_references.json',
          'artifact_manifest.json', 'audit.json', 'deployment_audit.json', 'statistics_audit.json',
          'summary.json', 'budget.json', 'routing_diagnostic.json', 'lock.json', 'design.json', 'checks.json',
          'checks_prelock001.json', 'preflight.json']]
paths += [ROOT / 'bank' / split / 'manifest.json' for split in ['train', 'dev', 'final', 'real']]
paths += [ROOT / name for name in ['training.log', 'evaluation.log', 'final_prepare.log',
          'PRELOCK_SOURCE_DELTA.json']]
for p in paths:
    assert p.is_file(), p
result = dict(status='PASS', scope='Supplementary receipt manifest; prediction NPZ hashes are in artifact_manifest.json, calibration NPZ hashes in each global_trace.json, bank input hashes in each bank manifest. Large evidence stays local and untracked.',
              files={p.relative_to(ROOT).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
                     for p in sorted(set(paths))},
              full_lock_sha256=sha(ROOT / 'results/lock.json'),
              source_limit='Earlier preparation utility bytes remain unavailable; PRELOCK_SOURCE_DELTA.json is retained, not repaired or rewritten.')
out = ROOT / 'results/provenance_manifest.json'
with out.open('x', encoding='utf8', newline='\n') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status='PASS', files=len(result['files']))))

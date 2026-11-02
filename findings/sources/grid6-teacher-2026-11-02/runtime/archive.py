"""Verbatim experiment snapshots; omit large checkpoints/raw probability banks."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parents[1]
work = root.parent
assert (root / 'results/completed.json').exists()
target = work / 'research-harness/findings/sources/grid6-teacher-2026-11-02'
files = [root / n for n in ['README.md', 'METHOD.md', 'REPORT_KO.md', 'REVIEW_KO.md', 'protocol.json', 'pilot.py', 'report.py', 'cost_audit.py']]
files += [p for p in sorted((root / 'results').glob('*.json')) if p.name != 'metrics.json']
for folder in sorted((root / 'results').glob('*_s*')):
    if folder.is_dir():
        files += [folder / n for n in ['training.json', 'train_log.json', 'overflow_retries.json']]
files += [root / 'teachers/manifest.json', root / 'bank/manifest.json', root / 'references/manifest.json']
files += [root / 'runtime/finish.py', root / 'runtime/archive.py']
records = []
for source in files:
    destination = target / source.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    blob = source.read_bytes()
    assert destination.read_bytes() == blob
    records.append({'alias': source.relative_to(root).as_posix(), 'original': str(source.resolve()),
                    'snapshot': destination.relative_to(work / 'research-harness').as_posix(),
                    'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)})
(target / 'manifest.json').write_text(json.dumps({'date': '2026-11-02', 'sources': records,
    'scope': 'Exact file copies. Checkpoints and raw train/test/teacher/probability NPZ banks remain in workspace; manifests, checkpoint digests, source locks and replay audits are included. Historical dependencies are preserved in the preceding disjoint-factor archive.'}, ensure_ascii=False, indent=2), encoding='utf8')
print('ARCHIVED', len(records), 'files', sum(r['bytes'] for r in records), 'bytes')

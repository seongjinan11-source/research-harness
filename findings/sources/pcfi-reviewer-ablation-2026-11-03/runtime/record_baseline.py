import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent/'research-harness'
out=ROOT/'runtime/harness_before.json'
assert not out.exists()
digest=lambda data:hashlib.sha256(data).hexdigest()
entries={}
for path in (REPO/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    for entry in yaml.safe_load(path.read_text(encoding='utf8'))['entries']:
        entries[entry['id']]=digest(json.dumps(entry,sort_keys=True,ensure_ascii=False).encode())
files={p.relative_to(REPO).as_posix():digest(p.read_bytes()) for directory in ['papers','findings/sources'] for p in (REPO/directory).rglob('*') if p.is_file()}
out.write_text(json.dumps({'entries':entries,'files':files},indent=2),encoding='utf8')
print('HARNESS BASELINE',len(entries),len(files))

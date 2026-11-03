import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent/'research-harness'
digest = lambda data:hashlib.sha256(data).hexdigest()


def snapshot():
    entries = {}
    for path in (REPO/'findings').glob('*.yaml'):
        if path.name.startswith('_'):
            continue
        for entry in yaml.safe_load(path.read_text(encoding='utf8'))['entries']:
            entries[entry['id']] = digest(json.dumps(entry,sort_keys=True,ensure_ascii=False).encode())
    files = {}
    for folder in [REPO/'papers',REPO/'findings/sources']:
        for path in folder.rglob('*'):
            if path.is_file():
                files[path.relative_to(REPO).as_posix()] = digest(path.read_bytes())
    return {'entries':entries,'files':files}


def main():
    target = ROOT/'runtime/harness_before.json'
    if sys.argv[1]=='before':
        assert not target.exists()
        data = snapshot()
        target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        print('HARNESS BASELINE',len(data['entries']),len(data['files']))
    else:
        before = json.loads(target.read_text())
        after = snapshot()
        assert all(after['entries'][k]==v for k,v in before['entries'].items())
        assert all(after['files'][k]==v for k,v in before['files'].items())
        print('HARNESS PRESERVED',len(before['entries']),len(before['files']))


if __name__=='__main__':
    main()

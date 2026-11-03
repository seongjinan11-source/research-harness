import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import re
import subprocess
import unittest
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
from preserve_harness import snapshot

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent/'research-harness'
DEST = REPO/'findings/sources/pcfi-partial-clamp-2026-11-03'
digest = lambda data:hashlib.sha256(data).hexdigest()


def tests():
    sys.path.insert(0,str(REPO/'src'))
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(REPO/'tests')))
    assert result.wasSuccessful()


def verify():
    before = json.loads((ROOT/'runtime/harness_before.json').read_text())
    after = snapshot()
    assert all(after['entries'][k]==v for k,v in before['entries'].items())
    assert all(after['files'][k]==v for k,v in before['files'].items())
    manifest = json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text())
    assert all(digest((DEST/p).read_bytes())==v for p,v in manifest.items())
    import openpyxl
    book = openpyxl.load_workbook(REPO/'out/research_harness.xlsx',read_only=True,data_only=True)
    paper_count = sum(1 for p in (REPO/'papers').glob('*.yaml') if not p.name.startswith('_'))
    idea_count = sum(1 for p in (REPO/'ideas').glob('*.yaml') if not p.name.startswith('_'))
    assert book['My Findings'].max_row==len(after['entries'])+1
    assert book['Papers'].max_row==paper_count+1
    assert book['Ideas'].max_row==idea_count+1
    tables,links = 0,0
    for path in [ROOT/'METHOD_KO.md',ROOT/'REPORT_KO.md']:
        content = path.read_text(encoding='utf8')
        rows = []
        for line in content.splitlines()+['']:
            if line.startswith('|'):
                rows.append(line)
            elif rows:
                assert len(rows)>=2 and len({r.count('|') for r in rows})==1
                tables += 1
                rows = []
        for target in re.findall(r'\[[^\]\n]+\]\(([^\n)]+)\)',content):
            if '://' not in target:
                assert (path.parent/target).is_file(),target
                links += 1
    result = {'status':'PASS','old_findings_preserved':len(before['entries']),
        'old_paper_source_files_preserved':len(before['files']),'snapshot_files_verified':len(manifest),
        'findings':len(after['entries']),'papers':paper_count,'ideas':idea_count,
        'markdown_tables_verified':tables,'local_links_verified':links,'workbook_sheets':book.sheetnames}
    (ROOT/'runtime/final_harness_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


def index():
    manifest = json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text())
    for path,value in manifest.items():
        content = subprocess.check_output(['git','-c','core.excludesFile=.git/info/exclude',
            'show',':'+(DEST/path).relative_to(REPO).as_posix()],cwd=REPO)
        assert digest(content)==value,path
    print('STAGED SOURCE HASHES VERIFIED',len(manifest))


if __name__=='__main__':
    {'tests':tests,'verify':verify,'index':index}[sys.argv[1]]()

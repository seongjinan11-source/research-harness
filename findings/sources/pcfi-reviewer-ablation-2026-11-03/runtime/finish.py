import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,subprocess,unittest
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent/'research-harness'
DEST=REPO/'findings/sources/pcfi-reviewer-ablation-2026-11-03'
digest=lambda data:hashlib.sha256(data).hexdigest()


def tests():
    sys.path.insert(0,str(REPO/'src'))
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(REPO/'tests')))
    assert result.wasSuccessful()


def verify():
    before=json.loads((ROOT/'runtime/harness_before.json').read_text())
    entries={}
    for path in (REPO/'findings').glob('*.yaml'):
        if path.name.startswith('_'):continue
        for entry in yaml.safe_load(path.read_text(encoding='utf8'))['entries']:
            entries[entry['id']]=digest(json.dumps(entry,sort_keys=True,ensure_ascii=False).encode())
    assert all(entries[key]==value for key,value in before['entries'].items())
    assert all(digest((REPO/path).read_bytes())==value for path,value in before['files'].items())
    snapshot=json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text())
    for path,value in snapshot.items():
        assert digest((DEST/path).read_bytes())==value,path
    import openpyxl
    book=openpyxl.load_workbook(REPO/'out/research_harness.xlsx',read_only=True,data_only=True)
    assert book['My Findings'].max_row==len(entries)+1
    assert book['Papers'].max_row==58
    assert book['Ideas'].max_row==17
    result={'status':'PASS','findings':len(entries),'snapshot_files':len(snapshot),
            'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files']),
            'workbook_sheets':book.sheetnames}
    (ROOT/'runtime/final_harness_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


def index():
    for path,value in json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text()).items():
        staged=subprocess.check_output(['git','-c','core.excludesFile=.git/info/exclude','show',':'+(DEST/path).relative_to(REPO).as_posix()],cwd=REPO)
        assert digest(staged)==value,path
    print('STAGED HASHES VERIFIED')


if __name__=='__main__':
    {'tests':tests,'verify':verify,'index':index}[sys.argv[1]]()

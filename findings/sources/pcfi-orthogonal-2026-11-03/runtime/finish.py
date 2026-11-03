import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,unittest,subprocess
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness';DEST=REPO/'findings/sources/pcfi-orthogonal-2026-11-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def tests():
    sys.path.insert(0,str(REPO/'src'));r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(REPO/'tests')));assert r.wasSuccessful()
def verify():
    before=json.loads((ROOT/'runtime/harness_before.json').read_text());entries={}
    for p in (REPO/'findings').glob('*.yaml'):
        if p.name.startswith('_'):continue
        for r in yaml.safe_load(p.read_text(encoding='utf8'))['entries']:entries[r['id']]=sha(json.dumps(r,sort_keys=True,ensure_ascii=False).encode())
    assert all(entries[k]==v for k,v in before['entries'].items())
    assert all(sha((REPO/p).read_bytes())==v for p,v in before['files'].items())
    manifest=json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text())
    for p,d in manifest.items():assert sha((DEST/p).read_bytes())==d,p
    import openpyxl
    w=openpyxl.load_workbook(REPO/'out/research_harness.xlsx',read_only=True,data_only=True)
    assert w['My Findings'].max_row==len(entries)+1 and w['Papers'].max_row==58 and w['Ideas'].max_row==16
    result={'status':'PASS','findings':len(entries),'snapshot_files':len(manifest),'old_entries_preserved':len(before['entries']),'old_paper_source_files_preserved':len(before['files']),'workbook_sheets':w.sheetnames}
    (ROOT/'runtime/final_harness_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False))
def index():
    for p,d in json.loads((DEST/'SNAPSHOT_MANIFEST.json').read_text()).items():
        b=subprocess.check_output(['git','-c','core.excludesFile=.git/info/exclude','show',':'+(DEST/p).relative_to(REPO).as_posix()],cwd=REPO);assert sha(b)==d,p
    print('STAGED HASHES VERIFIED')
if __name__=='__main__':{'tests':tests,'verify':verify,'index':index}[sys.argv[1]]()

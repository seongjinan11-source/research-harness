import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil,runpy,unittest
sys.path.append('C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'research-harness';DEST=REPO/'findings/sources/sheaf-pcfi-2026-11-03'
sha=lambda b:hashlib.sha256(b).hexdigest()
def revise():
    changes={
      '음수허용함수족이지만test의음수transport비율은0이었고cross-stalk만사용했다.':'첫좌표직접전달계수a<0비율은0이고cross-stalk사용은관측됐다. 행렬의다른성분·고유값·반복경로의signed효과는이통계로판정할수없다.',
      ';negativefraction0.':';first_diagonal_a_negative_fraction0;other_entries/eigenvalues_not_claimed.',
      'Synthetictest음수transport비율은0이다.':'Synthetictest의첫좌표직접전달계수a<0비율은0이며2D행렬전체의부호와구분한다.',
      'Synthetictest의음수transport비율은0이므로음의geometry를학습한성공/실패로과대해석하지않는다.':'Synthetictest의첫좌표직접전달계수a<0비율은0이다. 이통계는2D행렬의다른성분·고유값·반복경로의signed효과를판정하지못하며음의geometry학습의성공/실패로확대하지않는다.',
      '음수transport실사용0':'첫좌표직접전달계수a<0비율0'}
    for p in [ROOT/'runtime/archive.py',REPO/'findings/sheaf-pcfi.yaml',REPO/'RESULTS_SUMMARY_KO.md',REPO/'RESEARCH_HANDOFF_KO.md',REPO/'README.md']:
        text=p.read_text(encoding='utf8')
        for old,new in changes.items():text=text.replace(old,new)
        p.write_text(text,encoding='utf8')
    runpy.run_path(str(ROOT/'report.py'),run_name='__main__')
    for p in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'runtime/archive.py',Path(__file__)]:
        q=DEST/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    m={p.relative_to(DEST).as_posix():sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SNAPSHOT_MANIFEST.json'}
    (DEST/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
    print('SNAPSHOT REFRESHED',len(m))
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
    assert w['My Findings'].max_row==len(entries)+1
    assert w['Papers'].max_row==58
    assert w['Ideas'].max_row==13
    out={'status':'PASS','findings':len(entries),'snapshot_files':len(manifest),'old_entries_preserved':len(before['entries']),
         'old_paper_source_files_preserved':len(before['files']),'workbook_sheets':w.sheetnames}
    (ROOT/'runtime/final_harness_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False))
def tests():
    sys.path.insert(0,str(REPO/'src'))
    suite=unittest.defaultTestLoader.discover(str(REPO/'tests'))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
if __name__=='__main__':{'revise':revise,'verify':verify,'tests':tests}[sys.argv[1]]()

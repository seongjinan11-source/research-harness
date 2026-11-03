"""Finalize a completed experiment and save a local, unpushed research record."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
REPO = WORK/'research-harness'
DEPENDENCIES = 'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages'
PYTHON = [sys.executable,'-B','-X','utf8','-u']
GIT = ['git','-c','core.excludesFile=.git/info/exclude']


def run(command,cwd=WORK):
    process = subprocess.run(command,cwd=cwd,check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


def cli(name):
    code = f"import sys,runpy;sys.path.insert(0,'src');sys.path.append({DEPENDENCIES!r});sys.argv=['src/harness.py',{name!r}];runpy.run_path('src/harness.py',run_name='__main__')"
    run(PYTHON+['-c',code],REPO)


def main():
    assert 'TRAINING_AND_EVALUATION_COMPLETE' in (ROOT/'runtime/TASK_STATE.md').read_text()
    assert not subprocess.check_output(GIT+['status','--porcelain'],cwd=REPO).strip(),'Preserve unrelated repository changes'
    roots = {}
    for name in ['graph.html','research_harness.xlsx']:
        path = WORK/name
        if path.exists():
            data = path.read_bytes()
            backup = ROOT/'runtime/prior_root_outputs'/name
            backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_bytes(data)
            roots[name] = hashlib.sha256(data).hexdigest()
    for name in ['analyze.py','audit.py','report.py']:
        print('FINALIZE',name,flush=True)
        run(PYTHON+[str(ROOT/name)])
    print('FINALIZE archive',flush=True)
    run(PYTHON+[str(ROOT/'runtime/archive.py')])
    cli('validate')
    run(PYTHON+[str(ROOT/'runtime/finish.py'),'tests'])
    cli('build')
    run(PYTHON+[str(ROOT/'runtime/finish.py'),'verify'])
    for name in ['graph.html','research_harness.xlsx']:
        target = WORK/name
        if name in roots:
            assert hashlib.sha256(target.read_bytes()).hexdigest()==roots[name],'Root output changed during finalization'
        shutil.copyfile(REPO/'out'/name,target)
    paths = ['findings/pcfi-partial-clamp.yaml','findings/sources/pcfi-partial-clamp-2026-11-03',
        'ideas/pcfi-partial-clamp.yaml','RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md','README.md',
        'tools/build_review_packet.py','out']
    run(GIT+['add','-f','--']+paths,REPO)
    run(PYTHON+[str(ROOT/'runtime/finish.py'),'index'])
    run(GIT+['diff','--cached','--check','--','.',':(exclude)findings/sources/**'],REPO)
    run(GIT+['commit','-m','Record partial-coordinate confidence diffusion pilot and novelty review context'],REPO)
    assert not subprocess.check_output(GIT+['status','--porcelain'],cwd=REPO).strip()
    packet_code = f"import sys,runpy;sys.path.append({DEPENDENCIES!r});sys.argv=['tools/build_review_packet.py'];runpy.run_path('tools/build_review_packet.py',run_name='__main__')"
    run(PYTHON+['-c',packet_code],REPO)
    commit = subprocess.check_output(GIT+['rev-parse','HEAD'],cwd=REPO,text=True).strip()
    result = {'status':'COMPLETED','local_commit':commit,'pushed':False,
        'packet':str(WORK/'graphpfn_review_packet_2026-11-03_v20.zip'),
        'root_outputs_updated':list(roots),'prior_root_outputs_preserved':True}
    (ROOT/'runtime/finalization.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    (ROOT/'runtime/TASK_STATE.md').write_text(
        '# Partial-clamp experiment\n\nStatus: COMPLETED.\n\n'
        'All training/evaluation, paired analysis, audit, report, local research-harness commit and verified v20 review packet complete. No push.\n',encoding='utf8')
    print('FINALIZATION COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

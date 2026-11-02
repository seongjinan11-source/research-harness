import sys
sys.dont_write_bytecode=True
from pathlib import Path
import subprocess
import json
import time
root=Path(__file__).resolve().parents[1];logs=root/'logs';logs.mkdir(exist_ok=True)
started=time.time()
def launch(action,seed=None):
    tag=action+(f'_s{seed}' if seed is not None else '')
    args=[sys.executable,'-B','-X','utf8',str(root/'probe.py'),action]
    if seed is not None:args+=['--seed',str(seed)]
    with (logs/(tag+'.stdout.log')).open('w',encoding='utf8') as out,(logs/(tag+'.stderr.log')).open('w',encoding='utf8') as err:
        return subprocess.Popen(args,cwd=root.parent,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
for action in ['preflight','generate','controls']:
    code=launch(action).wait()
    if code:raise RuntimeError(f'{action} failed: {code}')
jobs=[launch('evaluate',s) for s in [11,12,13]]
codes=[j.wait() for j in jobs];assert codes==[0,0,0],codes
for action in ['summary','audit','cost']:
    code=launch(action).wait()
    if code:raise RuntimeError(f'{action} failed: {code}')
(root/'results/complete.json').write_text(json.dumps({'seconds':time.time()-started,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'new_updates':0},indent=2),encoding='utf8')
print('COMPLETE',flush=True)

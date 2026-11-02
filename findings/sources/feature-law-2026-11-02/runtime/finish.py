import sys
sys.dont_write_bytecode = True
from pathlib import Path
import subprocess
import time
root = Path(__file__).resolve().parents[1]
arms = ['original','static','adaptive','mask_pfn','pcfi_pfn']
begun = time.time()
while not all((root/'results'/f'{a}_s{s}'/'evaluation.json').exists() for a in arms for s in [11,12,13]):
    if time.time()-begun>7200:
        raise RuntimeError('Worker completion timeout; inspect logs, do not silently restart')
    time.sleep(5)
time.sleep(3)
for action in ['audit','cost','summary']:
    print('START',action,flush=True)
    subprocess.run([sys.executable,'-B','-X','utf8',str(root/'runtime/analyze.py'),action],cwd=root,check=True)
subprocess.run([sys.executable,'-B','-X','utf8',str(root/'runtime/write_report.py')],cwd=root,check=True)
print('FINISHED',flush=True)

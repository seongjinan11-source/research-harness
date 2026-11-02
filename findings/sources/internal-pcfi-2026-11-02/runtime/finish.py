import sys
sys.dont_write_bytecode=True
from pathlib import Path
import time
import subprocess
root=Path(__file__).resolve().parents[1]
start=time.time()
while not all((root/'results'/f'{a}_s{s}'/'evaluation.json').exists() for a in ['external','internal_fixed','context','label_blind'] for s in [11,12,13]):
    if time.time()-start>7200:raise RuntimeError('Timeout; preserve run, inspect workers')
    time.sleep(5)
time.sleep(3)
for action in ['audit','summary','cost']:
    print('START',action,flush=True)
    subprocess.run([sys.executable,'-B','-X','utf8',str(root/'pcfi_analysis.py'),action],cwd=root,check=True)
subprocess.run([sys.executable,'-B','-X','utf8',str(root/'runtime/report.py')],cwd=root,check=True)
print('FINISHED',flush=True)

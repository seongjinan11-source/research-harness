import sys
sys.dont_write_bytecode = True
from pathlib import Path
import datetime
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
log_path = ROOT/'runtime/execution.log'
assert not log_path.exists(),'Do not silently repeat an orchestrated run'
with log_path.open('w',encoding='utf-8') as log:
    for seed in [11,12,13]:
        (ROOT/'runtime/TASK_STATE.md').write_text(
            f'# Partial-clamp experiment\n\nStatus: RUNNING seed {seed}.\n'
            'Six new frontends,6144 total updates; all PFNs and real-graph weights fixed.\n',encoding='utf-8')
        command = [sys.executable,'-B','-X','utf8','-u',str(ROOT/'run.py'),'--seed',str(seed)]
        process = subprocess.Popen(command,cwd=str(ROOT.parent),stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        if code:
            (ROOT/'runtime/TASK_STATE.md').write_text(
                f'Status: FAILED seed {seed}, exit {code}. Preserve logs and checkpoints; do not restart silently.\n',encoding='utf-8')
            raise SystemExit(code)
(ROOT/'runtime/TASK_STATE.md').write_text(
    '# Partial-clamp experiment\n\nStatus: TRAINING_AND_EVALUATION_COMPLETE.\n'
    'Pending paired analysis, audit and Korean report.\n',encoding='utf-8')
print('ALL SEEDS COMPLETE',flush=True)

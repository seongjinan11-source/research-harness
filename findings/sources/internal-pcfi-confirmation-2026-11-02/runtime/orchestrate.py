import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import subprocess
import time

root = Path(__file__).resolve().parents[1]
logs = root / 'logs'; logs.mkdir(exist_ok=True)
script = root / 'confirmation.py'
started = time.time()


def run(action, seed=None):
    name = action + (f'_s{seed}' if seed is not None else '')
    args = [sys.executable, '-B', '-X', 'utf8', str(script), action]
    if seed is not None: args += ['--seed', str(seed)]
    with (logs / (name + '.stdout.log')).open('w', encoding='utf8') as out, (logs / (name + '.stderr.log')).open('w', encoding='utf8') as err:
        p = subprocess.Popen(args, cwd=root.parent, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
        return p


for action in ['lock', 'generate']:
    code = run(action).wait()
    if code: raise RuntimeError(f'{action} failed: {code}')
jobs = [run('evaluate', seed) for seed in [11, 12, 13]]
codes = [p.wait() for p in jobs]
assert codes == [0, 0, 0], codes
for action in ['summary', 'audit', 'cost']:
    code = run(action).wait()
    if code: raise RuntimeError(f'{action} failed: {code}')
(root / 'results/complete.json').write_text(json.dumps({'seconds': time.time()-started, 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'new_updates': 0}, indent=2), encoding='utf8')
print('COMPLETE', flush=True)

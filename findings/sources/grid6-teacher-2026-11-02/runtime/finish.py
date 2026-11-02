"""Wait for all declared evaluations, then run the fixed reporting/audit sequence."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[1]
names = ['mask_pfn_ce', 'pcfi_pfn_ce', 'factor_pfn_ce', 'mask_pfn_distill', 'factor_pfn_distill']
expected = [root / 'evaluation' / f'{arm}_s{seed}' / 'completed.json' for arm in names for seed in [11, 12, 13]]
deadline = time.monotonic() + 1800
while not all(path.exists() for path in expected):
    if time.monotonic() > deadline:
        raise TimeoutError('Evaluation did not complete; inspect declared worker logs')
    time.sleep(3)
for args in [['report.py'], ['cost_audit.py', 'benchmark'], ['report.py'], ['cost_audit.py', 'audit']]:
    print('RUN', args, flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', *args], cwd=root, check=True)
print('PIPELINE_COMPLETE', flush=True)

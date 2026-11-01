"""Finish the two authorized experiments; no changes to training or selection."""
import json,subprocess,sys,time
from pathlib import Path
WORK=Path(__file__).resolve().parents[2]
def ready(folder,arms):return all((WORK/folder/'results'/f'{arm}_s{seed}'/'evaluation.json').exists() for arm in arms for seed in [11,12,13])
def run(script,folder):
    out=WORK/folder/'runtime'/('finalize_'+Path(script).stem)
    with out.with_suffix('.stdout.log').open('w',encoding='utf8') as stdout,out.with_suffix('.stderr.log').open('w',encoding='utf8') as stderr:
        r=subprocess.run([sys.executable,'-X','utf8',script],cwd=WORK,stdout=stdout,stderr=stderr)
    if r.returncode:raise RuntimeError((script,r.returncode))
memory=(WORK/'feature_memory_pilot/results/completed.json').exists()
direct=(WORK/'direct_marginal_pilot/results/completed.json').exists()
while not(memory and direct):
    if not memory and ready('feature_memory_pilot',['context','all_nodes','observed','pcfi']):
        run('feature_memory_pilot/analyze.py','feature_memory_pilot');memory=True;print('MEMORY_ANALYZED',flush=True)
    if memory and not direct and ready('direct_marginal_pilot',['circuit','direct_graph','mask_pfn']):
        run('direct_marginal_pilot/runtime/benchmark_corrected.py','direct_marginal_pilot')
        run('direct_marginal_pilot/analyze.py','direct_marginal_pilot');direct=True;print('CIRCUIT_ANALYZED',flush=True)
    if not(memory and direct):time.sleep(10)
print('ALL_FINISHED',flush=True)

"""Same-device measurements after the experiment's training/evaluation jobs finish."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,time,gc
import numpy as np
import torch
from experiment import C,ROOT,cfg,make_model,rows,read,probabilities
from data import GridReference,dump


def measure(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize();steady=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();values=[]
    for _ in range(30):
        t=time.perf_counter();fn();torch.cuda.synchronize();values.append(1000*(time.perf_counter()-t))
    return {'median_ms':float(np.median(values)),'samples_ms':values,'additional_peak_mib':(torch.cuda.max_memory_allocated()-steady)/2**20,'resident_allocated_mib':steady/2**20}


def main():
    C.setup();torch.set_num_threads(2);row=next(r for r in rows('test') if r['case']=='matched' and r['context_total']==2 and r['mechanism']=='full')
    inp,q,y=read(row['path']);result=[]
    for arm in cfg()['arms']:
        for seed in cfg()['seeds']:
            out=ROOT/'results'/f'{arm}_s{seed}';model=make_model(arm,seed)
            state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if p.requires_grad:p.copy_(state[n].to(p.device))
            result.append({'arm':arm,'seed':seed,**measure(lambda:probabilities(model,arm,inp,q,111))})
            del model,state;gc.collect();torch.cuda.empty_cache()
    engine=GridReference('cuda')
    with torch.inference_mode():
        result.append({'arm':'grid6_bp','seed':0,**measure(lambda:engine.infer([inp],with_bp=True))})
        result.append({'arm':'grid6_exact','seed':0,**measure(lambda:engine.infer([inp],with_bp=False))})
    dump(ROOT/'results/benchmark.json',{'device':torch.cuda.get_device_name(),'row':row,'measurements':result,
        'scope':'All visible-input processing and forward/reference inference on the same GPU; excludes model construction, bank I/O and pretraining. Neural workers finished before measurements. External GPU activity is not controlled; raw repetitions retained.'})
    print('BENCHMARK_DONE',flush=True)
if __name__=='__main__':main()

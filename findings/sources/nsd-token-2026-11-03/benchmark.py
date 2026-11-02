import sys
sys.dont_write_bytecode=True
import sklearn
import json,gc,time
import numpy as np
import torch
from dataclasses import replace
from core import ROOT,C,cfg,dump,load_episode,load_model,predict,adjacency,impute
def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['graph'] in ['Cora','texas'] and r['repetition']==0 and r['case']=='uniform90'];out=[]
    for arm in cfg()['arms']:
        model=load_model(arm,11);before=C.parameter_digest(model)
        for row in rows:
            inp,_=load_episode(row)
            def run():
                z=inp
                if arm=='pcfi100':
                    z=replace(inp,pcfi=impute(adjacency((inp.src,inp.dst),len(inp.observed)),np.where(inp.observed,inp.observed_values,np.nan),'pcfi','structural',iterations=100)[0].astype('float32'))
                return predict(model,z)
            for _ in range(3):run()
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(10):
                t=time.perf_counter();run();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
            out.append({'arm':arm,'graph':row['graph'],'median_ms':float(np.median(times)),'p10_ms':float(np.quantile(times,.1)),'p90_ms':float(np.quantile(times,.9)),
                'samples_ms':times,'peak_additional_gpu_mib':(torch.cuda.max_memory_allocated()-base)/2**20,
                'scope':'CPU visible RP16input toCPUprobability;pcfi includes100CPUspatial+channelcorrection;NSD includesconfidencepreparation/maps/hiddenupdates. Excludescheckpointloading/initialRP/nativeoriginalfeaturePCFI.'})
            print('BENCH',arm,row['graph'],flush=True)
        assert C.parameter_digest(model)==before;del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/cost.json',{'device':torch.cuda.get_device_name(),'threads':2,'warmups':3,'repeats':10,'records':out,
        'limits':'Run serially after own GPU jobs;other userGPU activity uncontrolled. Current implementations,not a universal latency comparison. No timing claim versusnativePCFI.'})
if __name__=='__main__':main()

import sys
sys.dont_write_bytecode=True
import sklearn
import gc,json,time
import numpy as np
import torch
from core import ROOT,C,cfg,dump,load_model,load_episode,geometry,predict

def main():
    C.setup();torch.set_num_threads(2)
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['graph'] in ['Cora','texas'] and r['repetition']==0 and r['case']=='uniform90']
    output=[]
    for arm in cfg()['arms']:
        model=load_model(arm,11);before=C.parameter_digest(model)
        for row in rows:
            inp,_,_=load_episode(row)
            def run():
                # Geometry is rebuilt each time: no warm diffusion or input cache in these times.
                ex=geometry(inp.src,inp.dst,inp.observed,inp.observed_values,inp.context,inp.labels_context,inp.classes)
                return predict(model,inp,ex)
            for _ in range(3):run()
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(10):
                t=time.perf_counter();run();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
            output.append({'arm':arm,'graph':row['graph'],'case':row['case'],'seed':11,'repetition':0,
                'median_ms':float(np.median(times)),'p10_ms':float(np.quantile(times,.1)),'p90_ms':float(np.quantile(times,.9)),
                'samples_ms':times,'peak_additional_gpu_mib':(torch.cuda.max_memory_allocated()-base)/2**20,
                'scope':'CPU visible RP16 input to CPU probabilities, includes rebuilt PCFI100 warm geometry and descriptors even for mask, remaining front propagation and one PFN. Excludes checkpoint/file loading, initial RP and native original-feature PCFI. Current common implementation, not optimized method latency.'})
            print('COST',arm,row['graph'],flush=True)
        assert C.parameter_digest(model)==before
        del model;gc.collect();torch.cuda.empty_cache()
    dump(ROOT/'results/cost.json',{'device':torch.cuda.get_device_name(),'torch':torch.__version__,'cpu_threads':2,'warmup':3,'repeats':10,'records':output,
        'limitations':'Serial measurement after all own training/evaluation jobs finish; other-user GPU activity not controlled. Common preparation includes unnecessary work for mask. No speed comparison to native-PCFI or original public full preprocessing/ensemble.'})
if __name__=='__main__':main()

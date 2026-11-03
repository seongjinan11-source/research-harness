import sys
sys.dont_write_bytecode=True
import sklearn
import json,time,gc
import numpy as np
import torch
from core import ROOT,N,C,dump,load_episode,load_model,predict

def main():
    C.setup();torch.set_num_threads(2)
    rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['graph'] in ['Cora','texas'] and r['repetition']==0 and r['case']=='uniform90']
    out=[]
    for arm in ['identity','dynamic','node_frame']:
        obj=load_model(11) if arm=='node_frame' else N.load_model(arm,11);before=C.parameter_digest(obj)
        for row in rows:
            inp,_=load_episode(row)
            for _ in range(3):predict(obj,inp)
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(10):
                st=time.perf_counter();predict(obj,inp);torch.cuda.synchronize();times.append((time.perf_counter()-st)*1000)
            out.append({'arm':arm,'graph':row['graph'],'median_ms':float(np.median(times)),
                        'p10_ms':float(np.quantile(times,.1)),'p90_ms':float(np.quantile(times,.9)),
                        'peak_additional_gpu_mib':(torch.cuda.max_memory_allocated()-base)/2**20,'samples_ms':times})
        assert C.parameter_digest(obj)==before;del obj;gc.collect();torch.cuda.empty_cache();print('BENCH',arm,flush=True)
    dump(ROOT/'results/cost.json',{'device':torch.cuda.get_device_name(),'warmups':3,'repeats':10,'records':out,
          'scope':'Serial after own jobs. CPU visible RP16input toCPUprobabilities, includesconfidenceandmapdiffusion; excludesmodel/data loading andprojection. Existingedge-sparseimplementation retained; doesnotexploitaligned-coordinatefactorization. OtheruserGPUactivitynotcontrolled;additionalworkmemorynotfullVRAM.'})

if __name__=='__main__':main()

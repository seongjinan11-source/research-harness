import sys
sys.dont_write_bytecode=True
import sklearn
import json,time,gc
import numpy as np
import torch
from dataclasses import replace
from core import ROOT,C,N,dump,load_episode,load_model,predict,geometry,adjacency,distances,impute

def base_geometry(inp):
    a=adjacency((inp.src,inp.dst),len(inp.observed));pc=np.power(np.float32(.9),distances(a,inp.observed,True));ii,jj=a.nonzero()
    return {'pc':pc,'ii':ii.astype('int64'),'jj':jj.astype('int64'),'weights':(pc[jj]/np.maximum(a@pc,1e-30)[ii]).astype('float32')}

def main():
    C.setup();torch.set_num_threads(2);rows=[r for r in json.loads((ROOT/'bank/manifest.json').read_text())['episodes'] if r['split']=='real' and r['graph'] in ['Cora','texas'] and r['repetition']==0 and r['case']=='uniform90'];out=[]
    for arm in ['pcfi_numpy','pcfi','scalar','edge','node']:
        obj=load_model('pcfi' if arm=='pcfi_numpy' else arm,11);before=C.parameter_digest(obj.pfn)
        for row in rows:
            inp,_,_=load_episode(row)
            def run():
                if arm=='pcfi_numpy':
                    fill=impute(adjacency((inp.src,inp.dst),len(inp.observed)),np.where(inp.observed,inp.observed_values,np.nan),'pcfi','structural',iterations=100)[0]
                    return N.predict(obj.pfn,replace(inp,pcfi=fill))
                return predict(obj,inp,base_geometry(inp) if arm=='pcfi' else geometry(inp))
            for _ in range(3):run()
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(10):
                st=time.perf_counter();run();torch.cuda.synchronize();times.append((time.perf_counter()-st)*1000)
            out.append({'arm':arm,'graph':row['graph'],'median_ms':float(np.median(times)),'p10_ms':float(np.quantile(times,.1)),
                        'p90_ms':float(np.quantile(times,.9)),'peak_additional_gpu_mib':(torch.cuda.max_memory_allocated()-base)/2**20,'samples_ms':times})
        assert C.parameter_digest(obj.pfn)==before;del obj;gc.collect();torch.cuda.empty_cache();print('BENCH',arm,flush=True)
    dump(ROOT/'results/cost.json',{'device':torch.cuda.get_device_name(),'warmups':3,'repeats':10,'records':out,
         'scope':'Serial after own jobs,visibleRP16CPUinput toCPUprobability. Recomputesconfidenceandgeometry;PCFItorchskipunusedLPdescriptors;PCFInumpyisoriginalCPUimputerplusidenticalPFN. Includes100spatial/onecorrection;excludesmodel/data load/RP. AdditionalmemorynotfullVRAM;otheruserGPUloaduncontrolled. Nooptimizednode-framealignmentimplementation.'})

if __name__=='__main__':main()

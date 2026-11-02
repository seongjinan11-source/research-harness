import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pilot as P
import sampler as S
import numpy as np
import torch
import json,time,gc
def measure(fn):
    for _ in range(3):fn()
    tt=[]
    for _ in range(15):
        torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize();tt.append(time.perf_counter()-start)
    return {'median_ms':1000*float(np.median(tt)),'p10_ms':1000*float(np.quantile(tt,.1)),'p90_ms':1000*float(np.quantile(tt,.9)),'repeats':15}
def main():
    P.C.setup();torch.set_num_threads(2);P.lock();c=P.cfg();row=next(r for r in P.test_rows() if r['case']=='matched' and r['context_total']==2 and r['observed_nodes']==6)
    inp,q,y=P.read_test(row);factor=P.load_factor(11);net=S.Denoiser(c['width']).cuda().eval();net.load_state_dict(torch.load(ROOT/'results/denoiser_s11/checkpoint.pt',weights_only=True,map_location='cuda'))
    costs={}
    for arm in c['arms']:
        def fn():
            cc=P.condition(factor,[inp]);p,_,_=S.sample(net,cc,arm,18800001,particles=c['particles'],stages=c['stages'],sweeps=c['gibbs_cost_sweeps'] if arm=='smc_gibbs_cost' else 1,cess=c['cess_fraction'])
            return p[0,q].cpu().numpy()
        costs[arm]=measure(fn)
    for name,model in [('factor_bp',factor),('constant_bp',P.load_factor(11,True))]:
        def fn():
            with torch.inference_mode():return model(P.D.tensors([inp]))[0,q].cpu().numpy()
        costs[name]=measure(fn)
    engine=P.D.NATIVE.GridReference('cuda')
    for name,with_bp in [('grid6_exact',False),('grid6_bp',True)]:costs[name]=measure(lambda:engine.infer([inp],with_bp=with_bp)[name][0,q])
    for arm in ['mask_pfn','pcfi_pfn']:
        model=P.C.Model('mask_plain' if arm=='mask_pfn' else 'pcfi_plain',11)
        state=torch.load(P.OLD/'results'/f'{arm}_s11/checkpoint.pt',weights_only=False,map_location='cpu')['state_dict']
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.requires_grad:p.copy_(state[name].to(p.device))
        def fn():
            given={k:v.copy() for k,v in inp.items()}
            if arm=='pcfi_pfn':given['pcfi']=P.D.pcfi(given)
            return P.C.predict(model,P.D.public_input(given,q,18800001))
        costs[arm]=measure(fn);del model;gc.collect();torch.cuda.empty_cache()
    P.dump(ROOT/'results/benchmark.json',{'device':torch.cuda.get_device_name(0),'scope':'Single primary-like fresh graph/seed11;visible CPU input to CPU probabilities;include frozen factor extraction/SMC/RB or public PCFI;preload weights and reusable enumerator;exclude I/O;external GPU load not controlled. Dev cost control2sweeps was fixed before test. No latency winner claim beyond this measurement.',
        'row':row,'methods':costs,'source_sha':P.D.sha(Path(__file__))})
    print('BENCHMARK',json.dumps(costs),flush=True)
if __name__=='__main__':main()

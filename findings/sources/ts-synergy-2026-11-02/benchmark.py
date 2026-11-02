import sys
sys.dont_write_bytecode = True
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import torch
import run as r
from chronos import Chronos2Pipeline
R=Path(__file__).resolve().parent

def main():
    B=dict(np.load(R/'bank.npz'));Q=json.loads((Path(r.P['chronos_model'])/'config.json').read_text())['chronos_config']['quantiles']
    torch.set_num_threads(4)
    pipe=Chronos2Pipeline.from_pretrained(r.P['chronos_model'],device_map='cuda',dtype=torch.float32,local_files_only=True)
    pipe.model.eval()
    ts=np.arange(4,r.L-1)
    def classical(i):
        ids=r.view_ids(B,i,'uvN8');xx=r.features(B,i,ids,ts,True);xt=r.features(B,i,ids,B['origins'],True);yy=B['y'][i,ts+1]
        sc=StandardScaler().fit(xx[:320]);tr=sc.transform(xx[:320]);va=sc.transform(xx[320:416]);scores=[]
        for alpha in r.P['lasso_alphas']:
            f=Lasso(alpha=alpha,max_iter=4000,tol=1e-5).fit(tr,yy[:320])
            scores.append(np.mean((f.predict(va)-yy[320:416])**2))
        alpha=r.P['lasso_alphas'][np.argmin(scores)];sc=StandardScaler().fit(xx[:416])
        f=Lasso(alpha=alpha,max_iter=4000,tol=1e-5).fit(sc.transform(xx[:416]),yy[:416])
        res=yy[416:]-f.predict(sc.transform(xx[416:]))
        return f.predict(sc.transform(xt))[:,None]+np.quantile(res,Q)[None]
    def neural(i):
        ids=r.view_ids(B,i,'uvN8')
        gs=[np.concatenate([B['y'][i,None,t-r.L+1:t+1],B['x'][i,ids,t-r.L+1:t+1]]) for t in B['origins']]
        with torch.inference_mode():
            return np.concatenate([r.group_forward(pipe.model,gs[j:j+8]).cpu().numpy() for j in range(0,len(gs),8)])
    times={};errs={}
    with threadpool_limits(limits=4):
        for name,fn in [('interaction_lasso',classical),('chronos2',neural)]:
            for _ in range(3):fn(0)
            vals=[]; ee=[]
            ref=np.load(R/'results/classical.npz')[name] if name=='interaction_lasso' else np.load(R/f'results/{name}.npz')['predictions']
            for i in range(8):
                for _ in range(5):
                    torch.cuda.synchronize();st=time.perf_counter();pred=fn(i);torch.cuda.synchronize();vals.append(1000*(time.perf_counter()-st))
                    ee.append(np.max(np.abs(pred-ref[i,5])))
            times[name]={'median_ms_per_task_16_origins':float(np.median(vals)),'p10_ms':float(np.quantile(vals,.1)),'p90_ms':float(np.quantile(vals,.9)),'measurements':len(vals)}
            errs[name]=float(max(ee))
    assert max(errs.values()) < 2e-5
    out={'times':times,'replay_max_abs':errs,'scope':'8producttasks/5repeats/3warmup, full10covariates, CPU inputs -> CPU21quantile forecasts for16origins; classical includes all-candidate features, alpha validation, refit, residual calibration; neural excludes shared checkpoint loading; isolated implementations on shared RTX3090,4CPUthreads; not real-data deployment benchmark.'}
    (R/'results/benchmark.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out)
if __name__=='__main__':main()

"""Independent small-state validation; not test-bank tuning or a rescue arm."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pilot as P
import sampler as S
import torch
def main():
    P.C.setup();torch.set_num_threads(2);P.lock();torch.manual_seed(18300999);model=S.Denoiser(24).cuda().eval()
    j=torch.zeros((1,4,4,4),device='cuda')
    for a,b,v in [(0,1,.7),(1,2,-.5),(2,3,.6)]:j[:,:,a,b]=j[:,:,b,a]=v
    c={'j':j,'const':j.new_tensor([[.1,-.2,.3,-.1]]),'mass':j.new_tensor([[.5,0,0,.5]]),
       'context':torch.tensor([[1,-1,-1,-1]],device='cuda'),'types':torch.zeros((1,4),dtype=torch.long,device='cuda')}
    exact=P.exact(c);result={}
    for arm in ['smc_gibbs1','diffusion_smc_fixed','diffusion_smc_ess']:
        p,raw,diag=S.sample(model,c,arm,18301000,particles=4096,stages=8,sweeps=1)
        error=float((p-exact['p1']).abs().max());role_error=float((diag['role']-exact['role']).abs().max())
        assert error<.025 and role_error<.04,(arm,error,role_error)
        result[arm]={'node_max_error':error,'role_max_error':role_error}
    P.dump(ROOT/'results/convergence_audit.json',{'status':'PASS','particles':4096,'steps':8,'condition':'Independent4spin/2allowedrole tree,onehardcontext;untrainedproposal;notproductiondata or a scaling claim','results':result,
        'scope':'Post-analysis numerical audit of sampler correctness after production finite-particle degradation. No training/config/source/checkpoint/primary changes.','source_sha':P.D.sha(Path(__file__))})
    print('SMALL_STATE_AUDIT',result,flush=True)
if __name__=='__main__':main()

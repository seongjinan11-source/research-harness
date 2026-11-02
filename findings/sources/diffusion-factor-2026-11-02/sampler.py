"""Discrete conditional diffusion and correctly MH-corrected SMC mutations."""
import sys
sys.dont_write_bytecode=True
import sklearn
import math,time
import torch
from torch import nn
from torch.nn import functional as F

def energy(j,const,s,h):
    # j[B,H,L,L],s[B,K,L],h[B,K]
    chosen=j[torch.arange(len(j),device=j.device)[:,None],h]
    return .5*torch.einsum('bki,bkij,bkj->bk',s,chosen,s)+const.gather(1,h)

class Denoiser(nn.Module):
    def __init__(self,width=24):
        super().__init__();self.odd=nn.Linear(6,width,bias=False)
        self.gate=nn.Linear(30,width)
        self.blocks=nn.ModuleList([nn.Linear(5*width,width,bias=False) for _ in range(2)])
        self.out=nn.Linear(width,1,bias=False)
        self.role=nn.Sequential(nn.Linear(2*width+22,width),nn.SiLU(),nn.Linear(width,4))
    def forward(self,c,s,h,t,stages):
        j,const,mass,context,types=[c[k] for k in ['j','const','mass','context','types']]
        b,k,l=s.shape;row=j.abs().sum(-1);norm=j/row.clamp_min(1)[...,None]
        fields=torch.einsum('brij,bkj->bkri',norm,s).transpose(-1,-2)
        known=context>=0;clamp=(2*context-1).to(s.dtype)*known
        odd=torch.cat((s[...,None],clamp[:,None,:,None].expand(-1,k,-1,-1),fields),-1)
        eh=.5*torch.einsum('bki,brij,bkj->bkr',s,j,s)+const[:,None]
        eh=(eh-eh.mean(-1,keepdim=True))/32
        ct=(const-const.mean(-1,keepdim=True))/32
        timev=torch.stack((t/stages,1-t/stages),-1)
        glob=torch.cat((F.one_hot(h,4).float(),ct[:,None].expand(-1,k,-1),mass[:,None].expand(-1,k,-1),
                        row.mean(-1)[:,None].expand(-1,k,-1)/8,eh,timev),-1)
        even=torch.cat((known.float()[...,None],F.one_hot(types,3).float(),row.transpose(1,2)/8),-1)
        gate=torch.sigmoid(self.gate(torch.cat((even[:,None].expand(-1,k,-1,-1),glob[:,:,None].expand(-1,-1,l,-1)),-1)))
        x=torch.tanh(self.odd(odd))*gate
        for block in self.blocks:
            msgs=torch.einsum('brij,bkjd->bkrid',norm,x).transpose(2,3).flatten(-2)
            x=x+.25*torch.tanh(block(torch.cat((x,msgs),-1)))*gate
        p1=.98*torch.sigmoid(self.out(x).squeeze(-1))+.01
        p1=torch.where(known[:,None],context.clamp_min(0)[:,None].to(p1.dtype),p1)
        role_logits=self.role(torch.cat((x.square().mean(-2),(x*s[...,None]).mean(-2),glob),-1))
        role_logits=role_logits.masked_fill(mass[:,None]==0,-torch.inf)
        ph=.98*role_logits.softmax(-1)+.02*mass[:,None]
        return p1,ph

def reset_reverse(clean,current,base,t,stages):
    # clean[B,K,L,C],current[B,K,L],base[B,1,L,C] or broadcast.
    at=1-t/stages;ap=1-(t-1)/stages;r=at/ap.clamp_min(1e-12)
    classes=clean.shape[-1];one=F.one_hot(current,classes).to(clean.dtype)
    at=at[...,None,None];ap=ap[...,None,None];r=r[...,None,None]
    base=base.expand_as(clean);bj=(base*one).sum(-1,keepdim=True)
    denom=(at*one+(1-at)*bj).clamp_min(1e-12)
    transition=r*one+(1-r)*bj
    weighted=clean/denom
    result=transition*(ap*weighted+(1-ap)*base*weighted.sum(-1,keepdim=True))
    return result/result.sum(-1,keepdim=True)

def proposal(model,c,s,h,t,stages):
    p1,ph=model(c,s,h,t,stages);binary=torch.stack((1-p1,p1),-1)
    ps=reset_reverse(binary,((s+1)/2).long(),binary.new_full((1,1,1,2),.5),t,stages)
    pr=reset_reverse(ph[:,:,None],h[:,:,None],c['mass'][:,None,None,:],t,stages).squeeze(-2)
    known=c['context']>=0;clamp=F.one_hot(c['context'].clamp_min(0),2).to(ps.dtype)
    ps=torch.where(known[:,None,:,None],clamp[:,None],ps)
    return ps,pr

def draw_from(ps,ph,generator):
    ss=torch.multinomial(ps.flatten(0,-2),1,generator=generator).reshape(ps.shape[:-1])
    hh=torch.multinomial(ph.flatten(0,-2),1,generator=generator).reshape(ph.shape[:-1])
    return (2*ss-1).to(ps.dtype),hh
def logq(ps,ph,s,h):
    return ps.gather(-1,((s+1)/2).long()[...,None]).squeeze(-1).clamp_min(1e-30).log().sum(-1)+ph.gather(-1,h[...,None]).squeeze(-1).clamp_min(1e-30).log()
def base_draw(c,k,generator):
    b,_,l,_=c['j'].shape
    s=(2*torch.randint(2,(b,k,l),device=c['j'].device,generator=generator)-1).to(c['j'].dtype)
    s=torch.where(c['context'][:,None]>=0,(2*c['context'][:,None]-1).to(s.dtype),s)
    h=torch.multinomial(c['mass'],k,replacement=True,generator=generator)
    return s,h
def rb(c,s,h,w):
    j=c['j'][torch.arange(len(s),device=s.device)[:,None],h]
    field=torch.einsum('bkij,bkj->bki',j,s)
    p1=torch.sigmoid(2*field);p1=torch.where(c['context'][:,None]>=0,c['context'][:,None].to(p1.dtype),p1)
    return (w[...,None]*p1).sum(1),(w[...,None]*((s+1)/2)).sum(1)

def resample(s,h,w,generator):
    b,k=h.shape;u=(torch.rand((b,1),device=s.device,generator=generator)+torch.arange(k,device=s.device))/k
    ids=torch.searchsorted(w.cumsum(1).contiguous(),u.contiguous()).clamp_max(k-1)
    return s.gather(1,ids[...,None].expand_as(s)),h.gather(1,ids)

def mh_diff(model,c,s,h,beta,t,stages,generator):
    q,ph=proposal(model,c,s,h,t,stages);ss,hh=draw_from(q,ph,generator)
    rev,rh=proposal(model,c,ss,hh,t,stages)
    loga=beta[:,None]*(energy(c['j'],c['const'],ss,hh)-energy(c['j'],c['const'],s,h))
    loga+=c['mass'].gather(1,hh).log()-c['mass'].gather(1,h).log()+logq(rev,rh,s,h)-logq(q,ph,ss,hh)
    accept=torch.rand(h.shape,device=s.device,generator=generator).log()<loga
    return torch.where(accept[...,None],ss,s),torch.where(accept,hh,h),accept.float().mean(1)

def gibbs(c,s,h,beta,sweeps,generator):
    b,k,l=s.shape;ar=torch.arange(b,device=s.device)[:,None]
    for _ in range(sweeps):
        chosen=c['j'][ar,h]
        for i in range(l):
            field=(chosen[:,:,i]*s).sum(-1)
            value=(2*(torch.rand((b,k),device=s.device,generator=generator)<torch.sigmoid(2*beta[:,None]*field))-1).to(s.dtype)
            s[:,:,i]=torch.where(c['context'][:,None,i]>=0,(2*c['context'][:,None,i]-1).to(s.dtype),value)
        eh=.5*torch.einsum('bki,brij,bkj->bkr',s,c['j'],s)+c['const'][:,None]
        prob=(beta[:,None,None]*eh+c['mass'][:,None].log()).softmax(-1)
        h=torch.multinomial(prob.flatten(0,1),1,generator=generator).reshape(b,k)
        # A cheap global proposal prevents making the neural proposal the only nonlocal mutation.
        ss,hh=base_draw(c,k,generator)
        la=beta[:,None]*(energy(c['j'],c['const'],ss,hh)-energy(c['j'],c['const'],s,h))
        take=torch.rand((b,k),device=s.device,generator=generator).log()<la
        s=torch.where(take[...,None],ss,s);h=torch.where(take,hh,h)
    return s,h

@torch.inference_mode()
def sample(model,c,arm,seed,particles=64,stages=8,sweeps=1,cess=.8):
    gen=torch.Generator(device=c['j'].device).manual_seed(seed);s,h=base_draw(c,particles,gen)
    b=len(s);w=s.new_full((b,particles),1/particles);beta=s.new_zeros(b);ess_trace=[];betas=[];accepts=[];forced=s.new_zeros(b)
    if arm=='diffusion_only':
        for level in range(stages,0,-1):
            t=s.new_full((b,particles),level);ps,ph=proposal(model,c,s,h,t,stages);s,h=draw_from(ps,ph,gen)
    else:
        for stage in range(1,stages+1):
            e=energy(c['j'],c['const'],s,h)
            if arm=='diffusion_smc_ess' and stage<stages:
                low=beta.clone();high=torch.ones_like(beta)
                for _ in range(32):
                    mid=(low+high)/2;ww=((mid-beta)[:,None]*e).softmax(1);ess=1/ww.square().sum(1)
                    low=torch.where(ess>=cess*particles,mid,low);high=torch.where(ess<cess*particles,mid,high)
                new=low
            else:new=s.new_full((b,),stage/stages) if arm!='diffusion_smc_ess' else torch.ones_like(beta)
            if arm=='diffusion_smc_ess' and stage==stages:forced=(1-beta)
            w=((new-beta)[:,None]*e).softmax(1)
            ess_trace.append(1/w.square().sum(1));betas.append(new);beta=new
            if stage<stages:s,h=resample(s,h,w,gen);w=s.new_full((b,particles),1/particles)
            if arm.startswith('smc_gibbs'):s,h=gibbs(c,s,h,beta,sweeps,gen)
            else:
                t=((1-beta)*stages).ceil().clamp(1,stages)[:,None].expand(-1,particles)
                s,h,acc=mh_diff(model,c,s,h,beta,t,stages,gen);accepts.append(acc)
    p,praw=rb(c,s,h,w)
    role=torch.zeros((b,4),device=s.device).scatter_add_(1,h,w)
    info={'ess':1/w.square().sum(1),'role':role,'final_jump':forced,
          'acceptance':torch.stack(accepts,1).mean(1) if accepts else s.new_full((b,),-1),
          'beta':torch.stack(betas,1) if betas else s.new_zeros((b,stages))}
    return p,praw,info

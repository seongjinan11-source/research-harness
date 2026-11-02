"""Fixed compressed-input Gaussian/probit evidence; no learned parameters."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, log_ndtr, ndtri_exp, logsumexp
from scipy.stats import qmc


def kernel(graph):
    p=graph.toarray().astype(np.float64);p/=p.sum(1,keepdims=True)
    r=np.linalg.solve(np.eye(len(p))-.92*p,np.eye(len(p)))
    r-=r.mean(0,keepdims=True)
    k=r@r.T;k/=np.trace(k)/len(k)
    return (k+k.T)/2


def observed_score(inp):
    mask=inp.observed
    assert np.all(mask==mask[:,:1]), 'Whole-node model only'
    ids=np.flatnonzero(mask[:,0])
    if not len(ids):return ids,np.empty(0)
    x=inp.observed_values[ids].astype(np.float64)
    u,s,_=np.linalg.svd(x,full_matrices=False)
    z=u[:,0]*s[0]
    rms=np.sqrt(np.mean(z*z))
    return ids,z/max(rms,1e-12)


def sobol(d,bits=10):
    u=qmc.Sobol(d=d,scramble=True,seed=184021).random_base2(bits)
    return np.concatenate((u,1-u),0).T


def feature_infer(inp,k,b):
    """Strictly label-blind feature-law evidence; never access context/labels."""
    sources,score=observed_score(inp);logits=np.zeros((2,3))
    for g in [0,1]:
        value=score*(b[sources] if g else 1)
        for ai,amplitude in enumerate([.5,1.,2.]):
            if len(sources):
                covariance=amplitude**2*k[np.ix_(sources,sources)]+np.eye(len(sources))*.15**2
                ch=cho_factor(covariance,lower=True,check_finite=False)
                alpha=cho_solve(ch,value,check_finite=False)
                logits[g,ai]=-.5*(len(sources)*np.log(2*np.pi)+2*np.log(np.diag(ch[0])).sum()+value@alpha)
    logits=logsumexp(logits-np.log(3),axis=1)
    return np.exp(logits-logsumexp(logits))


def orthant_predict(mean,cov,context,signs,query,query_signs,uniforms):
    """P(signs*a_C>0) and conditional query sign probabilities under one Gaussian."""
    c=np.asarray(context);q=np.asarray(query)
    cc=cov[np.ix_(c,c)];cc=(cc+cc.T)/2
    jitter=max(float(np.diag(cc).max()),1.)*1e-10
    cc=cc+np.eye(len(c))*jitter
    factor=cho_factor(cc,lower=True,check_finite=False)
    qc=cov[np.ix_(q,c)]
    regression=cho_solve(factor,qc.T,check_finite=False).T
    variance=np.maximum(np.diag(cov)[q]-(regression*qc).sum(1),1e-10)
    flipped=cc*signs[:,None]*signs[None,:]
    l=np.linalg.cholesky(flipped)
    mu=mean[c]*signs
    z=np.zeros_like(uniforms);log_weight=np.zeros(uniforms.shape[1])
    for j in range(len(c)):
        shift=l[j,:j]@z[:j] if j else np.zeros(uniforms.shape[1])
        mass=log_ndtr((mu[j]+shift)/l[j,j])
        z[j]=-ndtri_exp(np.log1p(-uniforms[j])+mass)
        log_weight+=mass
    logz=logsumexp(log_weight)-np.log(len(log_weight))
    weight=np.exp(log_weight-logsumexp(log_weight))
    residual=signs[:,None]*(l@z)
    query_mean=mean[q,None]+regression@residual
    probability=ndtr(query_signs[:,None]*query_mean/np.sqrt(variance)[:,None])@weight
    ess=1/np.sum(weight*weight)
    return float(logz),probability,float(ess)


def infer(inp,k,b,*,bits=10,without_features=False):
    """Only visible graph kernel, BFS parity, values/mask and context labels enter."""
    context=np.asarray(inp.context);order=np.argsort(context);context=context[order]
    labels=np.asarray(inp.labels_context)[order];spin=2*labels-1
    query=np.asarray(inp.query);u=sobol(len(context),bits)
    sources,score=observed_score(inp)
    if without_features:sources=np.empty(0,np.int64);score=np.empty(0)
    branches=[];logits=[];feature_logs=np.zeros((2,3));pred=[];ess=[]
    for g in [0,1]:
        value=score*(b[sources] if g else 1)
        for ai,amplitude in enumerate([.5,1.,2.]):
            covariance=k*amplitude**2
            mean=np.zeros(len(k));conditional=covariance
            log_feature=0.
            if len(sources):
                ss=covariance[np.ix_(sources,sources)]+np.eye(len(sources))*.15**2
                ch=cho_factor(ss,lower=True,check_finite=False)
                alpha=cho_solve(ch,value,check_finite=False)
                log_feature=-.5*(len(sources)*np.log(2*np.pi)+2*np.log(np.diag(ch[0])).sum()+value@alpha)
                cross=covariance[:,sources]
                mean=cross@alpha
                conditional=covariance-cross@cho_solve(ch,cross.T,check_finite=False)
                conditional=(conditional+conditional.T)/2
            feature_logs[g,ai]=log_feature-np.log(3)
            for t in [0,1]:
                bc=b[context] if t else np.ones(len(context))
                bq=b[query] if t else np.ones(len(query))
                for sigma in [-1,1]:
                    log_context,p,e=orthant_predict(mean,conditional,context,sigma*bc*spin,query,sigma*bq,u)
                    branches.append([g,t,sigma,amplitude])
                    logits.append(log_feature+log_context-np.log(24))
                    pred.append(p);ess.append(e)
    weight=np.exp(np.asarray(logits)-logsumexp(logits));pred=np.asarray(pred)
    branch=np.asarray(branches)
    joint_g=np.array([weight[branch[:,0]==g].sum() for g in [0,1]])
    joint_t=np.array([weight[branch[:,1]==t].sum() for t in [0,1]])
    feature_g=logsumexp(feature_logs,axis=1);feature_g=np.exp(feature_g-logsumexp(feature_g))
    py=weight@pred
    result={'probability':np.column_stack((1-py,py)),'joint_g':joint_g,'joint_t':joint_t,'feature_g':feature_g,
        'joint_weights':weight,'branches':branches,'weighted_ess':float(weight@np.asarray(ess)),
        'min_branch_ess':float(min(ess)),'particles':u.shape[1]}
    assert all(np.isfinite(result[key]).all() for key in ['probability','joint_g','joint_t','feature_g','joint_weights'])
    return result

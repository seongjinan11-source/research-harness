"""Supplementary stored-control and mixture audit; no changes to locked study."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import experiment as E
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    rows=E.rows();index=E.inference_index();saved={};count=0;max_norm=0.;mix_error=0.;minimum=0.;maximum=1.
    for i,row in enumerate(rows):
        inp,y,_=E.read(row);missing=~inp.observed[inp.query].any(1)
        with np.load(root/index[i]['path']) as z:
            probabilities=z['probabilities'];fg=z['feature_g'];jg=z['joint_g'];lt=z['label_t']
        assert probabilities.shape==(len(E.CONTROL_METHODS),len(y),2)
        # Gaussian quadrature can put a mathematically unit probability a few ulps above1.
        # Keep raw predictions intact; inspect/report bounds with1e-12 numerical tolerance.
        assert np.isfinite(probabilities).all() and (probabilities>=-1e-12).all() and (probabilities<=1+1e-12).all()
        minimum=min(minimum,float(probabilities.min()));maximum=max(maximum,float(probabilities.max()))
        err=float(abs(probabilities.sum(-1)-1).max());max_norm=max(err,max_norm);assert err<1e-10
        for method,probability in zip(E.CONTROL_METHODS,probabilities):
            a=E.C.metrics(probability,y)
            b=E.C.metrics(probability[missing],y[missing]) if missing.any() else None
            assert all(np.isfinite(list(a.values())))
            if b is not None:assert all(np.isfinite(list(b.values())))
            count+=1
        # Label-only predictions must not depend on the feature law or number of sources.
        key=(row['graph'],row['label_law'],row['k'])
        subset=probabilities[[E.CONTROL_METHODS.index(m) for m in ['positive_lp','signed_lp','adaptive_lp','gaussian_label_only']]]
        if key in saved:
            assert np.max(abs(saved[key][0]-subset))<1e-12
            assert np.max(abs(saved[key][1]-lt))<1e-12
        else:saved[key]=(subset,lt)
        if row['sources'] in [0,1]:
            assert np.max(abs(fg-.5))<1e-8 and np.max(abs(jg-.5))<1e-8
        for seed in E.cfg()['seeds']:
            path=root/'results'/f'pcfi_plain_s{seed}'/'predictions'/f'e{i}.npz'
            with np.load(path) as z:p=z['p']
            for method,w in [('uniform_two',.5),('feature_adapt',fg[1]),('joint_adapt',jg[1])]:
                expected=(1-w)*p[0]+w*p[1]
                error=float(abs(expected-p[E.PCFI_METHODS.index(method)]).max())
                mix_error=max(error,mix_error);assert error<2e-7
    result={'classical_probability_arrays_checked':count,'label_only_feature_invariance_groups':len(saved),
        'max_probability_normalization_error':max_norm,'probability_min':minimum,'probability_max':maximum,
        'probability_bound_tolerance':1e-12,'stored_neural_mixture_max_error':mix_error,
        'scope':'Additional audit of stored controls and neural mixture algebra. No source/protocol/prediction modification; no new training.'}
    E.dump(root/'results/classical_audit.json',result)
    print(json.dumps(result),flush=True)

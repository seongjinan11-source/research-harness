"""Read-only supplementary comparisons of the completed ADTR bank; no training."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
import numpy as np
root=Path(__file__).resolve().parent;study=root.parent/'adaptive_transport_pilot'
sys.path.insert(0,str(study))
import experiment as E
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    test,index,data,lookup=E.collect();vectors={}
    for method,records in data.items():
        grouped=[[] for _ in range(64)]
        for row in records:
            if row['sources']==8 and row['k']==4:
                grouped[row['graph']].append([row['missing'][m] for m in E.METRICS])
        assert all(len(v)==(12 if method not in E.CONTROL_METHODS else 4) for v in grouped)
        vectors[method]=np.asarray([np.mean(v,axis=0) for v in grouped])
    comparisons={m:E.difference(vectors['gaussian_joint'],v) for m,v in vectors.items() if m!='gaussian_joint'}
    nonprivileged=[m for m in comparisons if m not in ['known_g_reference','known_t_lp_reference']]
    summary=json.loads((study/'results/statistics.json').read_text(encoding='utf8'))
    means={m:dict(zip(E.METRICS,v.mean(0).tolist())) for m,v in vectors.items()}
    for m in means:
        for metric in E.METRICS:assert abs(means[m][metric]-summary['primary']['means'][m][metric])<1e-12
    counterexamples=[]
    for key,cell in summary['cells'].items():
        if key=='primary':continue
        means_cell=cell['means'];prob=means_cell['gaussian_joint']['accuracy']
        others={m:v['accuracy'] for m,v in means_cell.items() if m not in ['gaussian_joint','known_g_reference','known_t_lp_reference']}
        best=max(others,key=others.get)
        if others[best]>prob:
            counterexamples.append({'cell':key,'gaussian_accuracy':prob,'best_nonprivileged':best,'best_accuracy':others[best]})
    paths=[study/'protocol.json',study/'bank/manifest.json',study/'results/statistics.json']
    paths+=list((study/'results').glob('inference_part*.json'))+list((study/'results').glob('*/evaluation.json'))
    inputs={p.relative_to(root.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    result={'scope':'Supplementary comparisons on the existing primary cohort; not new evaluation/training or a replacement primary. Pointwise95CI, no multiplicity correction.',
        'graphs':64,'fixed_training_seeds':[11,12,13],'law_cells':4,'primary_sources':8,'context_per_class':4,
        'bootstrap_repeats':5000,'bootstrap_seed':202611030101,'means':means,'gaussian_minus':comparisons,
        'nonprivileged_comparators':nonprivileged,
        'all_nonprivileged_accuracy_pointwise_ci_lower_positive':all(comparisons[m]['accuracy']['ci95'][0]>0 for m in nonprivileged),
        'all_nonprivileged_nll_brier_pointwise_ci_upper_negative':all(comparisons[m][metric]['ci95'][1]<0 for m in nonprivileged for metric in ['nll','brier']),
        'point_estimate_cell_counterexamples':counterexamples,'input_hashes':inputs,'new_models':0,'new_updates':0,'new_graphs':0}
    (root/'PRIMARY_COMPARISONS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['graphs','all_nonprivileged_accuracy_pointwise_ci_lower_positive','all_nonprivileged_nll_brier_pointwise_ci_upper_negative','new_updates']}))

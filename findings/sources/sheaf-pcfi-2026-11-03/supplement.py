import sys
sys.dont_write_bytecode=True
import sklearn
import json
from collections import defaultdict
import numpy as np
from core import ROOT,C,cfg,dump

def avg(rows,key):return float(np.mean([r[key] for r in rows]))
def main():
    main=json.loads((ROOT/'results/statistics.json').read_text());records=json.loads((ROOT/'results/rescored_records.json').read_text())
    manifest=json.loads((ROOT/'bank/manifest.json').read_text());full={};factorial={};diags={}
    for split in ['test','real']:
        groups=cfg()['real_datasets'] if split=='real' else ['all']
        full[split]={}
        for group in groups:
            full[split][group]={}
            for arm in cfg()['arms']:
                vals=[]
                for seed in cfg()['seeds']:
                    rows=json.loads((ROOT/'results'/f'{arm}_s{seed}'/f'{split}_evaluation.json').read_text())
                    vals += [r['all'] for r in rows if r['case']=='full' and (group=='all' or r['graph']==group)]
                full[split][group][arm]={k:avg(vals,k) for k in ['accuracy','nll','brier']}
    for label in ['homo','hetero']:
        for feature in ['smooth','rough']:
            rows=[r for r in records if r['split']=='test' and r['label_mode']==label and r['feature_mode']==feature]
            factorial[label+'_'+feature]={a:{k:avg([r for r in rows if r['arm']==a],k) for k in ['accuracy','nll','brier']} for a in cfg()['arms']}
    reconstruction=json.loads((ROOT/'results/reconstruction.json').read_text())
    mse={d:{a:avg([r for r in reconstruction if r['graph']==d and r['arm']==a],'missing_projected_mse') for a in cfg()['arms']} for d in cfg()['real_datasets']}
    assert max(r['observed_max_change'] for r in reconstruction)==0
    for arm in ['weighted','signed','sheaf']:
        vals=[]
        for seed in cfg()['seeds']:
            vals += [r['diagnostics'] for r in json.loads((ROOT/'results'/f'{arm}_s{seed}/test_evaluation.json').read_text()) if r['case']!='full']
        diags[arm]={k:avg(vals,k) for k in ['negative_transport_fraction','mean_abs_cross_stalk','observed_change']}
    controls=json.loads((ROOT/'results/controls.json').read_text());contrasts={}
    for ds in cfg()['real_datasets']:
        contrasts[ds]={}
        for arm in ['pcfi100','sheaf']:
            orig=[r for r in records if r['split']=='real' and r['graph']==ds and r['arm']==arm and r['seed']==11]
            contrasts[ds][arm]={'full_accuracy':avg(orig,'accuracy')}
            for control in ['no_features','shuffled_sources']:
                altered=[r['original_missing'] for r in controls if r['graph']==ds and r['arm']==arm and r['control']==control]
                contrasts[ds][arm][control+'_accuracy']=avg(altered,'accuracy')
                contrasts[ds][arm]['full_minus_'+control]=avg(orig,'accuracy')-avg(altered,'accuracy')
    native=json.loads((ROOT/'results/native_inference.json').read_text());native_table={}
    for ds in cfg()['real_datasets']:
        native_table[ds]={}
        for arm in ['pcfi200','weighted','signed','sheaf']:
            native_table[ds]['native_'+arm]={k:avg([r['missing'] for r in native if r['graph']==ds and r['arm']==arm],k) for k in ['accuracy','nll','brier']}
        for arm in ['pcfi100','native_pcfi100','sheaf','signed']:
            rr=[r for r in records if r['split']=='real' and r['graph']==ds and r['arm']==arm and r['seed']==11 and r['repetition']==0]
            native_table[ds][arm]={k:avg(rr,k) for k in ['accuracy','nll','brier']}
    source_counts={}
    for ds in cfg()['real_datasets']:
        source_counts[ds]={case:[r['observed_nodes'] for r in manifest['episodes'] if r['split']=='real' and r['graph']==ds and r['case']==case] for case in cfg()['masks']}
    dump(ROOT/'results/supplement.json',{'full_feature_reference':full,'synthetic_factorial':factorial,'numeric_reconstruction_rp16_mse':mse,
        'transport_diagnostics_synthetic_missing':diags,'feature_intervention_seed11':contrasts,'native_first_repetition_seed11':native_table,
        'real_sources_by_case':source_counts,'primary_gates_unchanged':main['gates'],
        'limits':'Native all-physical-channel inference is an extra first-repetition/seed11 diagnostic, not separately trained or a replacement for primary. All contrasts preserve original missing queries. No resampling empty-source episodes.'})
    print('SUPPLEMENT COMPLETE',flush=True)
if __name__=='__main__':main()

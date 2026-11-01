"""Graph-paired NLL kill test and exact-reference diagnostics."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
from experiment import C,ROOT,cfg,rows,read
from data import dump


def ci(x):
    x=np.asarray(x,float);rng=np.random.default_rng(2026110211);res=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'difference':float(x.mean()),'ci95':[float(v) for v in np.quantile(res,[.025,.975])],'graphs':len(x)}
def kl(a,b):
    a=np.clip(a,1e-10,1);b=np.clip(b,1e-10,1);a/=a.sum(-1,keepdims=True);b/=b.sum(-1,keepdims=True)
    return float((a*(np.log(a)-np.log(b))).sum(-1).mean())
def primary(r):return r['case']=='matched' and r['mechanism'] in ('uniform_nodes','clustered_nodes') and r['rate'] in (.5,.9)


def main():
    c=cfg();records=[];training=[];files=0;references={};baselines=[]
    for i,row in enumerate(rows('test')):
        inp,q,y=read(row['path'])
        with np.load(ROOT/'references'/(Path(row['path']).stem+'.npz')) as z:ref={k:z[k][q].astype(float) for k in z.files}
        references[row['path']]=ref
        n=len(inp['x']);a=inp['adj'].sum(0)+np.eye(n);a/=a.sum(1,keepdims=True)
        p=np.full((n,2),.5);known=inp['context']>=0;seeds=np.eye(2)[inp['context'][known]]
        for _ in range(40):p=a@p;p[known]=seeds
        prior=(np.bincount(inp['context'][known],minlength=2)+1)/(known.sum()+2)
        for arm,pr in list(ref.items())+[('lp',p[q]),('class_prior',np.tile(prior,(len(q),1))),('uniform',np.full((len(q),2),.5))]:
            baselines.append({**row,'arm':arm,'seed':0,'mode':'standard',**C.metrics(pr,y),
                'grid6_kl':kl(ref['grid6_exact'],pr),'true_parameter_kl':kl(ref['oracle_true_parameters'],pr)})
    for arm in c['arms']:
        for seed in c['seeds']:
            out=ROOT/'results'/f'{arm}_s{seed}';training.append(json.loads((out/'training.json').read_text()))
            for r in json.loads((out/'evaluation.json').read_text()):
                with np.load(ROOT/r['prediction']) as z:p=z['p'];y=z['y']
                score=C.metrics(p,y);assert all(abs(v-r[k])<1e-9 for k,v in score.items());files+=1
                ref=references[r['path']]
                # Reference KL has matching evidence only for standard predictions.
                if r['mode']=='standard':r.update(grid6_kl=kl(ref['grid6_exact'],p),true_parameter_kl=kl(ref['oracle_true_parameters'],p))
                records.append(r)
    records+=baselines
    metrics=['accuracy','nll','brier','grid6_kl','true_parameter_kl'];arms=c['arms']+['grid6_exact','grid6_bp','oracle_true_parameters','lp','class_prior','uniform']
    summaries=[]
    for arm in arms:
        for case in c['cases']:
            rr=[r for r in records if r['arm']==arm and r['mode']=='standard' and r['case']==case]
            for subset,selection in [('all',rr),('primary',[r for r in rr if primary(r)])]:
                if selection:summaries.append({'arm':arm,'case':case,'subset':subset,**{m:float(np.mean([r[m] for r in selection])) for m in metrics}})
    def vector(arm,metric,select=primary,mode='standard'):
        chosen=[r for r in records if r['arm']==arm and r['mode']==mode and select(r)]
        return {g:np.mean([r[metric] for r in chosen if r['graph']==g]) for g in sorted({r['graph'] for r in chosen})}
    paired={}
    for ref in [a for a in arms if a!='circuit']:
        for metric in metrics:
            a=vector('circuit',metric);b=vector(ref,metric);paired[f'circuit-{ref}/{metric}']=ci([a[g]-b[g] for g in a])
    for case in c['cases']:
        for ref in ['mask_pfn','direct_graph','grid6_bp','grid6_exact']:
            for metric in ['accuracy','nll','grid6_kl']:
                selector=lambda r,case=case:r['case']==case and r['mechanism'] in ('uniform_nodes','clustered_nodes') and r['rate'] in (.5,.9)
                a=vector('circuit',metric,selector);b=vector(ref,metric,selector)
                paired[f'{case}/circuit-{ref}/{metric}']=ci([a[g]-b[g] for g in a])
    interventions=[]
    for arm in c['arms']:
        for mode in ['no_observed','shuffled']:
            for metric in ['accuracy','nll','brier']:
                selector=lambda r:r['context_total']==2 and r['mechanism']=='clustered_nodes' and r['rate']==.5
                a=vector(arm,metric,selector,mode);b=vector(arm,metric,selector)
                interventions.append({'arm':arm,'mode':mode,'metric':metric,**ci([a[g]-b[g] for g in a])})
    mechanism=all(paired[f'circuit-{ref}/nll']['ci95'][1]<0 for ref in ['direct_graph','mask_pfn'])
    summary=next(r for r in summaries if r['arm']=='circuit' and r['subset']=='primary')
    bench=json.loads((ROOT/'results/benchmark.json').read_text());bb=bench['measurements']
    circuit_ms=np.median([r['median_ms'] for r in bb if r['arm']=='circuit']);grid_ms=next(r['median_ms'] for r in bb if r['arm']=='grid6_bp')
    competitive=mechanism and summary['grid6_kl']<=.01 and circuit_ms<=.5*grid_ms
    decision={'circuit_validity':json.loads((ROOT/'results/preflight.json').read_text())['pass_all'],'mechanism_gate':mechanism,'competitive_gate':bool(competitive),
        'matched_grid6_predictive_kl':summary['grid6_kl'],'circuit_median_ms':float(circuit_ms),'grid6_bp_median_ms':grid_ms,
        'primary':'NLL, matched case, four whole-node50/90 settings; contexts1/2/4 averaged within48 base graphs','fixed_seed_conditional_CI':True,
        'scope':'conditional tree circuit only; no global exact cyclic inference, variable schema, GSPN/MaM reproduction claim'}
    dump(ROOT/'results/summary.json',summaries);dump(ROOT/'results/paired.json',paired);dump(ROOT/'results/interventions.json',interventions);dump(ROOT/'results/baselines.json',baselines)
    dump(ROOT/'results/audit.json',{'prediction_files_rescored':files,'models':len(training),'successful_updates':sum(r['updates'] for r in training),'task_exposures':sum(r['examples'] for r in training),'all_frozen_preserved':all(r['frozen_weights_unchanged'] for r in training),'zero_observation_test_episodes':sum(r['observed_cells']==0 for r in rows('test'))})
    dump(ROOT/'results/decision.json',decision)
    lines=['# Direct marginal GraphPFN — valid conditional circuit kill test','',
        f"- Circuit validity: {decision['circuit_validity']}; mechanism gate: {mechanism}; competitive gate: {competitive}.",
        '- Newly implemented normalized graph-conditioned tree mixture; not a GSPN or MaM reproduction. Exactness holds inside the learned conditional tree model, not the original cyclic SBM.',
        '- Three fixed training seeds; 2048 task exposures and256 updates per model. Public Mask GraphPFN uses accumulated graphs and strong typed-context metadata. Architecture parameter counts differ and are reported.',
        '- Known-family exact Grid6 marginalizes parameters, roles, all hidden node labels and feature orientations. Grid6-BP uses the same coherent weights,64 damped iterations and observed-feature unary fields.',
        '- Primary: matched48 fresh base graphs, whole-node uniform/clustered50/90%, contexts1/2/4. No test selection.','',
        '| Method | Accuracy (%) | NLL | Brier | Grid6 predictive KL |','|---|---:|---:|---:|---:|']
    for arm in arms:
        r=next(v for v in summaries if v['arm']==arm and v['subset']=='primary')
        lines.append(f"| {arm} | {100*r['accuracy']:.4f} | {r['nll']:.6f} | {r['brier']:.6f} | {r['grid6_kl']:.6f} |")
    lines+=['','| NLL contrast | Difference | Graph95% CI |','|---|---:|---|']
    for ref in ['direct_graph','mask_pfn','grid6_bp','grid6_exact']:
        v=paired[f'circuit-{ref}/nll'];lines.append(f"| circuit − {ref} | {v['difference']:.6f} | [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    lines+=['','A valid marginalization operator is not evidence that the learned probability family fits the data. Gaussian features and the tree approximation are restrictive.',
        'Only the declared tiny schema was run. Full sparse observation tokens, continuous variable-schema MaM, and a globally valid cyclic graph circuit remain untested.',
        'See protocol.json and results/summary.json, paired.json, interventions.json, benchmark.json, audit.json for all secondary conditions and raw measurement scopes.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');dump(ROOT/'results/completed.json',decision);print(json.dumps(decision),flush=True)
if __name__=='__main__':main()

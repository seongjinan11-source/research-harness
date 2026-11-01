"""Preregistered graph-paired accuracy analysis; no model selection."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parent
from experiment import C,cfg,read,rows,write


def interval(x):
    x=np.asarray(x,np.float64); rng=np.random.default_rng(2026110207)
    draws=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'difference':float(x.mean()),'ci95':[float(v) for v in np.quantile(draws,[.025,.975])],'graphs':len(x)}
def group(row):
    fam=row['pair']; mech=row['mechanism'];return fam+'/'+mech
def primary(r):return r['pair'] in ('official','offdiagonal') and r['mechanism'] in ('uniform_nodes','clustered_nodes')


def main():
    c=cfg(); allrows=[]; audit=0; training=[]; costs=[]
    for arm in c['routing_arms']:
        for seed in c['seeds']:
            out=ROOT/'results'/f'{arm}_s{seed}'
            records=json.loads((out/'evaluation.json').read_text(encoding='utf8'))
            for r in records:
                with np.load(ROOT/r['prediction']) as z:
                    rescored=C.metrics(z['p'],z['y'])
                assert all(abs(r[k]-v)<1e-9 for k,v in rescored.items()); audit+=1
            allrows+=records;training.append(json.loads((out/'training.json').read_text()))
            costs.append({'arm':arm,'seed':seed,**json.loads((out/'timing.json').read_text())})
    assert len({r['initial_trainable_sha256'] for r in training if r['seed']==11})==1
    assert all(r['updates']==c['training_updates'] for r in training)
    # Recompute training-free baselines on identical context/query/graph inputs.
    baselines=[]; coverage=[]
    for row in rows():
        inp,y=read(row); n=len(inp.observed); a=C.BASE.adjacency(inp.src,inp.dst,n).astype(np.float64)
        degree=np.asarray(a.sum(1)).ravel(); p=np.ones((n,2))*.5
        seed=np.eye(2)[inp.labels_context.astype(int)]
        for _ in range(40):
            p=(a@p)/degree[:,None];p[inp.context]=seed
        prior=(np.bincount(inp.labels_context.astype(int),minlength=2)+1)/(len(inp.context)+2)
        for name,probs in [('lp',p[inp.query]),('uniform',np.ones((len(y),2))*.5),('class_prior',np.tile(prior,(len(y),1)))]:
            baselines.append({**row,'arm':name,'seed':0,'mode':'standard',**C.metrics(probs,y)})
        obs=inp.observed[:,:row['features']]; labeled=np.zeros(n,bool);labeled[inp.context]=True
        coverage.append({'path':row['path'],'observed_cells':int(obs.sum()),'observed_cells_outside_label_context':int(obs[~labeled].sum()),
            'zero_observed_cells':not bool(obs.any()),'real_features':row['features']})
    records=allrows+baselines
    summary=[]
    for arm in c['routing_arms']+['lp','uniform','class_prior']:
        for key in sorted({group(r) for r in records}):
            rr=[r for r in records if r['arm']==arm and r['mode']=='standard' and group(r)==key]
            if rr:summary.append({'arm':arm,'group':key,'episodes':len(rr),**{m:float(np.mean([r[m] for r in rr])) for m in ['accuracy','nll','brier']}})
    def paired(a,b,metric,selector=primary,mode='standard'):
        def vector(arm):
            return {g:np.mean([r[metric] for r in records if r['arm']==arm and r['mode']==mode and r['graph']==g and selector(r)]) for g in range(c['test_graphs'])}
        av,bv=vector(a),vector(b);return interval([av[g]-bv[g] for g in av])
    contrasts={}
    for ref in ['context','all_nodes','pcfi','lp','uniform','class_prior']:
        for metric in ['accuracy','nll','brier']:contrasts[f'observed-{ref}/{metric}']=paired('observed',ref,metric)
    for family in ['official','offdiagonal']:
        for mech in ['uniform_nodes','clustered_nodes']:
            for ref in ['context','all_nodes','pcfi','lp']:
                contrasts[f'{family}/{mech}/observed-{ref}/accuracy']=paired('observed',ref,'accuracy',lambda r,f=family,m=mech:r['pair']==f and r['mechanism']==m)
    interventions=[]
    for arm in c['routing_arms']:
        for fam in ['official','offdiagonal']:
            for mode in ['no_observed','shuffled']:
                for metric in ['accuracy','nll','brier']:
                    diff=[]
                    for g in range(c['test_graphs']):
                        rr=[r for r in allrows if r['arm']==arm and r['graph']==g and r['pair']==fam and r['mechanism']=='clustered_nodes' and r['rate']==.9]
                        diff.append(np.mean([r[metric] for r in rr if r['mode']==mode])-np.mean([r[metric] for r in rr if r['mode']=='standard']))
                    interventions.append({'arm':arm,'family':fam,'intervention':mode,'metric':metric,**interval(diff)})
    gate=lambda ref:contrasts[f'observed-{ref}/accuracy']['ci95'][0]>0
    decision={'mechanism_gate':gate('context') and gate('all_nodes'),'strong_method_gate':all(gate(x) for x in ['context','all_nodes','pcfi','lp']),
        'primary':'accuracy on official+offdiagonal, uniform+clustered, rates90/99.5 averaged within base graph',
        'no_test_selection':True,'fixed_training_seeds':c['seeds'],'novelty_not_established_by_gate':True}
    write(ROOT/'results/summary.json',summary);write(ROOT/'results/paired.json',contrasts);write(ROOT/'results/interventions.json',interventions)
    write(ROOT/'results/coverage.json',coverage);write(ROOT/'results/costs.json',costs);write(ROOT/'results/baselines.json',baselines)
    write(ROOT/'results/decision.json',decision);write(ROOT/'results/audit.json',{'prediction_files_rescored':audit,'trainings':len(training),'successful_updates':sum(r['updates'] for r in training),'all_frozen_parameters_unchanged':all(r['frozen_weights_unchanged'] for r in training)})
    lines=['# Feature-memory GraphPFN — matched pilot','',f"- Mechanism gate: {decision['mechanism_gate']}; strong-method gate: {decision['strong_method_gate']}.",
        '- Fresh 64 graph seeds; three fixed training seeds. Same declared families; real-graph/schema transfer untested.',
        '- Primary is accuracy. NLL/Brier remain separate outcomes. CIs resample base graphs, not individual queries or training seeds.',
        '- Observed memory is native two-coordinate feature-group memory with original two-bit mask. It is not a fully sparse cell-token architecture.','',
        '| Model | Primary accuracy (%) | NLL | Brier |','|---|---:|---:|---:|']
    for arm in c['routing_arms']+['lp','uniform','class_prior']:
        rr=[r for r in records if r['arm']==arm and r['mode']=='standard' and primary(r)]
        lines.append(f"| {arm} | {100*np.mean([r['accuracy'] for r in rr]):.4f} | {np.mean([r['nll'] for r in rr]):.6f} | {np.mean([r['brier'] for r in rr]):.6f} |")
    lines+=['','| Accuracy contrast | Difference (%p) | Paired graph 95% CI (%p) |','|---|---:|---|']
    for ref in ['context','all_nodes','pcfi','lp']:
        v=contrasts[f'observed-{ref}/accuracy'];lines.append(f"| observed − {ref} | {100*v['difference']:.4f} | [{100*v['ci95'][0]:.4f}, {100*v['ci95'][1]:.4f}] |")
    lines+=['','The context-only key/value restriction is verified in the source; this run does not establish it as the cause of earlier failed experiments.',
        'All-node feature attention is a direct prior-work control. Observed-only must improve it to establish an observation-specific effect.',
        'Details: protocol.json, results/paired.json, results/interventions.json, results/costs.json, results/audit.json. No automatic epoch/loss rescue.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');write(ROOT/'results/completed.json',decision)
    print(json.dumps(decision),flush=True)
if __name__=='__main__':main()

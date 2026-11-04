"""Independent saved-probability rescore and stratified graph bootstrap."""
from collections import defaultdict
import json
from pathlib import Path
import hashlib
import sklearn
import numpy as np
from util import ROOT, config, dump, sha, sources, metrics, array_hash


KEYS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'nll', 'brier']


def independent_metrics(p, y):
    # Separate implementation from the runner; class metrics include all K for
    # macro-F1 and only truth-present classes for balanced recall.
    p = np.maximum(np.minimum(p.astype('float64'), 1), 1e-8)
    p = p / p.sum(axis=1, keepdims=True)
    pred = np.argmax(p, axis=1)
    recalls, f1s = [], []
    for c in range(p.shape[1]):
        truth = y == c
        guess = pred == c
        tp = int(np.sum(truth & guess))
        if truth.any():
            recalls.append(tp/int(truth.sum()))
        den = int(truth.sum()+guess.sum())
        f1s.append(2*tp/den if den else 0.)
    brier = np.sum(p*p, axis=1)-2*p[np.arange(len(y)),y]+1
    return dict(accuracy=float(np.sum(pred==y)/len(y)), balanced_accuracy=float(np.mean(recalls)),
                macro_f1=float(np.mean(f1s)), nll=float(np.mean(-np.log(p[np.arange(len(y)),y]))),
                brier=float(np.mean(brier)))


def interval(values, strata, seed):
    values = np.asarray(values, float)
    if not len(values):
        return dict(n_graphs=0, mean=None, ci=None, verdict='undefined')
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    rng = np.random.default_rng(seed)
    draws = np.zeros(config()['statistics']['bootstrap'])
    for ids in groups:
        samples = rng.choice(ids, size=(len(draws), len(ids)), replace=True)
        draws += values[samples].sum(1)/len(values)
    tail = (1-config()['statistics']['ci'])/2
    low, high = np.quantile(draws, [tail, 1-tail])
    return dict(n_graphs=len(values), n_strata=len(groups), mean=float(values.mean()),
                ci=[float(low),float(high)], confidence=config()['statistics']['ci'],
                unit='base graph; fixed seeds/settings averaged first; resample within strata',
                verdict='PASS' if low > 0 else 'FAIL')


def main():
    from data import rows, read, external_records
    cfg = config()
    scientific_lock = json.loads((ROOT/'results/lock.json').read_text(encoding='utf8'))
    assert scientific_lock['sources'] == sources(), 'Source differs from scientific lock'
    assert sha(ROOT/'results/checks.json')==scientific_lock['checks']
    assert sha(ROOT/'results/preflight.json')==scientific_lock['preflight']
    records = []
    training = []
    files = []
    all_sets = {'final':rows('final'), 'real':rows('real')}
    for split, meta in scientific_lock['banks'].items():
        rr=rows(split)
        assert len(rr)==meta['count'] and sha(ROOT/'bank'/split/'manifest.json')==meta['manifest']
        assert all(sha(ROOT/r['path'])==r['sha256'] for r in rr)
    data_index = {r['path']:r for split in all_sets.values() for r in split}
    canonical = {}
    base_identity={}
    preprocess=[]
    eval_cost=[]
    max_error = 0.
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            out = ROOT/'results'/f'{arm}_s{seed}'
            tr = json.loads((out/'training.json').read_text(encoding='utf8'))
            assert tr['updates'] == cfg['train']['updates']
            assert tr['presentations'] == cfg['train']['updates']*cfg['train']['batch']
            assert tr['total_parameters'] == tr['trainable_parameters']
            assert sha(out/'checkpoint.pt') == tr['checkpoint_sha256']
            schedule=json.loads((out/'schedule.json').read_text(encoding='utf8'))
            expected=np.random.default_rng(cfg['train']['schedule_seed']+seed).integers(
                scientific_lock['banks']['train']['count'],size=(cfg['train']['updates'],cfg['train']['batch']))
            assert np.array_equal(np.asarray(schedule['indices'],np.int64),expected)
            assert array_hash(expected)==schedule['sha256']==tr['schedule_sha256']
            import torch
            checkpoint=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
            assert checkpoint['initial']==tr['initial_sha256'] and checkpoint['lock_sha256']==sha(ROOT/'results/lock.json')
            state_digest=hashlib.sha256()
            for name,value in sorted(checkpoint['state_dict'].items()):
                state_digest.update(name.encode());state_digest.update(value.contiguous().numpy().tobytes())
            assert state_digest.hexdigest()==tr['final_sha256']
            training.append(tr)
            for split in ['final','real']:
                meta = json.loads((out/f'{split}_evaluation.json').read_text(encoding='utf8'))
                assert len(meta['records']) == len(all_sets[split])
                assert len({r['path'] for r in meta['records']}) == len(all_sets[split])
                assert {r['path'] for r in meta['records']} == {r['path'] for r in all_sets[split]}
                eval_cost.append(dict(arm=arm,seed=seed,split=split,seconds=meta['seconds'],
                    model_forward_seconds=sum(r['timing']['inference_seconds'] for r in meta['records']),
                    additional_peak_mib=max(r['timing']['additional_peak_mib'] for r in meta['records'])))
                for rec in meta['records']:
                    row = data_index[rec['path']]
                    if rec['path'] not in canonical:
                        inp, query, truth = read(row)
                        canonical[rec['path']] = (query, truth, inp['context'])
                        identity=(query,truth,inp['context'])
                        key=(split,str(row['base_id']))
                        if key in base_identity:
                            assert all(np.array_equal(a,b) for a,b in zip(identity,base_identity[key]))
                        else:
                            base_identity[key]=identity
                        if split=='real':
                            preprocess.append(dict(graph=row['graph'],repetition=row['repetition'],mode=row['mode'],
                                native_features=inp['x'].shape[1],pcfi_raw_maxabs=float(np.abs(inp['pcfi']).max()),
                                represented_clip_fraction=float(np.mean(np.abs(inp['pcfi'])>10)),
                                raw_pcfi_cache_unchanged=True))
                    query, truth, ctx = canonical[rec['path']]
                    path = ROOT/rec['prediction']
                    assert sha(path) == rec['prediction_sha256']
                    association=dict(arm=arm,seed=seed,input_sha256=row['sha256'],
                        checkpoint_sha256=tr['checkpoint_sha256'],lock_sha256=sha(ROOT/'results/lock.json'))
                    assert rec['association']==association
                    sidecar=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
                    assert sidecar['association']==association and sidecar['prediction_sha256']==sha(path)
                    with np.load(path) as z:
                        prob = z['p']
                        assert np.array_equal(z['query'], query) and np.array_equal(z['y'], truth)
                        assert np.array_equal(z['context'],ctx)
                    assert prob.shape == (len(truth),int(row['classes'])) and np.isfinite(prob).all()
                    assert prob.min() >= 0 and abs(prob.sum(1)-1).max() < 1e-5
                    m = independent_metrics(prob, truth)
                    for key in KEYS:
                        max_error = max(max_error,abs(m[key]-rec['metrics'][key]))
                    records.append({**row,'arm':arm,'seed':seed,'metrics':m})
                    files.append(dict(path=rec['prediction'],sha256=sha(path),bytes=path.stat().st_size))
    assert max_error < 1e-9, max_error
    assert len({t['trainable_parameters'] for t in training}) == 1
    for seed in cfg['seeds']:
        tt = [t for t in training if t['seed']==seed]
        assert len({t['schedule_sha256'] for t in tt}) == 1
        assert len({t['initial_sha256'] for t in tt}) == 1
    final = [r for r in records if r['split']=='final']
    index = {(r['arm'],r['seed'],str(r['base_id']),r['setting'],r['variant']):r for r in final}
    base_rows = {}
    for r in all_sets['final']:
        base_rows.setdefault(str(r['base_id']),r)
    assert len(base_rows) == cfg['final']['graphs']
    base_ids = sorted(base_rows)
    strata = [base_rows[b]['stratum'] for b in base_ids]
    def value(arm,seed,base,setting,variant='standard',key='accuracy'):
        return index[arm,seed,base,setting,variant]['metrics'][key]
    def graph_value(arm,base,key='accuracy'):
        return np.mean([value(arm,s,base,m,key=key) for s in cfg['seeds'] for m in cfg['primary_settings']])
    gates = {}
    for j, other in enumerate(['mask','pcfi']):
        vals = [graph_value('observed',b)-graph_value(other,b) for b in base_ids]
        gates['observed_minus_'+other] = interval(vals,strata,cfg['statistics']['seed']+j)
    utility_ids, utility_strata, eligible_rates = [], [], {}
    for b in base_ids:
        if not base_rows[b]['disjoint']:
            continue
        eligible = [m for m in cfg['primary_settings'] if index['observed',cfg['seeds'][0],b,m,'standard']['sources']>=2]
        eligible_rates[b] = eligible
        if eligible:
            utility_ids.append(b)
            utility_strata.append(base_rows[b]['stratum'])
    for j, other in enumerate(['none','shuffle']):
        vals = []
        for b in utility_ids:
            changes = []
            for seed in cfg['seeds']:
                for m in eligible_rates[b]:
                    ref = value('observed',seed,b,'none') if other=='none' else value('observed',seed,b,m,'shuffle')
                    changes.append(value('observed',seed,b,m)-ref)
            vals.append(float(np.mean(changes)))
        gates['observed_minus_'+other] = interval(vals,utility_strata,cfg['statistics']['seed']+2+j)
        if set(utility_strata) != set(strata):
            gates['observed_minus_'+other]['missing_strata'] = sorted(set(strata)-set(utility_strata))
    primary = []
    for arm in cfg['arms']:
        primary.append(dict(arm=arm,**{key:float(np.mean([graph_value(arm,b,key) for b in base_ids])) for key in KEYS}))
    breakdown = []
    for label, selected in [('family',sorted({r['family'] for r in base_rows.values()})),
                            ('shape',sorted({r['shape'] for r in base_rows.values()}))]:
        for group in selected:
            subset = [b for b in base_ids if base_rows[b][label]==group]
            for arm in cfg['arms']:
                breakdown.append(dict(axis=label,group=group,arm=arm,n_graphs=len(subset),
                    **{key:float(np.mean([graph_value(arm,b,key) for b in subset])) for key in KEYS}))
    support = defaultdict(lambda:defaultdict(int))
    for row in all_sets['final']:
        if row['setting'] in cfg['primary_settings'] and row['variant']=='standard':
            support[row['setting']][str(min(row['sources'],2))] += 1
    support_metrics=[]
    for bucket in [0,1,2]:
        rr=[r for r in final if r['setting'] in cfg['primary_settings'] and r['variant']=='standard' and min(r['sources'],2)==bucket]
        for arm in cfg['arms']:
            aa=[r for r in rr if r['arm']==arm]
            if aa:
                support_metrics.append(dict(source_bucket='2plus' if bucket==2 else str(bucket),arm=arm,
                    graph_rate_cases=len(aa)//len(cfg['seeds']),
                    scope='descriptive case-weighted within source bucket; not primary graph-weighted estimator',
                    **{k:float(np.mean([r['metrics'][k] for r in aa])) for k in KEYS}))
    real_rows = [r for r in records if r['split']=='real']
    real_summary = []
    for graph in cfg['real']['graphs']:
        for mode in cfg['real']['modes']:
            for arm in cfg['arms']:
                rr = [r for r in real_rows if r.get('graph')==graph and r.get('mode',r['setting'])==mode and r['arm']==arm]
                assert len(rr)==len(cfg['seeds'])*len(cfg['real']['repetitions']), (graph,mode,arm,len(rr))
                real_summary.append(dict(graph=graph,mode=mode,arm=arm,n_records=len(rr),
                    **{key:float(np.mean([r['metrics'][key] for r in rr])) for key in KEYS},
                    accuracy_min=float(min(r['metrics']['accuracy'] for r in rr)),
                    accuracy_max=float(max(r['metrics']['accuracy'] for r in rr))))
    external=[]
    real_index={(r['graph'],r['repetition'],r['mode']):r for r in all_sets['real']}
    for ref in external_records():
        assert ref['identity_pass'] and ref['native_reconstruction_verified']
        assert sha(ref['path'])==ref['sha256'] and sha(ref['sidecar_path'])==ref['sidecar_sha256']
        assert sha(ref['legacy_bank_path'])==ref['legacy_bank_sha256']
        row=real_index[ref['graph'],ref['repetition'],ref['mode']]
        query,truth,context=canonical[row['path']]
        with np.load(ref['path']) as z:
            assert np.array_equal(z['query'],query) and np.array_equal(z['y'],truth)
            assert np.array_equal(np.sort(z['context']),np.flatnonzero(context>=0))
            assert np.array_equal(context[z['context']],z['yc'])
            checked=independent_metrics(z['p'],z['y'])
        assert max(abs(checked[k]-ref['metrics'][k]) for k in KEYS)<1e-9
        external.append({**ref,'metrics':checked})
    external_summary=[]
    external_arms=sorted({r['arm'] for r in external})
    for graph in cfg['real']['graphs']:
        for mode in cfg['real']['modes']:
            for arm in external_arms:
                rr=[r for r in external if r['graph']==graph and r['mode']==mode and r['arm']==arm]
                if rr:
                    external_summary.append(dict(graph=graph,mode=mode,arm=arm,n_records=len(rr),
                        **{k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS},
                        scope='fixed pretrained/reference model,3 context repetitions; different preprocessing/budget'))
    summary = dict(cycle='FP04',gate='PASS' if all(g['verdict']=='PASS' for g in gates.values()) else 'FAIL',
                   gates=gates, primary=primary, breakdown=breakdown, real=real_summary,external=external_summary,
                   support_0_1_2plus=dict(support),support_metrics=support_metrics, utility=dict(planned_disjoint_graphs=sum(bool(r['disjoint']) for r in base_rows.values()),
                   eligible_graphs=len(utility_ids), eligible_cases=sum(map(len,eligible_rates.values())),
                   rates_by_base=eligible_rates),
                   scope='96 fresh synthetic base graphs; reused real development, 3 train seeds/3 context repetitions; no target updates')
    budget = dict(fits=len(training),updates=sum(t['updates'] for t in training),
                  task_presentations=sum(t['presentations'] for t in training),
                  train_seconds=sum(t['seconds'] for t in training),
                  trainable_parameters=training[0]['trainable_parameters'],
                  train_peak_mib=max(t['peak_mib'] for t in training),training=training,evaluation=eval_cost,
                  preparation={s:{k:v for k,v in json.loads((ROOT/'bank'/s/'manifest.json').read_text(encoding='utf8')).items()
                                   if k in ['prepare_wall_seconds','prepare_cpu_seconds','pcfi_wall_seconds','native_raw_load_seconds']}
                               for s in scientific_lock['banks']},
                  native_representation=preprocess)
    dump(ROOT/'results/summary.json',summary)
    dump(ROOT/'results/paired.json',gates)
    dump(ROOT/'results/budget.json',budget)
    dump(ROOT/'results/audit.json',dict(status='PASS',prediction_records=len(records),final_records=len(final),
         real_records=len(real_rows),max_rescore_error=max_error,matched_initialization_and_schedule=True,
         independent_external_records=len(external),
         no_frozen_trainable_parameters=True,source_lock_sha256=sha(ROOT/'results/lock.json'),
         original_sources=sources(),utility_eligible_graphs=len(utility_ids)))
    dump(ROOT/'results/artifact_manifest.json',dict(files=files,checkpoints=[dict(path=f"results/{t['arm']}_s{t['seed']}/checkpoint.pt",sha256=t['checkpoint_sha256']) for t in training]))
    text = ['# FP04: End-to-end observed-cell PFN','',f"감사 PASS; 사전 복합 gate {summary['gate']}. 새9 fits/3 seeds, query CE만 사용, 실제 그래프 업데이트0회.",'',
            '## 합성 primary: 자기 피처가 없는 query, 95/99.5% whole-node missing','',
            '| Arm | Accuracy % | BAcc % | NLL | Brier |','|---|---:|---:|---:|---:|']
    for r in primary:
        text.append(f"| {r['arm']} | {100*r['accuracy']:.3f} | {100*r['balanced_accuracy']:.3f} | {r['nll']:.5f} | {r['brier']:.5f} |")
    text += ['', '| 비교 | Accuracy 차이 %p | 사전97.5% CI %p | Graph 수 | Gate |','|---|---:|---:|---:|---|']
    for key,g in gates.items():
        ci='undefined' if g['ci'] is None else f"[{100*g['ci'][0]:+.3f}, {100*g['ci'][1]:+.3f}]"
        point='undefined' if g['mean'] is None else f"{100*g['mean']:+.3f}"
        text.append(f"| {key} | {point} | {ci} | {g['n_graphs']} | {g['verdict']} |")
    text += ['',f"Primary는96개 base graph에서 고정3 train seeds와2 rates를 먼저 평균했다. Utility는 disjoint48개 예정 graph 중 source>=2인 {len(utility_ids)} graphs/{sum(map(len,eligible_rates.values()))} cases의 별도 조건부 비교다.",
             '', '## 실제 그래프 개발 평가','',
             '각 셀은3 training seeds×3 고정 context/mask 반복 평균이다. Cora/CiteSeer 전체, PubMed 기존2000-node induced graph. 모두 이전에 검토한 데이터이며 독립 새 graph 일반화 CI가 아니다.', '',
             '| Graph | Mode | Observed acc % | Mask acc % | PCFI acc % |','|---|---|---:|---:|']
    for graph in cfg['real']['graphs']:
        for mode in cfg['real']['modes']:
            rr={r['arm']:r for r in real_summary if r['graph']==graph and r['mode']==mode}
            text.append('| '+graph+' | '+mode+' | '+' | '.join(f"{100*rr[a]['accuracy']:.3f}" for a in cfg['arms'])+' |')
    text += ['', '## 외부 강한 기준선: 정확히 같은 context/query/mask의3반복 평균', '',
             '| Graph | Mode | Observed % | Public PFN % | PCFI+Public PFN % | FISF+Public PFN % | Restart LP % | Harmonic LP % |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for graph in cfg['real']['graphs']:
        for mode in cfg['real']['modes']:
            vals={r['arm']:r['accuracy'] for r in external_summary if r['graph']==graph and r['mode']==mode}
            obs=next(r['accuracy'] for r in real_summary if r['graph']==graph and r['mode']==mode and r['arm']=='observed')
            names=['public_graphpfn','pcfi_public_graphpfn','fisf_public_graphpfn','lp_restart','lp_harmonic']
            text.append('| '+graph+' | '+mode+' | '+f'{100*obs:.3f}'+' | '+' | '.join(f'{100*vals[a]:.3f}' if a in vals else 'missing' for a in names)+' |')
    text += ['', '외부 모델과 새 모델은 물리적 관측값·graph/context/query/truth가 같지만 전처리·구조·학습 예산은 다르다. 새 모델은 native 전체 column/RMS, public 모델은 원래 preprocessing/PCA/ensemble을 사용한다. 같은 예산의 세 arm 대조와 외부 pretrained 기준선의 비교를 혼합해 인과 해석하지 않는다.',
             '', '관측값 이용은 none/shuffle 비교로 별도 판정하며, 단순히 피처 없이 유지되는 성능을 missing-feature inference 성공으로 부르지 않는다. Source0/1과 실패한 family/shape/real 데이터도 제거하지 않는다.']
    (ROOT/'results/REPORT_KO.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(dict(gate=summary['gate'],gates=gates,primary=primary,audited=len(records)),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

"""Prespecified graph-paired analysis; no model selection or additional training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiment as E
import argparse
import json
import numpy as np


def lock_analysis():
    path=ROOT/'results/analysis_lock.json'
    state={'analysis_sha256':E.C.sha(__file__),'protocol_sha256':E.C.sha(ROOT/'protocol.json'),
        'sanity_predictors':['uniform','laplace_context_prior','context_clamped_LP_40steps'],
        'primary_gates':'Exact Gate A/B in protocol.json; sanity comparisons do not replace the registered contrasts.',
        'scope':'New synthetic graph seeds; controlled coupled/crossed isolates a constructed finite-bank dependence, not the failure mechanism of the official prior.'}
    if path.exists():assert json.loads(path.read_text())==state
    else:E.write(path,state)
    return state


def grouped(rows,name):
    family,pair,kind=name.split(':')
    selected=[r for r in rows if r['family']==family and (pair=='any' or r['pair']==pair)]
    if kind=='node':return [r for r in selected if r['mechanism'] in ['uniform_nodes','clustered_nodes']]
    if kind=='uniform':return [r for r in selected if r['mechanism']=='uniform_nodes']
    if kind=='clustered':return [r for r in selected if r['mechanism']=='clustered_nodes']
    if kind=='full':return [r for r in selected if r['mechanism']=='full']
    if kind=='cell':return [r for r in selected if r['mechanism']=='uniform_cells']
    raise ValueError(name)


def sanity():
    records=[]
    for row in E.rowsets():
        inp,y=E.read(row);classes=int(inp.classes);n=len(inp.observed)
        counts=np.bincount(inp.labels_context,minlength=classes)+1
        prior=counts/counts.sum();p=np.broadcast_to(prior,(n,classes)).copy()
        a=E.C.BASE.adjacency(inp.src,inp.dst,n);transition=E.C.sp.diags(1/np.asarray(a.sum(1)).ravel())@a
        p[inp.context]=np.eye(classes)[inp.labels_context]
        for _ in range(40):
            p=.9*(transition@p)+.1*prior;p[inp.context]=np.eye(classes)[inp.labels_context]
        lp=.99*p[inp.query]+.01*prior
        for arm,prob in [('context_prior',np.broadcast_to(prior,(len(y),classes))),
                ('uniform_prior',np.full((len(y),classes),1/classes)),('LP',lp)]:
            records.append({**row,'arm':arm,'seed':-1,'mode':'standard',**E.C.metrics(prob,y)})
    return records


def analyze():
    analysis_lock=lock_analysis();E.lock();c=E.cfg();rows=[];train=[]
    for seed in c['seeds']:
        for arm in c['arms']:
            out=ROOT/'results'/f'{arm}_s{seed}';record=json.loads((out/'training.json').read_text())
            logs=json.loads((out/'train_log.json').read_text())
            assert record['updates']==c['updates'] and [r['step'] for r in logs]==list(range(1,c['updates']+1))
            assert E.C.sha(out/'trainable.pt')==record['checkpoint_sha256'] and record['frozen_weights_unchanged']
            train.append(record);rows+=json.loads((out/'evaluation.json').read_text())
        same=[t for t in train if t['seed']==seed]
        for key in ['schedule_sha256','initial_trainable_sha256','frozen_sha256','trainable_parameters']:
            assert len({t[key] for t in same})==1,key
    for row in rows:
        with np.load(ROOT/row['prediction']) as z:
            measured=E.C.metrics(z['p'],z['y']);assert np.isfinite(z['p']).all()
        assert all(abs(measured[k]-row[k])<1e-10 for k in measured)
    sanity_rows=sanity();E.write(ROOT/'results/sanity.json',sanity_rows)
    groups=[f'{family}:{pair}:{kind}' for family,pair in [('official','any'),('controlled','diagonal'),('controlled','offdiagonal')]
        for kind in ['uniform','clustered','full','cell','node']]
    summary=[];paired=[];boot=np.random.default_rng(2026110204).integers(c['test_graphs'],size=(5000,c['test_graphs']))
    def contrast(subset,left,right,mode='standard',name=None):
        for metric in ['nll','accuracy','brier']:
            def values(arm,which):
                return np.array([np.mean([r[metric] for r in subset if r['arm']==arm and r['mode']==which and r['graph']==g]) for g in range(c['test_graphs'])])
            delta=values(left,mode)-values(right,'standard');lo,hi=np.quantile(delta[boot].mean(1),[.025,.975])
            paired.append({'group':name,'contrast':f'{left}:{mode} - {right}:standard','metric':metric,
                'delta':float(delta.mean()),'low':float(lo),'high':float(hi)})
    allrows=rows+sanity_rows
    for group in groups:
        subset=grouped(allrows,group)
        for arm in c['arms']+['uniform_prior','context_prior','LP']:
            for mode in sorted({r['mode'] for r in subset if r['arm']==arm}):
                selected=[r for r in subset if r['arm']==arm and r['mode']==mode]
                summary.append({'group':group,'arm':arm,'mode':mode,**{k:float(np.mean([r[k] for r in selected])) for k in ['nll','accuracy','brier']},
                    'per_seed_nll':{str(s):float(np.mean([r['nll'] for r in selected if r['seed']==s])) for s in sorted({r['seed'] for r in selected})}})
        for other in ['official_mask','coupled_mask','coupled_pcfi','uniform_prior','context_prior','LP']:
            contrast(subset,'crossed_mask',other,name=group)
    interventions=[]
    for group in groups:
        subset=grouped(rows,group)
        for arm in c['arms']:
            for mode in ['no_observed','shuffled']:
                altered=[r for r in subset if r['arm']==arm and r['mode']==mode]
                if not altered:continue
                allowed={(r['seed'],r['path']) for r in altered}
                original=[r for r in subset if r['arm']==arm and r['mode']=='standard' and (r['seed'],r['path']) in allowed]
                for metric in ['nll','accuracy','brier']:
                    # NLL: intervention minus full, so positive implies benefit from actual values.
                    diff=np.array([np.mean([r[metric] for r in altered if r['graph']==g])-np.mean([r[metric] for r in original if r['graph']==g]) for g in range(c['test_graphs'])])
                    lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
                    interventions.append({'group':group,'arm':arm,'contrast':mode+' - standard, paired episode subset',
                        'metric':metric,'delta':float(diff.mean()),'low':float(lo),'high':float(hi),'rate':.9})
    sampling=[]
    for kind in ['uniform_nodes','clustered_nodes']:
        subset=[r for r in rows if r['family']=='official' and r['mechanism']==kind and r['rate']==.995]
        for arm in ['official_mask','crossed_mask']:
            for other in ['standard','zero4']:
                for metric in ['nll','accuracy','brier']:
                    diff=np.array([np.mean([r[metric] for r in subset if r['arm']==arm and r['mode']=='random4' and r['graph']==g])-np.mean([r[metric] for r in subset if r['arm']==arm and r['mode']==other and r['graph']==g]) for g in range(c['test_graphs'])])
                    lo,hi=np.quantile(diff[boot].mean(1),[.025,.975])
                    sampling.append({'group':kind,'arm':arm,'contrast':'random4 - '+other,'metric':metric,
                        'delta':float(diff.mean()),'low':float(lo),'high':float(hi)})
    def comparison(group,other,metric):return next(r for r in paired if r['group']==group and r['contrast']==f'crossed_mask:standard - {other}:standard' and r['metric']==metric)
    gate_a=all(comparison(f'controlled:offdiagonal:{kind}',other,'nll')['high']<0 for kind in ['uniform','clustered'] for other in ['coupled_mask','coupled_pcfi'])
    gate_b=all(comparison(f'official:any:{kind}','official_mask','nll')['high']<0 for kind in ['uniform','clustered']) and comparison('official:any:node','official_mask','accuracy')['low']>0
    decision={'gate_a_controlled_offdiagonal':gate_a,'gate_b_official_transfer':gate_b,'primary_passed':gate_a and gate_b,
        'scope':'No real graph or unseen semantic schema tested. Coupled/crossed matched marginals do not establish the original official prior failure cause.'}
    E.write(ROOT/'results/summary.json',summary);E.write(ROOT/'results/paired.json',paired)
    E.write(ROOT/'results/interventions.json',interventions);E.write(ROOT/'results/sampling.json',sampling)
    E.write(ROOT/'results/decision.json',decision)
    E.write(ROOT/'results/audit.json',{'checkpoints':len(train),'successful_updates':sum(t['updates'] for t in train),
        'probability_files_rescored':len(rows),'same_initialization_and_schedule_per_seed':True,
        'frozen_weights_unchanged':True,'analysis_lock':analysis_lock,'sources':E.sources()})
    lines=['# Mechanism-crossed GraphPFN — controlled prior pilot','',
        f"2026-11-02. 실제 학습 완료: {len(train)} checkpoints × {c['updates']} successful updates. Gate A={gate_a}, Gate B={gate_b}.",
        '', '모델은 기존 공개 GraphPFN Mask Plain 그대로다. Official Mask, Coupled Mask, Crossed Mask, Coupled PCFI를 같은 초기화·update 수로 학습했다. 추가 posterior head·refiner·auxiliary loss는 없다.',
        '', '## 주요 결과','','| Test family / 조건 | 모델 | NLL ↓ | Accuracy % | Brier ↓ |','|---|---|---:|---:|---:|']
    for group in ['controlled:offdiagonal:uniform','controlled:offdiagonal:clustered','official:any:uniform','official:any:clustered']:
        for r in summary:
            if r['group']==group and r['mode']=='standard':lines.append(f"| {group} | {r['arm']} | {r['nll']:.5f} | {100*r['accuracy']:.2f} | {r['brier']:.5f} |")
    lines+=['','## Crossed의 paired NLL 차이','','| 조건 | 대조군 | Δ NLL [95% CI] |','|---|---|---:|']
    for r in paired:
        if r['group'] in ['controlled:offdiagonal:uniform','controlled:offdiagonal:clustered','official:any:uniform','official:any:clustered'] and r['metric']=='nll':
            lines.append(f"| {r['group']} | {r['contrast']} | {r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}] |")
    lines+=['','## 관측 피처 개입 (90% whole-node, 고정 checkpoint)','','| 조건 | Crossed 개입 | Δ NLL intervention−standard [95% CI] |','|---|---|---:|']
    for r in interventions:
        if r['arm']=='crossed_mask' and r['metric']=='nll' and r['group'].endswith((':uniform',':clustered')):
            lines.append(f"| {r['group']} | {r['contrast']} | {r['delta']:+.5f} [{r['low']:+.5f}, {r['high']:+.5f}] |")
    lines+=['','## 정보량·실행 범위','',
        'Controlled train은 64 base graphs × 4 styles다. Coupled와 Crossed의 concrete feature field·rule instance 주변 exposure를 정확히 맞췄다. Crossed는 style×rule×mask 64조합이 균등하다. One graph/update이므로 pair loss·paired minibatch 효과는 없다.',
        '', 'Feature law는 graph Gaussian field의 4개 변환이며 label rule은 self linear/neighbor linear/product/two-hop contrast다. Label rule 함수는 style ID를 받지 않고, sample quantile threshold나 class-balance rejection을 쓰지 않는다. Context는 별도로 클래스당 최대4개를 고른다. 전체 realized graph가 single-class여도 binary output을 유지했다.',
        '', 'Gaussian/linear subset의 관측 sign-only 진단은 알려진 rule을 받는 privileged reference다. 실제 RMS 입력이 보존하는 첫 관측 sign만 사용하고 unknown magnitudes를 적분한다. 전체 context posterior나 실제 배포 predictor가 아니다. 정보량이 적은 과제와 zero-observation 과제는 제거하지 않았다.',
        '', 'Test32 fresh base graphs의 controlled diagonal/off-diagonal과 official native tasks를 동일 입력으로 평가했다. Style/rule 종류는 학습에서 본 것이며 새로운 graph/function seed다. 실제 graph·미지 semantic schema 성과가 아니다.',
        '', 'Official vs Crossed는 feature/label 함수족까지 다르므로 coupling만의 효과를 분리하지 못한다. Coupled vs Crossed가 통제된 비교다. Constructed biased prior를 고쳤다는 결과를 기존 공식 prior 실패 원인의 확인으로 확대하지 않는다.',
        '', 'Bootstrap은 base graph32개를 paired resampling5000회 했고 rates·고정 training seeds3개를 먼저 평균했다. 동일 base graph의 tasks를 독립 표본으로 세지 않았다. Training-seed 모집단 CI나 다중 비교 보정은 없다. 마지막 checkpoint만 사용했다.',
        '', 'PCFI 기준선도 같은 update 수로 해당 입력을 학습한 모델이다. FISF-trained 대조와 real graph 비교는 이 pilot에 없다. 원래 공개 모델의 공식 preprocessing 재현이 아닌 공통 RMS+mask interface다.',
        '', 'Sampling 대조는 사전 지정 official99.5% 조건의 4-view zero와 iid-unit noise다. Primary predictor는 1-view이므로 호출 수를 구분하며, 병렬 실행 시간을 방법의 latency 우위로 주장하지 않는다.',
        '', f"모든 {len(rows)} saved probability outputs를 독립 재채점했다. Frozen original transformer 보존과 같은 seed의 initialization·update schedule·trainable parameter 수를 검증했다. 성공 optimizer updates 합계={sum(t['updates'] for t in train)}."]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('DECISION',decision,flush=True)
    for r in summary:
        if r['group'] in ['controlled:offdiagonal:uniform','controlled:offdiagonal:clustered','official:any:uniform','official:any:clustered'] and r['mode']=='standard':
            print(r['group'],r['arm'],round(r['nll'],6),round(r['accuracy'],6),flush=True)
    E.write(ROOT/'results/completed.json',{'decision':decision,'report_sha256':E.C.sha(ROOT/'REPORT.md')})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',action='store_true');args=ap.parse_args()
    if args.lock:lock_analysis();print('ANALYSIS_LOCKED')
    else:analyze()

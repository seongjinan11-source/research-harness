"""Render prespecified results; context-frequency controls remain secondary."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from pcfi_shared import C,rows,read,dump
from pcfi_analysis import collect,difference,METRICS
stats=json.loads((root/'results/statistics.json').read_text(encoding='utf8'))
cost=json.loads((root/'results/cost.json').read_text(encoding='utf8'))
audit=json.loads((root/'results/audit.json').read_text(encoding='utf8'))
test,data=collect();base={};context=[];other={a:[] for a in ['external','internal_fixed','label_blind']}
for gid in range(64):
    ids=[i for i,r in enumerate(test) if r['graph']==gid and r['mechanism'] in ['uniform_nodes','clustered_nodes'] and r['k']==4]
    current=[];priors={'uniform':[],'context_frequency':[]}
    for i in ids:
        inp,extra,y=read(test[i]);missing=~inp.observed[inp.query,:test[i]['features']].any(1)
        classes=int(inp.classes)
        frequencies=np.bincount(inp.labels_context,minlength=classes)+1.
        for name,prob in [('uniform',np.ones(classes)/classes),('context_frequency',frequencies/frequencies.sum())]:
            p=np.broadcast_to(prob,(len(y),classes));m=C.metrics(p[missing],y[missing]);priors[name].append([m[k] for k in METRICS])
        for seed in [11,12,13]:current.append([data['context'][seed][i]['missing_metrics'][k] for k in METRICS])
    context.append(np.mean(current,0))
    for arm in other:
        other[arm].append(np.mean([[data[arm][seed][i]['missing_metrics'][k] for k in METRICS] for seed in [11,12,13] for i in ids],0))
    for name in priors:base.setdefault(name,[]).append(np.mean(priors[name],0))
diagnostic={name:{'means':dict(zip(METRICS,np.mean(val,0).tolist())),'context_minus_prior':difference(np.array(context),np.array(val))} for name,val in base.items()}
dump(root/'results/prior_controls_secondary.json',{'controls':diagnostic,'scope':'Nonlearned probability controls computed for transparency after design lock. Secondary; no change to preregistered primary or choice of model/checkpoint.'})
placement=difference(np.array(other['internal_fixed']),np.array(other['external']))
dump(root/'results/placement_secondary.json',{'internal_fixed-external':placement,'scope':'Secondary decomposition of the planned four-arm comparison, computed after training. Not the primary context-routing success test; multiplicity unadjusted.'})
names={'external':'외부 PCFI → PFN','internal_fixed':'내부 고정 PCFI','context':'내부 context PCFI','label_blind':'내부 label-blind PCFI'}
primary=stats['subsets']['primary']
def delta(contrast,metric='accuracy',s=primary):
    d=s['contrasts'][contrast][metric];scale=100 if metric=='accuracy' else 1
    a,lo,hi=[x*scale for x in [d['delta'],*d['ci95']]]
    return f'{a:+.5f} [{lo:+.5f}, {hi:+.5f}]'
lines=['# 실제 PCFI 내부화 GraphPFN — 결과', '',
f"사전 primary accuracy gate={stats['primary_gate']}, full-feature2%p guard={stats['full_guard']}, 확대 기준={stats['expand']}.", '',
'## 무엇을 바꿨나', '',
'이전 generic MLP refiner를 다시 돌린 실험이 아니다. 실제 PCFI의 거리 pseudo-confidence, 관측값 고정100회 전파와 마지막1회 피처 상관 보정을 사용했다. 네 모델은 공개 GraphPFN/LimiX의 같은 원래 초기화에서 새로 학습했고, 같은12층을 한 번 통과한다.', '',
'External은100회 PCFI를 앞에서 마치고 PFN을 실행한다. Internal은25회씩 PFN0/3/6/9층 앞에 배치하고 encoder increment를 feature tokens에 더한다. 마지막 block에만 원래 상관 보정을 한다. Context와 label-blind는 같은513-parameter router로 각 block의 전파 가중치만 조절한다. 임의의 숫자 혁신·추가 posterior head·재구성 loss는 없다.', '',
'Context router는 현재 PFN feature+label token의 context-class prototype support를 읽는다. Label-blind는 해당4개 입력을 모두0으로 하고 PFN hidden/labels를 읽지 않는다. 두 모델의 main classifier는 context labels를 쓴다. 따라서 차이는 context+PFN-feedback 경로 전체에 대한 것이며 동일 hidden representation에서 labels만 뺀 효과가 아니다. Blinded 입력 일부 weights는 비활성이다.', '',
'## Primary: 자기 피처 없는 query accuracy', '',
'새64 base graphs, context최대4/class, uniform/clustered node90/99.5% 네 조건과 고정3 seeds를 graph 안에서 평균했다. Missing query는 원래 physical features 전체가 미관측인 query다. Odd padding은 제외했다. 전체 query와 k1/cell/full은 별도다.', '',
'| 방법 | Accuracy % | NLL | Brier |','|---|---:|---:|---:|']
for arm,name in names.items():
    m=primary['means'][arm];lines.append(f"| {name} | {m['accuracy']*100:.5f} | {m['nll']:.6f} | {m['brier']:.6f} |")
for name,rec in diagnostic.items():
    m=rec['means'];lines.append(f"| 보조: {name} | {m['accuracy']*100:.5f} | {m['nll']:.6f} | {m['brier']:.6f} |")
lines+=['','| Paired 비교 | Accuracy 차이 %p [95% CI] | NLL 차이 [95% CI] |','|---|---:|---:|']
for contrast in primary['contrasts']:lines.append(f'| {contrast} | {delta(contrast)} | {delta(contrast,"nll")} |')
place=placement['accuracy'];pm=placement['nll']
lines+=['',f"보조 위치 비교 internal-fixed−external: accuracy{place['delta']*100:+.5f}%p CI[{place['ci95'][0]*100:+.5f},{place['ci95'][1]*100:+.5f}], NLL{pm['delta']:+.6f} CI[{pm['ci95'][0]:+.6f},{pm['ci95'][1]:+.6f}]. 이 보조 분석은 primary를 대체하지 않는다.", '',
'CI는 base graph5000회 paired bootstrap이다. Query/node를 독립 표본으로 늘리지 않았다. Training seeds는 고정 반복이며 seed 모집단 uncertainty를 추정하지 않았다. Subgroup/보조 CI는 다중 비교 미보정이다. CI에0을 포함한 것은 무효·동등성의 증명이 아니다. 두 비학습 prior는 design lock 이후 추가로 재채점한 보조 대조이며 primary를 교체하지 않는다.', '',
'## 조건별 결과', '', '| 조건 | External % | Internal-fixed % | Context % | Label-blind % | Context−external %p [CI] |','|---|---:|---:|---:|---:|---:|']
for key,s in stats['subsets'].items():
    if key=='primary':continue
    v=s['means'];lines.append('| '+key+' | '+' | '.join(f"{v[a]['accuracy']*100:.5f}" for a in names)+' | '+delta('context-external',s=s)+' |')
lines+=['','Full은 전체 query, 나머지는 자기 피처 없는 query다. Primary 밖의 k1/cell/개별 mask 결과를 사후 main 성공으로 승격하지 않는다. 모든 조건의 전체 query 지표도 statistics.json에 별도 보존했다.', '',
'## 비용', '', '| 방법 | CPU 입력→CPU 확률, ms |','|---|---:|']
for arm,name in names.items():lines.append(f"| {name} | {cost['methods'][arm]['mean_of_condition_medians_ms']:.3f} |")
lines+=['','우리 학습 worker 종료 후 같은RTX3090, 사전4조건 각각5 warmup/30회 median의 평균이다. 거리/confidence 전처리와 실제100회 전파,1회 correlation,1회 PFN 및 CPU 확률 반환을 포함했다. 모델 로딩과 bank I/O는 제외했고 외부 GPU 부하는 통제하지 않았다. Kernel만의 본질적 속도나 큰 그래프 메모리 우위로 해석하지 않는다.', '',
'## 학습·검증', '',
f"- 새12모델, 성공 updates{audit['successful_updates']}, exposures{audit['exposures']}. 동일64 train graphs×4 settings=256 episodes를 seed별 한 번씩 같은 순서로 학습했다. CE만, 마지막 checkpoint만 평가했고 target fitting은 없다. Dev optimizer4 steps는 폐기했다.",
f"- Trainable counts: {audit['counts']}. 모든 모델의 common/frozen 초기값과 task schedules가 같고 frozen backbone은 유지됐다. Context/blind는 정확히 같은 수, fixed/external보다513개 많다.",
f"- Dev 작은 그래프에서 Torch/SciPy PCFI 차이 최대4.43e-7. 실제 dev28 episodes의 FP32 kernel/cache 차이는 최대8.54e-5였다. Float16 public PFN의 node permutation 최대 차이는0.00244 이내였으므로 전체 모델의 exact permutation equivalence를 주장하지 않는다.",
'- 최초 GPU atomic scatter 구현은 같은 입력 반복에서도 FP16 최종 확률이0.000976 변했다. Final 생성 전 수치 연산을 같은 수학의 dense row-normalization/bmm로 바꾸고 dev 전체 검사를 재실행했다. 이 수정은 학습·최종 성능에 따른 tuning이 아니다.',
'- Hidden-value poison, 모든 관측값 clamp, class-name invariant router statistics, strict label-blind router, nonzero router/hidden-state gradient, 실제 optimizer helper와 trainable checkpoint 저장/복원을 검증한 뒤 설계를 잠갔다. Learned arm 초기 예측은 internal-fixed와 정확히 같았다.',
f"- 모든 source/config/bank/checkpoint/prediction hashes와 저장예측 재채점, {len(audit['replays'])}개 checkpoint replay를 확인했다. Primary256 episodes 중 실제 관측 cell이0개인 episode{audit['zero_observation_primary_episodes']}개도 제외하지 않았다.", '',
'## 해석 범위와 판정', '',
'같은 size-restricted official graph/SCM generator family의 새 graph seeds다. 실제 graph, unseen semantic schema, 원 논문의 공식 downstream classifier 재현이 아니다. 공통 observed-only RMS/native-pair/mask interface가 공식 전처리 전체와 같다고 주장하지 않는다.', '',
'Context가 external·internal-fixed·label-blind 모두를 primary에서 이겨야 학습된 조절의 증분을 인정한다. Fixed internal만 external을 이기면 배치 위치 효과이고, context가 fixed만 이기고 external은 못 이기면 전체 파이프라인 개선은 아니다. 실패 시 이 bank에서 새 loss/head/반복 횟수/학습량으로 구제하지 않는다. 원인이 분리되지 않으면 미상으로 남긴다.', '',
'Row-stochastic 전파는 상수 feature를 그대로 보존한다. 관측0개인 column에서0으로 시작하면 가중치 조절만으로 새 신호가 생기지 않는다. 이는 연산의 알려진 제약이며 전체 실험 실패의 단일 원인으로 확인한 것은 아니다.', '',
'원 수치는 [statistics.json](results/statistics.json), [cost.json](results/cost.json), [audit.json](results/audit.json), 정확한 설계는 [protocol.json](protocol.json), 계산 정의는 [METHOD_KO.md](METHOD_KO.md)를 확인한다.']
(root/'REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print('REPORT_WRITTEN',flush=True)

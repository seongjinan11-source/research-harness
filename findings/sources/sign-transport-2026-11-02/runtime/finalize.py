"""Report frozen diagnostic and append records without revising old findings."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml

root=Path(__file__).resolve().parents[1];repo=root.parent/'research-harness'
def read(p):return json.loads((root/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read('results/statistics.json');audit=read('results/audit.json');cost=read('results/cost.json');complete=read('results/complete.json')
corrected=read('results/common_query_interventions.json')
p=s['primary'];c=p['contrasts'];headroom=s['primary_headroom'];floor=corrected['strong_floor_corrected']
def delta(d,m='accuracy'):
    scale=100 if m=='accuracy' else 1
    a,lo,hi=[z*scale for z in [d[m]['delta'],*d[m]['ci95']]]
    return f'{a:+.6f} [95% CI {lo:+.6f}, {hi:+.6f}]'
if not headroom:
    verdict='알려진 부호 법칙을 반영한 복원도 현재 frozen PCFI→PFN의 정확도 이득을 사전 기준대로 확인하지 못했다. 이 진단을 근거로 새 PFN 구조나 학습을 시작하지 않는다.'
elif not floor:
    verdict='같은 PCFI→PFN checkpoint에서 부호 법칙을 반영한 복원의 예측적 가치는 확인했다. 그러나 강한 직접 예측·graph-only 대조를 모두 넘는 기준은 통과하지 못해, 새 학습 모델이 필요하다는 근거로 확대하지 않는다.'
else:
    verdict='알려진 부호 법칙을 반영한 복원이 현재 frozen PFN과 고정 graph-only 대조를 넘는 headroom을 보였다. 다만 참 feature-law bit를 받은 진단이므로 task-adaptive 학습 성과나 새 메소드가 아니다. 새로운 학습은 아직 실행하지 않는다.'

body='# 부호 전달 진단 — PCFI 복원과 GraphPFN 분류의 한계 분리\n\n'+verdict+'\n\n'
body+='기존 Mask Plain/PCFI Plain의 최종 checkpoint를 3개 seed씩 그대로 재사용했다. 새 모델과 학습 update는 0개다. 두 arm은 이전 official graph/SCM bank에서 같은 schedule과 192-update 예산으로 학습됐다. 이번 graph-signal family는 그 학습 prior 밖이며, frozen 실패를 architecture의 표현력 불가능성으로 해석하지 않는다. 여기서 Plain은 공개 GraphPFN Mask Plain이고 과거 소형 typed-relation Plain이 아니다.\n\n'
body+='## 무엇을 분리했나\n\n192-node connected bipartite graph에서 저주파 Gaussian field a를 만들었다. Feature law g=0은 이를 그대로, g=1은 bipartition b를 곱해 관측한다. Label law t=0은 a, t=1은 b*a를 median threshold한 binary class다. 여덟 column은 같은 field의 서로 다른 방향·scale·작은 noise 버전이다. Feature law와 label law를 독립적으로 바꿨다.\n\n'
body+='각 label 조건 내 graph·labels·context·query·observation mask는 feature-law 두 설정에서 같다. 실제 달성된 label homophily와 feature energy를 기록했으며 원하는 수치가 나올 때까지 graph를 재추출하지 않았다. 새 base graphs32개와 개발 graphs2개, 최종32조건/graph=1,024 episodes다. Edge observation missing은 없다.\n\n'
body+='Context는 클래스당1/4개, 관측 source는 context 밖에서 정확히0/1/8개다. 1개 source는99.48%,8개는95.83% whole-node missing이고 Bernoulli99.5%와 다르다. Full-feature reference만 context에도 피처가 있다. Primary는 자기 feature가 없는 query다. Hidden values와 query labels는 predictor에 들어가지 않는다.\n\n'
body+='## Primary\n\nLabel/feature 둘 다 sign-gauged(t=1,g=1), sources8, context4/class 조건이다. 세 고정 training seed를 base graph 안에서 평균한 뒤32개 graph를 paired bootstrap5,000회 했다. CI는 training-seed population에 대한 값이 아니다.\n\n'
body+='| 방법 | Missing-query accuracy % | NLL | Brier |\n|---|---:|---:|---:|\n'
names={'mask':'기존 Mask Plain','pcfi':'기존 PCFI Plain','known_signed':'같은 PCFI Plain + known-law signed 복원','positive_lp':'고정 양의 LP','signed_lp':'고정 signed LP','uniform':'Uniform'}
for method,v in p['means'].items():body+=f"| {names[method]} | {v['accuracy']*100:.6f} | {v['nll']:.6f} | {v['brier']:.6f} |\n"
for key,label in [('zero_source','같은 PCFI checkpoint, 피처 제거·동일 query'),('full','같은 PCFI checkpoint, full·동일 query')]:
    v=corrected['means'][key];body+=f"| {label} | {v['accuracy']*100:.6f} | {v['nll']:.6f} | {v['brier']:.6f} |\n"
body+='\nSigned 복원 reference−대조 accuracy 차이(%p):\n\n'
for key,d in c.items():body+='- '+key+': '+delta(d)+'\n'
body+='- 같은 PCFI checkpoint의0-source 대조, 동일 query 집합: '+delta(corrected['contrasts']['known_signed-zero_source'])+'\n'
body+='\n원 PCFI checkpoint의 full−missing accuracy 차이, 동일 query 집합: '+delta(corrected['contrasts']['full-pcfi'])+'%p.\n\n'
body+=f"사전 reconstruction headroom={headroom}, 강한 direct/LP/feature-use 기준={floor}. Primary signed−PCFI의 NLL 차이는 {delta(c['known_signed-pcfi'],'nll')}, Brier 차이는 {delta(c['known_signed-pcfi'],'brier')}다.\n\n"
body+='원 PCFI−피처 제거의 동일-query accuracy 차이는 '+delta(corrected['contrasts']['pcfi-zero_source'])+'%p다. Signed 복원−피처 제거 차이의 CI는0을 포함한다. 따라서 signed 교체가 원 PCFI의 분류를 개선한 것은 맞지만, 피처가 없는 입력보다 추가 예측 정보를 유의하게 회수했다고 주장할 수는 없다. 이 cross-budget 보조 CI는 다중비교 미보정이다.\n\n'
body+='**Known-law reference는 참 feature-law g를 안다.** Bipartition b는 adjacency만으로 구하지만, g가 주어졌다는 점은 privileged information이다. 실제 missing X를 공개하지 않고 g=1에서 b*x를 PCFI로 복원한 뒤 b를 다시 곱했다. 새 그래프에서 g를 알아낸 모델이 아니며 이 결과만으로 신경망 학습이 필요하다고 할 수 없다. 양의/signed LP도 두 후보를 모두 보고하며 query truth로 좋은 쪽을 고르는 predictor는 만들지 않았다.\n\n'
body+='## Label/feature law factorial\n\nSources8,context4/class의 보조 결과다. Label H는 edge 양끝의 클래스가 같은 비율이며 feature energy와 다른 관측량이다.\n\n| Label law t | Feature law g | Label H | Mask % | PCFI % | Known signed % | 양의 LP % | Signed LP % | PCFI MSE | Signed MSE |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for t in [0,1]:
    for g in [0,1]:
        v=s['cells'][f't{t}_g{g}_k4_o8'];means=v['means']
        body+=f"| {t} | {g} | {v['mean_label_homophily']:.6f} | {means['mask']['accuracy']*100:.6f} | {means['pcfi']['accuracy']*100:.6f} | {means['known_signed']['accuracy']*100:.6f} | {means['positive_lp']['accuracy']*100:.6f} | {means['signed_lp']['accuracy']*100:.6f} | {v['mean_pcfi_mse']:.6f} | {v['mean_signed_mse']:.6f} |\n"
body+='\nMSE는 실제 관측값에서 구한 RMS scale과 clip10을 적용한 hidden query feature에 대한 값이다. Full-X의 통계로 model 입력을 정규화하지 않았다. MSE 개선은 classification 이득과 별개다. 원 PCFI−Mask 차이에 대한 feature flip difference-in-differences는 '+delta(s['feature_flip_difference_in_differences'])+'%p다. 이는 보조 interaction 분석이고 primary를 대체하지 않는다.\n\n'
other=s['cells']['t0_g1_k4_o8']
body+='특히 label은 smooth하고 feature만 sign-gauged인 t0/g1에서는 signed 복원이 MSE를 낮췄지만 accuracy 점추정은 '+f"{other['means']['pcfi']['accuracy']*100:.6f}%에서 {other['means']['known_signed']['accuracy']*100:.6f}%로 바뀌었다. 보조 차이 {delta(other['contrasts']['known_signed-pcfi'])}%p다. 복원 이득을 모든 task의 분류 이득으로 확대할 수 없다.\n\n"
body+='## Source 수와 context 수\n\nLabel/feature 모두 sign-gauged인 보조 분할이다. Source0은 피처 제거 대조이며, full은 관측 query도 포함하는 별도 reference다.\n\n| k/class | Sources | Mask accuracy % | PCFI accuracy % | Known signed accuracy % | 양의 LP % | Signed LP % |\n|---|---|---:|---:|---:|---:|---:|\n'
for k in [1,4]:
    for o in [0,1,8,'full']:
        v=s['cells'][f't1_g1_k{k}_o{o}']['means']
        body+=f"| {k} | {o} | {v['mask']['accuracy']*100:.6f} | {v['pcfi']['accuracy']*100:.6f} | {v['known_signed']['accuracy']*100:.6f} | {v['positive_lp']['accuracy']*100:.6f} | {v['signed_lp']['accuracy']*100:.6f} |\n"
body+='\n개별 분할 CI와 전체2×2×2×4 factorial, seed별 결과는 statistics.json에 있다. 보조 CI는 multiplicity 미보정이다. Source 수 표는 각 source 조건의 자체 missing-query 집합이므로 직접적인 intervention 차이는 위 primary와 동일한 query 집합으로 다시 계산한 값을 사용한다.\n\n'
body+='## 보조 집계 감사 보정\n\n원 statistics.json의 zero-source/full cross-budget 차이는 각각의 missing-query 집합(full은 전체 query)을 사용해 query 집합 변화가 섞였다. Primary8-source의 missing-query 집합으로 두 예측을 모두 제한해 results/common_query_interventions.json에 재계산했다. 원 statistics·예측·bank·checkpoint는 그대로 보존했다. Primary와 같은 cell 안의 모든 비교 및 strong-floor 판정은 바뀌지 않았다. Full 입력은 context feature도 관측하며 자체 RMS를 사용하므로 완전 입력 reference로만 해석한다.\n\n'
body+='## 수학적 반례와 실험의 범위\n\n단일 column·연결된 path에서 관측값이+1이고 결측 시작값이0이면 PCFI의 양의 row-stochastic 전파는 음수 값을 생성하지 못한다. 한 column이라 channel correction도0이다. 개발 검사에서 일반 PCFI는 음수 값을 만들지 못했고 signed 변환은 필요한 반대 부호를 만들었다. 이는 복원 연산의 반례다. GraphPFN이 graph/context로 정답을 복구할 수 없다는 증명은 아니다.\n\n'
body+='Signed diffusion/BP는 우리 기존 연구에서 이미 구현했다. 이 실험은 새로운 attention·posterior·loss·메소드가 아니라, 현재 PCFI와 frozen 공개 PFN의 역할을 분리하는 known-family 진단이다. 고정 Gaussian/signed 추론의 기존 연구와 구분되는 학습 원리를 확보한 결과도 아니다.\n\n'
body+='## 비용과 검증\n\n| 경로 | Median latency ms | 추가 peak work memory MiB |\n|---|---:|---:|\n'
for a,v in cost['methods'].items():body+=f"| {names[a]} | {v['median_ms']:.6f} | {v['additional_peak_mib']:.3f} |\n"
body+='\n같은 RTX3090에서 입력→CPU 확률, seed11/첫 graph primary cell,5warmups/30repeats다. PCFI 경로는 실제 PCFI를 다시 계산하고 signed reference는 parity 탐색도 포함한다. 로드/파일I/O를 제외하고 외부 GPU 부하는 통제하지 않았다. 이는 추가 작업 메모리이며 총 GPU 메모리가 아니다.\n\n'
body+=f"원 checkpoint replay18건을 lock 전에 확인했다. 저장 예측 {audit['predictions_rescored']}건을 재채점했고 새 replay{len(audit['replays'])}건의 최대 확률 차이는 {max(r['max_abs_error'] for r in audit['replays']):.9g}였다. Source/config/bank/checkpoint/prediction SHA, 입력 factorial pairing, hidden feature poison, 관측값 clamp, zero-source 입력/출력 동일성과 평가 전후 모델 파라미터 해시를 확인했다. 새 모델/updates는0이다.\n\n"
body+='## 최종 판정\n\n'+verdict+'\n\nFrozen 결과의 고유 원인은 분리하지 않았다. 새로운 graph-signal law와 학습 prior의 차이, 입력/표현·최적화·decoder 한계 등이 섞일 수 있다. 모든 heterophily나 PCFI/PFN 통합의 무효로 확대하지 않는다. 현재 bank를 보고 온도·checkpoint·학습량·feature-law를 조정하지 않는다.\n'
(root/'REPORT_KO.md').write_text(body,encoding='utf8')

prefix='findings/sources/sign-transport-2026-11-02/';dest=repo/prefix;assert not dest.exists();dest.mkdir(parents=True)
paths=list(root.glob('*.py'))+[root/p for p in ['protocol.json','METHOD_KO.md','REPORT_KO.md','bank/development.json','bank/manifest.json']]
paths+=list((root/'results').rglob('*.json'))+list((root/'logs').glob('*.log'))+list((root/'runtime').glob('*.py'))
manifest=[]
for path in sorted(set(paths)):
    q=dest/path.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,q)
    manifest.append({'alias':path.relative_to(root).as_posix(),'original':str(path),'snapshot':q.relative_to(repo).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Frozen sign-transport probe;0 training updates. Sources copied verbatim; raw inputs/probabilities/checkpoints remain local.','sources':manifest},ensure_ascii=False,indent=2),encoding='utf8')
numbers='; '.join(f"{a}:accuracy{v['accuracy']*100:.6f}%/NLL{v['nll']:.6f}/Brier{v['brier']:.6f}" for a,v in p['means'].items())
numbers+='; '+'; '.join(a+' accuracy%p '+delta(d) for a,d in c.items())
refs=[prefix+f for f in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/common_query_interventions.json','results/audit.json','results/cost.json','results/preflight.json']]
core={'id':'SIGN-P01' if headroom else 'SIGN-F01','kind':'preserved' if headroom else 'failure',
 'claim':'관계를 따라 feature 부호가 바뀌는 과제에서, 알려진 법칙으로 PCFI 복원을 바꾸면 같은 frozen 공개 GraphPFN의 classification이 개선된다.',
 'observation':numbers+f". Primary headroom={headroom}, strong direct/LP/feature-use floor={floor}.",
 'tags':['observation-mask','graph-structure-use','eval-protocol','inference-cost'],
 'keywords':['feature heterophily','label heterophily','bipartite sign gauge','positive diffusion limitation','frozen GraphPFN','privileged known feature law','signed transport'],
 'source_refs':refs,'evidence':'기존 matched192-update Mask/PCFI Plain checkpoint6개 재사용, 새 training0. 새32base graphs×32conditions=1024episodes. 세 fixed training seeds 평균 후5000 paired graph bootstrap. Primary t1/g1/k4/8sources.'}
if headroom:
    core['do_not_claim']='Known-law g bit를 받은 reconstruction 진단이다. 새 graph에서 부호 법칙을 추론한 학습 결과, arbitrary heterophily·real graph·schema 전이 또는 메소드 novelty가 아니다. 강한 baseline floor='+str(floor)+'. Signed LP 대비 우위와 동일-query zero-source 대비 feature-use 이득은 확인하지 못했다. Frozen out-of-prior 비교이며 새 prior에서 matched 학습한 구조 우열이 아니다. 새로운 학습은 실행하지 않았다.'
else:
    core.update(verdict=verdict,reason_status='unknown',
      failure_reason='Known-law 복원 교체가 사전 primary accuracy CI 기준을 통과하지 못했다. Frozen checkpoint의 prior mismatch, 입력/표현, 복원/decoder 중 고유 원인은 분리하지 않았다. MSE 개선을 분류 성공으로 바꾸지 않는다.',
      closed_scope='현재 bipartite sign-gauged Gaussian signal/binary label/8 correlated columns family에서 기존192-update 공개 Mask/PCFI Plain을 재사용한 known-law reconstruction primary와 이를 근거로 한 자동 확장.',
      not_closed='PCFI 단일-channel 양의 전파의 부호 반례, 기존 signed BP와 새 family에 맞춰 학습하는 방법, 임의 heterophily·관계 schema·실제 graph 일반을 기각하지 않는다. 과거 typed-relation Plain의 실패로 세지 않는다.',
      reopen_when='Known-law reconstruction의 MSE를 넘어 실제 missing-query classification에 쓰이는 feature 정보를 회수하는 원리가 생기고, 새 prior matched 학습 및 강한 고정 signed/통계 추론 대조를 사전에 설계할 수 있을 때.',
      comparison_floor='Matched 기존 Mask/PCFI Plain, positive/signed LP,full-feature와zero-source, 알려진 부호 법칙 reference. 새 메소드 주장에는 동일 새 prior 학습 및 고정 known-family 추론과 비용 비교도 필요.',
      recheck='현재 bank를 tuning하지 않고, label/feature heterophily를 교차하며 graph·labels·mask pairing과 feature-use intervention을 유지한 독립 검증을 한 번 수행한다.',
      reopen_tags=['observation-mask','graph-structure-use'],reopen_keywords=['task adaptive feature transport','signed feature likelihood','heterophily missing features'],revisit_status='open')
lesson={'id':'SIGN-L01','kind':'lesson','claim':'Feature 복원의 부호 제약, label heterophily, frozen PFN 분류, known-law 정보 우위를 구분한다.',
 'observation':'단일 channel path의+1 source/0 missing start에서 PCFI는 음수를 만들지 못했고 parity 변환은 반대 부호를 만들었다. 최종 factorial은 graph/labels/context/query/mask를 feature law 간 고정했다. Primary 결과 '+numbers+'.',
 'interpretation':'PCFI의 단일-channel counterexample은 뒤 PFN의 분류 불가능성 증명이 아니다. Label/feature heterophily는 독립 축이며 실제 달성된 homophily를 보고한다. Privileged g reference의 개선은 unknown-g adaptation의 성과가 아니다. Signed diffusion/BP는 기존 자산이므로 새 이름으로 novelty를 주장하지 않는다. 기존 training prior 밖의 frozen 실패를 새 architecture 필요성으로 곧바로 해석하지 않는다. Source 수가 바뀌면 missing-query 집합도 바뀌므로 feature intervention은 원 primary query 집합으로 양쪽 예측을 제한한다. 보조 집계 보정은 원 statistics·예측을 보존해 별도 기록했다.',
 'tags':['eval-protocol','graph-structure-use','observation-mask'],'keywords':['reconstruction versus classification','feature versus label heterophily','known law oracle','frozen prior mismatch','signed BP already implemented'],
 'source_refs':refs,'evidence':'원137finding과590source 파일은 byte/구조화 hash 그대로 유지한다. 기존 model Plain 이름도 구분한다.'}
finding={'project':'Frozen GraphPFN sign-transport diagnostic','date':'2026-11-02','source':prefix+'REPORT_KO.md',
 'setting':'Known bipartite sign-gauged signal family; label/feature-law2x2;0/1/8/full sources;context1/4 per class;6 frozen checkpoints;0 updates.','related':['GraphPFN'],'entries':[core,lesson]}
(repo/'findings/sign-transport.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
old=read('runtime/previous_harness.json');entries=[];projects=0
for path in (repo/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    projects+=1;entries+=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('entries',[])
hashes={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(hashes[k]==v for k,v in old['entries'].items())
assert all(sha(repo/p)==v for p,v in old['sources'].items())
counts=Counter(e['kind'] for e in entries)
(root/'results/archive_integrity.json').write_text(json.dumps({'old_findings_unchanged':len(old['entries']),'old_sources_unchanged':len(old['sources']),'new_ids':[core['id'],lesson['id']],'counts':dict(counts)},ensure_ascii=False,indent=2),encoding='utf8')
section='## 부호 전달과 기존 frozen GraphPFN 진단\n\n기존 matched192-update Mask/PCFI Plain checkpoint6개를 새32graph sign-gauged family에서 추가 학습 없이 비교했다. Label heterophily와 feature heterophily를 교차했으며, graph/label/context/mask를 유지하고 feature 부호 패턴을 바꿨다.\n\n| Primary: t1/g1,k4/class,sources8 | Missing-query accuracy % | NLL |\n|---|---:|---:|\n'
for a in ['mask','pcfi','known_signed','positive_lp','signed_lp']:
    v=p['means'][a];section+=f"| {names[a]} | {v['accuracy']*100:.6f} | {v['nll']:.6f} |\n"
section+='\nKnown-law signed−PCFI accuracy '+delta(c['known_signed-pcfi'])+'%p. '+verdict+' Known-law g는 privileged diagnostic 정보이며 학습된 adaptation이나 새 메소드가 아니다. [보고서]('+prefix+'REPORT_KO.md), [계산 정의]('+prefix+'METHOD_KO.md), [구조화 판정](findings/sign-transport.yaml).\n\n'
for name in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
    path=repo/name;body=path.read_text(encoding='utf8');anchor='## 내부 고정 PCFI의 새 graph 재검증';assert anchor in body
    body=body.replace(anchor,section+anchor,1)
    body=body.replace('32개 프로젝트 묶음의137항목',f'{projects}개 프로젝트 묶음의{len(entries)}항목').replace('연구 기록 137항목',f'연구 기록 {len(entries)}항목')
    body=body.replace('실패·중단41개, 보존29개, 교훈45개, 미실행22개',f"실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개")
    body=body.replace('32개 프로젝트 묶음이며 실패·중단41, 보존29, 교훈45, 미실행22항목',f"{projects}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목")
    body=body.replace('137번의 독립 실험',f'{len(entries)}번의 독립 실험').replace('[137개 전체 기록]',f'[{len(entries)}개 전체 기록]')
    body=body.replace('graphpfn_review_packet_2026-11-02_v7.zip','graphpfn_review_packet_2026-11-02_v8.zip')
    body=body.replace('v7는 내부 고정 PCFI의 독립 graph 재검증까지 포함한다.','v8은 부호 전달 frozen 진단까지 포함한다. v7는 직전 내부 고정 재검증 snapshot이다.').replace('과거 v1/v2/v3/v4/v5/v6 ZIP','과거 v1/v2/v3/v4/v5/v6/v7 ZIP')
    body=body.replace('최근 업데이트: 내부 고정 PCFI 재검증까지 포함한','최근 업데이트: 부호 전달 진단까지 포함한')
    if name=='RESEARCH_HANDOFF_KO.md':
        body=body.replace('1. [내부 고정 PCFI 재검증]', '1. [최신 부호 전달 진단]('+prefix+'REPORT_KO.md)과 [내부 고정 PCFI 재검증]',1)
        anchor_prompt='FP/FISF 내부화 후속 학습은 시작하지 않는다.'
        # Append within the final handoff text block only.
        i=body.rfind('```text');assert i>=0
        pre,tail=body[:i],body[i:]
        marker='\n```';j=tail.rfind(marker);assert j>=0
        tail=tail[:j]+'\n추가로 label/feature heterophily를 분리한 sign-transport frozen 진단을 했어. Primary known-law signed−PCFI는 '+delta(c['known_signed-pcfi'])+'%p였어. '+verdict+' 참 feature-law g를 안 reference라서 새로운 학습/novelty의 성과로 부르지 말아줘.\n'+tail[j:]
        body=pre+tail
    path.write_text(body,encoding='utf8')
path=repo/'README.md';body=path.read_text(encoding='utf8')
body=body.replace('내부 고정 PCFI의 추가 학습 없는 독립 graph 재검증','내부 고정 PCFI 재검증과 부호 전달 frozen 진단')
path.write_text(body,encoding='utf8')
path=repo/'tools/build_review_packet.py';body=path.read_text(encoding='utf8');assert 'VERSION = 7' in body
path.write_text(body.replace('VERSION = 7','VERSION = 8',1),encoding='utf8')
print('REPORT_AND_ARCHIVE_READY',core['id'],dict(counts),flush=True)

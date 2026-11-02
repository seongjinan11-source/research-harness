"""Append findings; preserve every previous observation and source byte."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml
root = Path(__file__).resolve().parents[1]
repo = root.parent/'research-harness'
snapshot = repo/'findings/sources/feature-law-2026-11-02'
assert not snapshot.exists()
snapshot.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[root/p for p in ['REPORT_KO.md','METHOD.md','protocol.json']]
files+=list(root.glob('*.py'))
files+=list((root/'results').rglob('*.json'))
files+=[root/p for p in ['bank/manifest.json','bank/development_manifest.json','references/manifest.json']]
files+=list((root/'logs').glob('*.log'))
files+=list((root/'runtime').glob('initial_*'))
files+=list((root/'runtime').glob('*.py'))
sources=[]
for path in sorted(set(files)):
    dest=snapshot/path.relative_to(root)
    dest.parent.mkdir(exist_ok=True,parents=True)
    shutil.copyfile(path,dest)
    sources.append({'alias':path.relative_to(root).as_posix(),'original':str(path),'snapshot':dest.relative_to(repo).as_posix(),'sha256':sha(dest),'bytes':dest.stat().st_size})
(snapshot/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Locked feature-law cell-factor comparison. Raw task/prediction arrays and checkpoints remain in workspace. Snapshots are verbatim; independent final graph seeds, not unseen-law generalization.','sources':sources},ensure_ascii=False,indent=2),encoding='utf8')
stat=json.loads((root/'results/statistics.json').read_text(encoding='utf8'))
p=stat['subsets']['primary_heldout'];g=stat['subsets']['gaussian_guard']
cost=json.loads((root/'results/cost.json').read_text(encoding='utf8'))
def d(comp,metric='accuracy',subset=p):
    e=subset['differences'][comp][metric]; k=100 if metric=='accuracy' else 1
    a,l,h=[x*k for x in [e['delta'],*e['ci95']]]
    return f'{a:+.6f} CI[{l:+.6f},{h:+.6f}]'
def v(arm):
    m=p['means'][arm]
    return f"{arm}: accuracy{m['accuracy']*100:.6f}%/NLL{m['nll']:.6f}/Brier{m['brier']:.6f}/referenceKL{m['reference_kl']:.6f}"
def c(arm):return f"{cost['single_graph'][arm]['median_ms']:.3f}ms"
prefix='findings/sources/feature-law-2026-11-02/'
refs=[prefix+x for x in ['REPORT_KO.md','METHOD.md','protocol.json','results/statistics.json','results/cost.json','results/audit.json','results/preflight.json','results/runtime_amendment.json']]
assert stat['verdict']['adaptation_accuracy_gate'] is False, 'Do not write a failure verdict if results changed'
entry={'id':'FLAW-F01','kind':'failure',
 'claim':'관측 cell의 부호 제약을 제거하고 새 graph의 column 통계에 조건부인 factor를 학습하면 static과 기존 Factor보다 자기 피처가 없는 query 분류가 좋아진다.',
 'observation':'; '.join(v(a) for a in ['original','static','adaptive','fixed_exact','fixed_bp','mask_pfn','pcfi_pfn'])+'. Adaptive−static accuracy%p '+d('adaptive-static')+', adaptive−original '+d('adaptive-original')+'. 두 primary CI 모두0 포함. Gaussian guard: adaptive−original '+d('adaptive-original',subset=g)+', adaptive−static '+d('adaptive-static',subset=g)+f". Adaptation gate=False, Gaussian guard={stat['verdict']['gaussian_2pp_noninferiority_guard']}, competitive gate={stat['verdict']['competitive_gate']}. Single-graph cost adaptive{c('adaptive')}, fixed exact{c('fixed_exact')}, fixed BP{c('fixed_bp')}.",
 'verdict':'사용자가 정한 primary accuracy 기준을 충족하지 못했으므로, 현재 cell 함수 교체만으로 독립 메인 메소드가 된다는 가설은 닫는다. 추가 head/loss/temperature/updates로 이 bank에서 구제하지 않는다.',
 'failure_reason':'부호 제약 제거와 관측 column 통계 conditioning만으로 이 예산에서 분류 이득을 확인하지 못했다. 관측 정보량·통계 요약·함수족·비선형 최적화·finite BP/Bethe·학습량의 기여를 분리하지 않아 고유 원인은 미상이다. CI에0을 포함한 것을 효과0 또는 동등성의 증명으로 쓰지 않는다.',
 'reason_status':'unknown',
 'excluded_explanations':['세 Factor와 두 fixed-law controls는 사전 선택한 동일 CE seed11 graph encoder/factor를 고정한다. Static1547/adaptive1575 trainable params, 동일 초기 predictive scores, 동일 seed별2048 tasks와256 successful updates다. Frozen graph 재학습 효과가 아니다.',
 '새15개 모델을 CE만으로 학습했다. Public Mask/PCFI는 새 prior로 실제 학습한 별도 adapter architecture다. Law·actual parameter·orientation·query truth를 predictor 입력에 넣지 않았다.',
 'Dev12 graphs의 수치·finite-gradient smoke weights3개를 폐기하고 설계를 잠근 뒤 final192 graphs를 생성했다. 최종 bank 생성 후 optimizer 전 발생한 keyword-only API 오류만 수정해 같은 bank와 초기화로 재시작했고 원 lock/로그를 보존했다. 분석 entrypoint는 같은 이름의 역사적 모듈과 import 충돌을 분리했다.',
 '전체 source/config/bank/reference/checkpoint hashes, freeze/schedule, 모든 mask와45개 prediction replay를 확인했다. 작은 tree의 normalized law/orientation 완전 열거와 fixed BP, hidden-value poison, class/relation/column permutation 검사도 통과했다.'],
 'closed_scope':'12-node binary native graph/2relations/2features, context총1/2/4개와 observed nodes3/6개 완전 분리, Gaussian mean/Student-t3 mean/variance law를 학습에 모두 넣고 diagonal train/off-diagonal heldout parameter tuples를 비교한 frozen-graph cell-only256-update CE pilot.',
 'not_closed':'기존 disjoint feature/label 연결 연산·signed likelihood·자기 role-update와 이전 Factor/teacher의 국소 개선은 보존한다. 모든 학습된 feature factor, unseen feature laws, 가변 feature/schema, 더 큰 실제 graph,99.5% missing의 효과를 기각한 결과는 아니다. 이번에도 law 이름은 이미 보았다.',
 'reopen_when':'관측 통계로 pointwise rho만 조절하는 것과 다른, 현재 함수족이 놓치는 feature–relation 의존성을 학습·전이하는 구체적인 연산이 등장하고 강한 명시적 법칙 주변화보다 유리할 근거가 생길 때.',
 'comparison_floor':'Original sign-constrained Factor, parameter-matched Flexible-static/adaptive, 동일 frozen graph의 full normalized law/parameter/orientation fixed-exact 및 same24-step fixed-BP, 실제 새 prior로 학습한 Mask/PCFI public PFN. Accuracy/NLL/Brier/reference KL과 end-to-end cost를 분리한다.',
 'recheck':'이 final bank를 추가 tuning에 쓰지 않는다. 새 연산의 구체적인 차이를 먼저 고정하고 독립 base graphs에서 자기 피처 없는 query accuracy를 primary로 비교하며 Gaussian 보존과 고정 추론 품질–비용을 함께 확인한다.',
 'tags':['observation-mask','amortized-inference','model-based-baseline','distribution-shift','eval-protocol','inference-cost'],
 'keywords':['feature law adaptation','variance evidence','heavy tail','heteroscedastic feature','signed cell factor','static versus adaptive','disjoint context feature','same frozen graph','no-feature query accuracy'],
 'reopen_tags':['amortized-inference','graph-structure-use','observation-mask'],
 'reopen_keywords':['joint feature law inference','conditional feature evidence','shared column latent','feature relation dependency','law marginalization'],
 'source_refs':refs,'evidence':'새15개 모델3840 successful updates/30720 exposures, dev3 smoke updates 별도. Fresh192 base graphs/1152episodes, primary heldout96 graphs. Query→episode6→fixed seeds3→graph 평균 후 paired5000 bootstrap. Reference는 broad known-family prior의 exact 분포이며 참 OOD posterior가 아니다.','revisit_status':'open'}
lesson={'id':'FLAW-L01','kind':'lesson','claim':'점별 cell 함수의 표현력 제약 제거와, 새 graph에서 feature law를 추론해 분류를 개선하는 것은 다른 검증이다.',
 'observation':'분산 차이의 참 half log-likelihood ratio는 x²의 함수라 큰 양수와 큰 음수가 같은 evidence를 준다. Original x*positiveNN은 이를 그대로 표현할 수 없지만 새 static/adaptive 함수는 짝수 evidence도 허용한다. 그 구조 변경 이후에도 primary adaptive−static과 adaptive−original accuracy 우위는 확인되지 않았다.',
 'interpretation':'새 함수가 필요한 예시를 구성한 것만으로 그 함수를 유한 예산에서 학습해 효과적으로 활용한다는 결론은 나오지 않는다. 학습 law3종은 모두 알려진 상태이며 heldout은 parameter tuple 조합뿐이다. Exact fixed-law와 학습 factor의 차이에는 law uncertainty 처리와 정확한 latent 합산/BP 근사 차이도 포함된다. Same-BP 고정 대조도 함께 보고하며 어느 하나만 단일 실패 원인으로 확정하지 않는다. Normalized known-family reference KL과 실제 sampled-label accuracy를 분리한다.',
 'tags':['amortized-inference','model-based-baseline','distribution-shift','eval-protocol'],
 'keywords':['expressivity versus learnability','even feature evidence','law known parameter tuple heldout','normalized reference versus true Bayes','no teacher loss'],
 'source_refs':[prefix+x for x in ['METHOD.md','REPORT_KO.md','results/preflight.json','results/statistics.json']],
 'evidence':'Variance half-LLR의 ±1.5 일치/양수 및0에서 음수, 세 함수의 동일 initial predictions를 dev에서 확인했다. 실제 최종 판정은 independent final bank의 primary accuracy다.'}
preserved={'id':'FLAW-P01','kind':'preserved',
 'claim':'세 feature law를 포함한 새 bank에서도 adaptive Factor는 같은 prior로 실제 학습한 공개 Mask/PCFI PFN보다 자기 피처 없는 query accuracy가 높았다.',
 'observation':'Adaptive−Mask accuracy%p '+d('adaptive-mask_pfn')+', NLL '+d('adaptive-mask_pfn','nll')+'. Adaptive−PCFI accuracy%p '+d('adaptive-pcfi_pfn')+', NLL '+d('adaptive-pcfi_pfn','nll')+'.',
 'do_not_claim':'두 공개 PFN 비교는 보조·multiplicity 미보정이다. Adaptive는 primary에서 static과 original을 이기지 못했으므로 adaptive cell의 독립적인 기여나 논문 메소드 성공이 아니다. 공개 PFN과 Factor는 architecture/사전학습/graph factor 경로가 달라 정확한 이득 원인을 분리하지 않았다. 이번에는 feature 제거 개입을 하지 않았으므로 이 차이만으로 variance feature를 활용했다고 단정하지 않는다. 고정 law 주변화보다 정확도가 낮은 것이라고 주장하지도 않는다. Adaptive−fixed exact accuracy CI도0을 포함한다.',
 'tags':['observation-mask','graph-structure-use','model-based-baseline','eval-protocol'],
 'keywords':['same law mixture bank','structured factor versus public PFN','no-feature query','preserved local improvement','not adaptive contribution'],
 'source_refs':[prefix+x for x in ['REPORT_KO.md','results/statistics.json','results/audit.json']],
 'evidence':'동일 새2048 training tasks·256 updates·3 fixed seeds, independent final heldout96 base graphs의 paired 비교다. 기존 Gaussian bank의 절대 수치를 뺀 비교가 아니다.'}
finding={'project':'Feature-law adaptive observed-cell Factor','date':'2026-11-02','source':prefix+'REPORT_KO.md',
 'setting':'동일 frozen graph checkpoint 위에서 Original/static/adaptive cell만 바꾼3seeds CE. 새 prior로 Mask/PCFI도 실제 학습. Final192 graphs 중 off-diagonal parameter tuple96개, 자기 피처 없는 query accuracy primary.','related':['GraphPFN','CSBMInference'],'entries':[entry,preserved,lesson]}
(repo/'findings/feature-law-factor.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
# Preserve old entry values and snapshots exactly, then derive new counts.
previous=json.loads((root/'runtime/previous_harness.json').read_text(encoding='utf8'))
entries=[]; projects=0
for path in (repo/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    projects+=1
    entries+=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('entries',[])
current={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(current[k]==v for k,v in previous['old_entries'].items())
assert all(sha(repo/p)==h for p,h in previous['old_sources'].items())
counts=Counter(e['kind'] for e in entries)
auditpres={'old_entries_unchanged':len(previous['old_entries']),'old_source_files_unchanged':len(previous['old_sources']),
 'new_entries':['FLAW-F01','FLAW-P01','FLAW-L01'],'projects':projects,'findings':len(entries),'kinds':dict(counts),'new_source_files':len(sources)}
(root/'results/archive_integrity.json').write_text(json.dumps(auditpres,ensure_ascii=False,indent=2),encoding='utf8')
# Focused updates to cumulative prose; all historical results stay in place.
latest='''## 최신 feature-law Factor 실험

새15개 모델을 동일2048 tasks/256 updates/3 seeds/query CE로 학습했다. Gaussian 평균·heavy-tail 평균·분산 차이 law를 모두 학습하고, 새 graph와 미학습 parameter tuple 조합에서 cell만 바꿨다. Graph encoder/factor는 한 역사적 checkpoint로 고정했다. Law 이름 자체의 미관측 전이나 공개 GraphPFN 개선 모듈은 아니다.

| Primary: heldout96 graphs, 자기 피처 없는 query | Accuracy % | NLL | Reference KL |
|---|---:|---:|---:|
'''
for arm,label in [('original','기존 부호 제약 Factor'),('static','Flexible-static'),('adaptive','Flexible-adaptive'),('fixed_exact','고정 law 주변화 exact'),('fixed_bp','고정 law 주변화 BP'),('mask_pfn','새로 학습한 Mask PFN'),('pcfi_pfn','새로 학습한 PCFI-PFN')]:
    m=p['means'][arm];latest+=f"| {label} | {m['accuracy']*100:.4f} | {m['nll']:.6f} | {m['reference_kl']:.6f} |\n"
latest+='\nAdaptive−static accuracy '+d('adaptive-static')+'%p, adaptive−original '+d('adaptive-original')+'%p다. 두 CI에0을 포함해 적응 primary gate를 통과하지 못했다. 이 cell 교체만으로 메인 메소드가 된다는 가설은 닫고 새 loss/head/학습량으로 구제하지 않는다. 원인은 미상이며 부호 제약 제거 자체나 모든 feature-law 학습을 기각하지 않는다. 모든 seed/episode 평균 뒤 **base graph**를 bootstrap했다.\n\n'
latest+=f"Adaptive {c('adaptive')}, fixed exact {c('fixed_exact')}, fixed BP {c('fixed_bp')}. Reference KL은 broad known-family exact reference에 대한 값으로 참 OOD Bayes KL이 아니다. [보고서]({prefix}REPORT_KO.md), [수식]({prefix}METHOD.md), [구조화 판정](findings/feature-law-factor.yaml).\n\n"
for name,anchor in [('RESULTS_SUMMARY_KO.md','## 실제 그래프에서 확보한 강한 비교'),('RESEARCH_HANDOFF_KO.md','## 최신 Grid6 teacher 결과')]:
    path=repo/name;body=path.read_text(encoding='utf8');assert anchor in body
    body=body.replace(anchor,latest+anchor,1)
    body=body.replace('29개 프로젝트 묶음의130항목',f'{projects}개 프로젝트 묶음의{len(entries)}항목')
    body=body.replace('실패·중단38개, 보존28개, 교훈42개, 미실행22개',f"실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개")
    body=body.replace('130번의 독립 실험',f'{len(entries)}번의 독립 실험').replace('[130개 전체 기록]',f'[{len(entries)}개 전체 기록]')
    body=body.replace('연구 기록 130항목',f'연구 기록 {len(entries)}항목').replace('29개 프로젝트 묶음이며 실패·중단38, 보존28, 교훈42, 미실행22항목',f"{projects}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목")
    body=body.replace('graphpfn_review_packet_2026-11-02_v4.zip','graphpfn_review_packet_2026-11-02_v5.zip')
    if name=='RESEARCH_HANDOFF_KO.md':
        body=body.replace('최근 업데이트: exact Grid6 teacher 후속까지 포함한','최근 업데이트: feature-law Factor cell 교체 결과까지 포함한')
        body=body.replace('v4는 이번 연구 포지션과 다음 후보 기준까지 포함한다.','v5는 feature-law Factor 최종 결과까지 포함한다. v4는 직전 연구 포지션 snapshot이다.')
        body=body.replace('연결 연산·signed evidence·자기 role-update는 이미 구현했어.', '추가로 feature-law3종을 학습한 cell-only 비교에서도 adaptive64.47%, static64.41%, 기존64.63%로 primary 적응 이득을 확인하지 못했어. 부호 제약을 푸는 것만 다시 제안하지 말아줘.\n연결 연산·signed evidence·자기 role-update는 이미 구현했어.')
    else:
        body=body.replace('과거 v1/v2/v3 ZIP','과거 v1/v2/v3/v4 ZIP')
        body=body.replace('새로운 feature law·semantic schema·실제 graph·큰 graph의 teacher/Factor 전이 | 작은 Gaussian family의 새 graph seeds만으로 이 범위를 검증하지 않았다.', 'Unseen feature law·semantic schema·실제 graph·큰 graph의 teacher/Factor 전이 | 새 cell 실험은 law3종을 모두 학습한 parameter tuple 전이이며 이 범위의 검증은 아니다.')
    path.write_text(body,encoding='utf8')
print('ARCHIVED',auditpres)

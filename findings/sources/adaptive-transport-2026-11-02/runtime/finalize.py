"""Append a completed frozen inference study; preserve all previous findings."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil
from collections import Counter
sys.path.append(r'C:/Users/seongjin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import yaml

root=Path(__file__).resolve().parents[1];repo=root.parent/'research-harness'
def read(p):return json.loads((root/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=read('results/statistics.json');a=read('results/audit.json');cost=read('results/cost.json')
complete=read('results/complete.json');latent=read('results/latent_accuracy_audit.json');ca=read('results/classical_audit.json')
p=s['primary'];c=p['contrasts']
assert not s['primary_gate'] and not s['strong_floor']
assert s['gaussian_joint_minus_zero_same_queries']['accuracy']['ci95'][0]>0
def delta(d,m='accuracy'):
    scale=100 if m=='accuracy' else 1
    v,lo,hi=[x*scale for x in [d[m]['delta'],*d[m]['ci95']]]
    return f'{v:+.6f} [95% CI {lo:+.6f}, {hi:+.6f}]'
names={'mask':'기존 Mask Plain','identity':'기존 PCFI Plain·일반 복원','parity':'같은 PCFI Plain·고정 반전 복원',
    'uniform_two':'같은 PCFI Plain·50:50 두 복원 평균','feature_adapt':'같은 PCFI Plain·피처만 사용해 적응',
    'joint_adapt':'같은 PCFI Plain·피처와 context로 적응','known_g_reference':'참 g 복원 reference (privileged)',
    'positive_lp':'양의 LP','signed_lp':'Signed LP','adaptive_lp':'Context 적응 LP (graph-only)',
    'gaussian_joint':'직접 Gaussian/probit 추론','gaussian_label_only':'같은 Gaussian/probit·피처 제거',
    'known_t_lp_reference':'참 t LP reference (privileged)','uniform':'Uniform'}
verdict='참 법칙을 주지 않는 적응 PCFI→frozen PFN은 사전 primary를 통과하지 못했다. 피처만 쓰는 적응과50:50 평균을 정확도에서 확실히 넘지 못했고, NLL/Brier는 두 대조보다 악화했다. 같은 입력의 직접 Gaussian/probit 추론은 유의한 피처 활용 이득을 보였다. 이번 입력 가설 선택기를 새 PFN 메소드로 확대하지 않는다.'
body='# 정답 법칙 없는 적응 전달 — frozen PFN과 직접 추론 비교\n\n'+verdict+'\n\n'
body+='## 설정과 고정한 것\n\n새64 base graphs×32조건=2,048 episodes다. 기존 sign-transport와 같은192-node connected bipartite graph와8개의 상관된 Gaussian-signal column family이며, graph seeds는 새로 생성했다. Label law t와 feature law g를 독립적으로 교차하고, 같은 t에서 g를 바꿀 때 graph·labels·context·query·mask를 고정했다. 개발 graph2개에서 계산을 확인한 뒤 source/config/checkpoint를 lock하고 최종 bank를 생성했다.\n\n'
body+='기존 matched192-update Mask Plain/PCFI Plain의 checkpoint를 각각 seeds11/12/13으로 재사용했다. 이번 새 모델/optimizer updates는0이다. 여기서 Plain은 공개 GraphPFN 기반 모델이며 과거 소형 typed-relation Plain과 다르다. 이 graph family는 기존 학습 prior 밖이다. 따라서 frozen 실패를 새 prior matched 학습이나 전체 PFN 구조의 실패로 해석하지 않는다.\n\n'
body+='Context는 클래스당1/4개다. Feature source는 context 밖의 정확히0/1/8개 또는 full이다. 1/192와8/192는 각각99.48%,95.83% whole-node missing이며 Bernoulli99.5% 조건이 아니다. Primary는 sources8·k4/class에서 자기 feature가 없는 query이며 네(t,g) 조합과 세 고정 training seeds를 각 graph 안에서 평균한다. 64개 graph 단위 paired bootstrap5,000회를 사용했다. Training-seed population CI가 아니며 subgroup CI는 다중비교 미보정이다.\n\n'
body+='## 어떤 적응을 계산했나\n\nAdjacency만으로 bipartition b와 저주파 Gaussian covariance K를 구성한다. 관측행만으로 계산한 leading uncentered singular score를 feature evidence로 사용했다. (g,t,상대 방향 σ,amplitude)의24개 가설에서 Gaussian feature density와 context-label orthant probability를 결합하고, 같은 evidence로 query sign probability를 계산한다. Context 적분은 고정 Sobol/antithetic2,048 particles다. 법칙·query 정답·hidden feature는 적응 입력에 주지 않았다.\n\n'
body+='Feature-only q(g)는 labels/context를 읽지 않는 별도 함수다. Joint q(g)는 context labels까지 이용하고 t·σ·amplitude를 주변화한다. PFN 경로는 일반 PCFI와 parity-PCFI 두 completion에 같은 checkpoint를 각각 실행해 q(g)로 확률을 평균한다. 50:50 평균도 동일한 두 PFN branches를 사용한다. 참 g를 받은 reference는 별도 평가용이며 실행 가능한 적응 성과가 아니다.\n\n'
body+='직접 Gaussian/probit 대조는 같은24개 가설에서 query 확률을 직접 주변화하며 PFN을 사용하지 않는다. 이는 알려진 family의 Gaussian covariance/score-noise 가정을 사용하는 강한 통계 대조다. 모델은 원8columns의 joint, RMS 압축, median threshold, class-balanced context 선택의 정확한 likelihood가 아니다. 원 데이터 exact Bayes posterior·calibration·새 neural 추론의 성과로 부르지 않는다. CPGNN/CLP 재현도 아니다.\n\n'
body+='## Primary: 네 label/feature 법칙 평균\n\n| 방법 | Accuracy % | NLL | Brier |\n|---|---:|---:|---:|\n'
for method,v in p['means'].items():body+=f"| {names[method]} | {v['accuracy']*100:.6f} | {v['nll']:.6f} | {v['brier']:.6f} |\n"
body+='\n적응 PCFI→PFN에서 대조를 뺀 paired 차이다. Accuracy 단위는%p이며 NLL/Brier는 낮을수록 좋다.\n\n| 대조 | Accuracy 차이 [95% CI] | NLL 차이 [95% CI] | Brier 차이 [95% CI] |\n|---|---:|---:|---:|\n'
for method in ['uniform_two','feature_adapt','mask','identity','parity','adaptive_lp','gaussian_joint']:
    d=c['joint_adapt-'+method];body+=f"| {names[method]} | {delta(d)} | {delta(d,'nll')} | {delta(d,'brier')} |\n"
body+='\n사전 primary=False, strong floor=False다. 일반 PCFI 하나와의 개선은 보존하되, 같은 두 completion의 고정 평균과 feature-only 적응을 넘지 못한 결과를 새 context 적응 메소드의 성공으로 바꾸지 않는다. 참 g를 받는 reference도 강한 classification upper bound가 아니다. True feature law에 맞게 복원하는 것과 frozen classifier의 최종 예측에 좋은 입력은 다를 수 있다.\n\n'
body+='## 피처의 예측 기여: query 집합을 고정한 제거 대조\n\n양쪽 예측 모두 원8-source primary의 missing-query 집합으로 제한했다. 관측 source 수가 바뀌면서 query 집합도 바뀌는 효과를 섞지 않았다.\n\n| 경로 | 피처 있음 accuracy % | 피처 제거 accuracy % | 차이%p [95% CI] | NLL 차이 [95% CI] |\n|---|---:|---:|---:|---:|\n'
for method,key,zero in [('joint_adapt','joint_adapt_minus_zero_same_queries','zero_source_primary_cohort_means'),('gaussian_joint','gaussian_joint_minus_zero_same_queries','gaussian_zero_source_primary_cohort_means')]:
    body+=f"| {names[method]} | {p['means'][method]['accuracy']*100:.6f} | {s[zero]['accuracy']*100:.6f} | {delta(s[key])} | {delta(s[key],'nll')} |\n"
body+='\n직접 통계 추론은 피처가 context와 분리된 상태에서도 accuracy·NLL·Brier의 추가 이득을 보였다. 반면 frozen PFN의 적응 completion은 피처 제거 대비 NLL/Brier를 개선했지만 accuracy 우위를 확인하지 못했다. 이 차이는 통계 모형/branch decoder/기존 prior mismatch 등 여러 요소가 섞여 있어 고유 원인을 분리한 결과는 아니다.\n\n'
body+='## Label/feature 법칙별 보조 결과\n\nSources8,k4/class. 이 네 조건의 평균이 위 primary이며 가장 좋은 한 cell로 primary를 교체하지 않는다.\n\n| t | g | 일반 PCFI % | 50:50 % | 피처만 적응 % | Context 적응 % | 직접 Gaussian % | 적응 LP % |\n|---|---|---:|---:|---:|---:|---:|\n'
for t in [0,1]:
    for g in [0,1]:
        means=s['cells'][f't{t}_g{g}_k4_o8']['means']
        body+=f'| {t} | {g} | '+' | '.join(f"{means[m]['accuracy']*100:.6f}" for m in ['identity','uniform_two','feature_adapt','joint_adapt','gaussian_joint','adaptive_lp'])+' |\n'
body+='\n## 법칙 식별과 source/context 수\n\n현재 family는 feature column의 숨겨진 방향이 대칭적이다. Source0/1에서는 g를 바꾼 관측값의 부호를 숨겨진 orientation으로 흡수할 수 있어 g에 대한 추가 정보가 없다. Context를 넣어도 q(g)는0.5로 남았다. 이 식별 불가능성을 복잡한 router로 해결할 수 있다고 가정하지 않는다. 이는 현재 family의 g에 관한 결과이며 임의 데이터의 모든 one-source 식별 불가능성을 주장하지 않는다.\n\n| k/class | Sources | Feature-only g 정확도 % | Joint g 정확도 % | Label-only t 정확도 % | Joint t 정확도 % |\n|---|---|---:|---:|---:|---:|\n'
for k in [1,4]:
    for o in [0,1,8,'full']:
        q=latent['groups'][f'k{k}_o{o}'];body+=f'| {k} | {o} | '+' | '.join(f"{q[f]['fractional_tie_accuracy']*100:.6f}" for f in ['feature_g1','joint_g1','label_t1','joint_t1'])+' |\n'
body+='\n법칙 정답은 이 사후 진단에만 사용했다. 수치상0.5 근처(1e−8)의 tie는 기대 정확도0.5로 집계했다. 이 보조 규칙은 weight 결과를 보기 전에 runtime/latent_accuracy_addendum.json에 기록했고 원 statistics/weights/예측을 보존했다. 실제 query argmax accuracy와 primary는 수정하지 않았다. 원 statistics의 strict q>0.5 latent accuracy 대신 latent_accuracy_audit.json을 참조한다. Source0/1의 g 확률은 최대 약4.4e−16만큼0.5와 달랐다.\n\n'
body+='8-source,k4에서 context가 g의 hard identification을 개선했다는 결과는 없다. Joint t 점추정이 label-only보다 높더라도 latent 식별 정확도를 최종 PFN 분류 개선으로 바꾸지 않는다. 같은 Gaussian 추론기가 직접 예측한 결과와 PFN 입력 가중치로 쓴 결과가 다르다.\n\n'
body+='## 비용\n\n| 경로 | Median latency ms | 추가 peak GPU work memory MiB |\n|---|---:|---:|\n'
for method,v in cost['methods'].items():body+=f"| {names[method]} | {v['median_ms']:.6f} | {v['additional_peak_mib']:.3f} |\n"
body+='\nRTX3090, seed11/첫 primary input,5warmups/30repeats다. CPU visible input→CPU 확률이며 kernel/PCA/orthant와 실제 PCFI/PFN branches를 포함하고 model load·bank I/O는 제외한다. Across-graph kernel cache는 없다. 같은 PCFI checkpoint를 쓰지만 적응/50:50는2회 PFN 실행이 필요하다. External GPU load는 통제하지 않았고 메모리는 추가 작업량이지 총 GPU 메모리가 아니다. 단일 입력의 비용을 일반 scaling 우위로 확대하지 않는다.\n\n'
body+='## 검증과 보존\n\n'
body+=f"저장된 neural variant probabilities {a['neural_variant_probabilities_rescored']:,}개를 재채점했다. Classical controls {ca['classical_probability_arrays_checked']:,}개는 finite/정규화·metric 계산을 확인했다. 그중 label-only 네 대조는 graph/label/context가 같은256그룹에서 피처-law/source 변경에 불변인지 확인했다. 저장된 PFN mixture 재계산 오차는 {ca['stored_neural_mixture_max_error']:.9g}였다. Neural replay18건, working-model replay3건, 이전 sign-probe checkpoint replay18건을 확인했다. Source/config/입력/예측/checkpoint SHA와 모델 파라미터 불변, 새/기존 graph overlap0, source0/1 대칭성을 확인했다.\n\n"
body+='개발 검사에는 해석 가능한 Gaussian orthant 해, class permutation/global gauge, node permutation 수치오차,2048→4096 particle 근사 비교와 strict feature-only label blindness가 포함된다. 일부 classical 확률이 합산 roundoff로[0,1]을 최대1.47e−14 벗어나 audit의1e−12 수치 허용치로 기록했다. Raw probabilities를 바꾸지 않았으며 합 정규화 오차는8.89e−16 이하다.\n\n'
body+='## 판정과 닫지 않은 범위\n\n'+verdict+'\n\n'
body+='닫는 것은 이번 Gaussian/probit q(g)를 기존 frozen PCFI classifier의 두 completion 가중치로만 사용하는 구현이다. 모든 adaptive propagation, 새 prior에서 직접 학습한 PFN, context-conditioned feature inference를 기각하지 않는다. 반대로 아직 하지 않은 재학습을 이번 실패의 해결책이라고 단정하지도 않는다. 새 구조를 논의하려면 알려진 법칙/고정 Gaussian 추론이 이미 하는 기능과 구분되는 연산, 강한 직접 추론 대조, 동일-query 피처 활용·accuracy·비용 기준을 먼저 제시해야 한다. 이 bank에서 temperature/noise/amplitude/particle/checkpoint를 조정하거나 새 head/loss로 구제하지 않았다.\n'
(root/'REPORT_KO.md').write_text(body,encoding='utf8')

prefix='findings/sources/adaptive-transport-2026-11-02/';dest=repo/prefix;assert not dest.exists();dest.mkdir(parents=True)
paths=list(root.glob('*.py'))+[root/p for p in ['protocol.json','METHOD_KO.md','REPORT_KO.md','bank/development.json','bank/manifest.json','runtime/latent_accuracy_addendum.json']]
paths+=list((root/'results').rglob('*.json'))+list((root/'logs').glob('*.log'))+list((root/'runtime').glob('*.py'))
manifest=[]
for path in sorted(set(paths)):
    q=dest/path.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,q)
    manifest.append({'alias':path.relative_to(root).as_posix(),'original':str(path),'snapshot':q.relative_to(repo).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'date':'2026-11-02','scope':'Unknown-law frozen adaptive transport versus fixed Gaussian/probit inference;0 new models/updates. Verbatim sources; raw arrays and checkpoints remain local.','sources':manifest},ensure_ascii=False,indent=2),encoding='utf8')
refs=[prefix+f for f in ['REPORT_KO.md','METHOD_KO.md','protocol.json','results/statistics.json','results/audit.json','results/classical_audit.json','results/latent_accuracy_audit.json','runtime/latent_accuracy_addendum.json','results/cost.json','results/preflight.json']]
numbers='; '.join(f"{m}:accuracy{v['accuracy']*100:.6f}%/NLL{v['nll']:.6f}/Brier{v['brier']:.6f}" for m,v in p['means'].items())
comparisons='; '.join(m+' accuracy%p '+delta(c['joint_adapt-'+m]) for m in ['uniform_two','feature_adapt','identity','gaussian_joint'])
evidence='Fresh64 base graphs,32conditions each,3 fixed historical training seeds;5000 graph-paired bootstrap. Six frozen public GraphPFN-based checkpoints reused;0 new models/updates. Known bipartite sign-gauged Gaussian-signal family, primary8sources/k4 per class/all4 label-feature-law cells.'
failure={'id':'ADTR-F01','kind':'failure','claim':'관측 feature와 context로 추론한 q(g)를 두 PCFI completion의 가중치로 쓰면 frozen PFN이 고정 평균·feature-only 적응보다 정확해진다.',
    'observation':numbers+'; '+comparisons+'. Primary=False,strong_floor=False.',
    'verdict':verdict,'failure_reason':'정확도 primary 두 CI가0을 포함하고 NLL/Brier는 두-completion 평균과 feature-only보다 악화했다. Known-family Gaussian/probit는 직접 예측에서 더 좋았다. 입력 후보·prior mismatch·decoder·weight 추정 중 고유 원인은 분리하지 않았다.','reason_status':'unknown',
    'closed_scope':'고정 Gaussian/probit24가설에서 q(g)를 계산하고, 기존192-update PCFI Plain의 identity/parity completion2개 예측만 가중 평균하는 frozen 진단 및 이를 근거로 한 자동 확대.',
    'not_closed':'새 prior matched 학습, 내부 관측-라벨 공동추론, 임의 heterophily·heterogeneous schema와 실제 graph 전이를 기각하지 않는다. 이번에 하지 않은 neural training을 학습 실패로 세지 않는다. 직접 통계 추론의 피처 활용 이득은 보존한다.',
    'reopen_when':'추정한 법칙을 입력 가중치로만 전달하지 않고 classifier의 조건부 추론에 활용하는 구체적 연산이 생기며, 고정 직접 Gaussian/signed 추론이 하는 기능과 학습의 추가 가치를 분리할 수 있을 때.',
    'comparison_floor':'같은 정보·2branches의50:50와 strict feature-only; Mask/identity/parity PCFI; graph-only adaptive LP; 직접 Gaussian/probit; 동일-query zero-feature; privileged known-law는 reference로만 보고; 비용도 비교.',
    'recheck':'현재 bank를 조정하지 않는다. Label/feature law 교차와 source-context 분리를 유지한 새 graph bank에서 missing-query accuracy·동일-query 피처 제거·NLL/Brier·실제 비용을 사전 고정해 확인한다.',
    'tags':['observation-mask','graph-structure-use','adaptation','eval-protocol','inference-cost'],
    'keywords':['adaptive PCFI','feature-only evidence','context joint evidence','two completion average','frozen GraphPFN','known-family Gaussian probit','feature heterophily','label heterophily'],
    'reopen_tags':['adaptation','observation-mask','graph-structure-use'],'reopen_keywords':['context conditional feature transport','decoder conditional law','joint relational feature inference'],
    'source_refs':refs,'evidence':evidence,'revisit_status':'open'}
preserved={'id':'ADTR-P01','kind':'preserved','claim':'같은 관측 feature·분리된 context labels를 작은 Gaussian/probit working model에서 직접 결합하면, 이 family에서 피처 없는 예측보다 missing-query classification이 개선된다.',
    'observation':f"Gaussian accuracy{p['means']['gaussian_joint']['accuracy']*100:.6f}%,without-features{s['gaussian_zero_source_primary_cohort_means']['accuracy']*100:.6f}%; paired accuracy%p {delta(s['gaussian_joint_minus_zero_same_queries'])}; NLL {delta(s['gaussian_joint_minus_zero_same_queries'],'nll')}; Brier {delta(s['gaussian_joint_minus_zero_same_queries'],'brier')}. Frozen joint-adaptive PFN accuracy{p['means']['joint_adapt']['accuracy']*100:.6f}%; PFN-Gaussian accuracy%p {delta(c['joint_adapt-gaussian_joint'])}.",
    'do_not_claim':'고정 known-family 통계 추론의 보존 결과다. 새 neural PFN 학습·novelty·참 Bayes posterior·calibration·실제 graph/schema·unknown Gaussian covariance 일반화의 성과가 아니다. 이 결과만으로 adaptive completion의 실패 원인을 특정하지 않는다. 보조 feature-intervention CI이며 primary 성공으로 바꾸지 않는다.',
    'tags':['graph-structure-use','observation-mask','eval-protocol'],'keywords':['Gaussian probit','disjoint feature label inference','same query feature ablation','classical strong baseline'],
    'source_refs':refs,'evidence':evidence}
lesson={'id':'ADTR-L01','kind':'lesson','claim':'법칙 식별 가능성, 입력 후보의 가중치, 조건부 예측 연산과 비용을 구분한다.',
    'observation':'0/1source q(g)는수치오차내0.5다. 8source/k4 feature-only g accuracy60.15625%,joint59.375%; label-only t60.9375%,joint64.0625%. Primary joint PCFI52.555985%,50:50 52.354767%,feature-only52.461961%,direct Gaussian56.154827%. Gaussian은 동일-query feature-removal accuracy이득을 보였지만 adaptive PFN은 확인하지 못했다.',
    'interpretation':'숨겨진 orientation 대칭으로 정보가 없는 g는 router로 복원할 수 없다. Law weight 진단과 classification을 혼동하지 않는다. 같은 q를 입력 선택에 쓰는 것과 일관된 conditional decoder에 쓰는 것은 다르며 고유 원인은 미상이다. 두 PFN forward·적분 비용을 포함한다. Secondary latent0.5 ties는 결과 열람 전 선언한1e-8 범위에서 기대0.5로 집계하고 원 raw/statistics를 보존한다. Query cohort 고정·기존 기록 보존을 유지한다.',
    'tags':['eval-protocol','adaptation','inference-cost','observation-mask'],'keywords':['latent orientation symmetry','identifiability','numerical ties','two forward cost','law inference versus prediction','no new training'],
    'source_refs':refs,'evidence':evidence}
finding={'project':'Unknown-law adaptive feature transport around frozen GraphPFN','date':'2026-11-02','source':prefix+'REPORT_KO.md',
    'setting':'Known bipartite Gaussian-signal family;64 new graphs/2048episodes;all4 label-feature-law cells;sources0/1/8/full;k1/4 per class;6 frozen checkpoints;0 new models/updates.','related':['GraphPFN'],'entries':[failure,preserved,lesson]}
(repo/'findings/adaptive-transport.yaml').write_text(yaml.safe_dump(finding,allow_unicode=True,sort_keys=False,width=170),encoding='utf8')
old=read('runtime/previous_harness.json');entries=[];projects=0
for path in (repo/'findings').glob('*.yaml'):
    if path.name.startswith('_'):continue
    projects+=1;entries+=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('entries',[])
hashes={e['id']:hashlib.sha256(json.dumps(e,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for e in entries}
assert all(hashes[k]==v for k,v in old['entries'].items())
assert all(sha(repo/p)==v for p,v in old['sources'].items())
counts=Counter(e['kind'] for e in entries)
(root/'results/archive_integrity.json').write_text(json.dumps({'old_findings_unchanged':len(old['entries']),'old_sources_unchanged':len(old['sources']),'new_ids':[e['id'] for e in finding['entries']],'counts':dict(counts)},ensure_ascii=False,indent=2),encoding='utf8')
section='## 참 법칙 없는 적응 전달과 직접 통계 추론\n\n새64graph·2,048episodes에서0 training으로 비교했다. Primary는 sources8,k4/class,네 label/feature-law 조합 평균이다. 8/192관측=95.83% whole-node missing이며99.5% Bernoulli 조건이 아니다.\n\n| 경로 | Missing-query accuracy % | NLL |\n|---|---:|---:|\n'
for method in ['identity','uniform_two','feature_adapt','joint_adapt','adaptive_lp','gaussian_joint']:
    v=p['means'][method];section+=f"| {names[method]} | {v['accuracy']*100:.6f} | {v['nll']:.6f} |\n"
section+='\nContext 적응−50:50 accuracy '+delta(c['joint_adapt-uniform_two'])+'%p, context 적응−feature-only '+delta(c['joint_adapt-feature_adapt'])+'%p. Primary=False다. 직접 Gaussian/probit의 동일-query feature-removal 차이는 '+delta(s['gaussian_joint_minus_zero_same_queries'])+'%p로 보존한다. 새 neural 학습이나 novelty의 성과가 아니며 frozen 후보 선택기를 확대하지 않는다. [보고서]('+prefix+'REPORT_KO.md), [고정 프로토콜]('+prefix+'protocol.json), [구조화 판정](findings/adaptive-transport.yaml).\n\n'
for name in ['RESULTS_SUMMARY_KO.md','RESEARCH_HANDOFF_KO.md']:
    path=repo/name;body=path.read_text(encoding='utf8');anchor='## 부호 전달과 기존 frozen GraphPFN 진단';assert anchor in body
    body=body.replace(anchor,section+anchor,1)
    body=body.replace('33개 프로젝트 묶음의139항목',f'{projects}개 프로젝트 묶음의{len(entries)}항목').replace('연구 기록 139항목',f'연구 기록 {len(entries)}항목')
    body=body.replace('실패·중단41개, 보존30개, 교훈46개, 미실행22개',f"실패·중단{counts['failure']}개, 보존{counts['preserved']}개, 교훈{counts['lesson']}개, 미실행{counts['untested']}개")
    body=body.replace('33개 프로젝트 묶음이며 실패·중단41, 보존30, 교훈46, 미실행22항목',f"{projects}개 프로젝트 묶음이며 실패·중단{counts['failure']}, 보존{counts['preserved']}, 교훈{counts['lesson']}, 미실행{counts['untested']}항목")
    body=body.replace('139번의 독립 실험',f'{len(entries)}번의 독립 실험').replace('[139개 전체 기록]',f'[{len(entries)}개 전체 기록]')
    body=body.replace('graphpfn_review_packet_2026-11-02_v8.zip','graphpfn_review_packet_2026-11-02_v9.zip')
    body=body.replace('v8은 부호 전달 frozen 진단까지 포함한다.','v9는 참 법칙 없는 적응 전달과 직접 통계 추론까지 포함한다. v8은 직전 부호 전달 frozen 진단 snapshot이다.')
    body=body.replace('과거 v1/v2/v3/v4/v5/v6/v7 ZIP','과거 v1/v2/v3/v4/v5/v6/v7/v8 ZIP')
    body=body.replace('최근 업데이트: 부호 전달 진단까지 포함한','최근 업데이트: 참 법칙 없는 적응 전달까지 포함한')
    if name=='RESEARCH_HANDOFF_KO.md':
        body=body.replace('1. [최신 부호 전달 진단]', '1. [최신 적응 전달 비교]('+prefix+'REPORT_KO.md)와 [부호 전달 진단]',1)
        i=body.rfind('```text');assert i>=0;pre,tail=body[:i],body[i:];j=tail.rfind('\n```');assert j>=0
        tail=tail[:j]+'\n추가로 참 부호 법칙을 주지 않고 Gaussian/probit evidence로 두 PCFI 입력 후보를 가중 평균했어. Context 적응−50:50는 '+delta(c['joint_adapt-uniform_two'])+'%p로 primary를 통과하지 못했어. 같은 정보를 직접 처리한 고정 Gaussian/probit는56.154827%였고 feature-removal 대비 '+delta(s['gaussian_joint_minus_zero_same_queries'])+'%p였어. 새 학습은0이야. 이미 구현한 통계 추론 기능과 학습의 독립적인 필요성을 구분해 줘.\n'+tail[j:];body=pre+tail
    path.write_text(body,encoding='utf8')
path=repo/'README.md';body=path.read_text(encoding='utf8').replace('내부 고정 PCFI 재검증과 부호 전달 frozen 진단','내부 고정 PCFI 재검증·부호 전달 frozen 진단·참 법칙 없는 적응 전달 비교');path.write_text(body,encoding='utf8')
path=repo/'tools/build_review_packet.py';body=path.read_text(encoding='utf8');assert 'VERSION = 8' in body;path.write_text(body.replace('VERSION = 8','VERSION = 9',1),encoding='utf8')
print('REPORT_AND_ARCHIVE_READY',len(entries),dict(counts),flush=True)

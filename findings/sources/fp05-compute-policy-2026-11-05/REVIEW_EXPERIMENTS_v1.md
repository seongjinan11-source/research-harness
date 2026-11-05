# FP05 experiments/statistics/budget review

2026-11-05. 대상은 `PLAN_v1.md`다. FP04 `DECISION_KO.md`/`RESPONSE_RESULTS.md`, observation-budget pilot 및 internal-PCFI pilot/confirmation의 실제 보고서를 함께 읽었다. 다른 현재-cycle review 파일은 읽지 않았다. Source/plan/branch 수정, GPU·새 실험, agent 재귀 호출은 하지 않았다.

**판정: 3/5, 아래 경계 정의를 V2/protocol에서 닫은 뒤 고정된 한 번의 실제 실험으로 진행할 수 있다.** 공개 pretrained predictor를 사용하고 CE-only000, uniform computation, exact expected-gradient, sampled policy-gradient를 비교하는 것은 FP04보다 강한 다음 검증이다. 추가 architecture/head/loss sweep나 새 진단 bank를 요구하지 않는다. 현재 목표는 observation 획득이나 새 imputation이 아니라, 공유 PCFI 입력 위에서 inference computation을 조절하는 학습의 증분이다.

## 축적된 결과에서 가져와야 할 조건

FP04에서는 matched arms보다 observed가 +0.423%p 높았지만 CI가0을 포함했고, conditional observed−none은+0.043%p, observed−shuffle은−0.326%p였다. Real arm 우위도 none에서 대체로 유지됐다. 따라서 FP05의 CE/정책 우위와 실제 feature-value utility를 계속 분리해야 한다. FP04의 항상 feature-free query 훈련은 관측 query 학습 가능성을 직접 검증하지 않았으나, 그것이 유일한 실패 원인이라고 확정된 것은 아니다.

Observation-budget pilot은 observation-aware policy가 structure/PPR보다 좋았어도 evidence-paths를 확실히 넘지 못했고, context-frequency probability baseline의 NLL이 더 낮았다. Internal-PCFI의 context router 역시 fixed internal 대비 primary accuracy가 낮았으며, 개발에서 선택한 fixed placement의 +0.657%p 신호는 새 bank에서+0.220%p, CI[−0.332,+0.797]로 확인되지 않았다. 이번에는 fixed extra depth·train-selected global trace·단순 prior를 약한 대조 뒤로 숨기거나 같은 final을 반복 튜닝하지 않는 것이 핵심이다. 이 기록들은 모든 RL/PFN/internal computation의 불가능성을 뜻하지 않는다.

## 주요 수정 1 — primary unit, 빈 query와 full guard를 완결해야 한다 [blocking]

Primary mask를 **uniform whole-node95와99.5의 두 조건**으로 명시하고, clustered/cell은 secondary로 고정한다. 현재 표현만으로는 clustered 두 조건도 primary인지 불명확하다. 각 base의 query16을 먼저 고정하고, 원래 물리적 mask에서 own row가 전부 missing인 subset `Q_gm`을 정의한다. Dummy/padding feature는 판정에서 제외한다. 이 subset은 모든 arms/seeds 및 none/shuffle에서도 그대로 사용한다. None에서 query가 모두 missing이 됐다고 더 많은 query를 새로 평가하면 paired effect가 달라진다.

권고하는 최소 null 규칙: 기존의 **all96 graphs/two rates** estimand를 유지하려면, 어느 primary graph-rate든 `Q_gm`이 비어 있을 때 metric을 null로 보존하고 해당 primary gate를 undefined/non-PASS로 처리한다. Mask나 graph를 다시 뽑아 구제하지 않는다. 모든 case가 유효할 때만 graph 안에서 고정3 training seeds×2 rates를 평균하고 graph 차이를 bootstrap한다. 빈 case를 평균에서 조용히 빼면서 계속 “all96/two rates”라고 쓰면 안 된다. Query/node/trace/seed를 독립 표본으로 resample하지 않는다.

Final96의 구성을 **requested shape2 × context-cap3(k1/3/5) × query/source design2 × repetition8**로 고정하면 12 strata×8로 집계가 명확해진다. K2..4는 official generator의 고정 분포에서 뽑고 실제 K별 결과를 보조 보고하며, class outcome으로 graph를 reject하지 않는다. Requested/actual N,F,K 및 구조적 retry seed도 보존한다. Bootstrap5000은 이 사전 strata 안에서 graph를 재추출한다. Training seeds는 고정된 세 모델 반복으로, seed population CI가 아니다.

Full guard는 실제 query feature가 관측된 **natural/full48 graphs**에서만 정의한다. Disjoint의 full-nonquery-visible을 완전 관측 guard에 합치지 않는다. `Δ=GRPO−fixed accuracy`의 paired **95% CI lower>−0.02**로 noninferiority를 고정하고, 점추정이−2%p보다 크다는 것만으로 PASS시키지 않는다. 네97.5% superiority/utility intervals와 별도 full guard를 전체 동시95% coverage로 부르지 않는다.

## 주요 수정 2 — 50:50 query support와 mask 분모를 정확히 설명해야 한다 [blocking]

절반이 natural mask, 절반이 forced-missing/disjoint라는 것은 **관측 query와 feature-free query가 절반씩이라는 뜻이 아니다**. Natural node95/995에서도 query 대부분은 missing이다. 이 설계를 유지할 수는 있지만, 이를 50:50 mask regimes라고 명명하고 actual observed/missing-query CE 수를 setting별로 기록해야 한다. Full-natural training이 실제 schedule에서 몇 update/query를 차지하는지도 사전에 보장한다. 간단한 고정 권고는128 updates 중16을 full-natural로 예약하고 나머지112를 고정 분포의 나머지 settings/regimes에 배정하는 것이다. 모든 arm은 같은 schedule을 쓴다. 이를 넘어 추가 trainability sweep를 요구하지 않는다.

Uniform node/cell은 Bernoulli인지, cluster는 exact rounded observed-cluster인지 등 mask 생성법과 nested RNG도 protocol에 적어야 한다. 기존 local mask는 uniform Bernoulli와 cluster rounded count를 섞는다. 특히 N128…192의 cluster99.5는 반올림 관측 수가 보통1이라 source>=2가 거의/전혀 없을 수 있다. 이 상태를 자연스럽게 보존하고 source0/1 counts를 보고한다. Cluster/cell을 학습에 넣었다면 같은 mechanism을 heldout-mask OOD라고 부르지 않는다.

Utility는 disjoint graph에서 **원 standard mask 기준 source>=2 및 |Q_gm|>=1인 rate cases**만 고른 별도 conditional estimand다. Eligible rates와 고정 seeds를 graph 안에서 먼저 평균하고, eligible graph마다 같은 weight를 준다. Planned/eligible graph 수·case 수·source0/1 및 빠진 strata를 보고하고, 전체 eligible graph가0이면 undefined다. None는 mask/value 모두 제거하며, shuffle은 같은 physical observed rows를 permute한다. RMS/PCFI를 새로 계산하고 policy도 intervened input에서 다시 실행한다. 이 효과는 rerouting까지 포함한 전체 pipeline feature utility이며, 입력 효과만으로 고정 action의 효능을 증명하는 것은 아니다. Root가 추가하겠다고 한 utility difference-in-differences vs fixed는 secondary로 고정하면 좋다.

## 주요 수정 3 — RP, normalization, PCFI의 순서와 reference 명칭을 잠가야 한다 [blocking]

Real whole-node masks는 관측 행만 투영할 수 있으므로 RP16에서도 row-level missing semantics를 보존할 수 있다. 다만 **native PCFI→RP와 RP→PCFI는 같은 함수가 아니다.** “FP04 native bank를 읽고 RP16”이라는 설명만으로는 입력 연산이 잠기지 않는다. 권고 순서는 immutable physical raw observed rows → feature-independent fixed Rademacher RP16 → 관측-only column RMS → literal100-step/alpha.9 PCFI → represented clip10이다. R은 rawF×16, ±1/sqrt(rawF), graph별 독립 seed를 사전에 고정하고 모든 reps/modes/arms/train seeds에서 재사용한다. 값이나 labels로 fit하지 않는다. NaN reference는 같은 RP16/RMS visible input의 hidden rows를 NaN으로 둔다.

Native 원값은 FP04의 RMS/clipped visible array만으로 복원하지 말고 immutable raw source/기존 mapping에서 읽어야 한다. 물리적 row/context/query/graph identity와 raw/projection/processed input hashes를 따로 남긴다. Projection의 상쇄로0이 된 값도 관측값이다. Cell mask를 미래 real 조건으로 추가하면서 zero-filled partial native row를 투영한 모든 RP 좌표를 “관측”으로 표시하면 안 된다. 이번 real panel은 whole-node만으로 제한하면 그 문제가 없다.

같은 RP16 전처리의 **frozen NaN000, frozen PCFI000**를 pipeline reference로 보존한다. 이를 “native-NaN GraphPFN”이라 부르면 native feature 차원을 그대로 사용한 것처럼 보인다. Historical full-native/public preprocessing/PCA/ensemble PCFI/FISF references와 LP도 그대로 제시하되, 정보 표현·capacity·학습 예산이 다른 비교라고 명시한다. Panel은 **Cora2708/CiteSeer3327 full graphs, PubMed inherited2000 induced graph**, 3 fixed context/mask reps×5 modes×3 trained seeds다. 새12 fits의 real saved records는540개이며, external frozen records의3 repetitions를 새로운3 training seeds처럼 복제하지 않는다. Synthetic gate 실패와 무관하게 전부 실행하고 새 real-population CI는 만들지 않는다.

## 주요 수정 4 — global trace calibration으로 adaptivity 해석을 지켜야 한다

Uniform8-trace inference mixture는 강한 control이지만8 forwards이고, greedy policy는1 trace다. Train-only best-global trace는 한 개 forward라 별도로 유용하다. 선택을 “train calibration”이라고만 쓰지 말고 다음을 고정한다: **처음16개로 사전 지정한 train base graphs × primary2 masks ×8 traces**, 마지막 checkpoint로 평균 `CE+lambda*extra_cost`를 최소화하고, exact tie는 낮은 cost 후 lexicographic 순으로 선택한다. Query CE를 쓰는 supervised train-only 선택이며 label-free calibration이 아니다. Dev/final/real score로 trace나 선택 criterion을 고르면 안 된다. Calibration input indices/forward RNG/predictions/aggregate costs와 선택된 bits를 저장한다.

Uniform checkpoint의 global trace와 함께 **group-policy 자기 checkpoint의 train-selected global trace를 secondary로 재생**하면, 좋은 adapter/depth를 학습한 효과와 episode별 routing 효과를 더 잘 분리할 수 있다. 새 fit은 필요 없다. GRPO가 fixed/direct를 이겨도 자기 global trace를 넘지 못하면 “해당 training objective의 predictor 개선”은 보존할 수 있지만, adaptive episode routing의 증분은 입증됐다고 하면 안 된다. 이 보조 control은 primary winner 선택에 쓰지 않는다.

과거 observation-policy의 교훈을 반영해 **uniform probability 및 Laplace1 context-frequency probability**를 이번에는 사전 reference로 포함한다. 계산 비용이 거의 없으며 새 학습 arm도 아니다. Capped contexts의 class-ID argmax tie에 accuracy가 민감할 수 있으므로 NLL/Brier를 같이 기록한다. Query truth는 이 prior 구성에 쓰지 않는다.

## 주요 수정 5 — trajectory 수와 실제 계산·평가 예산을 구분해야 한다

검토 중 root는 update별 feature-position RNG1개를 모든 arm/8 traces에 공유하고 action RNG만 분리하겠다고 알렸다. 이 수정이 필요하다. 서로 다른 embedding RNG를 trace bits에 붙이면 action 효과와 augmentation 효과가 섞이고 exact8-trace distribution조차 불명확해진다. Fixed000의8회는 같은 task/RNG의 duplicate presentations임을 보존한다. One-step clipping은 최초 ratio1에서 활성화되지 않으므로 root가 제안한 **group-normalized REINFORCE/group_pg, entropy0** 정정도 실제 estimator와 맞는다. PPO multiple-epoch/GRPO clipping 효과를 주장하지 않는다.

12 fits×128 updates×8=12,288 trajectory presentations지만, **episode-level update exposures는1536회**이며 같은 update의8 rollouts는8 independent tasks가 아니다. 실제 사용한 graph/setting unique counts·trace duplicates·zero-advantage fraction·entropy·extra-adapter calls·optimizer updates·grad retries를 기록한다. Direct의 exact expectation과 group sampling의 서로 다른 objective/variance는 의도된 차이로 남긴다. Same8 presentations는 equal total FLOPs/실제 walltime/gradient informativeness를 보장하지 않는다.

V1의 final7 standard+4 interventions를 그대로 세면1056 episodes/checkpoint, saved synthetic outputs는12,672개다. Uniform mixture8 forwards 때문에 trained arms의 synthetic PFN full-pass 수는 **34,848**, real45 episodes까지 더하면 **36,333**이다. Frozen/global references, train calibration, discarded dev smoke, checks/retries는 이 밖이다. Prefix caching을 쓰면 full-pass 등가치와 실제 layer/adapters 실행 수를 별도로 기록한다. None가 같은 input/query16의 두 rate-copy라면 shared none cache/prediction을 한 번 저장하고 각 rate의 원래 subset으로 score를 계산해도 되지만, protocol의 episode/output counts를 그에 맞게 먼저 수정한다. 이것을 독립 두 case로 해석하면 안 된다.

실제 real GPU pass가 학습보다 총비용을 지배할 수 있다. Resource smoke는 N/F native→RP 변환·PCFI cache·GPU transfer·policy/all8 traces·forward/backward·저장 시간을 분리해서 projected total walltime과 peak를 제시한다. 3090 FP16의 loss scaling retry는 성공 optimizer update 수를 줄이거나 task를 건너뛰는 장치가 아니어야 한다. Finite update 실패는 원인/중단을 보존하고, 같은 final을 보고 예산을 늘려 구제하지 않는다. 지금은 임의 축소를 권하지 않는다. 정해진 smoke 후 성능을 보지 않고 수정이 필요한 경우에만 한 번의 reviewed prelock budget revision으로 닫으면 된다.

## 최소 실행 전 확인 3개

1. **Bank/metric identity tiny check:** physical padding 제외 missing-query membership, Q 고정된 none/shuffle, source0/1, 빈 Q/null 처리, primary graph 선평균 및 conditional utility 분모를 저장-array toy oracle로 확인한다. Projection과 normalization/PCFI hidden-poison·row shuffle identity도 이 단계에 포함한다.
2. **Trace/control accounting check:** common feature RNG에서000 public parity,8 leaf probabilities sum1, direct exact expected objective/gradient, group_pg reward sign·sampling duplicates·detached read와 live adapter CE를 확인한다. Train-global selection은 final truth 없이 deterministic하게 재구성 가능해야 한다. 이 검사는 새 성능 실험이 아니다.
3. **One resource/optimizer smoke:** 실제 train shape의 update와 RP16 real shape의 greedy/8-trace passes를 한 worker에서 확인하고 비용 projection을 남긴 뒤, 고정256/8/96 bank와12 fits를 실행한다. Final 성능을 보기 전에 source/config/checkpoint hashes를 lock하고 저장 확률을 독립 재채점한다.

위 수정은 실험을 끝없이 진단으로 미루자는 요구가 아니다. 통계·입력·비용의 세 blocking 정의를 먼저 잠그고 기존 계획의 실제 experiment를 수행하자는 권고다. 실패하면 이번 pretrained computation-policy/128-update 예산의 후보를 닫고, 성공해도 pure RL backbone training·새 imputer·heterogeneous schema 일반화·새 RL 알고리즘으로 확대하지 않는다.

## PLAN_v2/protocol 대조 — 작성 종료 직전 확인

Root가 V2/protocol을 저장했다고 알려 두 파일을 추가로 읽었다. Current-cycle 다른 reviewer 문서는 여전히 읽지 않았다. 다음 변경은 수용할 수 있다.

- Uniform node95/995 primary, shape×k×regime strata, 실제 own-missing query subset과 conditional utility 분모가 분명해졌다. V2는 내 위의 더 엄격한 two-valid-rates 권고 대신, **각 base에 적어도1 eligible rate가 있으면 valid rates를 평균하고,96개 중 eligible base가 하나라도 없으면 primary undefined**로 고정했다. 최종 결과 전에 정한 명시적 available-rate estimand이므로 타당하다. One-rate/two-rate graph 수를 결과에 기록하고 이를 매 graph 두 rates가 있는 것처럼 설명하지 않으면 된다.
- Full-natural만 사용하는 **97.5% CI lower>−2%p**는 위95% 권고보다 보수적이나 사전 기준으로 타당하다. 50:50을 regimes로 정정하고 actual query CE exposures/full-natural counts를 보고하기로 한 것도 맞다.16개 full-natural update 예약은 내 권고일 뿐 V2의 채택 조건으로 추가하지 않는다. 실제 count가 없으면 없다는 사실을 그대로 보고하면 된다.
- Real RP→observed RMS→PCFI→clip, projected-NaN/PCFI reference 명칭, 실제 context/query identity와 historical representation 한계가 명확해졌다. R의±1/sqrt16은 내 예시±1/sqrt(rawF)와 다른 고정 convention이나, 그 뒤 관측 RMS가 있어 임의의 전역 scale은 소거된다. Literal pipeline·seed·hash는 V2를 따른다.
- Global trace는 **dev8 graphs×7 standard settings**에서 mean CE로 선택한다. 이는 위 train16×2/CE+cost 권고와 다르지만, 별도 heldout synthetic dev calibration을 이용한 합법적인 model selection이다. “train-only” 또는 label-free 선택이라고 부르지 않는다.8 traces를 모두 평가하면448 forwards/seed, uniform3 seeds에1344 forwards다. Final/real에는 접근하지 않으며 fixed lexical tie 규칙과 선택 receipt를 남기면 된다. Same-checkpoint000와 DiD는 사전 secondary로 명시됐다. 자기 best-global 추가 replay는 adaptivity 해석을 더 강하게 하는 권고이며, 새 primary gate를 요구한 것은 아니다.
- Common feature RNG, independent policy RNG, fixed duplicate actual-work accounting 및 group_pg/entropy0 정정은 핵심 혼동을 닫는다. Uniform8 deployment의 추가 비용과 dev calibration/secondary000 replay 비용도 cost budget에 함께 남겨야 한다.

**남은 한 문장 수정:** V2의 “All trajectory losses divided by8”이 direct의 `sum p_a L_a`에도 적용되는지 모호하다. Direct는 `sum p_a=1`인 weighted expectation 자체가 이미 평균이므로 추가1/8을 하면 원래 선언한 CE/policy gradient와 effective learning rate가 달라진다. Sampled uniform/group mean과 fixed duplicate mean만1/8, direct는Σp-weighted sum이라는 뜻으로 명확히 해야 한다. 이는 수학 reviewer/구현 checks와 닫을 작은 specification correction이며, 새 모델·새 diagnostic loop나 성능 기반 tuning을 요구하지 않는다. 이를 닫으면 현재 실험·통계 관점에서 새 broad blocker 없이 구현/고정 실행으로 진행할 수 있다.

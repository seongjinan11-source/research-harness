# 독립 실험설계 종합 및 FP02 Stage 0/1 검토

2026-11-04. Experiments/evidence reviewer. 이 검토는 FP01 최종 판정, transfer diagnostics의 원래 범위, 누적 findings, `factor_nuisance_cycle/PLAN_v1.md`를 대상으로 한다. 현재 cycle의 다른 두 리뷰 내용은 읽지 않았다. Source·bank·checkpoint·prediction·branch를 변경하거나 실험을 실행하지 않았다.

**추천 후보는 하나다: context를 clamp하고 관측 degree exposure·class population·class-pair affinity를 갱신하는 DC block BP operator.** 기존 학습 Factor와 강한 고전 추론기를 실제로 대조한다. 새 head/loss/gate를 추가하는 다음 실험은 추천하지 않는다. 이 후보는 기존 DC-SBM/BP/경험적 Bayes 계산을 현재 실패에 연결하는 수정이며, 현재 단계에서 새로운 PFN 원리나 neural 필요성의 증명이 아니다.

V1 판정: Stage 0는 실행 전 명세 보완 후 진행할 가치가 있다. 실험 방향 4/5, V1 Stage 0 명세 완결성 3/5, 원인에 대한 현재 확신 3/5. **아래 V2 재검토가 현재 판정이며, 뒤의 V1 제안 수치가 V2 protocol을 대체하지 않는다.**

## 0. PLAN_v2 재검토 및 현재 실행 판정

Root의 `factor_nuisance_cycle/PLAN_v2.md`를 추가로 읽었다. **Stage 0 실험설계 이슈는 실행 가능한 수준으로 해소됐다. 독립 수치 검사 PASS 후 계획된 10개 개발 episode를 실행하는 데 실험설계상 blocker는 없다.** V2 명세 완결성은 4/5다. 이 검토는 아직 작성 중인 구현의 정확성이나 성능을 통과시킨 것이 아니다. Stage 1의 학습은 별도 lock과 검토가 필요하다.

- 관측 degree를 곧바로 고정 theta로 넣는 V1에서, 고정 density offset와 Gamma theta latent를 갖는 joint nuisance update로 바뀌었다. 아래 V1 configuration-offset 설명은 이전 제안 기록이다. 현재 추천 후보의 실제 정의는 V2의 theta/B/population 공동 갱신이다.
- Gamma의 prior+fresh statistics, unordered H/count/exposure, theta update, Dirichlet concentration와 `Elogpi` unary/보고용 mean의 차이, parameter damping=.5가 명시됐다. 아래 posterior-mean unary 제안은 채택되지 않았고, V2의 Elog unary가 현 고정값이다. 문서대로 일관되게 구현하면 실험설계상 허용된다.
- 4×6=24 BP와 nuisance update 3회, 마지막 output 시점이 고정됐다. 기존에 지적한 미사용 마지막 갱신 및 evidence 반복 누적의 혼동을 해소했다.
- `poisson_fixed`/`sbm_update`/`dc_update`를 사전 고정해 link/volume repair와 degree 추가 효과를 구분할 수 있다. 단, sbm_update 대비 dc_update 차이도 이 전체 알고리즘 아래의 기여이며 real 실패의 유일 원인 증명은 아니다.
- 10 episode를 유지하며 새 shuffle 없음과 alignment 주장 불가를 명시했다. V2의 관측−none은 joint inference의 전체 효과다. 같은 C_global/centroid grid와 동일 homophilic conditional old reference를 쓰므로 feature module 또는 role 선택 변화가 graph 비교를 대신하지 않는다.
- Physics +20%p/CS +5%p, CiteSeer accuracy −1%p/NLL +.05 guard와 5-graph BAcc/NLL gate가 사전에 고정됐다. 아래 V1의 +5%p 및 개별 guard 제안과 다른 수치를 채택했으며, 결과 전에 고정했으므로 허용한다. Mean은 **5개 source graph에 동일 가중**으로 계산하고 n_query 가중 pooled score로 바꾸지 않는다. 개별 graph의 minority/NLL 악화도 평균과 함께 보존한다.

작은 보고 명확화 하나만 권한다. Stage 0의 주 선택은 사전 `dc_update`로 읽으며, gate 미통과 뒤 sbm_update의 결과를 DC 통과로 대체하지 않는다. SbM이 충분하면 더 단순한 후속 후보를 Stage 1 명세 전에 채택할 수 있지만 그것은 개발 단계 선택이라고 기록한다. Source/query가 기존 bank와 같더라도 observed source가 query에 포함되면 95% 점수의 own-feature-missing subset도 기술 통계로 병기한다. 이 둘은 계산 후 유리한 metric/operator를 선택할 수 있는 예외가 아니다.

Stage 0에 추가 대형 분석이나 재학습을 요구하지 않는다. Root의 수치 검사→고정 10 episode 실제 실행→구현 오류와 정상 성능 실패 분리→별도 Stage 1의 6 fits 순서에 동의한다.

## 1. 즉시 적용할 Stage 0 수정: 다섯 가지

이 절은 PLAN_v1에 대한 원래 검토다. 현재 채택/수정 여부는 위 V2 재검토를 따른다.

### 1) Population·Gamma 갱신과 feature control의 의미를 수치로 고정

`Dirichlet alpha=1`만으로 posterior mean과 MAP가 정해지지 않는다. 권장 population 갱신은 unlabeled 집합 U에 대해

    pi_a = (1 + sum_{i in U} b_i(a)) / (K + |U|).

이는 label-balanced context를 모집단 빈도로 세지 않는 posterior-mean 형태의 근사 갱신이다. MAP의 alpha−1 식을 구현한다면 다른 연산이므로 별도로 명시한다. Degree-weighted volume은 pi의 node count를 대체하지 않으며, Gamma exposure에는 context를 포함한 모든 node가 들어가야 한다. `shape=kappa+edge soft-count`, `rate=kappa/prior_mean+pair exposure`의 unordered pair convention, 마지막 6-step block 뒤 갱신한 parameter를 예측에 반영하는지, update 전후 output 시점도 고정한다.

None/observed 각각에서 population과 affinity를 다시 추정하면 두 예측 차이는 **feature가 graph nuisance 추정을 바꾸는 전체 효과**다. 이것은 허용되는 joint predictor지만 graph parameter를 고정한 feature likelihood 효과와 같지 않다. Stage 0에서는 이 범위를 정확히 써도 충분하다. 이후 추가 feature 효용을 주장할 때는 graph-only로 얻은 pi/W를 feature mode들에 공통 고정한 보조 대조도 남기는 것이 가장 작은 분리 검증이다. 이 대조의 결과를 보고 주 predictor를 선택하지 않는다.

### 2) 10개 episode에는 shuffle이 없으므로 해당 주장을 제한

현재 Stage 0는 5 graph×none/95=10 episode다. 같은 문서의 “none/observed/shuffle을 모두 보고”와 불일치한다. 실행 전 둘 중 하나를 고정하면 된다.

- 권장: 동일 95% source/mask에서 feature row를 고정 seed로 섞은 5 episode를 추가해 총 15개로 한다. Query/context/source/normalization과 source 개수는 그대로 둔다. 4 operator라면 60 predictions이며 새 training은 0회다.
- 10 episode를 유지할 경우: Stage 0는 graph 회복 및 observed−none 기술 통계까지이며, feature alignment를 검증했다고 쓰지 않는다. Shuffle은 Stage 1의 사전 명세에 넣는다.

어느 쪽이든 query 자신의 feature는 hidden인 공통 query를 써야 한다. 기존 episode 전체 query에 observed source가 섞여 있다면 원래 점수도 보존하고 own-feature-missing subset을 별도 재채점한다. Query truth를 사용해 source나 operator를 고르지 않는다. Source0/1 수치 대칭성은 tiny check이며, 그것만으로 실제 95% episode의 피처 정보가 없다고 해석하지 않는다.

### 3) “회복하고 보존”을 첫 예측 전에 정량화

Stage 0는 rep0/seed11의 이미 본 graph 5개를 쓰므로 population CI나 최종 우월성 검정을 붙이지 않는다. 그래도 Stage 1로 넘어갈 개발 gate는 필요하다. 다음은 **결과를 보기 전에 채택할 수 있는 보수적 실행 가능성 gate 제안**이다.

- None 조건에서 고정된 DC-update 하나가 old-full 대비 Physics와 CS accuracy를 각각 최소 +5%p 회복한다.
- CiteSeer accuracy가 old-full보다 1%p 넘게 하락하지 않는다.
- 다섯 graph 모두에서 old-full 대비 balanced accuracy가 1%p 넘게 하락하거나 NLL이 +0.05 nats 넘게 악화하면 “전체 회복”으로 통과시키지 않는다. Accuracy·BAcc·macro-F1·NLL·class별 recall·predicted class mass를 함께 남긴다.
- 모든 확률/메시지가 유한하고 clamp·대칭성·누수 검사 PASS여야 한다. 이 구현 gate가 실패한 경우와 정상 계산의 성능 gate 실패를 구분한다.

Root는 다른 수치를 명시적으로 채택할 수 있으나 첫 결과 전에 이유와 수치를 기록해야 한다. 이 gate는 새 operator가 실용적으로 작동하는지 보는 개발 결정이고, fixed+update 중 graph별 승자를 고르는 규칙이 아니다. Old-edge와 두 LP도 동일 표에 남긴다. DC가 old-full만 이기고 더 강한 old-edge/LP보다 못하면 그 한계를 보존한다. 미통과 후 eta/kappa/iteration을 쓸어 같은 5 graph에서 통과시키는 흐름은 피한다. 다음 변경은 새 가설·cycle이 된다.

### 4) Fixed와 update의 차이를 단일 원인 규명으로 쓰지 않기

DC-fixed-affinity/uniform-population과 DC-update는 affinity와 population을 함께 바꾼다. 이 두 arm으로 “degree가 원인”, “class prior가 원인” 또는 “positive coupling만 원인”을 따로 확인할 수 없다. Stage 0에서 굳이 ablation을 늘릴 필요는 없다. **합쳐진 operator가 하나의 고정 규칙으로 개선하는지**가 이번 질문이다. Tiny correctness와 parameter trajectory를 저장하고, 개선 뒤 원인이 필요하면 별도 개발 명세에서 하나씩 고정한다. Full/edge와 DC를 graph별로 고르는 oracle는 최종 방법이 아니다.

고전 DC baseline을 약화시키지 않는다. 고정 prior만 쓴 DC-fixed를 강한 통계 추론기의 대리로 부르면 안 되고, update arm이 실질 baseline이다. Old-full/old-edge에는 동일 seed11 C_global 계수, 동일 feature scaling·source/query·context를 써야 한다. 새 DC에는 true degree/class/affinity oracle를 입력하지 않는다. 관측 d_i는 공통 graph에서 얻지만 true theta/pi/W는 generator-side audit 전용이다.

### 5) Stage 1의 학습 비교와 “같은 task”를 나중 lock에서 명확히

6 fits는 적절한 최소 범위다. 다만 “같은 Factor”의 학습 parameter·init·graph operator·feature 경로를 별도 명세에 적어야 실제 학습 대조가 된다. Original/shifted prior 사이에는 labels와 degree가 달라져 adjacency도 달라진다. 같은 graph를 썼다고 할 수는 없다. **공통 난수·N/K·signal·expected degree·feature noise·mask/context 선택 규칙·노출 순서**를 가능한 한 결합하고, 바뀌는 생성 메커니즘만 적는다. 각 prior 안에서는 비교 모델에 동일 저장 task를 제공하고 모든 모델은 공통 locked final bank에서 평가한다.

또한 기존 historical Factor를 새로 재학습한 6 fits와 섞지 않는다. 같은 구조/init/optimizer의 F_balanced와 F_degree-skew 각 3 seed×256 updates×2 tasks=6 fits, 총 1,536 새 updates/3,072 task presentations로 고정한다. Inherited training과 Stage 0 inference 비용은 별도다. 이 비교가 끝나기 전에는 Stage 0의 classical 회복을 “synthetic-pretrained PFN 개선”이라고 쓰지 않는다.

## 2. 누적 실패가 이번 계산 변경을 요구하는 이유

| 관측 | 직접 확인된 범위 | 아직 확인하지 못한 원인/일반화 |
|---|---|---|
| FP01 C−C_global +1.187%p, 97.5% CI [+0.213,+2.446] | Adaptive centroid encoder의 해당 합성 비교 이득 | C−B +0.952%p CI [−0.585,+2.696]로 전체 gate 미통과. C−none +0.579%p의 95% CI도 0 포함. 이를 새 feature 효용 확립으로 바꿀 수 없음 |
| 실제 C가 B보다 개선되지만 none에 매우 가까움 | 여러 graph에서 feature harm 완화 | Graph-only 근처로 수축한 효과는 추가 feature 정보를 회수한 증거가 아님 |
| Physics none nonedge 삭제 46.30→86.14%, CS 61.74→72.67% | Frozen score의 nonedge 포함 계산이 이 개발 사례들에 큰 손해를 줌 | Closed-world 가정, degree/skew, 약한 positive edge 중 고유 원인 미분리. CS BAcc는 58.77→58.64로 개선되지 않음 |
| CiteSeer는 edge-only로 크게 하락 | Nonedge를 항상 삭제하는 규칙은 부적절 | Dataset 이름으로 full/edge를 선택할 근거가 아님 |
| Physics empirical edge log-ratio 3.077 vs learned mean 1.412, nonedge −0.00965 vs −0.01039 | Edge 대비 nonedge 상대 강도 mismatch를 지지하는 사후 진단 | Query truth로 만든 pooled 값은 배포 estimator가 아님. “Nonedge 확률만 과대”라는 설명은 불충분 |
| Physics K=5, CS K=15; PubMed degree CV가 더 큼 | K 외삽 또는 degree 하나로 두 실패를 모두 설명할 수 없음 | Degree correction 하나면 실제 전이가 해결된다고 예측할 근거 없음 |
| Physics 48→96 iteration argmax 동일 | 해당 frozen 사례는 반복만 늘려 복구되지 않음 | 모든 graph의 수렴·global optimum 증명은 아님 |

원자료: [FP01 판정](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/DECISION_KO.md), [전이 진단](C:/Users/seongjin/Desktop/code/gts_pfn/factor_transfer_diagnostics_20261104/REPORT_KO.md), [FP01 독립 결과 검토](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/RESULT_EXPERIMENTS.md).

누적 knowledge의 183 findings(실패 53, 보존 효과 43, lesson 64, untested proposal 23)는 모두 실패 183개라는 뜻이 아니다. 특히 다음 구분을 이번 설계에 반영한다.

- Positive-only diffusion의 한계와 signed edge/nonedge decoder의 기존 해결을 새 발견으로 재명명하지 않는다. 이번에는 그 **학습된 상대 강도와 class volume**이 실제 전이에 맞는지가 질문이다.
- Context에서 떨어진 feature source를 연결해 유효 정보를 전파할 수 있었던 binary 효과는 보존한다. Categorical/global uncertainty·실제 graph에서도 자동으로 성립한다는 결론은 보존된 효과의 범위 밖이다.
- Likelihood/posterior auxiliary loss, memory 경로, feature-law 변경, gate/router/LP 혼합 등의 미통과가 반복됐다. 이번 변경은 출력 조절 대신 입력 graph의 degree exposure·class mass·affinity를 추정하는 계산에 둔다. 모든 augmentation이나 모든 neural 방법이 불가능하다는 결론은 내리지 않는다.
- Grid/explicit Gaussian 등 강한 고전 추론기가 학습 모델을 따라잡거나 앞선 사례가 있다. 예를 들어 알려진 family의 Gaussian/probit 대조는 label-only 대비 +2.467%p CI [+0.884,+4.167] 효과를 보였다. 이는 기존 evidence를 회수하는 강한 baseline의 필요성을 뜻하며, 다른 real graph의 성능 보장은 아니다.

참조 findings: [LLBP/EAJ](C:/Users/seongjin/Desktop/code/gts_pfn/research-harness/findings/llbp-eaj.yaml), [adaptive transport](C:/Users/seongjin/Desktop/code/gts_pfn/research-harness/findings/adaptive-transport.yaml), [signed transport](C:/Users/seongjin/Desktop/code/gts_pfn/research-harness/findings/sign-transport.yaml), [factor projection](C:/Users/seongjin/Desktop/code/gts_pfn/research-harness/findings/factor-projection.yaml). Retrieval tag나 paper match는 해결의 증거가 아니다.

## 3. 추천 operator와 강한 baseline의 범위

PLAN_v1의 관측 configuration exposure `e_ij=d_i*d_j/(2m)`, Gamma affinity, unlabeled node-count population, edge cavity BP+nonneighbor negative field가 한 후보의 구체적인 실행안이다. `E log W`와 `log E W`를 구분하고 negative field가 self/neighbor를 이중 포함하지 않게 하는 현재 수치 검사 계획을 지지한다. Full Poisson factor의 tiny exact oracle, 동일 factor의 dense BP, sparse field 근사 오차를 서로 구분해야 한다. Loopy BP를 exact enumeration과 무조건 같게 요구하지 않는다.

이것은 binary simple graph에 대한 Poisson working score와 degree plug-in 근사다. Exact degree-conditioned likelihood, exact joint Bayes 또는 published AMP의 faithful reproduction으로 부르지 않는다. Gamma posterior에서 true label pair count를 쓰지 않는다. `m=0`과 isolated node는 정의된 fallback을 쓰고, context 없는 component는 label 식별 불가능성을 결과에 남긴다. 초기 eta=.6, kappa=5, damping=.5, BP24/4 outer를 고정해 실제 실행한 뒤 정상 성능 실패도 보고한다.

Degree-corrected block model과 BP/EM 기반 추론은 기존 문헌이다. Degree heterogeneity가 ordinary SBM을 왜곡할 수 있다는 근거와 이번 데이터의 원인 확정은 다르다. [Karrer & Newman, 2011](https://arxiv.org/abs/1008.3926), [Yan et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4498413/)을 근거로 고전 baseline을 구성하되 현재 hybrid를 원 논문의 정확 재현이라고 하지 않는다.

Stage 1의 normalized propagation은 추가 neural 후보가 아니라 강한 비교군이다. 최소 설정 예시는 `S=D_tau^(-1/2) A D_tau^(-1/2)`, `tau=mean_degree`, alpha=.9, 40 iterations, 매회 context hard clamp다. Isolated/unseeded component fallback과 최종 확률 정규화를 고정한다. 기존 harmonic/restart LP의 동일 episode 결과도 보존한다. Classical estimator만으로 회복되면 그것을 결과로 인정하며 neural superiority를 만들려고 고전 update를 끄지 않는다.

## 4. Stage 1에서 잠글 가장 작은 검증 설계 제안

다음은 아직 승인된 실행 명세가 아니라, Stage 0 뒤 별도 lock에서 구체화할 제안이다. Stage 0를 지연시키는 선행 조건은 아니다.

**개발/학습.** 새 512개 base task를 F_balanced/F_degree-skew 두 prior에 공통 난수로 결합한다. 같은 N/K·degree scale·homophily signal·feature law·context/source protocol을 쓰고 class imbalance와 degree propensity 생성만 바꾼다. Original support를 그대로 쓸지 N=128/256 등으로 양쪽 모두 넓힐지는 미리 명시한다. 넓힌 경우 F_balanced를 역사적 원래 prior의 완전 재현이라고 하지 않는다. 강한 skew에서 모든 class의 context+query가 가능하도록 count 생성 규칙을 사전에 정하고 rejection 횟수/이유를 남긴다. 평가 query에서 class를 삭제하거나 결과가 나쁜 graph를 버리지 않는다. 새 개발 bank 16개 정도로 correctness만 확인하고 final을 생성한다.

**새 final.** 실행 가능한 예시는 N={128,256}, K={3,5}, degree heterogeneity={낮음,높음}, population skew={균등,dominant .75}의 16 cell×4 graph=64 base graphs다. Dominant class index는 permutation한다. Degree propensity는 평균1의 고정된 lognormal σ=0/.9, 사전 clipping 범위와 renormalization을 기록한다. Binary adjacency 생성 방식도 고정한다. 예를 들어 `p=1-exp(-lambda_ij)`를 사용하면 candidate의 Poisson working score와 같은 likelihood라고 주장하지 않는다. 모든 arm은 같은 binary adjacency를 받는다. True pi/theta/W는 평가 sidecar에만 둔다. 실제 class count/CV/edge density를 기록하며 이 축이 real graph의 고유 인과를 증명한다고 하지 않는다.

**Primary.** 이번 cycle의 첫 질문은 graph-only 구조 회복이다. Shifted 세 cell(heterogeneity만, skew만, 둘 다)의 48 graph를 주평가, balanced/low-heterogeneity 16 graph를 safety panel로 사전 지정하는 것을 권한다. Class skew를 다루므로 balanced accuracy를 primary로, accuracy/NLL/macro-F1을 반드시 같이 둔다. Candidate DC-update와 (a) 새 F_degree-skew, (b) 고정 normalized LP의 두 비교를 co-primary로 하되 각각 97.5% paired graph-bootstrap CI, 평균 +1%p 및 하한>0, accuracy guard −1%p/NLL guard +.01을 사전에 고정할 수 있다. Graph strata는 N×K×세 shifted cell=12개, 각각4 graphs다. 같은 graph의 seed·rate를 먼저 평균한 뒤 base graph를 resample하며 node/seed를 독립 표본으로 세지 않는다. F_degree-skew−F_balanced는 prior 변경의 사전 secondary 비교다. Classical candidate가 학습 Factor를 이기더라도 그것이 learned PFN 성공은 아니다.

**Feature 평가.** Primary 구조 판단 후 동일 frozen C_global를 graph arm들에 붙이고 none/95/shuffle95와 공통 own-feature-missing query로 평가한다. 99/99.5는 source 개수별 secondary panel로 남긴다. `observed−none>0`뿐 아니라 `observed−shuffle>0`의 방향과 uncertainty를 보고한다. 다른 comparator보다 feature harm이 작지만 none보다 유용하지 않으면 “harm reduction”이다. Source0/1의 class-exchangeability 제한은 학습량으로 해결 가능한 오류와 구분한다. RP/raw/orthogonal은 이번 graph primary를 바꾸는 선택지가 아니라 고정 interface의 범위 설명이며, FP01 negative panel도 지우지 않는다.

**공정성과 비용.** 새 fits는 6개, updates1,536/presentations3,072, target-gradient updates0이다. 512 task bank를 두 번 보이거나 repeated epochs를 쓰면 독립 graph 수와 exposure를 구분한다. F0/Fshift의 초기 parameter·optimizer·학습 schedule·입력/CE query를 동일하게 고정한다. 공통 graph-only role weight 또는 homophilic conditional arm 중 하나를 protocol에서 고정해 role-selection 차이와 operator 차이를 섞지 않는다. 모든 final method는 64개 공통 graph, 동일 query와 context, 동일 preprocessing을 받는다. 먼저 단일 worker로 실행하고 GPU peak·cold preprocessing·graph encoding·prediction·analysis 시간과 실패/재계산 비용을 분리해 기록한다. Classical inference의 parameter estimation 비용도 prediction 시간에 포함한다. 6 fits가 지정 업데이트를 모두 마치지 못하면 wall-budget 종료를 성능 실패와 구분하고 불완전성을 기록한다. 실용적 pilot ceiling은 단일 GPU 60분으로 먼저 정할 수 있으나 source/graph 크기 smoke 측정 뒤, 결과를 보기 전에 최종 lock해야 한다.

## 5. 실제 graph 전이의 다음 범위

현재 Cora/CiteSeer/PubMed/CS/Physics는 계속 development다. Stage 0의 rep0/seed11에서 본 뒤 rep1/2를 추가해도 새 graph holdout은 아니다. 그래프별 full/edge/DC의 최댓값을 보고하는 것은 oracle diagnostic이며 배포 predictor가 아니다. 기존 외부 PCFI/FISF/PFN와 LP array는 input/context/query/hash identity를 확인한 episode에만 재사용한다. 특히 CS/Physics accuracy 우세와 minority recall/BAcc 열세를 함께 보고한다.

새 real 검증은 **Amazon Computers/Photo**를 다음 lock 전에 후보로 사전 지정할 수 있다. 공식 PyG 문서상 각각 10/8 class의 co-purchase graph다. 현재 local knowledge에서 이름이 보이지 않는다는 이유만으로 untouched라고 확정하지 말고 과거 사용/캐시/전처리 provenance를 먼저 감사한다. [PyG 공식 데이터 설명](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Amazon.html).

Historical dense Factor의 N² 비용 때문에 전체 graph를 못 돌리면 모든 방법에 동일한 label-blind fixed 2,000-node induced graph를 사용하고 induced 범위만 주장한다. Seed·추출법·context 수·own-missing query·rate·single predictor를 label 결과 전에 고정한다. 다른 source graph 두 개는 단순 synthetic seed 재사용보다 강한 전이 확인이지만, 2 graphs만으로 일반적인 real-population 우월성 CI를 붙이지 않는다. 데이터 미확보를 새 실험 결과처럼 쓰거나 이 graph 결과를 보고 candidate를 바꾼 뒤 같은 holdout을 재사용하지 않는다.

실행 순서는 **수치 검사 → 고정 Stage 0 실제 실행 → 구현 실패/정상 성능 분리 → Stage 1 별도 lock·6 fits → 새 final 재채점 및 독립 리뷰**가 적절하다. 현재 추천은 구체적 operator를 실행하는 것이며, 다시 진단만 늘리거나 FP01 final의 loss/head를 바꿔 구제하는 것이 아니다.

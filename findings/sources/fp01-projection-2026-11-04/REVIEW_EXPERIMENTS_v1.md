# FP01 independent experiment-design review v1

Reviewer: experiments / baselines / statistics. 2026-11-04. 대상은 `PLAN_v1.md`; 다른 current-cycle review 파일은 읽지 않았다. Root `AGENTS.md`, 기존 edge-only 및 feature-law 보고서, 기존 categorical protocol/data, homophily bank manifest를 대조했다. 이번 작업은 설계 검토뿐이며 학습·GPU·새 예측 실행·모델/source 수정은 없다.

## Verdict

**조건부 진행 권고.** FP01은 이전 두 실패를 반복하지 않을 가능성이 있는 구체적인 가설이다. Preferred-class column family가 projection에 닫혀 있지 않다는 문제와, 고전 Gaussian centroid inference를 강한 비교군으로 두려는 방향은 타당하다. 다만 16→16 Rademacher surrogate, 적은 feature-task 학습, 이전에 공동 학습한 graph checkpoint 고정이라는 조건에서 검증하는 **feature-module repair**다. 실제 native feature 분포를 정확히 맞췄다거나 새 Gaussian/AMP 원리를 발명했다고 해석할 수 없다. 최종 lock 전 아래 항목을 명세하면 작고 반증 가능한 실험이 된다.

Root가 검토 중 전달한 후속 결정—새 arm 공통의 frozen graph-only role weights, same-budget C-global, 제한된 RP surrogate와 orthogonal eval-only diagnostic—은 다음 권고에 반영했다. 다른 reviewer의 결론을 읽거나 동의 여부를 맞춘 것은 아니다.

## 보존해야 할 긍정적 근거

- 기존 edge-only 결과에서 Coauthors의 큰 accuracy 복구는 실제 결측 조건에도 남았다. 반면 CiteSeer 손상과 대부분 음수인 full−no-observed 효과도 남았다. 두 결과 모두 FP01의 출발점이며, Coauthors 복구를 feature 성공으로 세면 안 된다.
- Feature-law pilot는 실제 학습 PCFI/Mask 대비 약 11%p 효과를 보존하지만 adaptive−static/original 성공은 확인하지 못했다. FP01은 더 유연한 함수를 하나 추가했다는 이유로 성공을 선언하지 않고, C-global 대비 증분을 요구한다는 점에서 개선된다.
- 동일 task schedule, feature head reset, graph freeze, own-feature-missing query, source/context disjointness, final bank lock, 모든 full/edge 결과 보존은 좋은 통제다.

## 최종 lock 전 필요한 최소 수정

### 1. A/B가 검증하는 가설과 projection 진단을 좁힌다

Raw row를 x=m_c+epsilon, epsilon~N(0,0.36 I)로 쓰면 projected noise covariance는 **0.36 R^T R**이다. Rademacher 16×16/sqrt(16)은 직교행렬이 아니다. 따라서 mean shape 외에 상관, singular values/conditioning, 경우에 따라 rank까지 바꾼다. 관측-only RMS와 clipping도 그 뒤에 추가된다. 계획에 있는 “B−A는 전체 interface 변화”라는 caveat는 정확하다.

**필수:** A/B는 같은 latent raw graph/labels/noise/context/mask 및 feature-head 초기 parameter tensor를 공유하고, training view만 다르게 한다. Projection RNG를 별도 stream으로 분리해 R을 추가하면서 A/B의 뒤쪽 graph/noise/mask 샘플이 달라지지 않게 한다. Train base-task/초기화/순서 hash를 비교한다. Final에서 A/B×raw/projected를 교차 평가한다. B(projected)−A(raw)라는 대각선 차이만 비교하면 train-interface 효과와 eval-view 효과가 혼합된다.

**작은 eval-only 진단:** fixed-seed QR로 만든 직교 Q를 사용한다. 같은 raw x에 Q를 곱하면 preprocessing 전 Gaussian noise의 isotropy와 class-mean 거리를 보존하면서 column mean의 preferred-class 모양을 바꿀 수 있다. RMS/clipping 뒤에는 이 보존이 일반적으로 유지되지 않으므로 pre/post를 구분한다. R/Q별 rank, singular-value 범위, class-mean 거리, noise covariance off-diagonal 크기, clipping fraction을 generator-side audit로 저장한다. 참 centroid/noise는 이 감사와 scoring에만 쓰고 predictor에는 전달하지 않는다. Q 결과를 보고 다음 training view로 선택하지 않는다.

**범위:** existing bank에서 CoauthorCS는 native D=6805, Physics D=8415, PubMed D=500이다. 원 실제 RP는 D→16인 반면 FP01은 16→16이다. “RP operation을 포함한 작은 surrogate”라고 부르고 정확한 real-distribution matching이라고 부르지 않는다. C의 independent Gaussian class centroids도 원 preferred-class means가 보존하는 sum_c mu_cf=0 및 projected cross-column 의존성을 정확히 모델링한 것은 아니다. 더 넓은 conditional score family라는 가설로 평가하면 된다.

### 2. 공통 frozen graph-only role weights와 inherited/reset 비용을 명확히 한다

각 checkpoint seed, graph/context, full/edge panel에 대해 **feature-none graph만으로 role weights를 한 번 계산**하고 모든 A/B/C/C-global에 공유한다. Full과 edge는 자기 graph-factor 정의에 맞춘 graph-only weights를 별도로 계산한다. 그 뒤 observed values, encoder 출력, final labels로 role weights를 바꾸지 않는다. 이는 feature operator 비교를 분리하는 정직한 protocol이며, FP01에서는 feature evidence가 role posterior를 개선했다는 주장을 하지 않는다. Historical Factor의 원 feature-dependent role update는 별도 reference로만 둔다.

A는 이전 trained Factor 자체가 아니다. Original co-trained graph checkpoint를 고정하고 feature head를 원 초기값으로 reset한 뒤 재학습하는 arm이다. A/B head weight hash가 같아야 하고 graph-module hash는 학습 전후 같아야 한다. C도 matched moment initialization으로 출발시키되 representation/parameter 차이를 기록한다. C가 실패해도 원 graph encoder와의 co-adaptation 또는 독립 최적화 한계를 제거한 실험은 아니므로 centroid family 자체의 불가능성으로 해석하지 않는다.

Budget는 A/B/C/C-global 각각 3 seeds×256 updates×2 tasks, 즉 **새 12 model fits / 3072 successful updates / 6144 task presentations**로 고정하는 것이 명확하다. C-global은 inference가 싸지 않을 수 있으므로 비용 장부에서 제외하지 않는다. 별도 C-fixed(no training)는 0 updates reference이며 inference 비용은 포함한다. 원 3 graph checkpoint의 합계 768 updates/1536 task presentations는 inherited shared cost로 별기하고, graph checkpoint를 12번 새로 학습한 것처럼 더하지 않는다. 새 512 tasks가 원 checkpoint tasks의 replay인지 fresh인지를 명시하고 unique task 수와 presentations를 구분한다. Smoke/실패 시도/수치 preflight도 별도 장부에 남긴다.

### 3. C-fixed를 약한 임의 설정으로 만들지 않는다: 강한 Gaussian 비교군

**주 비교군은 C-global로 둔다.** C와 정확히 같은 centroid graph, class-exchangeable prior, quadrature, messages, iterations, damping, shared graph checkpoint/role weights를 사용하고, observed-input-dependent encoder 대신 shared location/noise/prior-scale의 소수 global coefficients만 synthetic training CE로 같은 256×2 budget 동안 학습한다. Global coefficients는 target graph에서 고정한다. C−C-global은 input-adaptive feature parameterization의 증분이다. “학습 vs 학습 없음”으로 부르지 않는다.

No-training C-fixed를 추가할 경우 다음과 같은 **구체적 moment-matched initialization**을 baseline specification으로 쓸 수 있다. Original raw family의 class-balanced second moment를 v_K=0.36+0.64/(K−1)로 두고, RMS가 약 1인 input 단위에서

    m_f = 0
    tau_0^2 = [0.64/(K−1)] / v_K
    sigma_0^2 = 0.36 / v_K
    mu_cf ~ Normal(0,tau_0^2), x_if|s_i=c ~ Normal(mu_cf,sigma_0^2)

를 사용한다. C-global은 tau=alpha_tau*tau_0, sigma=alpha_sigma*sigma_0, 두 positive global multiplier를 1에서 시작해 학습하면 된다. 추가 global location coefficient가 필요하면 C에도 같은 centering convention을 쓰고 수를 보고한다. 이 고정 초기값은 family moment를 이용하는 통계 비교군이며 실제 R, empirical RMS/clipping 이후의 정확한 generative law는 아니다. True realized centroids/labels/noise를 제공하지 않는다. source=0일 때 observed statistics의 fallback과 positive noise floor를 final 전 고정한다. C-fixed가 약하면 C-global을 빼고 그 값만 비교하는 선택은 하지 않는다.

**Gaussian-centroid cavity baseline으로 정직하게 이름 붙인다.** Grid/정규화 weight를 유지한 cavity update에서 centroid cf의 i를 제외한 분포는 개념적으로

    q_cf\i(mu) ∝ prior_f(mu)
       × product_{j in O\i} [1 − a_j(c) + a_j(c) Normal(x_jf;mu,sigma_f^2)]

이며 a_j(c)는 해당 factor를 제외한 node cavity probability다. 이를 grid에서 적분해 gated likelihood message를 보낸다. 이는 C와 같은 sum-product 통계 비교이며, point pseudo-label fitting이나 보통 GMM EM과 구별된다. 11-point grid가 좁은 posterior까지 충분한지 dev에서 21/41-point 또는 1D integration reference와 검사한다. Tiny tree만으로 source100–166 또는 K15의 해상도까지 보장할 수 없다. 더 촘촘한 reference를 final score를 보고 고르지 않는다.

**AMP 관련 선행연구를 정확히 취급한다.** Duranthon–Zdeborová의 AMP–BP는 Gaussian-mixture latent centroids와 graph inference를 결합하며, Appendix D에는 이미 multiple unbalanced communities가 있다. 원 논문은 sparse nonedge 정보를 effective field로 처리한다. 그러므로 “AMP는 binary 전용”이나 “nonedge를 단순히 버린 알고리즘”이라는 설명은 틀리다. [원 논문 §3·Appendix D](https://arxiv.org/html/2306.07948), [저자가 연결한 코드](https://gitlab.epfl.ch/spoc-idephics/csbm).

FP01의 finite quadrature/class-gated cavity, input-conditioned noise, masking, projected covariance, frozen neural graph potential은 그 published AMP–BP와 다르다. 이를 **Gaussian-centroid BP baseline**이라고 보고하고 AMP 재현/AMP의 최적성/AMP를 이겼다는 주장을 하지 않는다. 실제 AMP 재현이 필요하면 별도 companion setting으로 K=2, iid Gaussian v, B_beta,i=sqrt(mu/N)*v_beta*s_i+Z_beta,i, known common prior parameters, authors' graph/Onsager updates를 사용해 작은 논문-matched check를 먼저 둔다. Missing-row/low-F/projected 설정으로 바꾸면 그것도 modified AMP다. 이번 FP01을 시작하기 위해 그 별도 재현을 필수 대형 작업으로 확장할 필요는 없다.

### 4. Query와 sparse-source identifiability를 final endpoint 설계에 반영한다

모든 synthetic training task에 feature-missing query를 최소 하나 예약하고 source/context와 분리해야 한다. CE를 전체 non-context에 적용한다면 observed source도 training target이라는 사실을 명시한다. A/B/C/C-global의 같은 query/label vector를 사용한다. Final primary는 반드시 own-feature-missing query로 고정하고 모든 rates/views에서 같은 query를 사용한다. Label balancing/context selection은 허용된 generator/training labels에만 의존하며 final query scoring labels는 inference에 전달하지 않는다.

Exchangeable centroid prior와 모든 class에 공통인 hyperparameters에서 source row가 하나라면

    integral prior(mu_cf) Normal(x_if;mu_cf,sigma_f^2) dmu_cf

는 선택한 class c와 무관하다. source=0/1 whole-node condition은 graph/context만 남는 것이 정상이다. 이 결론은 feature hyperparameters 및 role weights가 class/role별 추가 정보를 주지 않는 현재 protocol의 조건이다. C가 singleton에서 큰 feature gain을 보이면 성공보다 leakage, symmetry, normalization 또는 cavity 문제를 먼저 의심해야 한다. Counterexample가 가능한 다른 prior까지 포괄하는 불가능성 정리는 아니다.

기존 n=64/96/128에서는 99% source가 주로 1개, 99.5%는 0 또는 1개다. 이 조건이 평균 primary의 절반을 차지하면 feature operator의 필요성이 원리상 없는 조건에 학습/평가 예산을 집중한다. **50/75/90/95%를 primary feature-utility stratum으로 사전 고정하고 99/99.5%도 모두 보고하되 singleton/empty sanity strata로 구분**하는 것이 작고 해석 가능하다. 사후에 유리한 source count만 고르는 방식은 피한다.

새 graph-only-role protocol에서는 feature-none A/B/C/C-global 확률이 같은 frozen graph predictor와 일치해야 한다. 따라서 (U_C−U_B), U_m=ACC_m(full features)−ACC_m(none), 는 같은 query에서 ACC_C−ACC_B와 동일한 대조다. 이를 두 독립적인 성공 증거로 중복 계산하지 않는다. Full−none은 실제 feature 추가 가치의 절대 부호를 확인하는 데 여전히 필수다.

### 5. 작은 final specification과 명확한 성공·중단 규칙을 고정한다

다음은 실행 가능한 구체적 최소안이다. Root가 다른 정확한 숫자를 선택해도 되지만 final 생성 전에 파일에 잠가야 한다.

| 항목 | 권고 명세 |
|---|---|
| Development | 12 base graphs, final과 disjoint seeds; numerical/leakage/quadrature/resource 수리에만 사용 |
| Final base graphs | 48 = N{64,96,128}×K{2,3,5,7}×4; graph·labels·noise·R 독립 seed streams. Role/degree는 전체에서 균형되게 사전 배정. 층화된 finite bank임을 명시 |
| Masks/context | Existing train-context recipe, nested 6 missing rates, common feature-missing Q; all arms/views same |
| Full primary | Projected view, rates 50/75/90/95의 graph 내 동등 평균, 3 fixed seeds 평균 후 base graph를 paired 단위로 사용 |
| Extreme-rate sanity | 99/99.5 모두 공개, source 0/1 parity를 별도로 검사 |
| Cross-view | A/B는 raw/projected 둘 다 평가; C/C-global의 main projected comparison은 동일 input |
| Orthogonal Q | Final의 사전 고정한 12 base graphs, 50/95% 두 rate만; eval-only mechanism diagnostic, primary 대체 금지 |
| Edge panel | 같은 projected bank의 prespecified secondary; full과 별도 표/결론. Per-graph 최선 full/edge selector 금지 |
| Real | 원 75 episodes×all 3 seeds; 5 graphs, 3 rates+none+shuffle95의 5 조건의 기존 정의 유지. Full/induced 구분 및 all full/edge 결과 공개. 모두 development evidence |

**핵심 성공 대조:** full projected primary에서 C−B와 C−C-global의 paired ACC 차이를 두 co-primary comparisons로 고정하고, C의 full−none도 양수인지 확인한다. 실용적 크기 기준을 원한다면 +1%p point estimate 및 CI lower>0처럼 수치를 사전 고정한다. Synthetic paired bootstrap은 base graph 단위로 하고 N/K strata 구조를 보존한다. 두 core comparisons에는 simultaneous 또는 Holm familywise 처리를 고정한다. NLL/Brier/balanced accuracy는 모두 보고하고, NLL 허용 악화 폭을 예컨대 0.01 nats로 사전 고정하면 accuracy만으로 probability quality 손상을 덮는 것을 막을 수 있다. 기준 숫자는 final 후 바꾸지 않는다.

Real per-graph/rate ACC와 full−none은 여전히 mandatory다. 그러나 이 재사용 개발 집합에는 성공 p-value나 graph-population claim을 부여하지 않는다. “5개 중 좋은 것만”, “full/edge 중 큰 값”, “C-fixed/C-global 중 약한 값”으로 판정을 바꾸지 않는다. 이미 알려진 Coauthors/CiteSeer 반대 방향을 양쪽 모두 보존한다. 재사용 public/PCFI/FISF와 두 LP는 정확한 episode/query/label/hash 및 입력 인터페이스가 동일한 saved probabilities만 사용하며, best baseline은 descriptive summary일 뿐 실행 selector가 아니다.

**Resource 선행 조건:** Training은 small synthetic full graph만 한다는 설계를 유지한다. Real resource preflight는 최소 CoauthorCS N=2000,K=15,F=16, observed100과 CiteSeer의 최대 source count를 포함한다. Pooled-state dense allocation을 만들면 graph Potts보다 centroid grid가 커질 수 있으므로 model-level peak memory와 prediction CPU-return까지 latency를 기록한다. Q/추가 grid resolution을 평가 진단으로 제한해 architecture별 update budget을 늘리지 않는다. 실패한 model은 best seed/epoch로 대체하지 않는다.

**Final lock:** code/config/base train/dev hashes, quadrature rule, all thresholds, seeds, checkpoint paths, primary stratum, full/edge role rule을 먼저 잠그고 final bank를 생성한다. Final을 보고 design choice를 바꾸면 그 bank는 development evidence가 되고 다음 cycle에는 새 final seeds가 필요하다. 실제 구현 결함은 원본과 수정 provenance를 보존하며 영향을 받는 계산만 재실행하고, 이를 방법 개선과 혼동하지 않는다. “New scalar/shape settings”는 현재 문구대로 열어두지 말고 구체 값/범위/가중치를 잠그거나 FP01에서는 제외하는 편이 낫다.

## 예상 가능한 결과의 해석

- **B>A, C≈B≈C-global:** RP surrogate에 맞춘 재학습의 효용이지 adaptive centroid contribution이 아니다.
- **C-global>B, C≈C-global:** richer statistical feature family의 성과이며 adaptive encoder 필요성은 미확인이다.
- **C>C-global/B + full−none positive + probability guard:** 이 frozen graph/synthetic family 안에서 adaptive feature inference를 지지한다. Real development feature gains가 별도로 재현되어야 real utility로 확장할 수 있다.
- **Edge accuracy만 개선, feature-none이 동일/더 좋음:** 기존 구조 복구 현상의 연장이지 missing-feature 성공이 아니다.
- **R에서 실패·Q에서 성공:** covariance/conditioning/RMS 효과를 포함한 유력한 진단이며 mean-shape mismatch 하나를 유일 원인으로 확정하지 않는다.
- **모든 새 arm 실패:** 이 budget·checkpoint·score family의 음성 결과를 기록한다. 같은 final bank에서 noise/head/loss/temperature를 바꾸며 계속 평가하지 않는다.

검토 완료. 위 수정 중 필수는 C-global과 공통 role/none parity, endpoint·bank·예산의 구체적 lock, A/B cross-view attribution이다. Q diagnostic와 별도 AMP companion은 주가설을 흐리지 않을 작은 범위로 제한해야 한다.


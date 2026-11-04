# FP03 PLAN_v1 — 독립 method / prior-art review

2026-11-04. 최초 대상은 `factor_context_cycle/PLAN_v1.md`이며, 마지막에 `PLAN_v2.md`와 `protocol.json`도 읽어 수정 반영을 확인했다. 과거 FP01/FP02 실패 기록, FP02 입력 manifest와 원본 context/edge 배열을 읽었으며 신규 추론·학습·평가·branch 변경·추가 reviewer는 실행하지 않았다. 작성 파일은 이 문서뿐이다.

**판정: 제한된 context-calibration 개발 실험으로 진행할 수 있다.** 원 PLAN_v1의 individual LOO는 class-count cue와 empty-class 평균, log-score 수치 convention을 수정해야 했다. 아래에서 root가 제안한 balanced leave-one-per-represented-class folds와 고정 probability floor를 검토했으며, 이를 PLAN_v2/protocol에 정확히 반영하면 구현을 막는 방법상 이슈는 없다. 설계의 실행 가능성은4/5, context로 두 contrast를 식별한다는 과학적 근거는2/5, 독립 PFN 방법 novelty는1/5다. 성능 결과는 아직 없으며 이 점수는 성공 확률 추정이 아니다.

## 보존할 강점

FP02의 잘못된 self-belief population/affinity feedback을 반복하지 않고, 실제로 관측한 context label의 held-out predictive score를 쓰는 변경은 계산적으로 구체적이다. Query truth를 weighting에 넣지 않고 graph-only weights를 none/95에 공유하는 점, prior/fixed/MAP/graph-off controls를 같은 candidate predictions에서 재구성하는 점, 기존 strong references와 실패 gate를 모두 보존하는 점이 좋다. 15개 family·unit mean score·수정된285-branch calibration 예산을 결과 전에 고정하고, 실패 후 grid/temperature rescue를 금지해야 한다.

Conditional Potts score는 finite label space에서 정당하게 정의할 수 있다. Nonedge strength를 단일 homophily rate로 묶지 않는 것도 타당하다. 이 형식의 validity는 실제 posterior calibration이나 충분한 표현력을 보장하지 않는다.

## 주요 이슈와 실행 전 수정 — 최대5개

### 1. [수정안으로 해소] Individual LOO의 class-count cue와 empty class

FP02 manifest의 같은 none 입력에서 원본 `context_ids/labels_context/src/dst`만 읽은 결과는 다음과 같다. Query labels는 이 점검에 사용하지 않았다.

| Graph | context 수 | class별 context 수 | known-known unordered edges | same-class edges |
|---|---:|---|---:|---:|
| Cora | 28 | 모든7개 class에4 | 3 | 3 |
| CiteSeer | 24 | 모든6개 class에4 | 0 | 0 |
| PubMed | 12 | 모든3개 class에4 | 1 | 0 |
| CoauthorCS | 55 | `[4,4,4,3,4,4,4,4,4,0,4,4,4,4,4]` | 2 | 2 |
| CoauthorPhysics | 20 | 모든5개 class에4 | 2 | 2 |

원 명세의139 contexts·2085 individual-LOO branches 계산은 맞지만, class당4개인 context에서 하나만 숨기면 특정 class count가4→3이 되어 held class의 단서가 된다. 또한 CS class9의 per-class mean은 정의되지 않는다. Root가 제안한 다음 수정이 이 두 문제를 닫는다.

- `C+={c:n_c>0}`, `m=min_{c in C+}n_c`로 정의한다. 각 represented class에서 stable provided node ID 순으로 m개를 선택한다. m개의 fold마다 class당 한 개씩 동시에 숨기고, 나머지 m−1개/class만 clamp한다.
- CS는 C+=14,m=3이므로42개 selected contexts,3folds다. 남는13개 context는 **모든 calibration fold에서 unclamp**하고 final full-context prediction에서만55개 모두 사용한다. 나머지 graph는 모두4folds다. 합계126 held points,19folds,285 graph-only candidate branch forwards이며 final150 branches는 그대로다.
- Stable IDs와 per-class selection은 split metadata일 뿐 model covariate가 아니다. Node permutation 때 ID도 node와 함께 이동시킨다. Hidden-node 목록·one-per-class constraint·held labels는 forward에 전달하지 않고 예측을 뽑는 단계에서만 읽는다.

선택된 class별 node를 i(c,t), fold t에서 숨긴 전체 node 집합을 H_t라 하면 score는

\[
S_g=\frac1{|C_+|}\sum_{c\in C_+}\frac1m
\sum_{t=1}^{m}\log\widetilde p_g^{(-H_t)}(y_{i(c,t)}).
\]

비어 있는 class는 평균에서 제외하고 수/coverage를 저장한다. 0점·chance score·query class 빈도로 대입하면 다른 알고리즘이다. Context가 전혀 없으면 사전 균등 weights라는 기존 규칙을 쓴다. m=1이면 모든 calibration anchor가 꺼지고, graph-only의 class symmetry로 모든 candidate가 uniform1/K, 따라서 같은 score와 prior weights를 반환해야 한다. 이것은 직접 검사할 좋은 무정보 경계다. 대표되지 않은 class의 predictive risk는 여전히 검증되지 않는다.

이 수정은 calibration condition도 바꾼다. 예를 들어 CS fold는 class당2개·총28개 anchors를 사용하지만 final에는55개를 모두 쓴다. 따라서 score가 final full-context risk의 정확한 추정값이거나 candidate ranking이 불변이라는 보장은 없다. Class별 동일 가중의 context prediction은 불균형 query population의 accuracy/NLL에 대한 불편 추정량도 아니다. 이 coverage 차이와 macro-risk 선택을 결과 해석에 남겨야 한다.

### 2. [수정안으로 해소] log-score와 finite-precision zero의 convention

`b=±2`의 field는 degree에 따라 강해질 수 있고 `a=log16`도 강한 edge evidence다. Float32 probability가0으로 underflow한 뒤 `log(p)`를 취하면 이론상 유한한 score를 −inf로 바꿀 수 있다. 그 결과 clipping convention이 weighting에 암묵적인 새 setting이 된다.

Root는 calibration의 probability floor1e−8와 float64 row normalization을 채택했다. 이를 다음처럼 정확히 잠그면 원래의 log-domain 제안 대신 유효한 convention이다.

\[
\widetilde p(c)=\frac{\operatorname{clip}(\operatorname{float64}(p(c)),10^{-8},1)}
{\sum_k\operatorname{clip}(\operatorname{float64}(p(k)),10^{-8},1)}.
\]

`log(tilde_p(true_label))`로 score하고 raw held probabilities, floor, clipping count와 derived log scores를 저장한다. 그러면 모든 weight를 저장 arrays에서 재구성할 수 있다. 이것은 floored/normalized predictive distribution의 log score이며 원래 exact Potts distribution의 log score와는 다르다. Calibration과 보고 NLL의 floor/row-normalization convention도 각각 명시한다. 결과를 보고 floor를 바꾸지 않는다.

`(a,b)=(0,0)` graph-only candidate는 항상 uniform1/K이므로 finite fallback score가 있다. 모든 score가 NaN/−inf여서 무조건 prior로 복구하는 일반 fallback은 수치 오류를 감출 수 있다. 유효한 finite log-domain 결과와 실제 예외를 구분한다. NaN이나 실행 실패를 후보에서 사후 제거하고 남은 후보로 정규화하면 사전 정의한15개 predictor가 바뀐다.

### 3. Predictive weighting은 identifiable relation posterior가 아니다

Balanced context CV는 직접 labeled edge가0개인 CiteSeer에서도 다른 context와 unlabeled paths를 통해 predictive signal을 얻을 수 있다. 그러므로 위 적은 edge 수가 곧 calibration 불가능을 뜻하지 않는다. 반대로 같은 unlabeled graph와 거의 같은 context를 공유하는 folds는 독립 관측이 아니고, a/b가 서로 보상하면 유사한 held prediction을 낼 수 있다. 이 score로 true edge/nonedge contrast나 homophily를 유일하게 복원했다고 말할 수 없다.

현재 `w_g∝exp(S_g)`는 균등 prior 위에 **mean class log score**를 적용한 보수적인 exponential predictive weighting이다. Individual-LOO-sum pseudo-BMA도, mixture log score를 최적화하는 stacking도, exact Bayes posterior도 아니다. 표본 수가 늘어도 평균 score가 같으면 같은 weights가 나온다. ESS/entropy는 이 선택된 weighting의 dispersion이지 finite-context epistemic uncertainty의 credible interval이 아니다. Unit mean scale을 사전 고정한 것은 타당하지만 data-size-adaptive confidence라고 설명하면 안 된다.

LOO model averaging 자체는 [Yao et al., *Using stacking to average Bayesian predictive distributions*](https://arxiv.org/abs/1704.02030)의 stacking/pseudo-BMA 비교와 직접 연결된다. 현재 규칙이 그 논문에서 최선이라고 검증된 구현이라는 뜻은 아니다. Finite-sample selection variance는 [Cawley–Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html)의 일반적인 경고도 적용된다. Context를 이용한 weighting은 허용된 statistical adaptation이다. “Neural optimizer updates0”과 “새 graph에서 아무 parameter/weight도 바꾸지 않는다”는 다른 문장이다.

실행 전에 저장할 항목으로 class coverage, 모든 held-node score, candidate 간 score range/near-tie, a/b의 marginal weights, entropy/ESS를 확정한다. 실제 정답을 쓴 oracle candidate는 성능 ceiling 진단으로 별도 표시할 수는 있지만 배포 predictor나 성공 기준으로 쓰지 않는다. 같은 데이터를 더 확인하며 grid를 늘리지 않는다.

### 4. 이 family는 원인을 모두 고치지 않으며, b 부호가 homophily 정답은 아니다

현재 d_i/sqrt(sum d) offset과 pair contrast는 **conditional working score**다. 정확한 degree-conditioned likelihood, 누락 edge exposure의 식별, 또는 semantic class와 structural community의 분리를 구현한 것은 아니다. Observed-edge a는 여전히 모든 연결에서 같은 equal-label contrast이고, nonedge b<0는 degree-volume 분할을, b>0는 global coassignment를 장려한다. 후보가 valid하다는 것과 그 충분통계가 실제 class 구조를 설명한다는 것은 다르다. Positive b가 선택됐다고 실제 graph가 heterophilic이거나 absence가 generative하게 positive evidence라고 단정하지 않는다.

Homophily와 absence의 informativeness는 별개다. Physics에서 strong homophily와 큰 nonedge deletion 이득이 공존했다. 이번 calibration이 검사하는 것은 “이 context와 graph에서 어떤 conditional prediction이 덜 틀리는가”다. A0가 observed absence인지 미관측 opportunity인지 판별한 것은 아니다. Nonedge MF와 finite loopy errors도 score의 일부로 선택되므로 exact score family의 true parameter 추정이라고 하지 않는다.

주어진 degree offset에서 무정보 self/neighbor exclusion과 negative/positive field의 부호는 모두 독립 검증해야 한다. 강한 후보의 saturation·entropy·predicted class mass도 진단으로 보존한다. 결과가 나쁘다고 field clip, population regularizer 또는 degree normalization을 사후 추가하면 새 cycle이다.

### 5. Primary gate는 classical graph calibration을 시험한다 — PFN novelty는 별도다

Feature-none에서 새 predictor는 conditional Potts/BP+MF와 LOO weighting의 **standalone classical inference mixture**다. C_global의 학습된 feature 계수는 이 조건에 사용되지 않는다. 따라서 none Physics/CS/CiteSeer gate를 통과하면 context-calibrated graph SSL의 개발 feasibility가 지지된다. Synthetic PFN 학습 성공, 새 feature-coupling 원리, 그래프 foundation model의 우월성으로 바꿔 쓰면 안 된다.

현재 prior mixture·fixed edge·MAP·graph-off·old full/edge 및 harmonic/restart LP는 올바른 핵심 대조다. Mean-score mixture−prior mixture는 weighting의 가치, prior mixture−fixed edge는 family/averaging 변경의 가치로 구분한다. Learned feature utility를 주장하려면 동일 weights에서 observed−none와 observed−shuffle 및 minority/NLL를 함께 보여야 한다. 이번10개 real rows에는 신규 shuffle이 없으므로 feature alignment 성공은 확정할 수 없다.

더 큰 방법 주장 이전에는 같은 context calibration 기회를 준 강한 LP/restart 또는 conditional Gaussian classical control이 필요하다. 지금 작은 pilot에 추가 구현을 강요할 blocker는 아니지만, 고정 LP만 이겼다는 것으로 learned BP/PFN의 필요성을 판정해서는 안 된다. 실제 learnable component는 inherited C_global일 뿐 새 network fitting은 없다는 계획의 표현을 유지한다. 실패를 성공으로 구제하기 위해 centroid weights를 target에서 fitting하지 않는다.

## 최소 검증 — 최대3개

1. **Fold와 score의 정확한 재구성:** CS의 empty class·13 extras, m=1, C=0을 포함하여 위 식과 독립 reference를 비교한다. Fold selection을 먼저 고정한 뒤 held label을 forward 경로에서 poison해도 해당 fold prediction은 같아야 한다. 남는13개 labels도 모든 calibration forward에서 차단한다. 저장 probabilities→floor/renormalization→class score→weights→final mixture를 재구성한다. Stable-ID node permutation, class permutation, chunk parity, m=1의 prior weights를 함께 검사한다.
2. **Family boundaries와 근사 분리:** b=0 tree의 exact oracle, a=b=0 uniform, empty/isolate, b 부호, self/neighbor exclusion, degree offset conservation, source0/1을 확인한다. Tiny full nonedge score의 exact enumeration·dense BP·sparse MF 차이를 각각 남긴다. 강한 후보의 underflow/entropy를 명시적 log-score convention으로 처리한다.
3. **유효성의 범위가 맞는 결과 판정:** primary adaptive mixture를 prior/fixed/LP와 사전 기준 그대로 비교한다. Candidate나 MAP를 graph별로 선택해 primary를 대체하지 않는다. 모든15개와 비용/실패를 보존하고, 신규 feature alignment·independent real CI·PFN novelty는 별도의 검증 없이는 주장하지 않는다. 재사용 개발 결과를 본 뒤 synthetic final generator를 유리하게 정의한 것을 독립 최종 확인이라고 부르지 않는다.

## 최종 권고

Balanced classwise folds·extras masking·stable IDs·고정 floor 공식을 PLAN_v2/protocol에 명시하고, 위 limited classical-calibration scope를 유지하면 구현으로 진행할 수 있다. 이 실험은 사용자가 요청한 “context를 보고 relational strength를 바꾸는 실제 계산”을 검증한다. 그 답은 oracle homophily 추정이나 자동적인 PFN method success가 아니라, 적은 context가 이 사전 고정 predictor family의 weighting을 실제로 개선하는지에 달려 있다.

**최종 amendment 확인:** 저장된 PLAN_v2/protocol은 balanced folds, C+ scoring,13 extras masking, stable-ID mapping,1e−8 float64 clip→row normalization,285/150 branch budget, 공통 none/95 weights, 두 NLL guard를 모두 명시한다. 위 필수 수정은 문서상 해소되었다. 구현·수치 검증을 진행할 수 있으며 이 review를 동결한다. 식별성·coverage shift·classical/PFN claim 범위는 해소된 것으로 바꾸지 않고 그대로 제한으로 남긴다.

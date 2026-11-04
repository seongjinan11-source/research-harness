# FP03 PLAN_v1 실험설계 검토

2026-11-04. Experiments/statistics reviewer. PLAN_v1과 FP02의 immutable manifest/실제 context 배열을 읽었다. 현재 cycle의 다른 reviewer 문서는 읽지 않았다. GPU/평가/학습/branch 조작을 수행하지 않았으며 이 파일만 작성했다.

**판정: 아래 명세 보완 후 제한된 개발 inference 실험으로 실행할 가치가 있다(4/5).** 고정15-family와 graph-only LOO, 공통 none/95 weights, prior/MAP/fixed-edge controls가 있어 FP02의 자기 예측 기반 nuisance update와 다른 질문을 실제로 검사할 수 있다. 새 neural 학습/새 PFN 또는 feature-conditioned Bayes posterior 검증은 아니다. 동일 개발 graph를 반복 본 상태이므로 성능이 나와도 독립 real 전이·논문 성공으로 확대하지 않는다.

## 0. 실행 전 수정안 재검토: balanced leave-one-per-class CV 채택

다른 검토에서 individual LOO의 **class-count cue**가 중요한 문제로 제기됐고, root가 결과를 계산하기 전에 balanced leave-one-per-represented-class CV를 제안했다. 이에 동의한다. V1처럼 class당4개에서 한 label만 가리면 held class만3개, 다른 class는4개가 남는다. Count와 nonedge class competition에 민감한 후보를 평가할 때 실제 graph signal 대신 이 selection cue가 score에 개입할 수 있다. 내 최초 검토는 이 문제를 충분히 지적하지 못했다. 현재 권고는 아래 수정안이며 뒤의139/2085 표와 individual LOO 문장은 **V1의 원래 카운트/검토 기록**이다.

수정할 split을 정확히 고정한다.

1. 원 context에서 `C+={c:n_c>0}`, `m=min_{c in C+} n_c`를 정한다. K는 원 task class 수다. Class별 context node를 **ascending stable node ID**로 정렬해 처음 m개를 calibration subset으로 선택한다. 다른 사전 고정 hash ordering을 쓰려면 seed/hash rule을 먼저 문서화해야 하며 node degree/feature/query 결과로 subset을 고르지 않는다. Stable-ID 선택이 무작위 대표 표본이라는 주장도 하지 않는다.
2. Fold r=0..m−1은 각 represented class에서 선택된 r번째 node 하나를 동시에 가린다. Clamped calibration nodes는 선택된 나머지 m−1개/class뿐이다. **선택되지 않은 extra context label은 모든 calibration fold의 입력·clamp·initialization·label-dependent cache에서 제거한다.** CS의 extra13개를 계속 clamp하면 balanced 수정의 목적이 사라진다. Extra도 최종 full-context prediction에는 원래대로 복구한다.
3. Saved split에 원 context IDs/labels, C+, m, selected IDs, unused IDs, fold별 held IDs 및 clamped IDs와 SHA를 보존한다. 모든 fold에서 represented class별 clamp count가 m−1로 같고, held/unused IDs가 clamp와 disjoint임을 검사한다. Held label은 완료된 예측의 log-score 계산에만 쓰며 query truth는 calibration에 전혀 쓰지 않는다.
4. Score는 selected m개/class의 held log probabilities를 class-macro 평균한다. Float64 clip[1e-8,1]+row renormalization, mean-log score 부호, exponent1, MAP order 및 none/95 common weights는 원래 보완대로 고정한다. CS는 C+14개만 calibration하며 class9의 query4개와 K15는 그대로 유지한다.

| Graph | 원 full context | Calibration held points | Balanced folds | Candidate branch forwards |
|---|---:|---:|---:|---:|
| Cora | 28 | 28 | 4 | 60 |
| CiteSeer | 24 | 24 | 4 | 60 |
| PubMed | 12 | 12 | 4 | 60 |
| CoauthorCS | 55 | 42=14×3 | 3 | 45 |
| CoauthorPhysics | 20 | 20 | 4 | 60 |
| 합계 | **139** | **126** | **19** | **285** |

따라서 현재 계획의 비용은 **285 calibration branch forwards+150 final branches=435**다. Calibration에서 저장되는 held probability vectors는126×15=1890개이며 이것은 forwards285개 또는 독립1890표본이라는 뜻이 아니다. 각 branch 한 번에 fold의 여러 held node를 예측한다. Calibration은 none/95에 공통으로 한 번만 실행한다. 원 context139개를 모두 calibration target으로 썼다고 보고하지 않는다.

Context가 전혀 없으면 prior weights다. m=1이면 모든 selected context label을 한 fold에서 제거하므로 graph-only·class-exchangeable 초기화 아래 anchors가0이고, candidate별 held prediction은 uniform, adaptive weights도 prior와 같아야 한다. 이것은 tiny-context에서 의미 있는 mechanism adaptation을 식별할 수 없는 사전 경계이지 구현 실패가 아니다. 별도의 불균형 clamp fallback으로 m=1을 구제하지 않는다.

이 수정은 class-count cue를 제거하지만 calibration의 m−1 anchors/class와 final의 더 많은/full context 사이 조건 차이, CS 미관측 class, dependent folds 및 query-risk mismatch를 없애지는 않는다. 새로운 uncertainty/posterior 정당화도 아니다. Root가 위 split identity와 비용/경계 조건을 revised plan/protocol에 반영하면, 수치·누수 검사 후 실행하는 데 추가 실험설계 blocker는 없다. 아직 어떤 candidate outcome도 계산하지 않았다.

## Manifest 카운트: 139/2085는 맞다

이 절의139/2085는 V1 individual-LOO 명세에 대한 사실 확인이다. 현재 balanced-CV 실행 예산은 위의126 held points/19 folds/285 calibration forwards를 따른다.

Manifest의 none episode NPZ에서 context_ids/labels_context와 실제 src/dst를 읽어 확인했다. None/95는 같은 graph/context이므로 calibration은 graph당 한 번만 한다.

| Graph | K | Context | Class별 context 수 | LOO×15 | Context 간 undirected edge(동일 label) |
|---|---:|---:|---|---:|---:|
| Cora | 7 | 28 | 모두4 | 420 | 3(3) |
| CiteSeer | 6 | 24 | 모두4 | 360 | 0(0) |
| PubMed | 3 | 12 | 모두4 | 180 | 1(0) |
| CoauthorCS | 15 | 55 | class3은3, class9는0, 나머지4 | 825 | 2(2) |
| CoauthorPhysics | 5 | 20 | 모두4 | 300 | 2(2) |
| 합계 | — | **139** | — | **2085** | — |

CS의 전체 context counts는 `[4,4,4,3,4,4,4,4,4,0,4,4,4,4,4]`다. Context 없는 class9에도 query4개가 있으므로 이를 제거해 평가하면 안 된다. Calibration class support는14개이며 task의 K=15는 유지한다. 새 final candidate forwards는10 episode×15=**150**이고, 총 개념적 branch forwards는2235개다. Batch로 묶어 호출 수를 줄여도 계산량/독립 표본 수가 같은 뜻은 아니다.

## 실행 전 주요 보완 다섯 가지

### 1. 빈 class를 제외한 class-balanced log-score와 수치 규칙 고정

`loss_g`는 양의 loss를 음수로 더하는 값이 아니라 **평균 log predictive score**라고 명명해야 한다. 권장 정의는

    C+ = {c: number of visible original context labels in class c > 0}
    score_g = mean_{c in C+} mean_{i in C_c} log p_g^{-i}(y_i)
    w = softmax(log(1/15) + score).

CS의 denominator는14개 observed context class다. 빈 class에0 또는 가상의 uniform term을 추가하지 않는다. Group membership/held target label은 **예측이 끝난 뒤 score 계산에만** 사용하며 해당 fold의 입력·clamp·initial message·class-count feature에는 들어가지 않는다. K는 task interface 그대로 유지한다. Context0이면 score 계산 없이 uniform weights, 향후 singleton-context fold에서도 held label이 사라진 채 실행되어야 한다.

Log0/underflow를 위한 epsilon과 재정규화를 고정한다. 기존 saved-score 관례와 맞추려면 저장 확률은 그대로 두고, score 계산만 float64에서 clip[1e-8,1] 후 row normalize하여 log를 취하면 된다. 모든 후보가 낮은 확률을 내더라도 softmax가 NaN이 되지 않아야 한다. Epsilon/score scale을 결과 후 바꾸면 새 방법 변경이다. Mean-score exponent1은 composite/generalized predictive weighting이며139개의 독립 likelihood를 곱한 posterior가 아니다.

### 2. Gate의 mode와 “either”를 명확히

Primary development gate의 모든 mean 비교는 **none의 source graph5개 동일 가중**으로 정의하는 것이 현재 계획과 일관된다. `mean NLL<=either+.01`은 하나만 만족하면 되는지 두 control 모두 만족해야 하는지 모호하다. 권장은 **adaptive NLL<=prior-mixture NLL+.01 AND adaptive NLL<=fixed-edge NLL+.01**이다. 즉 min(two NLL)+.01을 ceiling으로 쓴다. Additional mean accuracy≥prior+1%p 및≥fixed-edge 역시 none에서 계산한다고 명시한다. Query 수 가중 pooled mean 또는 none/95 pooled mean으로 바꾸지 않는다.

Physics/CS/CiteSeer 조건, mean BAcc≥old-edge, mean NLL≤old-edge+.05는 결과 전에 그대로 고정한다. 조건별 차이/threshold/PASS를 모두 출력하고 conjunction으로 판정한다. MAP가 더 좋거나 특정 candidate가 더 좋아도 adaptive mixture의 FAIL을 대체하지 않는다. Gate PASS는 family weighting의 개발 실행 가능성만 뜻한다. 95−none은 별도의 feature 효과다. None gate만 통과한 것을 missing-feature 방법 성공으로 부르지 않는다.

### 3. 공통 weights와 fold 누수 검증을 실제 저장 identity로 연결

Calibration은 다섯 graph 각각에서 feature를 모두 끈 뒤 한 번만 계산하고 none/95에서 같은 저장 weights를 재사용한다. None/95의 topology/context/hash identity를 검증해 잘못된 cache reuse를 막고, 같은 graph에서 두 번 calibration해2085가4170으로 늘지 않게 한다. Candidate order는 a-major/b-minor Cartesian order처럼 고정한다. 이 순서라면 fixed edge는 index10, graph-off는 index4(0-based)다. MAP tie는 첫 index이며 near-tie threshold를 사후 추가하지 않는다.

Fold에서는 held context label을 initialization과 모든 label-dependent cache에서 제외해야 한다. Graph topology/degree opportunity는 공유할 수 있지만 full-context BP의 node/centroid/cavity를 fold seed로 재사용하면 안 된다. Own-fold label poison 테스트는 score 산출용 별도 target과 예측 입력을 분리해서 수행한다. Future synthetic의 context1/class에서도 같은 규칙이다. Source features는 calibration에서 꺼지고 frozen C_global hyperparameters에도 hidden statistics가 들어가지 않아야 한다.

최종 candidate query probabilities와 LOO `[15,C,K]`를 저장하면 모든 mixtures/MAP/fixed controls를 **모델 재실행 없이 독립 재구성**할 수 있다. Save NPZ/sidecar SHA, source/checkpoint/bank/query/context identity, candidate score/weights/ESS/entropy와 all controls를 남긴다. None/95 common weights는 bit-identical cached 값 또는 엄격한 수치 tolerance로 검증한다. Graph-off는 own-feature-missing query에 graph 경로가 없고 uniform class prior이므로 uniform query prediction이 되는지 좋은 추가 sanity check다.

### 4. Prior/fixed controls는 mandatory, MAP/개별 winner는 진단

Uniform mixture와 adaptive mixture는 같은150 candidate predictions를 사용해 **가중치 변화만** 비교한다. Fixed-edge와 graph-off도 해당 index를 그대로 꺼낸다. MAP와 모든15개 개별 candidate를 보존하되, 결과 후 primary로 승격하지 않는다. Best-candidate-by-query 또는 graph별 full/edge 선택은 oracle diagnostic으로도 deploy predictor와 분리한다. 이15개는 여러 모델의 posthoc prediction ensemble이 아니라 계획에 고정된 conditional-family inference mixture라는 범위를 유지한다.

70 reused references는2 old C_global panels+5 external per episode로 한 번씩 센다. FP02 manifest/reference identity가 현재도 맞는지, old weights가 homophilic conditional임을 다시 확인한다. LP harmonic/restart, public/PCFI/FISF-PFN을 같은 query/context에서 모두 남긴다. Class별 recall/support·BAcc·macro-F1·NLL/Brier를 같이 보고하며 CS class9 query4개도 평가에 남긴다. Accuracy만 좋아지고 minority/NLL이 나쁜 결과는 별도로 기록한다.

Calibration의 context score 순위가 좋아졌다는 사실은 query accuracy의 이득이 아니다. 이전 reliability selector 실패는 proper-score 순위와 accuracy 효용이 어긋날 수 있음을 이미 보여줬다. 따라서 prior 대비 +1%p와 fixed-edge/NLL guard는 필요하다. FP02에서처럼 잘못된 분할을 더 확신하는 결과를 confidence 변화만으로 개선이라 해석하지 않는다.

### 5. 비용과 증거 단위를 분리

실제 실행은 graph5개/episode10개/139 dependent held nodes다. 2085 branch forwards와150 final branches를139 또는2235개의 독립 관측으로 세지 않는다. Real CI를 붙이지 않는 계획에 동의한다. 전체 score 평균은5 graph 동일 가중이고 LOO score는관측 class 동일 가중이며 두 목적은 다르다.

Calibration cost를 graph당 한 번, final candidate inference를 episode당 한 번, mixtures/score 재구성 cost를 별도로 기록한다. Uniform/MAP/fixed controls는 추가 BP 호출이 없다. Aggregate control5개×10episode=50 prediction records와 reused70 references를 점수화한다면120 aggregate scores이며, 별도150 candidate diagnostic probabilities와 혼동하지 않는다. Candidate/fold chunk16은 source lock 전에 parity·memory 검사로 고정하고, OOM에 따른 재시도/실패도 비용에 포함한다. Chunk를 바꾸는 operational repair가 필요하면 probability parity와 사유를 보존하고 outcome 선택에 쓰지 않는다.

Cold import/model load/preprocessing/H2D/calibration/final/D2H/scoring/end-to-end wall, CUDA peak/incremental peak를 보고한다. 한 번의 warmed-up branch time만 전체 latency로 쓰지 않는다. Learned feature checkpoint의 역사적 학습 비용은 inherited로 분리하고 이번 neural training은0이라고 적는다. Partial NPZ/sidecar strict resume, stale single-worker marker 처리, Git metadata once-per-run을 FP02 수준으로 유지하면 된다.

## Homophily와 다음 검증의 범위

Physics induced rep0의 전체-label homophily0.95817과 CiteSeer0.73550은 앞 cycle의 **audit-only** 숫자다. Context 간 edge2개/0개로 이 값을 신뢰성 있게 추정할 수 없다. 이번 LOO는 직접 context-edge 비율보다 multi-hop conditional prediction을 활용한다는 점에서 다른 검사지만, context가 작거나 semantic class와 graph partition이 다르면 여전히 식별이 어려울 수 있다. Mean-score unit1이 uncertainty를 완벽히 보정했다거나 ESS가 calibrated posterior confidence라고 주장하지 않는다.

새 synthetic factorial 검증은 현재 locked real 결과를 보고 grid를 고르는 수단이 아니어야 한다. Homophily×semantic-class coarsening을 별도로 교차하고 degree heterogeneity/context1·4/결측률을 사전 고정하는 방향에 동의한다. 같은 class histogram·density에서 pooled edge homophily와 pooled nonedge 빈도는 연결되므로 “nonedge 정보량을 완전히 독립 조작”했다고 쓰기보다 latent microcommunity coarsening/조건부 비동질성을 명시해야 한다. Generator·final seeds·primary contrast·graph bootstrap 단위는 그 bank 생성 전에 별도 검토/lock한다. Unseen real graph는 이 개발5개와 별도 확보해야 한다.

추가 학습이나 대형 실험을 이번 파일의 선행 조건으로 요구하지 않는다. 위 정의를 고정하고 수치/누수 검사 PASS 후 계획된10개 개발 case를 실제 실행하는 것이 적절하다. 실패하면 a/b/score-temperature를 같은 사례에서 재탐색하지 않고 이 한정된 context-adaptive hypothesis의 미통과로 남긴다.

## 최소 검증 세 가지

1. 원 context139개, balanced calibration126 held points/19 folds와 CS의 C+14/m3/unused13을 정확히 재현하고, fold별 균형 clamp·held-label poison/none95 weight identity/candidate-fold chunk parity를 검사한다.
2. 저장 LOO 확률→score→weights 및 final candidate 확률→5 controls를 독립 CPU code로 재구성하고 NPZ/sidecar/index의 hash·query·truth를 확인한다.
3. 모든 graph/none95/70 references/15 candidates/5 controls와 gate 각 항목, class별 지표 및 calibration 포함 실제 비용을 보존한다. Audit PASS와 performance FAIL을 별개로 보고한다.

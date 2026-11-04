# FP03 PLAN_v1 독립 추론 검토

2026-11-04. 검토 대상은 현재 PLAN_v1이다. 다른 현재 review는 읽지 않았다. 코드 구현·GPU·새 모델 실행·branch 변경은 하지 않았다.

**조건부 PASS.** 제안된 conditional Potts family, graph-only context calibration, 최종 probability mixture는 일관되게 정의할 수 있다. 검토 중 root가 채택한 **class-balanced leave-one-per-class mask**를 primary로 반영한다. PLAN_v1의 ordinary single-label LOO는 primary가 아니다. 아래 field/readout·score·fold 격리를 PLAN_v2/API에 고정해야 한다. 평균 log score의 softmax가 Bayesian posterior나 ambiguity 보존을 보장한다는 해석은 금지한다. FP02의 Gamma 자기 갱신을 제거한 것은 실제 연산 변화지만 성공은 아직 미확인이다.

## 1. 정확히 구현해야 할 graph 갱신식

Directed symmetric topology의 `u_{i→j}(c)`를 source i가 target j의 class c에 보내는 log message라 하자. `theta_i=d_i/sqrt(v)`이므로 `theta_i theta_j=d_i d_j/v`다. 아래 volume에 v로 다시 나누면 안 된다.

이전 node beliefs p와 이전 graph/feature messages에서

\[
V_i(c)=\sum_{j\notin\{i,N(i)\}}\theta_jp_j(c),\quad
H_i(c)=b\theta_i V_i(c),
\]
\[
F_i(c)=\log(1/K)+H_i(c)+\sum_{j\in N(i)}u_{j\to i}(c)
+\sum_{f:\,M_{if}=1}A_{ifc}
\]

를 만든다. `H`는 **더한다**. b<0이면 같은 class의 nonneighbor volume을 벌하고 b>0이면 지지한다. Self와 모든 observed neighbor의 theta-weighted volume을 제외한다. Observed edge는 Potts a만 포함하고 nonedge field에도 다시 포함하지 않는다. a=0인 observed edge도 topology상 neighbor이므로 nonedge field에서 제외하는 것이 맞다.

Graph cavity는

\[
r_{i\to j}=\operatorname{clampedSoftmax}(F_i-u_{j\to i}),\qquad
\hat u_{i\to j}(c)=\log[1+(e^a-1)r_{i\to j}(c)].
\]

다음 message는 gauge를 뺀 proposal과 old message의 `.5/.5` **log damping**이다. 음수 a도 같은 식으로 처리한다. Reverse edge index가 정확히 `j→i`를 가리켜야 한다. Node clamps는 cavity와 최종 readout 모두에 적용한다. Nonneighbor field는 node-product MF라 exact all-pair BP가 아니다.

**필수 정의 수정:** 24번째 step 뒤 hidden 25번째 MF update를 하지 않도록 정한다. FP02와 같이 마지막 step에서 실제 사용한 `field_node_belief`, H를 저장하고, 그 H와 갱신된 messages로 final field/p를 재구성하는 규칙을 권한다. Final edge/feature pair 진단도 같은 저장된 field/cavity를 사용한다. 다른 규칙도 가능하지만 하나만 명시해야 한다. Dense MF reference에는 정확한 이 iterate를 비교하고, dense full BP/exact enumeration과의 차이는 별도 근사오차로 보고한다.

## 2. Feature gate는 active class의 해당 factor만 제외한다

Observed cell마다 class c의 centroid factor는 `psi(y_i,z_cf)=exp(ll_if(z_cf))` if `y_i=c`, otherwise1이다. `A_ifc`는 그 factor의 active-class log contribution, `B_ifc(r)`는 centroid로 가는 log message라 하자.

\[
T_{cf}(r)=\log w_f(r)+\sum_{i:M_{if}=1}B_{ifc}(r),\quad
\log q_{cf\setminus i}(r)=\operatorname{logsoftmax}_r(T_{cf}-B_{ifc}).
\]

Node→factor cavity의 active numerator는 `F_i(c)−A_ifc`지만, inactive d≠c에는 **F_i(d)를 그대로 유지**한다. 다른 class gate까지 모두 빼면 다른 모형이다. 따라서

\[
\rho_{ifc}={e^{F_i(c)-A_{ifc}}\over e^{F_i(c)-A_{ifc}}+\sum_{d\ne c}e^{F_i(d)}},
\]
\[
\hat A_{ifc}=\operatorname{logsumexp}_r(\log q_{cf\setminus i}(r)+\ell_{if}(r)),\quad
\hat B_{ifc}(r)=\log[(1-\rho_{ifc})+\rho_{ifc}e^{\ell_{if}(r)}].
\]

Context이면 rho는 정확한0/1이다. 모든 proposals는 같은 old field/messages에서 동기적으로 계산한다. Normal 상수와 normalized GH prior를 보존하고, prior를 두 번 곱하지 않는다. Hidden cell의 ll은 선택해서 읽지 않아야 한다. Missing query에는 centroid factor를 만들지 않는다. Feature evidence의 경로는 source→centroid→source class→graph→missing query다.

Source0 또는 각 column당 source1이면 singleton integral은 class-independent다. Candidate별 final p가 corresponding no-feature p와 같아야 하며 mixture에서도 같아야 한다. Source row 전체1뿐 아니라 서로 다른 row에 column별 singleton을 둔 검사도 유지한다. 최종 mixture의 likelihood gradient는 graph-only calibration weights를 고정한 상태에서 검사한다.

## 3. 채택된 balanced-mask와 배치/cache 격리

Ordinary LOO에서는 held class의 남은 context count만 하나 줄어 nonedge field에 설계상의 단서가 들어간다. Root의 수정은 이 단서를 primary에서 제거하므로 타당하다. `C+={c:n_c>0}`, `m=min_{c∈C+} n_c`를 두고, 각 class의 context를 **제공된 stable node ID**로 정렬해 앞 m개만 calibration에 쓴다. Fold r는 각 represented class의 r번째 node 하나씩을 동시에 숨기고, 나머지 선택된 m−1개/class만 clamp한다. Stable IDs는 split metadata이며 모델의 숫자 feature가 아니다.

1. Fold별 held set H_r와 **선택되지 않은 extra context 모두**를 모든 초기화 전에 −1로 만든다. CS는 선택42개 중 fold당14개를 숨겨28개를 clamp하며 extra13개도 항상 calibration에서 숨긴다. 원래 full-context p/messages, original class histograms, feature-informed initialization을 warm-start하면 안 된다. Held labels/C+ grouping은 prediction 뒤 score evaluator에서만 읽는다.
2. Share 가능한 것은 topology/reverse indices, degree/theta, candidate constants 및 label-independent 배열이다. Label-dependent node/graph/centroid messages와 H는 fold마다 별도다. 최종 full-context run도 fold state에서 warm-start하지 않는다.
3. Packed batch의 각 element는 `(candidate_id,fold_id)`다. Volume/index_add는 같은 element의 node axis에만 합한다. Candidate/fold 축을 합친 class volume이나 messages를 사용하지 않는다. Serial·chunk1·chunk16·마지막 partial chunk parity를 확인한다.
4. Own-label poison 검사는 **split을 먼저 고정하고 calibration score 이전의 fold prediction**에 적용한다. Held set 전체와 unused extra labels를 바꿔도 그 fold p가 같아야 한다. 라벨 변경 후 class selection/split을 다시 만들면 다른 fold가 되어 올바른 poison 비교가 아니다. Held truth는 정당한 calibration target이므로 score/최종 weights까지 poison 불변을 요구하면 잘못된 검사다. 다른 실제 clamped labels는 당연히 해당 fold prediction에 영향을 줄 수 있다.
5. Graph-only calibration은 observed mask가 완전히 꺼진 별도 경로다. 95의 C_global statistics/centroid messages를 먼저 계산해 seed로 넣지 않는다. None/95의 topology/context 동일성을 hash로 확인한 뒤 weights를 graph당 한 번 저장한다. Calibration cache identity에도 graph·context·candidate order·operator SHA·fold mask를 포함한다.

**균형 수정의 제한:** 각 represented class의 남은 node count는 같지만 theta-weighted volume까지 같아지는 것은 아니다. 이를 균등 degree·관계 없는 context 배치의 null diagnostic과 구분해 기록한다. C+가 K보다 작아도 모델 prediction/normalization은 **전체 K class**를 유지한다. 평가자가 모르는 class에 probability0을 주거나 C+로 재정규화하면 안 된다. m=1이면 extra도 모두 숨겨 graph-only labels가 하나도 남지 않으므로 모든 held prediction은1/K, candidate weights는 정확한 prior여야 한다. 이때 모델에 C+ 목록이나 원래 histogram을 전달하면 symmetry를 깨는 우회 입력이 된다.

선택된 stable IDs, unused IDs, fold별 held/clamped IDs, class counts/hash를 evaluator metadata로 보존한다. Node permutation 검사에서는 stable IDs도 같은 node를 따라 이동해야 하고 class permutation에서는 같은 physical held sets가 유지되어야 한다. Stable-ID 선택은 deterministic split이지 무작위 대표 표집이 아니다. Calibration에는 m−1개/class만 있고 최종에는 원래 모든 context가 있으므로 더 적은 anchor에서 full-context로의 전이 차이가 남는다. 특히 CS의 extra13개는 calibration에서 제외되고 최종에는 모두 돌아온다. 비용·성능 해석에 이 차이를 공개한다.

## 4. Mean log score와 혼합의 필수 명세

`loss_g`는 이름과 달리 **큰 값이 좋은 log score**다. 혼동을 막기 위해 `score_g`로 표기하기를 권한다. K_C=|C+|, h_rc를 fold r에서 숨긴 class c의 node라 하면 채택된 balanced-mask의 점수는

\[
s_g={1\over K_C}\sum_{c\in C+}{1\over m}\sum_{r=1}^m
\log\bar p_g^{-H_r}(y_{h_{rc}}),
\quad w_g=\operatorname{softmax}_g(\log(1/15)+s_g).
\]

Mean score를 다시 context 수로 곱하지 않고, weight temperature를 추가하지 않는다. Context0이면15개 uniform prior다. 일부 class에 context가 없으면 empty mean을 계산하지 않는다. 가중 평균의 grouping에 쓰는 원래 held class는 score evaluator에만 존재한다.

**필수 수치 수정:** float32의 정확한0 probability를 어떻게 score할지 고정한다. 기존 scoring과 같이 `pbar=clip(p,1e-8,1)` 후 row normalization을 사용하는 것이 한 명확한 선택이다. 그렇지 않으면 모든 s=-inf에서 weights가 NaN일 수 있다. Floor는 scoring에만 쓰고 BP likelihood나 pair opportunity를 clip하는 규칙과 섞지 않는다. Raw p와 scored p 정의를 함께 보존한다.

최종은 `p_mix=Σ_g w_g p_g`, 그 뒤 `−log p_mix(y)`를 재채점한다. Branch NLL의 가중 평균, weighted log probabilities의 softmax, message/field를 먼저 섞는 계산은 다른 방법이다. Candidate table의 Cartesian enumeration 순서와 control indices를 고정한다. a-outer/b-inner라면 fixed edge index10, graph-off index4(0-based)다. MAP exact tie의 lowest index 규칙도 그대로 저장한다.

Mean-class log score는 overlapping folds의 composite predictive weight다. 이 softmax에는 posterior consistency, calibrated uncertainty, 충분히 큰 ESS가 보장되지 않는다. 애매한 sparse context에서도 score가 우연히 갈릴 수 있고, 반대로 mean을 쓴 덕분에 과도하게 평평할 수도 있다. Entropy/ESS와 선택된126×15 held probability vectors를 그대로 보고, weight의 분산 자체를 epistemic calibration 성공으로 부르지 않는다.

현재 개발5graph의 balanced-mask는4/4/4/3/4=19fold, candidate별 branch forwards285개다. 각 fold가 여러 held nodes를 함께 채점하므로 held node수126개/확률 vector1890개와 forward count를 혼동하지 않는다. Final candidate150개를 더해435 branch forwards다. Stable-ID selection/grouping/mask 생성의 CPU 시간도 preprocessing/calibration 비용에 포함하고, graph당 한 번 수행한 calibration 비용을 none/95에 중복 집계하지 않는다. Original context139개 중 calibration126개만 채점하지만 final에는139개 전부 쓰는 차이를 보고한다.

## 5. 구현 착수 전 닫을 항목과 최소 checks

필수 수정은 **최종 MF/readout iterate**, **log-score floor·empty class·부호**, **candidate 순서/배치별 context 초기화·cache identity**, **balanced-mask에서 extra labels까지 제외**의 명문화다. 위 식의 feature cavities를 그대로 적용하면 별도의 새 inference 아이디어를 추가할 필요는 없다. Ordinary single-label LOO count cue는 primary에서 제거하며, reduced-anchor/selection/coverage 차이는 결과 해석의 제한으로 기록한다.

최소 검사는 b=0 tree exact, arbitrary old-state graph contraction, arbitrary previous-p의 직접 nonneighbor sum, complete graph field0, a=0인 observed edge의 nonneighbor 제외, positive/negative a/b sign, dense MF serial parity, exact/loopy error 분리, chunk parity, frozen split의 held/extra-label poison, stable-ID/class permutation, m=0/1, source0/column-singleton, hidden NaN과 finite gradient다. a=b=0에서는 own-feature-missing query가 uniform이어야 하며 feature를 통해 가짜 query factor를 만들지 않아야 한다.

이 정의를 고정한 뒤 numerical checks와 source lock을 거쳐 계획된 개발 실행으로 진행할 수 있다. 현재 문서의 검토는 새 모델 성공·Bayesian mechanism identification·learned PFN의 추가 기여를 승인한 것이 아니다.

## PLAN_v2 대조와 최종 판정

Root의 `PLAN_v2.md`와 `protocol.json`을 읽어 위 필수 수정의 반영을 확인했다. Balanced masks/extras 제외, full-K prediction, m0/1, score floor와 row normalization, candidate 순서, 마지막 MF state, 공통 none/95 weights, NLL의 두 comparator guard가 명시됐다. **계획 수준의 추론 blocker는 해소되어 구현·독립 numerical checks로 진행할 수 있다.** Stable-ID carry와 frozen-split poison 및 cache 격리는 구현 검증에서 확인해야 한다. Reduced-anchor/represented-class coverage와 generalized weighting의 불확실성 제한은 여전히 남으며 scientific gate를 대체하지 않는다. 본 review를 이 상태로 고정한다.

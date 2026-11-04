# FP02 Stage0 독립 추론 결과 검토

2026-11-04. `PLAN_v2`, protocol, operators/checks, REPORT/summary/audit/manifest 및 저장 NPZ를 검토했다. 다른 현재 결과 review는 읽지 않았다. CPU에서 원래 확률/통계만 재계산했으며 source·설정·prediction·branch를 변경하거나 GPU/학습을 실행하지 않았다.

**판정: audit·작은 수학 구현은 PASS, 사전 scientific gate는 명백한 FAIL이다. 현 dc_update를 그대로 Stage1의 성공 후보로 승격할 근거는 없다.** 이는 학습된 PFN 전체나 모든 DC-SBM의 실패 판정이 아니라, 이번 고정 Poisson/BP/mean-field/nuisance 연산의 개발 실패다.

## 1. 구현 오류와 성능 실패를 구분한다

- Manifest의260개 파일 SHA를 독립적으로 모두 확인했다. 새30개 NPZ를 직접 재채점한 accuracy/BAcc/macroF1/NLL/Brier는 summary와 정확히 같았다. 원래 query/context/truth를 바꾸거나 node를 독립 표본으로 세지 않았다.
- 실제 저장된 Gamma proposals도 `원래 prior + 현재 통계`였다. 상대오차 최대8.73e-8, unordered exposure conservation 상대오차8.74e-7, edge-count conservation3.65e-8, population-count conservation1.24e-7로 float32 수준이다. 반복마다 같은 관측을 posterior에 다시 누적한 오류 증거는 없다.
- 독립 checks는 full-Poisson N=2 exact marginal, 별도 tree/dense oracle, unordered counts/exposures, theta/node-count 구분, self/neighbor 제외, full edge penalty 한 번 포함, 세 nuisance 갱신/최종 readout, permutation·poison·clamp·source0/1·gradient를 검증했다. 검증된 source와 실행 manifest가 연결된다.
- 이 PASS가 큰 graph의 BP 정확성이나 calibration을 보장하지는 않는다. Loopy BP, nonedge product MF, nuisance factorization의 오차는 남는다. Tiny sparse−dense 최대.02067735는 공개된 근사 차이다. 이번 review는 real CPU inference를 재실행하거나 exact real posterior와 대조하지 않았다.

따라서 현재 증거로는 **실제 코드 fault를 선언해 설정을 조정하고 같은 결과를 교체할 이유가 없다.** 실패한 결과와 연산을 그대로 보존한다.

## 2. 무엇이 실패했는가

None의 equal-five-graph 평균은 다음과 같다.

| 방법 | Accuracy % | BAcc % | NLL |
|---|---:|---:|---:|
| poisson_fixed | 47.226 | 51.961 | 4.07794 |
| sbm_update | 42.150 | 49.246 | 5.79886 |
| dc_update | 43.805 | 48.989 | 4.21414 |
| old_full | 55.691 | 54.599 | 2.53411 |
| old_edge | 63.750 | 57.432 | 1.62087 |
| LP restart | 66.319 | 59.848 | 1.63741 |

DC의 Physics는20.290% 대 old_full48.551/old_edge88.801%, CS는62.294% 대69.018/77.324%, CiteSeer는35.090% 대52.898/41.728%다. 세 사례 모두 목표 방향과 반대다. CiteSeer NLL도1.43452→2.89470으로 악화했다. 사전 일곱 DC 조건과 SBM의 대응 조건이 모두 실패했다.

보존할 작은 사실은 있다. DC의 Cora BAcc64.299%는 old_full60.022%보다 높고, CS의 SBM은 poisson_fixed보다 accuracy가2.307pp 높다. 그러나 Cora DC NLL은1.73434→2.59998로 나빠졌고 CS도 strong references보다 낮다. 이를 전체 repair나 calibration 성공으로 바꾸어 해석하면 안 된다.

Feature95−none의 DC 평균은 accuracy+.245pp/BAcc+.036pp/NLL−.00507이다. Physics는 accuracy−.198pp다. 신규 shuffle도 없어 feature alignment/utility 회복을 입증하지 못한다. 새 신경 학습은0회이며, 이전 C_global의3계수는 단지 공통 frozen feature factor다.

## 3. Gamma 자기 강화와 class mass 붕괴: 직접 확인된 범위

Physics query에서 실제 class2 비중은70.026%다. 저장 배열을 직접 읽으면 다음과 같다.

| Physics none | class2 hard 예측 비중 % | class2 population % | 평균 최고 confidence % | 전체 query 중 confidence>=.99인 오답 % |
|---|---:|---:|---:|---:|
| poisson_fixed | 13.241 | 20.000 | 96.352 | 50.527 |
| sbm_update | 4.743 | 3.394 | 98.075 | 66.864 |
| dc_update | 3.426 | 1.410 | 98.031 | 67.325 |

**붕괴는 첫 Gamma 갱신 이전에 이미 시작된다.** 첫6step 뒤 DC/SBM의 class2 unknown-node belief share는.10756%이고, 첫 population 갱신 뒤.20752%다. 이후 DC의 통계 share는.59120→2.05143%, population은.49566→1.41018%로 일부 반등한다. 따라서 “매 Gamma 갱신이 class2를 단조롭게 더 붕괴시켰다”는 설명도 틀리다. 최초 고정 score/BP 상태가 이미 잘못된 partition을 만들었고 nuisance 갱신은 그것을 실제 population 통계처럼 받아들였다.

반대로 잘못 크게 잡힌 class1의 population은 첫 갱신58.877%, 최종53.644%다. 그 class의 diagonal affinity Gamma shape는3459→4950→5448이고 conditional Gamma CV는약1.70→1.42→1.35%다. 이는 **잘못된 latent assignment에 조건부인 매우 좁은 affinity 추정**이다. Assignment/model uncertainty가 작다는 증거는 아니다. 원래 prior를 매번 한 번만 더하더라도, 자기 예측을 충분통계로 쓰는 approximate EM/Bayes feedback은 남는다.

같은 현상은 Physics만의 문제가 아니다. PubMed SBM은 실제13.61%인 class0에 hard mass76.40%, population81.45%를 주고 NLL7.70026, wrong99 query 비율61.34%에 이른다. CiteSeer DC는 실제 최대 class3의22.97%를 hard3.37%/population6.82%로 축소한다. **Population을 갱신할 수 있다는 사실과 실제 population을 알아냈다는 사실은 다르다.**

DC의 max class-conditional pair intensity는 Cora10.77/Cite1.74/PubMed2.66/CS2.58/Physics5.59로1을 넘는다. Binary saturation은 추가 working-model 위험이다. 그러나 theta=1인 poisson_fixed도 낮은 max intensity에서 이미 크게 실패하므로, saturation만 고치면 전체가 해결된다는 결론도 불가하다.

## 4. Semantic class 하나 = structural block 하나라는 가정

Degree correction은 node propensity의 이질성을 설명하지만, 같은 semantic class 안의 서로 다른 연구 주제/지역적 community까지 자동으로 분리하지 않는다. 여러 subcommunity가 같은 class를 공유한다면 이들 사이의 부재를 모두 하나의 class-pair rate로 설명하는 score가 실제 label task와 충돌할 수 있다. Free B와 pi를 추정해도 graph marginal fit이 semantic classification에 맞는 partition을 선택한다는 보장은 없다.

이번 결과는 이 가설과 일치하지만 **멀티커뮤니티가 고유 원인이라고 실증한 것은 아니다.** 초기 strength, finite iterations, sparse MF, conditional uncertainty, simple-graph likelihood도 함께 남는다. Metadata와 구조 group의 관계를 반드시 동일시할 수 없다는 점은 이미 annotated DC-SBM 연구에 명시되어 있다. [Newman–Clauset, 원문 Methods](https://arxiv.org/html/1507.04001). 구조 group 수를 semantic K와 별개로 다루는 hierarchical/degree-corrected SBM과 model selection도 기존이다. [Peixoto, 원문 §II](https://arxiv.org/html/1310.4377). 이를 이번에 새로 발견한 모델 원리로 주장하지 않는다.

## 5. Homophily에 따라 nonedge를 켜고 끈다는 해석의 교정

단순 Bernoulli SBM에서

\[
J_E=\log(p_{in}/p_{out}),\qquad
J_0=\log\{(1-p_{in})/(1-p_{out})\}.
\]

Homo면 `J_E>0, J_0<0`, hetero면 반대다. Sparse일 때 `J_0≈−(p_in−p_out)`다. Edge homophily `P(same class|edge)`는 class opportunity/degree와 함께 변하며, 이 값 하나가 부재의 정보량·모형 적합성·삭제 이득을 결정하지 않는다. Physics의 강한 homophily와 nonedge 삭제 이득은 동시에 가능하다. 반대로 heterophily라고 nonedge가 무조건 불필요한 것도 아니다. True query labels로 homophily를 계산해 weight를 정하면 leakage다.

더구나 현재 context끼리 연결된 unordered edges는 Cora3/Cite0/PubMed1/CS2/Physics2개뿐이다. Root의 read-only `context_support.py`는 PubMed의1개는 다른 class, 나머지 있는 edge는 same class임을 확인한다. “Physics h=1이므로 parameter를 확실히 안다”는 추정은 두 edge에 근거한 과신이다. 수많은 nonedge도 독립 label evidence 수가 수많다는 뜻은 아니다.

Balanced context의 raw counts만으로 두 contrast가 식별되지 않을 수도 있다. Held node i에 대해 class c의 edge count가 E_ic, nonedge count가 NE_ic이고 `E_ic+NE_ic=m_c`면

\[
\beta_E E_{ic}+\beta_0 NE_{ic}
=(\beta_E-\beta_0)E_{ic}+\beta_0m_c.
\]

`m_c`가 같으면 마지막 항은 class softmax에서 상쇄된다. 따라서 두 free contrast를 따로 추정했다고 해도 실제로는 차이 하나만 식별했을 수 있다. Degree/opportunity를 명시하고 uncertainty/rank를 검사해야 한다.

## 6. 다음 수정 하나: context 예측으로 보정하고 불확실성을 평균하는 conditional contrast BP

**다음 권고는 unlabeled q로 K×K Gamma/pi를 재적합하는 feedback을 제거하고, degree/opportunity를 고정한 두 graph contrast의 conditional predictive posterior를 context에서 추론하는 것이다.** 현재 Stage0를 닫은 뒤 새 계획으로 검토할 미실행 제안이다. Eta/temperature를 query accuracy로 sweep하거나 h 하나를 nonedge gate로 바꾸는 제안은 아니다.

실행 가능한 working conditional score의 예는

\[
S_\beta(y;A)=S_{feature}(y)
+\beta_E\sum_{i<j}A_{ij}\mathbf1[y_i=y_j]
+\beta_0\sum_{i<j}(1-A_{ij})o_{ij}\mathbf1[y_i=y_j],
\]

여기서 `o_ij=1−exp(−d_i d_j/sum d)`는 graph-only opportunity의 한 고정된 정의다(빈 graph는0). 이는 exact degree-conditioned generative likelihood라고 부르지 않는다. Class unary/population과 opportunity를 자기 semantic q로 다시 적합하지 않는다. Positive/negative contrast를 같은 h의 결정적 함수로 묶지도 않는다. Prior 중심·scale·유한 적분 grid는 inherited model/synthetic calibration으로 **결과 전에** 고정한다.

각 context label을 한 번씩 숨기고 나머지만 clamp한 `q_beta^{-i}(y_i)`를 구해

\[
\tilde p(\beta\mid C,A)\propto
p_{prior}(\beta)\exp\{\sum_{i\in C}\log q_\beta^{-i}(y_i)\},\quad
p(y_q\mid\cdot)=\int q_\beta(y_q)\tilde p(\beta\mid C,A)d\beta
\]

로 예측을 평균한다. 이는 overlapping held-context predictions의 **composite conditional predictive calibration**이지 exact marginal Bayes가 아니다. Class-stratified context selection도 population의 random sample로 취급하지 않는다. `E_ic`, degree/opportunity-weighted negative counts, observed class-pair opportunities와 posterior 폭을 저장한다. 위 collinearity나 context edge 부족으로 식별이 약하면 posterior가 넓게 남아야 하며, 임의로 단일 β/성능 좋은 branch를 확정하면 안 된다. 직접 context-pair Beta/Gamma count 진단도 opportunity/선택과정을 구분한 보조 정보로 사용하고 `동일 h→동일 nonedge weight`로 환원하지 않는다.

이 수정은 **첫 갱신 전 잘못된 graph score와 이후 feedback을 함께 건드린다.** CiteSeer의 유용한 absence는 context predictive evidence가 지지할 때 보존하고, global semantic absence penalty가 해로우면 불확실성을 포함해 약해질 수 있다. 잘된다고 보장하지 않으며, 희소 context가 아무 것도 식별하지 못한다는 결과도 정당한 실패다. 여러 community가 같은 class를 공유하는 문제도 이 conditional score가 완전히 해결했다고 주장하지 않는다.

최소 다음 검증은 query truth 없이 실행되는 위 규칙 하나를 새 synthetic matched-homophily/different-opportunity 사례와 같은5개 development graph 모두에 적용하는 것이다. 고정 old_full/old_edge, 강한 restart/harmonic LP 및 degree-conditioned CPL을 그대로 보고한다. Unknown q만으로 재적합하는 현재 DC와 context-calibrated graph-only를 먼저 비교하고, 같은 frozen C_global의 observed−none은 별도로 확인한다. 결과를 본 graph별 선택은 하지 않는다.

이 역시 classical conditional inference/calibration이며, DC-SBM/BP/Gamma 원리나 두 contrast를 쓴 사실이 새 PFN 기여가 되지는 않는다. 추후 synthetic-only 학습이 이런 강한 비학습 통계 추론보다 독립적인 전이·feature 효용을 보인 뒤에만 learned-factor 기여를 판단할 수 있다.

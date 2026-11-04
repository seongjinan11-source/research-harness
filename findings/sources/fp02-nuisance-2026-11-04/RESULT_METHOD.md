# FP02 Stage0 결과: 수학·추론 관점의 방법 판정과 다음 한 가지 수정

2026-11-04. **현 Gamma–Poisson/SBM/DC 경로는 Stage0에서 중단하고, adjacency에 조건화한 continuous label field와 centroid evidence를 공동 추론하는 연산자 하나를 다음 cycle에서 검증할 것을 권한다.** 실패한 통계 추론을 더 큰 head, nonedge gate, 동일 bank의 prior/temperature sweep으로 구제할 근거는 없다. 이 권고는 기존 community inference 전체의 불가능성 판정이 아니다.

검토 자료는 [저장 확률 재채점 보고서](results/REPORT_KO.md), `summary.json`, Physics none의 세 신규 NPZ, `audit.json`, `checks.json`, `PLAN_v2.md`, 기존 transfer diagnosis와 FP01 결론이다. 새 실행·예측·학습·source/branch 변경은 하지 않았다. 이 reviewer는 FP02 `operators.py` 구현자이기도 하므로 구현의 독립적인 재검증은 별도 담당의 `checks.json`과 저장 결과 감사에 근거한다. 아래에서는 수치 검사의 통과와 모델의 성능 실패를 구분한다.

## 1. 결과 판정: 실제로 실패했으며, fixed arm부터 원래 구조보다 나쁘다

아래는 동일한 rep0/seed11, feature-none 개발 사례다. 수치는 모두 graph 동일 가중 또는 해당 한 graph의 저장 확률 재채점이다.

| 방법 | Physics accuracy % | Physics NLL | CiteSeer accuracy % | 5-graph BAcc % | 5-graph NLL |
|---|---:|---:|---:|---:|---:|
| poisson_fixed | 32.740 | 9.26854 | 44.784 | 51.961 | 4.07794 |
| sbm_update | 23.123 | 11.77810 | 37.935 | 49.246 | 5.79886 |
| dc_update | 20.290 | 7.79559 | 35.090 | 48.989 | 4.21414 |
| old_full | 48.551 | 5.09251 | 52.898 | 54.599 | 2.53411 |
| old_edge | 88.801 | .72202 | 41.728 | 57.432 | 1.62087 |
| restart LP | 87.220 | 1.48294 | 40.042 | 59.848 | 1.63741 |

DC는 사전 gate의 7개 조건을 모두 실패했다. Physics는 old_full보다 −28.261%p, old_edge보다 −68.511%p이며, CS도 old_full보다 −6.724%p다. CiteSeer의 old_full 대비 accuracy −17.808%p와 NLL +1.46019는 허용 손상 범위를 크게 넘는다. SBM도 별도로 모두 실패했으며 DC의 실패를 대체하지 않는다. Audit PASS는 입력·정답·예측·source의 일관성을 뜻하고 성능 gate PASS가 아니다.

중요한 반례는 `poisson_fixed`다. 업데이트가 전혀 없는 이 arm도 Physics에서 old_full보다 −15.811%p다. 따라서 이번 실패를 latent theta 갱신의 결함 하나로 환원할 수 없다. 새 score의 positive-edge contrast, prior uncertainty, all-pair absence field, fixed affinity, 연산/근사 구성 자체가 함께 바뀌었다. old_full→poisson_fixed를 nonedge 또는 degree correction 하나의 순수 인과 효과라고 해석하면 안 된다.

관측95−none의 5-graph 평균 DC 변화는 accuracy +.245%p, BAcc +.036%p, NLL −.00507이다. 이것은 심하게 틀린 graph-only 기준 주변의 작은 변화다. 신규 shuffle이 없으므로 feature alignment 성공을 뜻하지 않는다. FP01의 centroid가 주로 feature harm을 줄였다는 결론을 새 feature 정보 회복으로 바꿀 근거도 없다.

## 2. 실제 계산과 결과가 보여 주는 다섯 가지 한계

1. **부정 증거의 손해는 남았고 nuisance 학습이 해결책이 아니었다.** 기존 frozen deletion은 Physics 9/9 사례에서 약 +38–43%p accuracy 개선을 보였다. 이번 Gamma/Dirichlet 갱신은 이를 복구하지 못했다. 다만 이번 arm들은 nonedge만 조작한 실험이 아니므로 “수정된 nonedge만 원래보다 정확히 몇 pp 더 해로웠다”는 분해도 아직 없다. 직접 입증된 것은 기존 nonedge deletion 효과와 새 통계 모델 전체의 실패다.
2. **추론된 class mass는 true population의 신뢰할 복원값이 아니었다.** `dc_update/CoauthorPhysics_r0_none.npz`의 query1518개에서 실제 class2 비율은70.026% (1063/1518)인데, DC argmax 예측은3.426% (52/1518), 전체 unknown-node population posterior mean은 약1.41%다. 마지막 값은 query만의 비율과 분모가 다르다. SBM population도 약3.39%다. 고정된 균등 prior를 없앴다고 실제 큰 class를 찾지 못했다. Soft belief로 만든 충분통계는 잘못된 class identity와 분할을 feedback할 수 있다. 이는 “analytic update이므로 label population을 올바르게 알아낸다”는 해석을 반박하지만, 갱신식에 evidence를 중복 가산했다는 증거는 아니다.
3. **Poisson hub saturation은 한계이나 전체 실패의 단독 원인이 아니다.** Physics DC의 최대 class-conditional rate는5.5888, 최대 observed-edge q-weighted rate는1.3370이다. 그러나 rate>1인 observed edge 비율은 약.000556이며, 이것으로 전 query의 실패를 설명할 수 없다. 특히 fixed arm의 최대 rate는 .01834로 작아도 성능은 매우 나빴다. 이 최대값은 class-state 전체의 상계이지 대다수 pair의 posterior 평균이나 실제 edge 확률이 아니다. 결과를 보고 rate clip을 넣는 것은 새 모델이다.
4. **정상 구현과 정확한 posterior는 다르다.** 독립 tree/local oracle 오차는 약10⁻¹⁶이고, counts/exposure·fresh-prior 갱신·대칭·누출·gradient 검사를 통과했다. 반면 tiny N=5 예시의 sparse-vs-dense 차이는 .02068, sparse-vs-exact 차이는 .01272였다. 비이웃 MF, finite loopy BP, 독립 Gamma factorization, 3회 nuisance 업데이트는 여전히 근사다. 이 수치들은 실제 Physics의 오차 크기를 정량화하지 않는다. 현재 결과를 “새 버그”라고 부를 근거도, approximate Bayes가 충분히 정확해서 가정만 완전히 분리됐다고 부를 근거도 없다.
5. **Class=graph-generating block의 강한 가정을 계속 보존했다.** Degree propensity와 affinity를 자유롭게 해도 semantic class 하나에 단일 class-pair connectivity law를 둔다. 한 semantic class가 여러 지역 community/주제/계층에 걸치거나, sampling/exposure가 block별로 달라지는 경우를 충분히 표현하지 않는다. 이번 결과는 이 family에서 벗어나 볼 강한 개발 근거다. 그러나 한 개발 rep·초기화·finite inference만으로 모든 DC-SBM이나 generative graph model을 기각하지는 않는다.

현재 main-method의 실증 지지는 **1/5**다. 보존할 자산은 명세·audit·수치 검증과 부정 결과이며, 다음 연구의 구현 출발점으로서 가치는 있다. Source0/1 경계, 숨긴 값 차단, 전체 비교 보존은 다음 cycle에도 유지해야 한다.

## 3. 다음 한 가지 수정: conditional Gaussian label-field + 공동 centroid 메시지

### 3.1 무엇을 바꾸는가

**p(A|class partition)를 제거하고, 관측 graph A 위의 continuous class-logit field g를 조건부 prior로 둔다.** 기존 observed-cell/class-centroid factor는 유지한다. Adjacency는 field의 전달 경로와 regularization을 정의하며, 모든 A_ij=0을 독립적인 반대 증거로 곱하지 않는다. Absence가 connectivity·degree·walk 경로에 영향을 주는 것과 all-pair negative likelihood를 주는 것은 다르다.

대칭 binary A에 대해

\[
w_{ij}=\frac{A_{ij}}{\max(d_i,d_j,1)},\qquad
L_M=\operatorname{diag}(W\mathbf1)-W,
\quad Q_A=\epsilon I+\beta L_M,\quad\epsilon>0,\;\beta\ge0.
\]

Metropolis 형태의 W는 symmetric이고 row sum≤1이다. 따라서 L_M은 PSD, 상수 field가 null direction이고 spectrum≤2다. `P_M=I−L_M`은 self-transition을 포함한 대칭 row-stochastic walk다. 단순 symmetric normalized Laplacian의 degree-scaled constant field를 무심코 class logits의 기준으로 쓰는 문제도 피한다. 이 normalization은 degree-marginalized graph likelihood가 아니라 bounded conditional transport의 정의다.

K개의 continuous field와 이산 label을

\[
g_{:c}\mid A\sim\mathcal N(0,Q_A^{-1}),\qquad
z_i\mid g_i\sim\operatorname{Categorical}(\operatorname{softmax}(g_i)),
\]

\[
\mu_{cf}\sim w_f\text{ on the fixed finite grid},\qquad
x_{if}\mid z_i=c,\mu\sim\mathcal N(\mu_{cf},\sigma_f^2)
\quad\text{only if observed}_{if}=1
\]

로 정의한다. Context z는 clamp하고 query z는 관측하지 않는다. Proper Gaussian normalization은 Q의 determinant를 포함한다. Fixed A의 MAP field step에는 그 g-independent 항을 생략할 수 있지만, 이후 모델 evidence나 prior-hyperparameter Bayes를 계산한다면 생략할 수 없다.

이 조건부 모델에서 한 semantic class가 여러 graph region에 나타나는 것이 허용된다. K가 topology의 community 수라는 제약이 없다. Positive smoothness는 여전히 homophily 가정이므로 arbitrary heterophily까지 해결했다고 주장하지 않는다. 균등한 label marginal prior는 남지만 큰 class에 O(N²) absence penalty를 부과하는 mechanism은 없다. 모든 population bias가 사라졌다는 주장도 하지 않는다.

### 3.2 실행할 안정된 field 연산

Centroid의 class-gated factorization과 엄격한 opposite-message exclusion은 FP01처럼 쓴다. Graph-to-node categorical BP 대신 node unary를 g_i로 둔다. 이전 g와 feature 메시지에서 얻은 q_i(c)로, g block의 objective는

\[
J(g;q)=\tfrac12\sum_c g_{:c}^{\top}Q_Ag_{:c}
+\sum_i\{\operatorname{LSE}(g_i)-q_i^{\top}g_i\}.
\]

한 inner update를

\[
g^{t+1}=(I+\eta Q_A)^{-1}
\big[g^t+\eta\{q^t-\operatorname{softmax}(g^t)\}\big]
\]

로 정의한다. 이것은 Gaussian prior 부분을 implicit하게 처리한 proximal-gradient field step이다. Fixed q에서 likelihood gradient의 Lipschitz constant는≤1/2이므로 예를 들어 0<eta≤2의 보수적인 step을 사전 고정할 수 있다. Linear inverse는 SPD이며 operator norm≤1/(1+eta*epsilon)이다. Sparse matvec/CG로 K개 right-hand side를 푼다. Dense N×N inverse를 만들지 않는다. CG tolerance/고정 반복 수와 residual을 사전 명세하고, finite solver를 미분한다.

관측 cell의 centroid→node, node→centroid proposal과 g proposal은 이전 iterate에서 동시에 계산한다. 각각 필요한 damping은 결과 전에 고정한다. Context의 q는 정확히 one-hot이다. **이로부터 전체 g–centroid coupled unroll의 수렴·정확 Bayes·calibration은 따라오지 않는다.** Field block 자체의 boundedness와 전체 비선형 inference의 성공을 구분한다. q(g)를 point field로 대체하는 MAP/finite inference approximation임을 명시하고, exact Gaussian posterior predictive라고 부르지 않는다.

특히 feature가 없는 unknown node에서는 q_i=softmax(g_i)이므로 local residual `q_i−softmax(g_i)`가 정확히0이다. 관측하지 않은 query의 pseudo-label count로 population을 새로 가산하는 힘이 없다. Context와 informative feature cell만 field에 forcing을 넣고 graph solve가 이를 전달한다. 이것은 FP02의 자체 추론 class counts→population/affinity→다음 분할 강화와 구별되는 실제 계산 변화다.

경로는 다음과 같다.

\[
Y_C\to g_{source}\to q(z_{source})\to q(\mu_{cf})
\to q(z_{other\ source})\to g_{query}\to p(Y_q).
\]

Query의 자신의 x는 여전히 없다. Feature를 한 번 impute한 뒤 고정 head에 넣는 순차 pipeline도 아니다. Centroid와 field가 같은 finite unroll 안에서 feedback한다. 열당 source0/1이면 동일 centroid prior의 exchangeability로 cell-to-node scalar가 class-independent하여 그 column의 class evidence는 사라져야 한다. 이 LOO boundary를 유지하는 것이 mean-field 자기확신을 추가하는 것보다 낫다.

## 4. Homophily rate로 센스 있게 처리할 수 있는가

**동종 class끼리 연결되는 경향과, 연결되지 않았다는 사실의 예측 정보는 다르다.** Degree/exposure가 주어졌을 때의 graph density, class mass, observation mechanism에 따라 strong homophily와 weak/noninformative absence가 공존할 수 있다. Physics의 삭제 개선이 바로 “homophily면 nonedge가 항상 유용하다”는 단순 규칙의 반례다. Heterophily라고 nonedge 삭제가 항상 좋은 것도 아니다.

이번 Stage0 입력의 known-known unordered edge에 대한 root의 read-only 점검은 Cora3개(3개 same), CiteSeer0개, PubMed1개(diff), CS2개(2개 same), Physics2개(2개 same)였다. 이처럼 소수 표본에서는 0%,100% 같은 plug-in rate가 실제 graph의 이질성을 판정할 수 없다. 특히 CiteSeer는 직접 관측한 connected-label contrast 자체가 없다. “Context labels를 읽으면 homophily rate를 정확히 안다”는 가정을 새 operator에 넣으면 안 된다. 합성 prior를 통한 shrinkage와 coverage/uncertainty가 실제 추론의 일부여야 한다.

Context로 계산할 수 있는 입력은 true homophily oracle 하나가 아니라, degree/opportunity strata를 맞춘 두 conditional relation의 차이다. 예를 들어 Z_ij=1[y_i=y_j]에 대해 context-context connected pair와 degree-bin을 맞춘 unconnected pair에서 각각 shrunken p_E(s), p_0(s)를 추정하고

\[
\Delta_s=\operatorname{logit}p_E(s)-\operatorname{logit}p_0(s)
\]

와 uncertainty/유효 coverage를 보존한다. 같은 estimator가 작은 합성 graph와 큰 graph에서 얼마나 보수적으로 작동하는지를 synthetic heldout에서 배운다. Class-balanced context selection은 실제 class population을 대표하지 않으며, pair들은 node를 공유하므로 독립 Bernoulli 표본이 아니다. 단순 Beta의 pair 수를 그대로 effective sample size로 쓰면 과신한다. Degree matching만으로 selection bias나 unknown exposure가 자동 교정되는 것도 아니다.

이 contrast는 위 **동일 field operator의 regularization precision을 조건화하는 관측 요약**으로 사용할 수 있다. Mapping은 synthetic-only query NLL로 학습하고 uncertainty가 클 때 prior 쪽으로 수축시켜야 한다. 그러나 이것을 새 router나 edge/full winner selection으로 만들거나 `nonedge_weight=homophily`로 되돌리면 이번 수정의 이유가 사라진다. 첫 후보가 nonnegative Laplacian field인 이상 negative contrast가 나타나는 arbitrary heterophily는 범위 밖이다. 그 경우 smoothness를 약하게 하는 것이 가능한 동작이며, heterophilic compatibility를 복원했다는 주장은 금지한다. Scope 확대는 별도 operator 명세가 필요하다.

## 5. 합성 학습도 class-first SBM에서 실제로 바꿔야 한다

이 후보는 기존 checkpoint를 새 field에 끼우는 것으로 끝내면 안 된다. 다음 한 cycle은 **operator 교체와 graph-first synthetic prior의 실제 query-loss 학습**을 함께 사전 명세해야 한다.

1. 먼저 degree/skew/구조 community 수가 다양한 A를 생성한다. 구조 community 수는 semantic K와 별개다. 그 위에서 여러 scale의 smooth latent field를 만들고 nonlinear/random label cut으로 Y를 정한다. 동일 class가 여러 region에 나타나는 경우를 포함한다. 단순 Gaussian/probit teacher 하나와 동일한 Gaussian student의 matched-prior 성공만으로 transfer를 주장하지 않는다.
2. Attribute는 latent label/field와 coupling된 관측값과 nuisance 성분으로 만들고, projection·mask를 적용한다. C/O/query 분리와 feature 없는 query 조건을 모든 학습 task에 적용한다. Graph-only task와 informative/independent feature task를 같은 사전 비율로 포함한다. Label, node, column permutation을 보존한다.
3. 동일 parameter budget·task exposure·seed로 원래 SBM prior에서 학습한 **같은 conditional operator**와 새 graph-first prior에서 학습한 operator를 비교한다. 기존 old-edge/old-full, centroid controls, harmonic/restart LP, 강한 conditional Gaussian/LP+centroid 통계 추론, PCFI/FISF+public PFN을 같이 남긴다. 그 비교 없이 prior 변경과 operator 변경의 기여를 구분할 수 없다.

β/epsilon 및 feature-law hyperparameters는 이 finite joint operator의 synthetic training에서 배우는 parameter다. 실제 graph에서 neural optimizer update는0이어야 한다. 신규 real graph에서 선택하거나 동일 다섯 개발 graph의 temperature·β·solver 횟수로 구제하면 안 된다. 이미 본 real graph는 개발 신호이며 Stage0 gate가 실패한 현재 상태에서 원래 Stage1을 자동 실행할 근거는 없다. 바뀐 operator/prior/예산/최종 seed에 새 명세·검토·lock이 필요하다.

## 6. 기존 문헌과의 관계: 원리는 이미 존재한다

- Degree correction과 block likelihood는 [Karrer–Newman](https://arxiv.org/html/1008.3926v1)의 기존 원리이며, degree-conditioned pseudo-likelihood도 [Amini et al.](https://arxiv.org/html/1207.2340v3)에 있다. FP02의 실패는 이 문헌 전체를 반박하지 않는다. 이 논문들이 semantic class와 structural block의 동일성을 보장해 주는 것도 아니다.
- Conditional graph fields, Laplacian priors와 classification은 오래된 family다. [NeurIPS 2005 hyperparameter/kernel learning](https://proceedings.neurips.cc/paper_files/paper/2005/file/c429429bf1f2af051f2021dc92a8ebea-Paper.pdf)와 [Graph Gaussian Processes, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/1fc214004c9481e4c8073e85323bfd4b-Abstract.html)을 가까운 classical 비교로 봐야 한다. 위 Gaussian-softmax conditional model 자체를 새로운 확률 원리라고 부를 수 없다.
- Bounded/implicit learned graph diffusion도 [GRAND](https://proceedings.mlr.press/v139/chamberlain21a.html)에 있다. Missing feature의 Dirichlet-energy diffusion은 [Rossi et al. 2022 Feature Propagation](https://proceedings.mlr.press/v198/rossi22a.html)의 직접 prior다. 여기서 제안한 식은 이 문헌들의 정확한 알고리즘을 재현했다는 뜻이 아니라 위 conditional model에서 유도한 특정 hybrid inference다.
- Centroid/class joint inference와 unrolled BP는 FP01의 [prior-art scope](../factor_projection_cycle/PRIOR_ART_SCOPE.md)에 이미 정리된 CSBM AMP–BP/BPGCN과 겹친다. 새로운 graph-first synthetic PFN 자체도 [GraphPFN 공식 구현 설명](https://github.com/yandex-research/graphpfn)처럼 이미 선행한다. 검색에서 확인한 2026-10-28 [GSCM-PFN abstract](https://arxiv.org/abs/2610.38335)는 joint feature–edge–label SCM synthetic PFN을 주장한다. 이 마지막 논문은 abstract 수준 확인이므로 동등 구현/정확 우선권 판정은 하지 않는다.

따라서 남는 기여 후보는 **disjoint evidence에서 source feature가 query label에 추가 정보를 주는 공동 conditional inference를 synthetic-only로 학습하고, 강한 classical/LP/imputation PFN보다 얻는 실측 가치**다. Gaussian field, diffusion, graph-first prior, differentiable inference라는 이름만으로 novelty를 확보할 수 없다.

## 7. 다음 실행 전 최소 세 가지 확인

1. **Field block의 수학:** Q SPD·constant-null Laplacian·spectral bound, 작은 dense inverse와 sparse solve의 일치, residual, fixed-q objective 감소, class/node permutation과 context clamp를 확인한다. 전체 coupled algorithm은 별도 tiny discrete-grid/field quadrature 또는 고정-field oracle과 비교하여 approximation error를 남긴다.
2. **정보 경로:** source0/1 경계, hidden poison, source-row/column permutation, query-feature absence를 확인한다. Synthetic informative feature에서 observed>none 및 observed>shuffle가 모두 있어야 하고 independent feature에서는 calibration/harm을 검사한다. Feature를 끊어도 같은 성능이면 joint feature mechanism 성공으로 쓰지 않는다.
3. **반증 가능한 equal-budget 비교:** operator/prior 비교를 분리하고, none와95/99/99.5의 accuracy·BAcc·NLL을 함께 사전 고정한다. Physics만 회복한 뒤 CiteSeer 손상을 허용하는 식으로 endpoint를 바꾸지 않는다. Stable feature utility가 없어도 classical graph-only를 이긴 성과로 바꿔 주장하지 않는다.

**최종 권고:** 현재 Poisson/DC cycle의 실패를 그대로 보존한다. 다음에는 graph-generating class partition의 nuisance를 더 맞추는 대신, bounded conditional continuous field와 기존 observed-cell centroid를 공동으로 업데이트하는 한 연산자를 실제 synthetic query loss로 학습한다. 이는 실행 가능한 연구 가설이며, 아직 성능·novelty·calibration이 입증된 해결책은 아니다.

# 누적 실패를 바탕으로 한 추론 연산자 수정안

독립 method/literature synthesis · 2026-11-04. Transfer diagnostics, FP01 `DECISION_KO.md`, research-harness의 CSBMInference·factor-projection·LLBP·missingness/decoder 실패 기록을 읽고 primary paper를 확인했다. 코드·branch·역사적 결과를 변경하거나 학습을 실행하지 않았다. 아래 방법은 **구체적인 미실행 제안**이며 이미 병목을 해결한 결과가 아니다.

## 결론: graph score를 먼저 교체한다

**우선안은 class population·node degree propensity·class-pair rate를 함께 갱신하는 shrinkage Poisson DC-SBM + cavity BP다.** 기존의 `density × same/different × 3-role` potential을 유지한 채 nonedge gate만 붙이거나 feature head를 더 키우는 방향은 권하지 않는다. 대신 positive-edge 강도와 negative exposure를 **같은 rate의 numerator/denominator**로 추정한다. Node 수에 따른 class-size 벌점을 degree-weighted exposure로 바꾸고, balanced context 빈도를 population prior로 오인하지 않는다.

먼저 강한 고전 추론기로 이 계산을 구현·비교하고, 그 위에서 학습된 prior/shrinkage가 추가 가치를 주는지를 묻는다. DC-SBM/BP 자체를 새 방법으로 명명하지 않는다. 이것은 새 head 제안이 아니라, 실패한 graph likelihood의 통계량과 갱신 방식을 바꾸는 수정이다.

## 1. 무엇이 확인됐고 무엇은 아직 모르는가

**확인된 것**

- Frozen no-observed Physics에서 nonedge coupling 삭제는 accuracy 46.30→86.14%, NLL 5.8485→.9253으로 크게 개선됐다. 동일 개입의 CS는 61.74→72.67%였지만 balanced accuracy는 거의 개선되지 않았다. 따라서 전체 accuracy 증가를 minority 복원으로 대체할 수 없다. [진단 원문](C:/Users/seongjin/Desktop/code/gts_pfn/factor_transfer_diagnostics_20261104/REPORT_KO.md)
- Nonedge를 버리는 연산은 CiteSeer에 불리하다. FP01 C의 full/edge 평균 accuracy는 45.827/36.348%다. 기존 known-family 기록에도 observed nonedge를 버리면 손해를 보는 경우가 있다. **Nonedge는 원래 쓸모없다는 결론은 배제한다.**
- Physics의 model positive-edge coupling 평균 1.412가 label-aware 사후 pooled contrast 3.077보다 약한 반면 nonedge contrast는 −.01039 대 −.00965로 가까웠다. “negative 항의 크기만 과대”라는 설명은 불충분하다. Positive rate와 absence penalty의 상대적인 추정이 핵심이다. 이 label-aware 수치는 모델 입력으로 사용하면 안 된다.
- Physics에 피처가 없어도 문제가 남고, 반복 수를 늘려도 해당 사례는 복구되지 않았다. FP01 C는 대부분 actual feature gain보다 기존 feature 손상을 줄였다. 투영된 feature의 mean law만 넓히는 것으로 graph 병목은 해결되지 않았다. [FP01 결정](C:/Users/seongjin/Desktop/code/gts_pfn/factor_projection_cycle/DECISION_KO.md)

**미확인인 것**

Class imbalance, degree heterogeneity, 약한 positive-edge score, semantic class와 structural community의 불일치, finite BP 및 관측 과정의 상대 기여는 분리되지 않았다. 특히 benchmark의 모든 내부 adjacency 0을 “측정되지 않은 관계”라고 바꾸어 부를 근거는 없다. DC-SBM을 쓰면 CiteSeer와 Physics를 동시에 고친다는 결과도 아직 없다.

## 2. 가까운 방법을 정확히 구분한다

| 방법 | 실제로 바꾸는 계산 | 이번 병목에 대한 의미 |
|---|---|---|
| **Degree-conditioned conditional pseudo-likelihood (CPL)** | Degree를 조건으로 둔 block-neighbor counts의 multinomial mixture를 적합한다. Degree nuisance를 조건화해 제거한다. [Amini–Chen–Bickel–Levina, AoS 2013, §2.2 식 (5)](https://arxiv.org/html/1207.2340v3) | 빠르고 강한 고전 대조다. Degree 자체의 class 정보도 버릴 수 있으며 joint undirected posterior의 exact BP는 아니다. DC-SBM과 같은 방법이라고 부르면 안 된다. |
| **DC-SBM** | Edge rate에 node propensity를 넣고 class-pair 연결률을 degree exposure로 정규화한다. [Karrer–Newman, PRE 2011, §III 식 (14)–(18)](https://arxiv.org/html/1008.3926) | 큰 class가 단지 node 수가 크다는 이유로 과도하게 분할되는 것을 완화할 구조를 제공한다. Degree correction, general affinity 및 class population은 별개 항목이며 함께 명시해야 한다. |
| **DC-SBM BP/EM** | Nonedge도 likelihood에 포함하고, sparse 조건에서 nonneighbor messages를 공통 field로 압축한다. 삭제가 아니다. [Yan et al., Biometrika, §3 식 (8)–(11)](https://arxiv.org/html/1207.3994v2) | 이번 우선안의 가장 직접적인 계산 기반이다. 원문은 degree-corrected 모델이 무조건 더 좋은 것도 아니며 sparse model selection에 단순 χ² 규칙이 맞지 않을 수 있음을 지적한다. |
| **JSBM/BPGCN** | Joint item/feature BP와 유한 BP의 label-loss 학습이다. BPGCN 원문은 학습 라벨이 group sizes를 조절한다는 이유로 BP의 adaptive nonedge field를 생략할 수 있다고 설명한다. [§V](https://arxiv.org/html/1911.00197v2) | Target label loss로 재학습하는 설정의 근거를 frozen zero-update predictor에 그대로 옮길 수 없다. 특히 class-balanced context 빈도는 실제 population skew를 알려주지 않는다. Edge-only BPGCN이라는 이름만 붙여도 해결되지 않는다. |
| **CSBM AMP–BP** | Gaussian latent centroids와 graph class를 공동 추론하며 Appendix D는 불균형 multiclass와 일반 affinity를 허용한다. [Duranthon–Zdeborová, TMLR 2024](https://arxiv.org/html/2306.07948v2) | Class prior/affinity 갱신 및 graph–feature 결합의 직접 prior다. 그러나 기본 graph model이 node별 arbitrary degree correction을 자동 제공하는 것은 아니다. 고차원 Gaussian AMP를 현재 RP/clipped/disjoint 입력에 그대로 이식해 optimality를 주장할 수 없다. |

보강 prior: [Newman–Clauset, *Structure and inference in annotated networks*, 2016](https://www.nature.com/articles/ncomms11863)는 degree-corrected graph와 metadata-dependent prior를 BP/EM으로 결합한다. 따라서 “degree correction + side evidence” 자체도 기존이다. Missing metadata를 별도 값으로 취급하는 원문 경로를 현재 numeric missing-cell likelihood의 동일 정보 대조라고 부르지는 않는다.

## 3. 우선 후보: population/exposure를 갱신하는 shrinkage DC-SBM-BP

### 3.1 모델과 실제 수정 대상

Undirected graph, classes `z_i∈{1,…,K}`, context C, unknown nodes U를 둔다. Positive class-pair matrix `B=Bᵀ`는 diagonal/off-diagonal 두 값이나 전역 homophily 역할에 제한하지 않는다.

\[
A_{ij}\mid z,\theta,B\sim\operatorname{Poisson}
\left(e_{ij}\theta_i\theta_j B_{z_i z_j}/N\right),\quad i<j.
\]

`theta_i`는 degree propensity, `e_ij`는 **주어진 graph exposure**다. 현재 완전 관측 benchmark 내부 pair는 `e_ij=1`이다. 알려진 pair mask가 있으면 관측 안 된 pair는 likelihood에서 생략한다. Feature 관측 mask `M_if`를 `e_ij`로 사용하거나, A만 보고 unknown exposure를 발명하지 않는다. Unknown exposure와 작은 B/degree를 관측 A만으로 자동 분리할 수 있다는 주장은 하지 않는다.

Prior는

\[
\theta_i\sim\Gamma(a_\theta,a_\theta),\quad
B_{cd}\sim\Gamma(a_B,b_B),\quad
\pi\sim\operatorname{Dirichlet}(\alpha_0/K,\ldots,\alpha_0/K).
\]

Gamma는 shape/rate 표기다. `a_theta`는 degree variation의 shrinkage다. 큰 값이면 theta가 1에 가까워져 일반 SBM 쪽으로 가고, 작은 값은 넓은 degree variation을 허용한다. Citation에 유용할 수 있는 degree/absence 정보를 강제로 모두 버리는 CPL과 차이가 있다. 이 prior와 아래 variational 구성은 **본 검토의 제안**이며 특정 원 논문의 그대로인 재현식이라고 주장하지 않는다.

Class-balanced로 선택한 context는 고정 boundary label로 clamp한다. `pi`의 통계는 U의 latent population에서 추정한다. Context의 class당 4개를 population class 비율 1/K의 증거로 넣지 않는다. 엄밀한 selection posterior를 이미 해결한 것으로 부르기보다, **context를 고정한 conditional population model**로 명시한다.

Class-dependent graph score의 핵심은

\[
\ell_{ij}(c,d)=A_{ij}\,\mathbb E\log B_{cd}
-{e_{ij}\over N}\,\mathbb E\theta_i\mathbb E\theta_j\mathbb E B_{cd}.
\]

기존의 임의 positive/negative 두 gain을 따로 gate하지 않는다. Positive-edge와 absence는 같은 B에서 나온다. `d_i log theta_i` 같은 class-independent 항은 class BP에서 상쇄되지만 theta posterior 갱신에는 필요하다.

### 3.2 학습할 것과 추론할 것

먼저 `q(z) ∏q(theta_i) ∏q(B_cd) q(pi)` 형태의 고전 variational/BP 반복을 구현한다. `q(z)`는 context를 clamp한 cavity BP로 근사하고, 나머지는 다음 통계로 갱신한다. BP가 정확한 variational optimizer가 아니므로 모든 outer iteration의 ELBO 단조 증가나 수렴을 보장한다고 쓰지 않는다.

Unordered class pair c≤d의 membership indicator를 H_ij^cd라 하자. c=d이면 두 class가 c인지, c<d이면 두 class의 집합이 {c,d}인지 표시한다. `t_ij^cd=E H_ij^cd`는 가능하면 pair belief에서 구한다.

\[
n_{cd}=\sum_{i<j} A_{ij}t_{ij}^{cd},\qquad
E_{cd}={1\over N}\sum_{i<j}e_{ij}\,\bar\theta_i\bar\theta_j t_{ij}^{cd},
\]
\[
q(B_{cd})=\Gamma(a_B+n_{cd},\ b_B+E_{cd}).
\]

이 **edge count / degree-weighted exposure** 갱신이 핵심 repair다. Degree volume이 다른 두 class를 node count만으로 같은 밀도 family에 억지로 맞추지 않는다. Gamma uncertainty/shrinkage는 희귀 class의 거의 없는 edge count를 곧바로 확률 0/과신으로 바꾸는 것도 줄인다.

\[
q(\theta_i)=\Gamma\!\left(a_\theta+d_i,\ a_\theta+
{1\over N}\sum_{j\ne i}e_{ij}\bar\theta_j
\sum_{c,d}b_{ij}(c,d)\bar B_{cd}\right),
\]
\[
q(\pi)=\operatorname{Dirichlet}
\left(\alpha_0/K+\sum_{i\in U}q_i(1),\ldots,
\alpha_0/K+\sum_{i\in U}q_i(K)\right).
\]

여기서 degree는 관측 positive count, `b_ij`는 class pair belief다. Nonedge pair를 node-belief product로 대체하면 그것도 명시적인 근사다. Soft node count, edge pair count, degree exposure를 한 번 갱신하는 것이며 test truth는 입력하지 않는다.

**첫 비교에서는 neural training을 추가하지 않는다.** 고정된 고전 hyperparameters와 사전 지정 초기화/outer budget으로 repair 자체를 시험한다. 이후 학습을 붙일 대상은 dense pair potential이나 새 posterior head가 아니라 `a_theta, a_B, b_B, alpha0` 같은 소수 **hyperprior/shrinkage**다. Synthetic predictive CE로 학습한다면 degree-skew/exposure를 독립적으로 바꾼 prior와 동일 정보의 classical DC-SBM 대조가 필요하다. Target에서는 neural weights를 고정하지만 nuisance posterior는 갱신하므로, 넓은 의미의 “target parameter update가 전혀 없다” 대신 “target neural-optimizer update가 없고 transductive statistical inference를 한다”고 구분한다.

### 3.3 Sparse 연산은 nonedge 삭제가 아니다

e=1이면 `V_d=Σ_j bar(theta_j) q_j(d)`를 두고 부정 증거의 leading field는

\[
H_i(c)={\bar\theta_i\over N}\sum_d\bar B_{cd} V_d
\]

형태다. Cavity에서는 자기/상대 항을 적절히 제외하고, edge messages에는 `exp(E log B_cd)`를 사용한다. Sparse/nonneighbor approximation을 쓰면 대략 `O(E K² + N K²)`의 message/field 계산을 목표로 할 수 있다. 이는 원래 all-pair potential을 버리는 것이 아니라 많은 약한 absence factors를 population field로 모으는 것이다. Tiny graph의 dense pair score와 독립 enumeration부터 대조해야 하며 이 sparse 식을 유한 graph의 exact equality로 쓰면 안 된다.

실제 adjacency는 binary이고 위 모형은 Poisson working model이다. Hub pair의 predicted rate가 커지면 simple-graph saturation 오차가 남으므로 최대/상위 rate와 그 pair 수를 보고한다. 이 문제를 몰래 clip하거나 “Poisson이라 항상 맞다”고 넘기지 않는다.

### 3.4 왜 이번 병목을 건드리는가

- **Physics class skew:** uniform node prior와 fixed same-class absence penalty를 함께 교체한다. pi와 B를 class population/degree exposure에 맞춰 갱신하므로 큰 class가 커졌다는 이유만으로 같은 고정 벌점을 누적시키는 계산을 바꾼다.
- **Degree heterogeneity:** theta가 node별 연결 기회를 설명한다. 높은 degree를 class evidence와 그대로 혼동하거나 두 class의 exposure를 동일 node 수로 비교하지 않는다.
- **CiteSeer의 negative evidence:** nonedge를 0으로 gate하지 않고 현재 posterior population에 맞는 field로 보존한다. Degree shrinkage가 강한 한계는 일반 SBM에 가깝다. 이것이 CiteSeer 성능 보존을 보장하지는 않지만, 보존할 수 있는 모델 요소가 있다.
- **약한 positive contrast:** context/현재 beliefs로 class-pair edge counts를 재추정한다. 기존 graph-only encoder의 고정 signal 약 .38을 그대로 들고 가지 않는다.
- **Feature path:** 구조가 안정된 뒤 기존 C의 관측-cell/centroid messages를 같은 class BP에 연결한다. Query의 가짜 feature edge를 만들지 않는다. C head를 다시 확장하지 않아도 graph path를 통해 관측 evidence가 전달되는 통계적 조건이 달라진다. Utility 회복은 반드시 observed−none으로 새로 검증한다.

## 4. 두 번째 후보: conditional pseudo-likelihood — 우선 구현의 강한 대조

현재 partition e에 대한 `b_ik=Σ_j A_ij 1[e_j=k]`와 d_i를 사용해

\[
q_i(c)\propto\pi_c\prod_k P_{ck}^{b_{ik}},\quad
P_{ck}={\sum_i q_i(c)b_{ik}\over\sum_i q_i(c)d_i}
\]

를 반복하는 Amini 계열 CPL을 context-clamped baseline으로 둔다. Class population pi도 추정하고, label-count imbalance와 degree normalization을 분리한다. 빠르고 density mismatch에 강할 가능성이 있어 **DC-SBM만 구현하고 약한 old Factor에만 비교하는 일을 막는 기준선**이다.

다만 이는 neighbor-count composite likelihood다. Soft neighbor labels를 넣어 반복했다는 이유로 exact BP 또는 calibrated graph posterior라고 부르면 안 된다. Degree의 class 신호와 absence information을 조건화해 잃을 수 있어, Physics에 유리하고 CiteSeer에 불리한 현상을 함께 다룰 우선안으로는 위 shrinkage DC-SBM을 선택한다. CPL에 graph별 gate를 추가하는 세 번째 후보는 제안하지 않는다.

## 5. 가장 강한 classical comparison과 최소 실행 순서

1. **비교 하한:** Context-clamped published DC-SBM BP/EM, 위 CPL, 일반 asymmetric-population SBM BP/EM(theta=1), 기존 full/edge Factor·graph-only, 두 LP, 같은 입력의 PCFI/FISF-PFN. DC-SBM+metadata BP/EM도 정보 경로가 맞는 형태로 대조한다. “새 shrinkage가 old uniform Potts를 이겼다”만으로 충분하지 않다. 모든 방법의 초기화/multistart·outer budget·실패 처리 규칙은 먼저 고정하고 query truth로 run을 고르지 않는다.
2. **새 독립 synthetic bank:** class skew와 degree variation을 따로 교차시키고, 실제 정보가 있는 full observation과 주어진 pair-mask/exposure 조건을 구분한다. K=15와 기존 K 범위, minority/majority, homophily와 class-specific compatibility를 포함한다. Tiny exact score·Gamma moments·pair counts의 대칭/중복계수·mask poisoning을 검증한 뒤 큰 graph로 간다. 단순 generator만 넓히고 old operator를 유지하는 재시작과 다르다.
3. **동일 operator로 5개 real 개발 graph 전부:** Physics만 좋아지고 CiteSeer를 크게 해치면 동시 repair 가설은 미확인이다. Graph마다 full/edge를 고르지 않는다. Accuracy/NLL/Brier·balanced accuracy·예측 class mass와 degree strata를 함께 보고, structure-only repair와 observed−none/shuffle을 분리한다. 이 개발 graph의 결과로 최종 설정을 골랐다면 새 synthetic final 및 별도 real holdout에 대한 주장은 새 검증이 필요하다.

구체적인 reopening 조건은 **고정 graph prior를 실제 population·degree exposure에 맞춰 추론하는 연산 변화**다. 따라서 FP01의 닫힌 범위(같은 graph core 아래 새 centroid head) 밖에 있다. 이것은 새 이름으로 같은 head를 재학습하는 반복이 아니다.

## 6. Novelty와 불확실성의 범위

DC-SBM, Gamma conjugacy, BP/EM, CPL, 불균형 multiclass CSBM은 기존이다. 현재 우선안은 먼저 **이 프로젝트의 잘못 제한된 graph likelihood를 고치는 실행 가능한 조합**이다. 고전 baseline이 병목을 풀면 그 사실을 성과로 받아들이고 새로운 neural method의 공으로 돌리지 않는다. 이후 learned hyperprior가 같은 정보/예산의 강한 statistical inference보다 분포 전이·feature utility·비용에서 독립적인 이득을 내야 방법 기여 후보가 된다.

위 식은 구현 가능한 시작점을 제공하지만 finite BP/VB의 calibration, semantic labels와 구조 group의 일치, local optimum, 희귀 class에 대한 B 추정, Poisson saturation 및 unknown exposure 식별은 해결되지 않았다. 이 한계 때문에 시작하지 말자는 것이 아니라, **어떤 가정을 실제로 교체하고 어떤 실패를 같은 operator로 고칠 것인지**를 고정하고 실행하자는 권고다.

## 7. FP02 `PLAN_v1.md`의 명시적 Stage0 검토

검토 대상: [FP02 계획](C:/Users/seongjin/Desktop/code/gts_pfn/factor_nuisance_cycle/PLAN_v1.md). **조건부 PASS: 아래 수식·일정 정의를 구현 전에 닫고 numerical check를 통과하면 Stage0를 실행하는 데 찬성한다.** 추가 문헌 검토나 Stage1 학습을 기다릴 이유는 없다. 결과가 좋을지는 아직 판정하지 않는다.

### 7.1 수용하는 계산과 주장 범위

계획의 `e_ij=d_i d_j/D`, `D=2m`은 위 우선안의 latent theta posterior를 생략한 **고정 configuration-offset Poisson score**다. Degree를 정확히 조건화한 CPL도, theta uncertainty를 적분한 DC-SBM posterior도 아니다. 그래도 명시한 working score 안에서 Gamma affinity와 population을 추론하는 연산은 일관되게 정의할 수 있다. Degree/skew/exposure를 함께 건드리는 첫 pilot으로 타당하며, 이 단순화 때문에 Stage0를 막을 필요는 없다. Degree correction이 degree 자체의 semantic class 정보를 제거할 수 있다는 위험은 결과 해석에 남긴다.

`A L-e R`에서 `L=E log W`와 `R=E W`를 분리한 점, observed edge의 음수 exposure를 보존하고 nonneighbor만 평균장으로 다루는 점, population에는 U의 **node count**를 쓰되 exposure에는 context를 포함한 **degree volume**을 쓰는 점은 수용한다. Gamma는 shape/rate이며 이는 BP/mean-field hybrid다. Exact joint Bayes나 일반적인 수렴 보장은 없다. `e`가 A로 추정되므로 이 점수로 unknown observation process까지 식별했다는 해석도 불가하다.

### 7.2 구현 전에 닫아야 할 세 항목

**1. Gamma 갱신은 원래 prior에서 다시 계산하고, unordered pair 계수를 명시한다.** 초기 mean을 `mu_ab`라 할 때 `s0_ab=kappa`, `r0_ab=kappa/mu_ab`다. 매 outer의 posterior는 반드시 `s_ab=s0_ab+n_ab`, `r_ab=r0_ab+E_ab`로 재계산한다. 이전 posterior에 같은 graph의 counts를 계속 더하면 반복할수록 관측량을 복제하는 오류다.

다음 식은 계획의 pair correction을 구현하는 한 가지 명확한 기준이다. `q_i(a)`는 node belief, `b_ij(a,b)`는 **observed-edge full Poisson factor**를 포함한 normalized pair belief다. Unordered class pair의 `t_ij^aa=b_ij(a,a)`, `t_ij^ab=b_ij(a,b)+b_ij(b,a)`(a<b)를 둔다. Product 버전 `tprod`는 b 대신 `q_i q_j`를 넣는다. `v_a=sum_i d_i q_i(a)`일 때

\[
n_{ab}=\sum_{i<j:A_{ij}=1}t_{ij}^{ab},\qquad
E_{ab}=E^{prod}_{ab}+\sum_{i<j:A_{ij}=1}e_{ij}(t_{ij}^{ab}-tprod_{ij}^{ab}),
\]
\[
E^{prod}_{aa}={v_a^2-\sum_i d_i^2 q_i(a)^2\over 2D},\qquad
E^{prod}_{ab}={v_a v_b-\sum_i d_i^2q_i(a)q_i(b)\over D}\quad(a<b).
\]

전체 exposure의 baseline은 **모든 서로 다른 node pair**다. 위 edge correction을 더한 뒤 observed edge의 exposure를 한 번 더 별도 합산하면 중복이다. Context의 one-hot beliefs도 v와 self subtraction에 포함한다. 독립 작은 graph에서 직접 `i<j` 합과 비교하면 대각 1/2·off-diagonal·edge correction 오류를 잡을 수 있다.

Observed-edge factor가 `exp(L_ab-e_ij R_ab)`이면 nonneighbor field는

\[
F_i(a)=-{d_i\over D}\sum_b R_{ab}
\left[v_b-d_iq_i(b)-\sum_{j\in N(i)}d_jq_j(b)\right].
\]

이어야 한다. 이 convention에서는 edge 음수항을 field에 다시 넣지 않는다. 대안적인 동치 분해를 쓰더라도 dense working score와 대조해 동일한 포함/제외를 확인한다. Nonedge product approximation의 오차를 exact BP 오류와 구분한다.

**2. 반복 일정과 최종 prediction 시점을 숫자로 잠근다.** `4 outer × 6 inner = 총24 BP sweep`인지 총96인지 확정하고, outer 사이 cavity message를 warm-start하는지, 어느 node/pair beliefs로 W와 pi를 갱신하는지 명시한다. 마지막 sweep 뒤 W/pi만 갱신하고 그 값을 아직 반영하지 않은 old beliefs를 output하는 혼선을 피한다. Feature/cavity/population도 동일한 iteration 상태를 참조하도록 정한다. 어느 선택이든 유한 연산자로 고정할 수 있지만 평가 전 하나로 정해야 한다.

**3. Population prior와 D=0 규칙을 정한다.** `alpha=1`이 총농도인지 class당 농도인지, pi에 posterior mean을 넣는지 `E log pi`를 unary로 넣는지 확정한다. 예컨대 총농도1의 posterior mean이면 `pi_a=(1/K+sum_{i∈U}q_i(a))/(1+|U|)`다. Context selection count는 넣지 않는다. D=0에서는 e=0, n=E=0으로 정해 Gamma가 원래 prior에 머물고 division by zero가 없도록 한다. Missing isolated query는 연결 정보 없이 population prior를 따라야 하며 query truth로 보정하지 않는다.

이 세 항목은 새 가설이나 튜닝 제안이 아니라, 현재 계획을 유일한 실행 연산자로 만드는 정의다. Source0/1 exchangeability 검사는 고정 W에서만 끝내지 말고 **W/pi outer update까지 포함한 전체 operator**에 적용한다.

### 7.3 Stage0와 Stage1의 분리

Stage0의 same-C_global old-full/edge, DC-fixed/uniform, DC-update 비교는 적절하다. DC-update가 W와 pi를 함께 바꾸므로 그 한 비교만으로 개선을 affinity와 population 중 하나의 인과 효과로 분리할 수는 없다. 이 제한 때문에 첫 pilot에 모든 ablation을 추가할 필요는 없다. Full/edge를 graph마다 골라 승자를 만드는 대신 하나의 고정 update가 Physics/CS와 CiteSeer를 함께 어떻게 바꾸는지 보고한다. BAcc/NLL과 minority mass를 함께 봐야 기존 CS의 accuracy-only 회복을 반복하지 않는다.

계획은 feature none/observed/shuffle을 약속하지만 Stage0의 명시 row는 none/95뿐이다. Lock 때 **shuffle95의 5개 개발 row를 추가하거나 shuffle 비교를 Stage1로 명시적으로 미룬다.** 관측치가 없는 query로의 경로는 source→centroid→source class→graph→missing query이며 직접적인 centroid→missing-cell 증거를 생성하면 안 된다. Observed−none이 없으면 graph score 개선을 feature 추론 성공으로 부르지 않는다.

Hub pair의 `e_ij R_ab` 분포/최댓값은 기록한다. 값이 크면 binary graph를 Poisson working score로 근사하는 포화 오차가 남지만, 결과를 보고 clip/새 gate를 고르는 근거로 사용하지 않는다. 원래 강한 LP/공개 PFN 비교와 C_global 고정은 유지한다.

Stage1의 여섯 same-budget fit은 **Stage0 뒤 별도 lock**이 맞다. 원래 prior와 degree/skew prior가 서로 다른 graph를 생성한다면 “동일 graph”를 문자 그대로 보장할 수 없으므로, Stage1 명세에서 공유 seed/template와 N/K·mask/context·노출 순서 중 무엇을 pairing하는지 구체화한다. 이는 Stage0 실행 blocker가 아니다. Stage0는 고전적 통계 추론 수정의 타당성을 시험하며, 새 synthetic-pretrained factor의 기여 판정은 Stage1 결과 이후에만 가능하다.

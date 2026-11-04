# Independent inference synthesis — one graph repair candidate

대상: transfer diagnostics 보고서, 원래 categorical Factor의 `model.py`, FP01의 `common.py`·`operators.py`·`DECISION_KO.md`. 요청에 적힌 `DECISION.md`는 없어 같은 결론 문서인 `DECISION_KO.md`를 읽었다. 코드 수정·실험·branch 변경·추가 agent는 수행하지 않았다. 아래는 아직 구현되지 않은 후보와 구현 전 검증 조건이다.

## 1. 제안: degree-offset Gamma–Poisson graph inference

**기존 graph-only 역할 가중치와 pair별 same/different Bernoulli potential을, 관측 degree를 exposure offset으로 분리하고 block affinity 및 unlabeled class population을 analytic하게 갱신하는 Gamma–Poisson inference로 교체할 것을 제안한다.** FP01 C의 feature factors는 우선 그대로 둔다. 새 gate/head를 붙이는 변경이 아니다. Graph score가 큰 class에 가하는 증거 자체를 바꾸는 변경이다.

근거의 강도는 다음과 같다.

- Physics의 no-observed nonedge deletion은 accuracy 46.30→86.14%, NLL 5.8485→0.9253으로 개선됐고 9/9사례에서 accuracy가 개선됐다. 이는 graph computation을 고칠 직접 근거다.
- 같은 삭제는 CiteSeer를 손상시킨다. 따라서 모든 absence evidence를 없애는 것을 답으로 고정할 근거는 없다.
- RP 노출 학습 B−A는 FP01 primary 약 −0.017%p였고 centroid C의 실제 관측 피처 추가 가치는 대체로 작았다. Feature head만 더 바꾸기보다 구조적 posterior가 source label evidence를 올바르게 전달하는지 고칠 우선순위가 높다.
- 다만 degree variation·class imbalance·edge 대비 nonedge 강도 중 유일한 원인을 판정한 결과는 아니다. 이 후보는 그 결합을 다루는 가설이지, 이미 원인이 입증됐다는 선언이 아니다.

### 기존 알고리즘과의 관계

Degree correction은 [Karrer–Newman의 DC-SBM](https://arxiv.org/html/1008.3926v1)의 기존 원리다. Poisson DC-SBM의 BP와 sparse external-field 근사도 [Yan et al.의 model-selection/BP 연구](https://arxiv.org/html/1207.3994v2)에 있다. Degree에 조건화하는 multinomial pseudo-likelihood 자체도 [Amini et al. 2013 §2.2](https://arxiv.org/html/1207.2340v3)에 있다. **DC-SBM·Gamma conjugacy·BP를 새 알고리즘이라고 포장할 수 없다.** 기여 후보는 이 알려진 구조 수정을 disjoint-evidence predictor에 올바르게 결합하고, 기존 실패를 실제로 줄이는지 강한 classical DC comparator와 분리해 보이는 데 있다.

아래 수식은 이 후보의 conditional offset model에서 직접 유도한 정의다. Degree-conditioned CPL을 동시에 또 하나의 새 방법으로 제안하는 문서는 아니다.

## 2. 어떤 항을 바꾸는가

원래 homophilic Potts score를 공통 strength로 단순화하면

\[
S=(J_e-J_0)E_{\rm within}+J_0\sum_a\binom{n_a}{2},\qquad J_0<0.
\]

Edge evidence가 약한 채 이 두 번째 항을 적용하면 큰 class를 과도하게 나눌 수 있다. 실제 network의 pair strengths는 상수는 아니지만 같은 부정 증거의 합산 경로는 코드에 존재한다. 현재 encoder는 graph/context의 class별 충분통계로 strength를 갱신하지 않는다. Physics의 경험적 positive edge log ratio가 모델보다 훨씬 컸다는 진단도 보존해야 한다. Nonedge strength 하나가 과대였다고 단정하지 않는다.

새 모델은 observed degree \(d_i\), edge 수 \(M=\frac12\sum_i d_i\)로

\[
w_i=d_i/\sqrt{2M},\qquad e_{ij}=w_iw_j=d_id_j/(2M),\quad i<j
\]

를 고정한다. \(M=0\)이면 graph factor를 모두 inactive로 정의한다. Degree 0 노드의 \(w_i=0\)도 정당하며, 존재하지 않는 edge의 `0*log(0)`을 계산하지 않는다.

Class-pair affinity \(\Omega_{ab}=\Omega_{ba}>0\)를 latent로 두고

\[
A_{ij}\mid s,\Omega,e\sim\operatorname{Poisson}(e_{ij}\Omega_{s_i s_j}).
\]

Drop 가능한 label-independent 항과 남겨야 할 항을 분리하면

\[
\log\psi^G_{ij}(a,b)=A_{ij}\log\Omega_{ab}-e_{ij}\Omega_{ab},
\]

\[
\log p(A\mid s,\Omega,e)=
\sum_{i<j}\log\psi^G_{ij}(s_i,s_j)
+\sum_{i<j}\{A_{ij}\log e_{ij}-\log(A_{ij}!)\}.
\]

이것은 **degree를 이용한 conditional/empirical-Bayes exposure score**다. Observed adjacency에서 e를 만들었으므로 정규화된 원래 graph 생성 likelihood라고 부르면 안 되고, 정확한 \(p(A\mid d)\)도 아니다. Degree sequence에 정확히 조건화하면 그 sequence를 만족하는 모든 graph에 대한 추가 partition이 생기며, 위 offset만 넣는 것으로 그 정규화가 수행되지는 않는다.

그래도 기존과의 계산 차이는 명확하다. 모든 node pair를 같은 opportunity로 보지 않고 degree-product exposure를 사용한다. Class별 연결 강도도 고정된 global role이 아니라 실제 edge count를 exposure로 나눈 통계에 적응한다. 큰 class의 낮은 pair density를 작은 class와 같은 fixed contrast로 벌하지 않는다. **Population 편향을 완전히 제거했다는 보장은 없다.** Degree offset, affinity regularization와 population 추론을 함께 검증해야 한다.

## 3. Block-rate nuisance를 주변화하는 정확한 기준식

아래 모든 block pair는 \(a\le b\), 모든 graph edge는 \(i<j\) convention이다. Within-class edge는 한 번 센다. Symmetric matrix로 저장한다고 Gamma prior나 count를 두 번 세면 안 된다.

\[
\Omega_{ab}\sim\operatorname{Gamma}(\alpha_0,\beta_0)
\quad\text{(shape, rate)},
\]

\[
M_{ab}(s)=\sum_{i<j}A_{ij}\mathbf1[\{s_i,s_j\}=\{a,b\}],
\quad
E_{ab}(s)=\sum_{i<j}e_{ij}\mathbf1[\{s_i,s_j\}=\{a,b\}].
\]

Affinity를 정확하게 적분한 block contribution은

\[
F_{ab}(M,E)=\log\Gamma(\alpha_0+M)-\log\Gamma(\alpha_0)
+\alpha_0\log\beta_0-(\alpha_0+M)\log(\beta_0+E).
\]

이는 tiny exact reference를 만들 수 있는 핵심이다. 각 label assignment의 counts/exposures를 계산해 \(\sum_{a\le b}F_{ab}\)로 비교하면 continuous affinity quadrature가 필요 없다. 정해진 assignment에서 posterior는 Gamma(\(\alpha_0+M,\beta_0+E\))다.

Profile MLE \(\widehat\Omega_{ab}=M_{ab}/E_{ab}\)라면 graph score가 대략 \(\sum_{ab}M_{ab}\log(M_{ab}/E_{ab})-M_{ab}\)로 바뀌는 것도 확인할 수 있다. 후보에서는 Gamma regularization으로 작은/비어 있는 class pair의 불안정을 다룬다. Profile objective와 Gamma-marginal objective를 같은 식이라고 쓰지 않는다. 이웃이 없는 class나 낮은 source coverage를 oracle count로 채우지 않는다.

Unknown class fraction은 별도로

\[
\pi\sim\operatorname{Dirichlet}(\alpha,\ldots,\alpha)
\]

로 두되 **context를 제외한 노드의 conditional prior**에만 적용하는 정의를 권한다. 현재 context는 class당 개수를 맞춰 선택하므로 population의 iid 표본이 아니다. 균등한 context frequency를 population estimate로 쓰면 바로 고치려는 균형 가정을 다시 넣는다. 이 conditional 정의는 context-selection process를 정확히 모델링한 generative Bayes라고 주장하지 않는 대신, stratified context를 exogenous evidence로 취급한다.

Unlabeled node count와 degree volume을 구분해야 한다.

\[
n_a^U=\sum_{i\notin C}b_i(a),\qquad
\kappa_a=\sum_{i\in V}w_i b_i(a).
\]

\(n^U\)는 population 갱신에, \(\kappa\)는 graph exposure에 쓴다. Context는 \(\kappa\)와 edge counts에는 포함된다. \(\kappa_a/\sum_b\kappa_b\)를 node class prior로 쓰면 degree nuisance를 population prior로 되돌려 넣는다.

## 4. 실제 실행할 inference: Gamma/Dirichlet update + BP

정확히 collapsed된 graph score는 global high-order factor다. 이를 기존 pair-Potts BP에 넣었다고 exact collapsed inference가 되지 않는다. 실행 후보는 **finite variational-Bethe / empirical-Bayes unroll**로 명시한다.

현재 pair/node beliefs로 \(\bar M_{ab},\bar E_{ab},n_a^U\)를 만들고

\[
q(\Omega_{ab})=\operatorname{Gamma}(a_{ab},b_{ab}),
\quad a_{ab}=\alpha_0+\bar M_{ab},\quad b_{ab}=\beta_0+\bar E_{ab},
\]

\[
L_{ab}=\mathbb E\log\Omega_{ab}=\operatorname{digamma}(a_{ab})-\log b_{ab},
\qquad W_{ab}=\mathbb E\Omega_{ab}=a_{ab}/b_{ab},
\]

\[
q(\pi)=\operatorname{Dirichlet}(\alpha+n^U),\qquad
g_a=\operatorname{digamma}(\alpha+n_a^U)
-\operatorname{digamma}(K\alpha+|U|).
\]

Node prior term \(g_a\)는 unknown nodes에만 쓰고 context는 clamp한다. Gamma uncertainty를 유지하는 이 variational convention에서는 `L=log(W)`로 바꾸면 다른 알고리즘이다. Posterior mean plug-in EM을 택한다면 그렇게 별도로 정의해야 한다.

고정 \(L,W,g\)에서 dense pair log potential은

\[
\Phi_{ij}(a,b)=A_{ij}L_{ab}-e_{ij}W_{ab}.
\]

C의 incoming feature log field를 \(f_i^X(a)\)라 할 때

\[
\log p_{i\setminus j}(a)\propto
g_a+f_i^X(a)+\sum_{k\ne i,j}u_{k\to i}(a),
\]

\[
u_{i\to j}(b)=\operatorname{LSE}_a
\{\log p_{i\setminus j}(a)+\Phi_{ij}(a,b)\}
\]

를 mean-zero gauge와 사전 고정한 damping으로 실행할 수 있다. 임의 K×K compatibility이므로 원래 `1+(exp(J)-1)*p` Potts shortcut을 그대로 사용할 수 없다. 이 dense reference는 conditional fixed-parameter local update의 정확성을 점검하는 기준이다. Continuous-rate posterior update 자체는 그와 다른 approximation layer다.

**동기 schedule도 명시해야 한다.** 예를 들어 이전 beliefs로 nuisance parameters를 갱신한 뒤 그 값을 고정하고 이전 graph/feature messages에서 다음 messages를 만드는 한 macro-step을 정의할 수 있다. 반복 중 hard argmax counts로 바꾸거나 무심코 `detach()`하면 intended gradient가 바뀐다. 아직 계산하지 않은 수렴 posterior의 gradient라고 부르지 않는다.

## 5. Sparse external field: absence를 없애지 않는 근사

Nonedge \(A_{ij}=0\)에서 exact message는

\[
\log m_{j\to i}(a)=\log\sum_b p_{j\setminus i}(b)e^{-e_{ij}W_{ab}}
\approx-e_{ij}\sum_b p_j(b)W_{ab}.
\]

즉 sparse approximation은 absence evidence를 **degree/class-volume field로 합치는 것**이지 무조건 0으로 만드는 것이 아니다. 두 음항을 중복 계산하지 않는 명확한 버전은 다음과 같다.

\[
R_i(b)=\kappa_b-w_i b_i(b)
-\sum_{j\in\partial i}w_j b_j(b),
\quad
H_i(a)=w_i\sum_bW_{ab}R_i(b).
\]

관측 edge는 full Poisson factor \(\Phi_{ij}=A_{ij}L-e_{ij}W\)로 유지한다. Unknown node의 graph cavity는

\[
\log p_{i\setminus j}(a)\propto
g_a+f_i^X(a)-H_i(a)+
\sum_{k\in\partial i\setminus j}u_{k\to i}(a).
\]

위 R은 self와 **모든 neighbor**를 제외한다. All-pair field에 이미 neighbor의 \(-eW\)를 넣고 edge factor에서도 이를 넣는 double count는 금지다. 반대로 edge에서 음항을 빼기로 정하면 field/conjugate exposure 식도 그 convention에 맞게 다시 정의해야 한다.

Exposure 충분통계도 일관돼야 한다. Independent node approximation에서

\[
E^{MF}_{aa}=\tfrac12\{\kappa_a^2-\sum_iw_i^2 b_i(a)^2\},
\]

\[
E^{MF}_{ab}=\kappa_a\kappa_b-\sum_iw_i^2b_i(a)b_i(b),\quad a<b.
\]

`soft b`에 hard-count 식 \(b^2=b\)를 적용하면 안 된다. Edge pair belief를 사용한다면 MF all-pair exposure에서 MF edge contribution을 빼고 actual edge pair contribution을 더한다. \(M\)는 observed edge pair beliefs에서 구한다. 구현 전 \(\sum_{a\le b}\bar M_{ab}=M\), \(\sum_{a\le b}\bar E_{ab}=\sum_{i<j}e_{ij}\) conservation을 확인해야 한다.

External-field 근사는 weak nonedge intensity를 전제로 한다. Physics에 hub가 있다는 사실 때문에 이 가정이 자동으로 맞는 것이 아니다. Tiny dense reference와의 차이, 특히 높은 \(\max_{i\ne j}e_{ij}W\) 영역을 따로 확인해야 한다. Poisson은 binary adjacency의 surrogate이기도 하므로 large intensity에서 Bernoulli와의 동등성을 주장하지 않는다. Target 결과를 보고 intensity를 clip하면 conjugacy/model을 바꾸므로 별도 cycle이 필요하다.

계산은 sparse version에서 graph messages \(O(|E|K)\), graph updates \(O(|E|K^2+NK^2)\) 및 기존 feature BP 비용이다. Dense exact local reference는 \(O(N^2K^2)\) work와 \(O(N^2K)\) messages이므로 작은 correctness reference부터 사용한다. 정당화되지 않은 sparse approximation을 비용 때문에 exact라고 부르지 않는다.

## 6. Feature evidence가 query로 가는 경로

현재 source/context 분리와 query-feature absence는 유지한다. C의 cell messages는 그대로이며 경로는

\[
Y_C\to\text{graph messages}\to b_{i\in O}
\to q(\mu_{cf})\to b_{j\in O}
\to\text{graph messages}\to b_q,
\quad i\ne j.
\]

Query에는 직접 cell factor가 없다. 새 graph field는 large-class assignment를 고정된 node-count penalty로 밀어내는 대신 observed degree volume과 inferred affinity에 따라 바뀐다. Context에서 source로 도착한 잘못된 과신이 완화된다면 centroid orientation과 source→query graph feedback이 올바른 쪽으로 작동할 여지가 생긴다. 이것이 **검증할 계산 가설**다. Graph repair가 feature utility를 반드시 양수로 만든다는 보장은 없다. Graph-only가 이미 충분히 잘 맞거나 feature law 자체가 맞지 않으면 추가 효용은 작을 수 있다.

Nuisance counts도 feature-conditioned beliefs를 읽으면 feature evidence가 \(W,g,H\)를 바꿀 수 있다. 이 피드백을 학습 때 미분하려면 soft counts→digamma/rate→messages 경로를 보존해야 한다. Graph potentials만 고정하고 메시지 전체를 `no_grad`로 감쌌던 유형의 실수를 반복하면 안 된다. No-observed, shuffle, source 0/1 경계는 계속 유지하며 source0/1의 centroid exchangeability 제한은 이 graph repair로 없어지지 않는다.

## 7. 식별성·누출·주장상의 함정

1. **Scale gauge:** w를 고정하면 rate scale의 convention이 고정된다. Node-specific multipliers까지 학습하면서 affinity도 자유롭게 두면 node/class scale gauge가 생긴다. 첫 후보에 그 자유도를 조용히 추가하지 않는다. Uniform thinning은 degree-based exposure에 흡수될 수 있지만 pair-specific unknown observation propensity는 adjacency만으로 식별되지 않는다.
2. **Population:** Class-balanced context frequency, query truth의 class counts, 후처리한 실제 block edge densities를 population/rate 입력으로 쓰지 않는다. Query의 *belief*를 transductive sufficient statistic에 쓰는 것은 허용되는 inference지만 query *정답*은 loss/scoring에만 쓴다.
3. **Symmetry and collapse:** 완전히 class-flat affinity와 uniform beliefs는 무정보 fixed point를 만들 수 있다. Permutation-equivariant 초기화와 hyperprior를 개발 전 고정하고 classical comparator에도 맞춘다. 초기화나 여러 restart 중 test accuracy가 높은 것을 고르면 안 된다. Sparse context가 class identity를 모두 식별한다고 가정하지 않는다.
4. **Posterior versus fitting:** Gamma/Dirichlet 갱신은 target graph에서 latent nuisance를 추론하거나 EM식으로 맞추는 계산이다. Network gradient update가 0이라는 표현은 맞지만 “target-side parameter fitting/adaptation도 전혀 없다”는 표현은 부정확하다. Finite Bethe beliefs가 locally inconsistent하면 analytic Gamma 식을 사용했다고 calibrated joint posterior가 되는 것도 아니다.
5. **Constants and Bethe:** Rate-integrated evidence를 비교하면 Gamma normalization과 exposure terms를 유지한다. Pairwise Bethe score는 parameter가 고정된 dense/sparse model에 대해서만 그 정의를 쓴다. Collapsed global Gamma factors에 old Potts entropy 식을 그대로 재사용하지 않는다. 첫 후보는 별도 role-selection mixture 없이 한 symmetric affinity model로 정의하는 편이 명확하다.

## 8. 구현 전에 필요한 최소 확인과 결론

- Fixed L/W와 hard context의 tiny tree에서 generic K×K Poisson pair message·partition·self-loop exclusion을 enumeration으로 검증한다. 이어 tiny whole model에서는 위 Gamma/Dirichlet 적분식으로 exact label/grid sum을 만들어 **nuisance approximation**과 **graph-BP approximation**을 분리한다. Zero error가 필요한 곳과 approximation을 보고할 곳을 구분한다.
- Label/name permutation, node permutation, zero-degree nodes, all-class volume conservation, dense versus sparse field, finite gradients를 확인한다. Class proportion과 degree propensity를 독립적으로 바꾼 새로운 개발 조건에서 기존 fixed-Potts 및 동일 classical DC inference와 비교해야 한다. 이 검사에는 실제 test labels로 추정한 rate를 입력하지 않는다.
- 실행 계획은 새 prior/새 final seeds/강한 classical DC comparator와 observed−none·shuffle·NLL/minority metrics를 먼저 고정해야 한다. FP01 final 또는 기존 다섯 real graph에서 field coefficient·class mass·initialization을 고르는 rescue를 하면 안 된다.

이 후보는 **기존 DC block-model 원리에 기반한 1차 구조 수정**으로 타당하다. 가장 직접적인 nonedge 손해를 겨냥하면서 CiteSeer에 유용했던 absence evidence를 전부 버리지 않는 계산적 이유가 있다. 다만 degree/exposure conditioning과 posterior approximation의 정의를 고정하고 위 오류들을 차단한 후에야 구현 가능한 사전 계획이다. Novel algorithm이나 실제 feature recovery의 성공은 현재 주장할 수 없다.

## Addendum — concrete FP02 PLAN_v1 review before code

추가로 `factor_nuisance_cycle/PLAN_v1.md`를 읽었다. Stage0의 frozen seed11 C_global, old-full/old-edge/DC-fixed/DC-update, rep0 none/95의 고정 개발 비교는 위 후보의 제한된 첫 검증으로 타당하다. 아래 네 가지는 **구현 전에 문서상 고정할 blockers**다. 새로운 성능 결과에 근거한 변경은 아니다.

### B1. Prior와 posterior를 분리해 evidence를 한 번만 센다

초기 mean을 \(\mu^0_{aa}=1+(K-1).6\), \(\mu^0_{ab}=.4\;(a\ne b)\), shape=5로 택했다면 고정 prior는

\[
a^0_{ab}=5,\qquad b^0_{ab}=5/\mu^0_{ab}.
\]

매 nuisance update는 **항상** \(a^{new}=a^0+\bar M\), \(b^{new}=b^0+\bar V\)다. 이전 posterior shape/rate에 같은 graph의 충분통계를 다시 더하면 outer 횟수만큼 evidence를 중복한다. Dirichlet도 이전 posterior에 counts를 누적하지 않고 `alpha + current unlabeled counts`로 다시 만든다. V는 본문의 E exposure와 같은 충분통계다.

`DC-fixed`는 이 **동일 Gamma prior를 갱신하지 않는** operator로 고정하는 것이 비교상 명확하다. 즉 \(R=\mu^0\)뿐 아니라 \(L=\operatorname{digamma}(5)-\log(5/\mu^0)\)도 사용한다. 여기서 fixed arm만 `L=log(R)`를 쓰면 nuisance update 외에 Gamma uncertainty convention도 달라진다. 만약 그 비교를 원한다면 다른 baseline으로 명시해야 한다.

### B2. 4×6 schedule의 마지막 parameter와 prediction을 일치시킨다

“6회마다 갱신, 4outer, BP24”만으로는 첫 block 직전 갱신 여부와 마지막 block 뒤 parameter의 사용 여부가 모호하다. 단순한 권장 명세는 **4개 BP blocks, 각6회**다. Block0은 초기 prior를 사용하고, block1/2/3 시작에서 이전 block beliefs로 nuisance를 갱신한다. 그러면 총24회 message update와3회 nuisance update이며 마지막 prediction은 마지막 block에 사용한 parameters와 연결된다.

다른 schedule도 유효하지만 명시해야 한다. 마지막6회 뒤 네 번째 nuisance update를 한 뒤 이전 messages로 만든 prediction을 반환하면서 새 parameters가 그 prediction을 생성했다고 기록하면 안 된다. 마지막 update를 진단으로만 보관하거나, 정확한 lagged-return 정의를 적어야 한다. 아직 graph 결과를 보지 않은 지금 하나를 선택한다.

### B3. Field와 pair statistics가 같은 iterate를 읽게 한다

Affinity/population을6회 동안 고정해도 nonneighbor field H는 **각 inner BP step의 이전 node beliefs**에서 다시 구해야 한다. Self와 neighbors는 동일한 beliefs와 동일한 w로 뺀다. Final edge pair belief는 실제 마지막 node field·opposite messages·그 block의 L/R에서 재구성한다. H만 새로 계산한 field와 이전 L/R의 cavity를 섞지 않는다. Finite messages가 locally consistent해지는 것은 보장하지 않지만, 어느 iterate를 쓰는지에 대한 정의는 정확해야 한다.

Sparse rate exposure는 `all-pair MF − edge MF + actual edge pair`다. M과 V를 unordered pair convention으로 저장하고 diagonal을 한 번만 센다. Population count는 U의 node counts, volume은 모든 node의 degree-weighted counts라는 구분도 그대로 유지한다. 이 부분은 conservation 및 dense/sparse hand-reference 검사에서 직접 assertion을 두는 것이 맞다.

### B4. Tiny exact 검사의 equality 범위를 고정한다

고정 L/R의 tree 또는 N=2 single-pair reference에는 exact local-update/marginal equality를 요구한다. N≥3 full Poisson graph나 multi-source centroid graph는 이미 cyclic하므로 dense BP도 일반적으로 exact enumeration과 같지 않다. Gamma/Dirichlet을 analytic하게 적분한 exact tiny label/grid reference와 variational update의 차이도 0을 강요하지 않는다. 세 오차를 섞지 않는다: **local implementation error, fixed-model loopy error, nuisance-factorization/sparse-field error**다.

### Nonblocking scope corrections

- Init eta .6은 과거 학습 potential의 실제 signal 약 .38과 다르다. 따라서 old-full→DC-fixed 차이를 degree correction 하나의 인과 효과로 부를 수 없다. 새 positive-edge contrast, prior uncertainty, affinity family, exposure가 함께 바뀌는 구조 수정의 결과다.
- Large hub pair에서 eR가 작다는 보장이 없고 Poisson과 binary likelihood의 차이도 커질 수 있다. Intensity/field diagnostic을 남기되 Stage0 결과를 보고 clip하거나 eta를 다시 고르는 rescue는 금지한다.
- Stage0 success를 “Physics/CS 회복 + CiteSeer 장점 보존”이라고 표현하려면 accuracy뿐 아니라 NLL/balanced accuracy·비교 대상·허용 손상 범위를 결과 전에 정해야 한다. Development pilot이면 기술 통계로 모두 보고할 수도 있으나, 결과 후 임의 문턱을 만들어 gate라고 부르면 안 된다.
- `DC-update`가 affinity와 population을 함께 바꾸므로 이 한 비교로 둘의 고유 원인을 분리했다고 쓰지 않는다. Strong classical inference가 더 잘 해결하면 그것이 구조 수정의 성과이며 neural factor의 필요성은 별도다.

**Concrete-plan verdict:** 위 네 구현 정의를 잠그면 Stage0 구현·수치검증으로 진행할 수 있다. 이 문서에서는 코드나 실행을 시작하지 않았다.

# FP01 independent review — mathematics, inference and implementation

검토 대상: `PLAN_v1.md`, root `AGENTS.md`, 원래 `factor_real_graph_pilot/model.py`, 이전 `REVIEW_METHOD.md`. 다른 current-cycle review는 읽지 않았다. 이 문서는 사전 설계 검토이며 구현·실험·학습을 수행하지 않았다.

## Verdict and required plan changes

**Conditional go.** Class-indexed centroid factorization은 joint score를 정확히 표현하며, Gaussian quadrature를 normalized categorical prior로 정의하면 유효한 finite-state conditional model이다. 구현도 dense heterogeneous pair tensor 없이 가능하다. 다만 다음을 계획 v2에서 고정해야 한다.

1. Gaussian prior quadrature와 discrete latent prior의 정의를 혼동하지 않는다. GH weights가 이미 prior measure를 포함한다. Normal prior density를 다시 곱하면 틀린다.
2. Pairwise factorization은 source가 둘 이상이면 graph edge가 없어도 loop를 만든다. 따라서 exact-tree 통과를 실제 centroid inference의 정확성 증명으로 삼지 않는다.
3. **FP01에는 A/B/C/C-fixed 모두에 공통으로 frozen graph-only role weights를 쓰는 방식을 권한다.** 새로운 centroid operator의 검증과 새 finite-Bethe role estimator의 검증을 분리할 수 있다. 이는 정규화된 conditional mixture이지, feature-conditioned joint role posterior가 아니다. 기존 historical Factor와 role protocol이 달라졌음을 표시해야 한다.
4. C를 우선 discrete Gaussian-centroid-grid structured model로 명시한다. 11-point rule의 continuous Gaussian fidelity는 source 수가 커질수록 당연히 보장되지 않는다. 더 큰 grid를 쓰더라도 같은 제한이 있으며, 고정-label 해석해와 별도로 대조해야 한다.

이하에 구현 가능한 메시지 식과, full role scoring을 유지할 경우 필요한 정확한 score 정의를 모두 적는다. 새 architecture/head/loss 제안으로 범위를 넓힐 필요는 없다.

## 1. Fixed discrete model and quadrature convention

Class 수를 \(K\), quadrature state 수를 \(R\)라 하자. \(R\)는 query 집합과 구별하기 위한 기호다. 각 column의 hyperparameters \(m_f,\tau_f>0,\sigma_f>0\)는 관측 입력으로부터 만들되 **모든 class와 role에 같은 값**을 사용한다. 모든 class에 같은 prior를 두어도 latent centroid \(\mu_{cf}\) 자체는 class마다 다르다. Class-specific prior를 만들려면 별도의 equivariant 정의가 필요하며 singleton 결론도 달라지므로 이번 계획에 조용히 추가하지 않는다.

`numpy.polynomial.hermite.hermgauss(R)`의 nodes/weights를 \((t_r,W_r)\)라 하면

\[
\mu_{cfr}=m_f+\sqrt2\tau_f t_r,
\qquad w_r=W_r/\sqrt\pi,\qquad\sum_rw_r=1.
\]

이는 \(\exp(-t^2)\) weight convention이다. `hermegauss`의 \(\exp(-t^2/2)\) convention과 섞으면 \(\sqrt2\)와 normalization이 틀어진다. 해당 convention은 [NumPy hermgauss 공식 문서](https://numpy.org/doc/2.1/reference/generated/numpy.polynomial.hermite.hermgauss.html)에 명시돼 있다. 이 변환 뒤 latent index \(r_{cf}\)에는 categorical prior \(w_r\)만 둔다. **추가로** \(\mathcal N(\mu_{cfr};m_f,\tau_f^2)\), \(\Delta\mu\), \(\tau_f\) Jacobian을 곱하지 않는다. Prior가 없는 column도 정확히 \(\sum_rw_r=1\)로 적분되어야 한다.

관측 cell \(e=(i,f)\)의 log likelihood table을

\[
\ell_{er}=-\tfrac12\log(2\pi)-\log\sigma_f
-\frac{(x_{if}-\mu_{cfr})^2}{2\sigma_f^2}
\]

로 둔다. Shared prior grid이므로 table 자체는 c/h와 무관하고, latent index 선택만 c에 따라 달라진다. \(x\)는 RMS/clipping 후 predictor 좌표의 값이다. Grid를 observed mean/std로 옮긴다면 그 값을 Gaussian prior의 location/scale로 정의하는 것인지, 동일한 고정 prior를 다른 integration grid로 근사하는 것인지 구분해야 한다. 위 정의는 전자다. Input-dependent hyperparameters이므로 continuous likelihood 모양을 사용해도 전체 모델은 conditional score다.

Node prior를 \(u_i(a)=\mathbf1[a=Y_i]\) for context, otherwise 1로 두는 convention을 권한다. Uniform \(1/K\) prior를 선택해도 되지만 absolute partition 검증 양쪽에 동일하게 포함해야 한다. 그때 \(-N\log K\)는 role 간 공통이다.

\[
\mathcal W_h(s,r)=e^{A_h}\prod_i u_i(s_i)
\prod_{cf}w_{r_{cf}}
\prod_{(i,j)\in E_G}e^{J^h_{ij}\mathbf1[s_i=s_j]}
\prod_{e=(i,f)\in\Omega}\prod_{c=1}^K
\psi_{ec}(s_i,r_{cf}),
\]

\[
\psi_{ec}(a,r)=
\begin{cases}e^{\ell_{er}},&a=c,\\1,&a\ne c.\end{cases}
\]

\(E_G\)는 full panel에서 모든 unordered node pair, edge-only panel에서는 유지한 graph edge다. Missing cell은 \(\Omega\)에 존재하지 않는다. \(\prod_c\psi_{ec}=\mathcal N(x_{if};\mu_{s_i f},\sigma_f^2)\)는 정확한 항등식이다.

Likelihood normalization을 단순히 버리지 않는다. Class-independent \(\sigma_f\)이면 \(-\log\sigma_f-\frac12\log2\pi\)를 **cell당 한 번** global constant로 뺄 수 있다. 하지만 gated factor마다 Normal을 독자적으로 재정규화하는 것은 메시지 factorization을 바꾼다. Exact joint와 partition의 보상 상수를 명시해야 하며, 같은 joint를 표현해도 finite loopy BP trajectory는 달라질 수 있다. 우선은 위 literal Normal convention을 그대로 고정하는 편이 검증하기 쉽다.

## 2. Exact factorization does not imply exact BP

두 source i,j와 한 column f, 두 class c,d만 있어도

\[
s_i-\mu_{cf}-s_j-\mu_{df}-s_i
\]

라는 4-cycle이 생긴다. 이는 graph edge가 하나도 없어도 존재한다. 전체 centroid-vector를 하나의 variable로 묶으면 feature-only model을 다른 형태로 표현할 수 있지만, 그 variable은 \(R^K\) states가 필요하다. 현재 classwise pair factorization은 이를 피하는 유효한 계산 선택이며 **loopy approximation을 새로 도입한다**.

따라서 “exact sum-product messages”는 각 pair update가 정확하다는 뜻으로 한정한다. Actual multi-source centroid model의 marginal/partition이 정확하다는 뜻이 아니다. Tree tests는 generic update/normalization 구현을 확인하고, 실제 전체 factorization의 작은 cyclic 사례를 별도로 exact enumeration해야 한다. 이 차이를 plan과 결과에 남기는 것이 중요하다.

## 3. Executable sum-product updates

각 role을 독립적인 BP lane으로 실행한다. \(H=3\), observed cell 개수를 \(E_X=|\Omega|\)라 하면 필요한 feature message 저장은 다음과 같다.

| Quantity | Shape | Meaning |
|---|---|---|
| `g[h,j,i,a]` | H×N×N×K (또는 sparse graph edges) | graph j→i log message |
| `a[h,e,c]` | H×E_X×K | centroid(c,f)→node i message의 active-class log ratio |
| `v[h,e,c,r]` | H×E_X×K×R | node i→centroid(c,f) log message |
| `T[h,c,f,r]` | H×K×F×R | centroid total log field |
| `Fnode[h,i,a]` | H×N×K | node total log field |

`a`의 c는 edge가 gate하는 class이자 node label 좌표다. K×K node-message tensor는 필요 없다. `a`라는 이름과 label state a가 충돌하지 않도록 실제 구현에서는 `to_node`, `class_state` 등으로 구별한다.

Old messages로부터

\[
F_i(k)=\sum_j g_{j\to i}(k)+
\sum_{e:i_e=i} a_{e,k},
\qquad
T_{cf}(r)=\log w_r+\sum_{e:f_e=f}v_{ec}(r)
\]

를 계산한다. Context는 softmax 이후 정확한 one-hot으로 override하며, hard-clamp의 \(-\infty\)를 finite fields에 넣었다 뺐다 하지 않는다.

### Centroid→node proposal

Centroid cavity는

\[
q^{\setminus e}_{cf}(r)=
\operatorname{softmax}_r\{T_{cf}(r)-v_{ec}(r)\}.
\]

Incoming factor message는 off-class states에서 1, class c에서 \(\sum_rq^{\setminus e}_{cf}(r)e^{\ell_{er}}\)이므로

\[
a^{\rm prop}_{ec}=
\operatorname{LSE}_r\{\log q^{\setminus e}_{cf}(r)+\ell_{er}\}.
\]

이 scalar는 active class와 off-class의 log ratio다. Vector softmax를 취할 필요가 없다.

### Node→centroid proposal

Node cavity에서 해당 feature edge 하나만 제외한다.

\[
p^{\setminus(ec)}_i(k)=
\operatorname{softmax}_k\{F_i(k)-a_{ec}\mathbf1[k=c]\},
\qquad \rho_{ec}=p^{\setminus(ec)}_i(c).
\]

Context이면 이 cavity도 context class에 clamp한다. 그 뒤

\[
v^{\rm raw}_{ec}(r)=
\log\{(1-\rho_{ec})+\rho_{ec}e^{\ell_{er}}\}
=\operatorname{logaddexp}
\{\log(1-\rho_{ec}),\log\rho_{ec}+\ell_{er}\}.
\]

`log(1-rho)`를 float32 `1-softmax`로 계산하면 rho≈1에서 정보가 사라진다. Node cavity의 off-class logsumexp와 active-class log weight를 직접 사용한다. \(\rho=0/1\)도 정확히 처리할 수 있다. Clamp context의 log probabilities는 0/−∞이므로 `0 * -inf`는 safe entropy 구현에서 피해야 한다.

Forward updates에는 \(\rho\)만 필요하다. 각 edge마다 K×K cavity를 materialize하지 않아도 된다. 예컨대 unclamped node에서 `L=log_softmax(F_i)`라 하면

\[
D_{ec}=\operatorname{logaddexp}
\{\operatorname{LSE}_{k\ne c}L_k,\;L_c-a_{ec}\},
\quad
\log\rho_{ec}=L_c-a_{ec}-D_{ec},
\quad
\log(1-\rho_{ec})=\operatorname{LSE}_{k\ne c}L_k-D_{ec}.
\]

\(\operatorname{LSE}_{k\ne c}\)는 작은 K에서 명시적으로 계산하거나 prefix/suffix logsumexp로 계산할 수 있다.

### Graph proposals, damping and final beliefs

Graph cavity에는 **모든 feature incoming messages를 포함**하고 반대 graph message 하나만 뺀다. 그 후 원래 Potts formula

\[
g^{\rm raw}_{i\to j}(k)=
\log\{1+(e^{J^h_{ij}}-1)p_{i\setminus j}(k)\}
\]

를 쓴다. Large negative coupling에서는 logaddexp 형태가 더 안전하다. 기존 `expm1/log1p` implementation의 bounds를 유지한다면 해당 범위 검증도 유지한다.

Gauge convention을 먼저 정한다. 예컨대 graph K-vector와 centroid R-vector proposal은 mean-zero로 center하고, scalar `a`는 off-class baseline 0을 유지한다. 모든 directed messages는 0으로 초기화한다. Log-space damping은

\[
m^{t+1}=(1-\lambda)m^t+\lambda m^{\rm prop},\qquad\lambda=.5
\]

이며 probability mixture damping과 다르다. 24번 **동기적**으로 업데이트한다. 한 iteration 안에서 `g`, `a`, `v` proposal은 모두 old messages/fields를 읽어야 한다. Column별 in-place update로 다른 column의 새 message를 읽으면 다른 inference operator다.

Final node beliefs는 clamp-aware softmax of `Fnode`, centroid beliefs는 softmax of `T`다. Frozen graph parameters라도 graph BP messages는 feature parameters에 의존한다. **Graph potentials/encoder를 no-grad로 캐시할 수는 있지만 conditional graph BP 전체를 no-grad로 감싸면 feature→graph→query gradient를 끊는다.** 이 오류가 이번 구현에서 가장 주의할 code hazard다.

## 4. Full heterogeneous Bethe score, if retained

Frozen graph-only role protocol을 선택하면 C의 이 score를 role mixing에 쓰지 않아도 된다. 그래도 구현 검증·향후 해석을 위해 올바른 식을 남긴다.

Final directed messages로부터 각 edge의 normalized cavities \(p_i,p_j\) 또는 \(p_i,q_{cf}\)를 구한다. Pair belief는

\[
\bar b_{uv}(a,b)\propto p_{u\setminus v}(a)
p_{v\setminus u}(b)\psi_{uv}(a,b).
\]

Feature edge에서는 \(\rho=p_i(c)\), \(A=\sum_rq(r)e^{\ell_{er}}\),

\[
Z_e=(1-\rho)+\rho A,
\quad
\bar b_i(c)=\rho A/Z_e,
\quad
\bar b_i(k\ne c)=p_i(k)/Z_e,
\]

\[
\bar b_{cf}(r)=q(r)\{(1-\rho)+\rho e^{\ell_{er}}\}/Z_e.
\]

`log Z_e`도 logaddexp로 계산한다. Full K×R pair table을 만들지 않고 pair energy-plus-entropy를

\[
B_e=\mathbb E_{\bar b_e}\log\psi_e+H(\bar b_e)
=\log Z_e-\sum_k\bar b_i(k)\log p_i(k)
-\sum_r\bar b_{cf}(r)\log q(r)
\]

로 계산할 수 있다. 원래 Potts edge score도 같은 identity를 사용한다.

Node graph degree를 \(d_i^G\), observed column 수를 \(n_i^X\)라 하면 centroid pair graph에서

\[
d_i=d_i^G+K n_i^X,
\qquad d_{cf}=\#\{i:(i,f)\in\Omega\}.
\]

각 observed cell이 **K개의 pair edge**를 만드는 것을 빠뜨리면 entropy correction이 틀린다. Pairwise Bethe 형태의 finite score는

\[
S_h=\sum_{e\in E_G\cup E_{\rm centroid}}B_e
+\sum_i(1-d_i)H(b_i)
+\sum_{cf}\left\{
\sum_rb_{cf}(r)\log w_r+(1-d_{cf})H(b_{cf})\right\}.
\]

Uniform node prior를 사용했다면 \(-N\log K\)를 추가한다. Hard context unary는 허용된 support에서 log 1=0이다. **GH prior의 \(\mathbb E_b\log w\) 항은 role-dependent일 수 있으며 빠뜨릴 수 없다.** Grid weights를 cavity initialization에만 넣는 것으로 이 free-energy prior term이 자동 포함되는 것은 아니다. Isolated centroid는 \(b=w\)이므로 \(\mathbb E_w\log w+H(w)=0\)이 되어야 한다.

Literal Normal factors라면 해당 상수는 \(B_e\)에 이미 들어 있다. 따로 global constant를 추출한 convention이면 cell당 한 번의 constant를 추가한다. Graph baseline은 원래와 같이 \(A_h=\sum_{i<j}L^h_{ij,\mathrm{diff}}\)를 유지하며 double-count하지 않는다. Full role weights를 택했다면

\[
q(h)=\operatorname{softmax}_h\{\log\pi_h+A_h+S_h\}.
\]

이 식도 24-step node beliefs와 pair marginal이 locally inconsistent하면 converged Bethe optimum이 아니다. Finite unrolled energy–entropy score로 정의해야 한다. 동일 joint의 다른 factorization/normalization이 finite approximation을 바꿀 수 있으므로 precision 향상만으로 해결되는 문제가 아니다. Centroid factor가 추가된다고 이 문제는 사라지지 않는다.

## 5. Recommended simpler role protocol for FP01

FP01의 목적은 projection interface와 feature operator다. **각 frozen graph checkpoint, graph/context/class-count, full/edge panel에 대해** feature가 없는 원래 graph-only Potts BP를 실행하여

\[
\bar q_G(h)=\operatorname{softmax}_h
\{\log\pi_h+A^G_h+S^G_h\}
\]

를 한 번 구해 cache/detach한다. A/B/C/C-fixed는 동일한 \(\bar q_G\)를 사용하고 각자의 conditional BP beliefs를

\[
\widehat p_C(s_i=c\mid D)=
\sum_h\bar q_G(h)b^C_{i,h}(c)
\]

로 섞는다. 이는 합이 1인 유효한 predictor다. Feature-conditioned joint posterior로 부르거나 C의 latent prior가 role weights에 반영됐다고 쓰면 안 된다. Feature로 graph role을 바꾸는 능력은 의도적으로 검증 범위 밖에 둔다. 잘못된 graph-only role 선택이 모든 arm에 공통 병목일 수 있으므로 negative result의 범위도 여기에 한정한다.

Full/edge panel에서 q를 각각 graph-only로 다시 계산할지, full q를 둘 다에 고정할지도 계획에 써야 한다. **각 panel의 자체 graph-only q를 사용**하는 방식을 권하지만, 이 경우 원래 “full role weights를 고정한 nonedge deletion” 진단과 다른 intervention이다. Edge-only의 model score를 쓴다면 nonedge \(J\)뿐 아니라 nonedge \(A_h\)도 제거한다. Full A를 유지하면서 edge-only partition을 Bayesian evidence라고 부르면 안 된다.

이 선택은 `C`에만 적용하지 말고 새 A/B/C/C-fixed 모두에 적용해야 한다. Historical original Factor의 feature-updated role protocol은 별도로 유지한다. Graph-only q는 query truth나 generator centroid 없이 graph/context만으로 산출한다. Training에서도 feature-module gradient가 q를 통해 흐르지 않는다는 점을 보고한다.

## 6. Singleton behavior and permutation semantics

관측 cell이 하나 있는 column에서 centroid priors가 class/role에 동일하면, 고정 class a에 대해

\[
\sum_{r_{1f},\ldots,r_{Kf}}
\prod_cw_{r_{cf}}\prod_c\psi_{icf}(a,r_{cf})
=\sum_rw_r\mathcal N(x_{if};\mu_{fr},\sigma_f^2)
\]

는 a/h와 무관하다. 여러 singleton column을 곱해도 같다. 따라서 exact class beliefs와 exact role posterior는 graph-only와 같다. **새 class-specific centroid representation이 singleton 한계를 해결하지는 않는다.** Context/source disjoint whole-node setting에서 source=1이면 여전히 feature utility를 검증할 수 없는 조건이다.

Literal paired BP에서도 degree-1 centroid cavity는 항상 prior다. 모든 c의 centroid→node message가 같은 scalar이므로 node feature field는 class-common이고 conditional node BP는 graph-only와 같다. Frozen graph-only q까지 쓰면 최종 prediction도 같은 것을 요구할 수 있다. Full finite-Bethe role score는 locally inconsistent beliefs 때문에 exact singleton cancellation을 자동 보장한다고 가정하지 않는다. Probability-level 검사를 따로 한다.

Class relabeling에서는 node labels와 latent centroid의 class index를 함께 permute한다. Shared class-independent hyperparameters라면 예측은 equivariant다. Column permutation은 grids, likelihoods, masks, stats도 함께 permute하면 prediction invariant다. 이는 rotation invariance나 임의의 feature semantics 추론을 의미하지 않는다. Observed-row reduction 순서 차이에 따른 float32 summation error는 허용 tolerance로 구분하고, row에 positional ID를 넣지 않는다.

## 7. Scalability and numerical constraints

위 feature updates의 시간·message storage는 각각

\[
O(T H E_X K R),\qquad O(H E_X K R+H K F R)
\]

이며 mean-zero graph message는 원래대로 full panel에서 \(O(HN^2K)\) storage와 \(O(THN^2K)\) work다. Whole-node observed count가 S이면 \(E_X=SF\)다. Feature tensor를 N×N에 붙이거나 K×R pair beliefs를 모두 저장할 필요가 없다. Grid를 11에서 31/41로 늘리면 이 feature 비용은 선형으로 증가한다. 하지만 전체 dense nonedge 비용은 그대로다.

Autograd의 24-step retained graph/intermediate 비용은 per-step message storage보다 크다. Frozen graph weights가 이를 없애지 않는다. CPU double correctness tests 후 실제 최댓값 N/K/source/R에서 forward+backward smoke를 통과해야 하며, 작은 parameter count를 낮은 메모리 비용으로 바꾸어 말하지 않는다. Windows 환경에서는 지시대로 sklearn을 torch보다 먼저 import한다.

\(\tau,\sigma\)의 positive floor와 필요시 upper bound, empty-column default, singleton scale fallback, dtype, clipping, `0 log 0` 처리는 v2에서 정한다. Source=1의 observed variance 0 때문에 prior scale/noise가 0이 되면 안 된다. Hyperparameter bounds를 final 결과를 보고 바꾸지 않는다.

## 8. Minimal correctness checks before training

### A. Discrete-model exact reference

작은 일반 heterogeneous tree에서 class variable K=3과 centroid variable R=3/5를 연결한다. One observed node와 그 K centroid leaves는 실제 완전 cell factorization의 tree다. 추가로 일부 class-gated edge와 graph edge가 섞인 generic tree로 양방향 cavity subtraction을 검사한다. Normal constants, nonuniform GH weights, context clamp, isolated centroid를 포함한다.

Exact enumeration과 비교할 때 같은 node-prior convention, \(A_h\), likelihood constants를 모두 포함한다. Damping 없는 BP는 tree diameter 이상, damping .5는 message residual이 충분히 작아진 상태에서 marginal/partition equality를 요구한다. 24회 damped iteration을 무조건 machine-precision exact로 요구하는 검사는 잘못이다.

그 다음 **실제 전체 factorization의 cyclic tiny case**를 별도로 검사한다. 예컨대 N=2, K=3, F=1, R=3이면 node/centroid states는 \(3^2\cdot3^3=243\)개다. R=5여도 1,125개다. 모든 centroid priors·all-class cell gates·선택한 graph factors를 열거해 exact conditional belief와 full q가 필요한 경우 exact role weight를 구한다. Finite BP의 오차와 residual을 보고하되 0을 강요하지 않는다. 이것이 quadrature error와 별개인 loopy inference error다.

### B. Continuous-Gaussian quadrature diagnostic with an analytic reference

고정 class에 할당된 n개 observation에서 \(\bar x\), \(SS=\sum_i(x_i-\bar x)^2\)라 하면 독립 Gaussian prior/likelihood의 centroid 적분은 해석적으로

\[
\log I_n=-\frac n2\log(2\pi)
-\frac{n-1}{2}\log\sigma^2
-\frac12\log(\sigma^2+n\tau^2)
-\frac12\left\{\frac{SS}{\sigma^2}
+\frac{n(\bar x-m)^2}{\sigma^2+n\tau^2}\right\},\quad n\ge1.
\]

\(I_0=1\)이다. Posterior variance는 \((\tau^{-2}+n\sigma^{-2})^{-1}\), mean은 그 variance에 \(m/\tau^2+\sum_i x_i/\sigma^2\)를 곱한 값이다. 이 식은 위 model 정의에서 완전제곱으로 직접 유도된다. Quadrature에서는 `logsumexp(log_w + sum_i log_likelihood_i)`로 비교한다.

n=1/5/20/100과 사전 고정한 \(\sigma/\tau\), off-grid sample mean을 포함하면 11 또는 더 큰 grid의 해상도 한계를 드러낼 수 있다. Source가 많을수록 posterior가 좁아져 prior-centered grid가 놓칠 수 있다. 공식 GH의 polynomial exactness는 이 Gaussian likelihood product의 exactness를 뜻하지 않는다. 이 검사는 label을 inference에 넣는 방법이 아니라 순수 numerical unit reference다. C를 discrete model로 정의했다면 continuous error가 있어도 discrete implementation 오류는 아니지만, continuous marginalization fidelity를 주장할 수 없다.

### C. End-to-end invariance, gradients and frozen inputs

No-observed와 source=1에서 graph-only와의 모든 query probability/role-weight 차이를 검사한다. Node/class/column permutation, hidden-value poison, observed-row order, context/source disjointness와 clamp를 확인한다. Finite feature gradients가 active likelihood와 centroid prior parameters까지 도달하는지, query loss가 graph message 경로를 지나 source-side feature parameters로 가는지 검사한다. Frozen graph parameter digest뿐 아니라 cached graph J/A와 graph-only q의 입력 identity도 확인한다. Query labels/true centroids를 poison해 forward 불변을 확인한다.

이번 review에서는 이 검증을 실행하지 않았다. Plan v2의 role protocol, grid semantics, hyperparameter bounds와 tensor/gauge convention을 먼저 잠근 뒤 구현 correctness 결과를 다시 검토하는 것이 적절하다.

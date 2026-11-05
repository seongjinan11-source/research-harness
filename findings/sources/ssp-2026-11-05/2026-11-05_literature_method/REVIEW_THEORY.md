# SSPd3-probe 강화: modular Bayes·semiparametric 문헌/수학 리뷰

2026-11-05. `RESEARCH_GOAL.md`와 최신 `count_support/DECISION_KO.md`를 읽었다. 아래 primary paper의 공식 proceedings·저자 arXiv 원문을 추가 검색했다. 새 방법 구현, fit, objective 호출, locked bank 변경은 하지 않았다. 기존 harness XLSX/추출본도 활용 가능한 자료이나, 이번 직접 근거는 별도 primary 문헌이다.

**핵심 판정:** “within-count 정보로 composition posterior를 보호하고, 이를 조건으로 count 함수의 posterior를 학습한다”는 접근은 **cut posterior의 직접적인 특수화**다. 수학적으로 자연스럽고 현재 병목에 맞지만 원리 자체의 신규성을 주장할 수 없다. Count 일반화를 덧붙여도 첫 모듈이 버린 between-count composition 정보가 복원되지는 않는다. 연구의 실질적 질문은 그 강건성·정보 손실의 대가를 어떤 관측으로 보완하고 실제 SSPd3-probe보다 나아지는가다.

## 1. 직접 확인한 선행과 읽은 범위

| Primary source | 확인 범위 | SSP에 대한 의미 |
|---|---|---|
| [Carmona & Nicholls, Semi-Modular Inference, AISTATS2020](https://proceedings.mlr.press/v108/carmona20a.html), [공식 PDF](https://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf) | Abstract, §2.1 Eq(1)(2), §3 Eq(4)–(6), §4의 framework 설명 | Full posterior의 feedback을 차단하는 cut과, feedback 정도를 조정하는 semi-modular 원리가 이미 있다. 아래 q(p)q(g|p) 구조의 가장 직접적인 선행이다. |
| [Frazier & Nott, Posterior risk of modular and semi-modular Bayesian inference, arXiv2301.10911v3](https://arxiv.org/html/2301.10911v3) | Abstract/intro, §2의 cut 정의와 likelihood factorization, §2.2 Assumption1의 iid 조건 | 완전 cut은 bias를 줄여도 variance를 키울 수 있다. 위험 비교 결과의 regularity/iid 조건을 작은 adaptive BO archive의 보장으로 옮길 수 없다. |
| [Tan, Nott & Frazier, Optimization-centric Cutting Feedback for Semiparametric Models](https://arxiv.org/abs/2509.18708), [최신v3 본문](https://arxiv.org/html/2509.18708v3) | 2025최초/2026-03v3 metadata, abstract/intro, §2 Eq(1)–(2)의 generalized/optimization formulation. §5 GP confounding 적용의 존재 확인; concentration 증명 전체 재검증은 하지 않음 | Parametric/nonparametric 모듈을 cut하여 prior/likelihood 오지정의 전파를 막는 연구와 GP 적용도 이미 있다. 단순 “cut+GP”나 semiparametric robustness를 새 발명으로 내세울 수 없다. Rényi/VI를 이번 선형 Gaussian 문제에 도입해야 한다는 근거는 아니다. |
| [Robinson1988 원문 PDF, 대학 호스팅](https://www.personal.soton.ac.uk/cz1y20/Reading_Group/mlts-2024s/week4/Robinson_1988_Root_N_Consistent_Semiparametric_Regression.pdf), [원 논문 식별 URL](https://www.jstor.org/stable/1912705) | 원문 서지와 introduction의 partially linear model·orthogonal projection·identification 논의. Publisher copy의 대학 호스팅본임을 확인 | Count를 nuisance로 둔 residualization은 classical partially linear estimation의 유한 categorical 특수형이다. Root-N/normality는 해당 가정 아래 결과다. |
| [Chernozhukov et al., Double/Debiased Machine Learning, 저자 원문](https://arxiv.org/pdf/1608.00060) | Abstract, §1.1의 partially linear model과 regularization-bias 설명, orthogonal score/cross-fitting 동기 | 좋은 nonlinear predictor가 target coefficient의 좋은 추정량을 자동으로 주지 않는다는 직접 선행. 그러나 단순 within-count ridge를 DML이라 부르거나 finite-sample BO coverage를 인용할 수는 없다. |
| [Lin, Khamaru & Wainwright, Semi-parametric inference based on adaptively collected data](https://arxiv.org/html/2303.02534v2) | Abstract, §1, §2.1–2.2의 sampling filtration, known selection probabilities, regressor support, weighted orthogonal score | Adaptive 자료에는 별도 explorability/weighting 조건이 필요하다. 이 논문의 주 regressor support는 basis vectors+zero여서 arbitrary multi-hot subset에 theorem을 그대로 적용할 수 없다. |

문헌에서 확인한 주장을 SSP의 새로운 실험 결과나 보장으로 확대하지 않았다. 특히 최신 semiparametric cut 논문이 있다는 사실만으로 subset BO의 size→composition 문제 분석 전체를 부정할 수는 없다. 다만 해결 원리의 attribution과 강한 대조군이 명확해야 한다.

## 2. 제안 구조는 정확히 어떤 cut인가

다음은 문헌의 cut 원리를 현재 Gaussian subset 모형에 적용한 독립 유도다. Fixed X, 고정 scales, iid Gaussian errors를 먼저 가정한다.

`y=Xp+Za+epsilon`, `epsilon~N(0,nu I)`.

Z는 observed count indicator다. C가 `null(Z')`의 orthonormal basis, D가 `col(Z)`의 orthonormal basis이면

`u=C'y=C'Xp+C'epsilon`,

`w=D'y=Fp+Ha+D'epsilon`, `F=D'X`, `H=D'Z`.

Fixed design에서는 두 noise projection이 독립이다. 따라서 u는 arbitrary observed-count a를 제거한 composition module, w는 count-level module이다. Count function에 prior를 주면 ordinary full posterior는 w의 count-function 가정을 통해 p도 갱신한다. 그 feedback을 자르는 분포는

`q_cut(p,a|u,w)=q(p|u) pi(a|p,w)`.

이는 Carmona–Nicholls Eq(2)의 parameter/data 이름을 바꾼 것이다. 새 Bayesian factorization 원리가 아니다. 첫 모듈에서 s/noise 등을 학습한다면 그 hyperparameters까지 보호 범위를 명시해야 한다. Stage2에서 공통 noise나 item scale을 다시 학습해 stage1에 돌려주면 완전 cut의 invariance가 깨질 수 있다.

**조건부 normalizer를 빠뜨리면 안 된다.** `pi(a|p,w)=L(w|p,a)pi(a)/m(w|p)`다. `q(p|u)L(w|p,a)pi(a)`만 곱하면 marginal p가 `m(w|p)`로 다시 갱신되므로 원하는 cut이 아니다. “All observations를 다시 사용”해도, fixed-scale Gaussian에서 g/a와 무관한 u likelihood는 a의 conditional posterior 정규화에서 소거된다. 이 소거를 확인하지 않은 채 같은 labels를 두 단계 objective에 중복 넣으면 다른 절차가 된다.

Count-table은 q(p|u)까지만 만들고 unsupported level을 예측하지 않았다. Cut+count GP는 그 p posterior를 유지하면서 second module이 미관측 count를 일반화하게 할 수 있다. **Composition class를 비선형으로 확장한 것은 아니며** p의 within-count 정보 부족을 고친 것도 아니다.

## 3. 정확 Gaussian cut covariance: 두 모듈을 독립으로 뽑으면 안 됨

First module `q(p|u)=N(m_p,S_p)`, full count-grid prior `a~N(m_a,G)`를 쓰자. H는 전체 count grid 중 관측된 count들을 w에 매핑하도록 확장할 수 있다. 고정 hyperparameters에서

`A=G H'(HGH'+nu I)^(-1)`,

`V_a=G−A H G`.

Cut joint draw는

`p~q(p|u)`,

`a=m_a+A(w−H m_a)−A F p+eta`, `eta~N(0,V_a)` independently of p.

따라서

`Cov_cut(p,a)=−S_p F'A'`.

Candidate gap을 `delta_x'p+delta_e'a`로 쓰면

`Var_cut(gap)=(delta_x−F'A'delta_e)' S_p (delta_x−F'A'delta_e)+delta_e' V_a delta_e`.

같은 K에서는 delta_e=0이어서 posterior mean뿐 아니라 gap variance도 q(p)만 남는다. Count prior를 바꿔도 fixed archive의 composition posterior를 보호한다는 정확한 성질이다. 반대로 p와 a의 marginal을 독립 sampling하거나 residual count GP를 fitted p 하나에서만 학습하면, 위 cross-covariance/uncertainty propagation을 잃는다. Plug-in variant는 별도 approximation control로 명시해야 한다.

동일한 구현을 count별 residual means로 할 수도 있다. Conditional on p, `rbar_K=mean(y−Xp|K)`의 noise variance는 nu/n_K다. **nu를 그대로 쓰지 않는다.** P uncertainty를 전파하면 count들 사이의 residual correlation과 p–a correlation이 생긴다. 수식은 고정 kernel/scales에서 정확하며, count GP hyperparameters를 data로 다시 고르면 그 uncertainty까지 자동 포함되는 것은 아니다.

이 유도는 cut literature와 Gaussian conditioning의 특수화로서 correctness 타깃이다. 기존 SSP appendix의 matrix retention/leakage처럼, 다시 이름 붙이는 것만으로 새 theorem 기여가 생기지 않는다.

## 4. 식별가능성: 두 모듈의 한계를 각각 적어야 함

Composition의 within-count information matrix는

`M=sum_K sum_{i:K_i=K}(x_i−xbar_K)(x_i−xbar_K)'/nu`.

필요한 contrast direction이 M의 nullspace에 있으면 보호된 모듈은 그 방향을 데이터에서 추정하지 못한다. Proper prior는 finite posterior를 주지만 정보를 만들지 않는다. 모든 count가 singleton이면 M=0이다. Count GP가 w를 잘 설명해도 **cut을 유지하는 한** 이 정보를 stage1에 돌려줄 수 없다. Full joint의 유용한 between-count 정보도 함께 버린다는 대가는 불가피하다.

Count 쪽에서는 arbitrary g에 대해 observed support 밖의 값을 식별할 수 없다. 관측된 모든 K에서 같고 미관측 K에서만 다른 두 함수는 현재 data likelihood가 같다. 따라서 안전한 unknown-count 최적화에는 추가 smoothness/RKHS norm/shape/bounded discrepancy 같은 가정이나 실제 query가 필요하다. GP를 붙이는 것은 그 일반화 가정을 선택하는 것이지 arbitrary nonlinear count 효과의 보편적 안전 보장을 주는 것이 아니다. 관측 stress shapes가 monotone이었다고 이후 방법이 monotonicity를 공짜로 가정해서도 안 된다.

또한 true composition이 linear/common across K라는 가정이 틀리면 within-count p가 그 자체로 misspecified다. Cut은 잘못된 첫 모듈을 보호할 수도 있다. 좋은 nuisance 처리만으로 nonlinear composition의 식별이 해결되지 않는다.

## 5. Count support 구매와 composition 정보 구매는 다른 query다

새 x를 기존 count K에 추가할 때 old sample 수를 m이라 하면 within-count scatter의 update는

`Delta M=[m/(m+1)]*(x−xbar_K)(x−xbar_K)'/nu`.

처음 보는 count이면 m=0이므로 **first query의 composition-module 정보 증분은0**이다. 반면 count level/generalization은 그 첫 label로 크게 바뀔 수 있다. 동일 K의 새로운 composition을 비교하는 query는 p를 배우고, 새로운 K의 첫 query는 count support를 산다. 이 두 목적은 같은 탐색 보너스로 자동 정렬되지 않는다.

위 scatter identity는 classical sufficient-statistic 업데이트이며 새 공식 자체가 기여는 아니다. 그러나 현재 관찰된 “table의 composition은 좋으나 online 선택은 나쁨”을 해결하려면 이 두 정보 이득이 실제 의사결정에 어떻게 연결되는지 구분하는 것이 핵심이다. 강한 기준선 SSPd3-probe와 같은 query budget에서 입증해야 한다.

## 6. iid/fixed design과 adaptive sampling을 구별

Fixed 또는 errors와 독립인 design에서는 Gaussian projection noise의 독립성으로 위 모듈 해석이 직접 성립한다. Adaptive BO에서는 realized X가 과거 noise를 담으므로 C(X)'epsilon와 D(X)'epsilon이 **X에 조건부로 독립 Gaussian sampling variables**라고 바로 주장할 수 없다. Group demeaning이 algebraic하게 count shifts를 제거한다는 사실과 frequentist unbiasedness/coverage는 다르다.

반대로 adaptive이면 모든 Bayesian likelihood가 무효라는 주장도 틀리다. Known predictable policy가 history에만 의존하고 추가 unknown-parameter factor를 만들지 않으면 full observed-path likelihood는 product of observation likelihoods로 Bayesian 갱신할 수 있다. 그 likelihood를 algebraic modules로 나누어 cut을 정의하는 것 역시 가능하다. 하지만 그 reduced module의 inference·repeated REML·data-dependent contrast normalization이 calibrated inference 또는 sequential Bayes coherence를 자동 보장하지는 않는다. Frazier–Nott의 일반 module factorization도 허용 범위가 넓으나 해당 risk 정리의 iid/regularity 조건은 별도로 확인해야 한다.

Lin–Khamaru–Wainwright는 adaptive semiparametric inference에 known selection probabilities와 explorability, weighted orthogonal scores를 두는 가까운 선행이다. 주어진 theorem의 regressor support가 multi-hot subset과 다르므로 인용만으로 해결되지 않는다. Deterministic argmax policy에서 propensity를 사후에 상수로 놓거나, random iid cross-fitting을 붙인 것으로 adaptive validity가 생긴다고 설명하면 안 된다.

## 7. 가장 싼 반증과 강한 대조

다음은 제안일 뿐 실행하지 않았다. 새 bank/계획 없이 기존 결과를 확인 bank로 다시 쓰지 않는다.

1. **0-objective numerical falsification:** 고정 X와 임의 count shift에서 q(p|u), scales, same-K posterior gap mean/variance가 보존되는지 검사한다. Exact Gaussian joint draw와 위 analytic covariance를 비교한다. Conditional normalizer를 누락하거나 independent p/a sampling한 구현은 여기서 탈락한다.
2. **고정 archive의 가장 싼 실질 반증:** 새 disjoint synthetic fixtures에서 같은 count kernel/scale-fit budget의 (a) full joint, (b) exact cut, (c) plug-in two-stage를 비교한다. Composition error뿐 아니라 미관측-count latent prediction/ranking을 동시에 본다. Cut이 composition을 보호해도 count regret가 여전히 table 수준이면 “generalization 연결로 online 병목을 풀었다”는 가설이 약해진다. In-span cubic에서도 불필요한 cut 손실을 보고한다.
3. **정보 불가능성 fixture:** 모든 count singleton 또는 required contrast가 within-K nullspace에 있는 X를 둔다. Cut에서 p가 데이터로 회복되면 구현의 feedback leak이거나 잘못된 평가다. 관측 support에서 동일하고 밖에서 다른 두 g fixture로 arbitrary-shape safety의 불가능성도 확인한다.
4. **그다음에만 bounded fresh online:** count_support에서 이긴 SSPd3-probe를 반드시 strong baseline으로 둔다. Exact cut의 좋은 composition만으로 성공 처리하지 말고 actual noisy-incumbent regret·count coverage·query 비용을 함께 평가한다. 같은 GP를 full joint한 baseline보다 유용한 regime와 무해한 regime가 있는지 구별한다.

최소한 1과2를 통과하지 못하면 복잡한 semi-modular feedback parameter/새 neural model/대규모 BO 실험으로 넘어갈 이유가 약하다. Semi-modular tuning은 classical bias–variance 해법의 후보이지만 이전 bank에서 eta를 sweep해 확인 성공으로 보고하면 안 된다.

## 추천 연구 주장

“기존 cut/partially-linear 원리로 subset의 within-count composition 추정을 size-shape misspecification에서 보호하되, count function에 대한 일반화와 그 불확실성을 연결한다. 새 count의 첫 관측과 within-count 비교가 서로 다른 정보를 사는 것을 명시하고 실제 query 비용에서 그 절충을 검증한다.”

이것은 수학적으로 정당하고 사용자 목표에 맞는 방향이다. **Cut+GP 자체는 새 해결 원리가 아니다.** 논문의 추가 가치는 subset BO에서 어떤 feedback을 끊어야 하고 어떤 정보를 다시 관측해야 하며, 그 결과가 강한 기존 SSPd3-probe보다 언제 실질적으로 나은지를 보여주는 데서 나와야 한다. 현재 단계에서 superiority나 novelty는 가설이다.

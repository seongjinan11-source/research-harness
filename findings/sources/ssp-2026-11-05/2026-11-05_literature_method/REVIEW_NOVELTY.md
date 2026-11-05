# SSPd3-probe 발전을 위한 직접 선행과 남는 문제

2026-11-05. RESEARCH_GOAL과 최신 count-support DECISION_KO를 읽고 primary papers를 웹에서 확인했다. 목표는 size/composition 문제와 실질적 해결의 연결이며 standalone kernel 발명을 요구하지 않는다. 새 실험·학습·잠긴 원고/결과 수정은 하지 않았다. 아래 여섯 편은 서로 다른 측면에서 가까운 선행이다. 모두 SSP의 정확한 문제를 이미 해결했다는 뜻은 아니다. 문헌 검색만으로 최초성을 확정하지 않는다.

## 판정

**Variable-size BO, nuisance partialling-out, regularization-induced confounding, trend/residual 분리, subproblem budget allocation은 이미 연구됐다.** 따라서 그 구성요소를 조합한 것만으로 강한 신규성을 주장하기 어렵다. 그러나 이 사실은 subset BO에서 size prior/shape 오지정이 composition을 오염하고 count conditioning으로는 복구되지 않는 구체적 메커니즘, 그리고 그 메커니즘을 교정한 의사결정 증거까지 중복이라는 뜻이 아니다.

현재 가장 구체적인 gap 후보는 **size nuisance를 잘못 지정했을 때 composition을 보호하면서, count 간 유용한 일반화를 버리지 않고, 제한된 support에서 실제 incumbent를 개선하는 조건과 방법**이다. 최신 실험은 이 tradeoff가 실제로 존재함을 보여주지만 해결이 완료되지는 않았다. Table_probe가 registered spread 비교에는 이기고 SSPd3_probe에는 진 것은 그 간극을 정확히 드러낸다.

## 1. Buathong, Ginsbourger & Krityakierne (2020)

**Kernels over Sets of Finite Sets using RKHS Embeddings, with Application to Bayesian (Combinatorial) Optimization**, AISTATS2020.

- [공식 proceedings](https://proceedings.mlr.press/v108/buathong20a.html), [공식 원문 PDF](https://proceedings.mlr.press/v108/buathong20a/buathong20a.pdf).
- 실제 확인: abstract/introduction, §2.1–2.2의 domain·mean embedding·DS/DE kernel 정의. 전체 실험/증명 재현은 하지 않았다.
- 이미 알려진 것: §2.1은 domain을 whole power set, fixed-p subsets 또는 모든 nonempty finite subsets로 허용한다. Normalized empirical RKHS embedding과 deep embedding kernel을 제시한다.
- 우리 주장에 주는 제약: “기존 set BO는 size를 고정해서 variable cardinality를 다루지 않았다”는 포괄 주장은 틀리다. Normalization도 size/composition 모델링의 실제 선택이므로 ‘모든 기존 kernel은 count 정보가 없다’고 묶으면 안 된다.
- 남는 gap: 확인한 model sections는 SSP의 archive-dependent retention 및 same-count leakage correction을 제시하는 분석이 아니다. Set input을 표현할 수 있다는 능력과 그 특정 오류를 진단·교정했다는 주장을 구분해야 한다.

## 2. Pelamatti et al. (2020)

**Bayesian optimization of variable-size design space problems**.

- [저자 arXiv record](https://arxiv.org/abs/2003.03300), [원문 PDF](https://arxiv.org/pdf/2003.03300).
- 실제 확인: abstract, §3 budget-allocation strategy, §3.1–3.3의 predicted optimum 비교·subproblem elimination·예산 배분·각 subproblem BO. 49-page 전체 증명/실험은 재검증하지 않았다.
- 이미 알려진 것: 구조를 결정하는 discrete variables를 고정해 fixed-size subproblems로 나누고, predicted best/worst/nominal optima와 dimension을 이용해 budget을 나눠준다. 원문은 별도의 cross-structure covariance 접근도 제안한다고 설명한다.
- 제약: “size/subspace를 먼저 고르고 내부 BO에 예산 배분” 자체는 신규 원리가 아니다. Count별 bandit allocation이나 confidence-based count elimination을 제안할 때 반드시 대조해야 한다.
- 남는 gap: 이 논문의 size는 활성 변수의 수·종류와 constraints가 달라지는 hierarchical design이다. 고정 ground set의 K=|x|, 공유 item contrasts, size-induced same-K contamination과 동일하지 않다. 그 통계적 연결과 실제 선택 회복은 따로 입증해야 한다.

## 3. Papenmeier, Nardi & Poloczek (Bounce)

**Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces**, NeurIPS2023; 확인한 arXiv PDF는 v2(2024).

- [원문](https://arxiv.org/pdf/2307.00618).
- 실제 확인: introduction/contributions, §4.6의 BODi/COMBO optimum-location sensitivity와 관련 Appendix H 부분. 원 논문의 모든 benchmark 수치를 재계산하지 않았다.
- 이미 알려진 것: 조합 BO가 특정 optimizer 구조와 representation에 편향되어 benchmark를 바꾸면 성능이 저하되는 문제를 분석하고 nested embeddings를 제시한다.
- 제약: “조합 BO의 구조적 bias를 처음 발견했다”는 넓은 모티브는 선취된다. Extreme-cardinality optimum 근처 성능 차이를 새 phenomenon 전체로 주장할 수도 없다.
- 남는 gap: Optimum-location/representation bias와 SSP의 common-item shrinkage·size-shape misspecification에 의한 finite-archive leakage는 다른 메커니즘이다. SSP는 size-only 변형과 동일-K 후보 대조로 차이를 보여야 한다.

관련 [BODi 원문](https://proceedings.mlr.press/v206/deshwal23a/deshwal23a.pdf)의 §3 dictionary construction도 확인했다. Extreme anchor가 있으면 Hamming distance가 K 또는 n−K를 표현한다. 그것은 count 표현의 잠재력이며 모든 실제 dictionary에서 anchor가 보장된다는 뜻은 아니다. 따라서 BODi 전체에 size 정보가 없다는 주장은 피한다.

## 4. Hahn, Carvalho, He & Puelz

**Regularization and confounding in linear regression for treatment effect estimation**, 확인한 원문 arXiv1602.02176.

- [저자 record](https://arxiv.org/abs/1602.02176), [원문 PDF](https://arxiv.org/pdf/1602.02176).
- 실제 확인: introduction, §2의 ridge-posterior bias 식(2.3), §2.1 reparameterized joint model, §3의 비교 대상 설명. Causal identification/전체 simulation은 재검증하지 않았다.
- 이미 알려진 것: 올바른 likelihood를 써도 shrinkage가 관련 nuisance coefficients를 줄여 관심 계수에 bias를 만들 수 있고, reparameterized simultaneous model과 별도 prior로 이를 완화한다.
- 제약: Regularization-induced contamination 및 correlated design에서 prior 분리를 통해 교정한다는 일반 원리를 새 발견이라고 주장할 수 없다. Count feature는 item의 합이므로 단순한 omitted-feature 설명만으로도 부족하다.
- 남는 gap: Treatment estimation과 subset optimization은 다른 문제다. Count의 특수한 중복/대칭 구조, archive에 따른 크기·같은-K ranking의 distortion, 그리고 noisy incumbent까지 연결하는 결론은 별도다.

## 5. Robinson (1988): flexible nuisance의 가장 직접적인 통계 선행

**Root-N-Consistent Semiparametric Regression**, Econometrica56(4),931–954.

- [원 출판 record/DOI](https://doi.org/10.2307/1912705), [원 논문 대학 미러 PDF](https://www.personal.soton.ac.uk/cz1y20/Reading_Group/mlts-2024s/week4/Robinson_1988_Root_N_Consistent_Semiparametric_Regression.pdf).
- 실제 확인: 원문 pp931–936, §1–2의 partially linear model, misspecified nuisance bias, response와 regressors 양쪽의 conditional-mean residualization. 이후 점근 정리의 모든 조건·증명을 재검증하지 않았다.
- 이미 알려진 것: E[Y|X,Z]=β'X+θ(Z)에서 θ의 오지정이 β를 왜곡할 수 있으며, Y−E[Y|Z]와 X−E[X|Z]를 이용해 slope를 추정한다.
- 제약: Count-table의 “Y와 X를 count 내에서 함께 demean하여 composition을 보호한다”는 원리는 이 partially linear/partialling-out 계열에 직접 놓인다. 이를 최초의 robust composition separation이라고 할 수 없다.
- 남는 gap: Z=K=1'X이면 residualized X의 합은0이라 common direction은 singular다. Zero-sum contrast gauge와 적절한 rank 조건이 필요하다. IID·점근 root-N 보장은 finite adaptive BO에 자동 이전되지 않으며, observed count에서 nuisance 제거가 unseen count 예측이나 decision regret를 해결하지도 않는다.

Robinson과 현재 table의 REML Gaussian shrinkage가 같은 알고리즘이라는 뜻은 아니다. 구조적 nuisance 제거 원리가 가까우며, 그 위의 regularization/variance fitting·query 선택·유한 예산을 별도로 평가해야 한다.

## 6. Plumlee & Joseph (2018)

**Orthogonal Gaussian Process Models**, Statistica Sinica28,601–619.

- [공식 저널 원문](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n23.pdf), DOI10.5705/ss.202015.0404.
- 실제 확인: §1의 mean/GP identifiability 예제, §2.1 Lemma1/Theorem1의 projected covariance, §2.2의 기본 성질. 전체 applications는 재현하지 않았다.
- 이미 알려진 것: Parametric trend와 GP residual이 likelihood에서 구별되지 않는 문제, 관측 지점만의 orthogonality와 전체 domain에서의 orthogonality 차이, covariance conditioning으로 trend와 stochastic part를 분리하는 방식.
- 제약: Trend/residual 분리나 covariance projection 자체는 새 원리가 아니다. Prior orthogonality를 얻었다고 posterior independence나 finite archive의 unbiased decision이 생겼다고 주장할 수 없다.
- 남는 gap: 원문은 subset count/support를 다루는 BO policy가 아니다. Size prior correction과 within-K decision recovery, shape misspecification 하의 acquisition 효과를 이 원리만으로 보장하지 않는다. 기존 nonlinear projection gate 실패도 그대로 유지해야 한다.

## 보조 baseline 및 미확인 항목

[BOCS 공식 원문](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf)의 §3.1 horseshoe sparse polynomial model도 확인했다. 이것은 shrinkage-based combinatorial surrogate의 직접 대조다. 그러나 특정 Gaussian linear retention 식을 모든 horseshoe fit에 동일하게 적용할 수는 없다.

이전 조사에서 Fukushima/Kurita2025 **Differentiable Subset Selection for Cardinality-Constrained Bayesian Optimization**의 저자 PDF 검색 abstract는 exact-k/Gumbel-Top-k/STE와 Matérn subset fixation를 보고했다. [PDF 주소](https://d197for5662m48.cloudfront.net/documents/publicationstatus/272382/preprint_pdf/c66b914af0552e7e11c1b10256662738.pdf), DOI10.36227/techrxiv.175432916.60787000/v1. 직접 fetch가 실패하여 kernel 식·구체적 고착 원인은 미확인이다. Exact-k sparse continuous design 설명이라 variable-count loss와 곧바로 같지 않다. 관련 항목으로 추적하되 fatal prior art나 검증 완료 문헌으로 세지 않는다.

## 현재 결과가 허용하는 구체적 gap

1. **진단의 gap:** Cardinality-sensitive representation의 존재와, archive/prior가 실제 size effect를 보존하는 것은 다르다. Original retention/leak theory와 size-injection controls는 이 차이를 정량화한다. 일반 regularization bias와 다른 subset-specific 함의를 정확히 쓰는 것이 필요하다.
2. **해결의 gap:** Flexible count nuisance를 완전히 자유롭게 두면 composition은 보호되지만 between-count 정보와 unseen K 예측을 잃는다. Count-support 실험에서 table composition은 좋아도 SSPd3_probe가 incumbent에서는 더 좋았다. 이 tradeoff는 미해결이며 새로운 estimator 이름보다 실제 연결을 해결하는 것이 중요하다.
3. **정책의 gap:** 단순 spread나 일정 간격 distance probes 자체는 classical이다. 언제 count support query가 필요한지, 언제 충분한 일반화를 이미 하고 있어 query가 낭비되는지 판별·활용하는 조건이 후보 gap이다. Conditional Gaussian SD가 misspecification를 놓친 current H1은 이 문제를 뒷받침하지만 universal uncertainty failure 정리는 아니다.
4. **증거의 gap:** 신뢰할 수 있는 기여는 같은 information/candidates에서 nuisance fitting 방식의 차이, 같은 policy에서 model 차이, 같은 model에서 query 차이를 각각 보이고 실제 incumbent와 composition 회복을 함께 평가해야 한다. SSPd3_probe가 현재 가장 강한 observed comparator이므로 이를 빼고 새 방법을 검증하면 해결의 필요성이 설득되지 않는다.

어떤 regularity도 없는 임의 g(K)에 대해 미관측 K의 값을 알아내겠다는 목표는 성립하지 않는다. Shape smoothness/variation/monotonicity, 필요한 support 관측 또는 제한된 claim 중 무엇에 기대는지 명시해야 한다. 이는 현재 확보한 새 하한 정리라는 뜻이 아니라, 해결 가정의 기본 제약이다.

## 엑셀 활용 기록

원본은 `C:/Users/seongjin/Desktop/code/gts_pfn/research-harness/out/research_harness.xlsx`이며 root가 fresh extraction을 담당한다. 이 reviewer는 기존 [read-only JSON](C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-04_method_candidates/harness_papers_and_claims.json)을 다시 열어 Papers5/46/47의 전체 요약·접점·URL을 확인했다. 기존 review에서 Claims24–27,286–298도 실제 읽었으며 이번 해석에 재사용했다.

- Papers46 **Spatial Context or Memory?**, Claims286–293: 같은 정보 아래의 reference로 information shortage와 fitting error를 구별하는 평가 원리. Current support tradeoff에 관련되나 직접 subset BO prior art는 아니다.
- Papers5 **Optimal Inference in Contextual Stochastic Block Models**, Claims24–27: joint inference 자체의 독창성을 과장하지 않는 대조 관점. 새로운 subset estimator의 근거로 직접 이전하지 않는다.
- Papers47 **Semi-supervised Gaussian Process Classifiers**, Claims294–298: covariance/evidence 수정의 classical 성격과 geometry 가정의 필요성. Constraint graph가 있다는 이유만으로 coefficient smoothness를 가정하지 않는다.

이 행들은 **로컬 문헌 노트**로 활용했으며 이번 six-paper primary 검증과 구분한다. Workbook은 PFN/graph/time-series 중심이어서 subset BO 최초성을 판정하는 comprehensive survey가 아니다.

## Contribution 문장과 최종 권고

현재까지 정당한 표현:

> We study how cardinality effects are mislearned in subset Bayesian optimization and can contaminate composition decisions even within a fixed cardinality. We quantify this failure for the original surrogate, test joint nonlinear-size corrections with controlled injections, and examine the tradeoff between protecting composition estimates and acquiring useful count support online. The modeling and design ingredients are classical; the contribution is the subset-specific diagnosis, its decision consequences, and evidence about when candidate corrections succeed or fail.

SSPd3에 정해진 주기로 distance probes를 붙였다는 사실만으로 충분한 method novelty를 얻었다고 판단하지 않는다. 반대로 자체 kernel이 없다는 이유로 논문 전체를 기각하지도 않는다. 가장 설득력 있는 발전 방향은 **prior/shape 오지정으로부터 composition을 보호하는 원리와, 실제 count 의사결정에 필요한 일반화·관측을 함께 연결하는 것**이다. 위 선행이 그 연결 전체를 해결했다고 확인하지는 못했지만, 아직 해결하지 않은 gap을 이미 달성한 contribution으로 표현하면 안 된다.

기존 matrix appendix·SSPd3는 출발점이다. Confirmation40의+1.31%, projection 실패, size-injection 성공, table_probe의 primary 통과와 SSPd3_probe 열세를 모두 보존한다. 새로운 candidate의 정당한 확인은 현재 development bank 밖에서 이루어져야 한다.

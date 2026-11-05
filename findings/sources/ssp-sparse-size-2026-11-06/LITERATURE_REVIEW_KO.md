# Sparse polynomial surrogate와 size 구조: 문헌 정리

2026-11-06. 사용자가 제시한 다섯 논문을 실제 원제목·전원 저자·연도·본문으로 확인하고, BOCS 원 논문과 Factorization Machines 원 논문을 연결했다. 기존 research-harness Papers/YAML 72편을 대조했으며 이 일곱 항목은 기존 목록에 없었다. 새 성능 실험·objective 호출·training은 하지 않았다.

## 문헌이 실제로 말하는 것

| ID | 확인한 원문·위치 | 확인한 내용 | SSP에서의 역할과 경계 |
|---|---|---|---|
| BOCS | Baptista·Poloczek, ICML 2018, §3.1–3.2 | Main/pairwise monomial의 horseshoe prior, posterior draw acquisition, known regularizer의 별도 처리 | Sparse polynomial family의 출발점. Size를 표현 못 하는 모델이라는 주장은 틀리며 isotropic Gaussian 정리의 직접 적용 대상도 아님 |
| BOCSRandom | Morita·Nishikawa·Ohzeki, JPSJ 2023, Eq.(3)–(7) 및 결론 | Dense SK coupling을 이유로 normal prior를 선택. 주 실험은 random duplicate-rejection postprocessing | Dense 구조와 prior 선택을 연결하는 명시적 동기. Horseshoe 대 normal의 matched size 실패 실험은 아님 |
| BOCSPSR | Deshwal·Belakaria·Doppa, arXiv 2020, §3·§4 | BOCS의 저차 모델 가정과 SDP 확장성 한계를 논의하고 PSR acquisition optimizer 제안 | Prior와 acquisition solver를 구별할 근거. Size/composition 누출 연구는 아님 |
| NonOrthogonalSet | Wendler·Amrollahi·Seifert·Krause·Püschel, AAAI 2021, §3·§5.2/Table2 | Facility-location preference 함수의 model4 sparsity와 WHT density를 대조 | Basis-dependent sparsity의 직접 선행. 모든 cardinality 함수가 최대 dense하다는 명제를 보인 것은 아님 |
| FABOLAS | Klein·Falkner·Bartels·Hennig·Hutter, AISTATS 2017, §3.1 Eq.(6) | Hyperparameter kernel과 training-size basis covariance의 곱을 사용 | 명시적 size basis 설계는 기존 아이디어. Training fidelity와 선택 subset의 composition은 다른 문제 |
| WavelengthBOCS | Kanebako·Koshikawa·Hitomi·Tanaka·Chiba·Mori·Ohzeki, arXiv 2026, §2.2 Eq.(7)–(8)·§5 | Horseshoe sparse quadratic surrogate, Thompson sampling, QUBO SA/QA로 wavelength subset 선택. 실제로 긍정 결과를 보고 | 최근 활용 사례. Size-prior 실패의 증거나 분야 전체의 보급률 통계로 인용할 수 없음 |
| FactorizationMachines | Rendle, 2010, §III Eq.(1)–(3) | Pair coefficient를 latent-vector 내적으로 공유 | 독립 coefficient sparsity와 다른 구조적 sharing 대조. 작은 rank는 함수공간도 바꿈 |

Wavelength 논문의 실제 제목은 **Robust Wavelength Selection for Partial Least Squares Sugar Content Estimation Using Combinatorial Bayesian Optimization**이며 arXiv:2607.27645, 2026-07-30 v1이다. 본문은 nested validation/independent test 부족과 prediction 결과를 보고 정한 wavelength 해상도를 향후 과제로 직접 인정한다. 해당 논문은 CBO가 잘 작동한 사례로 기록했으며 실패 사례로 바꾸지 않았다.

## 연결 가설에서 정확히 고칠 부분

### 1. Size 함수가 언제 monomial에서 dense한가

Boolean cube 전체에서 K=Σ_i x_i라 하자. r차 binomial basis는

    C(K,r) = Σ_{A:|A|=r} ∏_{i∈A} x_i

다. 유한한 count 함수 g(0),…,g(n)는 Newton 전개로

    g(K) = Σ_{r=0}^n Δ^r g(0) C(K,r)

라고 쓸 수 있다. 따라서 해당 finite-difference 계수가 0이 아닌 차수 r에서는 모든 C(n,r)개 monomial coefficient가 같은 값을 갖는다. 이것은 이 정리의 analyst 수학적 해석이며 위 논문이 그대로 제시한 SSP 정리라고 인용하지 않는다.

예를 들어 g(K)=aK+bK²이면 item 계수는 a+b, 모든 off-diagonal pair 계수는 2b다. b≠0이면 pair block이 dense하다. 하지만 g가 상수면 intercept 하나로 충분하고, 선형이면 pair block은 0이다. 특별한 a,b 조합에서는 main block도 사라진다. 그러므로 **“비영인 count 성분은 그 차수의 monomial block 전체에 퍼진 tied 구조다”**가 정확한 표현이다. “모든 size 함수가 가장 dense하다”는 표현은 쓰지 않는다. 이 전개는 새로운 최초성 주장이 아니다.

### 2. Dense tied 구조가 horseshoe에 반드시 손실되는가

아직 가설이다. Sparse coefficients에 맞춘 prior와 여러 작은 계수에 흩어진 common effect가 맞지 않을 가능성은 있다. 그러나 native horseshoe의 heavy tail·global/local adaptation·관측 구조가 이를 얼마나 보완하는지는 별도로 확인해야 한다. “가장 강하게 억누른다”는 순위 주장도 검증하지 않았다.

Conditional coefficient covariance는 σ²τ²diag(β_j²)이며 일반적으로 isotropic이 아니다. Marginal horseshoe에는 유한한 계수 variance가 없으므로 기존 Gaussian variance-budget 식을 그대로 대입하지 않는다. 이 한계는 [BOCS 원문](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf)과 공식 코드 감사에 근거한다.

### 3. FM rank-1 예의 적용 범위

Standard real-valued 2-way FM에서 모든 v_i=v로 놓으면 pair coefficient는 ||v||²≥0이다. 따라서 **양의** 공통 pair coefficient는 factor 차원 1로 표현할 수 있다. 이 예만으로 음의 quadratic count curvature까지 표현한다고 할 수는 없다. 입력/목적함수 부호와 FM의 실제 parameterization을 명시해야 한다. 표현 가능성과 제한된 관측에서 실제로 학습하는지는 여전히 다른 질문이다.

### 4. 네 arm 비교는 유용하지만 원인을 자동 식별하지 않는다

Horseshoe BOCS, normal-prior quadratic, FM, horseshoe+count 비교는 좋은 후보군이다. 다만 FM은 rank·함수공간·optimizer까지 달라질 수 있고 normal판도 normalization·intercept·noise·추론 방식이 다를 수 있다. Horseshoe만 낮은 성능을 보였다는 이유만으로 size-prior가 유일한 원인이라고 결론내리지 않는다.

Prior 효과를 보려면 같은 monomial basis·likelihood·archive·intercept·response scale·known costs·후보·budget에서 horseshoe/normal을 비교해야 한다. Count 항 추가도 같은 함수공간에서 count 방향의 prior만 바꾼 것인지 확인해야 한다. Native second-order 모델에 cubic count 항을 더하면 함수공간과 QUBO acquisition 최적화 문제가 함께 바뀔 수 있다. FM은 별도의 구조적 sharing 대조로 해석하는 편이 정확하다. 이 항목은 미래 실험의 설계 조건이며 이번에 실행한 결과가 아니다.

## 알려진 비용을 학습시키지 않는 간단한 대조

공식 BOCS Ising 구현은 λK를 surrogate label에서 제외하고 acquisition/report에서 정확히 더한다. 따라서 그 알려진 비용을 surrogate가 잃는다는 설명은 native 코드와 다르다. KL 내부의 unknown size trend는 별도의 질문으로 남는다. Contamination은 추가 λK만 분리하며 기본 처리비 K는 label에 들어가므로 **모든 알려진 비용을 정확히 residualize**하는 단순 대조도 필요하다.

근거: official commit `fef0d4e34e376e8bb0dae9955d70c2155530b9eb`의 [BOCS.m](https://github.com/baptistar/BOCS/blob/fef0d4e34e376e8bb0dae9955d70c2155530b9eb/algorithms/BOCS.m#L68-L81), [contamination_prob.m](https://github.com/baptistar/BOCS/blob/fef0d4e34e376e8bb0dae9955d70c2155530b9eb/test_problems/ContStudy/contamination_prob.m#L18-L22). 원본의 MATLAB/Python basis, Gibbs draw, intercept 처리도 고정해야 한다.

## 직접 선행 검색의 범위

기존 하네스 72편과 관련 구조화 Claims를 검색하고, 제시된 다섯 제목 및 다음 질의로 추가 탐색했다.

- `Random Postprocessing for Combinatorial Bayesian Optimization` / `Morita normal prior`
- `Scalable Combinatorial Bayesian Optimization with Tractable Statistical Models`
- `Learning Set Functions that are Sparse in Non-Orthogonal Fourier Bases`
- `Robust Wavelength Selection for PLS Using Combinatorial Bayesian Optimization`
- `Fast Bayesian Optimization of Machine Learning Hyperparameters on Large Datasets`
- `cardinality horseshoe Bayesian optimization`
- `size composition sparse polynomial Bayesian optimization`
- `cardinality BOCS prior`
- `Steffen Rendle Factorization Machines 2010`

마지막 세 연결 질의에서는 이 사슬을 직접 확인하는 논문을 식별하지 못했다. BOCS 약자가 다른 분야의 코드명으로 검색되는 등 잡음도 있었다. **한정된 검색에서 직접 선행을 찾지 못했다**고만 기록한다. 체계적 전수 조사·우선권 확정·관련 선행의 부재 증명은 아니다. 일반적인 basis dependence, shrinkage와 dense signal, structured prior 자체는 이미 알려진 내용이며 이를 합친 설명만으로 비자명성을 확정하지 않는다.

## 지금 원고에 쓸 수 있는 위치

가능한 연결 가설은 **“count의 tied 구조가 독립 monomial shrinkage 아래 어떻게 추정되며, 그 영향이 same-count 결정까지 전달되는가”**다. 새 kernel 발명을 목표로 바꾸지 않고 문제→prior→추정→선택 회복의 연결로 판단한다. Sparse polynomial family에서 중요한지는 native 실험으로 검증할 질문이다. 확인한 최근 응용 하나로 CBO 전체의 주류·보급률·구조적 최약점이라는 표현을 확정하지 않는다.

직전 corrected linear cycle의 반대 증거도 유지한다. 새24 tomography에서 d0 대 SSPd3 same-K regret 개선은 S에서 +0.006387이었지만, 예상한 양의 정렬·실제 오차 부호와 S−U 조절 효과는 실패했다. U에서도 개선이 있었다. BO archive의 양의 패턴은 기술 보고다. 이 결과를 horseshoe 실패의 전제나 모든 커널의 중요성 부재로 일반화하지 않는다. 원문 snapshot은 이 문서와 함께 `LINEAR_RESULT_KO.md`에 보존하고, 원래 파일은 `C:/Users/seongjin/Desktop/code/ssp-share/research_cycles/2026-11-06_linear_mechanism/DECISION_KO.md`다.

## 확인한 1차 자료

- [BOCS proceedings metadata](https://proceedings.mlr.press/v80/baptista18a.html)와 [본문](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf)
- [Morita metadata](https://arxiv.org/abs/2309.02842), [v2 본문](https://arxiv.org/html/2309.02842v2)
- [Deshwal metadata](https://arxiv.org/abs/2008.08177), [본문](https://arxiv.org/pdf/2008.08177)
- [Wendler metadata](https://arxiv.org/abs/2010.00439), [공식 AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/17232/17039), [arXiv 본문](https://arxiv.org/pdf/2010.00439)
- [FABOLAS proceedings metadata](https://proceedings.mlr.press/v54/klein17a.html), [본문](https://proceedings.mlr.press/v54/klein17a/klein17a.pdf)
- [Wavelength metadata](https://arxiv.org/abs/2607.27645), [v1 본문](https://arxiv.org/html/2607.27645v1)
- [Rendle 원문](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf), [저자 libFM bibliography](https://www.libfm.org/)

각 paper YAML은 problem/core idea/가정/강점/출처별 한계/열린 질문/SSP 접점/citation을 포함한다. 확인하지 못한 코드·게재 정보는 추정하지 않았다. BOCS→후속 비판 관계와 이 가설→문헌 citation은 하네스 그래프에 서로 다른 edge로 연결한다. Revisit keyword match는 재검토 단서이며 실험 재개나 해결 판정이 아니다.

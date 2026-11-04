# FP01 independent method / prior-art review

2026-11-04. `AGENTS.md`, `PLAN_v1.md`, 기존 `feature_law_pilot/REPORT_KO.md`, `REVIEW_NOVELTY.md`와 원래 multiclass 입력/모델 코드를 읽었다. 다른 FP01 reviewer 결과는 읽지 않았다. 새 학습·실험·모델 수정은 하지 않았다.

## 판단

**Projection mismatch를 시험하는 새 cycle은 타당하다. 아래 방법 명세를 보완한 뒤 실행을 권고한다.** 기존 binary feature-law의 음성 결과는 이 multiclass closure 가설을 반박하지 않는다. C는 합리적인 masked Gaussian-mixture/CSBM 계열 대안이지만, 새로운 centroid 추론 원리를 발명한 것으로 해석할 단계는 아니다. 먼저 projection 노출 학습의 효과(B−A), centroid 표현/추론의 효과(C-fixed−B 및 C−B), adaptive parameterization의 효과(C−C-fixed)를 구분해야 한다. C만 성공해야 cycle 전체가 의미 있다는 식의 단일 gate는 피한다.

## 1. 확인되는 수학과 미확인 인과

원래 class-mean column은

\[
v_z={0.8K\over K-1}\left(e_z-{\mathbf1\over K}\right),\quad
M_{:f}=\sum_d R_{df}v_{z_d}.
\]

K≥3에서 이 합은 일반적으로 하나의 preferred class와 동일한 나머지 평균을 갖지 않는다. 예를 들어 K=3에서 `v1=(.8,−.4,−.4)`, `v2=(−.4,.8,−.4)`를 `(v1−v2)/sqrt(2)`로 투영하면 `(.8485,−.8485,0)`의 세 서로 다른 평균이 생긴다. 따라서 **명시된 Gaussian 관측 family의 non-closure는 확정**이다. 이는 기존 network가 유용한 결정규칙을 전혀 근사할 수 없다는 불가능성 정리는 아니다. 실제 코드의 [mean 생성](C:/Users/seongjin/Desktop/code/gts_pfn/factor_real_graph_pilot/data.py:91), [real RP](C:/Users/seongjin/Desktop/code/gts_pfn/factor_real_graph_pilot/data.py:130), [기존 cell contrast](C:/Users/seongjin/Desktop/code/gts_pfn/factor_real_graph_pilot/model.py:120)가 이 차이를 확인한다.

- K=2의 antisymmetric centroid는 선형 투영 뒤에도 방향/크기만 바뀌는 이진 contrast다. 이전 feature-law pilot은 binary 2-column 환경이었고 multiclass projection closure를 시험하지 않았다. 다만 그 결과에서 adaptive가 static/original을 이기지 못했고 NLL도 나빴다는 근거는 보존해야 한다. 새 encoder를 자동적인 해결책으로 가정해서는 안 된다.
- Raw noise가 iid Gaussian이면 projected noise covariance는 `σ²RᵀR`이다. 관측 RMS scaling은 다시 좌우 diagonal scaling을 만들고 clipping은 Gaussian law를 바꾼다. C의 독립 Gaussian cell likelihood는 **평균 shape를 표현할 수 있게 한 것**이지 projection 전체 law를 정확히 맞춘 것이 아니다.
- 원래 preferred means의 class 합은 0이므로 투영 뒤에도 같다. 독립 Gaussian prior의 class centroids는 이 제약을 강제하지 않는다. 실제 데이터의 common offset에는 유연할 수 있지만, exact projected prior라는 표현은 부정확하다. 이번 cycle에 sum-zero 제약까지 추가할 필요는 없다. 남은 model mismatch로 기록하면 된다.
- Square16→16 Rademacher는 real native-D→16의 정확한 분포 matching이 아니다. Square R은 nonorthogonal이며 rank/SNR/correlation도 바꾼다. 첫 cycle의 controlled projection intervention으로는 적절하다. 명칭은 **projection-exposed / projection-mechanism-matched training** 정도로 제한한다.
- Real utility 부진의 원인에는 graph potential·nonedge assumption·source 수·role weighting·finite BP·학습 최적화도 남는다. A/B 비교와 C 결과만으로 projection을 유일 원인으로 확정할 수 없다. Coauthors와 CiteSeer의 상반된 nonedge 결과는 그대로 별도 panel에 유지한다.

## 2. 문헌상 위치와 더 강한 설명

**C는 기존 CSBM의 class-specific latent centroid 표현으로 되돌아가는 합리적인 확장이다.** Duranthon–Zdeborová의 [TMLR 2024 원문 Appendix D](https://arxiv.org/html/2306.07948v2)는 class one-hot와 Gaussian centroid vector의 내적을 관측 mean으로 쓰고 multiclass AMP–BP를 제시한다. 해당 asymptotic AMP update가 현재 작은 masked/clipped/RP 입력에 그대로 정확하다고 가정할 수는 없다. 이번 quadrature BP는 별도 finite approximation이며, prior-fitted hyperparameters와 disjoint observations의 효과가 검증 대상이다.

[JSBM/BPGCN, PRR 2020 §II–V](https://arxiv.org/html/1911.00197v2)는 item/feature latent group의 joint BP 및 finite BP 역전파 학습을 이미 다룬다. Bernoulli attribute-group와 Gaussian class-centroid는 다른 observation family다. 따라서 “feature node 대신 centroid를 썼다”만으로 기여를 선언하지 말고, **잘못 제한된 multiclass 관측 family를 넓히는 것이 zero-update feature utility를 실제로 회복하는지**로 질문을 세우는 편이 강하다. 기존 요소를 쓴다는 이유로 이 실험을 기각할 이유는 없다.

실행 전 설명을 다음 두 가설로 분리하는 것을 권한다.

1. **H-interface:** 동일 preferred-class operator도 projected synthetic values로 학습하면 같은 projected test input에서 더 유리하다. B−A가 검증한다. 이 차이는 covariance/SNR 변화까지 포함한다.
2. **H-centroid:** 동일 projected train/test interface에서 class-specific centroid inference가 더 좋은 feature 추가 가치를 준다. C-fixed−B는 통계 모델의 확장, C−C-fixed는 adaptive hyperparameters의 추가 가치를 검증한다. 후자가 실패하고 전자만 성공하면 고정 centroid inference의 성공으로 기록한다.

## 3. 실행 전 material amendments

### A. Role score까지 동시에 바꾸는 범위를 줄인다

C의 heterogeneous-state Bethe score가 아직 명세되지 않았다. 이전 Potts용 logZ 공식을 가져오면 틀린 비교가 된다. 완전한 score를 새로 구현할 수는 있지만, 이번 핵심은 feature law이며 role inference 개선이 아니다.

**최소 실험 권고:** episode와 사전 지정 graph arm마다 inherited graph-only/no-observed 계산으로 `q_h`를 한 번 정하고, A/B/C/C-fixed 및 source/none/shuffle 모드에서 동일하게 고정한다. Graph parameter도 고정한다. Full/edge-only 각각의 graph-only score 정의를 잠그고 어느 쪽이든 dataset별 선택하지 않는다. 결과는 “common graph-only role weights 아래의 feature-conditional prediction”으로 부르며 joint evidence role posterior라고 부르지 않는다. 이렇게 하면 Gaussian prior/grid의 새 partition normalization이 arm 간 role 선택을 바꿔 생기는 이득을 분리할 수 있다. Original trained Factor는 별도 historical reference로 둔다.

이 단순화를 채택하지 않는다면, train 전에 centroid prior masses, 모든 cell factor, context clamp, variable/factor entropy 및 role별 상수를 포함한 식을 완성하고 exact enumeration과 대조해야 한다. “추후 Bethe score를 정한다”는 상태로 학습을 시작하면 안 된다.

### B. 11-point grid와 C-fixed를 먼저 완전히 정한다

`11 points`, `observed statistics`, `fixed coefficients`만으로는 재현 가능한 method가 아니다. Gauss–Hermite인지 uniform grid인지, normalized prior weights, grid transform, scale/noise 양수 하한, column에 관측이 0/1개일 때 값, clip 경계, encoder 입력과 초기화가 필요하다. 모든 class에 같은 prior/hyperparameter law를 쓰고 class ID별 자유 parameter를 넣지 않는다. Learned location/scale이 관측에 의존하면 conditional score라는 현재 고지는 유지한다.

C-fixed의 수치를 임의로 약하게 정하지 않는다. Known synthetic Gaussian scale와 visible RMS로부터 유도해 고정하거나, 비교군을 **C-global**로 명명해 소수 전역 hyperparameter만 동일 합성 CE 예산으로 학습하는 둘 중 하나를 사전 선택한다. 둘의 여러 설정을 real 결과로 고르는 것은 금한다. A/B/C에도 동일하게 적용할 normalization convention을 먼저 잠근다.

### C. Exact product가 맞아도 분할 BP graph는 새 loop를 만든다

제안한 `psi_icf = Normal(...)` if `s_i=c`, else 1의 class별 곱은 원하는 likelihood와 **대수적으로 일치**한다. 그러나 두 source가 같은 column의 서로 다른 class centroids를 공유하면 `s_i → μ_cf → s_j → μ_c'f → s_i`의 length-4 loop가 생긴다. 원래 graph가 tree여도 inference graph는 tree가 아닐 수 있다. 따라서 tree exactness만으로 이 representation의 유용한 approximation을 확인했다고 할 수 없다.

Tree 검사에 더해 작은 **two-source/K=3 loop**에서 discrete exact marginals와 finite BP 오차를 기록한다. 전반적인 BP 수렴 보장을 요구하는 것이 아니라 새 factorization의 자체 증거 중복을 측정하는 최소 검사다. Singleton에서 identical class priors를 주변화하면 feature는 class 정보를 주지 않아야 한다. 이 성질은 centroid로 바꿔도 사라지지 않으며, source 한 개뿐인 real 조건을 표현력 개선의 성공/실패 gate로 쓰면 안 된다.

### D. Gaussian reference에는 numerical integration보다 conjugacy를 쓸 수 있다

Tiny reference는 class labels를 열거하고 centroid를 해석적으로 적분하면 된다. `μ_cf ~ N(m_f,τ_f²)`, `x_j|μ_cf ~ N(μ_cf,σ_f²)`에서 특정 class에 배정된 n개 관측의 `r_j=x_j−m_f`에 대해

\[
\log I_{cf}=-{n\over2}\log(2\pi\sigma_f^2)
-{1\over2}\log\left(1+{n\tau_f^2\over\sigma_f^2}\right)
-{1\over2\sigma_f^2}\left[\sum_j r_j^2-
{\tau_f^2\over\sigma_f^2+n\tau_f^2}\left(\sum_jr_j\right)^2\right].
\]

n=0이면 `log I=0`이다. 이는 본 reviewer의 conjugate integral 전개다. 고정 visible statistics/conditional hyperparameters에도 각 입력에 대해 계산할 수 있다. **Discrete-grid exact enumeration**, **continuous Gaussian analytic reference**, **finite BP**를 분리하면 quadrature error와 BP error를 구분할 수 있다. 연속 likelihood의 normalization을 포함한다. 실제 generative law가 clipping/RP와 다르다는 문제까지 해결하는 reference는 아니다.

## 4. 권고하는 최소 executable experiment

1. **CPU correctness/dev 단계:** K=3, 1–2 columns, 관측 source 0/1/2개, 작은 graph tree와 위 loop를 고정된 toy 집합으로 정한다. Raw와 projected mean counterexample을 포함한다. 위 analytic integration 및 label enumeration과 quadrature/finite BP를 대조한다. 관측 0/1개 invariance, class/column/node permutation과 hidden poison, gradients, graph freeze도 확인한다. 11-point 선택의 오차 기준은 이 단계에서 수치적으로 정하고 final 전에 잠근다. Dev에서 부족하면 grid 수를 기계적으로 늘리는 수정은 기록하되 real accuracy로 고르지 않는다.
2. **명세가 닫힌 뒤 A/B/C/(C-fixed 또는 C-global)만 실행:** 제안한 3 seeds·512 shared base tasks·256×2 task exposures를 유지한다. Paired raw/projected test views에서 **같은 입력을 받는 A/B**를 비교하고 C와 B도 같은 projected 입력으로 비교한다. 독립 projection seed와 base graph를 lock한다. Frozen graph가 old feature operator와 함께 학습된 checkpoint라는 inherited-cost/conditioning 제한을 기록한다. Synthetic final에서는 base graph 단위 paired CI와 accuracy/NLL을 보고한다.
3. **Real은 지정된 5개 graph/75 episodes를 그대로 개발 평가:** Full/edge-only panel, 모든 rates, no-observed 및 shuffle을 전부 보인다. Graph별 `full−none`, `C−B`, `C−C-fixed/global`를 나란히 보고하며 source 수·평가 전용 source-class coverage도 기록한다. Projection seeds/real graph arm을 보고 골라 쓰지 않는다. 무조건 C의 accuracy 우승을 요구하기보다, 관측 feature의 추가 가치와 probability quality가 함께 개선되는지 판단한다. Source singleton의 정보 제약은 별도로 해석한다.

**권고 결론:** FP01을 진행하되 첫 버전은 common graph-only role protocol과 정확한 Gaussian 참조로 단순화한다. C-fixed가 B보다 좋아도 adaptive의 가치가 없으면 그 결과를 그대로 받아들이고, B만 좋아도 interface correction 성과로 기록한다. 전부 실패하면 projection non-closure라는 수학적 사실은 남지만, 그것이 현재 실제 실패를 설명한다는 주장은 기각 또는 미확인으로 남는다.

# JR-PFN: exact Bayes benchmark and fixed-budget learned pilot

> **EAJ-PFN 학습 완료:** [Shared joint evidence MVP](evidence_amortized/REPORT.md)에서18개 모델을 동일 예산으로 학습했다. 주요 OOD6조건 모두 independent heads·analytic initializer·own-marginal product보다 KL이 낮았지만 exact-marginal product보다 여전히 나빴다. Density-high Coupled는 frozen0.07808→EAJ0.03608, exact joint0.01762다. Strong constant+updated-role control의 범용 우위·확실한 accuracy 증분·전체 계산비용 대비 exact 수준의 품질은 미확인이다. Learned evidence의 추가 이득은 확인했고 exact-grid 대체·schema 확대는 보류한다.

> **Joint role–parameter BP 검증 완료:** [Full-grid BP baseline](joint_role_parameter/REPORT.md)에서 같은 BP branches에 coherent joint/frozen-role/marginal-product weights를 비교했다. Density-high Coupled KL은 기존 MRP0.11282→frozen grid0.07808→joint BP0.01762였고 같은 marginals의 곱0.01905보다도 좋았다. Exact decoder에서만 보이던 효과가 아니다. 다만 matched에서는 constant potential의 role posterior만 갱신한 기준선도 강하고 prior별 실패가 남는다. 알려진 family의 model-based Conditional Go이며 새 학습은0회다.

> 위 Conditional Go는 확률 품질에 관한 판정이다. Density-high Coupled의 frozen-grid/joint-grid accuracy는 모두70.38%이며, 이 family에서는 role mass가 confidence를 바꾸고 argmax는 유지한다. 다른 두 prior도 같은 조건에서 role update의 독립적인 accuracy 증분 CI가0을 포함했다.

> **Known-family parameter 검증 완료:** [학습 없는 generalization test](parameter_generalization/REPORT.md)에서 constant·현재 point·통계/EB point·parameter distribution을 같은 masked decoder와 frozen q로 비교했다. Low-density Exclusive는 현재 point0.00725, 저렴한 density 추정0.00129, broad distribution0.00208이었다. Uncertainty는 일부 mean 대비 이득이 있지만 최강 통계 기준선 우위는 아직 없다. 같은 exact parameter predictive에서 role posterior만 coherent하게 업데이트한 density-high Coupled0.07936→0.01215도 확인했다. Potential뿐 아니라 role posterior의 OOD 적응도 남은 문제이며, 새 distributional head 학습은 진행하지 않았다.

> **Learned potential MVP 완료:** [Mask-aware Relational Potential PFN](learned_potential/REPORT.md)에서 posterior를 고정하고 oracle potential을 shared encoder로 대체했다.36개 모델·동일192-update budget의 matched Exclusive KL은 Positive0.05259→learned0.01127, oracle potential+동일 q0.01007이었다. 새 Large Plain과 learned signed보다 세 prior 모두 KL이 낮았다. 다만 추가 likelihood loss는 predictive-only보다 Independent/Exclusive에서 작게 나빴고, Coupled는 constant potential도 못 넘었다. 학습 가능한 decoder 치환은 Go, 새 학습 목적의 우월성·probability 외삽·schema 확장은 미검증이다.

> **Decoder 표현력 검증 완료:** [Contrastive Relational Decoder](contrastive_decoder/REPORT.md)에서 positive-only diffusion을 native edge/nonedge likelihood의 고정 sum-product 계산으로 교체했다. 새 학습0회, 새 graph768개다. 같은 frozen learned q에서 Exclusive Bayes KL0.04820→0.00867, exact joint q에서는0.00248이었다. 같은 decoder의 exact factorized0.00783 대비 joint 증분도 남았다. 알려진 생성 확률을 직접 이용하므로 새 학습 메소드의 성과는 아니다. Decoder 원리는 Go이며 observation-aware likelihood potential의 학습은 다음 미검증 과제다.

> **고정 decoder의 distillation 학습 완료:** [Role-Conditional Distillation ablation](distillation/REPORT.md)에서 추가 conditional 감독이 역할별 KL은 줄였으나 최종 predictive KL의 C−B 차이는 세 prior 모두 CI가0을 포함했다. Posterior 감독을 더한 D는 component posterior KL은 낮췄지만 predictive KL은 커졌다. 같은 exact teacher로 학습한 Plain과의 기존 JR 장점은 Exclusive에서 남지만 이번 추가 loss의 고유 성과는 아니다. 현재 loss-only 설계의 Stage2 확장은 보류한다. 이전 Exact Bayes 분석의 inference 가치 자체는 여전히 유효하다.

> **후속 분석 완료:** [Prior-to-Posterior Shift](posterior_shift/REPORT.md)에서 Exclusive·총 라벨 4개의 label 추가 역할 정보 0.12447 nats와 exact role 업데이트의 기대 NLL 이득 0.02278을 확인했다. 같은 exact conditional predictor에서 learned posterior는 가용 이득의 약53%를 회수했다. 따라서 아래 accuracy +0.14%p만으로 정보 부재나 posterior 학습의 완전 실패를 결론 내리지 않는다. Posterior와 conditional decoder의 오차를 함께 해결할 Conditional Go다. 새 학습은0회다.

> 2026-10-31 · 개발용 합성 실험. 공식 pretrained GraphPFN·실제 heterograph·unseen schema 평가가 아니다.

## 판정

**관계 역할을 평균하기 전후의 차이는 정확한 생성모형에서 확인했다. 그러나 그 차이 전체를 관계 간 상관관계의 기여로 해석할 수는 없다.** 관계별 분산을 유지하는 factorized 대조군과의 차이, 실제 posterior 근사 정도, 기존 Plain 대비 예측 이득을 각각 구분한다. 알려진 prior를 보존하는 효과와 context로 미지의 관계 의존성을 학습한 효과도 다르다.

이번에 구현한 것은 공유 staged graph adapter의 역할별 내부 계산을 확률 평균하는 소형 JR-PFN이다. 진단에 그치지 않고 총 66개 모델을 고정된 예산으로 학습했다. 모델 이름만으로 Bayesian posterior나 논문 novelty를 주장하지 않는다.

### 핵심 결과를 먼저 읽으면

- **Exclusive의 국소 positive signal:** JR은 capacity+long Plain 대비 +3.20%p [1.44, 4.98], factorized 대비 +1.76%p [0.28, 3.27].
- **범용 이득은 아님:** Independent의 JR−factorized는 +0.06%p로 CI가 0을 포함하며, Coupled에서는 개선이 없다.
- **새 posterior 학습 원리의 성과는 약함:** Exclusive에서 decoder를 고정하고 학습 posterior를 known prior로 되돌리면 accuracy 차이는 +0.14%p [−0.10, 0.41]에 그친다. 이 frozen 대조는 완전히 재학습한 prior-only baseline은 아니다.
- **보조목적의 예측 이득 미확인:** Exclusive에서 CE-only와 사실상 같고 Independent에서는 CE-only가 예측 KL이 더 좋다. Posterior 정확성과 decision quality를 분리해야 한다.
- **현재 판정:** 상관된 역할 prior를 보존하는 구조의 작은 근거는 확보했다. 미지의 관계 의존성을 극소 context에서 추론하는 새로운 학습 원리나 논문 메인 메소드는 아직 입증하지 못했다.

## 1. 생성모형과 정확한 기준

- Target 8개, 두 relation이 공유하는 helper 4개, binary class, role R=2의 네 상태를 전부 열거한다.
- 모든 노드의 잠재 라벨은 iid Bernoulli(0.5)다. 관측된 edge와 nonedge를 모두 likelihood에 포함한다.
- $P(B_{rij}=1\mid z,h)=0.25[1+0.72h_r(2z_i-1)(2z_j-1)]$. 역할 0은 잡음, 1은 homophilic signal이다.
- Helper 라벨과 query 라벨을 포함한 $2^{12}$개 잠재 할당을 전부 주변화한다. 생성 시 뽑힌 role을 posterior로 대신 사용하지 않는다.
- 피처는 전부 없다. **99.5% 결측과 동일한 실험이 아니다.**
- Context는 라벨과 독립적으로 뽑은 총 1·2·4개 target이다. **기존의 클래스당 k=1/3/5/10/20과 다르다.**
- 각 prior별 train graph 2,048개, 별도 eval graph 192개. Context mask는 nested이며 query 정답은 inference에 들어가지 않는다.

| Prior | Role 확률 (00,10,01,11) | 의미 |
|---|---|---|
| Independent | (1/4,1/4,1/4,1/4) | 독립 역할 prior; graph를 관측한 posterior는 독립일 필요가 없음 |
| Exclusive | (0,1/2,1/2,0) | 둘 중 하나만 signal |
| Coupled | (1/2,0,0,1/2) | 함께 signal 또는 함께 noise |

Exclusive/Coupled는 같은 **prior marginal**을 가진다. Graph/context 관측 뒤 posterior marginal까지 같도록 맞춘 SBM 쌍은 아니다. 정확히 같은 marginal을 갖는 제한된 반례는 [THEORY.md](THEORY.md)의 고정 typed-path 예제로 별도 검증했다. 알 수 없는 prior regime을 추론한 결과로 해석하지 않는다.

## 2. Gate 1: oracle 계산에서 무엇을 잃는가?

아래는 $\mathrm{KL}(p^*_{Bayes}(y_i\mid D)\Vert p_{control}(y_i\mid D))$의 graph 평균과 95% bootstrap CI다. 이는 정확한 conditional branch와 정확한 role posterior를 사용한다. Exact joint는 Bayes와 같아서 KL=0이며, 학습 성공 결과가 아니다.

| Prior | MAP | Mean-affinity plug-in | Product of exact marginals |
|---|---:|---:|---:|
| Independent | 0.05378 [0.04813, 0.05960] | 0.00805 [0.00725, 0.00890] | 0.00046 [0.00037, 0.00058] |
| Exclusive | 0.03252 [0.02949, 0.03612] | 0.01828 [0.01656, 0.02015] | 0.00645 [0.00583, 0.00707] |
| Coupled | 0.02865 [0.02421, 0.03328] | 0.00529 [0.00470, 0.00588] | 0.00348 [0.00307, 0.00391] |

Mean-first의 gap에는 marginal variance 손실도 포함된다. **Joint dependence의 필요성은 마지막 열로 판단해야 한다.** Independent prior에서 그 gap은 작다. 알려진 correlated prior에서는 더 크지만, 그 자체가 posterior를 학습했다는 증거는 아니다.

중요한 수학적 구분: 두 층의 선형 message passing도 $A(h)^2=\sum_{r,s}h_rh_sA_rA_s$이므로 역할에는 비선형이다. 필요한 null은 message-state 선형성이 아니라 **역할 변수에 대한 affine dependence**다. 상세 범위와 반례는 [THEORY.md](THEORY.md)에 있다.

## 3. 실제 구현한 모델과 공정성

각 가설의 typed message channel에 $h_r$를 곱한다. Relation identity를 adjacency 하나로 먼저 합치지 않고 공유 adapter 안에서 처리한 뒤 **확률분포**를 평균한다. Posterior scorer는 relation-shared unary MLP와 symmetric pair MLP로 Ising energy를 계산한다. 입력은 graph/context의 invariant statistics다. R=2에서는 energy family가 충분하지만 그 통계가 모든 posterior 정보를 담는다는 보장은 없다.

| 모델 | 역할 처리 | 파라미터 |
|---|---|---:|
| Plain | 기존 staged adapter를 같은 prior로 재학습 | 10,469 |
| Plain capacity+long | 실제 relation MLP 폭 78, diffusion 80/48회 | 14,655 |
| MAP / Mean / Factorized / JR | 공유 adapter + Ising scorer | 14,503 |
| JR CE-only | 같은 구조, posterior KL 보조목적 제거 | 14,503 |

기본 Plain/JR의 diffusion은 20/12회다. Capacity+long은 파라미터가 JR보다 약 1% 많고 반복 수는 네 배다. FLOPs를 정확히 일치시켰다는 주장은 하지 않는다. 실제 latency는 아래에 보고한다. Independent screen의 추가 Plain-wide는 피처 encoder만 넓힌 14,389개 모델이며, 피처가 없는 조건에서 유효 capacity 대조군으로 취급하지 않는다.

학습은 모델별 192 updates × batch 16, AdamW lr=0.001, seed 17/29/43이다. 동일 prior 안의 모든 모델은 같은 task schedule을 사용한다. 비교 가능한 기본 adapter는 같은 seed의 초기 가중치를 공유한다. Checkpoint는 마지막 update로 고정하고 test로 고르지 않았다. Query CE에 0.1 × exact-posterior KL을 더한다(Plain과 CE-only 제외). 따라서 CE-only 대조 없이 새 학습 원리의 이득을 주장하지 않는다.

Independent는 사전 고정한 첫 screen이다. Exclusive/Coupled는 그 결과를 본 후, 원래 제안된 correlated-prior 가설을 직접 확인하기 위해 별도로 고정한 개발용 후속 실험이다. 각 regime마다 별도 모델을 학습한다. **Unseen prior regime / schema adaptation 평가가 아니다.**

## 4. Gate 3: 학습한 예측 결과

총 context 1/2/4와 3개 학습 seed를 평균했다. Accuracy는 %, NLL/Brier/KL은 낮을수록 좋다. Bayes 행은 동일 graph에서 정확히 계산한 marginal predictor다.

| Prior | 모델 | Accuracy | NLL | Brier | Bayes predictive KL |
|---|---|---:|---:|---:|---:|
| Independent | plain | 50.86 | 0.68863 | 0.49561 | 0.06515 |
| Independent | plain_capacity_long | 51.07 | 0.68694 | 0.49400 | 0.06419 |
| Independent | map | 52.60 | 0.67955 | 0.48683 | 0.06058 |
| Independent | mean | 51.09 | 0.68653 | 0.49359 | 0.06454 |
| Independent | factorized | 52.35 | 0.68185 | 0.48911 | 0.05886 |
| Independent | joint | 52.41 | 0.68122 | 0.48850 | 0.05829 |
| Independent | joint_ce | 52.59 | 0.67805 | 0.48519 | 0.05533 |
| Independent | Exact Bayes | 64.72 | 0.61523 | 0.42868 | 0 |
| Exclusive | plain | 53.57 | 0.68872 | 0.49558 | 0.06431 |
| Exclusive | plain_capacity_long | 53.83 | 0.68658 | 0.49350 | 0.06215 |
| Exclusive | map | 54.80 | 0.67682 | 0.48430 | 0.05063 |
| Exclusive | mean | 54.29 | 0.67959 | 0.48706 | 0.05477 |
| Exclusive | factorized | 55.27 | 0.67584 | 0.48330 | 0.04960 |
| Exclusive | joint | 57.03 | 0.67186 | 0.47929 | 0.04309 |
| Exclusive | joint_ce | 57.06 | 0.67154 | 0.47897 | 0.04301 |
| Exclusive | Exact Bayes | 63.19 | 0.62777 | 0.44006 | 0 |
| Coupled | plain | 58.39 | 0.65871 | 0.46681 | 0.09636 |
| Coupled | plain_capacity_long | 58.39 | 0.65873 | 0.46682 | 0.09639 |
| Coupled | map | 56.33 | 0.66620 | 0.47417 | 0.10361 |
| Coupled | mean | 58.32 | 0.65570 | 0.46439 | 0.09518 |
| Coupled | factorized | 58.35 | 0.65845 | 0.46713 | 0.09628 |
| Coupled | joint | 58.22 | 0.65815 | 0.46663 | 0.09693 |
| Coupled | joint_ce | 58.26 | 0.65850 | 0.46675 | 0.09656 |
| Coupled | Exact Bayes | 67.87 | 0.56458 | 0.39002 | 0 |

### Paired contrasts

| Prior | JR − control | Accuracy 차이 (%p), 95% CI | Bayes KL 차이, 95% CI |
|---|---|---:|---:|
| Independent | plain_capacity_long | +1.35 [+0.55, +2.12] | -0.00590 [-0.00753, -0.00431] |
| Independent | factorized | +0.06 [-0.07, +0.21] | -0.00056 [-0.00070, -0.00042] |
| Independent | joint_ce | -0.18 [-0.70, +0.32] | 0.00296 [0.00193, 0.00399] |
| Exclusive | plain_capacity_long | +3.20 [+1.44, +4.98] | -0.01906 [-0.02201, -0.01640] |
| Exclusive | factorized | +1.76 [+0.28, +3.27] | -0.00651 [-0.00749, -0.00555] |
| Exclusive | joint_ce | -0.03 [-0.13, +0.06] | 0.00008 [-0.00024, 0.00039] |
| Coupled | plain_capacity_long | -0.17 [-0.40, +0.04] | 0.00055 [-0.00055, 0.00167] |
| Coupled | factorized | -0.13 [-0.71, +0.43] | 0.00065 [-0.00055, 0.00190] |
| Coupled | joint_ce | -0.04 [-0.20, +0.12] | 0.00037 [-0.00051, 0.00130] |

CI는 seed/context를 graph 내부에서 평균한 뒤 192개 graph를 4,000회 재표집했다. 따라서 graph sampling 불확실성에 대한 조건부 CI이며 3개 학습 seed의 모집단 불확실성을 모두 반영하지 않는다. 개별 seed와 context별 결과는 [all_prior_analysis.json](results/all_prior_analysis.json)에 있다.

## 5. Gate 2: posterior를 실제로 근사했는가?

| Prior | JR posterior KL | CE-only posterior KL | JR + oracle decoder KL | Oracle posterior + JR decoder KL |
|---|---:|---:|---:|---:|
| Independent | 0.18908 | 0.56336 | 0.01593 | 0.04740 |
| Exclusive | 0.04340 | 0.04821 | 0.00605 | 0.03749 |
| Coupled | 0.19563 | 0.39651 | 0.02800 | 0.08324 |

오른쪽 두 열은 기전 진단에만 쓰는 oracle-component substitution이며, 실행 가능한 학습 모델의 성적이 아니다. Posterior KL이 작은 것과 예측이 좋은 것은 다르다. 알려진 prior의 support 제약으로 올바른 상관관계의 일부를 처음부터 제공했다는 점도 중요하다. CE-only가 예측에서 이기더라도 그 mixture weight를 올바른 posterior라고 부르지 않는다.

### 같은 학습 branch에서 집계만 변경

추가 학습 없이 JR의 같은 conditional branch를 (a) 학습 joint posterior, (b) 그 marginal의 곱, (c) 고정 known prior로 집계했다. 재학습한 baseline이 아니라 집계 연산의 frozen ablation이다.

| Prior | learned joint − aggregation | Accuracy 차이 (%p), 95% CI | Bayes KL 차이, 95% CI |
|---|---|---:|---:|
| Independent | same_q_factorized | +0.12 [-0.02, +0.27] | -0.00047 [-0.00056, -0.00039] |
| Independent | fixed_known_prior | +0.17 [-0.10, +0.43] | -0.00138 [-0.00199, -0.00077] |
| Exclusive | same_q_factorized | +1.78 [+0.30, +3.31] | -0.00886 [-0.00996, -0.00780] |
| Exclusive | fixed_known_prior | +0.14 [-0.10, +0.41] | -0.00362 [-0.00494, -0.00238] |
| Coupled | same_q_factorized | -0.15 [-0.74, +0.43] | 0.00064 [-0.00039, 0.00168] |
| Coupled | fixed_known_prior | +0.00 [+0.00, +0.00] | -0.00289 [-0.00470, -0.00115] |

### Coupled에서 accuracy 효과가 제한되는 구조적 이유

현재 adapter의 h=00 branch는 uniform이다. Coupled는 00/11만 허용하므로 $p=q_{00}/C+q_{11}p_{11}$이고, $q_{11}>0$이면 argmax는 항상 $p_{11}$과 같다. 따라서 같은 decoder에서 posterior weight의 변화가 accuracy를 바꿀 수 없고 confidence만 바꾼다. Coupled의 frozen known-prior 대조에서 accuracy 차이가 정확히 0인 이유다. Coupled 결과를 posterior reasoning 전체의 불가능성으로 해석하지 않는다.

## 6. Context 정의와 decoder의 한계

총 라벨 1개에서는 positive diffusion이 그 라벨과 반대 class를 지지할 수 없다. Exact SBM은 nonedge로 반대 class를 지지할 수 있다. 따라서 현재 adapter에는 posterior만으로 메울 수 없는 표현상 차이가 있다. 이전의 클래스당 1개 조건과 같은 퇴화가 아니다.

아래는 관측 context만으로 정의한 balanced stratum의 2차 진단이다. Query 정오로 선별하지 않았고, 주 결과는 위 전체 graph 평균이다. 총 2개에서 각 class 1개, 총 4개에서 각 class 2개인 subset을 보인다.

| Prior | Total labels | Balanced graphs | Plain accuracy | JR accuracy |
|---|---:|---:|---:|---:|
| Independent | 2 | 97 | 54.35 | 58.25 |
| Independent | 4 | 85 | 56.27 | 62.25 |
| Exclusive | 2 | 108 | 57.77 | 59.93 |
| Exclusive | 4 | 61 | 57.65 | 62.30 |
| Coupled | 2 | 95 | 64.56 | 64.39 |
| Coupled | 4 | 77 | 64.61 | 64.18 |

## 7. 자원과 검증

| Arm | Median batch-16 inference ms | Total training seconds |
|---|---:|---:|
| plain | 8.61 | 68.7 |
| plain_capacity_long | 19.26 | 163.7 |
| map | 13.61 | 78.9 |
| mean | 14.87 | 80.9 |
| factorized | 14.15 | 80.9 |
| joint | 14.04 | 78.5 |
| joint_ce | 13.80 | 79.9 |

RTX 3090의 짧은 실측이며 공유 GPU의 부하 변동을 포함한다. 엄밀한 FLOP matching이나 안정된 throughput benchmark는 아니다. Train 초는 optimizer loop만 포함하며 oracle bank 구축·평가·감사는 제외한다.

- 독립 scalar 전수 열거와 vectorized exact inference의 최대 확률 차이는 약 4.4e−16.
- Edge signal=0, empty-context symmetry, relation/class permutation null 검사.
- Helper labels, nonedge likelihood, 모든 query latent labels 주변화.
- Query truth를 바꿔도 model input·예측·role posterior가 불변.
- 같은 prior/seed의 task schedule, 공통 초기 adapter, checkpoint/source/bank hash 검사.
- 지표를 저장된 확률에서 재집계. 최종 감사는 [audit.json](results/audit.json).

## 8. 이 결과로 주장할 수 있는 범위

Gate 1은 공통 conditional 연산에서 평균화나 factorization이 예측 정보를 잃을 수 있음을 지지한다. Gate 2는 posterior 근사 정도를 측정한 단계이며, 충분히 정확한 Bayesian inference를 달성했다고 자동 판정하지 않는다. Gate 3의 수치는 이 adapter·작은 prior·고정 예산에 한정된다. Gate 4는 미실시다.

**일반 MoE/ensemble, R=3/4, 미지의 role 조합·prior·semantic schema, 실제 heterograph는 비교하지 않았다.** 특히 알려진 Exclusive/Coupled 제약을 보존하는 능력은 context에서 미지의 relation 의존성을 발견하는 능력과 다르다. 이 pilot만으로 논문급 novelty나 unseen-schema 일반화를 확립했다고 판정하지 않는다.

## 재현

사용 Python: `relational_baselines/.venv-rt/Scripts/python.exe`. 새 output directory에서 순서대로 실행한다. 기존 bank/checkpoint의 덮어쓰기는 거부한다.

```powershell
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/build.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/run.py checks
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/run.py train
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/run.py eval
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/capacity_control.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/analyze.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/correlated.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/correlated_analyze.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/write_report.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/audit.py
```

![JR-PFN results](results/joint_role_results.png)

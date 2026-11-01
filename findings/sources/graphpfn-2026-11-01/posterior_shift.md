# JR-PFN: Prior-to-Posterior Shift와 예측 가치

> 2026-10-31 · 새 학습 0회 · 기존 checkpoint와 원래 결과 불변 · 소형 SBM의 개발용 기전 검증

**판정: posterior inference의 예측 가치는 있다. 현재의 약한 accuracy 증분을 정보 부재로 설명하기는 어렵다.** 다만 학습 posterior도 일부 이득을 회수하므로 완전 실패라고 부를 수 없고, conditional decoder의 오차도 크다. 새 메소드의 novelty나 unseen-schema 성능을 확립한 결과는 아니다.

- 기존 Exclusive / 총 라벨 4개에서 label 추가 정보는 **0.12447 nats [0.11344, 0.13623]**다.
- 정확한 conditional predictor에서 fixed prior를 exact posterior로 바꾸면 기대 NLL을 **0.02278 [0.01949, 0.02649]** 줄일 수 있다.
- 그래프만 본 posterior에서 context label까지 반영하는 순수한 role-weight 업데이트의 가치는 **0.01595 [0.01401, 0.01803]**다.
- 같은 exact conditional predictor에서 learned posterior도 일부 이득을 회수한다. 현 단계는 **정보 부재가 아니라 불완전한 posterior 근사와 decoder 오차가 함께 남은 Conditional Go**다.

## 1. 정보량과 predictor를 어떻게 정의했나?

$D_C=(G,Y_C)$ 전체의 역할 정보량은 graph 정보와 label 추가 정보를 모두 포함한다. 세 항을 분리했다.

\[I(H;G,Y_C)=I(H;G)+I(H;Y_C\mid G).\]

Graph별로 가능한 모든 context label pattern을 합산했다. 따라서 아래 rb 값은 해당 graph에 조건부로 context/query label의 Monte Carlo 잡음이 없다. Graph 자체는 prior에서 표집했으므로 그 불확실성은 남고, graph 단위 bootstrap으로 CI를 계산했다. 단위는 nats다.

같은 hypothesis-conditioned predictor $f_h=p^*(y_q\mid G,Y_C,h)$를 고정한 채 집계 weight만 바꾼다.

\[p_\pi=\sum_h\pi(h)f_h,\quad p_G=\sum_hp(h\mid G)f_h,\quad p_*=\sum_hp(h\mid G,Y_C)f_h.\]

여기서 $p_\pi$의 branch는 이미 graph/context를 조건으로 사용한다. **Fixed-prior aggregation**이라는 대조군이며, 모든 관측을 조건으로 한 올바른 Bayesian prior predictive라고 부르지 않는다.

정확한 Bayes 분포 아래 기대 NLL 이득은 $KL(p_*\Vert p_\pi)$ 또는 $KL(p_*\Vert p_G)$다. 실제 표집 query 정답으로 계산한 NLL 차이도 별도로 저장했다. 후자는 유한 표본에서 음수가 될 수 있다.

## 2. 기존 평가 bank 재분석

원래 세 prior별 graph 192개, 총 context 1/2/4개를 그대로 사용했다. 아래 표는 가능한 context label을 모두 평균한 값이다. CI는 graph 4,000회 재표집이며 같은 graph의 context count는 독립 표본으로 세지 않는다.

| Prior | 총 labels | Graph 정보 | Label 추가 정보 | 전체 정보 | Prior→exact 기대 NLL 이득 | Graph posterior→exact 이득 |
|---|---:|---:|---:|---:|---:|---:|
| independent | 1 | 0.15050 | 0.00000 | 0.15050 | 0.01159 | 0.00000 |
| independent | 2 | 0.15050 | 0.04072 | 0.19122 | 0.01643 | 0.00215 |
| independent | 4 | 0.15050 | 0.15675 | 0.30726 | 0.02553 | 0.01121 |
| exclusive | 1 | 0.02635 | 0.00000 | 0.02635 | 0.00287 | 0.00000 |
| exclusive | 2 | 0.02635 | 0.03060 | 0.05695 | 0.00723 | 0.00268 |
| exclusive | 4 | 0.02635 | 0.12447 | 0.15082 | 0.02278 | 0.01595 |
| coupled | 1 | 0.16705 | 0.00000 | 0.16705 | 0.02146 | 0.00000 |
| coupled | 2 | 0.16705 | 0.03560 | 0.20265 | 0.03346 | 0.00302 |
| coupled | 4 | 0.16705 | 0.13324 | 0.30030 | 0.05111 | 0.01614 |

**총 라벨 1개에서 label 추가 역할 정보는 정확히 0이다.** Binary label을 전부 뒤집어도 graph likelihood가 같으므로 $p(Y_c=0\mid G,h)=p(Y_c=1\mid G,h)=1/2$다. 따라서 이 라벨은 $p(H\mid G)$를 바꾸지 않는다. 그렇지만 label은 query class의 방향을 정하는 데 쓸 수 있으므로 분류에 무용하다는 뜻은 아니다. 또한 총 1개는 기존의 클래스당 1개 조건과 다르다.

### Exclusive / 총 4개에서 CI까지 보면

| 측정 | 평균 [95% CI] |
|---|---:|
| Graph 정보 | 0.02635 [0.01956, 0.03417] |
| Label 추가 정보 | 0.12447 [0.11344, 0.13623] |
| 전체 정보 | 0.15082 [0.13649, 0.16581] |
| Prior→exact 기대 NLL 이득 | 0.02278 [0.01949, 0.02649] |
| Graph-only→exact 기대 NLL 이득 | 0.01595 [0.01401, 0.01803] |

즉, 이전 accuracy +0.14%p만으로 $p^*(H\mid D_C)\approx p(H)$라고 결론 내릴 수 없다. 이 조건에서는 정확한 posterior가 label 정보를 반영하고, 그 변화가 예측에도 유용하다.

## 3. 같은 predictor에서 네 가지 posterior를 비교

다음은 기존 Exclusive / 총 4개에서 **동일한 exact conditional branch**를 쓴 결과다. 이 표는 실제 관측된 context에 조건부이고 query 정답으로 계산한 empirical NLL을 포함한다. 위 표는 모든 가능한 context를 평균하므로 기대값의 기준이 다르다.

| 집계 weight | Empirical NLL | Bayes predictive KL |
|---|---:|---:|
| Known prior | 0.61592 | 0.02278 |
| Exact posterior | 0.59123 | 0.00000 |
| Learned JR posterior | 0.60174 | 0.01070 |
| Product of exact marginals | 0.59730 | 0.00699 |

Known-prior KL 0.02278에서 learned posterior KL 0.01070으로 줄어든다. 회수한 기대 NLL 이득은 **0.01208 [0.00819, 0.01610]**로, 이 관측-context 기준 가용 이득의 약 53%다. **Learned가 전혀 정보를 쓰지 못했다는 해석도 맞지 않는다.** 그러나 exact posterior에 남은 차이는 분명하며, exact marginal을 factorize한 것보다도 오차가 크다.

### Posterior만 정확하게 만들면 현재 JR이 Bayes에 도달하나?

| Decoder | Weight | Bayes predictive KL |
|---|---|---:|
| Exact | Learned | 0.01070 |
| Learned | Learned | 0.06131 |
| Learned | Exact | 0.05054 |
| Learned | Known prior | 0.07002 |
| 별도 Plain capacity+long | 자체 예측 | 0.09106 |

현재 decoder에서 exact posterior로 바꾸면 KL 0.06131→0.05054로 줄지만, 큰 decoder 오차가 남는다. 따라서 posterior head만 개선하면 문제가 해결된다고 단정하지 않는다. 이 교체는 기전 진단이며 새로 학습한 모델의 성능이 아니다.

## 4. Paired context-signal sweep

각 prior별 새로운 master graph 512개를 사용했다. 같은 latent label·sampled role·edge uniform·context order를 모든 signal과 총 context 1/2/4/6개에서 공유한다. Context index를 라벨과 독립적으로 먼저 정한 뒤 graph를 생성한다.

\[P(B_{rij}=1\mid z,h,C)=0.25[1+\eta_i h_r(2z_i-1)(2z_j-1)].\]

Query target의 $\eta_i=0.72$는 고정하고 context target만 0 / 0.36 / 0.72 / 0.96으로 바꿨다. 고정 context set에서 query-target의 native incidence는 byte 단위로 동일하다. 기대 edge density는 0.25로 같지만 실제 degree를 고정한 실험은 아니다. Context edge 변화는 conditional query 계산 경로도 바꾸므로 posterior weight만의 순수 개입이라고 해석하지 않는다.

### Exclusive / 총 context 4개 / 새로운 master 512개

| Context signal | Label 추가 정보 | Prior→exact 기대 NLL 이득 | Graph-only→exact 이득 | Posterior max ≥0.9 비율 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00000 | 0.00000 | 0.00000 | 0.0% |
| 0.36 | 0.01194 | 0.00184 | 0.00087 | 0.0% |
| 0.72 | 0.12688 | 0.02388 | 0.01634 | 13.2% |
| 0.96 | 0.27420 | 0.07008 | 0.04151 | 58.5% |

Context signal=0에서는 label 추가 정보와 posterior 업데이트의 예측 가치가 수치 오차 수준으로 0이다. Signal=0.72에서 기존 graph의 정보량·예측 가치가 새 master에서도 재현되고, 0.96에서는 더 커진다. 그러나 총 4개·signal 0.96에서도 posterior가 0.9 이상 집중하는 비율은 약 58.5%이므로 완전히 Identifiable한 조건이라고 부르지 않는다.

### 가장 강한 signal / 총 context 6개

| Prior | Graph 정보 | Label 추가 정보 | 전체 정보 | Prior→exact NLL 이득 | 집중 비율 ≥0.9 |
|---|---:|---:|---:|---:|---:|---:|
| independent | 0.50254 | 0.44213 | 0.94468 | 0.08687 | 48.6% |
| exclusive | 0.17652 | 0.37620 | 0.55273 | 0.10177 | 82.8% |
| coupled | 0.47999 | 0.14879 | 0.62878 | 0.11078 | 92.8% |

Coupled에서는 signal이 커지면 graph 자체가 role을 상당히 알려주므로 label의 **잔여 추가 정보**는 단조 증가하지 않을 수 있다. 전체 정보량과 conditional MI를 분리해야 하는 또 하나의 이유다.

**Frozen learned head와 Plain의 sweep 수치도 전부 저장했지만, 동일 prior로 재학습한 비교는 아니다.** Checkpoint가 학습한 조건은 signal=0.72·총 context 1/2/4개뿐이다. 다른 signal과 총 6개에서의 수치는 OOD transfer 진단이다. 특히 oracle는 해당 signal을 알고 계산하지만 frozen 모델에는 signal metadata를 새로 넣지 않는다. 따라서 강한 signal에서 frozen head가 oracle보다 낮다는 이유만으로 그 새 prior의 학습 불가능성을 주장하지 않는다.

## 5. 연구 판정

“Exact posterior도 prior와 비슷해서 inference 자체가 필요 없다”는 설명은 현재 Exclusive의 총 2/4개 조건을 설명하지 못한다. 정확한 role 업데이트의 정보량과 예측 가치가 모두 존재한다. 따라서 posterior inference를 연구할 가설적 근거는 남는다.

다만 현재 모델은 learned posterior로 일부 이득을 회수하며, exact posterior로 교체해도 conditional decoder의 큰 오차가 남는다. 다음 메소드의 목표는 단순히 joint posterior head를 붙이는 것이 아니라 **role별 conditional prediction과 role evidence 추정을 함께 맞추는 것**이어야 한다. 이 분석 자체를 새로운 학습 원리의 novelty라고 주장하지 않는다.

총 라벨 1개는 role posterior 업데이트가 구조적으로 불가능하므로 새 router의 성공 조건으로 삼지 않는다. 기존 클래스당 k=1 문제를 주장하려면 그 sampling protocol과 likelihood를 별도로 맞춰야 한다. 미지의 prior·role 의미·schema·실제 heterograph의 일반화는 아직 검증하지 않았다.

## 6. 검증과 재현

- 새로운 optimizer update와 checkpoint 생성은 0개다. 기존 source·bank·checkpoint·저장 확률의 hash를 검사했다.
- 총 57개 확률 archive: 기존 9개 graph/context cell과 새 sweep 48개 cell, 합계 26,304 graph-context 사례.
- 모든 helper/query latent label 2^12개와 모든 context label pattern 2^k개를 주변화한다.
- 독립 scalar 열거로 posterior·conditional MI·기대 예측 가치를 검증했다. 최대 오차는 약 2.8e−16.
- MI chain rule, entropy chain rule, zero-signal null, one-label null, relation/class permutation을 검사했다.
- 원래 signal에서 원 생성기와 graph가 정확히 같고, query incidence pairing도 검사했다.
- [최종 감사](results/audit.json), [전체 분석과 CI](results/analysis.json), [고정 프로토콜](protocol.json), [입출력 hash manifest](results/manifest.json).

```powershell
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/posterior_shift/checks.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/posterior_shift/run_shift.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/posterior_shift/analyze_shift.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/posterior_shift/audit_shift.py
./relational_baselines/.venv-rt/Scripts/python.exe joint_role_pfn/posterior_shift/report_shift.py
```

Runner는 기존 manifest를 덮어쓰지 않는다. 재실행은 별도의 새 결과 경로에서 한다.

![Posterior information and predictive value](results/posterior_shift.png)

# Contrastive Relational Decoder — frozen representation kill test

> 2026-11-01 · 새 학습 0회 · 소형 native bipartite SBM · 개발용 표현력 검증

**판정: decoder 표현력 변경은 Go. 학습된 새 HeteroGraphPFN이나 논문 novelty가 입증된 것은 아니다.**

Positive diffusion의 한 방향 제약을 없앴을 뿐 아니라, 같은 frozen learned posterior를 유지한 decoder 교체에서도 Bayes KL이 크게 감소했다. 같은 contrastive decoder의 exact joint vs factorized 대조는 Exclusive에서만 확실한 증분을 보였다. 알려진 생성모형의 확률과 역할 집합을 이용한 분석이므로 일반적인 PFN 성능 개선과 구분한다.

![Decoder comparison](results/decoder_comparison.png)

## 1. 고정 프로토콜과 접근 권한

- 새 graph seeds: 9,100,000 / 9,200,000 / 9,300,000부터 prior별 256개, 총 768개. 기존 train/eval seed 범위와 다르다.
- Target 8개, helper 4개, binary class, relation 2개. Feature는 전부 없다. Context는 **총** 1/2/4개이며 클래스당 개수가 아니다.
- 존재 변수 B와 관측 mask O를 분리했다. MCAR 관측률 1.0/0.5/0.2, 같은 graph/context 및 nested masks로 총 27개 cell, 6,912 graph-context-mask 조합을 평가했다.
- BP는 64회, damping 0.5, zero messages로 고정했다. Signed diffusion은 64회, 계수0.9로 고정했다. 학습·계수 탐색·checkpoint 선택은 없다.
- Positive D와 Large Plain은 기존 최종 checkpoint 3 seeds(17/29/43)를 그대로 사용했다. 원래 모델은 O 입력이 없어 full observation에서만 비교했다.
- 모든 analytic decoder가 참 density0.25, signal0.72, 역할 집합을 안다. `joint`는 **exact role posterior oracle**, `factor`는 그 주변확률의 곱, `prior`는 known prior다. `learnedq`만 기존 frozen posterior다. 이것은 접근 정보가 같은 새 학습 모델 비교가 아니다.
- 각 seed의 지표를 평균했으며 예측확률 ensemble을 만들지 않았다. Context를 graph 안에서 평균하고 4,000회 paired graph bootstrap을 했다. CI는 fitted model seeds에 조건부다.

## 2. 실제 decoder 연산

잠재 spin $s_v\in\{-1,+1\}$에서 $p_{r,\pm}^{(h)}=d(1\pm\eta h_r)$로 둔다. 관측된 native pair의 coupling은 다음과 같다.

$$J_{it}^{(h)}=\frac12\sum_r O_{rit}\left[B_{rit}\log\frac{p_{r,+}^{(h)}}{p_{r,-}^{(h)}}+(1-B_{rit})\log\frac{1-p_{r,+}^{(h)}}{1-p_{r,-}^{(h)}}\right].$$

그러면 $p(s\mid B,O,h,Y_C)\propto\mathbf1[s_C=2Y_C-1]\exp(\sum_{it}J_{it}^{(h)}s_i s_t)$다. Helper label은 알려주지 않고 sum-product 메시지로 합산한다. 관계 채널은 동일 pair의 coupling에 합쳐지며 target-target path를 독립 Bernoulli로 취급하지 않는다.

$$u_{i\to j}^{(h)}\leftarrow\operatorname{atanh}\left[\tanh J_{ij}^{(h)}\tanh\left(\sum_{v\ne j}u_{v\to i}^{(h)}\right)\right].$$

Context의 cavity spin은 관측 label로 고정한다. 위 candidate와 이전 message를 0.5씩 섞어 고정 횟수만큼 계산하고 node marginal을 얻는다. 트리에서는 정확하지만 loopy graph에서는 근사다. Role posterior 계산에는 spin-independent likelihood 상수도 유지하며, 조건부 decoder에서만 그 상수가 상쇄된다.

최종 확률은 $\sum_h q(h\mid B,O,Y_C)\,p_{\mathrm{BP}}(y_q\mid B,O,Y_C,h)$이며 **확률을 평균**한다.

| 대조군 | 연산 / 분리하는 효과 |
|---|---|
| Frozen positive D | 이전 양의 diffusion decoder, learned q |
| Signed diffusion | Signed J의 절댓값 degree로 정규화한 선형 전달, context clamp. 참 확률계수를 알지만 likelihood 추론은 아님 |
| Context-star | Context와 해당 query만 남기고 helper를 log-cosh로 정확히 합산. 나머지 unknown target의 구조적 증거를 버림 |
| Edge-only BP | 같은 BP에서 nonedge coupling만 제거. 같은 exact q 비교는 **decoder 항**만 분리하며 q 경로에는 nonedge 정보가 남음 |
| Edge/nonedge BP | 관측된 양·음 coupling을 함께 사용 |
| Prior / factor / joint | **같은 decoder**에서 역할 불확실성의 집계만 교체 |
| Exact conditional / exact q | 모든 2^12 label 상태를 열거한 별도의 Bayes 기준. 제안 decoder가 아님 |

Known prior를 공통으로 쓴 edge-only vs edge/nonedge 대조도 포함해, exact q에 남아 있는 nonedge 정보와 별도로 absence 항의 가치를 확인했다.

## 3. 주 결과: full observation, context 1/2/4 평균

Bayes predictive KL (nats, 낮을수록 좋음):

| Decoder / aggregation | Independent | Exclusive | Coupled |
|---|---:|---:|---:|
| Frozen Large Plain | 0.066551 | 0.061710 | 0.090523 |
| Frozen positive D + learned q | 0.062244 | 0.048200 | 0.091293 |
| Frozen positive D + exact q | 0.051393 | 0.040829 | 0.076300 |
| Signed diffusion + exact joint | 0.046934 | 0.037511 | 0.074036 |
| Context-star + exact joint | 0.007772 | 0.007635 | 0.015853 |
| Edge-only BP + exact joint | 0.083179 | 0.052283 | 0.130809 |
| Contrast BP + known prior | 0.018365 | 0.012646 | 0.035223 |
| Contrast BP + exact factorized | 0.003540 | 0.007828 | 0.006817 |
| Contrast BP + frozen learned q | 0.014723 | 0.008665 | 0.028190 |
| Contrast BP + exact joint | 0.003699 | 0.002476 | 0.006146 |
| Exact Bayes | 0.000000 | 0.000000 | 0.000000 |

**같은 learned q를 유지한 decoder 교체**의 KL 차이, 새 decoder − 기존 positive D:

| Prior | 차이 | 95% paired graph CI |
|---|---:|---:|
| independent | -0.047521 | [-0.054638, -0.041361] |
| exclusive | -0.039534 | [-0.043449, -0.035635] |
| coupled | -0.063103 | [-0.072523, -0.054314] |

세 prior 모두 오차가 크게 감소했다. Decoder를 exact q로 공통 통제한 비교에서도 감소 CI가0을 포함하지 않았다. 다만 새 decoder가 참 생성 확률을 직접 이용한다는 접근 권한 차이는 남는다. 이 결과는 기존 함수족의 병목을 확인하고, 학습해야 할 연산을 제시하는 증거다.

### Accuracy와 proper scores

| Prior | 모델 | Accuracy (%) | NLL | Brier | Bayes decision regret |
|---|---|---:|---:|---:|---:|
| independent | Frozen positive | 55.95 | 0.67235 | 0.47993 | 0.09388 |
| independent | Contrast / learned q | 64.43 | 0.62543 | 0.43607 | 0.00496 |
| independent | Contrast / exact joint | 65.87 | 0.60995 | 0.42554 | 0.00100 |
| independent | Exact Bayes | 65.88 | 0.60683 | 0.42287 | 0.00000 |
| exclusive | Frozen positive | 54.94 | 0.67372 | 0.48117 | 0.08279 |
| exclusive | Contrast / learned q | 63.76 | 0.63388 | 0.44363 | 0.00508 |
| exclusive | Contrast / exact joint | 64.33 | 0.63070 | 0.43971 | 0.00084 |
| exclusive | Exact Bayes | 64.28 | 0.62757 | 0.43739 | 0.00000 |
| coupled | Frozen positive | 56.54 | 0.67023 | 0.47785 | 0.10130 |
| coupled | Contrast / learned q | 64.56 | 0.61647 | 0.42882 | 0.00162 |
| coupled | Contrast / exact joint | 64.56 | 0.60280 | 0.41901 | 0.00162 |
| coupled | Exact Bayes | 64.68 | 0.59762 | 0.41622 | 0.00000 |

Accuracy는 기존과 같은 argmax를 사용했다. 정확한 동률은 class0을 택하므로 label-swap에 따른 sampling 효과가 있을 수 있다. 별도로 tie-averaged accuracy도 JSON에 기록했다. Bayes decision regret는 동률을 균등하게 처리한 기대 정확도 손실이다. 근사 BP의 표본 accuracy가 Exact Bayes보다 조금 높아도 Bayes보다 우수함을 의미하지 않는다.

## 4. Joint-role uncertainty의 독립적인 효과

같은 contrastive BP에서 exact joint − product of exact posterior marginals의 KL 차이:

| Prior | 차이 | 95% CI | 판단 |
|---|---:|---:|---|
| independent | +0.000159 | [-0.000121, +0.000536] | 증분 불확실 |
| exclusive | -0.005352 | [-0.006113, -0.004559] | joint 증분 확인 |
| coupled | -0.000671 | [-0.001726, +0.000555] | 증분 불확실 |

Exclusive에서는 역할의 배타 제약을 깨는 factorization이 손해다. Independent도 graph/context를 본 posterior가 반드시 독립인 것은 아니다. 그러나 이번 근사 decoder의 최종 metric에서 그 차이가 작았고, factorized 혼합의 우연한 근사오차 상쇄도 가능하다. Coupled 역시 joint의 일반적 우월성을 주장할 근거는 부족하다.

Known prior보다 exact joint가 좋은 KL 차이는 세 prior 모두 CI가0을 포함하지 않았다. 반면 이것은 **exact posterior를 공급한 효과**이며 새 posterior network가 그만큼 학습했다는 뜻이 아니다.

## 5. 단일 context에서 표현력 제약이 사라졌는가?

관측된 context class의 반대 클래스를 top-1으로 예측하는 query 비율 (총 context 1개):

| Prior | Positive D | Contrast / learned q | Contrast / exact joint | Bayes가 반대 클래스를 선호 |
|---|---:|---:|---:|---:|
| independent | 0.00% | 51.23% | 51.34% | 52.12% |
| exclusive | 0.00% | 49.61% | 50.11% | 49.55% |
| coupled | 0.00% | 49.72% | 49.72% | 49.78% |

이는 정답 비율이 아니라 **Bayes의 decision 방향**에 대한 분석이다. Exclusive의 exact joint BP는 Bayes가 반대 클래스를 선호하는 query의95.72%를 포착하며, 자신이 반대 클래스를 고른 경우94.65%가 Bayes와 방향이 같다.

단일 context A–helper edge, query–helper measured nonedge인 트리 예제에서는 반대 클래스 확률이0.5864였다. Positive D의 반대 클래스 선택은 수치 허용오차1e−6 기준0건이다. 최초 분석 assertion은 float32의 약1e−8 roundoff를 위반으로 셌으므로 정규화와 명시적 허용오차로 수정했다. 예측값이나 decoder를 재학습·수정한 것은 아니다.

새 graph에서 계산한 positive-family 부분 KL 하한과 BP 결과:

| Prior | Positive family 부분 하한 | Contrast BP + exact joint KL |
|---|---:|---:|
| independent | 0.017074 | 0.003699 |
| exclusive | 0.014914 | 0.002476 |
| coupled | 0.028230 | 0.006146 |

이 부분 하한보다 낮은 KL은 새 decoder가 기존 positive-only 함수족 밖의 예측을 실제로 활용했다는 증거다. 하한은 기존 모델의 모든 오차를 설명하지는 않는다.

## 6. Unknown pair를 nonedge로 간주하면 안 된다

| Prior | 관측률 | 올바른 mask BP KL | 잘못된 zero-fill BP KL |
|---|---:|---:|---:|
| independent | 0.5 | 0.000570 | 0.018721 |
| independent | 0.2 | 0.000003 | 0.061496 |
| exclusive | 0.5 | 0.000092 | 0.017857 |
| exclusive | 0.2 | 0.000001 | 0.062904 |
| coupled | 0.5 | 0.001255 | 0.028514 |
| coupled | 0.2 | 0.000206 | 0.078431 |

둘 모두 **올바른 O로 계산한 exact q**를 공통 사용했다. Zero-fill 대조만 unknown pair를 observed zero로 decoder에 넣어, decoder의 missingness 처리 오류를 분리했다. Hidden B 값을 바꿔도 올바른 likelihood와 coupling이 그대로인 것도 확인했다.

관측률이 낮아질수록 올바른 BP의 Bayes KL이 작아지는 것은 정보가 늘어서가 아니다. 추론 graph의 loop가 줄고 Bayes target이 더 uniform에 가까워져 근사가 쉬워지는 효과가 함께 있다. Masked 조건의 낮은 KL을 더 높은 분류 성능으로 해석하지 않는다. MCAR 밖의 informative edge censoring은 검증하지 않았다.

## 7. 근사 오차와 검증

다음 검증이 통과했다: 독립 scalar enumeration과 masked exact posterior/conditional 일치, native likelihood와 Ising 표현의 상수항 일치, full observation에서 기존 likelihood와 일치, O=0 값 변경 불변성, 관측0개에서 query uniform, native tree의 BP exactness, relation 및 class permutation 불변성.

최대 probability 합 오차는3.58e-07, 독립 scalar/reference 오차는2.22e-16다. 새 optimizer update는0회이며 입력 source와18개 checkpoint의 hash가 계산 전후 동일했다.

**BP가 완전히 수렴한 것으로 보고하지 않는다.** 64회 고정 연산 후 residual>1e−8인 role의 posterior mass는 full-observation Exclusive에서 context1/2/4 각각 다음과 같다: 83.18%, 44.12%, 10.86%. 특히 context1에서 반복 잔차가 남는다. 낮은 marginal KL은 이 고정 연산의 관측 성능이지 fixed-point 수렴 보장이 아니다. Iteration 수를 test 결과에 맞춰 바꾸지 않았다.

## 8. 연구 판정과 다음 메소드의 경계

1. **Positive-only decoder의 병목은 확인됐다.** 같은 learned q에서 likelihood 기반 decoder 교체가 큰 개선을 주었다. 단순 signed diffusion보다 적절한 양·음 likelihood 전달이 더 정확했다.
2. **Decoder를 고친 뒤에도 Exclusive의 joint-role 보존 효과가 남는다.** 다만 세 prior에 보편적인 joint 우위로 확장하지 않는다.
3. **아직 새 학습 메소드의 성과는 아니다.** 알려진 SBM의 sum-product는 기존 추론 연산이다. 새 이름이나 nonedge 항 자체로 novelty를 주장하지 않는다.
4. **다음 단계의 구체적 학습 대상:** O를 명시적으로 받는 role-conditional signed likelihood potential을 학습하고, 고정된 cavity/sum-product 계산을 통해 예측한다. Density/signal을 직접 준 현재 oracle potential과의 차이를 측정해야 한다. 이 실험에서 그 후속 모델을 학습하지는 않았다.
5. 후속 학습에서는 같은 데이터·예산의 positive decoder, signed decoder, generic MLP/Plain distillation 및 joint/factorized/prior를 유지해야 한다. 실제 schema로 갈 때는 edge observation process를 별도로 모델링해야 한다.

## 산출물

- [고정 protocol](protocol.json), [native likelihood / decoder 구현](contrastive_ops.py)
- [평가 실행](execute_contrastive.py), [사전 독립 검증](contrastive_checks.py), [분석](analyze_contrastive.py)
- [전체 지표](results/aggregate.json), [context별 지표](results/cell_metrics.json), [paired CI](results/paired_contrasts.json)
- [역할별 조건부 KL](results/conditional_metrics.json), [단일 context 방향 분석](results/one_context_opposite.json)
- [수렴 잔차](results/convergence.json), [난이도·표현력 하한](results/difficulty_and_bounds.json), [audit](results/audit.json)
- [PNG](results/decoder_comparison.png), [PDF](results/decoder_comparison.pdf), [입력 hash / cell manifest](results/manifest.json)

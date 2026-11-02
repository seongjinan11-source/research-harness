# J / V 관측 prior — 학습 결과

2026-11-02. **두 prior를 모두 실제로 학습했다.** 새30개 공개 GraphPFN 기반 checkpoint, 총6,912 optimizer updates다. 기존 모델에 사후 mask 선택만 적용한 실험이 아니다. Primary는 새64 base graph, source4/96개, 총 context8개, 자기 피처 없는 query24개의 accuracy다. 3개 학습 seed를 graph 안에서 평균했다. 이전 192-node Gaussian/probit 결과와 생성 family·입력·context sampling이 달라 절대 정확도를 빼서 전후 개선으로 해석하지 않는다.

## Primary 결과

| 학습 prior | 입력 | Accuracy % | NLL | Brier |
|---|---|---:|---:|---:|
| Random | mask | 55.2951 | 0.690357 | 0.495384 |
| 거리/coverage | mask | 55.3819 | 0.688894 | 0.494924 |
| J: entropy 감소 | mask | 55.7943 | 0.690654 | 0.495232 |
| V: 분류 오류 감소 | mask | 55.6641 | 0.690199 | 0.495521 |
| Random, 2배 updates | mask | 56.7057 | 0.683048 | 0.489159 |
| Random | pcfi | 56.9878 | 0.680178 | 0.486830 |
| 거리/coverage | pcfi | 56.9227 | 0.680114 | 0.486976 |
| J: entropy 감소 | pcfi | 57.3134 | 0.680086 | 0.486717 |
| V: 분류 오류 감소 | pcfi | 56.9444 | 0.680549 | 0.487088 |
| Random, 2배 updates | pcfi | 57.5304 | 0.673814 | 0.480939 |

각 candidate와 같은 입력 방식의 대조를 짝지은 차이다. CI는64graph paired bootstrap5000회이며3개 seed를 고정한다. 개별95% CI이며 multiplicity 보정은 하지 않았다. J/V 각각 Mask에서 세 대조를 모두 넘어야 primary 통과다.

| 후보 | 대조 | Accuracy 차이 %p [95% CI] | NLL 차이 [95% CI] |
|---|---|---:|---:|
| entropy_mask | random_mask | +0.4991 [-0.3472, +1.3455] | +0.000297 [-0.001776, +0.002524] |
| entropy_mask | geometry_mask | +0.4123 [-0.5642, +1.3889] | +0.001760 [-0.001956, +0.006402] |
| entropy_mask | random_cost_mask | -0.9115 [-2.0839, +0.2393] | +0.007606 [+0.003210, +0.012690] |
| decision_mask | random_mask | +0.3689 [-0.7812, +1.6059] | -0.000158 [-0.003208, +0.002855] |
| decision_mask | geometry_mask | +0.2821 [-0.8898, +1.5408] | +0.001305 [-0.001898, +0.004584] |
| decision_mask | random_cost_mask | -1.0417 [-2.3437, +0.2604] | +0.007151 [+0.003281, +0.011135] |
| entropy_pcfi | random_pcfi | +0.3255 [-0.4340, +1.0851] | -0.000092 [-0.001843, +0.001640] |
| entropy_pcfi | geometry_pcfi | +0.3906 [-0.4340, +1.1936] | -0.000029 [-0.001995, +0.002023] |
| entropy_pcfi | random_cost_pcfi | -0.2170 [-1.6493, +1.1719] | +0.006272 [+0.000944, +0.011972] |
| decision_pcfi | random_pcfi | -0.0434 [-0.9115, +0.8464] | +0.000371 [-0.002393, +0.003273] |
| decision_pcfi | geometry_pcfi | +0.0217 [-0.8681, +0.9332] | +0.000435 [-0.002872, +0.003735] |
| decision_pcfi | random_cost_pcfi | -0.5859 [-1.7361, +0.5208] | +0.006735 [+0.002548, +0.011092] |

**사전 primary 판정: J=False, V=False.** 정확도 gate를 NLL/Brier나 특정 subgroup으로 대체하지 않는다. CI가0을 포함하는 것은 동등성 증명이 아니다.

## 관측 위치가 실제로 달라졌나?

두 후보는 source budget·graph·feature/label law·초기화·CE를 유지하고 학습 mask prior만 바꾼다. 각 후보는50% random과50% 점수 상위3개 mask 중 균일 선택의 mixture다. 학습에 사용한256-draw 점수는 선택 편향을 포함할 수 있으므로 아래는 별도4096-draw로 계산한 첫16 train graph primary 설계의 평균이다. 이는 새 test bank의 모델 선택 지표가 아니며 이 결과를 보고 mask를 바꾸지 않았다.

| Mask prior | 독립 감사 J | 독립 감사 V |
|---|---:|---:|
| Random | 0.062977 | 0.049572 |
| 거리/coverage | 0.057273 | 0.044977 |
| J: entropy 감소 | 0.063937 | 0.050298 |
| V: 분류 오류 감소 | 0.063891 | 0.050232 |

실제 J/V 선택 mask 일치율은6개 context×budget cell에서 73.44~87.50%였다. J와 V는 다른 정의지만 이번 finite family에서는 설계를 크게 구분하지 못했다. 이 결과로 일반적으로 J와 V가 같다고 주장하지 않는다.

## Feature를 활용했나?

같은 primary query를 유지하고 관측값을 모두 제거하거나, 관측 행끼리 값을 순열했다. 셔플에서는 PCFI를 다시 계산하고, 모든 피처 제거에서는 all-missing PCFI와 같은0행렬을 쓴다. 다음은 standard−intervention 정확도 차이다. Primary와 독립인 메소드 성공 판정으로 승격하지 않는다.

| Arm | 피처 제거 대비 %p [95% CI] | 피처 셔플 대비 %p [95% CI] |
|---|---:|---:|
| random_mask | -0.0434 [-1.3889, +1.3455] | +0.0651 [-1.3021, +1.4757] |
| geometry_mask | -0.1302 [-1.5191, +1.3021] | +0.3906 [-0.9983, +1.8229] |
| entropy_mask | +0.6076 [-1.1507, +2.3655] | +0.1085 [-1.0417, +1.2587] |
| decision_mask | +0.4774 [-1.2370, +2.3438] | -0.3689 [-1.7795, +1.0200] |
| random_cost_mask | +0.7812 [-0.9332, +2.5174] | +0.1519 [-1.2592, +1.5408] |
| random_pcfi | +1.2587 [-0.7161, +3.1467] | +0.9115 [-1.1285, +2.8212] |
| geometry_pcfi | +1.0417 [-0.9115, +2.9080] | +1.1719 [-0.7595, +2.9297] |
| entropy_pcfi | +1.4974 [-0.5859, +3.3854] | +1.4323 [-0.9337, +3.5807] |
| decision_pcfi | +0.9766 [-0.8247, +2.7995] | +1.0200 [-0.9120, +2.7561] |
| random_cost_pcfi | +1.1936 [-0.9115, +3.3203] | +0.4991 [-1.6710, +2.4957] |

## 전체 관측 조건과 seed

다음 accuracy는 사전 고정한 모든 standard setting이다. Primary와 동일한 역할로 판정하지 않는다. NLL/Brier는 results/statistics.json의 all_cells에 함께 남겼다.

### mask

| 조건 | Random | Geometry | J | V | Random 2배 |
|---|---:|---:|---:|---:|---:|
| clustered_c8_o4_standard | 54.6007 | 55.2951 | 54.7309 | 55.1866 | 54.7309 |
| full_c8_o96_standard | 57.0095 | 56.6189 | 56.6840 | 56.7925 | 57.6389 |
| uniform_c2_o1_standard | 51.3672 | 51.8012 | 51.1502 | 52.1701 | 51.9965 |
| uniform_c2_o4_standard | 49.5660 | 48.8064 | 49.2839 | 48.9800 | 50.6727 |
| uniform_c2_o8_standard | 51.0417 | 52.0399 | 51.2804 | 51.2370 | 51.2370 |
| uniform_c8_o1_standard | 54.1450 | 53.9714 | 54.5356 | 54.4705 | 55.3385 |
| uniform_c8_o4_standard | 55.2951 | 55.3819 | 55.7943 | 55.6641 | 56.7057 |
| uniform_c8_o8_standard | 52.6476 | 52.8212 | 53.1467 | 53.2335 | 53.4505 |

### pcfi

| 조건 | Random | Geometry | J | V | Random 2배 |
|---|---:|---:|---:|---:|---:|
| clustered_c8_o4_standard | 57.3568 | 57.0964 | 57.3785 | 57.3785 | 58.3767 |
| full_c8_o96_standard | 58.6155 | 58.3984 | 58.8325 | 58.8976 | 59.6354 |
| uniform_c2_o1_standard | 52.6042 | 52.1701 | 52.4306 | 52.5825 | 52.1918 |
| uniform_c2_o4_standard | 50.2821 | 50.2387 | 50.1085 | 50.0868 | 50.3472 |
| uniform_c2_o8_standard | 52.8212 | 52.6476 | 52.8863 | 52.4957 | 53.3203 |
| uniform_c8_o1_standard | 55.2300 | 54.5790 | 55.1649 | 55.2734 | 56.1632 |
| uniform_c8_o4_standard | 56.9878 | 56.9227 | 57.3134 | 56.9444 | 57.5304 |
| uniform_c8_o8_standard | 54.8828 | 54.6224 | 55.0998 | 54.9479 | 55.7726 |

각 seed의 primary accuracy도 그대로 보고한다. Final checkpoint를 seed별로 고르거나 ensemble probability를 만든 값이 아니라, graph별 metric을 seed 평균한 비교다.

| Arm | seed11 | seed12 | seed13 |
|---|---:|---:|---:|
| random_mask | 55.5990 | 55.3385 | 54.9479 |
| geometry_mask | 55.0781 | 55.4688 | 55.5990 |
| entropy_mask | 55.5990 | 54.5573 | 57.2266 |
| decision_mask | 55.0781 | 54.8828 | 57.0312 |
| random_cost_mask | 57.3568 | 55.9896 | 56.7708 |
| random_pcfi | 56.9010 | 57.0964 | 56.9661 |
| geometry_pcfi | 56.4453 | 58.0729 | 56.2500 |
| entropy_pcfi | 57.9427 | 57.6823 | 56.3151 |
| decision_pcfi | 56.9010 | 57.6172 | 56.3151 |
| random_cost_pcfi | 57.4870 | 57.3568 | 57.7474 |

## Known-family reference

Teacher는 graph에서 정해지는64개 latent state를 모두 열거한다. Gaussian feature noise와 Bernoulli label 생성 law는 reference와 일치한다. Posterior 계산은 이 유한 family에서 exact지만 관측의 기대 가치는 Monte Carlo 추정이다. 실제 full latent realization을 reference에 주지 않는다. 별도 Gaussian/probit 논문 구현이나 Grid6의 재현이 아니다.

| Reference | Accuracy % | NLL | Brier | 평균 posterior 최대확률 % |
|---|---:|---:|---:|---:|
| exact_family | 74.3490 | 0.534618 | 0.359182 | 72.1165 |
| exact_no_features | 70.3125 | 0.583000 | 0.398604 | 67.0610 |
| context_frequency | 50.5208 | 0.720915 | 0.525781 | 59.3750 |

Exact-family의 동일-query feature gain: +4.0365 [95% CI +1.8880, +6.3802]%p. 관측 feature가 쓸모 있는 family라는 사실과, 학습 observation prior를 바꾸는 것이 유리하다는 주장은 구분한다.

## 계산·재현·범위

- Train bank:64graph×6context/source cell×4prior =1,536episodes. Test bank:64graph×8setting =512episodes. Neural standard/ablation probabilities19,200배열, reference1,536배열.
- 일반 arm192updates, random_cost384updates. 두 입력 방식·다섯 prior·세 seed를 실제로 학습했다. 모델마다 trainable4,012,259개로 같다. 초기 상태·일반 arm schedule·frozen parameters·최종 checkpoint hash·실제 update 수와 저장된 모든 metric을 재검사했다. 최종 checkpoint30개를 각1회 replay했다.
- Train bank 전체 cold 생성 21.988s, 그중 모든 후보 J/V scoring 10.709s. 공유 계산을 한 후보에 전부 청구하는 보수적 비교도 남겼다.
- mask: random 2배 학습의 추가 시간 47.507s; 전체 cold bank 생성 시간을 덮는가=True. Seed 작업이 같은3090에서 겹쳐 실행됐으므로 isolated latency 결과는 아니다.
- pcfi: random 2배 학습의 추가 시간 32.198s; 전체 cold bank 생성 시간을 덮는가=True. Seed 작업이 같은3090에서 겹쳐 실행됐으므로 isolated latency 결과는 아니다.
- Primary는95.83% whole-node missing이다. Source1/96은98.96%, source8/96은91.67% missing이다. Total context2/8은 class-balanced sampling이 아니다. Query의 실제 정답과 latent mechanism은 mask 설계기의 인자가 아니다.
- 실제 X/Y draw는 코드상 graph 설계 loop 전에 생성하지만 별도 RNG이고 선택 함수에 전달되지 않아 mask와 독립이다. Protocol의 “after selection”은 시간 순서 표현이 부정확했다. 원 source/lock/bank를 바꾸지 않고 runtime/scope_errata.json에 정정했다.
- Context/source/query는 full reference를 제외하고 완전히 분리한다. 각 모델은 test graph에서 학습하거나 mask를 골라주지 않는다. 최종 random·clustered·full·feature-ablation 입력은 모든 arm에 공통이다.
- 새로운 graph seed의 동일 finite family다. 실제 graph·미학습 schema·feature law·무한 latent GP나 과거 공식 prior 전체에 대한 결론이 아니다. 짧은 학습·작은 family·mask score 추정·학생 fitting의 원인을 이 실험만으로 분리하지 않는다.
- 이번에 실제로 재학습한 pipeline 대조는 PCFI다. FISF-PFN·다른 imputer의 동일-prior 학습이나 실제 graph 전이는 실행하지 않았다.
- Source와 protocol은 final bank 생성 전에 잠갔다. 독립 MC 감사·통계·보고서 코드는 runtime/에 별도로 남겼고 학습 코드나 mask/최종 checkpoint를 바꾸지 않았다.

## 판정

**현재의 J/V mask prior를 메인 메소드로 확대하지 않는다.** 두 설계의 기대 정보 가치는 약간 높일 수 있었지만, 이것만으로 같은 PFN의 heldout accuracy가 강한 random·geometry·extra-training 대조를 안정적으로 넘는다는 기준은 통과하지 못했다. 이는 관측 가치 설계 전체의 불가능성이나 “feature가 필요 없다”는 결론이 아니다. 실제 실행한50:50 top3-mixture·finite family·192-update 범위를 닫는다.

추가 loss/head나 같은 bank에서 mask 분율·candidate 수·update 수를 바꾸는 구조 구제는 하지 않는다. 재개하려면 기존 결과와 다른 구체적인 정보/계산 병목을 제시하고, 단순 geometry와 같은 비용의 random training을 유지한 새 독립 프로토콜이 필요하다.

원시 source·protocol·bank·30 checkpoint·모든 확률은 workspace에 보존했다. 공유 snapshot에는 raw bank와 checkpoint를 제외하고 계산 정의·source·lock·설계 감사·전체 조건·seed별 통계·원본 보고서를 포함한다.

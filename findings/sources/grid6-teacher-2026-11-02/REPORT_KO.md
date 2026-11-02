# Grid6 exact teacher — 새 disjoint bank 비교

2026-11-02. 사전 primary는 공개 Mask GraphPFN의 exact-Grid6 predictive KL이다. 동일 생성 prior에서 sampled-label CE를 soft predictive KL로 바꿨다. Factor 학생은 보조 비교이고 primary 실패를 대체하지 않는다.

## 실제 학습과 평가

- 새 모델6개: Mask/Factor 각3 seeds. 각2048 기존 train tasks,256 successful updates. 원래 초기화·task order·optimizer·learning-rate schedule·trainable parameter 수를 CE arm과 일치시켰다.
- 기존 CE Mask/PCFI/Factor9개 checkpoint를 새96 graph bank에서 재평가했다. 이전 학습 결과에 추가 학습한 것이 아니다. Train tasks는 재사용했고 test seed만 새로 만들었다.
- Teacher는 실제 task parameter를 받지 않는 exact Grid6다. 네 role·여섯 parameter·네 feature orientation과 모든 미공개 node labels를 주변화한다. Student와 teacher는 graph/관측값/mask/context/family만 받는다.
- 총1152 standard episodes. Primary는 matched48 graphs의 feature–label disjoint 조건이며 context 총1/2/4개, 관측노드3/6개를 graph 내 평균한다. 세 fixed training seed도 graph 내 평균하고5000 base-graph bootstrap을 사용한다.

## Primary 결과

| 방법 | Accuracy % | NLL | Brier | Grid6 exact KL |
|---|---:|---:|---:|---:|
| mask_pfn_ce | 55.3475 | 0.678973 | 0.486110 | 0.104788 |
| pcfi_pfn_ce | 55.7650 | 0.676898 | 0.483947 | 0.103836 |
| factor_pfn_ce | 69.8468 | 0.575272 | 0.385773 | 0.067257 |
| mask_pfn_distill | 54.6255 | 0.679253 | 0.486305 | 0.104134 |
| factor_pfn_distill | 69.5877 | 0.568642 | 0.385723 | 0.049731 |
| constant_factor | 67.5182 | 0.610127 | 0.400815 | 0.107381 |
| grid6_exact | 68.3862 | 0.551923 | 0.380938 | 0.000000 |
| grid6_bp | 68.3284 | 0.551872 | 0.380899 | 0.000010 |
| feature_only | 50.4340 | 0.693147 | 0.500000 | 0.113227 |
| lp | 51.5294 | 1.147039 | 0.647525 | 0.564661 |
| class_prior | 50.1157 | 0.766172 | 0.563465 | 0.176592 |
| uniform | 49.4378 | 0.693147 | 0.500000 | 0.113227 |

## Paired 결과

각 값은 distill−대조, 괄호는 paired95%CI다. Accuracy만 %p이며 나머지는 원 단위다. CI가0을 포함하는 것은 동등성이나 무효의 증명이 아니다.

### mask_pfn_distill − mask_pfn_ce

- accuracy: -0.722002 [-3.917376, +2.542232]
- nll: +0.000280 [-0.002496, +0.002959]
- brier: +0.000195 [-0.002491, +0.002782]
- grid6_kl: -0.000654 [-0.002034, +0.000583]

### factor_pfn_distill − factor_pfn_ce

- accuracy: -0.259039 [-1.211179, +0.755071]
- nll: -0.006630 [-0.023930, +0.007280]
- brier: -0.000050 [-0.007740, +0.007348]
- grid6_kl: -0.017526 [-0.026031, -0.010538]

### mask_pfn_distill − grid6_bp

- accuracy: -13.702877 [-19.538380, -8.235505]
- nll: +0.127381 [+0.076315, +0.179524]
- brier: +0.105406 [+0.063084, +0.148102]
- grid6_kl: +0.104124 [+0.075245, +0.136454]

### factor_pfn_distill − grid6_bp

- accuracy: +1.259369 [-0.953690, +3.579868]
- nll: +0.016771 [-0.016408, +0.053202]
- brier: +0.004823 [-0.015209, +0.026042]
- grid6_kl: +0.049721 [+0.037172, +0.064317]

## Secondary와 intervention

| 분할 | 학생 | Accuracy % | NLL | Grid6 KL |
|---|---|---:|---:|---:|
| overlap | mask_pfn_ce | 53.5260 | 0.680081 | 0.257836 |
| overlap | mask_pfn_distill | 54.5470 | 0.679627 | 0.257001 |
| overlap | factor_pfn_ce | 74.8236 | 0.468115 | 0.017969 |
| overlap | factor_pfn_distill | 74.6046 | 0.469867 | 0.024641 |
| overlap | grid6_bp | 74.8305 | 0.448360 | 0.000062 |
| density_low | mask_pfn_ce | 47.3765 | 0.705604 | 0.072484 |
| density_low | mask_pfn_distill | 50.1213 | 0.701567 | 0.070225 |
| density_low | factor_pfn_ce | 59.6092 | 0.702401 | 0.060628 |
| density_low | factor_pfn_distill | 60.7308 | 0.670134 | 0.036706 |
| density_low | grid6_bp | 62.3181 | 0.605519 | 0.000003 |
| signal_high | mask_pfn_ce | 63.7153 | 0.650077 | 0.271209 |
| signal_high | mask_pfn_distill | 66.9257 | 0.649939 | 0.272281 |
| signal_high | factor_pfn_ce | 85.1742 | 0.380521 | 0.067669 |
| signal_high | factor_pfn_distill | 85.4029 | 0.373177 | 0.057864 |
| signal_high | grid6_bp | 87.7728 | 0.279820 | 0.000012 |

Query 자체도 feature가 없는 분할은 `results/query_without_features.json`에 별도로 기록했다. 아래 intervention은 matched disjoint context2/observed6의 보조 진단이다. Feature 제거/셔플은 새로운 재학습이나 선택된 hybrid가 아니다.

- mask_pfn_distill, no_observed−standard accuracy: +1.041667 [-2.777778, +6.018519]%p
- mask_pfn_distill, shuffled−standard accuracy: -0.810185 [-2.777778, +0.925926]%p
- mask_pfn_distill, graph_permuted−standard accuracy: -4.282407 [-11.458333, +2.662037]%p
- factor_pfn_distill, no_observed−standard accuracy: -11.921296 [-19.097222, -4.861111]%p
- factor_pfn_distill, shuffled−standard accuracy: -20.717593 [-28.359375, -12.731481]%p
- factor_pfn_distill, graph_permuted−standard accuracy: -29.050926 [-39.583333, -17.939815]%p

## 비용과 판정

Primary teacher-effect gate=False; KL<=.01 gate=False; competitive compression gate=False.

Teacher2048 tasks의 offline 생성 wall time=15.313s. 새6개 학생 학습 wall time 합=951.223s이며 일부 worker가 동시에 실행됐으므로 이 합은 전체 elapsed time이나 통제된 CE 대비 속도 비교가 아니다. Teacher 입력·출력 bank I/O를 포함한 준비 비용이고 배포시 teacher 호출은 없다.
완료 후 동일GPU 단일 graph median: Mask distill59.735ms, Factor distill8.764ms, Grid6-BP12.425ms. Grid6/Mask 비율=0.208배. PCFI·baseline 구성 비용은 비교 범위에 따라 별도이며 teacher 생성 비용을 inference 절감에 숨기지 않는다. 외부GPU 부하는 통제하지 않았다.

## 해석 범위

Distillation 자체는 기존 기법이며 예전 feature-free JR에서도 시험했다. 이번의 다른 범위는 연속 Gaussian 관측과 label context가 분리된 과제에서, unknown-parameter six-support predictive를 직접 입력형 학생에 가르친다는 것이다. 이 단일 비교를 새로운 PFN 메소드의 novelty로 주장하지 않는다.
정확한 matched prior predictive는 해당 family 안의 기준이다. Density/signal OOD에서 Grid6 KL은 Grid6 reference에 대한 일치도이며 참 OOD Bayes posterior의 KL이 아니다. Accuracy·NLL·Brier·teacher KL을 분리한다.
12-node binary/2-relation/2-Gaussian-feature,50/75% whole-node missing에 한정한다. 실제 graph,99.5% missing, 새로운 feature law·semantic schema·큰 graph를 검증하지 않았다. Full-grid teacher의 exponential label enumeration을 확장 가능한 방법으로 부르지 않는다.
고유 실패 원인은 이 실험만으로 식별하지 않는다. Teacher를 바꿔도 개선이 작으면 표현력·최적화·입력 경로·prior 중 무엇 때문인지 추가 분리 없이 확정할 수 없다. 결과에 맞춰 loss·temperature·학습량·primary를 바꾸지 않았다.

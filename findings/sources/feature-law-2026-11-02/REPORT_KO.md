# Feature-law Factor 실험 결과

질문: 피처–클래스 관계가 그래프마다 달라질 때, 관측 column 통계에 적응하는 cell factor가 고정 법칙 주변화와 static 함수보다 유리한가?

## 판정

- 적응 accuracy gate: **False**. Adaptive가 original과 static 모두보다 primary accuracy CI 하한>0이어야 한다.
- Gaussian 2%p 비열등 guard: **True**. Guard 미확보를 Gaussian 붕괴의 증명으로 읽지 않는다.
- 최종 품질–비용 gate: **False**. 더 빠른 고정 대조는 fixed_exact, fixed/adaptive latency 비는 1.311다.

이는 이 구현·bank·256-update 예산의 판정이다. 실패한 gate를 teacher KL 개선·다른 subgroup·더 약한 baseline으로 대체하지 않는다. 새로운 head/loss/temperature/학습량을 추가해 구제하지 않는다.

## Primary: heldout tuple, 자기 피처가 없는 query

96 base graphs, 각6 episode와 고정된3 training seeds 평균. Accuracy는%, Brier는 두 클래스 제곱오차의 합이다. Reference KL은 broad feature-law prior와 native6 graph-parameter prior로 정의한 exact reference에 대한 KL이며 참 OOD Bayes KL이 아니다.

| 방법 | Accuracy % | NLL | Brier | Reference KL |
|---|---:|---:|---:|---:|
| 기존 부호 제약 Factor | 64.6281 | 0.643004 | 0.445450 | 0.022515 |
| Flexible-static | 64.4125 | 0.643406 | 0.446408 | 0.021705 |
| Flexible-adaptive | 64.4675 | 0.650797 | 0.449291 | 0.027830 |
| 고정 law 주변화 / exact | 63.4065 | 0.631744 | 0.440029 | 0.004696 |
| 고정 law 주변화 / BP | 64.0811 | 0.646232 | 0.446208 | 0.023510 |
| 실제 학습 Mask PFN | 53.4883 | 0.693767 | 0.499510 | 0.068593 |
| 실제 학습 PCFI-PFN | 52.9697 | 0.693928 | 0.499014 | 0.068545 |
| Native exact reference | 62.7249 | 0.630731 | 0.439466 | 0.000000 |

| 사전/보조 paired 비교 | Accuracy 차이 %p [95% CI] | NLL 차이 [95% CI] |
|---|---:|---:|
| adaptive-original | -0.1607 [-1.2025, +0.8215] | +0.007794 [+0.001050, +0.017646] |
| adaptive-static | +0.0550 [-0.9763, +1.0379] | +0.007392 [-0.000386, +0.017698] |
| static-original | -0.2156 [-0.7225, +0.2788] | +0.000402 [-0.002196, +0.003255] |
| adaptive-fixed_exact | +1.0610 [-0.7191, +2.7765] | +0.019053 [+0.001102, +0.041914] |
| adaptive-fixed_bp | +0.3864 [-0.8565, +1.5806] | +0.004565 [-0.002509, +0.012021] |
| adaptive-mask_pfn | +10.9791 [+7.3423, +14.6310] | -0.042970 [-0.081804, -0.001247] |
| adaptive-pcfi_pfn | +11.4977 [+7.7188, +15.2561] | -0.043131 [-0.082211, -0.001461] |

CI는 query/node를 독립 단위로 세지 않고 base graph 단위5000회 paired bootstrap했다. 세 seed는 고정 반복이며 seed 모집단의 변동을 추론한 CI가 아니다. 핵심 두 적응 비교 이외 subgroup/보조 비교의 multiplicity는 보정하지 않았다. CI에0이 있다는 것을 효과0 또는 동등성으로 읽지 않는다.

## Law와 파라미터 조건

| 조건 | Graphs | Original % | Static % | Adaptive % | Fixed exact % | Fixed BP % | Adaptive−static %p [CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| gaussian_matched | 24 | 56.3272 | 55.8603 | 57.2184 | 56.4468 | 57.7315 | +1.3580 [-0.2432, +3.3180] |
| gaussian_heldout | 24 | 66.9637 | 66.7863 | 67.1605 | 66.4005 | 67.2106 | +0.3742 [-1.6744, +2.4732] |
| heavy_matched | 24 | 61.0995 | 61.6397 | 61.0080 | 59.1270 | 60.5721 | -0.6316 [-2.4460, +1.4760] |
| heavy_heldout | 24 | 67.0795 | 66.9500 | 65.7661 | 63.5450 | 65.0149 | -1.1839 [-3.4916, +1.1899] |
| variance_matched | 24 | 57.8819 | 57.7353 | 58.2986 | 55.9375 | 58.6690 | +0.5633 [-1.1190, +2.2608] |
| variance_heldout | 24 | 61.7609 | 61.9615 | 61.9417 | 61.9444 | 62.8489 | -0.0198 [-1.6243, +1.7313] |
| mixed_matched | 24 | 67.4691 | 67.2299 | 67.4074 | 68.4028 | 67.5231 | +0.1775 [-1.6242, +2.1067] |
| mixed_heldout | 24 | 62.7083 | 61.9522 | 63.0015 | 61.7361 | 61.2500 | +1.0494 [-0.9144, +2.9360] |

Gaussian guard는 matched+heldout48개 Gaussian graphs를 합친 사전 분석이다.

- Adaptive−original: +0.5440 [-0.8430, +2.2762]%p.
- Adaptive−static: +0.8661 [-0.4341, +2.3380]%p.

모든 law 이름과 개별 scalar 값은 학습에서 보았고 파라미터 tuple만 off-diagonal로 바뀌었다. 새 feature law 자체의 zero-shot 전이를 시험한 것이 아니다. Final은 네 law regime을 층화한 분포라 iid training law 빈도와도 다르다. 이 subgroup 수치로 학습 법칙·baseline·checkpoint를 골라내지 않았다.

## 비용

| 방법 | 단일 graph ms | Batch16 ms | Batch16 추가 작업 MiB |
|---|---:|---:|---:|
| 기존 부호 제약 Factor | 6.496 | 5.027 | 3.70 |
| Flexible-static | 4.892 | 5.204 | 3.70 |
| Flexible-adaptive | 6.759 | 5.147 | 3.70 |
| 고정 law 주변화 / exact | 8.861 | 8.525 | 88.23 |
| 고정 law 주변화 / BP | 10.077 | 41.172 | 335.50 |
| 실제 학습 Mask PFN | 58.741 | 916.034 | 8.64 |
| 실제 학습 PCFI-PFN | 62.766 | 963.051 | 8.64 |

같은RTX3090에서 우리 학습 worker 종료 후5 warmup+30회 측정. 단일 graph는 사전 선택한4개 law regime median의 평균이다. 모델 생성·bank I/O는 제외하고 tensor/statistics/graph/BP 또는 enumeration, PCFI 실제 대치를 포함했다. 최종 표는 모든 모델의 CPU probability 반환까지 포함해 인터페이스를 맞췄다. 최초 structured GPU-output 측정은 cost_gpu_structured.json에 보존하고 최종 판정에는 쓰지 않았다. 외부 GPU 부하는 통제하지 않았다. Public batch는16회 순차 forward, structured batch는 실제 병렬이므로 순수 kernel 속도 대조는 아니다. 메모리는 총 GPU가 아닌 추가 작업량이다.

## 구현·학습·독립 평가 확인

- 새 학습 모델15개, successful updates3840회, task exposures30720개. Worker 학습시간 합2055.0초는 병렬 wall-clock이 아니다. 별도 dev smoke3 updates는 폐기했다.
- Context total1/2/4, observed nodes3/6. Context와 physical feature 관측은 완전히 분리했고 label을 보지 않고 query 하나를 예약해 모든 episode에 자기 피처 없는 query가 존재한다. 학습 CE는 전체 query를 사용한다.
- 동일2048개 새 train tasks, 동일 seed별 task order와256 successful updates. CE만 사용하고 teacher/posterior loss를 추가하지 않았다. 마지막 checkpoint만 사용한다.
- 세 Factor와 fixed-law 대조는 한 역사적 CE seed11 graph encoder/factor checkpoint를 고정한다. 기존 feature head를 추가 학습한 것이 아니다. 과거 checkpoint 사전학습 비용은 새 학습 예산에 포함하지 않는다.
- Trainable parameters: {'original': 697, 'static': 1547, 'adaptive': 1575, 'mask_pfn': 4012259, 'pcfi_pfn': 4012259}. Static/adaptive 약1.8% 차이. Original의697개와는 정확히 같지 않다. Public PFN은 별도 원래 adapter architecture이므로 같은 graph checkpoint 비교라고 주장하지 않는다.
- 기존 함수는 rho=x*positiveNN; static/adaptive는 부호·홀함수 제약 없이 value basis+MLP를 사용한다. 시작 score는 모두1.55*x로 같고 출력 clip도 동일하다. 표현력 허용과 실제 그 표현을 학습했다는 주장은 구분한다.
- Fixed-law 대조는 actual law/tuple/orientation을 입력받지 않고 law3×tuple4×orientation2의 사전 uniform prior를 주변화한다. Feature likelihood 상수도 유지한다. Fixed exact와 학생의 차이는 feature 함수뿐 아니라 정확한 label 합산과 유한 BP 차이도 포함한다. Fixed BP 대조가 이 차이를 보완한다.
- Dev 검사에서 frozen graph parity, 작은 tree의 독립 완전 열거와 fixed BP, class/relation/column permutation, hidden value poison, finite gradients를 확인한 뒤 설계를 잠갔다. Final192 graphs는 이후 생성했다.
- 시작 hash 함수가 keyword-only 인자를 요구하는 API 오류로 optimizer 전에 중단됐다. 인자 표기만 수정하고 동일 bank·초기화를 유지했다. 첫 lock과 오류 로그를 runtime에 보존하고 runtime_amendment.json에 기록했다. 최종 bank 생성 후 수행한 기계적 수정이므로 모든 실행 source가 한 번도 바뀌지 않았다고 주장하지 않는다. 모델/protocol/평가 규칙과 예측 결과에 근거한 선택은 없었다.
- 모든 source/config/bank/reference/checkpoint hashes, graph freeze와 task order를 검사했다. 저장예측45개를 checkpoint에서 재실행해 일치했다. 전체 결과 배열과 재채점은 results에 남겼다.

## 해석 범위

이 실험은12-node binary graph/2 feature/2 relation의 별도 structured predictor다. Graph factors까지 새 law에 맞춰 학습한 결과나 공개 GraphPFN 개선 모듈,99.5% 결측, 가변 schema, 큰 실제 graph 전이가 아니다. 원인을 분리하지 않았으면 최적화·학습량·정보 부족 중 하나로 확정하지 않는다. Primary accuracy 성공 없이 reference KL만 낮아진 것은 분류 메소드 성공으로 승격하지 않는다.

설계와 수식은 [METHOD.md](METHOD.md), 고정 명세는 [protocol.json](protocol.json), 원 수치는 [statistics.json](results/statistics.json), 비용은 [cost.json](results/cost.json), 재현 검사는 [audit.json](results/audit.json)에 있다.

현재 판정은 adaptive cell의 추가 가치 미확인이다. Static도 original 대비 primary accuracy 우위를 확인하지 못했다. Gaussian guard는 통과했으므로 Gaussian 붕괴 때문에 기각한 결과는 아니다. Adaptive NLL은 original보다0.007794 높았고 CI는[+0.001050,+0.017646]이었다. 정확도 개선 없이 확률 품질만 좋아진 성공 사례도 아니다.

한편 adaptive Factor는 실제 학습한 Mask/PCFI보다 accuracy가 약11%p 높았다. 이 보조 개선은 보존하되 architecture 차이와 이미 있던 Factor 연산을 새 adaptive 함수의 공으로 돌리지 않는다. Fixed exact의 정확도 point estimate가 adaptive보다 낮았으므로 고정 추론이 모든 지표에서 이겼다고 쓰지 않는다. 그 accuracy 차이 CI는0을 포함하고 NLL은 fixed exact가 더 낮았다.

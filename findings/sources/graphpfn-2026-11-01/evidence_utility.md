# Frozen Evidence Utility Test — 확률 품질과 결정 오류 분리

**현재 Evidence Head의 No Go는 유지한다.** 기존 확률만 분석했으며 새 모델 forward·학습·loss 변경·temperature fit·확률 조합은 모두 0회다. 같은 개발용 합성 그래프에 대한 진단이며 독립 검증이 아니다.

**핵심 해석:** Plain의 과소확신이 크고, evidence 모델은 주로 기존 정답의 확률을 강화했다. Separate는 확률을 평탄하게 만든 것이 아니라 더 뾰족하게 만들었다. 동시에 Plain 정답을 잃는 사건이 구제보다 많아 accuracy는 낮아졌다. 이 현상은 세 seed에서 반복됐으며 최종 출력만으로 정보 중복과 representation 간섭 중 하나를 원인으로 확정할 수는 없다.

## 범위와 가중 방식

주 평가 base/node99.5%만 사용했다. Graph 24개 × scenario 3 × k=1/3/5/10/20 × 학습 seed 3 = 1,080개 matched triple, 기존 확률 파일 3,240개다. 각 task 안에서 query를 균등 평균하고 k·학습 seed·graph도 균등 평균한다. 조건부 비율은 이 가중 방식의 사건 mass를 분모로 쓴다. 따라서 raw query를 모두 합친 비율과 다를 수 있다.

## 1. Confidence, entropy, calibration

ECE15는 top-label confidence의 고정 equal-width 15개 bin에서 계산했다. 개별 task ECE를 평균하지 않고 task 가중치를 적용한 bin 통계를 합친 뒤 계산한다. Confidence−accuracy는 평균적인 방향만 보여주며 calibration 전체를 표현하지 않는다. Entropy는 log(C)로 나눴다.

| Scenario | Model | Accuracy (%) | Mean confidence (%) | Conf−Acc (%p) | ECE15 (%) | NLL | Brier | Normalized entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| consistent | plain | 96.53 | 65.74 | -30.79 | 30.79 | 0.4781 | 0.2508 | 0.7258 |
| consistent | mean | 95.76 | 91.56 | -4.21 | 4.21 | 0.1458 | 0.0765 | 0.2204 |
| consistent | separate | 95.82 | 91.81 | -4.01 | 4.01 | 0.1443 | 0.0764 | 0.2115 |
| noisy | plain | 88.49 | 64.47 | -24.02 | 24.02 | 0.5216 | 0.2847 | 0.7306 |
| noisy | mean | 84.15 | 83.18 | -0.97 | 1.80 | 0.4008 | 0.2293 | 0.3994 |
| noisy | separate | 85.08 | 83.65 | -1.43 | 1.74 | 0.3794 | 0.2153 | 0.3914 |
| conflicting | plain | 88.96 | 63.99 | -24.98 | 24.98 | 0.5320 | 0.2905 | 0.7340 |
| conflicting | mean | 85.34 | 83.85 | -1.49 | 1.97 | 0.3751 | 0.2151 | 0.3852 |
| conflicting | separate | 86.09 | 84.31 | -1.77 | 1.82 | 0.3585 | 0.2044 | 0.3764 |

**Conflict의 개선은 전체 확률을 평탄하게 만든 현상과 반대 방향이다.** Separate는 평균 confidence가 높고 entropy가 낮다. NLL/Brier가 좋아졌다는 사실을 과신 완화로 바로 설명할 수 없다.

| Conflict | 오답일 때 confidence (%) | 정답일 때 confidence (%) | 전체 query 중 confidence≥.75 오답 (%) | confidence≥.90 오답 (%) |
|---|---:|---:|---:|---:|
| plain | 43.11 | 66.58 | 0.11 | 0.02 |
| mean | 64.33 | 87.20 | 4.48 | 0.99 |
| separate | 63.90 | 87.61 | 4.09 | 0.94 |

### Bin 수와 class 수 민감도

| Scenario | Model | ECE10 (%) | ECE15 (%) | Equal-mass ECE15 (%) |
|---|---|---:|---:|---:|
| consistent | plain | 30.79 | 30.79 | 30.79 |
| consistent | mean | 4.21 | 4.21 | 4.21 |
| consistent | separate | 4.01 | 4.01 | 4.01 |
| noisy | plain | 24.02 | 24.02 | 24.02 |
| noisy | mean | 1.54 | 1.80 | 1.77 |
| noisy | separate | 1.61 | 1.74 | 1.75 |
| conflicting | plain | 24.98 | 24.98 | 24.98 |
| conflicting | mean | 2.03 | 1.97 | 1.96 |
| conflicting | separate | 1.85 | 1.82 | 2.09 |

Class 수가 섞여 생기는 평균 효과를 확인하기 위해 Conflict를 C별로도 나눴다. ECE는 bin·표본 수에 민감하고, top-label calibration이 좋아도 전체 확률의 calibration이나 resolution 개선을 각각 입증한 것은 아니다.

| C | Model | Accuracy (%) | Mean confidence (%) | ECE15 (%) |
|---:|---|---:|---:|---:|
| 2 | plain | 88.46 | 72.92 | 15.54 |
| 2 | mean | 86.54 | 86.69 | 2.28 |
| 2 | separate | 86.81 | 87.57 | 1.79 |
| 3 | plain | 88.90 | 61.06 | 27.84 |
| 3 | mean | 84.58 | 82.43 | 2.73 |
| 3 | separate | 85.50 | 82.87 | 3.02 |
| 4 | plain | 89.82 | 58.51 | 31.32 |
| 4 | mean | 85.50 | 83.28 | 2.65 |
| 4 | separate | 86.47 | 83.22 | 3.26 |

## 2. Disagreement partition

Rescued/Harmed는 전체 query mass다. Disagree 내 승률은 어느 한 모델만 정답인 mass를 전체 disagreement mass로 나눈 값이다. 둘 다 틀리면서 다른 클래스를 고른 query도 분모에 포함한다.

| Scenario | Control | Disagree (%) | Control만 정답 (%) | Separate만 정답 (%) | Disagree 내 Control 정답 (%) | Separate 정답 (%) | Sep−Control acc (%p) |
|---|---|---:|---:|---:|---:|---:|---:|
| consistent | plain | 3.56 | 2.06 | 1.35 | 57.98 | 38.03 | -0.71 |
| consistent | mean | 0.47 | 0.20 | 0.26 | 43.43 | 55.24 | +0.06 |
| noisy | plain | 9.32 | 5.95 | 2.55 | 63.87 | 27.30 | -3.41 |
| noisy | mean | 1.82 | 0.37 | 1.30 | 20.26 | 71.24 | +0.93 |
| conflicting | plain | 8.40 | 5.36 | 2.48 | 63.83 | 29.57 | -2.88 |
| conflicting | mean | 1.63 | 0.39 | 1.13 | 23.83 | 69.31 | +0.74 |

## 3. NLL/Brier 개선은 어느 query에서 오는가?

아래 loss 기여는 각 집단의 전체 query 대비 mass를 포함한다. 다섯 row의 ΔNLL/ΔBrier 합은 전체 Separate−Plain과 정확히 같다. 음수는 Separate 개선이다.

| Conflict partition | Mass (%) | ΔNLL 기여 | ΔBrier 기여 | Plain p(true) | Separate p(true) |
|---|---:|---:|---:|---:|---:|
| 둘 다 정답 | 83.60 | -0.24487 | -0.13482 | 0.6798 | 0.8819 |
| Control만 정답 | 5.36 | +0.03080 | +0.02487 | 0.4468 | 0.2803 |
| Separate만 정답 | 2.48 | -0.01419 | -0.01038 | 0.3837 | 0.6827 |
| 둘 다 오답·같은 결정 | 8.00 | +0.05133 | +0.03232 | 0.3498 | 0.2368 |
| 둘 다 오답·다른 결정 | 0.55 | +0.00351 | +0.00190 | 0.2842 | 0.1764 |

전체 ΔNLL = -0.17342, ΔBrier = -0.08611. 같은 결정을 유지한 query의 확률 변화와 class decision이 바뀐 query의 손익을 구분해야 한다.

Plain→Separate NLL 개선 중 산술적으로 90.5%는 이미 Plain→Mean 차이에 있고, Brier는 87.6%다. 큰 확률 점수 개선 대부분을 관계 출처 보존만의 효과로 읽을 수 없다. 이는 숫자의 차이 분해이며 공통 head의 인과적 효과 비율은 아니다.

## 4. 정답 확률을 높였는데도 argmax를 틀리는가?

Control만 정답인 집단에서 p_true가 증가한 경우를 센다. Multiclass에서는 다른 오답 클래스의 확률이 더 크게 증가하면 가능하지만, binary에서는 p_true가 증가하면서 correct→wrong이 되는 것은 불가능하다. 저장 예측에서도 binary 사건이 0인지 확인했다.

| Scenario / Plain→Separate | Plain만 정답 mass (%) | 그중 p_true 증가 (%) | Plain p_true | Separate p_true | Plain strongest wrong | Separate strongest wrong |
|---|---:|---:|---:|---:|---:|---:|
| consistent | 2.06 | 17.52 | 0.4169 | 0.2949 | 0.3503 | 0.6028 |
| noisy | 5.95 | 9.95 | 0.4477 | 0.2760 | 0.3321 | 0.6093 |
| conflicting | 5.36 | 12.48 | 0.4468 | 0.2803 | 0.3367 | 0.6151 |

| Conflict C | Harmed mass (%) | 그중 p_true 증가 (%) |
|---:|---:|---:|
| 2 | 4.75 | 0.00 |
| 3 | 5.68 | 13.60 |
| 4 | 5.45 | 24.93 |

Conflict에서 Plain 정답→Separate 오답이면서 p_true가 오른 사건은 전체 query의 0.67%다. Separate가 오답이면서 Plain보다 p_true가 높은 사건 전체는 2.48%이며, Separate 오답 집단의 17.82%다. 현상은 존재하지만 accuracy 하락의 주된 패턴은 아니다.

## 5. 학습 seed별 반복성

| Scenario | Seed | Δaccuracy (%p) | ΔNLL | ΔBrier | Δconfidence (%p) |
|---|---:|---:|---:|---:|---:|
| consistent | 17 | -1.03 | -0.3367 | -0.1764 | +26.26 |
| consistent | 29 | -1.21 | -0.3304 | -0.1711 | +26.06 |
| consistent | 43 | +0.11 | -0.3344 | -0.1757 | +25.87 |
| noisy | 17 | -3.79 | -0.1417 | -0.0687 | +19.36 |
| noisy | 29 | -3.37 | -0.1499 | -0.0735 | +19.46 |
| noisy | 43 | -3.06 | -0.1349 | -0.0659 | +18.70 |
| conflicting | 17 | -3.24 | -0.1732 | -0.0856 | +20.57 |
| conflicting | 29 | -3.04 | -0.1781 | -0.0885 | +20.53 |
| conflicting | 43 | -2.35 | -0.1689 | -0.0842 | +19.88 |

세 seed에서 같은 방향이면 단순히 한 번의 불안정한 학습 결과라는 설명은 약해진다. 그러나 세 초기화·한 task bank만으로 학습 안정성 전체를 입증하거나, 최종 확률만으로 정보 중복(A)과 representation 간섭(B)을 구분할 수는 없다.

## 6. Conflict 강도와의 관계

원래 생성기·seed로 복원한 실제 partition 변경 비율 범위는 0.306–0.601다. 이 비율은 그래프별 숨은 생성 정보이며 분석에만 사용했다. 새 forward나 강도 intervention은 없다.

동일 그래프에서 강도만 바꾼 저장 예측이 없으므로 아래는 서로 다른 24개 그래프 사이의 Spearman 연관이다. Classes·edge 강도·크기·context 구성도 함께 다르다. 실제 변경 비율은 구조적 충돌의 전 범위에서 단조 척도가 아니며, binary partition이 전부 뒤집히면 군집 구조가 다시 같아질 수도 있다. 이를 보완하는 기술 통계로 1−NMI와 context-supported evidence disagreement도 함께 보고한다.

| Strength | 비교 | Metric | Spearman ρ | 탐색적 95% 구간 |
|---|---|---|---:|---|
| corruption_fraction | Separate−Plain | Accuracy | -0.110 | [-0.499, 0.323] |
| corruption_fraction | Separate−Plain | NLL | 0.124 | [-0.291, 0.500] |
| corruption_fraction | Separate−Mean | Accuracy | 0.068 | [-0.334, 0.424] |
| corruption_fraction | Separate−Mean | NLL | -0.373 | [-0.635, 0.016] |
| partition_information_loss | Separate−Plain | Accuracy | 0.292 | [-0.162, 0.665] |
| partition_information_loss | Separate−Plain | NLL | 0.770 | [0.529, 0.905] |
| partition_information_loss | Separate−Mean | Accuracy | -0.417 | [-0.749, 0.035] |
| partition_information_loss | Separate−Mean | NLL | 0.117 | [-0.291, 0.498] |
| supported_evidence_disagreement | Separate−Plain | Accuracy | -0.250 | [-0.580, 0.168] |
| supported_evidence_disagreement | Separate−Plain | NLL | 0.065 | [-0.373, 0.475] |
| supported_evidence_disagreement | Separate−Mean | Accuracy | 0.197 | [-0.224, 0.529] |
| supported_evidence_disagreement | Separate−Mean | NLL | -0.377 | [-0.703, 0.077] |

### 실제 변경 비율의 중앙값 분할

분할은 strength만 보고 정했으며 성능에 맞춰 경계를 선택하지 않았다. 낮음/높음은 다른 graph 집단이므로 인과 비교가 아니다.

| 그룹 | Graph 수 | 평균 변경 비율 | Sep−Mean acc (%p) | Sep−Plain acc (%p) |
|---|---:|---:|---:|---:|
| low | 12 | 0.394 | +0.68 | -2.45 |
| high | 12 | 0.506 | +0.80 | -3.31 |

Separate−mean 증분의 high−low 차이는 +0.12%p [-0.45, +0.63]다.

Separate−plain 증분의 high−low 차이는 -0.86%p [-2.36, +0.76]다.

k별 변경 비율–accuracy 증분의 연관도 함께 확인했다.

| k/class | 비교 | ρ | 95% 구간 |
|---:|---|---:|---|
| 1 | Sep−mean | -0.258 | [-0.631, 0.201] |
| 1 | Sep−plain | -0.054 | [-0.520, 0.400] |
| 3 | Sep−mean | 0.084 | [-0.299, 0.444] |
| 3 | Sep−plain | 0.086 | [-0.302, 0.473] |
| 5 | Sep−mean | 0.242 | [-0.139, 0.577] |
| 5 | Sep−plain | -0.150 | [-0.529, 0.273] |
| 10 | Sep−mean | 0.056 | [-0.330, 0.424] |
| 10 | Sep−plain | -0.258 | [-0.645, 0.188] |
| 20 | Sep−mean | 0.016 | [-0.420, 0.461] |
| 20 | Sep−plain | -0.077 | [-0.484, 0.297] |

Class 수별 변경 비율–accuracy 연관:

| C | 비교 | Graph 수 | ρ | 95% 구간 |
|---:|---|---:|---:|---|
| 2 | Sep−Plain | 7 | -0.143 | [-0.961, 0.686] |
| 2 | Sep−Mean | 7 | -0.286 | [-1.000, 0.750] |
| 3 | Sep−Plain | 12 | 0.350 | [-0.307, 0.871] |
| 3 | Sep−Mean | 12 | -0.287 | [-0.829, 0.394] |
| 4 | Sep−Plain | 5 | -0.600 | [-1.000, 0.875] |
| 4 | Sep−Mean | 5 | 0.600 | [-1.000, 1.000] |

보조 지표 1−NMI와 ΔNLL의 전체 상관은 class 수와도 얽혀 있다. 아래 class별 기술 통계를 함께 봐야 한다. 양의 ΔNLL 상관은 강도가 높을수록 Separate의 NLL 이득이 줄어드는 방향이다.

| C | Graph 수 | 평균 1−NMI | 평균 Sep−Plain NLL | C 내부 ρ |
|---:|---:|---:|---:|---:|
| 2 | 7 | 0.983 | -0.0739 | 0.393 |
| 3 | 12 | 0.899 | -0.1865 | 0.350 |
| 4 | 5 | 0.812 | -0.2815 | 0.600 |

## 판정의 한계

NLL/Brier와 ECE 개선은 관계 evidence가 새로운 정보를 추가했다는 증명이 아니다. 공통 backbone이 이미 같은 graph/context를 사용하고, evidence branch뿐 아니라 backbone도 arm별로 함께 학습됐다. 최종 확률만으로 head의 직접 효과와 backbone의 학습 변화를 식별할 수 없다.

이 분석으로 새 selector·확률 조합·margin loss 채택을 정당화하지 않는다. 현재 head의 No Go는 유지한다. Relation-aware prior 가설은 남지만 기존 결정 능력을 유지하는 구체적 방법은 미검증이다.

Graph bootstrap은 k와 학습 seed를 같은 master 안에 유지한 2,000회 탐색적 구간이다. 학습 seed 모집단의 불확실성이나 다중 비교는 보정하지 않았다. Equal-mass ECE는 기술 통계다.

[원래 Kill Test](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/REPORT.md) · [진단 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/protocol.json) · [검증](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/results/audit.json)

## Figure

![Frozen evidence utility](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/results/frozen_evidence_utility.png)

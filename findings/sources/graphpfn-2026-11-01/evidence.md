# Relation Evidence PFN — synthetic conflict kill test

**판정: 현재 설정에서는 No Go. 대규모 학습으로 확장하지 않는다.**

공식 pretrained GraphPFN이 아닌 소형 일방향 quadratic prototype을 새로 학습했다. LP-conditioned residual은 재개하지 않았으며, 출력 log LP offset도 없다. 세 seed(17/29/43), arm별 384 updates × batch 4, 동일한 새 합성 과제 1,536개와 optimizer를 사용했다. 최종 checkpoint만 평가했으며 test 기반 tuning이나 선택은 없다.

## 메소드와 공정성

Plain은 기존 staged 모델(10,469 parameters)이다. Mean/Separate는 같은 staged backbone에 801-parameter shared class/relation head(7→32→16→1)를 붙인 11,270-parameter 모델이다. Mean/Separate의 전체 초기 가중치와 파라미터 수는 같고, Plain과도 초기 backbone이 정확히 같다. Evidence head는 출력 0으로 시작하여 세 arm의 초기 logits도 같다. Plain 대비 약 7.7%의 용량 차이가 있어, Plain 대비 차이를 representation 하나의 효과로만 해석할 수 없다.

K_r = B_r D_helper,r^(-1) B_r^T, E_i,r,c = sum_{j in C} K_r(i,j) 1[y_j=c]를 사용한다. K의 대각을 먼저 0으로 만들어 모든 2-hop self-return을 정확히 제거한다. Query seed는 애초에 없고, context utility 계산의 각 노드도 자기 seed 없이 평가한다. 확률만이 아니라 log1p(E), log1p(support mass), context LOO soft quality, context support 총량·coverage, class frequency를 함께 전달한다.

고정 전처리로 만든 7차원 evidence descriptor를 t_i,r,c라 하면 Mean은 phi(mean_r t_i,r,c), Separate는 mean_r phi(t_i,r,c)를 기본 logits에 더한다. Mean도 R개 동일 slot을 처리하여 shared head 계산량을 맞춘다. 이는 raw E 하나만 평균하는 대조보다 풍부한 descriptor 평균 대조다. Relation ID나 class ID embedding은 없다. Context utility라는 명시적 통계가 있어, 이 실험만으로 암묵적 PFN 추론을 발견했다고 주장하지 않는다.

세 arm은 전체 graph와 relation별 통계를 처리하는 공통 backbone을 유지한다. 따라서 Mean 모델 전체가 관계 정보를 잃은 입력만 받는 비교는 아니다. 이 실험은 evidence head에서 인코딩 전/후 어느 시점에 관계를 합치는지의 증분을 측정한다. 성능 차이가 있어도 관계 분리 입력의 통계적 정보 우위를 일반적으로 증명하지 않는다.

Context에 대한 CE는 사용하지 않는다. Utility는 context의 label을 정답으로 쓰는 LOO 통계이며 actual query 정답은 사용하지 않는다. Context 노드의 완전한 pseudo-query 예측이 필요하다면 그 label을 전역에서 숨기고 utility까지 다시 계산해야 한다. 이번에는 그런 예측으로 학습·평가하지 않았다.

## Paired scenario

같은 master의 label·feature·context 순서·관측 uniform·helper 크기·relation별 edge 난수를 공유한다. Consistent는 두 관계가 label partition을 따르고, Noisy는 한 관계가 독립 무작위 partition을 따른다. Conflicting은 한 관계의 노드 partition 35–55%를 다른 클래스로 바꾼다. Context와 query에 같은 규칙을 적용해 context로 utility를 추정할 여지를 남겼다. 고정된 클래스 이름 치환은 B D^(-1) B^T를 바꾸지 않아 conflict 생성 방식으로 쓰지 않았다.

유용한 역할의 relation ID는 task마다 무작위다. Role reversal은 ID·relation별 난수를 고정한 채 어느 partition이 들어가는지 교환해 topology를 다시 만든다. 순수 ID permutation과 구분한다. Context shuffle은 context 위치·클래스 수를 유지하고 그 안의 label을 무작위 순열한 뒤 evidence와 모든 utility를 다시 계산한다. 작은 context에서는 우연히 label이 그대로 남을 수 있으며 실제 변경률을 기록했다.

## Primary: 99.5% whole-node missing

새 master graph 24개 × scenario 3 × k=1/3/5/10/20/class × base/role reversal/context shuffle = 1,080 episodes. Base 360개를 주 비교로 쓴다. 추가 full/node90%는 k5/k20의 288 episodes이며 별도 표다. 각 episode에서 학습된 9개 모델과 고정 기준선 2개를 평가해 총 15,048개 확률 파일을 저장했다.

Uniform은 관계별 2-hop class histogram 평균, Context-weighted는 context LOO soft quality에 고정 softmax(5q)를 적용한 histogram 평균이다. 둘 다 새 학습이 없으며 반복 diffusion을 하는 공식 LP 재현은 아니다.

| Scenario / base | Uniform | Context-weighted | Plain | Mean | Separate |
|---|---:|---:|---:|---:|---:|
| consistent | 93.53 | 93.24 | 96.53 | 95.76 | 95.82 |
| noisy | 78.78 | 84.88 | 88.49 | 84.15 | 85.08 |
| conflicting | 79.90 | 85.05 | 88.96 | 85.34 | 86.09 |

| Scenario / base | Separate−Plain (%p) | Separate−Mean (%p) | Separate−Context weighted (%p) |
|---|---|---|---|
| consistent | -0.71 [-1.30, -0.15] | +0.06 [-0.06, +0.18] | +2.58 [+2.13, +3.08] |
| noisy | -3.41 [-4.68, -1.99] | +0.93 [+0.61, +1.25] | +0.21 [-0.37, +0.78] |
| conflicting | -2.88 [-3.66, -2.09] | +0.74 [+0.49, +1.03] | +1.03 [+0.47, +1.58] |

## k별 accuracy와 증분

| Scenario | k/class | Plain | Mean | Separate | Sep−Mean (%p) | Sep−Plain (%p) |
|---|---:|---:|---:|---:|---:|---:|
| consistent | 1 | 90.22 | 87.12 | 87.72 | +0.60 | -2.50 |
| consistent | 3 | 94.58 | 95.30 | 95.02 | -0.28 | +0.45 |
| consistent | 5 | 98.51 | 97.46 | 97.44 | -0.02 | -1.06 |
| consistent | 10 | 99.64 | 99.19 | 99.18 | -0.01 | -0.46 |
| consistent | 20 | 99.71 | 99.74 | 99.73 | -0.01 | +0.03 |
| noisy | 1 | 68.15 | 65.29 | 65.26 | -0.04 | -2.89 |
| noisy | 3 | 88.32 | 79.67 | 81.74 | +2.07 | -6.58 |
| noisy | 5 | 94.06 | 87.27 | 88.98 | +1.71 | -5.08 |
| noisy | 10 | 95.73 | 93.12 | 93.84 | +0.73 | -1.88 |
| noisy | 20 | 96.20 | 95.41 | 95.58 | +0.18 | -0.62 |
| conflicting | 1 | 69.16 | 67.68 | 67.56 | -0.11 | -1.60 |
| conflicting | 3 | 88.70 | 81.53 | 82.92 | +1.39 | -5.78 |
| conflicting | 5 | 94.73 | 87.62 | 89.24 | +1.62 | -5.49 |
| conflicting | 10 | 96.01 | 93.83 | 94.58 | +0.74 | -1.44 |
| conflicting | 20 | 96.21 | 96.05 | 96.12 | +0.07 | -0.09 |

## 학습 seed별 증분

| Seed | Scenario | Sep−Mean (%p) | Sep−Plain (%p) |
|---:|---|---:|---:|
| 17 | consistent | -0.02 | -1.03 |
| 17 | noisy | +0.56 | -3.79 |
| 17 | conflicting | +0.51 | -3.24 |
| 29 | consistent | +0.03 | -1.21 |
| 29 | noisy | +1.26 | -3.37 |
| 29 | conflicting | +0.87 | -3.04 |
| 43 | consistent | +0.15 | +0.11 |
| 43 | noisy | +0.96 | -3.06 |
| 43 | conflicting | +0.85 | -2.35 |

## NLL / Brier

| Scenario | Model | NLL | Brier |
|---|---|---:|---:|
| consistent | uniform | 0.3652 | 0.1406 |
| consistent | context_weighted | 0.3649 | 0.1430 |
| consistent | plain | 0.4781 | 0.2508 |
| consistent | mean | 0.1458 | 0.0765 |
| consistent | separate | 0.1443 | 0.0764 |
| noisy | uniform | 1.1321 | 0.3684 |
| noisy | context_weighted | 0.9370 | 0.2426 |
| noisy | plain | 0.5216 | 0.2847 |
| noisy | mean | 0.4008 | 0.2293 |
| noisy | separate | 0.3794 | 0.2153 |
| conflicting | uniform | 0.9907 | 0.3407 |
| conflicting | context_weighted | 0.8426 | 0.2432 |
| conflicting | plain | 0.5320 | 0.2905 |
| conflicting | mean | 0.3751 | 0.2151 |
| conflicting | separate | 0.3585 | 0.2044 |

## Negative controls

| Scenario | Control | Plain | Mean | Separate |
|---|---|---:|---:|---:|
| consistent | role_reversal | 96.53 | 95.76 | 95.82 |
| consistent | context_shuffle | 33.46 | 33.62 | 33.58 |
| noisy | role_reversal | 90.40 | 86.11 | 87.05 |
| noisy | context_shuffle | 34.73 | 34.87 | 35.04 |
| conflicting | role_reversal | 89.73 | 86.34 | 87.22 |
| conflicting | context_shuffle | 35.07 | 34.79 | 34.92 |

Context shuffle의 실제 label 변경률은 episode 평균 67.56%, 전혀 바뀌지 않은 episode는 15/360다.

순수 relation-ID permutation의 학습 후 확률 최대 차이는 2.38e-07, class 순열 등변성 차이는 1.19e-07였다. 이는 주어진 관계의 ID 재명명 검사이며 새로운 semantic schema 전이의 증명은 아니다.

## Conflict가 실제 evidence에 나타났는가?

아래는 query 양쪽 관계에 support가 있는 경우의 argmax 불일치율과 전체 query의 관계별 정확도다. 동일한 seed/label을 쓴 집계이므로 기전 진단으로만 쓴다. Tie와 support 부족은 k에 따라 달라질 수 있다.

| Scenario | k/class | 양쪽 support query (%) | 그 안의 불일치 (%) | 지정 good 관계 acc | 다른 관계 acc |
|---|---:|---:|---:|---:|---:|
| consistent | 1 | 40.10 | 25.76 | 67.49 | 68.00 |
| consistent | 3 | 84.68 | 20.46 | 83.69 | 85.27 |
| consistent | 5 | 94.63 | 15.62 | 90.27 | 90.40 |
| consistent | 10 | 99.49 | 11.53 | 94.04 | 93.56 |
| consistent | 20 | 100.00 | 8.50 | 95.86 | 95.54 |
| noisy | 1 | 35.05 | 63.88 | 67.49 | 35.97 |
| noisy | 3 | 81.54 | 64.03 | 83.69 | 36.37 |
| noisy | 5 | 93.62 | 63.14 | 90.27 | 36.87 |
| noisy | 10 | 99.36 | 63.88 | 94.04 | 36.35 |
| noisy | 20 | 100.00 | 64.53 | 95.86 | 35.02 |
| conflicting | 1 | 35.59 | 58.99 | 67.49 | 40.57 |
| conflicting | 3 | 80.48 | 57.45 | 83.69 | 42.96 |
| conflicting | 5 | 93.16 | 56.74 | 90.27 | 44.30 |
| conflicting | 10 | 99.40 | 54.88 | 94.04 | 45.84 |
| conflicting | 20 | 99.95 | 53.35 | 95.86 | 48.08 |

## Secondary: 덜 결측된 조건

| Condition | Scenario | Plain | Mean | Separate |
|---|---|---:|---:|---:|
| full | consistent | 99.39 | 99.40 | 99.41 |
| full | noisy | 95.20 | 95.01 | 95.59 |
| full | conflicting | 95.35 | 94.98 | 95.72 |
| node_0.9 | consistent | 98.78 | 98.82 | 98.81 |
| node_0.9 | noisy | 95.16 | 92.07 | 93.00 |
| node_0.9 | conflicting | 95.13 | 92.91 | 93.66 |

## 판정 범위

Go 규칙은 conflicting/base에서 Separate가 Mean과 Plain을 세 학습 seed 모두 이기고 두 graph-bootstrap 구간의 하한이 0보다 큰지 확인하는 소형 screening이다. 추가 capacity, 명시적 context quality 통계, 고정 기준선, 확률 품질과 다른 scenario까지 함께 판단해야 한다. Go가 되더라도 논문 novelty나 실데이터 전이 성공은 아니다.

구간은 학습 seed를 task별 평균하고 같은 master의 k 조건을 함께 묶어 24 master graph를 2,000회 resample했다. 학습 seed 모집단 전체의 불확실성이나 다중 비교를 보정하지 않는다. Train/eval은 seed와 graph 크기 범위가 다르지만 같은 생성기이며, eval relation 수는 2다. 새로운 relation 수·semantic schema·실제 heterograph 전이는 평가하지 않았다.

[사전 고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/protocol.json) · [구현 검사](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/implementation_checks.json) · [평가 검사](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/evaluation_audit.json)

## Figure

![Synthetic conflict kill test](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/conflict_kill_test.png)

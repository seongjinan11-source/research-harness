# LP conditioning: offset·내부 효과 및 relation/feature 분리 진단

새 학습·residual tuning 없이 기존 6개 소형 체크포인트를 고정했다. 공식 GraphPFN 및 실제 그래프 평가가 아니다. 기존 pilot A 재분석과 새 paired synthetic factorial B/C를 구분한다.

## A. 기존 pilot의 결측 조건별 효과

Δoffset = offset−plain, Δinternal = conditioned−offset. 수치는 accuracy의 %p다.

| 기존 setting | 결측 | Δoffset | Δinternal |
|---|---|---:|---:|
| matched | full | +0.29 | -0.15 |
| matched | node_0.9 | -0.58 | +0.10 |
| matched | node_0.995 | -0.83 | +0.05 |
| matched | cell_0.995 | -0.17 | -0.11 |
| heldout_shape | full | -0.33 | +0.26 |
| heldout_shape | node_0.9 | +1.07 | +0.17 |
| heldout_shape | node_0.995 | +0.41 | +0.42 |
| heldout_shape | cell_0.995 | +1.05 | +0.53 |

## B. 새 paired factorial

36개 master graph(각 family 12개) × relation 2/3 × feature narrow/wide × 결측 4조건 = 576 episodes. 각 episode에 LP와 6개 frozen 모델을 평가해 4,032개 확률 파일을 저장했다. 학습 업데이트는 0회다. 주 평가의 learned task forward 3,456회 외에 batch/singleton 일치 확인용 12회가 추가되어 실제 3,468회다. LP는 주 평가 576회와 추가 확인 2회이며, 공식 GraphPFN forward는 0회다.

Master는 원래 R3/wide heldout_shape 생성기를 재현한다. R2는 label/feature 관계를 모두 보존하고 나머지 nuisance 관계와 해당 helper 노드만 제거한다. Narrow는 동일 raw feature의 사전 고정 6–12열 prefix이며 wide는 16–24열이다. 공유 열의 값·column embedding·관측 mask, label/context/query, 보존한 관계의 edge는 동일하다. 정규화는 각 조건의 관측값만 사용한다. 99.5% 결측에서 관측 노드를 강제로 남기지 않는다.

따라서 relation 효과는 비정보성 관계와 helper 노드·edge 추가의 효과다. 생성기의 숨은 지정으로 제거 대상을 정한 평가 개입이며, 그 지정은 모델 입력에 전달하지 않았다. 실제 그래프에서 nuisance 관계를 알아내는 알고리즘을 평가한 것은 아니다. 숫자로서 relation 개수만의 효과나 새로운 semantic schema 전이를 뜻하지 않는다. Narrow 조건도 기존 matched 분포의 완전 재현은 아니다.

| R | Feature | LP | Plain | Offset | Conditioned | Δoffset | Δinternal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | narrow | 66.80 | 74.21 | 73.91 | 74.27 | -0.29 | +0.36 |
| 2 | wide | 66.80 | 73.64 | 73.71 | 74.14 | +0.06 | +0.43 |
| 3 | narrow | 66.56 | 73.88 | 73.95 | 74.42 | +0.07 | +0.47 |
| 3 | wide | 66.56 | 73.62 | 73.67 | 74.20 | +0.05 | +0.52 |

각 row는 네 결측 조건 평균이다. 아래는 조건별 증분이다.

| R | Feature | 결측 | Δoffset | Δinternal |
|---:|---|---|---:|---:|
| 2 | narrow | full | -0.35 | +0.56 |
| 2 | narrow | node_0.9 | -0.59 | +0.08 |
| 2 | narrow | node_0.995 | -0.26 | +0.56 |
| 2 | narrow | cell_0.995 | +0.02 | +0.23 |
| 2 | wide | full | +0.69 | +0.54 |
| 2 | wide | node_0.9 | -0.16 | +0.32 |
| 2 | wide | node_0.995 | -0.39 | +0.67 |
| 2 | wide | cell_0.995 | +0.11 | +0.21 |
| 3 | narrow | full | -0.41 | +0.71 |
| 3 | narrow | node_0.9 | +0.45 | +0.77 |
| 3 | narrow | node_0.995 | +0.22 | +0.39 |
| 3 | narrow | cell_0.995 | -0.00 | +0.02 |
| 3 | wide | full | -0.60 | +0.72 |
| 3 | wide | node_0.9 | +0.63 | +0.82 |
| 3 | wide | node_0.995 | +0.16 | +0.45 |
| 3 | wide | cell_0.995 | +0.01 | +0.10 |

### 요인 변화가 증분에 미치는 영향

R 효과는 R3−R2를 feature 두 조건에 평균, feature 효과는 wide−narrow를 R 두 조건에 평균했다. Interaction은 (R3−R2)wide−(R3−R2)narrow다. 표는 %p 및 탐색적 graph bootstrap 95% 구간이다.

| 결측 | 증분 | R 효과 | Feature 효과 | Interaction |
|---|---|---|---|---|
| all | offset | +0.17 [-0.27, +0.65] | +0.17 [-0.06, +0.40] | -0.37 [-0.68, -0.07] |
| all | internal | +0.10 [-0.21, +0.42] | +0.06 [-0.11, +0.23] | -0.02 [-0.26, +0.20] |
| full | offset | -0.67 [-1.02, -0.32] | +0.43 [-0.54, +1.41] | -1.23 [-2.21, -0.36] |
| full | internal | +0.17 [-0.17, +0.57] | -0.01 [-0.56, +0.56] | +0.04 [-0.58, +0.67] |
| node_0.9 | offset | +0.92 [+0.07, +2.04] | +0.30 [-0.13, +0.72] | -0.25 [-0.85, +0.28] |
| node_0.9 | internal | +0.60 [+0.06, +1.19] | +0.15 [-0.33, +0.67] | -0.18 [-0.70, +0.30] |
| node_0.995 | offset | +0.52 [-0.21, +1.30] | -0.09 [-0.19, -0.01] | +0.07 [-0.06, +0.22] |
| node_0.995 | internal | -0.19 [-0.61, +0.22] | +0.08 [+0.02, +0.16] | -0.05 [-0.18, +0.06] |
| cell_0.995 | offset | -0.06 [-0.67, +0.57] | +0.05 [-0.35, +0.48] | -0.07 [-0.88, +0.61] |
| cell_0.995 | internal | -0.16 [-0.69, +0.31] | +0.03 [-0.17, +0.26] | +0.10 [-0.31, +0.55] |

## C. LP quality에 따른 진단

**Test 정답으로 나누는 사후 기전 분석이다.** 추론 입력·선택기·모델 채택 규칙으로 사용하지 않았다. 고정 bin은 task-level LP accuracy <50%, 50–75%, ≥75%다. Task의 label mechanism과 난도가 섞이므로 bin 차이를 LP quality 자체의 인과 효과로 해석하지 않는다.

| LP task accuracy | 반복 task 수 | LP | Plain | Offset | Conditioned | Δinternal |
|---|---:|---:|---:|---:|---:|---|
| LP<50% | 152 | 33.04 | 47.91 | 48.08 | 47.91 | -0.17 [-0.47, +0.15] |
| 50%<=LP<75% | 144 | 61.27 | 71.00 | 69.34 | 69.95 | +0.60 [-0.14, +1.38] |
| LP>=75% | 280 | 87.72 | 89.38 | 90.08 | 90.78 | +0.70 [+0.10, +1.40] |

아래 query partition은 모든 factorial 조건과 두 학습 seed의 반복 decision을 합친 기술 통계다. 독립 query 수가 아니며 위 graph-평균 지표와 가중 방식이 다르다. Rescued/Harmed는 각 partition 내부 비율이다.

| LP query 판정 | 비교 대상 | 대상 acc | Conditioned acc | Rescued | Harmed |
|---|---|---:|---:|---:|---:|
| LP_correct | plain | 89.31 | 93.25 | 4.82 | 0.88 |
| LP_correct | lp_offset | 94.05 | 93.25 | 0.96 | 1.76 |
| LP_correct | LP | 100.00 | 93.25 | 0.00 | 6.75 |
| LP_wrong | plain | 43.61 | 36.94 | 1.95 | 8.62 |
| LP_wrong | lp_offset | 33.95 | 36.94 | 4.42 | 1.43 |
| LP_wrong | LP | 0.00 | 36.94 | 36.94 | 0.00 |

## 판정과 다음 연구의 경계

**LP-conditioned의 대규모 확장은 계속 보류한다.** 이번 새 master graph에서는 네 cell 모두 내부 conditioning의 평균 증분이 +0.36~+0.52%p였지만, conditioned−plain의 네 평균 bootstrap 구간은 모두 0을 포함했다. 단순 offset 대비 작은 이득과 전체 방법의 우위를 구분한다. 네 cell 모두 평균 NLL/Brier는 offset이 더 좋았다.

내부 효과의 relation 추가에 따른 평균 변화는 +0.10%p [−0.21,+0.42], feature 확장에 따른 변화는 +0.06%p [−0.11,+0.23]였다. Relation×feature interaction도 −0.02%p [−0.26,+0.20]로 0을 포함한다. 일부 결측별 탐색적 차이는 있지만 schema 변화에 특별히 강하다는 일관된 증거는 없다. Full에서도 내부 효과가 +0.54~+0.72%p 있으므로 심한 결측에만 국한된 현상도 아니다.

LP가 틀린 query에서 conditioned는 offset보다 2.99%p 높았지만 plain보다 6.67%p 낮았다. LP가 맞는 query에서는 plain보다 3.94%p 높고 offset보다 0.80%p 낮았다. 이는 두 방식 사이의 절충이라는 해석과 부합하며, LP의 정답을 보존하면서 plain의 교정 능력까지 얻었다는 증거는 아니다. Test 정답으로 나눈 집단이므로 인과적 효과나 실행 가능한 선택 규칙을 뜻하지 않는다.

이후 주력 가설은 관계별 context evidence의 출처를 보존하는 내부 연산으로 둔다. 아직 새 모델을 구현·학습한 것은 아니다. 기존 plain도 shared relation MLP로 전역 관계 가중치를 계산하므로 shared encoder 자체를 새 기여로 주장하지 않는다. 차이는 query별·class별 evidence를 관계 간 평균 이전에 모델에 제공하는 연산과 그 검증에서 보여야 한다.

단, 현재 합성 그래프의 native relation은 target–helper bipartite이고 context label은 target에만 있으므로 단일 native adjacency A_r Y_C는 target query에서 0이다. Target→helper→target kernel K_r = B_r D_helper^(-1) B_r^T 등의 typed path를 사용해야 한다. E_i,r,c = sum_{j in C} K_r(i,j) 1[y_j=c]와 support mass sum_{j in C}K_r(i,j)를 함께 보존하고, 필요하면 길이별 채널을 유지한다. 이 class histogram은 원래 그래프의 모든 정보를 보존하지는 않는다.

다음 최소 대조는 동일한 synthetic 과제·학습량에서 기존 relation-aware plain, 관계를 평균한 evidence 입력, 관계별 evidence 입력이다. 출력 LP offset 없이 비교하고 relation 순열 불변성, class 순열 등변성, context pseudo-query의 자기 label 기여 제거를 확인해야 한다. 관계 수뿐 아니라 유용/무관/상충 관계 조합을 바꿔 어느 조건에서 관계 출처 보존이 도움이 되는지 먼저 검증한다. 이 결과만으로 novelty나 새로운 semantic schema 전이를 주장하지 않는다.

## 검증과 한계

원래 학습 소스·6개 checkpoint hash, master와 기존 generator의 full 입력 일치, factorial pairing, query 정답 변경 시 입력 불변, batch/singleton 확률 일치, 평가 전후 weight 불변, 모든 저장 확률의 지표 재계산을 검사했다. 세부 결과는 results/audit.json이다.

구간은 두 학습 seed를 task별 평균하고 master graph 단위로 반복 조건을 묶어 family별 층화 2,000회 bootstrap했다. LP quality bin에서는 해당 bin에 기여한 graph를 resample하고 각 graph가 제공한 task 수를 보존해 task 평균과 bootstrap estimand를 맞췄다. 학습 seed 모집단의 불확실성, 다중 비교, 새로운 생성기·실데이터 전이는 다루지 않는다. 이 진단만으로 offset이 학습 regularizer로 작동했다는 기전도 입증되지 않는다. 학습과 추론 시 offset 효과를 분리하려면 별도 intervention이 필요하지만 이번에 추가하지 않았다.

[원래 pilot 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/REPORT.md) · [고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/diagnostics/protocol.json)

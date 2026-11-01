# Plain: exact-degree Relation Role Reversal

Evidence Head와 LP-conditioned residual을 재개하지 않았다. 기존 Plain checkpoint 세 개(seed17/29/43)만 고정하고 새 그래프에서 추론했다. 새 weight 학습·loss 변경·calibration은 없다. 공식 pretrained GraphPFN이 아닌 10,469-parameter staged synthetic prototype에 대한 진단이다.

## 먼저 알아야 할 구조적 사실

Plain은 relation별 8개 통계를 같은 MLP로 처리하며 relation-ID embedding이나 ID별 학습 파라미터가 없다. 그래서 특정 ID를 lookup table처럼 외운다는 설명은 이 구현에 맞지 않는다. 학습 prior도 유용한 역할의 relation ID를 task마다 무작위로 정했다. 검증 대상은 context-dependent utility 통계를 따라 global relation 가중치를 바꾸는지이며, 실패해도 곧바로 ID 암기라고 결론내리지 않는다.

## Degree와 역할을 어떻게 통제했나?

같은 master graph24개 × Consistent/Noisy/Conflicting × k1/3/5/10/20, 99.5% whole-node missing이다. Double-edge switch는 각 relation의 target/helper 개별 노드 degree를 정확히 유지한다. Node ID·type·feature 값·mask·context/query·label도 고정했다. 관계 역할을 유지하며 재샘플한 그래프와 반대로 바꿔 재샘플한 그래프에 같은 proposal 난수를 썼다. Consistent에서는 두 결과가 byte 단위로 같다.

조건부 edge potential은 원래 생성기의 혼합 확률을 사용하고, relation마다 edge 수의100배 proposal을 고정했다. MCMC의 평형 도달을 보장하지 않으며 degree 외 모든 고차 구조를 보존하는 것은 아니다. Hidden synthetic partition은 benchmark 구성·검증에만 쓰고 모델에는 전달하지 않았다. 성능을 보고 graph를 제외하거나 sampler를 조정하지 않았다.

48/48 Noisy/Conflict graph 쌍에서 전체 target kernel label homophily 기준으로 유용한 relation의 교환을 확인했다. 아래 homophily는 query 정답까지 쓰는 분석 전용 지표다.

| Scenario | Same: old good | Same: other | Reversed: new good | Reversed: old good | Same changed edges | Reversed changed edges |
|---|---:|---:|---:|---:|---:|---:|
| consistent | 0.805 | 0.811 | 0.811 | 0.805 | 0.733 | 0.733 |
| noisy | 0.814 | 0.359 | 0.791 | 0.361 | 0.730 | 0.881 |
| conflicting | 0.817 | 0.405 | 0.802 | 0.404 | 0.729 | 0.839 |

## End-to-end robustness

| Scenario | Original acc | Same-role acc | Reversed acc | Reversed−Same (%p, graph95% interval) | Same−Original (%p) |
|---|---:|---:|---:|---|---|
| consistent | 96.53 | 95.27 | 95.27 | +0.00 [+0.00, +0.00] | -1.26 [-2.64, -0.03] |
| noisy | 88.49 | 88.03 | 87.83 | -0.20 [-2.81, +2.45] | -0.46 [-3.22, +2.20] |
| conflicting | 88.96 | 89.19 | 87.11 | -2.09 [-3.99, -0.18] | +0.23 [-1.92, +2.36] |

Reversed−Same는 역할 변경의 matched 비교이고 Same−Original은 재배선 절차 자체의 shift를 보여준다. 두 값을 혼동하지 않는다. 구간이 0을 포함한다고 동등성이 증명되는 것은 아니다.

## Mechanism adaptation의 실제 기여

Frozen-old는 같은-role graph에서 계산한 wx/wy/coupling/ridge를 reversed graph에 고정한다. Shuffled-mechanism은 mechanism 계산에만 context label 순열을 넣고 실제 prediction의 context labels와 clamp는 원래 정답을 쓴다. Uniform은 reversed의 wx/wy를 relation 평균으로 바꾸고 총 scale·coupling·ridge를 보존한다. 모두 하나의 frozen 모델에 대한 진단용 intervention이다.

| Scenario | Adaptive | Old mechanism | Shuffled mechanism | Uniform weights | Adaptive−Old (%p) | Adaptive−Shuffled (%p) | Adaptive−Uniform (%p) |
|---|---:|---:|---:|---:|---|---|---|
| consistent | 95.27 | 95.27 | 94.68 | 95.92 | +0.00 [+0.00, +0.00] | +0.58 [-0.46, +1.55] | -0.65 [-1.28, -0.07] |
| noisy | 87.83 | 40.87 | 75.54 | 78.80 | +46.96 [+42.78, +51.18] | +12.30 [+8.94, +15.90] | +9.03 [+7.82, +10.31] |
| conflicting | 87.11 | 50.60 | 76.60 | 81.07 | +36.51 [+33.31, +39.99] | +10.51 [+8.18, +12.85] | +6.04 [+4.44, +7.54] |

## Label propagation weight tracking

wy share는 relation별 양수 가중치를 합1로 정규화했다. Before는 Same graph에서 old-good의 share, After는 Reversed graph에서 new-good의 share다. Preference는 다른 relation보다 share가1e−6 이상 큰 경우이며, near tie는 반대다. 조건부 추적률은 Before에서 good을 선호한 경우에만 After도 새 good을 선호한 비율이다. Consistent는 good 역할 구분 자체가 없으므로 해석하지 않는다.

| Scenario | Before good share | After new-good share | Before good preference (%) | After good preference (%) | Both followed (%) | Followed given before-good (%) |
|---|---:|---:|---:|---:|---:|---:|
| consistent | 0.500 | 0.500 | 47.22 | 52.78 | 0.00 | 0.00 |
| noisy | 0.894 | 0.913 | 90.00 | 94.44 | 86.11 | 95.68 |
| conflicting | 0.897 | 0.869 | 93.89 | 90.28 | 85.83 | 91.42 |

## k별 결과

| Scenario | k/class | Same acc | Reversed acc | Adaptive−Old (%p) | Adaptive−Shuffled (%p) | After good weight share |
|---|---:|---:|---:|---:|---:|---:|
| consistent | 1 | 88.88 | 88.88 | +0.00 | +0.00 | 0.440 |
| consistent | 3 | 92.24 | 92.24 | +0.00 | -0.31 | 0.509 |
| consistent | 5 | 96.95 | 96.95 | +0.00 | +1.97 | 0.515 |
| consistent | 10 | 99.20 | 99.20 | +0.00 | +0.93 | 0.506 |
| consistent | 20 | 99.06 | 99.06 | +0.00 | +0.34 | 0.530 |
| noisy | 1 | 67.82 | 65.64 | +17.87 | +0.00 | 0.680 |
| noisy | 3 | 88.80 | 89.86 | +48.27 | +28.76 | 0.938 |
| noisy | 5 | 91.60 | 93.72 | +52.45 | +19.45 | 0.961 |
| noisy | 10 | 95.88 | 94.70 | +58.19 | +10.21 | 0.993 |
| noisy | 20 | 96.05 | 95.24 | +58.02 | +3.07 | 0.995 |
| conflicting | 1 | 71.10 | 64.61 | +12.64 | +0.00 | 0.541 |
| conflicting | 3 | 87.35 | 87.18 | +35.36 | +10.87 | 0.887 |
| conflicting | 5 | 94.17 | 92.74 | +43.56 | +28.49 | 0.934 |
| conflicting | 10 | 96.40 | 95.06 | +45.48 | +8.93 | 0.991 |
| conflicting | 20 | 96.94 | 95.94 | +45.52 | +4.24 | 0.995 |

## 확률 품질

| Scenario | Variant | NLL | Brier |
|---|---|---:|---:|
| consistent | Original | 0.4781 | 0.2508 |
| consistent | Same-role resampled | 0.5052 | 0.2675 |
| consistent | Reversed adaptive | 0.5052 | 0.2675 |
| consistent | Reversed / old mechanism | 0.5052 | 0.2675 |
| consistent | Reversed / shuffled mechanism labels | 0.8943 | 0.5300 |
| consistent | Reversed / uniform relation weights | 0.5090 | 0.2698 |
| noisy | Original | 0.5216 | 0.2847 |
| noisy | Same-role resampled | 0.5299 | 0.2911 |
| noisy | Reversed adaptive | 0.5471 | 0.2986 |
| noisy | Reversed / old mechanism | 1.0641 | 0.6501 |
| noisy | Reversed / shuffled mechanism labels | 0.9739 | 0.5870 |
| noisy | Reversed / uniform relation weights | 0.8192 | 0.4788 |
| conflicting | Original | 0.5320 | 0.2905 |
| conflicting | Same-role resampled | 0.5253 | 0.2857 |
| conflicting | Reversed adaptive | 0.5463 | 0.3000 |
| conflicting | Reversed / old mechanism | 0.9971 | 0.6074 |
| conflicting | Reversed / shuffled mechanism labels | 0.9622 | 0.5807 |
| conflicting | Reversed / uniform relation weights | 0.7786 | 0.4529 |

## Seed별 robustness와 adaptation

| Scenario | Seed | Reversed−Same (%p) | Adaptive−Old (%p) | Adaptive−Shuffled (%p) |
|---|---:|---:|---:|---:|
| consistent | 17 | +0.00 | +0.00 | +0.51 |
| consistent | 29 | +0.00 | +0.00 | +0.68 |
| consistent | 43 | +0.00 | +0.00 | +0.57 |
| noisy | 17 | -0.41 | +46.91 | +12.24 |
| noisy | 29 | -0.22 | +47.31 | +12.02 |
| noisy | 43 | +0.05 | +46.66 | +12.63 |
| conflicting | 17 | -2.21 | +36.51 | +10.44 |
| conflicting | 29 | -2.28 | +36.78 | +10.42 |
| conflicting | 43 | -1.77 | +36.24 | +10.67 |

## k=1의 식별 한계와 구조 품질 대조

클래스당 context가 하나이면 서로 다른 context node의 centered one-hot 내적은 항상 −1/C다. 따라서 model의 label_alignment = sum(W_ij * inner(y_i,y_j))/sum(W_ij)는 context pair mass가 양수일 때 항상 −1/C다. Context label의 within-set shuffle은 이 값을 바꾸지 않으므로 k1에서 shuffled-mechanism 효과가0인 것은 구현의 수학적 성질이다. 이것을 전체 예측이 label을 사용하지 않는다는 뜻으로 읽으면 안 된다. Pair mass·feature alignment 등 다른 통계는 여전히 변할 수 있다.

아래 designated-good 2-hop accuracy는 숨은 생성 역할과 query 정답을 쓰는 구조 품질 진단이며 실행 가능한 role 선택기가 아니다. Role reversal은 degree를 보존해도 실제 path coverage·utility 강도까지 동일하게 보존하지는 않는다.

| Scenario | k | Good 2-hop Same | Good 2-hop Reversed | Uniform 2-hop Same | Uniform 2-hop Reversed |
|---|---:|---:|---:|---:|---:|
| consistent | 1 | 64.92 | 68.71 | 79.24 | 79.24 |
| consistent | 3 | 82.42 | 82.24 | 90.63 | 90.63 |
| consistent | 5 | 89.30 | 89.34 | 95.48 | 95.48 |
| consistent | 10 | 93.10 | 92.37 | 97.81 | 97.81 |
| consistent | 20 | 94.64 | 95.07 | 98.92 | 98.92 |
| noisy | 1 | 66.18 | 67.96 | 59.78 | 61.40 |
| noisy | 3 | 83.91 | 83.79 | 72.40 | 71.76 |
| noisy | 5 | 89.84 | 88.31 | 79.31 | 79.03 |
| noisy | 10 | 93.85 | 92.50 | 88.55 | 87.96 |
| noisy | 20 | 95.29 | 94.69 | 92.45 | 93.12 |
| conflicting | 1 | 66.01 | 64.76 | 61.37 | 60.89 |
| conflicting | 3 | 84.06 | 83.66 | 74.86 | 74.67 |
| conflicting | 5 | 90.59 | 89.16 | 80.93 | 79.13 |
| conflicting | 10 | 94.37 | 93.15 | 89.36 | 88.67 |
| conflicting | 20 | 96.41 | 94.87 | 92.94 | 92.58 |

## 현재 판정

Plain은 이 global utility 설정에서 이미 context-dependent relation adaptation을 수행한다. Noisy/Conflict에서 적응된 mechanism이 이전 역할의 mechanism 고정과 uniform/shuffled-context 대조보다 강했다. 그러나 Conflict의 end-to-end 정확도는 역할 교환 후 약2.09%p 낮아 완전한 robustness를 입증하지 않는다. 특히 k1 손실과 미세한 구조 품질 차이를 함께 고려해야 하며, ID 암기라는 결론은 지원하지 않는다.

현재 결과로 새로운 모듈이나 loss 학습을 시작하지 않는다. 다음 연구에서 필요한 것은 Plain이 이미 처리하는 전역 유용 관계 교환을 넘는 실패 조건이지 같은 현상을 새 아키텍처로 재현하는 것이 아니다. 아직 평가하지 않은 semantic operator·국소 utility 변화에 대한 주장은 남겨 둔다.

## 검증과 범위

주 평가3 models×360 tasks×6 variants=6,480 predictions. Implementation checks를 포함한 실제 custom task forward는 6501회, canonical forward는 9회다. 학습 업데이트는0회다. 기존 cached probability와 최대 차이는 2.38e-07, argmax 변경은 0/145080였다.

같은 개발용 graph와 generator에서 global role만 바꾼 실험이다. 새로운 relation 수·타입·semantic mechanism, 국소적으로 달라지는 relation utility나 heterophilic class 변환은 평가하지 않았다. 원래 prior가 이미 random ID의 role variation을 학습시켰으므로, 이 테스트를 통과해도 진짜 unseen-schema 일반화가 입증되는 것은 아니다.

Graph bootstrap은 k와 학습seed를 master 안에서 평균한24개 paired graph의2,000회 탐색적 구간이다. 학습seed 모집단과 다중 비교의 불확실성을 보정하지 않는다. 가중치 변화가 있었다는 관찰과, 그 변화가 정확도에 기여했다는 intervention을 분리해서 해석한다.

[고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/protocol.json) · [그래프 구성 검사](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/results/graph_audit.json) · [평가 검사](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/results/evaluation_audit.json)

## Figure

![Plain role reversal](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/results/plain_role_reversal.png)

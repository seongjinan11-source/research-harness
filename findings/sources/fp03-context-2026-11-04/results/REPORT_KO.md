# FP03 context calibration: 저장 확률 재구성

기존 개발 graph5개/none95의10episode다. 신경 학습과 optimizer update0회이며 context에서 predictive mixture weights를 계산했다. 새 PFN 또는 exact Bayes posterior 검증은 아니다.

감사 PASS; adaptive 사전 개발 gate FAIL. Audit와 성능 판정은 별개다.
139 original context 중126 held points/19 balanced folds,285 calibration branches 및150 final branches를 사용했다. 50 controls+70 reference=120 aggregate scores다. 독립 표본139/435/1890개로 세지 않으며 real CI가 없다.

| Mode | 방법 | Accuracy % | BAcc % | Macro-F1 % | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| none | adaptive | 60.910 | 53.272 | 51.527 | 1.33772 | 0.62142 |
| none | prior | 35.269 | 25.635 | 19.196 | 1.38693 | 0.66661 |
| none | map | 52.425 | 45.724 | 43.972 | 1.42016 | 0.62967 |
| none | fixed_edge | 62.523 | 56.611 | 53.695 | 1.62348 | 0.56913 |
| none | graph_off | 9.640 | 18.190 | 3.577 | 1.83075 | 0.81810 |
| none | old_full | 55.691 | 54.599 | 50.860 | 2.53411 | 0.70453 |
| none | old_edge | 63.750 | 57.432 | 54.869 | 1.62087 | 0.54456 |
| none | lp_harmonic | 50.689 | 43.137 | 39.853 | 1.23455 | 0.57337 |
| none | lp_restart | 66.319 | 59.848 | 57.962 | 1.63741 | 0.76178 |
| none | public_graphpfn | 41.455 | 43.366 | 37.556 | 2.08376 | 0.79619 |
| none | pcfi_public_graphpfn | 41.455 | 43.366 | 37.556 | 2.08376 | 0.79619 |
| none | fisf_public_graphpfn | 59.218 | 55.325 | 52.333 | 1.56541 | 0.59467 |
| 95 | adaptive | 60.804 | 52.935 | 51.226 | 1.33811 | 0.62132 |
| 95 | prior | 35.224 | 25.768 | 19.423 | 1.38907 | 0.66678 |
| 95 | map | 50.151 | 43.696 | 41.413 | 1.41955 | 0.63027 |
| 95 | fixed_edge | 61.347 | 55.173 | 52.336 | 1.63036 | 0.57349 |
| 95 | graph_off | 9.640 | 18.190 | 3.577 | 1.83075 | 0.81810 |
| 95 | old_full | 55.062 | 53.526 | 49.917 | 2.53919 | 0.70626 |
| 95 | old_edge | 62.935 | 56.277 | 53.767 | 1.63325 | 0.55055 |
| 95 | lp_harmonic | 50.689 | 43.137 | 39.853 | 1.23455 | 0.57337 |
| 95 | lp_restart | 66.319 | 59.848 | 57.962 | 1.63741 | 0.76178 |
| 95 | public_graphpfn | 35.785 | 37.387 | 29.083 | 2.16955 | 0.80115 |
| 95 | pcfi_public_graphpfn | 59.330 | 54.259 | 50.935 | 1.58471 | 0.60087 |
| 95 | fisf_public_graphpfn | 60.744 | 55.687 | 52.613 | 1.51081 | 0.58086 |

평균은5 source graph 동일 가중이다. MAP 또는 query-truth best candidate가 adaptive FAIL을 대체하지 않는다.

| Graph | Mode | 방법 | Accuracy % | BAcc % | NLL |
|---|---|---|---:|---:|---:|
| Cora | none | adaptive | 60.231 | 54.971 | 1.49042 |
| Cora | none | prior | 13.326 | 14.448 | 1.59074 |
| Cora | none | map | 46.380 | 43.851 | 1.72853 |
| Cora | none | fixed_edge | 63.169 | 58.730 | 1.49200 |
| Cora | none | graph_off | 13.221 | 14.286 | 1.94591 |
| Cora | none | old_full | 59.601 | 60.022 | 1.73434 |
| Cora | none | old_edge | 64.218 | 59.494 | 1.63194 |
| Cora | none | lp_harmonic | 24.554 | 25.844 | 1.58910 |
| Cora | none | lp_restart | 58.972 | 58.011 | 1.81476 |
| Cora | none | public_graphpfn | 38.615 | 42.119 | 1.80411 |
| Cora | none | pcfi_public_graphpfn | 38.615 | 42.119 | 1.80411 |
| Cora | none | fisf_public_graphpfn | 54.565 | 58.520 | 1.41163 |
| Cora | 95 | adaptive | 58.867 | 53.123 | 1.49137 |
| Cora | 95 | prior | 13.326 | 14.448 | 1.59292 |
| Cora | 95 | map | 47.639 | 45.130 | 1.71486 |
| Cora | 95 | fixed_edge | 62.225 | 58.135 | 1.49962 |
| Cora | 95 | graph_off | 13.221 | 14.286 | 1.94591 |
| Cora | 95 | old_full | 58.972 | 57.936 | 1.74559 |
| Cora | 95 | old_edge | 63.799 | 59.304 | 1.65828 |
| Cora | 95 | lp_harmonic | 24.554 | 25.844 | 1.58910 |
| Cora | 95 | lp_restart | 58.972 | 58.011 | 1.81476 |
| Cora | 95 | public_graphpfn | 20.252 | 26.187 | 2.05399 |
| Cora | 95 | pcfi_public_graphpfn | 48.059 | 52.604 | 1.44763 |
| Cora | 95 | fisf_public_graphpfn | 51.522 | 55.774 | 1.38783 |
| CiteSeer | none | adaptive | 40.674 | 39.839 | 1.66729 |
| CiteSeer | none | prior | 21.707 | 21.256 | 1.72664 |
| CiteSeer | none | map | 40.042 | 43.864 | 1.66886 |
| CiteSeer | none | fixed_edge | 40.991 | 44.664 | 1.63397 |
| CiteSeer | none | graph_off | 7.798 | 16.667 | 1.79176 |
| CiteSeer | none | old_full | 52.898 | 49.929 | 1.43452 |
| CiteSeer | none | old_edge | 41.728 | 45.326 | 1.69716 |
| CiteSeer | none | lp_harmonic | 32.561 | 36.028 | 1.60291 |
| CiteSeer | none | lp_restart | 40.042 | 43.560 | 1.73507 |
| CiteSeer | none | public_graphpfn | 27.292 | 23.924 | 1.73642 |
| CiteSeer | none | pcfi_public_graphpfn | 27.292 | 23.924 | 1.73642 |
| CiteSeer | none | fisf_public_graphpfn | 27.503 | 30.105 | 1.73072 |
| CiteSeer | 95 | adaptive | 40.780 | 40.089 | 1.67303 |
| CiteSeer | 95 | prior | 21.286 | 20.587 | 1.73728 |
| CiteSeer | 95 | map | 29.715 | 32.421 | 1.67276 |
| CiteSeer | 95 | fixed_edge | 38.883 | 41.964 | 1.65832 |
| CiteSeer | 95 | graph_off | 7.798 | 16.667 | 1.79176 |
| CiteSeer | 95 | old_full | 50.053 | 47.822 | 1.44589 |
| CiteSeer | 95 | old_edge | 40.042 | 42.945 | 1.72867 |
| CiteSeer | 95 | lp_harmonic | 32.561 | 36.028 | 1.60291 |
| CiteSeer | 95 | lp_restart | 40.042 | 43.560 | 1.73507 |
| CiteSeer | 95 | public_graphpfn | 9.062 | 16.666 | 2.04391 |
| CiteSeer | 95 | pcfi_public_graphpfn | 31.085 | 30.443 | 1.71881 |
| CiteSeer | 95 | fisf_public_graphpfn | 31.718 | 31.706 | 1.68237 |
| PubMed | none | adaptive | 45.694 | 49.155 | 1.07710 |
| PubMed | none | prior | 38.001 | 33.333 | 1.20724 |
| PubMed | none | map | 43.590 | 44.148 | 1.26842 |
| PubMed | none | fixed_edge | 46.483 | 50.525 | 2.71817 |
| PubMed | none | graph_off | 13.609 | 33.333 | 1.09861 |
| PubMed | none | old_full | 48.389 | 41.435 | 2.09107 |
| PubMed | none | old_edge | 46.680 | 50.326 | 2.20631 |
| PubMed | none | lp_harmonic | 44.970 | 39.965 | 0.90051 |
| PubMed | none | lp_restart | 71.598 | 59.703 | 1.06585 |
| PubMed | none | public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | none | pcfi_public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | none | fisf_public_graphpfn | 59.500 | 54.266 | 0.93989 |
| PubMed | 95 | adaptive | 45.431 | 47.018 | 1.07336 |
| PubMed | 95 | prior | 38.001 | 33.333 | 1.20728 |
| PubMed | 95 | map | 40.894 | 43.374 | 1.28081 |
| PubMed | 95 | fixed_edge | 44.050 | 46.778 | 2.73671 |
| PubMed | 95 | graph_off | 13.609 | 33.333 | 1.09861 |
| PubMed | 95 | old_full | 48.849 | 41.838 | 2.07963 |
| PubMed | 95 | old_edge | 44.773 | 47.173 | 2.22171 |
| PubMed | 95 | lp_harmonic | 44.970 | 39.965 | 0.90051 |
| PubMed | 95 | lp_restart | 71.598 | 59.703 | 1.06585 |
| PubMed | 95 | public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | 95 | pcfi_public_graphpfn | 63.840 | 55.709 | 0.98094 |
| PubMed | 95 | fisf_public_graphpfn | 59.435 | 53.237 | 0.96783 |
| CoauthorCS | none | adaptive | 72.050 | 59.930 | 1.67147 |
| CoauthorCS | none | prior | 23.995 | 14.220 | 1.65134 |
| CoauthorCS | none | map | 45.221 | 35.468 | 1.94141 |
| CoauthorCS | none | fixed_edge | 73.171 | 58.109 | 1.56637 |
| CoauthorCS | none | graph_off | 3.296 | 6.667 | 2.70805 |
| CoauthorCS | none | old_full | 69.018 | 62.970 | 2.31813 |
| CoauthorCS | none | old_edge | 77.324 | 60.989 | 1.84693 |
| CoauthorCS | none | lp_harmonic | 64.469 | 54.881 | 1.50775 |
| CoauthorCS | none | lp_restart | 73.764 | 62.583 | 2.08843 |
| CoauthorCS | none | public_graphpfn | 33.223 | 40.706 | 4.48554 |
| CoauthorCS | none | pcfi_public_graphpfn | 33.223 | 40.706 | 4.48554 |
| CoauthorCS | none | fisf_public_graphpfn | 67.831 | 58.904 | 2.93364 |
| CoauthorCS | 95 | adaptive | 72.314 | 60.097 | 1.67066 |
| CoauthorCS | 95 | prior | 23.204 | 13.896 | 1.65000 |
| CoauthorCS | 95 | map | 45.946 | 35.969 | 1.93160 |
| CoauthorCS | 95 | fixed_edge | 72.841 | 57.980 | 1.55964 |
| CoauthorCS | 95 | graph_off | 3.296 | 6.667 | 2.70805 |
| CoauthorCS | 95 | old_full | 68.952 | 61.786 | 2.32803 |
| CoauthorCS | 95 | old_edge | 77.324 | 60.956 | 1.84355 |
| CoauthorCS | 95 | lp_harmonic | 64.469 | 54.881 | 1.50775 |
| CoauthorCS | 95 | lp_restart | 73.764 | 62.583 | 2.08843 |
| CoauthorCS | 95 | public_graphpfn | 39.552 | 34.219 | 4.37208 |
| CoauthorCS | 95 | pcfi_public_graphpfn | 66.777 | 56.946 | 2.90460 |
| CoauthorCS | 95 | fisf_public_graphpfn | 70.138 | 63.404 | 2.68220 |
| CoauthorPhysics | none | adaptive | 85.903 | 62.465 | 0.78230 |
| CoauthorPhysics | none | prior | 79.315 | 44.917 | 0.75865 |
| CoauthorPhysics | none | map | 86.891 | 61.291 | 0.49357 |
| CoauthorPhysics | none | fixed_edge | 88.801 | 71.026 | 0.70690 |
| CoauthorPhysics | none | graph_off | 10.277 | 20.000 | 1.60944 |
| CoauthorPhysics | none | old_full | 48.551 | 58.641 | 5.09251 |
| CoauthorPhysics | none | old_edge | 88.801 | 71.026 | 0.72202 |
| CoauthorPhysics | none | lp_harmonic | 86.891 | 58.966 | 0.57250 |
| CoauthorPhysics | none | lp_restart | 87.220 | 75.383 | 1.48294 |
| CoauthorPhysics | none | public_graphpfn | 63.900 | 65.380 | 1.33187 |
| CoauthorPhysics | none | pcfi_public_graphpfn | 63.900 | 65.380 | 1.33187 |
| CoauthorPhysics | none | fisf_public_graphpfn | 86.693 | 74.830 | 0.81115 |
| CoauthorPhysics | 95 | adaptive | 86.627 | 64.350 | 0.78211 |
| CoauthorPhysics | 95 | prior | 80.303 | 46.576 | 0.75787 |
| CoauthorPhysics | 95 | map | 86.561 | 61.584 | 0.49774 |
| CoauthorPhysics | 95 | fixed_edge | 88.735 | 71.007 | 0.69749 |
| CoauthorPhysics | 95 | graph_off | 10.277 | 20.000 | 1.60944 |
| CoauthorPhysics | 95 | old_full | 48.485 | 58.249 | 5.09684 |
| CoauthorPhysics | 95 | old_edge | 88.735 | 71.007 | 0.71406 |
| CoauthorPhysics | 95 | lp_harmonic | 86.891 | 58.966 | 0.57250 |
| CoauthorPhysics | 95 | lp_restart | 87.220 | 75.383 | 1.48294 |
| CoauthorPhysics | 95 | public_graphpfn | 65.810 | 65.159 | 1.31693 |
| CoauthorPhysics | 95 | pcfi_public_graphpfn | 86.891 | 75.593 | 0.87156 |
| CoauthorPhysics | 95 | fisf_public_graphpfn | 90.909 | 74.315 | 0.83383 |

## 열 가지 개발 gate

| 조건 | 차이 | 사전 기준 | 판정 |
|---|---:|---:|---|
| physics_gain_old_full | 0.373518 | >= 0.200000 | PASS |
| cs_gain_old_full | 0.030323 | >= 0.050000 | FAIL |
| citeseer_accuracy_floor_old_full | -0.122234 | >= -0.010000 | FAIL |
| citeseer_nll_ceiling_old_full | 0.232778 | <= 0.050000 | FAIL |
| mean_balanced_accuracy_floor_old_edge | -0.041600 | >= 0.000000 | FAIL |
| mean_nll_ceiling_old_edge | -0.283157 | <= 0.050000 | PASS |
| mean_accuracy_gain_prior | 0.256414 | >= 0.010000 | PASS |
| mean_accuracy_floor_fixed_edge | -0.016125 | >= 0.000000 | FAIL |
| mean_nll_ceiling_prior | -0.049208 | <= 0.010000 | PASS |
| mean_nll_ceiling_fixed_edge | -0.285764 | <= 0.010000 | PASS |

## Calibration 진단

| Graph | Context/calibration points/folds | Entropy | ESS | KL to prior | MAP index |
|---|---|---:|---:|---:|---:|
| Cora | 28/28/4 | 2.21809 | 8.380 | 0.48996 | 9 |
| CiteSeer | 24/24/4 | 2.26197 | 8.809 | 0.44608 | 7 |
| PubMed | 12/12/4 | 1.72647 | 4.347 | 0.98158 | 1 |
| CoauthorCS | 55/42/3 | 2.17052 | 8.221 | 0.53753 | 6 |
| CoauthorPhysics | 20/20/4 | 1.74775 | 4.783 | 0.96030 | 7 |

CS represented classes14/K15, selected42/unused13이며 extras는 calibration에서 모두 가리고 final에서는 원 context55개를 사용했다. Class9 query4개를 포함한 모든 minority 지표는 summary.json에 남겼다. Weights/ESS/entropy는 predictive mixture의 기술 통계이며 calibrated posterior uncertainty가 아니다. None/95 weights는 bit-identical이다.

## 95−none 추가 피처 효과

| 방법 | 평균 Accuracy 차이 %p | 평균 BAcc 차이 %p | 평균 NLL 차이 |
|---|---:|---:|---:|
| adaptive | -0.107 | -0.337 | +0.00039 |
| prior | -0.045 | +0.133 | +0.00215 |
| map | -2.273 | -2.029 | -0.00061 |
| fixed_edge | -1.176 | -1.438 | +0.00687 |
| graph_off | +0.000 | +0.000 | +0.00000 |
| old_full | -0.629 | -1.073 | +0.00508 |
| old_edge | -0.816 | -1.155 | +0.01238 |
| lp_harmonic | +0.000 | +0.000 | +0.00000 |
| lp_restart | +0.000 | +0.000 | +0.00000 |
| public_graphpfn | -5.671 | -5.980 | +0.08579 |
| pcfi_public_graphpfn | +17.875 | +10.893 | -0.49905 |
| fisf_public_graphpfn | +1.526 | +0.362 | -0.05460 |

Graph-only calibration이므로 weight 변화가 feature mechanism adaptation을 뜻하지 않는다. 신규 shuffle이 없어 alignment 성공을 확정하지 않는다. None 근처로 수축한 harm reduction과 추가 feature 효용을 구분한다. Per-graph 효과는 summary.json에 있다.

## Query-truth oracle: family headroom 진단 전용

사전에 고정한 ANALYSIS_SPEC에 따른 기술 분석이다. 실제 predictor/weight/primary/tuning에는 쓰지 않았다. Accuracy/NLL 각각 다른 best candidate일 수 있다.

| Graph | Mode | best accuracy index | Accuracy % | best NLL index | NLL |
|---|---|---:|---:|---:|---:|
| Cora | none | 12 | 64.428 | 10 | 1.49200 |
| Cora | 95 | 12 | 64.218 | 10 | 1.49962 |
| CiteSeer | none | 13 | 41.834 | 10 | 1.63397 |
| CiteSeer | 95 | 13 | 41.623 | 10 | 1.65832 |
| PubMed | none | 2 | 48.389 | 4 | 1.09861 |
| PubMed | 95 | 2 | 48.389 | 4 | 1.09861 |
| CoauthorCS | none | 13 | 76.928 | 10 | 1.56637 |
| CoauthorCS | 95 | 13 | 76.928 | 10 | 1.55964 |
| CoauthorPhysics | none | 10 | 88.801 | 7 | 0.49357 |
| CoauthorPhysics | 95 | 10 | 88.735 | 7 | 0.49774 |

All15 candidate 지표·weighted contrast moments·prior KL·mean pairwise TV 및 weighted TV-to-mixture는 summary.json에 모두 보존했다. Family headroom이 있어도 극소 context로 회수할 수 있다는 증명은 아니다.

## 비용과 범위

Calibration inference 0.730s, final inference 0.654s, final preparation 0.030s, CUDA peak 399.53MiB. Execution attempts 합산 wall 3.891s이며 실패 0개다. Cold load/lock validation/audit 및 inherited 비용은 budget.json과 lock lineage에 분리했다.
Cora/CiteSeer full, 나머지 induced2000 범위이며 모두 재사용 development다. Graph별 predictor를 선택하지 않았고 no neural training/no independent real CI/no new PFN claim을 유지한다. 정상 성능 FAIL과 구현/audit 실패를 분리한다. Gate 미통과 후 a/b/score scale로 같은 bank를 구제하지 않는다.

원자료: summary.json, audit.json, budget.json, evaluation.json, artifact_manifest.json 및 calibration/predictions의 NPZ·sidecar. 각 저장 확률과 source/checkpoint/bank/정답 identity의 hash가 연결돼 있다.

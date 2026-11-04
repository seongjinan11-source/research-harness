# FP02 Stage 0: 저장 확률 재채점

기존 개발 graph 5개, rep0/seed11, none/95의 고정 10개 episode다. 새 학습·신경 optimizer 업데이트는 0회이며 통계 nuisance 추론 갱신은 포함한다.

감사: PASS. 새 예측 30개와 재사용 비교 70개를 동일 query/truth/context로 재채점했다.
사전 DC 개발 gate: FAIL. SBM은 별도 결과이며 DC의 실패를 대체하지 않는다.

| 조건 | 방법 | Accuracy % | BAcc % | Macro-F1 % | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| none | poisson_fixed | 47.226 | 51.961 | 45.244 | 4.07794 | 0.89969 |
| none | sbm_update | 42.150 | 49.246 | 42.303 | 5.79886 | 1.07325 |
| none | dc_update | 43.805 | 48.989 | 43.951 | 4.21414 | 0.98569 |
| none | old_full | 55.691 | 54.599 | 50.860 | 2.53411 | 0.70453 |
| none | old_edge | 63.750 | 57.432 | 54.869 | 1.62087 | 0.54456 |
| none | lp_harmonic | 50.689 | 43.137 | 39.853 | 1.23455 | 0.57337 |
| none | lp_restart | 66.319 | 59.848 | 57.962 | 1.63741 | 0.76178 |
| none | public_graphpfn | 41.455 | 43.366 | 37.556 | 2.08376 | 0.79619 |
| none | pcfi_public_graphpfn | 41.455 | 43.366 | 37.556 | 2.08376 | 0.79619 |
| none | fisf_public_graphpfn | 59.218 | 55.325 | 52.333 | 1.56541 | 0.59467 |
| 95 | poisson_fixed | 47.070 | 51.716 | 44.910 | 4.07563 | 0.90216 |
| 95 | sbm_update | 42.237 | 49.389 | 42.424 | 5.83288 | 1.07009 |
| 95 | dc_update | 44.050 | 49.025 | 44.185 | 4.20907 | 0.98369 |
| 95 | old_full | 55.062 | 53.526 | 49.917 | 2.53919 | 0.70626 |
| 95 | old_edge | 62.935 | 56.277 | 53.767 | 1.63325 | 0.55055 |
| 95 | lp_harmonic | 50.689 | 43.137 | 39.853 | 1.23455 | 0.57337 |
| 95 | lp_restart | 66.319 | 59.848 | 57.962 | 1.63741 | 0.76178 |
| 95 | public_graphpfn | 35.785 | 37.387 | 29.083 | 2.16955 | 0.80115 |
| 95 | pcfi_public_graphpfn | 59.330 | 54.259 | 50.935 | 1.58471 | 0.60087 |
| 95 | fisf_public_graphpfn | 60.744 | 55.687 | 52.613 | 1.51081 | 0.58086 |

모든 평균은 source graph 5개에 동일 가중이다. Node·seed를 독립 표본으로 세거나 real CI를 붙이지 않았다.

| Graph | 조건 | 방법 | Accuracy % | BAcc % | NLL |
|---|---|---|---:|---:|---:|
| Cora | none | poisson_fixed | 59.496 | 63.256 | 2.08437 |
| Cora | none | sbm_update | 54.984 | 59.142 | 3.05410 |
| Cora | none | dc_update | 59.077 | 64.299 | 2.59998 |
| Cora | none | old_full | 59.601 | 60.022 | 1.73434 |
| Cora | none | old_edge | 64.218 | 59.494 | 1.63194 |
| Cora | none | lp_harmonic | 24.554 | 25.844 | 1.58910 |
| Cora | none | lp_restart | 58.972 | 58.011 | 1.81476 |
| Cora | none | public_graphpfn | 38.615 | 42.119 | 1.80411 |
| Cora | none | pcfi_public_graphpfn | 38.615 | 42.119 | 1.80411 |
| Cora | none | fisf_public_graphpfn | 54.565 | 58.520 | 1.41163 |
| Cora | 95 | poisson_fixed | 59.811 | 63.365 | 2.06642 |
| Cora | 95 | sbm_update | 55.194 | 59.122 | 3.00523 |
| Cora | 95 | dc_update | 58.867 | 63.776 | 2.61049 |
| Cora | 95 | old_full | 58.972 | 57.936 | 1.74559 |
| Cora | 95 | old_edge | 63.799 | 59.304 | 1.65828 |
| Cora | 95 | lp_harmonic | 24.554 | 25.844 | 1.58910 |
| Cora | 95 | lp_restart | 58.972 | 58.011 | 1.81476 |
| Cora | 95 | public_graphpfn | 20.252 | 26.187 | 2.05399 |
| Cora | 95 | pcfi_public_graphpfn | 48.059 | 52.604 | 1.44763 |
| Cora | 95 | fisf_public_graphpfn | 51.522 | 55.774 | 1.38783 |
| CiteSeer | none | poisson_fixed | 44.784 | 42.472 | 1.90332 |
| CiteSeer | none | sbm_update | 37.935 | 36.404 | 2.98358 |
| CiteSeer | none | dc_update | 35.090 | 35.191 | 2.89470 |
| CiteSeer | none | old_full | 52.898 | 49.929 | 1.43452 |
| CiteSeer | none | old_edge | 41.728 | 45.326 | 1.69716 |
| CiteSeer | none | lp_harmonic | 32.561 | 36.028 | 1.60291 |
| CiteSeer | none | lp_restart | 40.042 | 43.560 | 1.73507 |
| CiteSeer | none | public_graphpfn | 27.292 | 23.924 | 1.73642 |
| CiteSeer | none | pcfi_public_graphpfn | 27.292 | 23.924 | 1.73642 |
| CiteSeer | none | fisf_public_graphpfn | 27.503 | 30.105 | 1.73072 |
| CiteSeer | 95 | poisson_fixed | 43.625 | 41.317 | 1.95669 |
| CiteSeer | 95 | sbm_update | 39.410 | 37.820 | 2.89625 |
| CiteSeer | 95 | dc_update | 36.459 | 36.229 | 2.95218 |
| CiteSeer | 95 | old_full | 50.053 | 47.822 | 1.44589 |
| CiteSeer | 95 | old_edge | 40.042 | 42.945 | 1.72867 |
| CiteSeer | 95 | lp_harmonic | 32.561 | 36.028 | 1.60291 |
| CiteSeer | 95 | lp_restart | 40.042 | 43.560 | 1.73507 |
| CiteSeer | 95 | public_graphpfn | 9.062 | 16.666 | 2.04391 |
| CiteSeer | 95 | pcfi_public_graphpfn | 31.085 | 30.443 | 1.71881 |
| CiteSeer | 95 | fisf_public_graphpfn | 31.718 | 31.706 | 1.68237 |
| PubMed | none | poisson_fixed | 38.462 | 48.536 | 3.59403 |
| PubMed | none | sbm_update | 31.755 | 44.291 | 7.70026 |
| PubMed | none | dc_update | 42.275 | 42.273 | 4.05819 |
| PubMed | none | old_full | 48.389 | 41.435 | 2.09107 |
| PubMed | none | old_edge | 46.680 | 50.326 | 2.20631 |
| PubMed | none | lp_harmonic | 44.970 | 39.965 | 0.90051 |
| PubMed | none | lp_restart | 71.598 | 59.703 | 1.06585 |
| PubMed | none | public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | none | pcfi_public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | none | fisf_public_graphpfn | 59.500 | 54.266 | 0.93989 |
| PubMed | 95 | poisson_fixed | 38.527 | 48.685 | 3.58693 |
| PubMed | 95 | sbm_update | 30.966 | 43.760 | 7.98863 |
| PubMed | 95 | dc_update | 42.406 | 42.376 | 4.03824 |
| PubMed | 95 | old_full | 48.849 | 41.838 | 2.07963 |
| PubMed | 95 | old_edge | 44.773 | 47.173 | 2.22171 |
| PubMed | 95 | lp_harmonic | 44.970 | 39.965 | 0.90051 |
| PubMed | 95 | lp_restart | 71.598 | 59.703 | 1.06585 |
| PubMed | 95 | public_graphpfn | 44.247 | 44.702 | 1.06083 |
| PubMed | 95 | pcfi_public_graphpfn | 63.840 | 55.709 | 0.98094 |
| PubMed | 95 | fisf_public_graphpfn | 59.435 | 53.237 | 0.96783 |
| CoauthorCS | none | poisson_fixed | 60.646 | 55.229 | 3.53945 |
| CoauthorCS | none | sbm_update | 62.953 | 59.417 | 3.47825 |
| CoauthorCS | none | dc_update | 62.294 | 57.803 | 3.72223 |
| CoauthorCS | none | old_full | 69.018 | 62.970 | 2.31813 |
| CoauthorCS | none | old_edge | 77.324 | 60.989 | 1.84693 |
| CoauthorCS | none | lp_harmonic | 64.469 | 54.881 | 1.50775 |
| CoauthorCS | none | lp_restart | 73.764 | 62.583 | 2.08843 |
| CoauthorCS | none | public_graphpfn | 33.223 | 40.706 | 4.48554 |
| CoauthorCS | none | pcfi_public_graphpfn | 33.223 | 40.706 | 4.48554 |
| CoauthorCS | none | fisf_public_graphpfn | 67.831 | 58.904 | 2.93364 |
| CoauthorCS | 95 | poisson_fixed | 60.646 | 55.229 | 3.53764 |
| CoauthorCS | 95 | sbm_update | 62.492 | 59.264 | 3.55101 |
| CoauthorCS | 95 | dc_update | 62.426 | 57.795 | 3.71378 |
| CoauthorCS | 95 | old_full | 68.952 | 61.786 | 2.32803 |
| CoauthorCS | 95 | old_edge | 77.324 | 60.956 | 1.84355 |
| CoauthorCS | 95 | lp_harmonic | 64.469 | 54.881 | 1.50775 |
| CoauthorCS | 95 | lp_restart | 73.764 | 62.583 | 2.08843 |
| CoauthorCS | 95 | public_graphpfn | 39.552 | 34.219 | 4.37208 |
| CoauthorCS | 95 | pcfi_public_graphpfn | 66.777 | 56.946 | 2.90460 |
| CoauthorCS | 95 | fisf_public_graphpfn | 70.138 | 63.404 | 2.68220 |
| CoauthorPhysics | none | poisson_fixed | 32.740 | 50.314 | 9.26854 |
| CoauthorPhysics | none | sbm_update | 23.123 | 46.977 | 11.77810 |
| CoauthorPhysics | none | dc_update | 20.290 | 45.380 | 7.79559 |
| CoauthorPhysics | none | old_full | 48.551 | 58.641 | 5.09251 |
| CoauthorPhysics | none | old_edge | 88.801 | 71.026 | 0.72202 |
| CoauthorPhysics | none | lp_harmonic | 86.891 | 58.966 | 0.57250 |
| CoauthorPhysics | none | lp_restart | 87.220 | 75.383 | 1.48294 |
| CoauthorPhysics | none | public_graphpfn | 63.900 | 65.380 | 1.33187 |
| CoauthorPhysics | none | pcfi_public_graphpfn | 63.900 | 65.380 | 1.33187 |
| CoauthorPhysics | none | fisf_public_graphpfn | 86.693 | 74.830 | 0.81115 |
| CoauthorPhysics | 95 | poisson_fixed | 32.740 | 49.986 | 9.23048 |
| CoauthorPhysics | 95 | sbm_update | 23.123 | 46.977 | 11.72330 |
| CoauthorPhysics | 95 | dc_update | 20.092 | 44.950 | 7.73067 |
| CoauthorPhysics | 95 | old_full | 48.485 | 58.249 | 5.09684 |
| CoauthorPhysics | 95 | old_edge | 88.735 | 71.007 | 0.71406 |
| CoauthorPhysics | 95 | lp_harmonic | 86.891 | 58.966 | 0.57250 |
| CoauthorPhysics | 95 | lp_restart | 87.220 | 75.383 | 1.48294 |
| CoauthorPhysics | 95 | public_graphpfn | 65.810 | 65.159 | 1.31693 |
| CoauthorPhysics | 95 | pcfi_public_graphpfn | 86.891 | 75.593 | 0.87156 |
| CoauthorPhysics | 95 | fisf_public_graphpfn | 90.909 | 74.315 | 0.83383 |

## dc_update 개발 gate: FAIL

| 사전 조건 | 실제 차이 | 기준 | 판정 |
|---|---:|---:|---|
| physics_gain_old_full | -0.282609 | >= 0.200000 | FAIL |
| cs_gain_old_full | -0.067238 | >= 0.050000 | FAIL |
| citeseer_accuracy_floor_old_full | -0.178082 | >= -0.010000 | FAIL |
| citeseer_nll_ceiling_old_full | 1.460187 | <= 0.050000 | FAIL |
| mean_balanced_accuracy_floor_old_full | -0.056103 | >= 0.000000 | FAIL |
| mean_balanced_accuracy_floor_old_edge | -0.084430 | >= 0.000000 | FAIL |
| mean_nll_ceiling_old_edge | 2.593265 | <= 0.050000 | FAIL |

## sbm_update 개발 gate: FAIL

| 사전 조건 | 실제 차이 | 기준 | 판정 |
|---|---:|---:|---|
| physics_gain_old_full | -0.254282 | >= 0.200000 | FAIL |
| cs_gain_old_full | -0.060646 | >= 0.050000 | FAIL |
| citeseer_accuracy_floor_old_full | -0.149631 | >= -0.010000 | FAIL |
| citeseer_nll_ceiling_old_full | 1.549068 | <= 0.050000 | FAIL |
| mean_balanced_accuracy_floor_old_full | -0.053531 | >= 0.000000 | FAIL |
| mean_balanced_accuracy_floor_old_edge | -0.081857 | >= 0.000000 | FAIL |
| mean_nll_ceiling_old_edge | 4.177984 | <= 0.050000 | FAIL |

## 피처 관측 95% − none

이 차이는 feature가 population/affinity/degree 추론에 미친 전체 효과다. 신규 shuffle 대조가 없으므로 alignment 또는 새 피처 효용 성공을 확정하지 않는다.

| Graph | 방법 | Accuracy 차이 %p | BAcc 차이 %p | NLL 차이 |
|---|---|---:|---:|---:|
| Cora | poisson_fixed | +0.315 | +0.110 | -0.01795 |
| CiteSeer | poisson_fixed | -1.159 | -1.155 | +0.05337 |
| PubMed | poisson_fixed | +0.066 | +0.149 | -0.00711 |
| CoauthorCS | poisson_fixed | +0.000 | +0.000 | -0.00182 |
| CoauthorPhysics | poisson_fixed | +0.000 | -0.328 | -0.03806 |
| Equal-five-graph mean | poisson_fixed | -0.156 | -0.245 | -0.00231 |
| Cora | sbm_update | +0.210 | -0.020 | -0.04887 |
| CiteSeer | sbm_update | +1.475 | +1.416 | -0.08734 |
| PubMed | sbm_update | -0.789 | -0.531 | +0.28837 |
| CoauthorCS | sbm_update | -0.461 | -0.154 | +0.07276 |
| CoauthorPhysics | sbm_update | +0.000 | +0.000 | -0.05480 |
| Equal-five-graph mean | sbm_update | +0.087 | +0.142 | +0.03403 |
| Cora | dc_update | -0.210 | -0.523 | +0.01051 |
| CiteSeer | dc_update | +1.370 | +1.039 | +0.05747 |
| PubMed | dc_update | +0.131 | +0.103 | -0.01995 |
| CoauthorCS | dc_update | +0.132 | -0.007 | -0.00845 |
| CoauthorPhysics | dc_update | -0.198 | -0.430 | -0.06492 |
| Equal-five-graph mean | dc_update | +0.245 | +0.036 | -0.00507 |
| Cora | old_full | -0.630 | -2.087 | +0.01125 |
| CiteSeer | old_full | -2.845 | -2.107 | +0.01137 |
| PubMed | old_full | +0.460 | +0.404 | -0.01144 |
| CoauthorCS | old_full | -0.066 | -1.184 | +0.00990 |
| CoauthorPhysics | old_full | -0.066 | -0.392 | +0.00433 |
| Equal-five-graph mean | old_full | -0.629 | -1.073 | +0.00508 |
| Cora | old_edge | -0.420 | -0.189 | +0.02634 |
| CiteSeer | old_edge | -1.686 | -2.380 | +0.03151 |
| PubMed | old_edge | -1.907 | -3.153 | +0.01539 |
| CoauthorCS | old_edge | +0.000 | -0.033 | -0.00338 |
| CoauthorPhysics | old_edge | -0.066 | -0.019 | -0.00797 |
| Equal-five-graph mean | old_edge | -0.816 | -1.155 | +0.01238 |
| Cora | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| CiteSeer | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| PubMed | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| CoauthorCS | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| CoauthorPhysics | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| Equal-five-graph mean | lp_harmonic | +0.000 | +0.000 | +0.00000 |
| Cora | lp_restart | +0.000 | +0.000 | +0.00000 |
| CiteSeer | lp_restart | +0.000 | +0.000 | +0.00000 |
| PubMed | lp_restart | +0.000 | +0.000 | +0.00000 |
| CoauthorCS | lp_restart | +0.000 | +0.000 | +0.00000 |
| CoauthorPhysics | lp_restart | +0.000 | +0.000 | +0.00000 |
| Equal-five-graph mean | lp_restart | +0.000 | +0.000 | +0.00000 |
| Cora | public_graphpfn | -18.363 | -15.932 | +0.24987 |
| CiteSeer | public_graphpfn | -18.230 | -7.258 | +0.30748 |
| PubMed | public_graphpfn | +0.000 | +0.000 | +0.00000 |
| CoauthorCS | public_graphpfn | +6.328 | -6.487 | -0.11346 |
| CoauthorPhysics | public_graphpfn | +1.910 | -0.221 | -0.01494 |
| Equal-five-graph mean | public_graphpfn | -5.671 | -5.980 | +0.08579 |
| Cora | pcfi_public_graphpfn | +9.444 | +10.484 | -0.35648 |
| CiteSeer | pcfi_public_graphpfn | +3.793 | +6.519 | -0.01762 |
| PubMed | pcfi_public_graphpfn | +19.592 | +11.007 | -0.07989 |
| CoauthorCS | pcfi_public_graphpfn | +33.553 | +16.240 | -1.58094 |
| CoauthorPhysics | pcfi_public_graphpfn | +22.991 | +10.213 | -0.46031 |
| Equal-five-graph mean | pcfi_public_graphpfn | +17.875 | +10.893 | -0.49905 |
| Cora | fisf_public_graphpfn | -3.043 | -2.746 | -0.02380 |
| CiteSeer | fisf_public_graphpfn | +4.215 | +1.600 | -0.04835 |
| PubMed | fisf_public_graphpfn | -0.066 | -1.029 | +0.02794 |
| CoauthorCS | fisf_public_graphpfn | +2.307 | +4.500 | -0.25144 |
| CoauthorPhysics | fisf_public_graphpfn | +4.216 | -0.515 | +0.02268 |
| Equal-five-graph mean | fisf_public_graphpfn | +1.526 | +0.362 | -0.05460 |

None 근처로 수축해 기존 feature 손상이 줄어든 것을 추가 feature 정보 활용과 구분한다. Class별 recall/support와 predicted/true class mass는 summary.json에 모두 보존했다.

## 비용과 판정 범위

새 forward 30회. Arm별 prediction seconds: {"dc_update":0.5241216998547316,"poisson_fixed":0.8980778995901346,"sbm_update":0.5456572994589806}. CUDA peak 52.40 MiB. Shared feature-table 비용 0.096s는 한 번씩만 집계했다.
기록된 실행 attempt 1개, 실패 0개, 합산 wall 5.374s. 로그 없는 중단 비용을 0이라고 추정하지 않는다.

Cora/CiteSeer는 기존 full graph, PubMed/CS/Physics는 기존 induced graph 범위다. Full/edge/DC를 graph마다 선택하지 않았다. Classical degree/affinity/population inference의 개발 실험이며 새 PFN 학습, 방법 novelty, 독립 real holdout 또는 Stage 1의 최종 성공을 뜻하지 않는다. 정상 계산의 성능 실패와 감사/구현 실패를 구분한다.

Theta·affinity·population 및 fresh sufficient statistics는 NPZ의 operator__ 배열에 보존했다. JSON의 긴 vector는 요약/hash이며 원래 값은 NPZ에 있다. max_rate는 모든 class 상태에 대한 pair intensity 최대값이며 q-weighted 평균 rate가 아니다. Binary graph의 Poisson saturation 한계를 숨기거나 rate를 몰래 clip하지 않는다.

원자료: summary.json, audit.json, budget.json, evaluation.json, artifact_manifest.json. Source·bank·checkpoint·예측 hash를 manifest에 연결했다. Stage 1은 별도 사전 명세와 source lock, 실제 6 fits 및 새 final 평가가 필요하다.

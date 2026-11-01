# Observation-aware Subgraph Policy — frozen GraphPFN pilot

2026-11-01. 판정: **현재 observation policy 우위 미입증; 확장 보류**.

이 실험은 예측기를 고르는 selector가 아니라 GraphPFN에 전달할 induced subgraph를 선택하는 작은 정책을 학습했다. GraphPFN의 모든 parameter는 고정했다. LP/BP/EAJ head나 추가 predictive loss를 사용하지 않았다.

## 결과 해석

가장 큰 제한은 사후 기본 대조에서 드러났다. Context의 class 빈도에 Laplace 1 smoothing만 적용한 확률 예측은 NLL 0.85241, Brier 0.55736이었다. Observation 정책의 64-node matched/cluster OOD NLL 1.07193/1.08551보다 낮다. 따라서 아래 PPR·Structure 대비 개선을 강한 확률 예측 메소드가 확보됐다는 근거로 해석하지 않는다. 이 sanity check는 학습과 평가 완료 후 추가했으며, 기존 정책·checkpoint·주 성공 기준을 바꾸지 않았다.

64-node 주 평가에서 mask 입력의 추가 가치는 NLL 기준으로 나타났다. Observation−Structure 차이는 matched −0.16128 [−0.27147, −0.05871], cluster OOD −0.24695 [−0.38936, −0.11246]였다. Observation−PPR도 각각 −0.14488 [−0.27591, −0.01535], −0.26857 [−0.47820, −0.07895]였다. 이는 같은 소형 정책에서 관측 상태를 활용한 입력 선택이 도움이 될 수 있다는 증거다. 다만 실제 feature 값의 인과적 기여를 분리한 intervention 결과는 아니다.

강한 Evidence-paths 규칙과 비교하면 달라진다. Observation−Paths NLL은 matched +0.01839 [−0.06529, +0.10213], cluster OOD +0.00866 [−0.11473, +0.11820]다. 두 점추정치는 규칙이 조금 낮고, CI는 모두 0을 포함한다. 규칙의 확실한 우위라고도, 학습 정책과 동등하다고도 말하지 않는다. 현재 학습 정책이 강한 규칙을 넘었다는 근거는 확보하지 못했다. 같은 조건의 accuracy 차이 역시 −0.91%p, −1.11%p이며 두 CI가 0을 포함한다.

사전 보조 평가인 32-node cluster OOD에서는 Observation NLL 1.26040, Paths 1.45324로 방향이 바뀐다. 차이는 −0.19284 [−0.39771, +0.00284], accuracy는 +5.01%p [−0.39, +11.00]다. 작은 예산에 관한 흥미로운 개발 신호지만 둘 다 CI에 0이 포함된다. 이 조건을 새 주 평가로 승격하거나, 예산을 골라 성공했다고 정리하지 않는다. 32-node matched의 규칙 대비 NLL 차이 −0.08314 [−0.26173, +0.09400]도 우위를 확정하지 못했다.

판정은 **mask-aware 입력 구성의 효용은 보존하되, 이 작은 학습 정책의 확장은 보류**다. PPR만 이겼다는 이유로 메인 메소드로 삼지 않는다. 이 결과가 모든 learned retrieval을 부정하지는 않는다. Shared node MLP가 mask·geometry만 사용하고, 작은 공식 prior graph와 128 updates로 학습한 구현에 대한 판정이다. 실제 heterograph 전이와 새로운 메소드 novelty는 아직 검증하지 않았다.

비용 측면에서는 matched budget 64의 관측된 선택 단계 평균이 policy 1.76ms, Paths 13.73ms였다. PCFI+forward까지 더하면 각각 약 146.82ms, 155.90ms다. 공통 full-graph 통계 전처리와 정책 학습 비용은 이 합계 밖에 있다. 측정 변동·quality noninferiority margin·큰 graph 검증이 없으므로 이 차이를 end-to-end 효율 우위나 새로운 효율 논문의 근거로 승격하지 않는다.

## 고정 프로토콜

- 공식 GraphPFN prior의 새 graph 64/32개를 train/test로 분리했다. 기존 regret/program/BP bank는 재사용하지 않았다.
- 원래 generator context에서 class당 최대 4개 label anchor를 뽑고, 원래 query에서 8개를 뽑았다. 선택한 context와 query는 모든 방법에서 동일하다. 다른 available context labels는 halo에 들어와도 제공하지 않는다.
- 모든 method는 전체 관측 graph와 mask를 볼 수 있다. 예산은 GraphPFN resident node 32/64/96개이며 context와 query도 포함한다. 탐색에 접근한 총 node 수나 feature 획득 비용을 제한한 실험은 아니다.
- 정책은 uniform whole-node 50/90/99.5% missing과 node budget 64에서만 학습했다. 평가는 uniform 90/99.5% matched, clustered 90/99.5% OOD, cell 99.5% OOD, full features다. 32/96 budget도 학습에서 보지 않았다. 같은 generator family이며 unseen semantic schema는 아니다.
- 관측값만으로 feature별 RMS를 구하고 centering 없이 나누고 ±10으로 clip했다. 공유 통계는 full observed graph에서 구한다. hidden value는 쓰지 않는다. 이전 regret pilot과 다른 전처리이므로 두 보고서의 숫자를 직접 전후 개선으로 연결하지 않는다.
- PCFI는 선택한 부분그래프 안에서 100회 다시 계산한다. 선택 밖 observed donor의 imputed feature를 가져오지 않는다. 추가 random feature는 0개, single-member 공개 GraphPFN이며 official ensemble benchmark 결과가 아니다.

## 모델과 대조군

Random은 uniform halo, PPR은 query와 context에서 출발한 두 PPR의 max-normalized 합으로 halo를 고른다. Evidence-paths는 query에서 context 또는 observed-feature donor로 이어지는 shortest path를, 추가 resident node 수가 적은 순서로 닫고 남는 예산을 PPR로 채운다. 이는 이 pilot의 고정 규칙이며 공식 LoGIC 구현이나 최적 set-cover가 아니다.

Structure/Observation 정책은 모두 17→32→32→1 shared MLP, 1,665 parameters다. degree, query/context PPR와 거리, terminal flag와 예산 비율을 받는다. Observation은 추가로 관측 비율·observed donor PPR/거리 등 7개 좌표를 받고 Structure는 그 좌표를 모두 0으로 받는다. raw feature 값, label 값, node ID는 정책 입력이 아니다. 정책은 mask와 geometry를 학습하는 제한된 모델이다.

각 seed에서 초기화와 128-update task schedule을 맞췄다. 4개의 ordered Plackett–Luce halo를 샘플링하고 query −NLL/log(C)를 reward로, 다른 rollout의 평균을 baseline으로 쓰는 REINFORCE로 학습했다. 정답 query label은 synthetic train reward와 평가에만 쓴다. 마지막 checkpoint만 사용하며 test NLL로 checkpoint나 rule을 고르지 않았다. 추론에서는 deterministic top-k를 쓴다. 따라서 학습 sampling과 deterministic deployment 사이의 차이도 이 구현의 한계다.

두 정책 × 3 seeds, 총 6개 모델을 학습했다. 모든 GraphPFN parameter가 처음과 끝에 동일한지 hash로 확인했다. 기존 published checkpoint의 사전학습 비용은 새 정책 학습 비용과 맞춘 것이 아니다.

## 주 결과: budget 64

Metric은 graph별 mask·seed 지표를 평균한다. 정책 3개를 prediction ensemble한 결과가 아니다. Random은 3개 sampling seed, PPR/path/full은 deterministic 1회다. NLL은 자연로그, Brier는 class별 제곱오차 합이다.

방법 열의 `full`은 같은 context/query를 유지하면서 전체 graph를 넣은 예산 밖 참조다. 64-node 방법이 아니며 이론적 성능 상한도 아니다. 조건 열의 `full_features`는 feature를 가리지 않은 조건으로, 이 두 의미를 구분한다.

| 조건 | 방법 | Accuracy (%) | NLL ↓ | Brier ↓ |
|---|---|---:|---:|---:|
| matched | random | 45.51 | 1.52351 | 0.75561 |
| matched | ppr | 45.70 | 1.21681 | 0.68233 |
| matched | evidence_paths | 48.24 | 1.05353 | 0.62737 |
| matched | structure_policy | 44.99 | 1.23321 | 0.70088 |
| matched | observation_policy | 47.33 | 1.07193 | 0.64067 |
| matched | full | 51.95 | 0.94011 | 0.58756 |
| cluster_ood | random | 46.94 | 1.55834 | 0.76456 |
| cluster_ood | ppr | 43.55 | 1.35408 | 0.72823 |
| cluster_ood | evidence_paths | 44.92 | 1.07685 | 0.63731 |
| cluster_ood | structure_policy | 44.79 | 1.33246 | 0.73269 |
| cluster_ood | observation_policy | 43.82 | 1.08551 | 0.65827 |
| cluster_ood | full | 45.90 | 0.93248 | 0.60626 |
| cell_ood | random | 47.40 | 1.81049 | 0.79504 |
| cell_ood | ppr | 43.75 | 1.05712 | 0.66135 |
| cell_ood | evidence_paths | 46.88 | 0.96354 | 0.60481 |
| cell_ood | structure_policy | 43.75 | 1.32095 | 0.74687 |
| cell_ood | observation_policy | 49.09 | 1.03419 | 0.62367 |
| cell_ood | full | 48.05 | 0.93150 | 0.60774 |
| full_features | random | 50.52 | 0.82433 | 0.53769 |
| full_features | ppr | 51.17 | 0.80363 | 0.52193 |
| full_features | evidence_paths | 50.78 | 0.82428 | 0.53649 |
| full_features | structure_policy | 51.04 | 0.82901 | 0.54186 |
| full_features | observation_policy | 54.04 | 0.81815 | 0.53157 |
| full_features | full | 52.73 | 0.80827 | 0.52545 |

## Observation − 대조군: paired graph CI

32 graph를 paired bootstrap 5,000회 재표집했다. Mask와 seed를 graph 안에서 평균한 뒤 resample했다. 이 CI는 주어진 3개 policy seed를 고정한 graph 변동이다. Training-seed 모집단 불확실성이나 다중 검정 보정은 포함하지 않는다.

| 조건 | 대조군 | Δ NLL, 95% CI | Δ Accuracy (%p), 95% CI |
|---|---|---:|---:|
| matched | random | -0.45159 [-0.58958, -0.31854] | +1.82 [-2.02, +5.99] |
| matched | ppr | -0.14488 [-0.27591, -0.01535] | +1.63 [-3.52, +6.84] |
| matched | evidence_paths | +0.01839 [-0.06529, +0.10213] | -0.91 [-5.40, +3.52] |
| matched | structure_policy | -0.16128 [-0.27147, -0.05871] | +2.34 [-1.11, +5.86] |
| cluster_ood | random | -0.47283 [-0.60072, -0.34704] | -3.13 [-7.88, +1.76] |
| cluster_ood | ppr | -0.26857 [-0.47820, -0.07895] | +0.26 [-5.73, +6.45] |
| cluster_ood | evidence_paths | +0.00866 [-0.11473, +0.11820] | -1.11 [-4.82, +2.34] |
| cluster_ood | structure_policy | -0.24695 [-0.38936, -0.11246] | -0.98 [-5.14, +3.00] |
| cell_ood | random | -0.77630 [-0.99018, -0.57004] | +1.69 [-4.43, +7.42] |
| cell_ood | ppr | -0.02293 [-0.14997, +0.08666] | +5.34 [+0.13, +10.42] |
| cell_ood | evidence_paths | +0.07065 [-0.01646, +0.16249] | +2.21 [-4.04, +8.20] |
| cell_ood | structure_policy | -0.28675 [-0.44249, -0.14799] | +5.34 [+1.17, +9.38] |
| full_features | random | -0.00618 [-0.02193, +0.00870] | +3.52 [+0.91, +6.12] |
| full_features | ppr | +0.01452 [-0.00297, +0.03277] | +2.86 [-0.91, +6.90] |
| full_features | evidence_paths | -0.00612 [-0.02472, +0.01282] | +3.26 [-1.04, +7.94] |
| full_features | structure_policy | -0.01086 [-0.03032, +0.00575] | +2.99 [+0.39, +5.73] |

사전 고정한 확장 기준은 budget 64의 matched와 cluster OOD에서 Observation이 PPR·Evidence-paths·동일 용량 Structure 정책 모두보다 낮은 NLL CI를 얻는 것이다. CI에 0이 들어가는 것은 동등성이나 비열등성의 증명이 아니다.

## 미학습 예산

| 조건 | Budget | Random NLL | PPR NLL | Paths NLL | Structure NLL | Observation NLL |
|---|---:|---:|---:|---:|---:|---:|
| matched | 32 | 1.47726 | 1.55307 | 1.35465 | 1.50479 | 1.27151 |
| matched | 64 | 1.52351 | 1.21681 | 1.05353 | 1.23321 | 1.07193 |
| matched | 96 | 1.42657 | 0.98201 | 0.94126 | 1.14267 | 0.95988 |
| cluster_ood | 32 | 1.39574 | 1.55884 | 1.45324 | 1.50134 | 1.26040 |
| cluster_ood | 64 | 1.55834 | 1.35408 | 1.07685 | 1.33246 | 1.08551 |
| cluster_ood | 96 | 1.47290 | 1.26201 | 0.94178 | 1.33210 | 0.96199 |
| cell_ood | 32 | 1.56157 | 1.63800 | 1.74724 | 1.77745 | 1.60535 |
| cell_ood | 64 | 1.81049 | 1.05712 | 0.96354 | 1.32095 | 1.03419 |
| cell_ood | 96 | 1.50110 | 0.99280 | 0.92771 | 1.20387 | 0.94825 |
| full_features | 32 | 0.82572 | 0.82221 | 0.82469 | 0.83194 | 0.83580 |
| full_features | 64 | 0.82433 | 0.80363 | 0.82428 | 0.82901 | 0.81815 |
| full_features | 96 | 0.82513 | 0.80603 | 0.82340 | 0.82763 | 0.81378 |

## 사후 기본 확률 대조

학습된 선택기의 상대 개선만으로 충분한지 확인하기 위해, 학습·평가 완료 후 다음 두 확률 기준선을 추가했다. Query 정답은 평가에만 쓰고, 확률 계산에는 selected context labels와 동일한 class 공간만 사용한다. 어떤 mask와 node budget에서도 같은 예측을 한다. 이는 새 학습 arm이나 사전 등록된 주 대조군이 아니며, checkpoint·하이퍼파라미터 선택에 사용하지 않았다.

| 기준선 | NLL ↓ | Brier ↓ |
|---|---:|---:|
| Class별 uniform probability | 0.86849 | 0.56771 |
| Context count + Laplace 1 smoothing | 0.85241 | 0.55736 |

Observation−Laplace-context-prior의 64-node paired graph 차이는 다음과 같다. 양수는 학습 정책이 더 나쁘다는 뜻이다.

| 조건 | Δ NLL, 95% CI | Δ Brier, 95% CI |
|---|---:|---:|
| Matched | +0.21952 [+0.12555, +0.31655] | +0.08331 [+0.03246, +0.13318] |
| Cluster OOD | +0.23310 [+0.14044, +0.32727] | +0.10090 [+0.05211, +0.15015] |
| Cell OOD | +0.18179 [+0.06946, +0.30777] | +0.06631 [+0.00944, +0.12640] |
| Full features | −0.03425 [−0.09375, +0.02188] | −0.02579 [−0.07318, +0.02095] |

이 비교는 확률적 품질에 관한 것이며 accuracy의 같은 결론을 자동으로 뜻하지 않는다. Context가 class별로 capped sampling되어 동점이 많으므로, prior의 lowest-class-ID argmax accuracy는 tie-sensitive 보조값으로 sanity.json에만 기록한다. 확률 자체의 NLL/Brier에는 argmax tie-breaking이 영향을 주지 않는다.

따라서 관측 상태를 이용한 선택이 기존 frozen PFN을 개선했다는 상대 결과와, 실제로 단순 확률 기준선보다 좋은 예측을 제공한다는 주장을 구분한다. 후자는 현재 missing 조건에서 성립하지 않았다. 원인이 backbone의 calibration, context 크기, 선택된 graph 분포, 전처리 중 무엇인지 이 실험으로 분리하지 않았고, 이를 구제하려는 temperature·추가 loss 실험도 하지 않는다.

## 비용과 관측값 전달

Bank 준비 44.80초, 전체 evaluation 1070.34초. Bank 준비에는 global PPR·거리·관측 통계를 만드는 비용이 포함된다. 아래 retrieval은 그 공통 특징 계산 이후의 선택 비용이다. Subgraph PCFI와 GraphPFN forward는 inference 열에 포함되며 파일 저장은 제외한다. 전체 graph를 확인하는 CPU 전처리가 있으므로 node-budget만으로 end-to-end scalability를 주장하지 않는다.

| 정책 | Seed | Updates | Train 초 |
|---|---:|---:|---:|
| structure_policy | 11 | 128 | 66.06 |
| observation_policy | 11 | 128 | 73.28 |
| structure_policy | 12 | 128 | 78.76 |
| observation_policy | 12 | 128 | 81.19 |
| structure_policy | 13 | 128 | 75.98 |
| observation_policy | 13 | 128 | 73.62 |

| Matched budget 64 | Retrieval ms | PCFI+forward ms | Observed rows | All-zero input fraction |
|---|---:|---:|---:|---:|
| random | 0.47 | 141.89 | 3.19 | 0.344 |
| ppr | 0.36 | 138.47 | 3.38 | 0.359 |
| evidence_paths | 13.73 | 142.17 | 9.38 | 0.141 |
| structure_policy | 1.81 | 143.94 | 3.27 | 0.359 |
| observation_policy | 1.76 | 145.06 | 11.39 | 0.125 |

Observed donor 전달률 자체는 label 예측 효용의 증거가 아니다. RMS가 singleton 값을 보존하더라도 PCFI 출력이나 backbone 내부 정규화에서 구별 가능한 피처 정보가 남는다고 보장할 수 없다. All-zero와 mask별 세부값은 summary.json에 보존한다.

## 검증·범위·관련 연구

숨긴 feature를 1e12로 바꾸어도 model-visible 관측값이 동일함을 검사했다. Query 정답 변경이 선택이나 예측을 바꾸지 않는지, context/query budget과 class permutation에 sampler가 의존하지 않는지, PL uniform log-probability와 공통 logit offset 불변성, policy gradient와 완전히 고정된 backbone을 검사했다. Saved prediction에서 metric을 독립 재계산하고 bank·checkpoint·학습 source hash를 확인했다.

PPR/path는 같은 budget과 입력을 쓰는 강한 규칙 대조군이다. 공식 LoGIC나 Wander checkpoint와 직접 비교한 결과는 아니며, 관계 타입·native feature block·실제 heterograph나 새로운 semantic schema는 다루지 않았다. Query 8개짜리 작은 synthetic episode와 mask-only node scorer의 결과를 전체 retrieval 학습의 가능성으로 확대하지 않는다.

- [LoGIC](https://arxiv.org/html/2609.05955v1)은 labeled context와 unlabeled halo 예산을 구분하고 PPR·feature·coverage retrieval을 다룬다. 따라서 budgeted halo나 PPR 자체를 새 기여로 주장하지 않는다.
- [Wander](https://arxiv.org/html/2610.06694v2)는 random-walk 기반 계산과 inference-time context 확장을 다룬다. 이 pilot은 Wander를 구현한 실험이 아니다.

실행 순서: prepare.py → preflight.py → run.py → analyze.py → audit.py. protocol.json과 run_manifest.json은 학습 시작 전 설정과 source hash를 고정한다. audit.py는 학습 후 추가한 기술 점검으로, 학습된 정책의 순열 검사와 checkpoint replay 및 최종 artifact hash 기록만 수행한다. 결과 판정 뒤 추가 loss·더 긴 학습·선택적 checkpoint로 이 구현을 구제하는 실험은 자동으로 추가하지 않는다.

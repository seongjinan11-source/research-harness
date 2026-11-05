# FP05 — 공개 GraphPFN의 missing 대응과 RL 계산 정책

2026-11-05. **이번 3-bit 계산 정책은 No-Go다.** 실제 공개 GraphPFN을 fine-tuning했지만, RL의 추가 정확도 이득이나 관측 피처 활용의 개선을 확인하지 못했다. 수치·입력·checkpoint·통계 감사는 모두 PASS이고, 사전 효능 gate 네 개는 FAIL, 완전 관측 보존 guard는 PASS다.

이 판정은 기존 GraphPFN 내부 세 지점에서 같은 adapter를 한 번 더 실행할지 선택하는 구조와 이번 학습·배포 규칙에 대한 것이다. Missing-feature PFN, RL 전체, heterogeneous graph용 PFN의 가능성을 기각한 결과가 아니다.

## 실제로 학습한 것

공개 GraphPFN1.3 adapter와 LimiX16M을 불러왔다. LimiX·입출력 encoder/decoder는 고정하고, 기존 graph adapter **3,564,288 parameters**를 query CE로 학습했다. Direct와 Group-PG는 별도 **513-parameter 정책**도 학습했다. 따라서 본체 Transformer에 policy-gradient를 적용한 실험으로 부르면 안 된다.

원래 12-block forward를 유지한 채 layer2/5/8에서 같은 residual graph-conv+MLP를 추가 실행하는 세 binary action을 두었다. 가능한 경로는 8개다. 모든 arm은 관측값만의 RMS→PCFI100→동일 clip을 사용한다. 이번 후보는 PCFI 대치법을 새로 학습한 모델이 아니다.

- Fixed: 원래 `000` 경로의 일반 fine-tuning.
- Uniform: 무작위 경로 8개로 학습하고, 주 배포 비교는 8경로 확률 평균. 개발 데이터로 고른 단일 고정 경로도 별도 보존.
- Direct: 8개 경로를 모두 열거해 정책의 기대 CE+0.01×추가연산 비용을 직접 최적화.
- Group-PG: 현재 정책에서 8경로를 뽑고 group-normalized reward로 REINFORCE; adapter는 CE 학습. 단일 on-policy update이므로 GRPO clipping의 효과나 새 RL 알고리즘을 검증한 것이 아니다.

각 arm×3 seed, 마지막 checkpoint, 같은 입력·일정·128 optimizer groups다. 총12 fits/1,536 scientific updates이며 실제 그래프의 weight update는0회다. Group-PG와 Direct의 hidden state는 둘 다 detach했다. Direct가 정확하다는 것은 고정 conditional state에서의 정책 기대 목적 계산을 뜻하며, adapter까지 포함한 전체 coupled objective의 완전 미분이라는 뜻이 아니다. Group normalization도 raw expected reward의 불편 추정량과 같지 않다.

## 사전 primary 결과

새 합성 base graph96개, whole-node missing95/99.5%, **자기 피처가 없는 query**만의 정확도다. 고정된3 training seeds와 두 rate를 graph 안에서 평균했다. Frozen references도 동일한 합성 입력·전처리다.

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| Frozen PFN, NaN 입력 | 47.774 | 0.862407 | 0.574038 |
| Frozen PCFI→PFN | 48.788 | 0.869112 | 0.572587 |
| Fixed, 일반 fine-tuning | 49.290 | 0.845051 | 0.555849 |
| Uniform, 8경로 평균 | 49.475 | 0.845582 | 0.555509 |
| Direct | 48.776 | 0.845467 | 0.556608 |
| Group-PG | 48.793 | 0.845406 | 0.556633 |
| LP | 47.015 | 0.834960 | 0.548308 |
| Context prior | 43.895 | 0.826377 | 0.542449 |

Uniform의 최고 점 추정과 일반 fine-tuning의 작은 개선은 보존한다. 둘의 우월성에 대한 사전 유의성 판정은 하지 않았다. Uniform은 추론8회이므로 단일 경로와 비용이 같지 않다. NLL/Brier가 더 좋은 context prior를 accuracy 성공으로 승격하지 않는다.

| 사전 비교 | 차이 %p | nominal97.5% CI %p | 판정 |
|---|---:|---:|---|
| PG−Fixed | −0.497 | [−1.034,+0.012] | FAIL |
| PG−Direct | +0.017 | [−0.241,+0.275] | FAIL |
| PG observed−none | −1.063 | [−5.567,+4.069] | FAIL |
| PG observed−shuffle | −0.260 | [−2.127,+1.584] | FAIL |
| 완전 관측 natural, PG−Fixed | +0.130 | [−0.825,+1.042] | 보존 guard PASS |

Primary는12strata의 base-graph bootstrap5,000회다. Utility는 source≥2인 disjoint48graphs/58cases에 한정한 다른 estimand다. Full guard는 실제로 모든 query가 관측된 natural48graphs/768queries에 대해 하한>−2%p를 확인한다. 개별 nominal97.5% 구간을 네 효능 비교의 동시95% 보장으로 해석하지 않는다. 음의 utility 점 추정도 유의한 피해의 증명은 아니다.

## 어떤 계산이 실제로 일어났는가

Direct와 PG는 **각 seed의 final1,056개와 real45개 모든 episode에서 greedy `000`**을 선택했다. 같은 checkpoint를 강제로 `000` 재생한 primary accuracy와 차이는 두 방법 모두 정확히0이다. 이번 배포에서는 context나 missing 상황에 따라 추가 연산 경로를 바꾸는 동작이 나타나지 않았다.

이것을 정책의 확률분포 전체가 붕괴했다거나 모든 branch가 무가치하다는 결론으로 바꾸면 안 된다. Greedy 배포와 학습 중 stochastic policy는 다르다. 실제 개발 NLL로 고른 PG 고정 경로는 세 seed 모두 `110`이고, 비용항0.01을 포함한 개발 평균 목적의 최선 경로도 전부 `000`은 아니다. 따라서 비용항만으로 설명하거나, PG의 sampling variance 하나만 원인으로 단정할 수 없다. 별도의 정확 기대 목적 Direct도 같은 배포 결정을 보였다.

PG의 feature-utility DiD는 none 대비−0.391%p CI[−1.845,+0.977], shuffle 대비−0.456%p CI[−1.519,+0.543]다. 기존 fine-tuning보다 관측 정보의 가치를 더 회수했다는 근거도 없다. 이런 결과에서 checkpoint 간 차이를 policy routing의 효과로 부르지 않는다.

## 실제 graph 개발 패널

Cora/CiteSeer 전체와 PubMed2,000-node induced graph, 입력 repetition3개와 학습 seed3개의 기술 평균이다. 원래 관측 행만 고정 RP16→RMS→PCFI로 처리했다. Native 전체 피처를 그대로 넣은 공개 점수와 동일한 입력 조건이 아니다. 세 graph는 과거에도 본 개발 데이터이므로 graph population CI나 미지 실제 데이터 일반화 주장을 하지 않는다.

실제 context는 **클래스당4개**다: Cora7classes/28labels, CiteSeer6classes/24labels, PubMed3classes/12labels. 과거 클래스당20개 실험과 같은 표로 비교하면 안 된다. 합성 추가 학습의 class 수는 최대4이므로 Cora/CiteSeer class 수 변화도 현재 전이 조건에 포함된다. 이것이 부진의 단독 원인이라고 분리 검증한 것은 아니다.

| Graph | Missing % | Frozen PCFI→PFN | Fixed | PG | LP |
|---|---:|---:|---:|---:|---:|
| Cora | 95 | 46.439 | 45.891 | 45.459 | 59.056 |
| Cora | 99 | 50.768 | 51.049 | 50.653 | 59.056 |
| Cora | 99.5 | 53.370 | 52.865 | 52.748 | 59.056 |
| CiteSeer | 95 | 28.547 | 29.354 | 28.334 | 38.410 |
| CiteSeer | 99 | 37.943 | 37.106 | 35.779 | 38.410 |
| CiteSeer | 99.5 | 33.576 | 33.427 | 32.275 | 38.410 |
| PubMed | 95 | 65.032 | 62.323 | 61.671 | 73.690 |
| PubMed | 99 | 69.375 | 67.193 | 66.767 | 73.690 |
| PubMed | 99.5 | 71.242 | 69.502 | 69.904 | 73.690 |

이 패널에서도 PG가 frozen PCFI→PFN이나 LP를 안정적으로 넘었다고 할 수 없다. Graph별 순위나 missing-rate 비단조성을 본 뒤 새 보정값을 고르는 검증으로 사용하지 않는다. Historical native PCFI/FISF references는 `summary.json`에 입력·예산 차이와 함께 별도 보존했다.

다만 **실제 관측 피처의 효용까지 사라진 것은 아니다.** 다음은95% real 개발 조건의 observed−none / observed−shuffle 정확도 차이(%p)다.

| Graph | Frozen PCFI→PFN | Fixed | PG |
|---|---:|---:|---:|
| Cora | +19.923 / +2.880 | +19.756 / +2.554 | +21.134 / +3.185 |
| CiteSeer | +11.597 / −0.038 | +10.736 / +0.757 | +11.180 / +1.385 |
| PubMed | +30.385 / +2.472 | +22.595 / +1.784 | +24.768 / +2.071 |

PG에서도 관측값 제거·노드 대응 셔플에 반응하는 점 추정은 존재한다. 이 비교는 소수 재사용 real graph의 기술 결과이며 synthetic utility gate와 다른 estimand다. `000` replay가 동일하므로 이를 적응적 routing의 효과로 해석할 수 없고, frozen/일반 fine-tuning에도 있는 PCFI→PFN 정보 활용과 학습된 adapter의 반응을 구분해야 한다. RL의 추가 가치가 없었다는 결과를 “remaining features 자체가 쓸모없다”로 바꾸지 않는다.

## 확인된 것과 남은 것

확인된 것은 실제 public model의 adapter 학습과 정책 경로·gradient 경계가 구현되고, 정해진 비교가 완료됐다는 점이다. 세 감사에서30,828개 평가 예측과2,688개 개발 calibration branches, checkpoint 내부 상태, 일치하는 일정·정답·context·query, 저장 지표를 확인했다. 독립 통계 재계산의 최대 차이는4.44e−16이다. 발견된 numerical/leakage 결함이나 필요한 scientific rerun은 없다.

관측 피처를 직접 연결하는 새로운 추론 연산, target graph에서의 RL, 본체 Transformer policy-gradient, heterogeneous schema 전이는 이번 실험의 대상이 아니다. 현재 반복 adapter 연산·상태 요약·예산·보상·greedy 배포 중 무엇이 실패의 주원인인지는 분리하지 못했다. 단순한 학습량 부족이나 비용항 탓을 확정 원인으로 기록하지 않는다.

원래 official prior의 class-coverage/feature sanity 필터는 유지됐고, 전체 내부 거절 횟수는 API에서 알 수 없다. Source0/1을 추가 제거하지 않았다. Final node99.5에서는 source0/1이76/96graphs이며, utility 안의 입력 불변 shuffle8cases도 그대로 유지했다. Requested/actual 크기도 구별한다: train actualN54..192/F7..12, final matchedN112..192/F8..12, heldoutN202..256/F15..16이다.

Train/dev/real 준비 시점의 이전 utility-file bytes를 보존하지 못한 제한은 `PRELOCK_SOURCE_DELTA.json`에 남겼다. 다른 준비 dependency와 모든 최종 locked source·bank는 일치하며, 실제 저장 입력을 별도 감사했다. 이전 utility의 모든 변경을 byte 단위로 복원했다고 주장하지 않는다.

## 비용과 기록

실제 scientific training forward/backward는 각각9,600회, calibration forward2,688회, 평가 forward51,747회다. Training 약46.86분, calibration3.28분, trained evaluation61.51분을 기록했다. Checks와 개발 smoke의 버린 optimizer steps/groups10개는 scientific1,536에 포함하지 않는다. 데이터 준비·PCFI와 reference/LP 시간의 기록 범위를 별도 cost audit에 남겼다. Hash/CPU 감사와 실행 overhead를 포함하는 workload 기록이므로 반복 latency benchmark나 작은 속도 우위의 증거가 아니다.

현재 branch는 `research/fp05-missing-compute-policy`다. Large bank/checkpoint/예측은 경로와 hash로 보존하고 Git에는 코드·계획·리뷰·compact results를 남긴다. 같은 final bank는 이후 개발 자료다. 새 hypothesis는 새 cycle·비교 명세와 별도 bank에서 검증해야 한다.

이 갈래에서 보존할 결론은 **“학습한 내부 계산 선택이 이번 missing 문제를 해결하지 못했고, greedy 배포는 원래 경로와 같아졌다”**다. 새로운 방법의 필요성을 입증하려면 먼저 실제로 유용한 대안 계산과, 그것을 배포에서 활용하는 학습 원리를 보여줘야 한다. 이 조건이 검증되기 전 반복 횟수나 loss를 늘리는 방식으로 현재 후보를 확대하지 않는다.

참고: [고정 설계](PLAN_v2.md), [원 지표](results/summary.json), [독립 통계](results/statistics_audit.json), [checkpoint·배포 감사](results/deployment_audit.json), [문헌 경계](PRIOR_ART_VERIFIED.md). 세 결과 리뷰와 root 응답은 별도 파일로 보존한다.

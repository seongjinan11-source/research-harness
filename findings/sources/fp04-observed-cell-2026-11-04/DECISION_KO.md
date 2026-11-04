# FP04 결과 판정: 직접 관측-cell PFN

2026-11-04. **실험과 산출물 감사는 완료했지만, 사전에 정한 네 성능 조건은 모두 통과하지 못했다. 현재 구성과 예산의 확대는 중단한다.** 관측-cell PFN 전체나 모든 sparse Transformer의 실패를 주장하는 결과는 아니다.

## 실제로 실행한 것

공개 GraphPFN의 입력만 바꾸는 대신, 관측값 인코더·그래프·공유 feature-column·분류 head의 **19,748개 파라미터 전체를 무작위 초기화에서 학습**했다. Width32, signed value basis rank4, 공유 message-passing 4회인 작은 PFN 방식의 predictor다. 자유로운 token 간 query/key attention을 가진 Transformer는 아니다.

- Observed: 실제 관측 cell만 계산에 포함.
- Mask: 모든 위치를 포함하고 결측값은 0, 실제 mask를 입력.
- PCFI: 실제 100회/alpha .9 PCFI로 채운 입력과 실제 mask를 사용.

세 arm은 seed별 초기화·task schedule·파라미터 수·update 수가 같다. 모든 arm을 각각 3개 seed로 학습했다. **9 fits, 6,912 updates, 27,648 task presentations**이며, CE만 사용했다. 실제 평가 그래프의 가중치 업데이트는 0회다. 동일한 노출 수가 동일 FLOPs를 뜻하지는 않는다.

학습 bank는 384 base graphs의 2,688 episodes다. 합성 최종 평가는 별도 seed로 만든 **96개 base graphs**이며, 두 feature/graph family와 두 shape, K=2/3/4를 포함한다. Query 16개는 모든 학습·평가에서 자기 피처가 없다. `full`은 **query를 제외한 노드의 피처 관측**을 뜻한다.

## 사전 primary와 utility 판정

Primary는 95%/99.5% whole-node missing의 accuracy다. 각 graph에서 고정 3개 학습 seed와 두 결측률을 먼저 평균했다.

| Arm | Accuracy % | Balanced accuracy % | NLL | Brier |
|---|---:|---:|---:|---:|
| Observed | 48.253 | 46.999 | 1.02142 | 0.61350 |
| Mask | 47.830 | 46.542 | 1.02359 | 0.61510 |
| PCFI | 47.830 | 46.549 | 1.02359 | 0.61511 |

| 고정 비교 | Accuracy 차이 %p | 사전 97.5% CI %p | Base graphs | Gate |
|---|---:|---:|---:|---|
| Observed − Mask | +0.423 | [−0.293, +1.134] | 96 | FAIL |
| Observed − PCFI | +0.423 | [−0.271, +1.118] | 96 | FAIL |
| Observed − 피처 제거 | +0.043 | [−0.629, +0.629] | 48 | FAIL |
| Observed − 관측값 셔플 | −0.326 | [−0.825, +0.087] | 48 | FAIL |

CI는 12개 factorial strata 안에서 base graph를 재표집한 paired bootstrap 5,000회다. 각 비교의 명목 97.5% 구간이며, 네 비교의 동시 95% 구간이라고 부르지 않는다. CI가 0을 포함한다는 결과는 동등성을 증명하지 않는다.

Feature utility는 context와 관측 노드가 분리되고, 관측 source가 2개 이상인 rate cases를 사용한 **별도 조건부 추정량**이다. 예정된 48 disjoint graphs가 모두 남았고 68 graph-rate cases였다. Primary에서는 source0/1을 제외하지 않았다. 특히 99.5%에서 source0/1/2+ graph 수는 22/25/49였다.

작은 양의 arm 차이는 보존하지만, 더 중요한 **실제 값의 예측 효용**은 입증하지 못했다. Field family에서는 observed의 NLL도 두 대조보다 약간 나빴다. 더 좋은 subgroup나 secondary metric으로 고정 gate를 바꾸지 않는다.

## 실제 그래프에서도 확인한 것

아래는 95% whole-node missing, 동일 context/query/mask에서의 accuracy다. 새 모델은 3 training seeds × 3 context/mask 반복 평균이며, 고정 공개 모델과 LP는 같은 3개 context 반복이다.

| Graph | Observed | Mask | 자체 PCFI arm | 피처 전부 제거한 Observed | PCFI + 공개 PFN | Restart LP |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 52.876 | 47.819 | 47.924 | 52.923 | 54.796 | 59.056 |
| CiteSeer | 29.977 | 25.840 | 25.828 | 30.119 | 34.078 | 38.410 |
| PubMed | 64.002 | 55.331 | 55.097 | 64.264 | 55.888 | 73.690 |

Observed는 자체 대조군보다 높지만 **같은 checkpoint에서 피처를 없애도 그 성능이 유지된다.** 95%의 관측값 셔플도 큰 하락을 만들지 않았다. 따라서 PubMed의 PCFI+공개 PFN 대비 +8.11%p 같은 점 차이를 관측 피처 추론의 성공으로 설명하지 않는다. Restart LP는 세 그래프 모두에서 더 높다. 99%/99.5%, FISF+공개 PFN, harmonic LP와 전체 NLL/Brier는 원보고서에 함께 보존했다.

Cora/CiteSeer는 전체 graph, PubMed는 이전의 2,000-node induced graph다. 셋 모두 이미 연구 중에 본 **개발 데이터**다. 이 결과는 미지 실제 graph population에 대한 독립 검증 CI가 아니다. 물리적 관측값·graph·context·query·정답 identity는 검증했지만, 새 모델은 native 전체 column/RMS, 공개 모델은 원래 전처리/PCA/ensemble을 사용한다. 외부 모델 비교와 동일 예산 세 arm 대조를 같은 인과 효과로 취급하지 않는다.

## 확인된 사실과 아직 모르는 원인

**확인됨:** 관측 cell 경로가 실제 계산 그래프에 존재한다. 독립 loop/NumPy oracle, CPU/GPU, sparse/dense forward·gradient, chunk·node/class/column permutation, hidden-value poison, self-label leakage, 두 component를 잇는 column 경로 검사를 통과했다. 9 fits와 8,181개 새 예측의 hash·schedule·checkpoint·receipt·metric 재계산도 일치했다. 별도 외부 기준선 225 records도 확인했다.

**확인됨:** 피처를 지운 real observed arm에서는 message passing이 4회에 제한된다. 균형 context에서 4-hop보다 먼 query는 column 입력도 없을 때 수학적으로 클래스 대칭이다. 리뷰의 저장 예측 분석에서는 CiteSeer 95%의 record별 top-2 margin 중앙값을 9개 record에서 평균하면 약 6e−8이었고, margin<1e−5 비율의 record 평균은 65.64%였다. 아주 작은 확률 변화로 argmax가 바뀔 수 있어, 이를 유의미한 feature evidence로 해석하면 안 된다. 이는 표현 범위와 FP32 동률 해석의 제한이며, 현재 저장 정확도를 다른 tie 규칙으로 바꾸어 gate를 구제하지 않는다. 이 4-hop 논증을 A=1인 Mask/PCFI arm의 column 경로에 확대하지 않는다.

**원인 미확정:** rank4 평균 압축, 짧은 graph 전달, graph-only shortcut, 합성 prior와 실제 데이터 차이, 최적화 예산 중 무엇이 실패를 얼마나 설명하는지는 분리하지 못했다. 올바른 forward·gradient가 유용한 feature 학습을 보증하지 않으며, 작동 경로의 존재가 곧 성능 기여는 아니다.

**별도 학습 가능성 미검증:** query가 피처를 가진 쉬운 조건에서 같은 encoder가 feature-label 법칙을 학습하는지 검사하지 않았다. `full` 및 완전 M=1 수치 parity 검사를 그런 학습 검증으로 대체할 수 없다. 이 훈련 support가 실패 원인이라는 주장도 현재는 가설이다. Train episode의 source0/1/2+ 비율은 19.42%/4.87%/75.71%여서, source0/1을 전혀 학습하지 않았다고 설명하는 것도 틀리다.

## 비용·보존·종료 범위

9 fits의 기록된 학습 시간 합은 448.41초, 평가 phase 합은 282.18초다. Bank 준비·PCFI 전처리·개발 smoke 시간은 별도로 기록했다. 이것은 planning/import까지 포함한 전체 사용자 대기시간이 아니다. 학습 peak는 약 475.58 MiB였으며, native feature 수를 줄이지 않은 평가를 실행했다. 단일 cold smoke 시간으로 강한 속도 우위를 주장하지 않는다.

PCFI raw cache는 그대로 보존했다. 모델 입력에서는 모든 arm의 값에 공통 ±10 clipping을 적용했으며, native PCFI의 최대 clipping 비율은 약 0.34%였다. 이것만으로 PCFI 실패의 원인을 정하지 않는다.

현재 닫는 것은 **이 19k/rank4/4-round 구조, 두 합성 family, 768-update 예산에서 missing-feature 분류 성과를 확보했다는 가설**이다. 계산 경로·native 전체 피처 처리·수치 검증·작은 arm 점 차이는 보존한다. 관측 subset 입력, PFN 전체, 자유로운 sparse attention, 기존 CE Factor의 과거 성과까지 기각하지 않는다.

다음 질문을 하나 남긴다면 **관측 query에서의 feature 학습 가능성 → query 관측 유무가 섞인 훈련 → feature-free query 전이**를 분리할 수 있는가다. 이는 미실행 제안이며 현 실패의 확정 설명이나 성공한 메소드가 아니다. 이번 결과를 본 뒤 같은 bank에서 rank/round/loss/temperature를 조정하지 않는다.

## 기록 위치

- 고정 계획: `PLAN_v2.md`, `protocol.json`, `results/lock.json`.
- 원결과·감사: `results/REPORT_KO.md`, `summary.json`, `paired.json`, `budget.json`, `audit.json`, `artifact_manifest.json`.
- 세 관점의 결과 리뷰: `RESULT_METHOD.md`, `RESULT_INFERENCE.md`, `RESULT_EXPERIMENTS.md`; 답변은 `RESPONSE_RESULTS.md`.
- 가벼운 로컬 branch: `research/fp04-observed-cell`.
- 축적 기록 branch: `research/fp04-observed-cell-record`.

현재 이 후보의 추가 학습이나 GPU 작업은 실행 중이지 않다.

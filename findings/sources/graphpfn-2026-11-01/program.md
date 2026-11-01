# Compiler–Executor GraphPFN: 완전 관측 pilot 결과

2026-11-01. 판정: **현재 compiler의 확장 보류 / 이 pilot 구현은 No-Go.**

Compiler는 같은 예산의 작은 Plain보다 정확도가 높았다. 그러나 미학습 조합에서 NLL 개선은 확인되지 않았고, 관계 연산 특징 + TabPFN과 캐싱된 ProgramGrid에 품질이 크게 뒤졌다. ProgramGrid는 측정 지연시간도 더 짧았다. 따라서 결측·대규모 heterograph로 곧바로 확장할 근거는 확보하지 못했다.

이는 이 작은 compiler 구현의 판정이다. 계산 구성 방식 일반화가 불가능하다는 결론도, 실제 missing-feature GraphPFN에서 실패했다는 결과도 아니다.

## 고정한 실험 범위

- Target 64개, helper 48개, relation 2종, helper Boolean feature 3개. 모든 edge와 feature를 관측하며 label noise는 없다.
- 고정 Boolean executor가 EXISTS, AND, OR, XOR, NOT을 실행한다. 같은 helper에서 predicate를 결합하는 식과 서로 다른 helper도 허용하는 식을 구별한다.
- 문법은 222개 프로그램이다. 186개는 학습 label 생성용, 36개는 미학습 조합이다. 제외한 조합은 다른 relation의 EXISTS(feature)와 EXISTS(AND(feature,feature))를 XOR로 묶은 식 및 그 전체 complement다. 각 primitive는 학습 중 등장한다.
- 전체 222개 후보와 그 trie는 compiler 학습 때부터 사용한다. **Heldout은 label을 생성하는 조합의 제외이며, 후보 구조를 처음 보는 평가는 아니다.** 무한 프로그램 합성이나 새로운 semantic schema를 주장하지 않는다.
- Train/validation/test graph 수는 512/32/64, episode 수는 3072/96/384다. 동일 원본 graph의 파생 task는 한 split에 묶었다. Test의 같은 64개 graph에서 본 조합과 미학습 조합을 각각 평가했다.
- Context는 클래스당 수가 아니라 총 4/8/16개다. Label과 무관하게 뽑은 노드 순열의 중첩 prefix를 모든 방법에 공유했다. Train은 graph당 규칙을 두 번 복원추출하므로 4개 graph에서는 두 추출 규칙이 같았다.
- Graph별 후보의 전체 target 양성 비율이 15~85%인 규칙만 label 생성에 사용한다. 이 eligibility 조건은 알려진 generator prior다. 각 가설을 관측 입력에서 실행해 계산하며, 실제 query 정답을 가설 선택기에 제공하지 않는다.

## 모델과 감독 신호

| 모델 | 파라미터 | seed당 update | 3 seed 학습 시간 합계 |
|---|---:|---:|---:|
| Plain | 41,426 | 768 | 30.65초 |
| Compiler | 41,868 | 768 | 92.45초 |

Seed는 11/12/13, batch 16이다. 같은 seed에서 shared encoder 초기값과 task 순서가 일치한다. 최종 768번째 checkpoint를 고정해 평가했으며 validation이나 heldout 결과로 checkpoint를 선택하지 않았다. **Update 수는 같지만 wall-clock은 같지 않다.** 위 시간은 update 계산만 포함하고 데이터 준비·평가·저장을 제외한다.

Plain은 relation마다 helper의 8개 joint Boolean pattern 빈도와 degree를 받는다. 이는 이 문법의 binding 정보를 보존하는 18차원 통계다. 단순 marginal 평균으로 같은 helper 조건을 제거한 기준선이 아니다. 2-layer Transformer가 context label을 사용한다. 다만 이 작은 예산의 Plain을 최적화된 GraphPFN 전체와 동일시하지 않는다.

Compiler는 같은 입력에서 class별 context 요약과 graph 요약을 만들고, 공유 GRU의 prefix-token 점수로 유효 프로그램의 weight를 계산한다. 고정 실행값을 그 weight로 합산한다. 개별 program ID embedding, 정답 program 감독, posterior auxiliary, 학습 가능한 실행 의미는 사용하지 않는다. 학습 목적은 query predictive CE뿐이다. 이 weight가 정확한 Bayesian program posterior라는 보장은 없다.

TabPFN은 binding·집계 순서가 포함된 전체 222개 프로그램 실행값을 특징으로 받는다. 로컬 v2.6 checkpoint, 8 estimators, seed 0이며 test graph에서 gradient update는 없다. 384개 중 24개 단일 클래스 context는 사전에 정한 Laplace empirical-frequency fallback을 사용했다. 이 사전학습 모델은 새로 학습한 두 모델과 pretraining 비용을 맞춘 대조군이 아니다.

## 주 결과

세 context 크기의 task별 지표를 동일 가중 평균했다. 학습 모델은 3 seed의 지표를 평균했으며 예측 ensemble은 아니다. Brier는 두 클래스 오차 제곱합이므로 binary scalar Brier의 2배다. NLL은 자연로그이고 평가 확률은 1e-8로 clipping했다.

| 조건 | 방법 | Accuracy (%) | NLL ↓ | Brier ↓ |
|---|---|---:|---:|---:|
| 본 조합 / 새 graph | Plain Transformer | 59.65 | 0.66131 | 0.46886 |
| 본 조합 / 새 graph | Compiler + 고정 Executor | 67.54 | 0.58764 | 0.40633 |
| 본 조합 / 새 graph | 관계 연산 특징 + TabPFN | 86.45 | 0.29850 | 0.19079 |
| 본 조합 / 새 graph | ProgramGrid / 전체 문법 | 88.76 | 0.21571 | 0.14636 |
| 미학습 조합 / 새 graph | Plain Transformer | 56.15 | 0.68425 | 0.49098 |
| 미학습 조합 / 새 graph | Compiler + 고정 Executor | 61.59 | 0.68349 | 0.47662 |
| 미학습 조합 / 새 graph | 관계 연산 특징 + TabPFN | 85.18 | 0.33352 | 0.21581 |
| 미학습 조합 / 새 graph | ProgramGrid / 전체 문법 | 87.73 | 0.23073 | 0.15655 |

Compiler−Plain 정확도는 본 조합 +7.89%p, 미학습 조합 +5.44%p다. 미학습 조합에서는 accuracy 이득이 있지만 NLL은 거의 같다. 따라서 “미학습 조합에서 모든 이득이 사라졌다”는 해석도 맞지 않는다. 다만 더 강한 두 대조군과의 큰 격차를 해결하지 못한다.

## Paired graph 신뢰구간

각 graph 안에서 context 크기와 3 training seed의 지표를 먼저 평균하고, 64개 graph를 5,000회 paired bootstrap했다. 이는 해당 3 seed를 고정한 graph 변동의 CI이며 training-seed 모집단 불확실성까지 포괄하지 않는다.

| 조건 | 비교 | Δ Accuracy (%p), 95% CI | Δ NLL, 95% CI |
|---|---|---:|---:|
| 본 조합 | Compiler − Plain Transformer | +7.89 [+4.64, +11.11] | -0.07367 [-0.10160, -0.04624] |
| 본 조합 | Compiler − 관계 연산 특징 + TabPFN | -18.92 [-21.48, -16.37] | +0.28914 [+0.25276, +0.32134] |
| 본 조합 | Compiler − ProgramGrid / 전체 문법 | -21.23 [-24.01, -18.52] | +0.37192 [+0.34176, +0.40103] |
| 미학습 조합 | Compiler − Plain Transformer | +5.44 [+2.14, +8.83] | -0.00076 [-0.03029, +0.02882] |
| 미학습 조합 | Compiler − 관계 연산 특징 + TabPFN | -23.59 [-26.34, -20.74] | +0.34997 [+0.31928, +0.37986] |
| 미학습 조합 | Compiler − ProgramGrid / 전체 문법 | -26.14 [-29.11, -23.17] | +0.45276 [+0.42026, +0.48402] |

## Context 증가와 조합 일반화

| 미학습 조합 | Plain Acc / NLL | Compiler Acc / NLL | TabPFN Acc / NLL | Grid Acc / NLL |
|---|---:|---:|---:|---:|
| 총 context 4개 | 55.12% / 0.6874 | 61.65% / 0.6915 | 72.97% / 0.5558 | 77.21% / 0.4399 |
| 총 context 8개 | 55.01% / 0.6910 | 61.55% / 0.6976 | 86.41% / 0.3376 | 88.81% / 0.2015 |
| 총 context 16개 | 58.32% / 0.6743 | 61.57% / 0.6613 | 96.16% / 0.1072 | 97.17% / 0.0509 |

Compiler의 미학습 조합 정확도는 context가 늘어도 약 61.6%에 머물렀다. 반면 전체 문법 Grid와 TabPFN은 context 16개에서 각각 97.17%, 96.16%다. 따라서 이 조건에는 관측 가능한 예측 정보가 남아 있다. 현재 학습된 compiler가 그 정보를 충분히 회수하지 못한 것으로 해석할 수 있다. 원인을 새로운 loss나 추가 head 하나로 단정할 자료는 없다.

## ProgramGrid prior와 ambiguity

전체 문법 Grid는 eligibility를 통과한 222개 후보에 uniform prior를 두고 context와 모순되는 후보를 제거한다. 공통 하위식을 캐싱하며 label noise는 0이다. 학습 문법 Grid는 186개만 사용한다. Case-informed oracle은 평가 case를 알고 본 조합에는 186개, 미학습 조합에는 36개 prior를 적용하는 privileged reference다. 두 prior를 context 결과를 보고 선택한 배포 모델로 제시하지 않는다.

| 방법 | 본 조합 Acc / NLL | 미학습 조합 Acc / NLL |
|---|---:|---:|
| ProgramGrid / 학습 문법 | 89.08% / 0.20269 | 71.89% / 1.04892 |
| ProgramGrid / 전체 문법 | 88.76% / 0.21571 | 87.73% / 0.23073 |
| Case-informed oracle | 89.08% / 0.20269 | 91.95% / 0.13570 |

학습 문법 Grid는 미학습 조합의 192 episode 중 51개에서 context와 맞는 후보가 없어 0.5로 fallback했다. 잘못된 support를 가진 prior의 confidence 문제와 후보 전체를 사용하는 계산의 장점을 구별해야 한다. 전체 문법 Grid의 우위를 neural model이 exact Bayes를 이겨야 한다는 요구로 해석하지 않는다.

다음은 실제 정답 program을 지정하지 않고 전체 문법 Grid를 기준으로 측정한 ambiguity다. 여러 후보가 남더라도 특정 query의 답은 모두 같을 수 있으므로 후보 수와 query ambiguity를 구분한다.

| 조건 | Context 총수 | 호환 후보 수 중앙값 | 모호한 query 비율 | Compiler의 Grid predictive KL ↓ |
|---|---:|---:|---:|---:|
| 본 조합 | 4 | 13 | 77.29% | 0.20246 |
| 본 조합 | 8 | 3.5 | 29.52% | 0.41367 |
| 본 조합 | 16 | 1.5 | 6.41% | 0.51749 |
| 미학습 | 4 | 13.5 | 78.57% | 0.20945 |
| 미학습 | 8 | 4 | 33.31% | 0.44577 |
| 미학습 | 16 | 1 | 8.53% | 0.59664 |

프로그램을 한 개로 강제 선택하지는 않았지만, 혼합 출력 형식만으로 적절한 불확실성 추정이 확보되지는 않았다. Case-informed oracle 대비 predictive KL과 program-weight 진단은 summary.json에 별도로 기록했다. 그 KL은 case 정보가 추가된 reference와의 거리이며 그대로 배포 calibration 지표가 아니다.

## 측정 비용

| 방법 | 입력 특징 구성 포함 graph당 지연시간 |
|---|---:|
| Plain | 3.181 ms |
| Compiler | 13.118 ms |
| 캐싱된 전체 문법 ProgramGrid | 1.119 ms |
| 특징 + TabPFN, 관측된 단일 실행 시간 | 1598.2 ms |

RTX 3090 / CPU·torch thread 4. 동일한 18개 test episode에서 Plain·Compiler·Grid는 1 warm-up 뒤 10회씩 측정했고, episode별 중앙값의 중앙값을 보고했다. CPU의 native adjacency·feature가 준비된 시점부터 필요한 표현 구성, neural host/device transfer와 예측까지 포함한다. 디스크 I/O·weight loading·합성 graph 생성·pretraining은 제외했다.

TabPFN은 동일 episode의 기존 평가 중 한 번 측정한 fit/predict 시간에 별도 측정한 특징 구성을 더했다. 반복 측정이 아니며 해당 실행에는 라이브러리 telemetry 연결 실패 로그도 있었으므로 본질적인 속도 비율 주장에 사용하지 않는다. 완료된 384개 예측에는 오류가 없었다. 향후 실행용 telemetry opt-out만 추가했고, 실행 당시 source snapshot과 수정 내역은 tabpfn_source_provenance.json에 보존했다.

확인된 범위에서 compiler는 ProgramGrid보다 약 11.7배 느리고 예측 품질도 낮다. 현재는 전체 문법 실행값을 compiler도 모두 계산하므로, 프로그램 탐색 비용을 amortize해서 줄였다는 성과는 없다. 대형 graph·큰 문법·메모리 scaling은 측정하지 않았다.

## 검증과 해석의 한계

- 가능한 두 relation의 Boolean witness-pattern 집합 65,536개를 전부 확인했다. 문법 222개는 전역 의미가 모두 다르고 train/heldout의 전역 semantic overlap은 0이었다.
- 다만 test composition graph 64개 중 5개에서는 해당 graph의 실행 벡터가 우연히 seen 규칙과 같았다. 사후 제거하지 않았다.
- Node permutation equivariance는 preflight에서 오차 6e-8 이하였다. 반면 class·relation·feature 재명명에 대한 정확한 대칭성은 없다. Seed 11의 16개 episode에서 compiler의 평균 절대 예측 변화는 각각 0.0582/0.0484/0.0291이었다. 이를 schema-invariant 모델이라고 부르지 않는다.
- Forward에는 context label만 전달한다. 후보 실행 table은 관측 graph/feature에서 모든 가설을 실행한 값이며 generating-rule ID나 query label을 선택기에 넣지 않는다. Query label은 학습 loss와 평가에만 사용한다.
- 파라미터 차이는 약 1.1%이며 초기 shared encoder와 task schedule hash를 확인했다. Prediction·checkpoint·source hash와 확률 유효성을 다시 검사했다. 이전 executor fixture와 종료한 BP/EAJ/R1 구현은 수정하지 않았다.
- 이것은 두 relation·세 predicate의 fully observed synthetic kill test다. Feature missing, edge observation missing, hidden-fact identity의 확률적 의존성, 실제 ACM/IMDB/FREEBASE, unseen semantic schema는 평가하지 않았다.

## 최종 판정

고정 실행기와 context-conditioned 프로그램 혼합을 작은 모델로 학습하는 것은 가능했다. 동일 예산 Plain 대비 정확도 개선도 재현됐다. 그러나 요구했던 미학습 조합의 확률적 품질 또는 강한 기준선 대비 품질–비용 이점은 확인되지 않았다.

**현재 compiler를 키우거나 missing 실험으로 확장하지 않는다.** ProgramGrid, binding을 보존한 특징 + TabPFN, 문법 semantic audit, paired task bank와 원시 예측을 재사용 가능한 비교 도구로 남긴다. 이 결과만으로 새로운 메소드의 novelty나 범용적 유용성을 주장하지 않는다.

## 저장 자료

- protocol.json: 사전에 고정한 설정
- models.py / dsl.py: 학습 모델과 고정 실행기
- results/summary.json: 지표, paired CI, seed별 결과, prediction hash
- results/task_metrics.csv: episode별 지표
- results/frozen_diagnostics.json: ambiguity와 무결성 검사
- results/semantic_audit.json: 의미 중복·대칭성 검사
- results/training.json / latency.json / tabpfn_metadata.json: 학습·비용·가중치 출처
- results/manifest.json: pilot 파일의 SHA-256 목록

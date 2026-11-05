# FP05 결과 심사 — method / prior art

검토일: 2026-11-05. **판정: 감사 PASS인 부정 결과이며, 사전 지정한 missing-feature 효능 주장은 지지되지 않는다.** 추가 경로를 선택하는 greedy 배포의 이득은 관측 범위에서 정확히 0이다. 이를 stochastic policy의 확률분포 붕괴, adapter 미학습, 또는 RL/PFN 전체의 불가능성으로 확대하면 안 된다.

본 reviewer는 사전 method review, `rl_checks.py`의 toy oracle 및 실제 모델 검사 코드 작성, 선행 논문 전문 검토에 참여했다. 따라서 이 문서는 **독립된 method 관점의 결과 검토이며 외부 blind review가 아니다.** 이번 검토에서는 저장된 JSON·계획·선행 검토문을 읽고 metadata와 수치를 확인했다. 새 forward, GPU 작업, 학습, checkpoint 변경 또는 source/bank 수정은 하지 않았다.

## 근거와 신뢰 범위

읽은 주 근거는 `PLAN_v2.md`, `results/summary.json`, `REPORT_KO.md`, `audit.json`, `statistics_audit.json`, `deployment_audit.json`, `budget.json`이다. 추가로 direct/group_pg의 세 seed에 대한 final/real evaluation metadata 및 update128 직전 progress 기록을 읽었다. 다른 결과 reviewer의 문서는 보지 않았다.

Main audit는 30,828 prediction records, paired input identity, matched schedules, target updates 없음과 source lock `18b27b3d9a4eb3c8fc321824bf32407ae8f888d27c4e2de4a10b66eba325c575`를 확인했다. 통계 audit는 production 분석 함수를 부르지 않고 저장 확률에서 지표를 재계산했으며 최대 저장 지표 오차는 약 4.44e−16이다. Deployment audit도 PASS이며 frozen parameter digest와 checkpoint association을 확인했다. 이로써 **계획된 계산과 저장 결과에 근거해 실패를 해석할 근거**가 있다. 감사 PASS 자체는 효능·수렴·인과적 원인을 보증하지 않는다. 본 reviewer가 전체 NPZ/훈련 과정을 별도로 재현했다는 뜻도 아니다.

## 사전 효능 기준

Primary는 96 base graphs의 적격 missing-query rate와 세 train seeds를 graph 내에서 평균한 값이다. 아래 CI는 계획된 5,000회 stratum bootstrap의 **각 contrast별 97.5% CI**이며 네 효능 주장 전체의 동시 신뢰구간으로 읽지 않는다. 단위는 accuracy percentage points(pp)다.

| 비교 | 평균 차이(pp) | 97.5% CI(pp) | 사전 기준 판정 |
|---|---:|---:|---|
| group_pg − fixed | −0.4974 | [−1.0336, +0.0121] | FAIL |
| group_pg − direct | +0.0172 | [−0.2409, +0.2748] | FAIL |
| group_pg observed − none | −1.0634 | [−5.5667, +4.0693] | FAIL |
| group_pg observed − shuffle | −0.2604 | [−2.1267, +1.5842] | FAIL |
| full/natural: group_pg − fixed | +0.1302 | [−0.8247, +1.0417] | PASS: lower > −2pp |

Utility는 계획된 disjoint 48 graphs 중 적격 48 graphs/58 cases(node95 48, node995 10)를 사용한다. 적격 조건은 source≥2와 missing query≥1이므로 이 utility 결론을 source0/1 전체에 확장할 수 없다. Full guard는 natural 48 graphs/768 queries의 관측 query 성능 보존 기준이다. 이 PASS는 missing-feature 개선이나 feature 추론 성공을 뜻하지 않는다. Feature CI의 폭 때문에 두 utility 차이의 부재·동등성도 입증되지 않았다.

| Primary arm/reference | Accuracy(%) | NLL | Brier |
|---|---:|---:|---:|
| fixed 000, adapter CE | 49.2904 | 0.84505 | 0.55585 |
| uniform, 8-trace probability mean | 49.4750 | 0.84558 | 0.55551 |
| direct, greedy | 48.7759 | 0.84547 | 0.55661 |
| group_pg, greedy | 48.7930 | 0.84541 | 0.55663 |
| frozen GraphPFN + same represented PCFI | 48.7883 | 0.86911 | 0.57259 |
| frozen GraphPFN + NaN representation | 47.7739 | 0.86241 | 0.57404 |
| LP | 47.0152 | 0.83496 | 0.54831 |
| context frequency prior | 43.8948 | 0.82638 | 0.54245 |

Uniform가 accuracy 점추정치상 가장 높다는 사실은 보존해야 한다. 다만 uniform은 8-trace mixture이고 다른 main 배포는 1 trace이므로 이를 계산량이 같은 정책 승리나 유의한 개선으로 바꿀 수 없다. PG와 동일 입력 frozen-PCFI의 accuracy 차이는 약 +0.0047pp에 불과하다. PG의 NLL/Brier는 frozen-PCFI보다 낮지만 LP와 frequency prior보다 높다. Accuracy 순위만으로 확률 예측까지 강해졌다고 주장할 수 없으며, 위 reference 차이에는 별도 유의성 판정을 추가하지 않는다.

## 주요 해석 — 다섯 가지

### 1. 학습된 θ와 greedy 경로 선택은 별개다

여기서 θ는 GraphPFN adapter weights, φ는 controller weights를 뜻한다. 원래 LimiX backbone·feature encoder·decoder는 frozen이고 약 3.564M adapter parameters가 supervised CE로 갱신된다. 513-parameter controller에만 direct objective 또는 group-normalized REINFORCE가 적용된다. 그러므로 이번 실험을 backbone 자체의 RL fine-tuning으로 기술하면 틀리다.

직접 확인한 모든 direct/group_pg seed의 main metadata는 **final 1,056 rows와 real 45 rows 모두 trace000**이다. 같은 θ의 replay000와 비교한 다섯 저장 지표 차이도 모두 0이며 primary routing contrast CI는 [0,0]이다. 따라서 실제 배포에서 controller가 추가 adapter pair를 선택해 얻은 이득은 이 표본에서 0이다. 000도 원래 12개 wrapper를 실행하므로 graph 연산 자체를 없앤 배포는 아니다.

그러나 학습 중 θ와 φ에 gradient가 있었고 checkpoint digest가 달라졌다. PG/direct의 θ는 여러 trace의 CE를 통해 학습되므로 000 배포라도 separately trained fixed θ와 다를 수 있다. **PG−fixed의 −0.4974pp는 이런 훈련 차이까지 포함하며, 배포 시 adaptive routing 개선으로 귀속할 수 없다.** PG와 direct의 작은 차이 역시 두 θ의 학습 결과 차이이지 서로 다른 greedy 경로의 효능이 아니다.

### 2. 000 greedy deployment는 stochastic distribution collapse의 증거가 아니다

Update128의 optimizer step **직전**, 마지막 sampled training episode에서 기록된 pφ(000)는 direct seeds11/12/13에서 약 .273/.305/.287, PG에서는 .289/.323/.341이다. PG의 마지막 8 samples에는 각각 5/4/4개 서로 다른 traces가 존재한다. Direct의 해당 full-tree entropy는 약 1.90–1.94 nats이다(균등 8 traces는 log8≈2.079). 이 기록만으로도 “훈련 내내 000만 sampling했다”는 설명은 맞지 않는다.

이 값들은 최종 checkpoint를 모든 evaluation state에서 다시 측정한 확률분포가 아니다. 최종 stochastic mass/entropy가 얼마나 입력별로 달랐는지는 여기서 확정하지 않는다. Greedy는 각 binary logit의 부호로 결정하므로 상당한 nonzero-action 확률이 남아도 모든 greedy trace가 000일 수 있다. 또한 sequential greedy는 일반적으로 joint-probability MAP trace나 minimum-risk action을 보장하지 않는다.

Stochastic training objective와 greedy deployment 사이의 차이는 확인된 설계 경계다. Cost penalty .01, 짧은 학습, state 표현, action family 또는 이 배포 규칙 중 무엇이 실패를 일으켰는지는 현 결과가 식별하지 못한다. “Penalty가 반드시 과했다”, “policy가 아무것도 학습하지 않았다”, “sampling inference면 성공한다”는 설명은 모두 미검증이다. 계획된 dev-global PG control은 세 seed 모두110을 골랐으나 primary 48.7952%로 greedy 48.7930%와 거의 같다. 이 secondary 결과도 사후 oracle trace로 main 실패를 대체할 근거가 아니다.

### 3. feature 이용 가능성과 RL이 추가한 feature utility는 분리해야 한다

Missing query가 source feature semantics의 도움을 받았다는 주장은 primary accuracy만으로 나오지 않는다. 조건부 utility 두 gate가 모두 실패했고, 추가 secondary difference-in-differences도 다음과 같다.

- [(PG observed−none)−(fixed observed−none)]: **−0.3906pp**, CI [−1.8446,+0.9766].
- [(PG observed−shuffle)−(fixed observed−shuffle)]: **−0.4557pp**, CI [−1.5191,+0.5425].

따라서 PG가 fixed보다 useful feature inference를 강화했다는 증거는 없다. 반대로 두 feature contrasts가 곧 “그래프만 사용했다”는 판정도 아니다. 원래 PFN/PCFI는 000에서도 source features를 받을 수 있고 실제 real development rows에는 observed가 none/shuffle보다 높은 점추정치가 있다. 이는 **기존 PCFI+PFN 경로의 feature 효과**와 양립하며 새 controller의 성과로 돌릴 수 없다. Source0의 graph/context 효과와 source≥2에서의 의미 있는 feature 전달을 계속 구분해야 한다.

### 4. 강한 reference와 real 전이까지 보면 효능 주장은 더 제한된다

아래는 같은 FP05 visible-row RP16→RMS→PCFI representation을 사용하는 real development 결과다. 각 값은 저장된 repetition/train-seed 평균 accuracy(%)다. Reused Cora/CiteSeer/PubMed 세 graphs의 기술 통계이며 새 unseen-domain generalization 또는 CI 주장이 아니다. LP는 features를 쓰지 않아 setting별 값이 같다.

| Graph | Missing(%) | fixed | uniform | direct | group_pg | frozen-PCFI | LP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 95 | 45.89 | 47.08 | 45.46 | 45.46 | 46.44 | 59.06 |
| Cora | 99 | 51.05 | 51.50 | 50.66 | 50.65 | 50.77 | 59.06 |
| Cora | 99.5 | 52.87 | 53.12 | 52.71 | 52.75 | 53.37 | 59.06 |
| CiteSeer | 95 | 29.35 | 29.06 | 28.37 | 28.33 | 28.55 | 38.41 |
| CiteSeer | 99 | 37.11 | 35.09 | 35.71 | 35.78 | 37.94 | 38.41 |
| CiteSeer | 99.5 | 33.43 | 33.84 | 32.23 | 32.27 | 33.58 | 38.41 |
| PubMed | 95 | 62.32 | 65.00 | 61.12 | 61.67 | 65.03 | 73.69 |
| PubMed | 99 | 67.19 | 69.06 | 66.25 | 66.77 | 69.37 | 73.69 |
| PubMed | 99.5 | 69.50 | 71.02 | 69.77 | 69.90 | 71.24 | 73.69 |

PG는 이 아홉 visible-feature setting의 점추정치 모두에서 frozen-PCFI 및 LP보다 낮다. 예를 들어 95%에서 PG의 (observed,none,shuffle95)는 Cora (45.46,24.32,42.27), CiteSeer (28.33,17.15,26.95), PubMed (61.67,36.90,59.60)다. Feature-associated difference는 존재하지만, 해당 차이가 원래 PCFI/frozen 모델보다 강화되었다는 결과는 아니다. Real none의 PG가 fixed보다 낮은 점도 “feature 없는 경우에는 RL이 graph-only 추론을 강화했다”는 설명을 지지하지 않는다.

이 비교는 native 수천 feature 차원의 public GraphPFN/PCFI/FISF와 다르다. Historical references는 원래 preprocessing과 budget을 보존한 별도 범위이며 현재 RP16 기반 수치를 그대로 native baseline과 연결해 우월성을 주장할 수 없다. 현재 LP가 강하다는 사실은 남겨야 하지만 RP16, prior mismatch, feature imputation 중 어느 요소가 성능 차이를 일으켰는지 이 실험은 분해하지 않는다.

### 5. 새 RL 알고리즘·meta-path 학습의 novelty는 성립하지 않는다

전문으로 확인한 [Policy-GNN](https://arxiv.org/html/2006.15097)은 이미 DQN으로 aggregation hop 수를 고른다. [RMS](https://arxiv.org/html/2112.12845v5)는 typed relation/STOP을 골라 meta-path를 탐색하며, validation reward로 downstream recommender를 평가한다. 세부 방법과 저자/분석자 한계는 `PRIOR_ART_VERIFIED.md`에 분리되어 있다. 이 두 논문이 extreme missing synthetic-only PFN 전이나 backbone RL을 이미 검증한 것은 아니다.

FP05의 구체적 차이는 frozen PFN 내부 세 위치에서 추가 adapter pair를 선택하고, synthetic labels로 actor와 CE adapter training을 결합해 비교한 제한된 응용이다. 그러나 실제 greedy 경로는 입력별로 달라지지 않았고 efficacy gates도 실패했다. 따라서 새 RL 알고리즘, heterogeneous meta-path discovery, 성공한 adaptive propagation, feature-aware policy inference 또는 missing-robust PFN의 성능 개선이라는 novelty 주장은 지지되지 않는다. 보존할 기여는 이 제한된 설계에 대해 direct/uniform/fixed를 포함한 **감사 가능한 부정 증거**다.

## 무엇이 닫혔고 무엇이 남는가

닫힌 범위는 released GraphPFN1.3+frozen LimiX16M, PCFI representation, detached 14-stat controller, 세 binary extra-adapter decisions, cost .01, 128-update/fit, 세 seeds, restricted official synthetic prior 및 사전 지정 greedy 배포다. 이 조건에서 PG는 fixed/direct보다 우월하지 않았고 feature utility 향상을 입증하지 못했다. 세 seed 평균을 고정한 graph-bootstrap 결과를 임의의 train seed population 또는 training-to-convergence 결과로 확장하지 않는다.

12 fits/1,536 optimizer groups는 계획된 bounded pilot로 충분히 실행되었다. 12,288 logical trace slots는 독립 task 수가 아니다. 각 fit은 128 task groups이며 seed별 124–125 unique episodes/101–107 bases다. 실제 training forwards/backwards는 각 9,600, 전체 preparation/calibration/deployment를 포함한 actual forwards는 64,035다. 약 6,974초의 component wall sum에는 LP/prior의 미기록 CPU 시간 등이 빠져 있어 통제된 latency benchmark가 아니다. 따라서 동일 logical budget과 동일 actual compute를 혼용하면 안 된다.

FP04와 달리 이번 학습에는 실제 observed-query support가 존재했다. Full/natural groups는 seeds11/12/13에 24/29/22개이며 observed queries는 384/459/347개였다. 그러므로 이번 실패를 “query가 전부 feature-free여서”라고 재활용할 수 없다. 다만 이것이 충분한 trainability·수렴을 입증하는 것도 아니다. Frozen backbone의 제약, 제한된 action family와 작은 task budget까지 포함한 결과다.

**남는 한 가지 질문:** 현재의 source/context 정보와 세 extra-adapter actions 안에, fixed CE training을 넘어서는 feature-conditional predictive benefit가 실제로 있고 그것을 context에서 식별할 수 있는가? 현재 결과는 이 편익의 존재와 policy/deployment의 식별 실패를 구분하지 못한다. 이 질문은 아직 검증되지 않은 범위로 남기며, 같은 final bank의 penalty/temperature/trace 사후 선택으로 답을 만들지 않는다. 이번 검토는 추가 실행을 제안하거나 승인하지 않는다.

# Reference-regret prior curriculum — 소형 GraphPFN pilot

2026-11-01. 판정: **현재 regret curriculum의 우위 미입증; 확장 보류**.

이번 실험은 공개 GraphPFN 1.3의 구조와 공식 generator를 유지하고, 동일한 합성 과제 bank에서 어떤 과제를 선택해 계속 사전학습할지만 바꾸었다. 배포 모델에 새 head, loss, router, 프로그램 실행기를 추가하지 않았다. 학습된 생성기나 새로운 mechanism library까지 구현한 실험은 아니다.

## 결과 해석

Reference-regret는 같은 128 update의 Random보다 Matched NLL을 0.00185 낮췄다(95% paired graph CI −0.00353~−0.00034). Manual 대비 개선도 확인됐다. 따라서 이 pilot을 아무 효과도 없었다고 정리하지 않는다. 다만 Random 대비 Matched accuracy 차이는 +0.01%p이고 CI가 0을 포함하므로, 더 많은 노드를 맞혔다는 근거는 없다.

핵심 가설인 reference의 독립적 가치는 아직 확인하지 못했다. Regret−Loss의 NLL 차이는 Matched −0.00090, cell-mask OOD −0.00218이며 둘 다 CI가 0을 포함한다. Matched accuracy에서는 Loss 대비 +0.49%p의 양수 CI가 있지만, 이 차이만으로 확률 예측과 OOD를 포함한 전반적 우위를 주장하지 않는다.

Reference 채점 비용까지 추가 학습에 쓴 RandomBudget은 Matched/OOD NLL의 점추정치가 Regret보다 낮다. 두 차이의 CI는 0을 포함하므로 RandomBudget의 유의한 우위라고도 부르지 않는다. 그럼에도 Regret이 채점 비용을 감수할 만큼 유리하다는 증거는 확보하지 못했다. Frozen FISF+GraphPFN 역시 Matched NLL 0.76227, accuracy 59.08%로 강한 대조군으로 남는다.

따라서 현재 finite-bank reference-regret sampling을 메인 메소드로 확대하는 것은 보류한다. 이는 모든 prior learning이나 학습된 generator가 불가능하다는 판정이 아니다. 이번 실험에서 구현하고 검증한 범위는 기존 generator의 과제 bank를 재가중하는 작은 curriculum뿐이다.

극단적 whole-node missing에서는 기존 전처리의 정보 손실도 확인됐다. Test 99.5% 조건의 uniform 20/32개, clustered 32/32개 episode는 PCFI 입력 전체가 0이었다. 이 조건에서의 이득을 실제 관측 feature를 잘 활용한 결과라고 해석하지 않는다. 다만 이 입력 손실이 curriculum 비교 결과를 일으켰는지는 별도 개입 실험으로 분리하지 않았으며, 전처리를 바꾸면 이 방법이 성공한다고 추정하지 않는다.

## 프로토콜

- 새 official-prior graph 64개를 train, 32개를 test로 사용했다. 이전 결과의 graph bank를 재사용하지 않았다.
- 공식 generator의 기존 소형 제한을 재사용했다: 요청 노드 256, feature 8~16, 평균 차수 4~12, 최대 4 class. 실제 sampler의 노드 수와 context 크기를 그대로 유지했다. 이것은 클래스당 k를 고정한 실제 ACM/IMDB 프로토콜이 아니다.
- Train은 whole-node uniform/clustered mask와 결측률 50/90/99.5%의 384 episode다. Zero-observation episode도 제거하지 않았다.
- 새 test graph에서 matched mask 192개, unseen cell-mask 64개, full-feature 32개를 평가했다. Cell-mask OOD는 관측 패턴 변화이며 새로운 graph family나 semantic schema는 아니다.
- 모든 학습 arm은 동일한 PCFI 입력 전처리, 공개 초기 가중치, trainable set, optimizer, 128 update를 사용했다. Query CE만 학습한다.
- 초기화는 고정하고 과제 선택 RNG만 11/12/13으로 바꿨다. 따라서 3개는 독립적인 backbone 초기화가 아니라 3개 schedule seed다.
- 첫 12 update는 모든 fixed-update arm에서 같은 stratified warmup이다. 실제 warmup 이후 전체 state hash가 seed별로 같음을 검사했다.
- Frozen LimiX transformer는 갱신하지 않고, 원래 adapters·encoder_x·y_decoder만 학습했다.

## 학습한 것과 선택 규칙

Random은 균등 선택한다. Manual은 같은 20% uniform floor를 유지하면서 50% → 90% → 99.5% missing task를 순서대로 강조한다. Loss는 task query CE를, Regret은 CE에서 reference NLL을 뺀 값을 priority로 사용한다. 초기 점수는 frozen PCFI+GraphPFN으로 계산하고, 선택된 episode의 점수를 EMA로 갱신한다. 미선택 episode의 점수는 최신 모델로 다시 계산하지 않고 이전 값을 유지한다.

Class 수에 따른 loss 크기 차이를 줄이기 위해 점수는 log(C)로 나눴다. Temperature 0.25, uniform floor 0.2, EMA old weight 0.8을 사전에 고정했다.

Reference는 각 합성 TRAIN episode에서 PCFI+GraphPFN, FISF+GraphPFN, LP의 query NLL 최솟값이다. 선택에 정답을 쓰는 것은 synthetic pretraining 단계로 한정한다. 이 최솟값은 배포 가능한 selector도, 정확한 Bayes risk도 아니다. 따라서 reference regret가 원리적으로 회수 가능한 정보량과 같다고 주장하지 않는다.

## 확률 예측과 정확도

지표는 query에서만 계산하고 graph별 각 mask의 task 지표를 동일 가중 평균했다. 학습 모델의 3 schedule seed는 지표를 평균하며 prediction ensemble은 아니다. NLL은 자연로그, Brier는 C개 class에 대한 오차 제곱합이다.

| 조건 | 방법 | Accuracy (%) | NLL ↓ | Brier ↓ |
|---|---|---:|---:|---:|
| Matched whole-node | Random | 58.78 | 0.76563 | 0.48949 |
| Matched whole-node | Manual curriculum | 58.66 | 0.77027 | 0.49092 |
| Matched whole-node | Loss priority | 58.30 | 0.76467 | 0.49108 |
| Matched whole-node | Reference-regret priority | 58.79 | 0.76377 | 0.48836 |
| Matched whole-node | RandomBudget | 58.84 | 0.76266 | 0.48806 |
| Matched whole-node | Frozen PCFI+GraphPFN | 58.46 | 0.80765 | 0.50396 |
| Matched whole-node | Frozen FISF+GraphPFN | 59.08 | 0.76227 | 0.48742 |
| Matched whole-node | LP | 54.48 | 0.84461 | 0.54237 |
| Unseen cell-mask | Random | 59.06 | 0.75271 | 0.48409 |
| Unseen cell-mask | Manual curriculum | 58.91 | 0.75252 | 0.48388 |
| Unseen cell-mask | Loss priority | 58.69 | 0.75412 | 0.48591 |
| Unseen cell-mask | Reference-regret priority | 58.96 | 0.75193 | 0.48362 |
| Unseen cell-mask | RandomBudget | 58.97 | 0.75074 | 0.48296 |
| Unseen cell-mask | Frozen PCFI+GraphPFN | 59.11 | 0.75808 | 0.48619 |
| Unseen cell-mask | Frozen FISF+GraphPFN | 59.53 | 0.75511 | 0.48407 |
| Unseen cell-mask | LP | 54.48 | 0.84461 | 0.54237 |
| Full features | Random | 61.25 | 0.72483 | 0.46268 |
| Full features | Manual curriculum | 61.22 | 0.72483 | 0.46249 |
| Full features | Loss priority | 60.99 | 0.72600 | 0.46403 |
| Full features | Reference-regret priority | 61.42 | 0.72482 | 0.46273 |
| Full features | RandomBudget | 61.02 | 0.72455 | 0.46221 |
| Full features | Frozen PCFI+GraphPFN | 61.02 | 0.72792 | 0.46412 |
| Full features | Frozen FISF+GraphPFN | 60.99 | 0.72733 | 0.46391 |
| Full features | LP | 54.48 | 0.84461 | 0.54237 |

## Regret 대조: paired graph CI

각 graph에서 mask와 schedule seed를 먼저 평균하고 32개 graph를 paired bootstrap 5,000회 재표집했다. CI는 이 3 schedule seed를 고정한 graph 변동을 나타낸다. 다중 검정 보정은 하지 않았으며, 짧은 exploratory pilot의 CI다.

| 조건 | Regret − 대조군 | Δ NLL, 95% CI | Δ Accuracy (%p), 95% CI |
|---|---|---:|---:|
| Matched | Random | -0.00185 [-0.00353, -0.00034] | +0.01 [-0.44, +0.50] |
| Matched | Manual curriculum | -0.00650 [-0.01389, -0.00093] | +0.14 [-0.23, +0.49] |
| Matched | Loss priority | -0.00090 [-0.00543, +0.00404] | +0.49 [+0.08, +0.87] |
| Matched | RandomBudget | +0.00111 [-0.00145, +0.00387] | -0.05 [-0.56, +0.48] |
| Cell OOD | Random | -0.00078 [-0.00283, +0.00114] | -0.10 [-0.63, +0.44] |
| Cell OOD | Manual curriculum | -0.00059 [-0.00225, +0.00106] | +0.05 [-0.42, +0.51] |
| Cell OOD | Loss priority | -0.00218 [-0.00605, +0.00133] | +0.27 [-0.23, +0.78] |
| Cell OOD | RandomBudget | +0.00119 [-0.00156, +0.00414] | -0.01 [-0.65, +0.62] |
| Full | Random | -0.00002 [-0.00195, +0.00189] | +0.17 [-0.37, +0.77] |
| Full | Manual curriculum | -0.00001 [-0.00207, +0.00214] | +0.20 [-0.33, +0.76] |
| Full | Loss priority | -0.00118 [-0.00435, +0.00158] | +0.43 [-0.03, +0.93] |
| Full | RandomBudget | +0.00026 [-0.00311, +0.00361] | +0.40 [-0.24, +1.11] |

Matched와 OOD에서 네 대조군 모두에 대한 NLL 개선이 확인됐는지의 보수적인 집계 판정은 No다. Full-feature의 비열등성 margin은 사전에 설정하지 않았으므로, CI가 0을 포함하는 것만으로 비열등성을 증명했다고 해석하지 않는다.

## 계산 비용

| 방법 | 3 seed 평균 update | Training loop (초) | Cold-score 비용 (초) | 합계 (초) |
|---|---:|---:|---:|---:|
| Random | 128.0 | 85.69 | 0.00 | 85.69 |
| Manual curriculum | 128.0 | 86.59 | 0.00 | 86.59 |
| Loss priority | 128.0 | 82.66 | 51.17 | 133.83 |
| Reference-regret priority | 128.0 | 79.99 | 116.84 | 196.83 |
| RandomBudget | 319.7 | 197.23 | 0.00 | 197.23 |

RandomBudget는 각 seed의 Regret training-loop 시간에 전체 train-reference cold-score 비용을 더한 시간을 추가 최적화에 사용하도록, 같은 초기화에서 새로 학습했다. 캐시는 실제로 공유했지만 비교에서는 reference 구축 비용을 seed마다 전액 부담시켰다. 따라서 여러 seed에 amortize한 경우보다 Regret에 보수적인 비교다.

RandomBudget는 학습 길이를 사전에 update 수로 알 수 없어 시간 기반 cosine LR를 사용했다. 나머지 네 arm은 update 기반 cosine LR다. 완전히 동일한 LR trajectory의 비교는 아니다. 시간 예산은 update 경계에서 검사하므로 마지막 update만큼 초과할 수 있으며 개별 초과분은 summary.json에 기록했다.

공통 bank 준비는 62.13초였고, 이 중 graph 생성 29.15초, 공통 PCFI 전처리 9.76초였다. 공통 준비·model loading·GPU warmup·checkpoint 저장·평가 비용은 위 training 비교와 분리해서 원시 기록에 남겼다. 전체 실험 총비용과 optimizer-loop 비용을 혼동하지 않는다.

## Prior 선택 진단

Train episode 중 초기 reference headroom이 양수인 것은 247/384개다. 평균 초기 reference NLL gap은 0.05404다.

초기 normalized loss와 normalized regret의 Spearman은 0.4264였다. 이를 비교해야 Regret이 단순 hard-example mining과 실제로 다른 과제를 강조했는지 평가할 수 있다. 상관관계 자체는 성능 개선이나 기전의 인과 증거가 아니다.

| 방법 | 선택 횟수: uniform .5/.9/.995 | clustered .5/.9/.995 | seed평균 unique episode |
|---|---|---|---:|
| Random | 20.3 / 21.0 / 20.3 | 23.7 / 19.3 / 23.3 | 108.0 |
| Manual curriculum | 19.7 / 21.0 / 21.0 | 22.7 / 21.3 / 22.3 | 109.7 |
| Loss priority | 16.3 / 16.7 / 24.7 | 15.3 / 21.0 / 34.0 | 100.7 |
| Reference-regret priority | 15.0 / 18.7 / 26.7 | 16.3 / 16.0 / 35.3 | 103.3 |
| RandomBudget | 53.3 / 49.3 / 51.3 | 55.7 / 55.3 / 54.7 | 212.7 |

## 검증과 한계

- 숨긴 feature 값을 1e12로 바꿔도 입력 전처리가 변하지 않는지 검사했다. 모델 입력 dataclass에는 query label이 없다. Query label은 synthetic train loss/score 또는 평가에만 사용한다.
- 반복 forward의 결정성, 초기 frozen reference와의 NLL 일치, 실제 weight update, frozen parameter gradient 부재, 공통 score offset에 대한 sampling 불변성을 확인했다.
- 원시 예측에서 모든 metric을 독립 재계산하고 checkpoint·data·source hash를 확인했다.
- 공식 prior 자체의 NaN/클래스 coverage/작은 SBM draw 검사는 유지했다. 성능에 따른 graph rejection은 하지 않았다.
- 모든 arm은 같은 학습 bank에 접근할 수 있다. Loss/Regret은 초기 채점에서 bank 전체의 synthetic query labels를 활용한다. 이 추가 감독 계산을 무료로 취급하지 않았다.
- 이 실험은 finite task-bank priority sampling이다. 학습된 생성 mechanism, 새로운 program library, adversarial generator, 무한 task stream은 구현하지 않았다.
- 실험은 같은 official graph/SCM family의 새 graph와 mask pattern OOD까지다. 새 semantic schema·실제 graph·원래 클래스당 20 labels 조건·논문 novelty는 입증하지 않는다.
- Input은 PCFI 고정 전처리다. 어떤 curriculum이 좋아도 그것만으로 missing 자체의 표현 한계나 information loss를 해결했다고 해석하지 않는다.

## 저장 자료와 재현

실행 순서: prepare.py → references.py → run.py → analyze.py → report.py.
기존 GraphPFN용 Windows virtualenv를 사용하며, upstream source와 공개 가중치는 읽기만 했다.
protocol.json은 학습 전 고정 설정, bank/manifest.json은 graph·mask·input hash, results/run_manifest.json은 학습 시작 시 source hash를 기록한다.
results/summary.json과 task_metrics.csv는 전체 조건·seed·paired CI를 포함한다.
각 arm 디렉터리에 training.json, train_log.json, trainable.pt와 query 예측을 보존했다.

새로운 loss나 더 긴 Regret 재학습으로 이 pilot을 구제하는 실험은 자동으로 추가하지 않는다.

## 최종 입력·실행 기록

학습 가능한 parameter는 4,002,059개, 고정한 parameter는 16,088,642개다. 추가 random feature 0개, single-member 추론으로 비교했으므로 공식 ensemble benchmark의 성능이라고 부르지 않는다.

입력 audit에서 확인한 test 99.5% 조건의 전체 PCFI input zero episode 수:

| Mask | 전체 입력이 0인 episode / 전체 |
|---|---:|
| uniform_nodes | 20 / 32 |
| clustered_nodes | 32 / 32 |
| uniform_cells | 1 / 32 |

Whole-node 관측 행이 0~1개이면 관측값 기반 centering 뒤 입력이 전부 0이 될 수 있다. 이 구현에서는 feature observation mask도 별도로 forward에 제공하지 않는다. 그러므로 이 부분에서의 개선을 실제 관측 피처 활용의 증거로 해석하지 않는다. 이 입력 손실은 직접 확인했지만, curriculum 비교 결과의 원인으로 인과적으로 분리한 실험은 아니다. SCOPE_NOTES.md에 자세히 적었다.

학습·평가가 진행되는 동안 protocol과 학습 소스는 고정했다. Input audit와 문서 작성은 그 이후의 기술 점검이며, 과제 선택·가중치·checkpoint 선택에 반영하지 않았다. results/environment.json은 runtime과 read-only import helper의 hash, results/manifest.json은 완료된 파일 전체의 SHA-256을 제공한다.

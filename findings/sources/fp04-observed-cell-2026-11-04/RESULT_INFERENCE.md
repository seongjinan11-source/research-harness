# FP04 결과 독립 심사: 추론·수치·관측 정보 사용

검토일: 2026-11-04. `PLAN_v2`, protocol, frozen model/checks와 `results/REPORT_KO.md`, summary/paired/audit/budget 및 저장된 예측을 검토했다. 다른 현 주기 결과 리뷰는 읽지 않았다. CPU 읽기 전용 진단만 실행했으며 새 모델 forward·학습·GPU 실행이나 기존 결과 교체는 하지 않았다.

**판정: 구현·저장 예측의 정합성과 과학적 성공을 구분해야 한다. Audit PASS는 지지하며 네 gate FAIL도 독립 재현했다. 이 고정 예산의 rank4 conditional-message 모델은 유용한 missing-feature 추론을 입증하지 못했다.** Observed arm의 작은 정확도 우위와 실제 그래프 성능은 부분 결과로 보존할 수 있으나, 관측 feature 내용의 유용성을 보여 주지는 않는다. Transformer나 observed-data PFN 전체의 실패로 일반화할 근거는 없다.

## 1. 독립 재검산

Production `analyze.py`나 그 metric/bootstrap helper를 호출하지 않는 별도 CPU NumPy 계산으로 evaluation metadata가 가리키는 **8,181 NPZ**를 읽었다. 각 파일 SHA256을 metadata와 대조하고, p를 float64에서 [1e−8,1] clip 후 행 정규화하여 accuracy/NLL/Brier를 다시 계산했다. 세 metric의 metadata 대비 최대 차이는 0이었다. 이는 root가 수행한 source/schedule/checkpoint/input-identity audit 전체를 다시 수행했다는 뜻은 아니다.

Primary는 base별 고정 3 train seeds×2 rates를 먼저 평균하고, 12개의 family×shape×K strata 안에서 base graph를 복원 추출했다. Utility는 disjoint이며 source≥2인 rate만 선택하여 seed/rate를 base 안에서 먼저 평균했다. 5,000 bootstrap, protocol seed47069000+j, 1.25/98.75 percentile로 네 CI를 재구성했다. 저장된 paired 값과 최대 차이는 2e−15 미만이다.

| 비교 | Accuracy 차이 %p | 97.5% CI %p | 독립 판정 |
|---|---:|---:|---|
| observed−mask | +0.4232 | [−0.2930,+1.1340] | FAIL |
| observed−pcfi | +0.4232 | [−0.2713,+1.1176] | FAIL |
| observed−none | +0.0434 | [−0.6293,+0.6293] | FAIL |
| observed−shuffle | −0.3255 | [−0.8247,+0.0868] | FAIL |

Primary는 96 graphs, utility는 별도 조건부 estimand인 48 graphs/68 rate cases다. Source0/1을 primary에 포함한 것은 계획과 일치한다. Training budget은 9 fits,19,748 trainable parameters,6,912 updates,27,648 exposures이며 기록상 target update는 없다. Frozen checks의 CPU/GPU PASS, full-loop/NumPy 참조와 forward/gradient 동치, bridge 검사는 구현된 연산과 경로를 지지한다. 최초 bridge assertion의 context-node 대칭 요구 오류는 이미 별도 prelock repair에 보존되어 있으며, 이번 gate 실패의 모델 결함 증거가 아니다.

검산 기준 파일 SHA256:

```text
protocol.json               1f302d384aefae7036dfe001442c5f2822a1ec88cea1dd7a20e7eb4684a72665
bank/final/manifest.json     292f65e8b2296f071541132852b23b02a9441cbdad805f7618c630cd02021ff3
results/summary.json        c3d931308c21e58337bb22885552f7327fc9a92b4a2453e172bae6311a5a9989
results/paired.json         e1c0643cd100b7fa2ba1b7c81196a9fbd8a824c15ad19574f04de6631b039fff
results/audit.json          cc77a21a72c7c74f77f3eb9746112e6f9e03e180e663ca54dd5d45bd741ee3d0
```

이 다섯 파일은 진단 전후 hash가 같았다. NPZ 원본과 evaluation metadata는 수정하지 않았다. 첫 임시 stdout 진단 명령의 구문 오류를 고친 뒤 재실행했으며, 해당 오류는 모델·결과 파일을 읽거나 쓰기 전에 발생했다.

## 2. +0.423%p와 NLL의 해석

Primary observed accuracy48.253%, mask/pcfi47.830%이다. Observed NLL1.021423은 mask1.023594보다 약0.002171 낮고 Brier도 약0.00160 낮다. 점 추정치의 방향은 보존할 수 있지만 CI가 0을 포함하는 accuracy gate를 NLL로 바꾸어 통과시킬 수는 없다. Seed별 observed−mask accuracy 차이는 +0.358,−0.228,+1.139%p로 일관된 seed별 우위도 아니다.

더 직접적인 내용 사용 비교인 utility의 NLL 차이는 observed−none −0.0000641, observed−shuffle −0.00000139에 불과했다. 같은 graph bootstrap으로 얻은 사후적 97.5% CI는 각각 [−0.0002047,+0.0000754], [−0.0000310,+0.0000299]다. 이는 secondary 진단이며 gate를 대체하지 않는다. Full/nonquery-visible 조건에서도 observed−none accuracy는 centroid family +0.304%p, field family −0.217%p이고, NLL은 각각 +0.00137,+0.00148로 악화했다. 따라서 단지 extreme mask의 source0/1 때문에만 feature utility를 검출하지 못했다고 설명하기도 어렵다.

같은 예산의 세 arm은 각자 학습된 서로 다른 파라미터를 갖는다. Observed가 mask보다 좋아졌다는 사실만으로 정확한 value→column 경로를 더 잘 학습했다고 식별할 수 없다. Routing이 학습한 graph/context prior나 regularization을 바꾼 효과도 포함된다. Tiny nonzero bridge derivative는 경로가 구현되었다는 증거이며, 학습 후 실질적인 정보 활용이나 utility의 증거가 아니다.

## 3. 실제 그래프: 작은 probability 변화와 많은 argmax 변화

Observed real95의 none/shuffle 비교를 저장 p에서 다시 계산했다. 아래 수치는 각 graph의 **3 train seeds×3 context repetitions, 총9개 record의 query 내부 평균을 동일 가중 평균**한 것이다. Query 수로 record를 가중하거나 새 graph 표본으로 간주하지 않았다. TV는 query별 `0.5*sum_K(abs(p95−pref))`, argmax 변화는 두 저장 분포의 top1 label 불일치율이다.

| Graph | 95−none acc %p | 95−shuffle acc %p | TV: none / shuffle | Argmax 변화: none / shuffle % |
|---|---:|---:|---:|---:|
| Cora | −0.0476 | −0.0466 | 1.73e−4 / 2.07e−5 | 10.46 / 7.58 |
| CiteSeer | −0.1419 | −0.0242 | 1.30e−4 / 6.38e−6 | 25.17 / 16.81 |
| PubMed | −0.2626 | −0.0214 | 8.63e−5 / 1.44e−5 | 7.15 / 4.87 |

실제 feature를 제거하거나 배치를 바꿔도 accuracy와 NLL에 유용한 이득은 드러나지 않는다. 그러나 확률 변화가 정확히 0은 아니고 argmax도 바뀌므로 “수치적으로 feature를 전혀 읽지 않는다”는 표현 역시 부정확하다. **의미 있는 확률 구분과 예측 효용을 학습했다는 증거가 부족하다**가 지지되는 결론이다.

Near-tie를 별도 확인했다. 각 query의 margin을 정렬한 p의 1위−2위로 정의했다. Real95에서 margin<1e−5 비율은 Cora27.03%,CiteSeer65.64%,PubMed16.55%다. 각 record의 median margin을9개 평균하면 각각5.99e−4,5.96e−8,1.56e−3이다. 평균 max probability도0.1645/0.1768/0.3492로 각 K=7/6/3의 uniform1/K에 가깝다. 특히 CiteSeer에서는 작은 probability 이동이 많은 top1 변경으로 증폭된다. 이 현상은 작은 accuracy 차이를 유의미한 feature 내용 활용으로 읽지 말아야 할 이유이며 gate를 변경할 이유가 아니다.

### 네 graph round의 수학적 범위와 재현 절차

`bank/real/manifest.json`의 mode=none 입력 NPZ에서 `src,dst,context,query,classes`를 읽고 입력 SHA를 확인했다. Directed edge adjacency에서 모든 context node를 시작점으로 multi-source BFS를 수행하여 가장 가까운 context까지의 unweighted hop 수를 계산했다. 반복별 query 집합에서 거리>4인 비율을 구한 뒤 세 repetitions를 동일 가중 평균했다. 저장된 `results/observed_s{11,12,13}/real_evaluation.json`에 연결된 none 예측은 같은 query 순서를 사용한다.

관측 셀이 없는 observed arm은 column read가0이다. 실제 context 수는 모든 class에4개씩으로 균형이므로, context에서4 hops보다 먼 node는 네 번의 local graph update 뒤에도 수학적으로 class 대칭이어야 한다. 정적 degree/count 통계는 이를 깨지 못한다. 해당 query 비율은 **Cora20.81%,CiteSeer63.73%,PubMed12.54%**였다. 이 집합에서 저장 probability의 class 간 최대 spread는 각각1.64e−7,1.79e−7,2.98e−7이다. 정확한 top2 tie 비율을 모든 none query에서 계산한9-record 평균은17.31%,55.35%,8.03%였다.

따라서 많은 query가 이 모델의 label-anchor receptive field 밖에 있고, 그 영역의 매우 작은 class 비대칭은 **FP32 합산 roundoff와 양립한다**. 정확한 원인을 GPU 재실행으로 식별한 것은 아니며 모델 버그나 특정 kernel을 원인으로 확정하지 않는다. Argmax tie/near-tie에 민감한 accuracy와 사실상 균등한 probability를 함께 보고해야 한다. 사후 threshold로 label을 바꾸거나 원래 점수·CI를 교체하지 않았다. 이4-hop 논증은 column support가 빈 **observed-none arm**에 대한 것으로, A=1인 mask/PCFI의 implicit global column 경로까지 같은 반경이라고 주장하지 않는다.

## 4. Native와 외부 PCFI의 비교 범위

실제95 observed는 matched mask/PCFI보다 약4–9%p 높지만 none에서도 높은 성능이 유지되므로 이를 missing feature 복원 성공이라 부를 수 없다. Restart LP 대비로는 Cora 약6.18%p,CiteSeer8.43%p,PubMed9.69%p 낮다. 강한 label-propagation 기준선을 능가한 결과가 아니다.

물리적 관측값·context/query identity가 맞아도 새 PCFI arm은 **native columns+observed-only RMS+공통 basis clipping+새 소형 모델**이고, legacy PCFI+public PFN은 원래 preprocessing/PCA/ensemble과 pretrained predictor를 사용한다. 두 결과의 차이는 completion 방식만의 효과가 아니다. Raw PCFI cache는 보존되었고 model representation clip 대상 비율의 최대치는 Cora0.096%,CiteSeer0.340%,PubMed0.238%였다. 이 수치만으로 clipping의 성능 영향이 0이라고 확정할 수도 없다.

새 모델의 graph reasoning은4 rounds이고 literal PCFI preprocessing은100 steps다. 이 둘은 서로 다른 연산이다. PCFI arm에도100-step fill은 이미 제공되므로 “모든 arm이4회만 보아서 PCFI를 시험하지 않았다”는 해석은 틀리다. 반대로 equal parameters/updates가 equal graph propagation reach나 equal FLOPs를 뜻하지도 않는다. 관측 정보가 없는 상황에서 global LP보다 범위가 짧다는 구조적 한계와, 내용 기반 column 추론을 충분히 학습했는지는 분리해야 한다.

## 최종 권고

이 고정 candidate는 gate FAIL로 종료하고 수치 정합성·전체 결과·작은 점 추정치 우위는 보존하는 것이 타당하다. 확인된 것은 작은 네-round/rank4 조건부 메시지 네트워크가 이 prior와768-update 예산에서 유용한 feature 영향을 입증하지 못했다는 사실이다. 이유를 단일하게 rank, propagation depth, synthetic law, CE 학습, native dimension 차이 중 하나로 확정할 식별 실험은 없다. 이 결과를 근거로 Transformer 전체를 폐기하거나, 반대로 학습량만 늘리면 해결된다고 결론 내리면 안 된다. 다음 계획은 관측 내용이 class contrast를 만들어 query까지 운반하는지를 먼저 명시하고, graph-only 성능과 별개의 성공 조건으로 유지해야 한다.

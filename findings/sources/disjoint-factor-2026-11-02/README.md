# Disjoint feature–label factor PFN pilot

질문은 feature가 있는 노드와 label context가 겹치지 않아도 graph 구조로 두 정보를 결합해 zero-shot 분류할 수 있는가다. 이전 bank의 사후 분석과 별개로 새 graph seeds에서 실제 학습한다.

## 계산

노드마다 잠재 class spin `s_i`, feature column마다 잠재 방향 `z_f`를 둔다. 관측 cell만 `exp(kappa_if s_i z_f)` factor로 연결한다. `kappa_if`는 관측값에 대해 odd인 shared neural function이다. Graph에는 role별 edge/nonedge factor가 있고, context label은 hard clamp한다.

하나의 계산에서 context→graph→관측 feature source→column state→다른 노드의 경로를 처리한다. 별도 numeric completion이나 role-posterior head를 만들지 않는다. 같은 factor로 conditional label belief와 Bethe role score를 계산하고 predictive CE를 유한 BP 계산 전체로 미분한다. Missing feature의 값을 정확히 생성·적분했다고 주장하지 않는다.

Binary CSBM와 attributed-graph BP는 기존 아이디어다. 이 실험은 새로운 이름의 novelty를 주장하기 위한 것이 아니라, 현 학습 모델이 회수하지 못한 disjoint evidence의 이득을 이 연산으로 회수할 수 있는지 확인하는 known-family kill test다.

가장 가까운 model-based 선행연구는 Duranthon·Zdeborová의 [Optimal Inference in Contextual Stochastic Block Models](https://arxiv.org/html/2306.07948)다. 이 연구도 node labels와 latent Gaussian feature centroids를 함께 추론하고 AMP–BP 및 parameter estimation을 제공한다. 우리의 binary column latent, missing-mask 처리, learned potential만으로 새 원리라고 주장할 수 없다. 현재 Grid6은 작은 family의 강한 비교이며 원 논문의 AMP–BP를 재현하거나 이긴 결과로 해석하지 않는다.

## 대조와 고정 조건

- Factor PFN / 실제 공개 Mask GraphPFN / 실제 PCFI 입력으로 학습한 GraphPFN: 같은2048 training tasks,256 successful updates,3 seeds, query CE만 사용.
- Known-family Grid6 exact/BP, training modal-parameter factor BP, 정확한 Gaussian family의 feature-only, LP/class prior/uniform.
- 새96 test graphs 중 matched48 primary; low density/high signal secondary. 각 graph context1/2/4, observed node3/6, disjoint/paired-overlap.
- 평가 graph에서 gradient update 없음. Parameter 수는 아키텍처마다 다르고 update/exposure만 맞춘다. 전체 비용은 따로 측정한다.

사전 조건과 gate는 `protocol.json`, 수치 점검은 `results/preflight.json`, source/bank lock은 `results/lock.json`에 있다. 실패 시 추가 loss·새 head·예산 확대·test 선택으로 구제하지 않는다.

실행: `experiment.py prepare`, `experiment.py preflight`, `experiment.py run --seed 11` (12/13도 동일), 모든 평가가 끝난 뒤 `analyze.py`와 `benchmark.py`.

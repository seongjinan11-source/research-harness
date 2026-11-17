# SPFN01 v1 method and prior-art review

**판정: 비교군을 보강하면 진행할 가치가 있는 bounded pilot.** 관측 라벨이 적은 클래스의 개별 증거를 보존하고, 많은 클래스는 첫 모멘트로 요약하는 설계는 명확히 비교할 수 있다. 초안의 가장 큰 약점은 같은 예산의 raw-anchor 대조군이 구조 하나에 의존하고 토큰의 실제 크기와 클래스의 전체 라벨 수를 섞는다는 점이다. 이 비교를 바로잡아야 성공해도 무엇이 나아졌는지 설명할 수 있다. 이 검토는 계획과 논문에 근거하며 구현이나 성능을 확인한 것이 아니다.

## 검증할 주장과 선행 경계

| 선행 | 이미 하는 계산 | 이 계획이 검증할 차이 |
|---|---|---|
| [LoGIC, §§IV–V](https://arxiv.org/html/2609.05955v1) | 고정된 backbone에 query cluster별 labeled row 예산을 두고 PPR, feature-kNN, coverage로 꺼낸다. Coverage는 비례 class strata를 허용한다. Adapter용 unlabeled halo에는 별도 예산을 두며, 논문의 channel/budget 선택은 held-out 검증 점수를 쓴다. | 전체 labeled pool의 일부를 그대로 가져오는 대신, 많은 라벨을 같은 class의 여러 anchor에서 만든 query별 diffusion moment로 보낸다. 이를 관측 라벨이 적은 class의 singleton 보존과 결합한다. PPR retrieval만 이겼다고 LoGIC 전체를 이겼다고 할 수 없다. |
| [Ephris, §3 및 App. B](https://arxiv.org/html/2609.37057v1) | Synthetic SCM pretraining, inducing-token refinement와 **노드별 feature compression**, sparse graph와 global nodes를 통한 label-conditioned message passing을 이미 결합한다. | 이 계획의 작은 예산은 노드당 feature token 수가 아니라 **decoder가 읽는 labeled support group 수**다. Graph ICL의 sparse 계산, compression, synthetic prior 자체는 새 주장이 아니다. |
| [NodePFN](https://arxiv.org/abs/2604.19028), [GraphPFN](https://proceedings.mlr.press/v306/eremeev26a.html) | Synthetic graph prior, 구조와 feature 관계를 포함한 ICL, target별 학습 없이 분류가 이미 제시됐다. NodePFN은 context-query attention과 local message passing을 결합한다. | 작은 shared head와 group allocation의 효과를 같은 계산 안에서 검증한다. Released NodePFN 참조가 가능하면 유용하나 지원 예산·backbone·학습 규모가 다르므로 내부 ablation과 분리한다. |
| [PPRGo](https://arxiv.org/abs/2007.01570), [ProtoNet](https://papers.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html), [G-META, §5](https://proceedings.neurips.cc/paper_files/paper/2020/file/412604be30f701b1b1e3124c252065e6-Paper.pdf) | PPR로 정보 집계, class prototype과의 거리, graph-local support embeddings를 prototype으로 요약하는 원리가 각각 선행한다. G-META는 support loss로 target task 적응도 수행한다. | 고정된 finite diffusion으로 group mass와 첫 모멘트를 모으는 target-optimizer-free 분류에서, support scarcity를 반영한 할당이 어느 정도 도움이 되는지를 검증한다. 새 prototype/PPR 원리나 최초 subgraph 기반 few-shot 방법으로 볼 수 없다. |

`F_T`는 seed에 선형이어서 동일 group 안 singleton들의 mass와 numerator를 합한 값과 일치한다. 이 성질은 pooled centroid를 입력으로 한 비선형 MLP가 singleton 전체의 logits를 보존한다는 뜻이 아니다. 예컨대 weighted mean까지 같아도 두 집단의 분산과 모드가 다르면 raw-anchor 유사도 분포는 다를 수 있다. 그래서 top/full 비교에 실질적 의미가 있다. 또한 현재는 전체 graph에서 finite diffusion을 계산한다. “Subgraph”는 local evidence의 의미로 쓰거나 제목에서 범위를 밝혀야 하며 LoGIC처럼 materialized subgraph만 방문하는 알고리즘으로 소개할 수 없다.

## 한 가지 중요 수정: 같은 quota의 raw-selection 대조군을 강하게 만들기

`top`은 tail과 같은 per-class quota라서 질문의 핵심 대조다. 하지만 PPR만 사용하면 유사한 feature가 더 좋은 증거인 graph의 대표 anchor를 놓칠 수 있다. 반면 tail은 graph proximity와 feature를 함께 사용한 요약으로 읽힌다. LoGIC §V-D도 PPR, feature, coverage 중 최선이 데이터에 따라 달라진다고 보고한다. PPR-only raw selection보다 이 계획의 summary가 이겼다는 좁은 결론은 가능하지만, 일반적인 “선택보다 압축” 설명에는 약하다.

**추천:** 기존 `top` arm을 같은 quota의 고정 PPR+feature rank-fusion raw selector로 교체한다. 한 class 안에서 각 anchor의 $D_T[q,i]$ (group에 쓴 동일 finite-diffusion kernel)과 $-\|H_q-H_i\|^2$ 순위를 각각 0–1로 바꾸고, 동률에 평균순위를 적용해 두 점수의 평균이 큰 순으로 뽑는다. 마지막 동률은 사전 지정한 stable node ID로 해소한다. 모두 같은 PPR이면 feature rank가 구분한다. 더 많은 training arms나 검증 점수 기반 채널 탐색 없이, 의도한 low/high structure 및 weak/strong feature 조건을 두루 다루는 비교다. 이 고정 혼합은 LoGIC의 재현이나 모든 selector의 최적값은 아니다.

이 비교에서 `member_count`는 실제 group 크기 $|g|$로 통일한다. Raw token은 singleton이므로 1이다. $n_c/\mathrm{quota}_c$는 집계를 하지 않은 anchor에 압축 그룹의 질량을 배정한 값이어서, 실제 크기 채널 및 log-count pooling과 섞으면 tail−top 차이가 선택/요약뿐 아니라 입력의 count 의미 차이도 반영한다. 원래의 관측 클래스 support $n_c$를 알려 주려면 모든 arm에 동일한 별도 class count 채널로 공급한다. 선택된 singleton의 diffusion mass는 해당 anchor의 실제 mass다. 따라서 tail−top은 **동일한 quota와 관측 support 정보에서 모든 anchor의 group moment 대 선택된 raw anchor**의 비교다. Tail−proportional은 동일 group budget과 moment 연산에서 quota 할당의 비교다. 두 비교 모두 matched pretraining을 포함한 알고리즘 비교이며 보편적 인과 분해를 뜻하지 않는다.

## Imbalance와 성공 판정의 범위

- **Tail은 관측 support가 1–4개인 class다.** Population minority, 그래프의 degree 불균형, query label imbalance, 관측 feature 결측은 다른 문제다. 관측 tail/head 역할과 실제 class occupancy를 독립 배정한다는 초안은 이 구분에 좋다. 생성 제약(최소 class 크기, head와 query 동시 확보)이 실제로 그 독립성을 깰 수 있으므로 realized population×support cross-tab을 evaluator에서 남긴다. Query truth로 정한 클래스별 strata는 점수 정의용이며 추론에는 노드 features/IDs만 전달한다. [Ochal et al.](https://arxiv.org/abs/2101.02523)도 few-shot episode의 support 불균형을 별도의 문제로 검토했다. 모든 라벨이 적어도 한 번 주어진 본 계획에는 zero-anchor/open-set 결론이 없다.
- **지표는 클래스 균등 의사결정이다.** Tail bAcc는 관측 support가 적은 클래스의 macro recall로 해석한다. Class-balanced CE와 class-stratified query sampling으로 얻은 확률 지표는 이 평가 분포의 NLL/Brier다. 이를 자연 class prevalence의 보정된 posterior나 long-tail population 성능으로 볼 수 없다. 보고서의 overall accuracy도 stratified queries에서 측정했다는 조건을 붙인다. Reused real graph의 무작위 support 역할은 transfer의 기술적 점검에 해당한다.
- **예산은 라벨 수가 아니라 decoder token 수다.** Tail은 모든 head 라벨을 읽어 seed moment를 만들고 top 역시 후보 전체를 점수화한다. Peak graph/preprocessing memory와 decoder context token 수를 나눠 기록한다. 계획의 held-out 조건에서 K≤5, head≥1, head size≥24, tail≤4이므로 worst case 한 head+tail 16개에서 `full/tail ≥ (24+16)/(2+16) = 2.22`다. “2× context” 조건은 해당 bank에서 실측 속도 가설이라기보다 구현/예산 확인이다. 실제 end-to-end 시간과 memory는 graph diffusion까지 포함해 별도로 보고한다.
- **간단한 규칙이 이미 답하는지 확인한다.** Balanced PPR와 H nearest-centroid를 유지하고 joint go/no-go에서 “loses”의 대상 metric과 판정 규칙을 source lock 전에 명시한다. Full guard는 같은 학습 절차의 all-anchor 대비 bAcc 비열등성이지 full-context logits의 정확한 복원이 아니다. 유리한 sub-stratum이나 optional external 모델 결과로 전체의 사전 gate를 대신하지 않는다.

이 수정 뒤 성공하더라도 주장 범위는 **주어진 완전 feature와 모든 class가 관측된 task에서, 고정 token budget의 support-skew 대응**이다. 실제 population 불균형, missing-feature 개선, million-node scale, 또는 novel graph pretraining은 이 실험으로 확인되지 않는다.

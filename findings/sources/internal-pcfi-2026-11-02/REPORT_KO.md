# 실제 PCFI 내부화 GraphPFN — 결과

사전 primary accuracy gate=False, full-feature2%p guard=True, 확대 기준=False.

## 무엇을 바꿨나

이전 generic MLP refiner를 다시 돌린 실험이 아니다. 실제 PCFI의 거리 pseudo-confidence, 관측값 고정100회 전파와 마지막1회 피처 상관 보정을 사용했다. 네 모델은 공개 GraphPFN/LimiX의 같은 원래 초기화에서 새로 학습했고, 같은12층을 한 번 통과한다.

External은100회 PCFI를 앞에서 마치고 PFN을 실행한다. Internal은25회씩 PFN0/3/6/9층 앞에 배치하고 encoder increment를 feature tokens에 더한다. 마지막 block에만 원래 상관 보정을 한다. Context와 label-blind는 같은513-parameter router로 각 block의 전파 가중치만 조절한다. 임의의 숫자 혁신·추가 posterior head·재구성 loss는 없다.

Context router는 현재 PFN feature+label token의 context-class prototype support를 읽는다. Label-blind는 해당4개 입력을 모두0으로 하고 PFN hidden/labels를 읽지 않는다. 두 모델의 main classifier는 context labels를 쓴다. 따라서 차이는 context+PFN-feedback 경로 전체에 대한 것이며 동일 hidden representation에서 labels만 뺀 효과가 아니다. Blinded 입력 일부 weights는 비활성이다.

## Primary: 자기 피처 없는 query accuracy

새64 base graphs, context최대4/class, uniform/clustered node90/99.5% 네 조건과 고정3 seeds를 graph 안에서 평균했다. Missing query는 원래 physical features 전체가 미관측인 query다. Odd padding은 제외했다. 전체 query와 k1/cell/full은 별도다.

| 방법 | Accuracy % | NLL | Brier |
|---|---:|---:|---:|
| 외부 PCFI → PFN | 49.24327 | 0.867128 | 0.559929 |
| 내부 고정 PCFI | 49.90070 | 0.863062 | 0.558362 |
| 내부 context PCFI | 49.41351 | 0.863255 | 0.558625 |
| 내부 label-blind PCFI | 49.69055 | 0.863001 | 0.558301 |
| 보조: uniform | 42.32140 | 0.888307 | 0.574219 |
| 보조: context_frequency | 45.99776 | 0.866755 | 0.560505 |

| Paired 비교 | Accuracy 차이 %p [95% CI] | NLL 차이 [95% CI] |
|---|---:|---:|
| context-external | +0.17025 [-0.55767, +0.86604] | -0.00387 [-0.01012, +0.00196] |
| context-internal_fixed | -0.48718 [-0.86182, -0.13011] | +0.00019 [-0.00140, +0.00161] |
| context-label_blind | -0.27704 [-0.70496, +0.17476] | +0.00025 [-0.00128, +0.00171] |

보조 위치 비교 internal-fixed−external: accuracy+0.65743%p CI[-0.11984,+1.37358], NLL-0.004065 CI[-0.010835,+0.002168]. 이 보조 분석은 primary를 대체하지 않는다.

CI는 base graph5000회 paired bootstrap이다. Query/node를 독립 표본으로 늘리지 않았다. Training seeds는 고정 반복이며 seed 모집단 uncertainty를 추정하지 않았다. Subgroup/보조 CI는 다중 비교 미보정이다. CI에0을 포함한 것은 무효·동등성의 증명이 아니다. 두 비학습 prior는 design lock 이후 추가로 재채점한 보조 대조이며 primary를 교체하지 않는다.

## 조건별 결과

| 조건 | External % | Internal-fixed % | Context % | Label-blind % | Context−external %p [CI] |
|---|---:|---:|---:|---:|---:|
| full | 53.62956 | 54.13411 | 53.95508 | 53.98763 | +0.32552 [-0.19531, +0.84635] |
| uniform | 49.23928 | 49.80654 | 49.24763 | 49.64635 | +0.00835 [-1.04766, +1.00764] |
| clustered | 49.24725 | 49.99485 | 49.57939 | 49.73476 | +0.33214 [-0.64186, +1.23944] |
| cell | 51.61333 | 50.97677 | 50.58957 | 51.44487 | -1.02376 [-2.90782, +0.75860] |
| k1 | 41.17709 | 41.32767 | 41.42379 | 41.61910 | +0.24670 [-1.34127, +1.73945] |
| uniform_nodes_0.9 | 50.19619 | 50.54551 | 50.31720 | 50.51182 | +0.12101 [-1.20081, +1.35708] |
| uniform_nodes_0.995 | 48.28237 | 49.06758 | 48.17806 | 48.78087 | -0.10431 [-1.54572, +1.32011] |
| clustered_nodes_0.9 | 50.84193 | 52.02001 | 51.54979 | 51.68201 | +0.70786 [-0.53197, +1.90962] |
| clustered_nodes_0.995 | 47.65257 | 47.96970 | 47.60900 | 47.78751 | -0.04358 [-1.44442, +1.24545] |

Full은 전체 query, 나머지는 자기 피처 없는 query다. Primary 밖의 k1/cell/개별 mask 결과를 사후 main 성공으로 승격하지 않는다. 모든 조건의 전체 query 지표도 statistics.json에 별도 보존했다.

## 비용

| 방법 | CPU 입력→CPU 확률, ms |
|---|---:|
| 외부 PCFI → PFN | 56.944 |
| 내부 고정 PCFI | 63.578 |
| 내부 context PCFI | 76.948 |
| 내부 label-blind PCFI | 68.752 |

우리 학습 worker 종료 후 같은RTX3090, 사전4조건 각각5 warmup/30회 median의 평균이다. 거리/confidence 전처리와 실제100회 전파,1회 correlation,1회 PFN 및 CPU 확률 반환을 포함했다. 모델 로딩과 bank I/O는 제외했고 외부 GPU 부하는 통제하지 않았다. Kernel만의 본질적 속도나 큰 그래프 메모리 우위로 해석하지 않는다.

## 학습·검증

- 새12모델, 성공 updates3072, exposures3072. 동일64 train graphs×4 settings=256 episodes를 seed별 한 번씩 같은 순서로 학습했다. CE만, 마지막 checkpoint만 평가했고 target fitting은 없다. Dev optimizer4 steps는 폐기했다.
- Trainable counts: {'external': 4012259, 'internal_fixed': 4012259, 'context': 4012772, 'label_blind': 4012772}. 모든 모델의 common/frozen 초기값과 task schedules가 같고 frozen backbone은 유지됐다. Context/blind는 정확히 같은 수, fixed/external보다513개 많다.
- Dev 작은 그래프에서 Torch/SciPy PCFI 차이 최대4.43e-7. 실제 dev28 episodes의 FP32 kernel/cache 차이는 최대8.54e-5였다. Float16 public PFN의 node permutation 최대 차이는0.00244 이내였으므로 전체 모델의 exact permutation equivalence를 주장하지 않는다.
- 최초 GPU atomic scatter 구현은 같은 입력 반복에서도 FP16 최종 확률이0.000976 변했다. Final 생성 전 수치 연산을 같은 수학의 dense row-normalization/bmm로 바꾸고 dev 전체 검사를 재실행했다. 이 수정은 학습·최종 성능에 따른 tuning이 아니다.
- Hidden-value poison, 모든 관측값 clamp, class-name invariant router statistics, strict label-blind router, nonzero router/hidden-state gradient, 실제 optimizer helper와 trainable checkpoint 저장/복원을 검증한 뒤 설계를 잠갔다. Learned arm 초기 예측은 internal-fixed와 정확히 같았다.
- 모든 source/config/bank/checkpoint/prediction hashes와 저장예측 재채점, 36개 checkpoint replay를 확인했다. Primary256 episodes 중 실제 관측 cell이0개인 episode24개도 제외하지 않았다.

## 해석 범위와 판정

같은 size-restricted official graph/SCM generator family의 새 graph seeds다. 실제 graph, unseen semantic schema, 원 논문의 공식 downstream classifier 재현이 아니다. 공통 observed-only RMS/native-pair/mask interface가 공식 전처리 전체와 같다고 주장하지 않는다.

Context가 external·internal-fixed·label-blind 모두를 primary에서 이겨야 학습된 조절의 증분을 인정한다. Fixed internal만 external을 이기면 배치 위치 효과이고, context가 fixed만 이기고 external은 못 이기면 전체 파이프라인 개선은 아니다. 실패 시 이 bank에서 새 loss/head/반복 횟수/학습량으로 구제하지 않는다. 원인이 분리되지 않으면 미상으로 남긴다.

Row-stochastic 전파는 상수 feature를 그대로 보존한다. 관측0개인 column에서0으로 시작하면 가중치 조절만으로 새 신호가 생기지 않는다. 이는 연산의 알려진 제약이며 전체 실험 실패의 단일 원인으로 확인한 것은 아니다.

원 수치는 [statistics.json](results/statistics.json), [cost.json](results/cost.json), [audit.json](results/audit.json), 정확한 설계는 [protocol.json](protocol.json), 계산 정의는 [METHOD_KO.md](METHOD_KO.md)를 확인한다.

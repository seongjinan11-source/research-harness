# FP05 결과의 수학·추론·구현 검토

2026-11-05. **실행·저장 정합성은 감사 범위에서 PASS, 사전 efficacy 네 항목은 FAIL, full-natural guard는 PASS다. 현재 후보에서 유용한 적응형 배포 경로를 학습했다는 증거가 없다.** Direct/group_pg의 모든 final·real greedy 출력은 각자 같은 체크포인트의000 replay와 비트 단위로 같았다. 이 결과로 일반적인 RL, Transformer 전체 또는 더 넓은 action family의 가능성을 부정할 수는 없다.

검토자는 `rl_model.py` 구현, 사전 추론 검토 및 별도 deployment audit 작성에 참여했다. 외부 blind review가 아니라, 잠긴 실행과 저장 결과에 대한 내부 교차 검토다. 이번에는 PLAN_v2, 실제 model/runner, REPORT/summary, 두 감사 결과, 입력 감사·prelock delta를 읽었다. 추가 작업은 저장 progress/calibration/evaluation JSON·NPZ를 NumPy로 읽은 CPU 진단뿐이다. 모델 import/forward, GPU, 새 fit, 과학 source·bank·lock·checkpoint·기존 prediction 수정은 없었다. 새 진단 JSON은 create-only로 저장했다.

## 사전 판정과 추정 단위

실제12 fits, 1,536 successful optimizer groups, 30,828 saved prediction records가 완료됐다. Record 수를 독립 graph 또는 독립 query 수로 해석하지 않는다. Primary는96 base graphs 각각에서 고정3 training seeds와 유효 node95/995 rates를 먼저 평균하고,12 strata 안에서 graph 단위 paired bootstrap5,000회를 수행한 결과다. 모두 두 rates가 유효했다. Utility는 별도의48 disjoint graphs/58 eligible graph-rate cases를 graph 안에서 먼저 평균한 조건부 추정량이다.

| 사전 비교 | 정확도 차이 %p | 양측 nominal97.5% CI %p | 판정 |
|---|---:|---|---|
| group_pg − fixed | −0.4974 | [−1.0336, +0.0121] | FAIL |
| group_pg − direct | +0.0172 | [−0.2409, +0.2748] | FAIL |
| PG observed − none, utility | −1.0634 | [−5.5667, +4.0693] | FAIL |
| PG observed − shuffle, utility | −0.2604 | [−2.1267, +1.5842] | FAIL |
| Full-natural PG − fixed | +0.1302 | [−0.8247, +1.0417] | guard PASS |

첫 비교는 우월성 실패이지, 이 CI에서 열등성이 확정됐다는 뜻은 아니다. Guard는 하한>−2%p라는 제한된 조건을 통과했으며 efficacy를 대신하지 않는다. 네 CI가 동시에95% coverage라는 주장도 하지 않는다.

동일한 primary own-missing subset·graph weighting으로 독립 재계산한 metric은 다음과 같다. Production `final_secondary`의 전체 query 평균과 혼합하지 않았다.

| Arm | 정확도 % | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|
| fixed | 49.2904 | 0.845051 | 0.555849 |
| uniform8 mixture | 49.4750 | 0.845582 | 0.555509 |
| direct | 48.7759 | 0.845467 | 0.556608 |
| group_pg | 48.7930 | 0.845406 | 0.556633 |

정확도 이외 metric이 PG의 주장을 구제하지 않는다. Uniform의 fixed 대비+0.1845%p는 기술적 관찰이며8-forward mixture의 사전 PG gate 성공이 아니다. PG의 feature-utility difference-in-differences도 none 대비−0.3906%p [−1.8446,+0.9766], shuffle 대비−0.4557%p [−1.5191,+0.5425]로 추가적인 feature 활용을 입증하지 못했다.

## Greedy000과 정책 분포 붕괴는 다르다

`research_cycles/fp05_routing_diagnostic.py`는 evaluation record의 physical episode ID로 main/replay000을 쌍짓고, 파일 SHA 및 y/query/context/yc 일치를 확인한 뒤 저장 p를 직접 비교했다. 두 arm 모두 seed11/12/13을 포함한다.

| Arm | Final main000/전체 | Real main000/전체 | Main↔replay000 bitwise exact |
|---|---:|---:|---:|
| direct | 3,168/3,168 | 135/135 | 3,303/3,303 |
| group_pg | 3,168/3,168 | 135/135 | 3,303/3,303 |

총6,606 쌍에서 최대 확률 차이0, TV0, argmax 변경0이었다. NumPy 기본 allclose 및 rtol0/atol1e−7·1e−6도 모두 통과했다. Final에는11 settings/interventions, real에는 모든45 rows/seed가 포함돼 있어 primary subset만의 우연한 동일성에 그치지 않는다. 따라서 현재 greedy 배포에서 추가 conv+MLP는 실제로0회였고, PG−fixed 또는 PG−direct 차이는 **각 arm의 adapter 학습 결과 차이**로 남는다. 동일 체크포인트에서 non000 routing이 만든 배포 증분은 관찰되지 않았다.

그러나 logit이 세 stage에서 모두0 이하이면 greedy000이며, 각 action0의 확률이 조금만0.5를 넘어도 이 현상이 생긴다. 결합확률이1에 가깝다는 뜻은 아니다. 후기16 logged training groups의 결과는 다음과 같다.

| Arm/seed | Entropy 또는 MC self-information, nats | p000 또는000 표본비율 | 평균 unique traces/8 |
|---|---:|---:|---:|
| direct11 | 1.9502, exact entropy | p000 0.2691 | 8, 전수열거 |
| direct12 | 1.9097, exact entropy | p000 0.2962 | 8, 전수열거 |
| direct13 | 1.9263, exact entropy | p000 0.2854 | 8, 전수열거 |
| group_pg11 | 2.0096, MC −mean(logπ) | 표본비율0.2656 | 5.125 |
| group_pg12 | 1.8535, MC −mean(logπ) | 표본비율0.3594 | 4.625 |
| group_pg13 | 1.7954, MC −mean(logπ) | 표본비율0.3672 | 4.375 |

세 비트 최대 entropy는 ln8=2.0794 nats다. Direct는 저장된8 leaf probabilities로 −Σp logp를 계산했다. 그 unique8은 열거 설계이므로 탐색의 증거로 쓰지 않는다. PG는8개 on-policy 표본의 −mean(logπ)이며 exact entropy나8-bin empirical entropy가 아니다. 개별 Monte Carlo 값은 ln8보다 클 수도 있다. PG의 다양한 실제 표본과 direct의 p000/entropy는 **기록된 후기 학습 분포가000에 완전히 붕괴했다는 설명과 맞지 않는다.** 두 arm의 φ gradient norm도 이 구간에서 계속 양수였고 PG zero-advantage group은0이었다.

이 로그는 update128도 optimizer step 직전 상태다. 최종 checkpoint의 evaluation-state action logits/conditional probabilities는 저장되지 않았으므로, 최종 분포의 exact entropy·p000까지 측정했다고 주장하지 않는다. 현재 확정할 수 있는 것은 모든 평가 입력의 greedy 선택000과 후기 학습 분포의 다양성이다.

## 비용만을 원인으로 단정할 수 없다

실제 global calibration 선택은 사전대로 **평균 CE**이며 cost는 동률 후순위였다. 별도 post-result 진단에서 각 saved branch NPZ를 독립 재채점하고, 추가로 CE+0.01×trace cost의 argmin도 계산했다. 이는 새 배포 선택이나 gate 변경이 아니다. 각 fit에는 dev8 bases×7 settings=56 episodes/trace가 있다.

| Calibration fit | 실제 mean-CE best | 진단 CE+cost best | (CE+cost)000 − 최선 |
|---|---|---|---:|
| uniform11 | 110 | 110 | 0.015442 |
| uniform12 | 010 | 010 | 0.017592 |
| uniform13 | 110 | 010 | 0.002285 |
| group_pg11 | 110 | 110 | 0.012729 |
| group_pg12 | 110 | 010 | 0.026654 |
| group_pg13 | 110 | 110 | 0.008804 |

여섯 fit 모두 비용을 넣어도 non000이 최선이다. 반복 연산이 모든 상황에서 무효이거나 현재 λ가 항상000을 최적으로 만든다는 설명은 이 dev 결과로 지지되지 않는다. 동시에 작은 dev panel의 global post-training 평균은 on-policy training episode별 목적식이나 전체 최적값을 대신하지 않는다. λ0 대조도 없어 비용의 기여 자체를 배제할 수 없다. PG의 dev-selected global control은 final primary 정확도48.7952%로 greedy48.7930%와 거의 같았다. Non000 dev best가 final accuracy 개선을 보장하지도 않았다.

## 수학·구현 오류와 과학적 실패의 구분

잠긴 코드의 direct adapter 목적식은 Σ stopgrad(p_a)L_a, actor는 Σp_a stopgrad(L_a+0.01c_a)다. p는 이미 합1이므로 추가/8이 없으며, detached14-state에서 φ에 대한 전수 기대목적식이다. PG는8개의 별도 NumPy RNG action 표본에 대해 mean CE와 −mean(A logπ)를 사용한다. Group 표준편차에 의존하는 정규화는 direct actor의 단순 불편 추정량과 같지 않다. Exact expectation보다 PG가 좋아야 한다는 수학적 보장은 없다. 둘 다 entropy coefficient0이며, stochastic expected-loss 학습과 deterministic greedy 배포의 목적 차이는 실제 존재한다. 이것이 관찰된 실패의 단독 원인이라고 확정할 대조는 없다.

사전 checks는 original000 forward/adapter-gradient parity,8-leaf/prefix 확률, actor gradient, RNG 분리, AMP/실제 discarded optimizer step, class-coordinate 처리와 save/load를 검증했다. 별도 deployment audit는 모든12 checkpoint 내부 arm/seed/lock/schedule,148 trainable tensor keys의 shape·FP32·finite, public frozen digest,128개 실제 progress와 optimizer 회수, dev criterion, 배포 variant/reference matrix와 checkpoint association을 확인했다. Logged direct actor와 PG reward/actor도 허용 FP32 오차 내에서 재계산됐다. 별도 statistics audit는 saved probability의 metric과 CI를 독립 식으로 재현했다. 현재 결과를 오류로 무효화해야 할 새 계산 결함은 발견하지 못했다.

이것은 모든 실제 forward를 재실행해 증명했다는 뜻이 아니다. Per-update hidden state/old-policy checkpoint가 없어 logged conditional probabilities 전체를 사후 재생할 수 없다. Calibration record 내부에는 checkpoint SHA가 없으나 immutable fit folder/training checkpoint와의 연결은 확인했다. Upstream의 고정 label embedding/feature-position 특성 때문에 summary의 equality-group invariance를 전체 pretrained predictor의 임의 class/column equivariance로 확대하지 않는다.

최소 원인 범위는 **현재 세 위치의 동일 adapter 반복 연산, detached14-summary controller,128-group co-training, 주어진 prior·PCFI 표현, greedy readout의 조합에서 유용한 조건별 추가 연산을 배포하지 못했다**는 것이다. Frozen φ, 항상 막힌 action1 또는 전체 Transformer를 제거한 대체 모델이라는 설명은 source/checks·활성 actor gradient·non000 dev branch 차이와 맞지 않는다. Summary의 정보 부족, 학습예산, 연산 표현력, 분포 차이, cost 편향 중 무엇이 지배적인지는 분리되지 않았다. 이 결과 뒤 같은 bank에서 λ/entropy/threshold를 골라 gate를 뒤집는 것은 적절하지 않다.

## 보존할 성과와 주장 한계

실제 released GraphPFN12 wrappers와 LimiX16M frozen backbone을 보존하고, adapters-only CE와 controller policy gradient를 분리해12 fits를 완료했다. 이는 작은 새 network를 pretrained Transformer로 부른 실험과는 구별되는 구현 성과다. Fixed CE는 frozen projected-PCFI000 대비 primary 정확도+0.5021%p, NLL0.869112→0.845051이라는 기술적 개선을 보였다. 별도 유의성 검정 없이 이 값을 성공 gate로 승격하지 않는다. Context prior의 NLL0.826377은 모든 trained arm보다 낮아서 전반적 calibration 우월성도 주장할 수 없다.

Reused real development의 node95 PG 정확도는 Cora45.459%, CiteSeer28.334%, PubMed61.671%였다. Fixed는45.891/29.354/62.323%, LP는59.056/38.410/73.690%였다. PG의95−none 차이는 각각+21.13/+11.18/+24.77%p여서 이 pretrained+PCFI pipeline이 feature 입력에 의존한다는 관찰은 남는다. 그러나 모든 PG main이 replay000과 같아 이를 routing 기여로 부를 수 없다. Fixed/frozen도 feature를 사용한다. 이 세 graph는 반복 사용한 개발 자료이며 seeds×repetitions를 독립 graph9개로 세지 않는다.

기술적 증분은 기존 PFN 위의 제한된 compute controller를 정확히 비교·감사한 pilot과 그 음성 결과다. 새로운 RL 알고리즘, 새로운 imputer, learned posterior mechanism inference, heterogeneous/schema 전이 또는 missing-robust PFN의 입증으로 제시할 수 없다. 현재 action/operator/budget 후보는 종료하는 판정이 타당하며, Transformer 전체나 더 넓은 정책학습의 실패라는 결론은 범위를 벗어난다.

## 입력·provenance·비용의 남는 한계

- Actual final은 N112…256/F8…16, K2/3/4 중69/96이 binary다. Heldout shape는 actual train 범위 밖이지만 schema holdout은 아니다. Official prior의 내부 class/feature sanity filtering은 남아 있고 내부 rejection 총수는 미상이다.
- Node995의76/96은 source0/1이다. 이를 유지한 primary와 source≥2인48-graph utility를 구분했다. Utility58 cases 중8개의 shuffle 입력은 원래와 같으며, 사후 제외/재추첨하지 않았다. Feature-value utility의 검정력 한계지만 gate 실패를 삭제할 이유는 아니다.
- Real은 visible native rows→고정 RP16→observed RMS→PCFI100→clip±10이다. Native F3703 전체 network 처리나 native historical public PCFI와 동일 표현·budget 실험이 아니다. 과거 native reference와의 차이를 RL 효과로 귀속하지 않는다.
- Train/dev/real preparation manifest가 기록한 과거 `rl_util.py` SHA는 `af7dba0d…f5c80d`, 잠긴 현 source는 `4ee7f2bf…164e18`이다. 과거 정확한 bytes가 보존되지 않아 “source enumeration만 바뀌었다”는 byte-exact 주장을 입증할 수 없다. 다른 기록 dependency는 일치하고 final preparation 및 scientific fits는 현 design/full lock 아래 수행됐다. 입력 identity·전처리 독립 감사와 원 manifest 보존은 완화 조치이지 과거 source gap의 소급 복구가 아니다.
- Logical slots12,288은 equal FLOPs를 뜻하지 않는다. 성공 학습 forward/backward9,600회, calibration2,688 forward, deployment51,747 forward로 감사 합계64,035 forward이며 training retry는0이다. 준비·PCFI·calibration을 별도 포함했고 PCFI를 중복 합산하지 않았다. 약6,974s component wall 합계에는 concurrent CPU audit/hash contention이 있을 수 있고 LP/prior timing·checks/smoke는 완전 포함되지 않는다. 통제된 latency benchmark로 주장하지 않는다.

## 재현 경로와 고정 식별

추가 진단 실행은 다음 한 번의 CPU 명령이었다. 기존 scientific artifacts는 읽기만 했으며 script가 입력·source SHA의 전후 일치를 검사하고 결과를 `open('x')`로 썼다. 진단의 TV는 record 동일 weight, argmax 변화는 query row count이며 어느 것도 CI의 graph 표본 수를 늘리지 않는다.

```powershell
& 'C:/Users/seongjin/Desktop/csag_share/research/www2027/results/graph_pfn_suite_20261016/windows_graphpfn_env/Scripts/python.exe' -B -X utf8 research_cycles/fp05_routing_diagnostic.py
```

| Artifact | SHA256 |
|---|---|
| results/lock.json | `18b27b3d9a4eb3c8fc321824bf32407ae8f888d27c4e2de4a10b66eba325c575` |
| rl_model.py | `db0b860547574658e0b67d17a9ab1d30687f3291902734fc6de929f5cd309c80` |
| results/summary.json | `9dc5546f6cd8eaf35880d905c0f59cc420579286fb6c1081925a92c57429b6ad` |
| results/deployment_audit.json | `c899cf4bc5d19fe0ea568aefc03b71010fa731f3c4e3d76bd486e0db05b94a1a` |
| results/statistics_audit.json | `7761a2a639b6a36ab7d7e3e33de6ae242d23e7c66c75bb16fe1f61c69f59dad8` |
| research_cycles/fp05_routing_diagnostic.py | `39031b7e8c8eedf490f0c1e1f57f744556dea8df46c4e272c11923fec2389f04` |
| results/routing_diagnostic.json | `2bb4297025725bfa06bbdc7a59383d83472e661330f4cd6feca96dfe71bfbd7e` |

전수 metric/CI의 별도 식과 tolerances는 deployment/statistics audit script·JSON에, routing pair·training window·dev branch rescore의 입력 hash와 계산은 새 diagnostic script·JSON에 보존돼 있다. 이번 리뷰로 잠긴 점수·gate·선택 trace를 변경하지 않았다.

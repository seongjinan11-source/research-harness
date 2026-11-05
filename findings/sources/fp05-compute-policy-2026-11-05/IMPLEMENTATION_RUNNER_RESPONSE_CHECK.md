# FP05 runner 수정 응답 확인

2026-11-05. 현재 `rl_run.py`, `rl_analyze.py`와 연결 계약을 읽기 전용으로 재검토했다. 결과 prediction/score/progress는 열지 않았다. 과학 source·bank·design/lock·모델은 수정하지 않았고 GPU/모델 forward도 실행하지 않았다. 새로 작성한 파일은 이 문서뿐이다.

**판정: 기존 지적의 핵심 수정은 반영됐다. θ/φ 목적식이나 현재 학습을 바꿔야 할 새 오류는 발견하지 못했다.** 다만 분석의 empty-secondary 경계와 감사 범위는 아래처럼 한정해야 한다. Source 수정 대신 우선 현재 입력·결과의 독립 audit에서 발생 여부를 확인하는 것이 적절하다.

## Source anchor

다음 실제 SHA256이 design과 scientific lock 양쪽 sources 항목에 같은 값으로 들어 있음을 작은 CPU 파일 검사로 확인했다.

```text
rl_run.py       d0926d0501b104e6fa24e9b1bd6502b1674c9136b4a6badf89ef89307b73d877
rl_analyze.py   03343d1972ae2b65a06551d04eee98b43edfd232f5abcf4ab77d81dd4184fb04
rl_model.py     db0b860547574658e0b67d17a9ab1d30687f3291902734fc6de929f5cd309c80
design.json     8d790c1a3c962c47c6fc01ae60433bfeaff5302e0607f9fb7483a850d12900fe
lock.json       18b27b3d9a4eb3c8fc321824bf32407ae8f888d27c4e2de4a10b66eba325c575
```

이는 source 고정을 확인한 것이며, 미래 checkpoint/예측이 모두 적절하다는 사전 판정은 아니다. Root가 보고한 checks/dev-smoke PASS는 여기서 재실행하지 않았다.

## 네 지적의 조치

| 지적 | 현재 조치 | 확인 |
|---|---|---|
| Dev criterion에 불필요한 cost | 평균 NLL만 누적하고 cost는 exact tie의 후순위로만 사용 | 수정 적절 |
| Dev8-branch probabilities 미저장 | episode×trace별 p/y/query/context/yc NPZ, input/prediction SHA 및 selection index 저장 | 수정 적절; 최종 독립 재검산 필요 |
| Retry NaN/Inf JSON 실패 | CE를 null로 바꾸고 ce_finite flags를 보존 | 지적한 직렬화 경로 수정됨 |
| Canonical bank/coverage/schedule 부족 | y/query/context/yc/standard_own_missing를 input NPZ에 직접 대조; row metadata/input SHA, variant별 unique ID coverage, RNG schedule 재생과 progress episode/epsilon 대조 | 핵심 수정 적절; 아래 provenance 잔여 감사 필요 |
| Empty primary | Caller가 values 없음에 n_graphs0/UNDEFINED를 기록하고 primary accuracy None 허용 | Primary gate 자체는 수정됨; secondary 경계는 아래 참조 |

Direct의 θ loss는 여전히 p.detach×CE이고 actor는 Σp×detached(CE+cost)로 추가/8이 없다. Group PG는 별도 action RNG와 group 이후 한 optimizer step을 유지한다. CE backward와 actor graph의 분리, AMP unscale 후 θ/φ 별도 clipping, common K 좌표와 standard-own-missing subset도 변경되지 않았다. 수정 과정에서 새로운 training-gradient 문제를 발견하지 못했다.

## 남아 있는 한정 사항

1. **Empty secondary 처리:** `interval()` 자체는 여전히 empty에서 assert한다. Primary caller는 이를 피하지만, 후반 `direct/group_pg_routing_minus_replay000`은 값이0개일 때도 interval을 호출한다. 따라서 모든 primary-own-missing case가 없는 이론적 경계에서는 UNDEFINED primary를 만든 뒤 secondary에서 report가 중단될 수 있다. 실제 final bank에서 그런 상황이 발생했다고 주장하는 것이 아니다. 최종 audit에서 available graph/rate 수를 먼저 확인하고, 발생 시 별도 analysis-only repair 기록으로 처리해야 한다. 학습을 수정하거나 bank를 다시 뽑을 이유는 아니다.

2. **새 secondary는 primary와 다른 estimand:** 추가된 `final_secondary`는 setting/intervention별 **전체 query** metric 평균이다. Primary는 own-missing query만의95/995 값을 base 안에서 평균한다. 둘을 같은 NLL/Brier/BAcc/macroF1 집계라고 부르면 안 된다. Final review에서 primary와 동일한 subset·graph weighting의 secondary metric을 저장 배열로 재구성하면 source 변경 없이 이 차이를 명확히 할 수 있다.

3. **Calibration/deployment 감사는 아직 별도다:** `rl_analyze`는 `global_trace.json`/dev calibration NPZ를 직접 검증하지 않는다. Expected variant 집합도 “존재하는 variant의 coverage”만 검사하며 missing variant 전체를 항상 검출하는 구조는 아니다. 예컨대 `global` control이 통째로 빠지는 경우를 별도 검사해야 한다. Group_pg global control은 locked source에 있지만 V2의 uniform-only global 설명보다 넓으므로 사전 source에 있던 추가 secondary로 표시하고 primary와 혼합하지 않는다.

4. **비용 합계 범위:** 현재 budget의 evaluation 합계에는 calibration과 bank의 projection/RMS/PCFI 준비 비용이 모두 자동 합산되지 않는다. 이 비용은 각각 global_trace metadata와 bank manifests에 있으므로 최종 audit에서 별도 항목으로 더해야 한다. 실패 retry의 forward/backward 비용도 successful group count와 구분한다.

## 다음 결과 audit의 최소 확인 항목

- **Checkpoint 내부:** CPU load로 내부 arm/seed/lock_sha/schedule_sha를 training.json·폴더와 대조한다. Parameter key 집합이 정확한 adapter whitelist+policy인지, shape/dtype/finite 여부, 초기 public/frozen identity 및 최종 tensor digest가 맞는지 확인한다. 단순 checkpoint 파일 SHA 일치와 내부 state 정합성을 구분한다. 현재 evaluation `load()`의 초기/최종/frozen digest 검사는 좋은 선행 방어다.
- **Actual schedule/exposures:** Seed별 schedule을 독립 RNG로 재현하고12 fits의 공유 episode/epsilon을 확인한다. Progress의 update1..128 순서, episode/input SHA/epsilon, record 수(고정1, 나머지8), trace/cost, direct probability 합1 및 expected CE/actor, PG reward population-SD/advantage-zero를 기록에서 재구성한다. Action seed는 schedule에 있고 runner가 직접 전달하지만 progress에 복제되지는 않으므로 이를 별도 association으로 보고한다. 성공/실패 retry와 논리 slots/실제 calls를 분리한다.
- **Dev selection:** Uniform/group_pg 각각 seed별 calibration이 canonical dev episodes×8 traces를 한 번씩 포함하는지 검사한다. Saved p/y/identity/hash에서 mean CE와 tie-break(lower cost, lexical)를 독립 재계산하여 선택 trace를 확인한다. Folder의 trained checkpoint/arm/seed/source lock와 연결하고, 같은 frozen 선택이 final과 real에 쓰였는지 대조한다. 현재 calibration record에 explicit checkpoint SHA가 없다는 점은 파일 위치·training artifact association으로 보완하여 감사 범위를 명시한다.
- **Deployment matrix:** Fixed={main}, uniform={main,global}, direct={main,replay000}, group_pg={main,replay000,global}가 각 canonical row를 정확히 한 번씩 포함하는지 확인한다. References는 frozen_pcfi/frozen_nan/prior/lp 각각 전체 rows를 확인한다. Record의 checkpoint SHA가 해당 fit과 같아야 한다. replay000은000, global은 dev 선택 trace, uniform main은8 traces 전체이며 평균 probability를 사용하는지, cost/forward 수가 맞는지 대조한다. Greedy route가 final label을 사용하지 않는 source 경계는 유지된다.
- **Metric/CI와 historical scope:** Runner와 같은 `metrics()` 재호출만으로 끝내지 말고 별도 NumPy 식으로 rescore한다. Canonical standard-own-missing mask와 eligibility를 사용해 fixed seeds→valid rates→base 평균 및 strata bootstrap을 재현한다. Full-natural own-observed guard, utility denominator/strata, replay/DoD의 동일 query subset을 확인한다. Native historical references는 기존 hash/identity lineage와 다른 RP16 representation이라는 한계를 함께 기록한다.

이 항목은 결과를 본 뒤 설계를 고치는 제안이 아니라 이미 고정된 계산과 배포가 실제 artifact에 반영됐는지 확인하는 감사다. 현재 학습 source/bank/locks를 바꿀 근거로 사용하지 않는다.

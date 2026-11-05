# FP05 independent correctness checks

2026-11-05. 소유 파일은 `rl_checks.py`와 이 문서다. PLAN_v2 및 모델 agent의 최종 API에 맞췄다. 이 작성 단계에서는 **GPU·Model 구성·실제 학습·train/dev/final/real bank 읽기를 실행하지 않았다.** CPU toy와 standalone policy-statistic helper만 실행했다. 실제 공개 모델 검사는 root가 실행한다.

## 실행/receipt 계약

- `rl_checks.run(device='cuda', toy_only=False)`는 JSON-serializable dict를 반환한다. 실패도 status=FAIL과 traceback, 실행 진행/폐기 optimizer count를 반환하며 성공처럼 취급하지 않는다. 성공적인 full run만 status=PASS다.
- CLI 기본 출력은 `results/checks.json`이다. `rl_util.dump`의 exclusive-create를 사용하므로 기존 PASS/FAIL을 덮어쓰지 않는다. 기존 경로가 있으면 검사를 시작하기 전에 실패한다. Retry는 명시적인 새 `--output` 경로를 사용하고 root가 이전 실패와 함께 보존한다.
- `--toy-only`는 CPU helper만 검사하고 `TOY_PASS`를 쓴다. 기본 경로도 `checks_toy.json`으로 분리해 scientific lock의 PASS receipt를 대체하지 못한다. 이번 작성자는 CLI를 실행하지 않고 함수의 CPU 결과만 메모리에서 확인했으므로 결과 receipt 파일은 만들지 않았다.
- Full run은 `rl_util.sources()`를 검사 전후 비교하며 upstream commit/checkpoint hash도 모델 결과에 기록한다. Config/plan/checks/model/util의 source가 실행 중 바뀌면 실패한다. 실제 frozen parameter digest는 폐기 optimizer step 전후 비교한다.

## 독립 CPU oracle

공유 linear policy와 prefix-dependent 6-dimensional toy state로 3단 binary tree를 구성했다. Production loss/helper를 호출하지 않고 8개 trace를 NumPy product와 Torch log-probability로 별도 계산한다. Exact expected `(CE + .01 cost)` gradient를 finite difference와 detached-weight score-function 합에 비교한다. Separate scalar Bernoulli toy에서는 더 좋은 reward의 action 확률이 gradient descent로 증가하는지, reward가 detach되는지, 같은 reward 8개의 population-standardized actor gradient가 정확히 0인지 검사한다.

실제 수행한 CPU float64 결과:

| 검사 | 결과 |
|---|---:|
| 8-leaf probability sum | 1.0 |
| NumPy/Torch probability max error | 5.55e−17 |
| Direct vs score-function gradient max | 0 |
| Direct vs central finite difference max | 5.62e−11 |
| Rewarded action logit gradient | −0.4999995 |
| 해당 action 확률, one toy gradient step | .5 → .5124974 |
| Equal-group actor loss/gradient | 0 / 0 |

Standalone `rl_model.policy_state`는 NumPy로 독립 구현한 14-statistic oracle와 비교했다. Population component variance, zero-norm context-pair 제외, unordered same/different pairs, validity flags, static observed/degree/count 통계를 포함한다. Class renaming과 context/noncontext block 내부 node 순열만 exact invariance로 검사하며 전체 pretrained model의 class equivariance를 요구하지 않는다.

실행 결과: statistic oracle max error 2.24e−8, class renaming 0, block-node permutation 1.12e−8, empty context finite, zero-norm same/different pair flags 0. Model instance는 만들지 않았다.

## Root가 실행할 실제 모델 검사

N=6/F=4/K=2의 고정 finite fixture를 사용한다. 두 context, 두 query, self edges와 두 chain component가 있다. `values`는 이미 표현된 PCFI 근사값이므로 M=0 위치도 합법적으로 nonzero다. Query targets는 입력 dict 밖의 CE test에만 있다.

1. `Model.original(inp)`와 forced000 output/adapter gradients를 동일 forward seed와 autocast에서 비교한다. Same connected-gradient pattern을 요구한다. Paired000의 절대 tolerance는 score 2e−3, adapter gradient abs3e−3/relative1e−2로 기록하며 실제 오차도 저장한다. 이는 FP16 수치 tolerance이고 exact bitwise equivariance 주장은 아니다.
2. Initial zero binary logits의 greedy tie가 000인지 확인한다. 정책 마지막 weight를 deterministic하게 잠시 바꿔 nonuniform 8-leaf tree를 만들고 logp의 conditional Bernoulli product, leaf probability sum, 같은 prefix state 일치, cost/trace/action-call metadata를 검사한다. 최소 한 trace의 예측 변화가 있어야 한다. 이는 operator 작동 검사이며 utility gate가 아니다.
3. Exact leaf actor loss의 policy gradient가 존재하면서 adapter gradient가 없고, forced-trace CE는 adapter gradient가 존재하면서 policy gradient가 없는지 검사한다. 모든 backward 전후 hook-call count가 같아 checkpoint recomputation이 action을 다시 만들지 않는지도 확인한다.
4. FP32 trainable parameters, FP16 autocast, 실제 AdamW lr(adapter3e−5/policy1e−3), weight_decay.01, GradScaler64, 각 group clip1로 **한 폐기 optimizer step**을 실행한다. Synthetic positive actor advantage를 사용하는 numerical smoke이며 실제 RL reward 성능이 아니다. 두 parameter group이 step 직전 digest에서 실제 바뀌고 finite하며 frozen weights/gradients는 유지돼야 한다. 시도/성공 step count는 뒤 검사 실패 시에도 receipt progress에 남긴다.
5. `trainable_state`를 메모리 BytesIO에 실제 serialize/deserialize하고 parameter를 잠시 바꾼 뒤 복원한다. Adapter/policy digest와 forward replay를 확인한다. 파일 checkpoint를 생성·수정하지 않는다.
6. 별도 NumPy Generator의 uniform-policy action sequence가 global Torch feature-RNG pairing에 재설정되지 않는지 확인한다. Context에 없는 클래스의 head-coordinate mapping과 남은 mass의 exchangeability도 검사한다(K4, context0/2 → head order0/2/1/3).
7. 공개 PFN의 class/column/node order sensitivity는 probability 차이를 **diagnostic으로만 기록**한다. Upstream의 fixed target/random feature embeddings에 대해 큰 exact equivariance를 요구하지 않는다. Policy summary 자체의 renaming invariance는 별도 strict test다.

## 명시적인 경계

- `values[~observed]`를 poison하고 동일 출력이기를 요구하면 안 된다. 이 값은 PCFI가 계산한 합법적 표현이다. **숨긴 raw truth를 preprocessing 전에 poison하는 검사와 query-target input 누출은 data pipeline/root의 별도 책임**이다. Receipt에 이 경계를 명시한다.
- CPU toy가 맞아도 실제 GraphPFN hook/gradient/parity는 미검증 상태다. Full status=PASS는 root 실행 이후만 주장할 수 있다. 은행 생성의 label-free mask, 실제 PCFI, projection, schedule/metric/CI/strict-resume는 이 파일이 대신 검증하지 않는다.
- 모델은 disposable instance이고 optimizer step은 버린다. 이 검사는 12 fits 학습이나 target adaptation이 아니다. Root는 실제 실행 비용과 실패 receipt도 scientific ledger에 포함해야 한다.

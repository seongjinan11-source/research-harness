# FP04 runner/statistics implementation review

2026-11-04. 검토 대상: `run.py`, `analyze.py`, `util.py`의 최초 구현 snapshot과 PLAN_v2/protocol v2. Data 구현은 미완성이므로 row metadata·전처리의 실제 충족 여부는 미검증이다. Runner·analyzer·util을 직접 수정하지 않았고 GPU·학습·bank 평가를 실행하지 않았다. 아래 발견 사항은 root에 즉시 전달했으며, root가 수정하겠다고 회신했다. 이 문서는 수정본의 최종 PASS가 아니다.

## 맞게 구현된 핵심 계산

- 기본 3 seeds×3 arms loop는 9 fits다. Seed별 NumPy schedule을 한 번 생성해 세 arm이 재사용하며, 매 arm의 `make(seed)`가 Torch CPU/CUDA RNG를 다시 초기화한다. Same-seed initial state digest도 실제 학습 직전에 비교한다.
- 한 optimizer step에서 네 task의 query CE를 각각 1/4로 나눠 backward하고, 합산 gradient를 한 번 clip/step한다. Task당 query 수가 고정 16이라는 data 계약이면 의도한 평균 CE다. Shared logits에 CE를 한 번 적용하고, 평가에서도 모든 class logits를 받은 뒤 softmax를 한 번 적용한다.
- 중간 best checkpoint 선택이 없고 768번째 update 뒤 state를 저장한다. Synthetic outcome gate를 검사하기 전에 final 다음 real evaluation을 무조건 호출한다. 중단된 fit을 progress marker가 있는데도 조용히 재시작하지 않도록 막는다.
- Primary는 각 base graph에서 고정 seeds와 두 rates를 먼저 평균한 뒤 arm 차이를 계산한다. CI는 독립 node나 seed를 표본 단위로 쓰지 않고, strata 안에서 graph index를 재추출한다. Utility의 eligible rates를 base별로 골라 seeds/rates를 먼저 평균하는 순서도 conditional estimand에 맞는다.
- Runner와 analyzer metric 구현은 별도이며, normalized/clipped probability에 대한 NLL·Brier와 truth-present balanced recall / supplied-K macro-F1의 정의가 일치한다. Prediction 내용과 metric 재계산을 대조하는 방향도 적절하다.

## 실행 전 닫아야 할 실제 누락/오류

### 1. P1 — PASS receipt와 현재 source/bank를 연결하는 lock 검증이 불충분하다

초기 `run.lock()`은 현재 source·bank hash를 잠그지만 checks에서는 status=PASS만 검사하고, preflight의 status나 두 receipt에 저장된 source fingerprint가 현재 source와 같은지는 검사하지 않는다. 따라서 검사 이후 source가 바뀌어도 이전 PASS receipt를 새 source와 함께 lock할 수 있다.

초기 `analyze.main()`은 source/bank의 현재 hash를 lock의 내용과 비교하지 않는다. audit에 lock 파일 자체 hash와 현재 `sources()`를 적는 것만으로는 일치 검증이 아니다. 분석 시 source가 바뀌어도 audit PASS가 나올 수 있다.

최소 수정: lock 생성 때 checks/preflight 모두 PASS이고 receipt의 관련 source/config fingerprint가 현재 것과 같은지 확인한다. Analysis 시작 때 lock을 다시 읽어 current scientific sources·bank manifests/content·receipts가 일치하는지 검증한다. Checkpoint 안의 lock hash도 같은 lock을 가리켜야 한다. Root가 이 gap 수정을 수락했다.

### 2. P1 — prediction/evaluation resume가 checkpoint와 input identity를 보증하지 않는다

초기 `evaluate()`는 `<split>_evaluation.json`이 존재하면 내용을 검증하지 않고 반환한다. `predict_and_save()`는 기존 NPZ에 대해 query/truth/context만 검사하며, 해당 probability가 어느 checkpoint·lock·arm·seed·input hash에서 나왔는지 확인하지 않는다. 같은 split row의 다른 checkpoint 예측을 해당 output directory에 둬도 내용 검사는 통과할 수 있다. 나중 analyzer의 probability hash/rescore 검사는 “그 probability의 metric이 맞는가”를 보증할 뿐, “이 모델이 만든 probability인가”는 보증하지 않는다.

최소 수정: prediction receipt에 checkpoint hash·lock hash·arm·seed·canonical input row hash를 결합하고 재개 때 이를 검증한다. Evaluation complete marker도 expected row set, receipts, prediction hashes와 같은 association을 검사한 후만 재사용한다. JSON/NPZ 생성 중 중단을 완성으로 오인하지 않도록 atomic completion 또는 strict validation이 필요하다. 불완전 파일을 조용히 덮어쓰거나 새 모델 예측으로 섞으면 안 된다. Root가 이 gap 수정을 수락했다.

### 3. P1 — utility에 계획에 없던 missing-strata gate가 추가됐다

초기 analyzer는 일부 원래 stratum에 eligible utility graph가 없으면 interval 계산 후 verdict를 undefined로 덮어쓴다. V2의 utility는 **source≥2인 disjoint case에 조건부인 별도 estimand**이고, 전체 eligible graph 수가 0일 때 undefined다. 일부 stratum의 관측 부재를 새 gate 조건으로 만드는 것은 사전 정의와 다르다.

최소 수정: interval의 empty input 처리는 유지하고, missing strata는 diagnostic으로 기록한다. Eligible strata의 graph bootstrap을 사용하며 표본 구성이 바뀐 conditional estimand라고 명시한다. Root는 “eligible graph 0이면 undefined; 일부 strata 부재는 diagnostic”으로 확정하고 RESPONSE에 기록하겠다고 회신했다.

### 4. P2 — matched schedule/initialization audit가 기록된 문자열의 일치에 의존한다

초기 analyzer는 training JSON의 `schedule_sha256`와 `initial_sha256`가 arm 간 같은지만 검사한다. `schedule.json`의 indices/hash를 읽거나 protocol seed·bank order로 다시 계산하지 않는다. Checkpoint file hash는 검사하지만 checkpoint 내부 lock/initial 및 state digest를 training JSON의 final digest와 연결하지 않는다. 따라서 현재 fresh training loop가 올바르게 동작한다는 사실과 독립 artifact audit의 보증 범위를 구분해야 한다.

최소 수정: 각 training record의 arm/seed/update/exposure를 expected identity와 대조하고, 실제 schedule indices의 array hash·expected generation을 검증한다. Checkpoint의 내부 lock·initial과 state digest도 metadata에 연결한다. 단순히 같은 문자열이 세 번 등장했다는 이유로 `matched_initialization_and_schedule=True`를 쓰지 않는다.

### 5. P2 — source0/1 성능을 별도로 보고하라는 요구가 빠져 있다

초기 summary의 `support_0_1_2plus`는 source0/1/2+의 **case 수**만 보고한다. Performance breakdown은 family/shape뿐이다. V2는 source0/1을 primary에 포함하는 동시에 별도 결과를 요구한다. Case count는 해당 조건에서의 accuracy/NLL·arm 차이를 대신하지 못한다.

최소 수정: rates별 source0/1/2+의 graph/case 수와 fixed-seed 평균 성능을 별도 표로 보존하고, conditional support subset의 집계 단위를 명시한다. Secondary full/node90/cell 표와 각 arm의 none/shuffle 비교도 결과 파일에서 접근 가능하게 한다. 이 보조 표로 primary gate를 바꾸지는 않는다.

## Data가 완성된 뒤 확인할 인터페이스 조건

아래는 지금 확인된 data 버그가 아니라 runner 분석이 의존하는 계약이다.

- `read(row)`의 y/query는 정수 index이고 y는 Torch CE가 받는 int64이며, query·context·K는 같은 base의 모든 rates/interventions에서 동일하다. Query는 feature-free이고 feature availability로 골라진 것이 아니다.
- `rows()`는 canonical 순서를 가지며 path가 전체 split에서 유일하다. `(base_id,setting,variant)`도 final에서 유일하다. `stratum`은 비교 가능한 scalar identifier이고 family/shape/K·disjoint metadata의 실제 의미가 protocol과 일치한다.
- Analyzer의 utility none reference는 standalone `(setting='none',variant='standard')`다. Primary-rate none variant가 따로 있다면, standalone none과 같은 query/context·전처리 결과임을 명시적으로 검사하거나 같은 rate의 none row를 직접 사용해야 한다.
- `sources`는 true observed source node 수이고 metadata 선택만으로 정해진다. Whole-node shuffle은 관측 행 전체를, cell shuffle은 해당 관측 column 내부를 순열하며 every intervention의 RMS/count/moments/literal PCFI를 재계산한다.
- Native identity-checked external references와 raw PCFI cache의 source/config/hash/time·representation clip fraction/maxabs는 data/manifest에 실제로 남아야 한다. 현재 analyzer의 external-summary 안내 문구만으로 reference가 검사됐다고 판단하지 않는다.

## Prelock numeric 수정 기록

이 review 도중 root가 Native PCFI의 큰 값에서 `v²/100`을 그대로 쓰는 표현 범위를 질의했다. 성능 관찰 전, raw literal PCFI cache는 그대로 보존하고 모든 arm의 basis 입력에 공통 `clamp(v,-10,10)`을 적용하도록 승인했다. 소유 model.py와 IMPLEMENTATION_METHOD.md만 수정했다. Tiny CPU extreme-input 확인 PASS이고, 새 model SHA256은 `112CFDBD73C115C9BCCBB64FE99EE73897DD8C266A0C0B1FF5E786C3F58854AC`다. 이 변경은 PCFI magnitude 정보의 일부를 버리는 표현 선택이며, unclipped PCFI와 같은 representation이라는 주장을 하지 않는다.

**종료 판정:** 학습·seed 평균·class normalization의 핵심 계산은 맞다. 위 provenance, utility verdict와 보고 누락을 수정하고 완성 data에 대한 독립 checks/lock을 통과한 뒤 실행해야 한다. 현재 문서는 초기 snapshot의 수정 요구 기록이다.

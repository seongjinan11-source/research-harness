# FP05 runner·analysis 독립 구현 검토

2026-11-05. 대상: `rl_run.py`, `rl_analyze.py`, 연결되는 frozen `rl_model.py`, `rl_util.py` 및 완성된 `rl_data.py`의 공개 read/extra 계약. 다른 agent의 현 단계 리뷰는 읽지 않았다. 읽기 전용 검토이며 코드 수정·GPU·모델 실행·학습·데이터 생성은 하지 않았다.

**판정: 핵심 θ/φ gradient 식은 V2와 일치한다. 아래 calibration 계약과 audit 보강은 source lock 전 수정이 필요하다.** 모델 구조를 바꿀 이유는 발견하지 못했다. 이 문서는 검사/실제 finite optimizer smoke를 대체하지 않는다.

검토한 스냅샷 SHA256:

```text
rl_run.py       bf98a46fac0bf860a3fc6da45b19d32d653981140dab505c3d9b879a50a2e924
rl_analyze.py   f4a377427b911a7832ba1994f39b9170c9188d9117e704c9fdd7d6fdea489ed8
rl_model.py     db0b860547574658e0b67d17a9ab1d30687f3291902734fc6de929f5cd309c80
```

## 정상인 핵심 연산

- **Direct의8중복 scale 없음:** `fit_group`에서 adapter loss는 `p.detach()*CE`이고 actor는 `sum(p*(CE.detach()+.01cost))`다. 양쪽 모두 추가 `/8`이 없어서 normalized eight-leaf expectation과 맞는다. Leaf 합1 tolerance 검사도 있다. Fixed는000 한 번, uniform/PG는8 sampled CE의 평균이므로 서로 다른 arm의 목적식도 맞는다.
- **Actor graph가 CE backward 뒤 살아 있다:** `rl_model`의 state는 detach되고 action은 Python 값이므로 CE graph에 φ 경로가 없다. θ loss에서 p도 detach한다. 따라서 CE backward가 actor의 logp/probability graph를 소모하지 않으며, 이후 actor backward에 `retain_graph=True`를 붙일 필요가 없다. CE 값 자체도 actor에서 detach한다. 이는 전체 coupled θ derivative가 아닌 계획한 conditional θ gradient다.
- **Sampling·한 번의 update:** 같은 inp forward_seed를8 traces에 사용하고 group action generator는 별도 NumPy RNG다. θ/φ는 모든 trajectories와 actor backward 뒤 한 번만 step한다. PG reward/advantage는 detach되고 population std, epsilon1e−6, mean actor loss, entropy0이 계획과 맞는다. Retries에서 action seed를 복원하므로 동일한 action draw를 재현한다.
- **AMP/clip:** 한 optimizer의 두 lr groups를 같은 scaler로 scale하고, 모든 θ/φ backward 뒤 한 번 unscale한다. θ/φ norm clip을 따로1로 적용한 뒤 한 번 step한다. Model의 adapter/policy 파라미터는 FP32이고 policy 및 CE 계산은 FP32다. Actual CUDA finite-step 검사는 여전히 필수다.
- **Checkpoint/frozen identity:** Save는 adapter와 policy 값을 포함하며 frozen base는 public checkpoint로 재구성한다. Load 시 checkpoint SHA, lock SHA, 초기 전체 parameter digest, 복원 뒤 최종 digest와 frozen digest를 검사한다. 새로운 Model마다 같은 public base를 만들기 때문에 training seed에 따른 public initialization 차이를 조용히 허용하지 않는다.
- **출력 좌표·대조 subset:** 모든 경로가 Model의 공통 first-K/class-coordinate/absent-mass 후처리를 거치며 CrossEntropy(log-prob)는 같은 normalized distribution에 대한 CE다. Evaluation은 policy에 y를 전달하지 않고 greedy/forced trace만 사용한다. Standard own-missing membership을 none/shuffle에도 재사용하므로 utility가 intervention마다 다른 query 집합을 비교하지 않는다. Uniform은 probability mixture이고8 forwards를 비용에 반영한다.

## 필요한 수정

### 1. Global-trace selection이 계획과 다르다 — 수정 후 lock

`rl_run.py:226`은 `NLL + .01*sum(trace)`를 dev criterion으로 사용한다. V2는 **dev base/rate 평균 CE**로 best-global trace를 선택한다고 명시했다. 이 차이는 단순 reporting 문제가 아니라 실제 배포 trace를 바꿀 수 있다. 최소 수정은 criterion에서 cost term을 제거하고 selection metadata도 mean CE로 바꾸는 것이다. Cost는 별도로 기록한다. Cost-aware criterion을 의도한다면 먼저 V2/protocol에 명시적 prelock 수정으로 남겨야 한다.

또한 현재 calibration은 metrics만 `global_trace.json`에 보관하며 **요구한8 dev branch prediction arrays를 저장하지 않는다**. 각 dev episode/trace의 p,y,query/context identity를 NPZ로 저장하고 input/checkpoint/prediction SHA를 index에 넣어야 selection을 독립 재검산할 수 있다. 동일 수의 settings를 가진8 dev bases라면 현재 case 평균과 base-then-rate 평균은 같지만, row count/기준을 명시하고 저장 배열에서 재현해야 한다.

`evaluate`는 uniform뿐 아니라 group_pg에도 global calibration/control을 추가한다. 누출은 아니지만 V2의 uniform global-control보다 넓은 secondary output이다. 유지한다면 prelock 운영/분석 명세에 적고 추가 dev forwards를 비용에 포함한다. Primary greedy PG와 이 control을 혼합하지 않는다.

### 2. Nonfinite retry를 성공 후 JSON으로 기록할 수 없는 경로 — 수정 권고

`rl_run.py:134`의 retry entry는 nonfinite CE를 그대로 float로 보관한다. 재시도 뒤 정상 update에 성공하더라도 `progress.jsonl`의 `json.dumps(...,allow_nan=False)`가 이 과거 NaN/Inf 때문에 실패한다. 이때 optimizer는 이미 step된 상태이므로 “정상 재시도 기록” 대신 완료되지 않은 run을 남긴다.

최소 수정은 retry의 nonfinite 숫자를 null/string과 명시적 finite/status flags로 직렬화하고 원인을 보존하는 것이다. Retry 횟수와 그 actual forward/backward 비용도 남긴다. GradScaler가 gradient overflow를 처리하는 경로와 forward 단계 예외로 즉시 중단되는 경로를 구분해야 한다. 원인 없이 재시도를 늘리거나 optimizer step을 다시 실행해서는 안 된다.

### 3. Saved-array audit를 canonical bank와 연결해야 한다 — audit PASS 전 필수

`rl_analyze.py:57–80`은 저장 NPZ 간 query/y/context/yc/own_missing 일치를 검사하지만 **locked input bank의 정답·query·context·standard membership과 직접 비교하지 않는다**. 모든 arm이 공통으로 잘못된 y나 own-missing mask를 저장하면 현재 audit는 통과할 수 있다. Prediction hash는 배열을 metadata에 연결할 뿐, 올바른 physical episode에 연결하지는 않는다.

각 split manifest의 canonical row를 `rl_data.read(row)`와 `extra(row)`로 읽어 다음을 대조해야 한다: query_ids, y, context_ids/labels, classes, standard_own_missing, standard source count, row/input SHA. 각 arm/seed/variant의 expected row IDs를 정확히 한 번씩 포함했는지도 확인해야 한다. 현재 main record 개수만 같다는 검사는 중복/누락을 배제하지 못한다. Full guard의 all-own-observed membership도 comment에 의존하지 말고 canonical full_natural mask로 확인한다.

분석은 training/checkpoint 파일 SHA와 schedule SHA의 arm 간 일치를 확인한다. 실제 schedule 내용이 seed로 재현되는지, progress의 성공 groups/exposure/epsilon/action seeds와 일치하는지, checkpoint에 기록된 arm/seed/lock/schedule association이 일치하는지는 별도 root audit 또는 이 분석에서 검증해야 한다. Evaluation의 `load()` digest 확인은 정상적이므로 모델 재학습 없이 감사만 보강하면 된다.

### 4. 빈 eligible set은 계획한 UNDEFINED가 되어야 한다 — 작은 경계조건 수정

`interval`의 `assert ids` 때문에 primary comparison의 공통 eligible base가0개이면 gate를 UNDEFINED로 기록하기 전에 analysis가 중단된다. Routing secondary의 available set이0인 경우도 같다. V2는 empty/불완전 primary를 undefined로 보고하고 재추출하지 않도록 했다. Empty interval은 `mean/ci=None,n_graphs=0`를 반환하고 caller가 UNDEFINED를 보존하도록 처리해야 한다. Utility empty는 현재 별도로 처리하고 있다.

## 집계 식과 남은 보고 범위

현재 nonempty-case 집계는 옳다. `grouped_mean`이 같은 physical case의 고정3 training seeds를 먼저 평균한 뒤, available95/995 rates를 base 안에서 평균한다. Paired graph 차이를 shape×k×regime strata 안에서 bootstrap하므로 nodes나 seeds를 독립 graph처럼 재표집하지 않는다. Utility는 disjoint/standard-source≥2 조건을 입력 기반으로 선택하고 같은 base/rate/query subset을 유지한다. DoD와 trained-policy replay000도 올바른 case 좌표를 사용한다. Natural full에서 모든 query가 실제 observed라는 data 계약이 검증된다면 whole-query guard는 own-observed guard와 동일하다.

Primary summary는 현재 accuracy만 집계한다. NLL/Brier/BAcc/macroF1가 raw record에 보존되므로 값이 유실되지는 않지만, V2의 판단 범위를 충족하려면 같은 graph weighting의 secondary metric summaries와 비용을 함께 보고해야 한다. 이를 accuracy gate 대체에 사용해서는 안 된다. Policy actor loss/expected CE와 greedy deployment score도 서로 다른 양임을 유지해야 한다.

수정 후 source hashes를 다시 고정하고 root의 actual-model parity/finite-step 및 canonical-array audit를 통과하면, 현재 발견한 사항 때문에 θ/φ 목적식이나 모델 구조를 바꿀 필요는 없다.

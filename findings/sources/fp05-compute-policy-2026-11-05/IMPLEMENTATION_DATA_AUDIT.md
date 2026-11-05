# FP05 prepared-data and planned-exposure audit

2026-11-05. **입력 내용·식별 감사 PASS; 준비 시점 utility 버전 차이는 명시적으로 기록됐으며 과거 bytes 미보존 한계는 남는다.** 검토 범위는 root가 생성한 train1792/dev56/real45 episodes와 train/dev264개 private raw tasks, 그리고 계획된128-group schedules다. Model predictions/정확도/loss/최종 성능을 읽거나 새 model forward·학습·GPU 작업을 실행하지 않았다. 감사 시작 시 생성 중이던 final bank는 이번 NPZ 전수 감사 대상에서 제외했다. Source/bank/원결과/branch는 변경하지 않았고 새 파일은 이 문서뿐이다.

검토자는 rl_data.py 작성자다. 이번 계산은 별도의 CPU NumPy 코드로 raw/processed NPZ를 대조하고, 실제 raw real sources에서 projection을 다시 구성했다. 완전히 독립된 데이터 구현 재현이라고 부르지는 않는다. 하지만 감사의 normalization/hash/schedule 계산은 production helper를 호출하지 않고 구현했으며, 실제 read()/extra() whitelist도 별도로 확인했다.

## 1. 검토한 고정 식별

| Artifact | SHA256 |
|---|---|
| rl_data.py | `2d72eff6ac0a8a1de243fc95511cfa20e13e37b7c33696921bac0a622770dc99` |
| train manifest | `77e866cc5d4bf060a6d309935dcad474840c82049cc7e8e9bd2ee3e9466cf2bc` |
| dev manifest | `bed5efcd8e11df7bc092907677ca71dbdb062945b641bef986c6077bf28d258d` |
| real manifest | `db3c14b1d61bec329d43ecdd65e63ee3b30e84b472bfbdffff44cde151c608be` |
| root-created design | `8d790c1a3c962c47c6fc01ae60433bfeaff5302e0607f9fb7483a850d12900fe` |

각 manifest의 protocol/data SHA, 모든1850 processed NPZ SHA,264 raw NPZ SHA, generator-config SHA와 raw-task fingerprint를 확인했다. Train/dev raw fingerprints는264개 모두 달랐다. Requested/actual N,F,K와 공식 n_train, context/query IDs 및 labels를 실제 raw arrays에서 대조했다.

모든 base의7 standards는 graph edge arrays·context/query·query truth·forward seed가 같다. Context는 공식 train segment, query는 evaluation segment에 있고 서로 겹치지 않는다. Query IDs는 중복되지 않는다. Context label/class-cap, graph bounds와 self/duplicate edge 수, raw full class counts가 저장된 audit와 일치했다. 실제 read()는 정확한9-key whitelist만 반환하며 target y는 별도다. Raw y_full·source/generator audit·own-missing flags는 model input에 들어가지 않는다.

## 2. 실제 크기와 support

| Split | Bases / processed rows | Actual N | Actual physical F | Actual K counts | Actual Q |
|---|---:|---:|---:|---|---|
| train | 256 / 1792 | 54…192 | 7…12 | K2=161,K3=52,K4=43 | 255 bases는16,1 base는11 |
| dev | 8 / 56 | 126…191 | 8…10 | K2=3,K3=3,K4=2 | 모든 base16 |

Requested N/F와 actual prior output은 다르다. `train_g0085`는 requested128에서 actual98이며, `train_g0162`는 requested160/F8에서 actual54/F7, n_train43 및 query11이다. 후자의 raw class counts는24/14/9/7이다. 이는 숨기거나 다시 뽑을 사유가 아니며, “학습 actual N이 모두128…192”라는 설명은 틀리다. Train에서는6 bases가 홀수 F이며, 데이터가 dummy column을 추가하지 않았음을 확인했다. Original GraphPFN preprocessing이 처리할 범위다.

Train natural/disjoint는128/128 bases, k1/3/5는86/86/84다. Dev는4/4, k별4/2/2다. 모든 disjoint mask에서 context/query가 실제로 비관측이다. Natural full에서는 query를 포함한 모든 physical cells가 관측됐다. 따라서 full_natural과 full_nonquery를 실제로 구분할 수 있다.

다음은 source **node** 수의 분포다. 2+는 두 개 이상이며, observed cells 수와 혼동하지 않는다. Source0/1을 제외하거나 mask를 다시 뽑은 흔적은 없다.

| Train setting | Source0 | Source1 | Source2+ | Missing / observed-query targets |
|---|---:|---:|---:|---:|
| full | 0 | 0 | 256 | 2048 / 2043 |
| node90 | 0 | 0 | 256 | 3874 / 217 |
| node95 | 1 | 3 | 252 | 3985 / 106 |
| node995 | 130 | 81 | 45 | 4077 / 14 |
| cluster95 | 0 | 0 | 256 | 3963 / 128 |
| cluster995 | 14 | 242 | 0 | 4074 / 17 |
| cell95 | 0 | 0 | 256 | 3287 / 804 |
| Total episode cases | 145 | 326 | 1321 | 25308 / 3329 |

Dev의 source0/source1/source2+ 합계는3/13/40이다. Node995는3/5/0, cluster995는0/8/0이며 나머지 setting은 각각8 source2+ cases다. 이는 모델 효과나 독립 표본 수가 아니다. 두 bank의 cluster995에는 source2+가 없으므로 이 조건이 다수 source의 feature-law 식별을 제공했다고 설명하면 안 된다.

Own-missing은 mask에서 query의 원래 물리적 행에 어떤 관측이라도 있는지로 정의하며 PCFI 값이0인지로 정하지 않는다. 일부만 관측된 cell95 query는 observed-query지만 fully-observed-query는 아니다. 표의 수는 고정 bank targets이며 아직 실제 학습 schedule 노출 수는 아니다.

## 3. Mask·정규화·cache consistency

모든 train/dev episodes의 uniform node/cell masks를 독립 RNG 벡터/행렬로 재생성해 완전 일치를 확인했다. Base별 node995⊆node95⊆node90 및 cluster995⊆cluster95도 확인했다. Disjoint는 원래 mask 뒤 context/query를 지우므로 cluster의 최초 rounded count보다 실제 source가 작아질 수 있으며 source0도 유지된다.

모든 processed episode에서 독립 float64 observed RMS/scale1/clip10 재계산이 visible_values와 원소별 완전히 같았다. 모든 hidden raw features를 NaN으로 바꿔도 같았다. Hidden 위치는 정확히0이고 실제 관측0은 observed=True로 유지된다. Source_nodes·standard_source_nodes·observed_cells·column-support0/1·context overlap·query-observed/missing counts와 NPZ flags는 mask에서 직접 계산한 값과 일치했다.

모든 pcfi_raw가 유한하며, 관측 위치는 visible_values와 정확히 같고 values는 clip(pcfi_raw,−10,10)과 정확히 같다. Source0 cache 전체와 source가 없는 column도0이다(train 합계1439 episode-columns, dev30). 이 열 수는 source node 수가 아니다. Input arrays와 graph/context/query-truth/standard-membership hashes를 재계산해 일치를 확인했다.

이번 감사가 모든1850 episodes에서100-step PCFI kernel을 따로 재실행한 것은 아니다. 저장 내용·pre-PCFI 입력·clamp/empty 지원·literal 구현의 출처를 확인했으며 kernel reference/gradient 검사는 독립 checks 담당이다. 유한성/clamp 검사만을 전체 imputation 수식의 독립 재현이라고 부르지 않는다. 어떤 cache도 분류 점수로 선택하지 않았다.

Train/dev는 standard rows만 있어 standard_own_missing과 actual_own_missing이 같다. Synthetic none/shuffle의 원래 standard subset 보존은 preimplementation CPU selfcheck에서 확인했지만 생성 중인 final1056 rows는 이후 root의 전수 감사가 필요하다. 이 보고서는 final intervention 전부가 통과했다는 판정이 아니다.

## 4. Real45의 원래 행과 RP identity

PyG가 있는 기존 Python 환경에서도 CPU read-only 스크립트를 실행했다. Immutable raw PyG 파일3개의 SHA를 확인하고 Cora/CiteSeer의 이전 context/source/query RNG와 shuffle 행 매핑, PubMed original/value-original ID 매핑과 induced edges를 재구성했다. 이 재구성을 production _real_base 호출로 대신하지 않았고 모델 예측도 읽지 않았다.

Projection NPZ3개의 SHA·array hash·seed를 재구성했다. R은 graph별 nativeF×16, seed51089000+graph index, entries±1/sqrt16이며 같은 graph의 모든 reps/modes가 같은 R을 사용한다. 독립 raw-visible 행렬과 physical mask hash가 native_visible_sha256과 일치했다. RP 뒤 독립 RMS와 저장 visible_values의 최대 차이는 **5.2154e−7**이었다(CPU 행렬곱 FP32 오차 규모로 감사 허용치3e−6 미만). 숨긴 행은 행렬곱의 visible 입력에 들어가지 않았으며 FP04 clipped/RMS 값을 native raw로 오인하지 않았다.

FP05와 FP04/legacy의 실제 ordered query/truth·context labels·graph가 완전히 같고 graph/repetition별5 modes에서도 같았다. 모든 real query/context의 원래 physical row는 missing이다.16열에 broadcast한 row mask는 native availability와 행별로 같으며 RP 결과0을 missing으로 오인하지 않았다. 새 NPZ의 physical_features=16은 projected input 열 수다. Native_features는1433/3703/500으로 따로 저장되므로 RP16을 원 차원 입력이라고 설명하면 안 된다.

| Graph / inherited scope | 95 sources | 99 sources | 995 sources | none | shuffle95 |
|---|---:|---:|---:|---:|---:|
| Cora2708 full | 135 | 27 | 14 | 0 | 135 |
| CiteSeer3327 full | 166 | 33 | 17 | 0 | 166 |
| PubMed2000 induced | 100 | 20 | 10 | 0 | 100 |

각 항목은 세 repetitions에서 source 수가 같지만 구체적 source/context/query는 다를 수 있다.45개 NPZ의 PCFI observation clamp·값 clip·own-missing/source flags와 모든 identity hashes도 통과했다. CPU 스크립트가 끝날 때 `torch.cuda.is_initialized()`는 여전히 False였다.

이번 감사는 물리적 데이터/처리된 입력의 일치를 확인한 것이며 RP가 모든 native 정보를 보존하거나 real transfer에 성공했다는 증거는 아니다. 이전 public/native-PCFI/FISF와 현재 RP16의 표현은 다르다. 외부225 predictions의 metric 재감사는 결과 단계 작업으로 이번 무점수 데이터 감사에서는 실행하지 않았다.

## 5. 계획된128-group schedule의 실제 학습 지원

현재 rl_run.schedule 규칙에 맞춰 seed11/12/13을 독립 재구성했다. Update0,8,…,120은 full-natural 집합에서, 나머지는1792 rows 전체에서 뽑고 같은 순서로 forward/action seed RNG도 소비한다. Seed별128 groups다. 아래는 arm마다 공유할 예정인 schedule이며 학습을 완료했다는 뜻이 아니다.

| Seed | Full-natural groups | Full-natural observed targets | Unique episodes | Unique base graphs | All observed / missing targets |
|---|---:|---:|---:|---:|---:|
| 11 | 24 | 384 | 125 | 107 | 460 / 1588 |
| 12 | 29 | 459 | 124 | 101 | 516 / 1527 |
| 13 | 22 | 347 | 124 | 101 | 423 / 1620 |

예약된16 full-natural groups를 모두 충족했고 uniform 추첨에서도 추가되어24/29/22가 됐다. Seed12/13은 Q11 full-natural episode를 각각 한 번 포함해 targets가 groups×16과 다르다. 어떤 query feature라도 관측된 target 비율은 각각약22.46%/25.26%/20.70%로50%가 아니다. Fully observed full-natural과 부분 관측 cell query도 구분해야 한다.

Natural/disjoint를 합쳐 setting별 `groups ; observed targets ; missing targets`로 요약하면 다음과 같다.

| Setting | Seed11 | Seed12 | Seed13 |
|---|---:|---:|---:|
| full | 33 ; 384 ; 144 | 33 ; 459 ; 64 | 33 ; 347 ; 176 |
| node90 | 19 ; 19 ; 285 | 16 ; 16 ; 240 | 16 ; 8 ; 248 |
| node95 | 18 ; 5 ; 283 | 10 ; 1 ; 159 | 17 ; 4 ; 268 |
| node995 | 13 ; 0 ; 208 | 18 ; 0 ; 288 | 11 ; 0 ; 176 |
| cluster95 | 13 ; 12 ; 196 | 16 ; 10 ; 246 | 19 ; 11 ; 293 |
| cluster995 | 17 ; 1 ; 271 | 20 ; 0 ; 320 | 17 ; 0 ; 272 |
| cell95 | 15 ; 39 ; 201 | 15 ; 30 ; 210 | 15 ; 53 ; 187 |

Schedule의 source0/source1 groups는 각각9/19,8/24,8/20이며 제거하지 않았다. 세 seed 합계6134 group-level query targets 중1399 observed,4735 missing이다. 네 arm은 각각 같은 seed schedule을 반복한다.8 logical trajectories/group은 독립 physical tasks8개가 아니며 fixed의 한 번 계산과 uniform/direct/group 분기 비용도 다르다. 학습 완료 후 saved schedule/progress로 이 planned exposure 수를 다시 대조해야 한다.

## 6. 원래 official prior 거절 규칙의 정확한 범위

원본 `check_dataset`은 feature/label 유한성, 최소4 features, 기대 class 수, 유효한 train/evaluation split 및 **모든 class가 train과 evaluation 양쪽에 등장하는지**를 검사한다. `_sample_dataset_with_retry`는 같은 sampled config에서 최대3회 검사하고 `sample_batch`는 SanityCheckError를 잡으면 config를 다시 뽑아 계속한다. 외부 prior_adapter는 binclass/multiclass task type도 선별한다.

따라서 모든 무조건부 SCM draw를 그대로 유지한 bank는 아니다. Accepted raw tasks 전체의 train/eval class coverage를 이번에도 직접 확인했다. 내장 label-dependent sanity filter는 채택한 원래 official prior의 일부이며 deploy 입력 누출이나 새 model-performance/class-balance filter는 아니다. 다만 그 존재를 명시해야 한다. 내부 생성 시 query segment labels도 class coverage 검사에 쓰이므로 “query labels는 synthetic 생성/acceptance에도 전혀 쓰이지 않았다”는 포괄적 주장은 틀리다. 이 labels가 policy/predictor에 전달된 것은 아니다.

Manifest의 structural_retries는 실제로 외부에서 잡은 AssertionError 수다. Train은총345회,140 bases에서 적어도 한 번, 최대16회였다. Dev는총11회,6 bases, 최대3회다. 저장된 message가 대개 비어 있어 이 필드만으로 원인을 전부 structural이라고 확정할 수 없다. 성공한 prior_adapter 호출의 task-type rejection은 train51회/dev0회로 따로 저장됐다. **원 sampler 내부의 class-coverage/feature sanity retries는 API가 모두 반환하지 않아 총 거절 draw 수와 원인별 횟수는 모른다.** 외부50회 한도도 내부 draw 전체 횟수의 상한이 아니다.

이는 초기 문서의 “only structural”을 과도하게 이해할 가능성을 정정한다. 현재 코드는 model score·accuracy·reward·추가 class balance·source-support acceptance filter를 넣지 않았고, 이번 감사도 더 이상적인 support를 얻으려고 데이터를 다시 뽑지 않았다.

## 7. Preparation utility 버전 차이

세 manifest.dependencies에 기록된 준비 시점 rl_util.py SHA는 `af7dba0d423b7f5fb0c94e9b744d32d19e61cf8b5b7b0edbc57674cec7f5c80d`이고 감사 시점 current는 `4ee7f2bfe2bfcf4c1e77ac19b7fb262d79c1abe02faaaa4623d4c9b733164e18`이다. 다른 dependencies는 현재 파일과 일치하며 rl_data.py는2d72eff6 버전이다. 차이를 즉시 root에 알렸고 원 manifest를 고쳐 일치하는 것처럼 만들지 않았다.

Root의 BUILD_EXECUTION은 prelock source-hash/bookkeeping 확장과 checks 재실행을 설명한다. 이 설명만으로 정확한 utility byte delta까지 확인되는 것은 아니므로 준비 시점과 현재 design/학습 버전, 알려진 변경과 미보존 범위를 구분해야 한다. 독립 raw/normalization/RP/schedule 대조에서는 이에 따른 입력 차이를 찾지 못했다. 따라서 확인된 데이터 내용 bug가 아니며 SHA를 맞추려고 bank를 덮어쓸 필요도 없다. 제한되는 주장은 “준비 시점 dependency bytes가 현재와 모두 같다”는 것이다.

**Root 대응을 추가 확인했다.** `PRELOCK_SOURCE_DELTA.json`과 `BUILD_EXECUTION.md`는 old SHA/current SHA, 영향받는3개 manifests, 다른 dependency의 일치와 current-design 아래 final 준비를 분리해 기록한다. 기억한 source-list 추가만 제거해도 old SHA를 복원하지 못했다는 사실, 이전 utility bytes가 보존되지 않아 “bookkeeping만 바뀌었다”는 byte-exact 증거는 없다는 한계도 명시했다. 과거 source snapshot을 만들어냈다고 주장하지 않고 원 bank/manifests를 그대로 남겼다. 이 대응은 적절하다. 불일치를 해소됐다고 가장하지 않으면서 실제 입력 감사와 provenance 완전성의 한계를 구분한다. Bank를 사용하는 학습/결과의 재현 가능성과 이전 bank 생성 프로그램 bytes까지 완전히 재현할 수 있다는 주장은 구분해야 한다. 새 source·bank 재생성이나 성능 기반 선택을 요구하지 않는다.

**최종 범위:** train/dev/real 내용, query/source 지원, visible 전처리와 native row/RP identity는 이번 CPU 감사를 통과했다. 실제12 fits 실행, final1056-row intervention bank, 정책 선택과 분류 효과는 고정 실험 및 후속 결과 감사로 판단해야 한다. 이 보고서는 그러한 모델 효과를 읽거나 추정하지 않았다.

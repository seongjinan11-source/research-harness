# FP05 최종 실험·통계 검토

2026-11-05. **판정: 실행·저장 확률·통계 감사 PASS, 사전 과학적 성공 gate FAIL. 현재 action/operator/objective/128-update 예산의 후보는 종료하는 것이 타당하다.** PG의 fixed/direct 대비 우월성과 disjoint feature utility가 확인되지 않았다. 실제 fully-observed query의 보존 guard는 통과했다. Real 개발 패널에는 양의 관측 효용이 있으며 이를 보존하되, adaptive routing 또는 PG 고유 효과로 승격하지 않는다.

검토자는 `rl_data.py`와 별도 `research_cycles/fp05_statistics_audit.py` 작성에 참여했다. 따라서 데이터/감사 구현의 저자와 완전히 독립된 심사라고 주장하지 않는다. 이번 문서는 실험설계·통계 관점의 별도 검토다. Root가 실행한 독립 NumPy 통계 재검산의 실제 출력과 다른 담당자의 deployment audit를 읽었고, 저장 확률의 main–replay 일치와 support/real/budget 집계를 CPU로 추가 확인했다. Production analyzer/채점 helper를 호출하지 않았으며 새 GPU forward·학습·예측·bank/source/checkpoint/원결과 수정은 하지 않았다. 이번에 추가한 파일은 이 문서뿐이다.

## 1. 실행과 통계 결과를 연결하는 근거

- 12 fits 모두 128 successful optimizer groups, frozen public parameters 불변, target updates 0이다. 공통 seed별 schedule과 입력 SHA, checkpoint/fit 식별, actor scalar 및 calibration/deployment variant 감사를 통과했다.
- Canonical final1056 episodes와 real45 episodes에 대해 **29,568 synthetic + 1,260 real = 30,828 저장 prediction records**를 재채점했다. 이 수는 여러 arms/seeds/variants/references를 포함한 기록 수이며 독립 graph 수가 아니다.
- Production 감사, 별도 NumPy 통계 감사, CPU deployment 감사는 모두 PASS다. 별도 통계 감사의 saved-metric 및 summary/CI 최대 차이는 각각 **4.44e−16**이다. Accuracy/NLL/Brier/BAcc/macro-F1, ordered query/truth/context 및 원래 standard own-missing membership이 연결됐다. BAcc는 truth에 등장한 classes의 recall 평균, macro-F1은 supplied K 전체를 포함하는 정의다.
- 별도 통계 스크립트는 production `rl_*`를 import하거나 실행하지 않고 locked analyzer의 난수 seed/CI recipe를 읽어 대응했다. Deployment audit도 scientific forwards/optimizer updates/GPU 사용이 모두 0이다. 다만 저장 actor 상태 전체를 다시 forward한 감사는 아니므로 로그의 conditional policy probabilities를 매 update 재현했다는 주장은 하지 않는다.

| 근거 artifact | SHA256 |
|---|---|
| `results/lock.json` | `18b27b3d9a4eb3c8fc321824bf32407ae8f888d27c4e2de4a10b66eba325c575` |
| `results/summary.json` | `9dc5546f6cd8eaf35880d905c0f59cc420579286fb6c1081925a92c57429b6ad` |
| `results/statistics_audit.json` | `7761a2a639b6a36ab7d7e3e33de6ae242d23e7c66c75bb16fe1f61c69f59dad8` |
| `results/deployment_audit.json` | `c899cf4bc5d19fe0ea568aefc03b71010fa731f3c4e3d76bd486e0db05b94a1a` |

## 2. 사전 estimand와 gate 판정

Primary는 node95/node995의 **원래 physical query row에 관측 cell이 전혀 없는 subset**이다. PCFI 출력이 0인지로 missing을 정의하지 않는다. 고정 3 training seeds를 case 안에서 평균하고, 두 유효 rates를 base graph 안에서 평균한 뒤 paired 차이를 만든다. Final96 graphs 모두 두 rates가 유효하므로 **96 graphs/192 graph-rate cases**, 12 strata(shape × context cap k × natural/disjoint)에 각각 8 graphs다. 여기서 k=1/3/5는 class 수 K가 아니다.

5000회 within-stratum graph bootstrap은 locked seed51079000과 비교별 offset을 사용한다. Node·query·rate·training seed를 독립 표본으로 부풀리지 않았다. CI는 고정된 세 training runs를 평균한 graph-task 분포에 관한 것으로, 학습 seed 모집단의 불확실성까지 추정하지 않는다. Stratum당 8 graphs인 제한된 official-prior task 표본이며 real population CI가 아니다.

| 사전 비교 | Accuracy 차이 %p | 명목 97.5% CI %p | Graphs / strata | 판정 |
|---|---:|---:|---:|---|
| PG − fixed | −0.4974 | [−1.0336, +0.0121] | 96 / 12 | FAIL |
| PG − direct | +0.0172 | [−0.2409, +0.2748] | 96 / 12 | FAIL |
| PG observed − none | −1.0634 | [−5.5667, +4.0693] | 48 / 6 | FAIL |
| PG observed − shuffle | −0.2604 | [−2.1267, +1.5842] | 48 / 6 | FAIL |
| Full-natural PG − fixed | +0.1302 | [−0.8247, +1.0417] | 48 / 6 | PASS: lower > −2%p |

**명목 97.5% intervals이며 네 개선 비교 전체에 대한 동시 95% coverage를 보장한다고 표현하면 안 된다.** PG−fixed는 음의 점추정이지만 CI가 0을 포함한다. 사전 우월성 실패를 확정된 열등성/정확한 효과0/동등성으로 바꾸지 않는다. PG−direct의 작은 양의 점추정도 우월성 증거가 아니다. Full-natural guard는 실제 own-row fully observed인 natural48 graphs의 768 queries에서 사전 −2%p 손상 한계를 넘지 않았다는 상대적 결과다. Fixed 대비 positive gain이나 absolute quality를 증명한 것은 아니다.

Utility는 primary와 별도 조건부 estimand다. Disjoint이며 **standard source nodes >=2**, own-missing query가 있는 cases만 사전 규칙으로 정하면 **48 eligible graphs / 58 cases**(node95 48, node995 10)다. 38 graphs는 한 rate, 10 graphs는 두 rates를 먼저 평균하고 graph에는 같은 weight를 줬다. 58 cases를 pooled 평균하거나 primary96/192와 같은 분모로 부르면 안 된다. None의 actual source0으로 eligibility를 다시 정하지 않았다. Utility58 안의 8 shuffle cases는 실제 입력이 unchanged인 허용된 순열이며 그대로 유지됐다.

## 3. 강한 기준선과 같은 primary subset의 다섯 metric

아래는 전부 같은 own-missing subset, seed→rate→base weighting이다. CI를 추가해 winner를 선택한 표가 아니라 저장된 prespecified/secondary predictions의 점추정이다. Global은 dev8 graphs ×7 settings에서 mean CE, exact tie 시 cost/lexical 순서로 고정된 단일 trace다. Final labels로 선택한 trace가 아니다.

| Arm / deployment | Accuracy % | NLL | Brier | BAcc % | Macro-F1 % |
|---|---:|---:|---:|---:|---:|
| Fixed000, CE continuation | 49.2904 | 0.845051 | 0.555849 | 48.8481 | 36.9919 |
| Uniform, 8-trace probability mean | 49.4750 | 0.845582 | 0.555509 | 48.5982 | 37.0127 |
| Direct, greedy | 48.7759 | 0.845467 | 0.556608 | 48.1801 | 36.3424 |
| PG, greedy | 48.7930 | 0.845406 | 0.556633 | 48.3129 | 36.4324 |
| Uniform, dev-selected global | 48.7381 | 0.846806 | 0.555781 | 47.5671 | 36.4522 |
| PG, dev-selected global | 48.7952 | 0.845002 | 0.554423 | 47.5869 | 36.3788 |
| Frozen released PFN + PCFI000 | 48.7883 | 0.869112 | 0.572587 | 48.4497 | 37.0680 |
| Frozen released PFN + NaN000 | 47.7739 | 0.862407 | 0.574038 | 47.2294 | 36.9934 |
| Restart LP | 47.0152 | 0.834960 | 0.548308 | 45.8178 | 34.5989 |
| Context-frequency prior | 43.8948 | 0.826377 | 0.542449 | 46.0069 | 25.7151 |

PG accuracy는 frozen-PCFI와 약 +0.0047%p 차이이며, fixed보다 낮다. PG의 frozen-PCFI 대비 NLL/Brier 감소는 실제 점추정으로 보존한다. 그러나 이를 learned routing의 효과라고 부를 수 없고 fixed CE continuation도 비슷한 개선을 보인다. Uniform8의 accuracy가 +0.1845%p 높다는 descriptive result 역시 추가 inference8회 비용을 가진 별도 control이며 PG gate를 대신하지 않는다. LP/prior의 accuracy는 낮지만 NLL/Brier는 PG보다 좋다. Accuracy만으로 전체 probabilistic quality에서 모든 classical baselines를 이겼다고 말하면 안 된다. 반대로 낮은 prior NLL만으로 primary accuracy를 사후 교체하지 않는다.

`secondary_all_query`와 `secondary_primary_same_missing_subset`은 서로 다른 표다. 예를 들어 PG node95는 all-query49.2839%/NLL0.836203, 원 own-missing subset49.2238%/0.836678이다. Node95에서는 47개, node995에서는 7개의 own-observed targets가 primary에서 빠진다. All-query full 표는 natural fully observed와 disjoint full_nonquery를 함께 포함하므로 full-natural guard와도 다르다. Control을 score할 때 none/shuffle에서 새로 missing이 된 query를 추가하지 않고 원 standard membership을 유지한 점이 적절하다.

## 4. 실제 routing과 sparse support가 말하는 것

Direct와 PG의 **모든 final+real greedy main records는 trace000**이다. 검토 중 추가 CPU 검사에서 **6606 main–replay000 쌍**(두 policy arms ×3 seeds ×[1056+45 episodes])의 query/truth/context/own-membership이 같고, 저장 확률의 최대 절대 차이는 **0.0**이었다. Primary routing−replay CI도 두 arm 모두 [0,0]이다. 따라서 현재 배포에서 input별 추가 adapter computation으로 얻은 이득은 없다. PG−fixed 차이는 서로 다르게 학습된 adapter parameters 등의 training-path 차이를 포함하며 routing의 기여를 분리한 양의 증거가 아니다.

Feature difference-in-differences도 none 기준 **−0.3906%p [−1.8446,+0.9766]**, shuffle 기준 **−0.4557%p [−1.5191,+0.5425]**로 0을 포함한다. PG에서 관측을 사용했다는 사실이나 PG가 특정 기준선보다 높다는 사실만으로 fixed보다 feature utility가 커졌다고 할 수 없다.

Node995는 source0=38, source1=38, source>=2=20이다. 즉 76/96(79.17%)가 source0/1이며 primary에는 그대로 포함했다. Node95는 96개 모두 source>=2다. 아래는 seed 평균 후 **graph-rate case-weighted descriptive subset**으로, primary의 graph 안 rate 평균이나 utility48/58를 대체하지 않는다.

| Source bucket | Cases | Fixed acc % | Uniform8 acc % | Direct acc % | PG acc % |
|---|---:|---:|---:|---:|---:|
| 0 | 38 | 50.4386 | 49.2873 | 49.6711 | 49.7807 |
| 1 | 38 | 47.8289 | 48.1981 | 47.5512 | 47.6060 |
| 2+ | 116 | 49.3931 | 49.9547 | 48.8838 | 48.8583 |

Source0 결과는 관측 feature 추론의 성공이 될 수 없다. Source0/1을 사후 삭제해 primary를 구제해서도 안 된다. 반대로 source>=2 조건에서도 정해진 utility가 실패했으므로 실패를 source0/1만으로 전부 설명할 수 없다. Matched/heldout primary의 PG−fixed 점추정도 각각 −0.4096/−0.5852%p로 어느 shape에서 양의 구제를 주장하지 않는다.

조건을 섞으면 feature-effect 부호가 바뀔 수 있다. 모든 두 rates를 그대로 평균한 descriptive natural PG observed−none는 +2.4017%p, disjoint는 −0.6510%p다. Natural에는 관측된 context rows도 허용되는 반면 disjoint는 context/query source를 제거한다. 이는 서로 다른 조건의 관찰이며 위 utility의 eligible-rate 분모와도 다르다. 이를 class anchoring 하나가 확정 원인이라고 단정하거나 natural subset을 새 primary로 삼으면 안 된다.

Trace000은 비용항, action operator의 가치, 짧은 학습, state summary, prior/real shift 또는 greedy objective mismatch 중 무엇이 원인인지 분리하지 않는다. Direct exact-expectation arm도 000이므로 PG sampling failure 하나로 확정할 자료가 없다. 수치/gradient/RNG 검사 PASS는 이 결과를 성능 성공으로 바꾸지 않으며, 성능 실패만으로 검사된 구현을 faulty implementation으로 재분류하지도 않는다.

## 5. 실제 real 개발 패널: rates 정렬과 긍정적 feature effect

Cora2708/CiteSeer3327은 기존 full graph, PubMed2000은 기존 induced graph다. Trained-arm 셀은 **3 training seeds ×3 고정 context/mask repetitions** 평균이다. Frozen/LP/prior는 3 repetitions이며 training seed를 인위적으로 복제하지 않았다. 동일 세 개발 graphs의 반복 재사용이므로 9 independent real graphs나 새 transfer population CI가 아니다.

Canonical NPZ에서 실제 context는 class당4개로 Cora7 classes/28 labels, CiteSeer6/24, PubMed3/12다. Rep0/1/2의 query 수는 각각 Cora953/946/956, CiteSeer949/945/955, PubMed1521/1516/1516이다. 셀은 이러한 repetitions의 metric 평균이지 query를 모두 pooled한 점수가 아니다. 과거 class당20-context benchmark와 수치를 직접 비교하면 안 된다. Synthetic actual K<=4에서 Cora7/CiteSeer6으로 class 수가 바뀌는 shift도 존재하지만 이것 하나가 실패 원인으로 확인된 것은 아니다.

Whole-row native observations만 fixed graph-specific Rademacher RP16에 넣고 observed RMS→literal PCFI100→clip을 적용했다. 원 physical row/context/query/target identity를 유지했고 hidden native rows를 projection에 넣지 않았다. Cora1433/CiteSeer3703/PubMed500 native features를 그대로 모델에 유지한 실험은 아니다. RPCache의 non-none PCFI raw 최대 abs 범위는 각각2.666…3.702/2.835…3.701/2.212…3.582이며 abs>10 fraction은 전부0이었다. 이번 projected real 결과를 clipping 손실로 설명할 근거는 없다. Historical native PCFI의 clipping/표현과 혼동하지 않는다.

아래는 실제 missing95%,99%,99.5% 순서다(`995`를 lexical 정렬해 99 앞뒤가 뒤섞이지 않도록 했다). 전부 accuracy%다.

| Graph | Missing % | Fixed | Uniform8 | Direct | PG | Frozen RP-PCFI | Frozen RP-NaN | LP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 95 | 45.891 | 47.083 | 45.459 | 45.459 | 46.439 | 31.148 | 59.056 |
| Cora | 99 | 51.049 | 51.504 | 50.665 | 50.653 | 50.768 | 33.495 | 59.056 |
| Cora | 99.5 | 52.865 | 53.121 | 52.714 | 52.748 | 53.370 | 31.220 | 59.056 |
| CiteSeer | 95 | 29.354 | 29.060 | 28.369 | 28.334 | 28.547 | 19.656 | 38.410 |
| CiteSeer | 99 | 37.106 | 35.089 | 35.708 | 35.779 | 37.943 | 22.259 | 38.410 |
| CiteSeer | 99.5 | 33.427 | 33.839 | 32.228 | 32.275 | 33.576 | 21.238 | 38.410 |
| PubMed | 95 | 62.323 | 64.996 | 61.122 | 61.671 | 65.032 | 28.938 | 73.690 |
| PubMed | 99 | 67.193 | 69.061 | 66.254 | 66.767 | 69.375 | 34.286 | 73.690 |
| PubMed | 99.5 | 69.502 | 71.024 | 69.772 | 69.904 | 71.242 | 32.489 | 73.690 |

PG95의 none/shuffle contrasts는 다음처럼 양수다. 이를 FP04의 near-zero feature response와 같은 결과라고 지워서는 안 된다.

| Graph | PG95 acc % | PG none % | PG shuffle95 % | 95−none %p | 95−shuffle95 %p |
|---|---:|---:|---:|---:|---:|
| Cora | 45.459 | 24.325 | 42.274 | +21.134 | +3.185 |
| CiteSeer | 28.334 | 17.154 | 26.949 | +11.180 | +1.385 |
| PubMed | 61.671 | 36.903 | 59.600 | +24.768 | +2.071 |

이 개발 패널에서는 입력 관측을 제거/셔플하면 pretrained-PCFI 경로의 점수가 낮아진다는 실제 descriptive 증거다. 다만 frozen RP-PCFI의 95−none도 +19.923/+11.597/+30.385%p이고, fixed는 +19.756/+10.736/+22.595%p다. 일부 contrasts는 PG가 더 크지만 일부는 더 작다. 모두 같은 PCFI front end를 사용하고 greedy000이므로 adaptive routing이나 PG 고유 feature increment가 확인된 것은 아니다. Synthetic disjoint utility FAIL을 이 real 패널로 덮지 않는다.

LP는 모든 9 graph/rate accuracy에서 PG보다 높지만 확률 metric에서는 일괄 우위가 아니다. 예를 들어 Cora95는 PG NLL1.50582/Brier0.68482가 LP1.80715/0.80647보다 낮다. 반면 frozen RP-PCFI의 Cora95 NLL1.46597/Brier0.66560은 PG보다 좋다. PG95 BAcc는 Cora51.085/CiteSeer30.427/PubMed60.975%, macro-F1은45.174/26.723/56.041%다. PubMed accuracy61.671%만으로 minority-class 문제를 해결했다고 말할 수 없으며 fixed의 같은 지표도 각각61.429/56.872%다.

기존 native-input public references도 약한 plain PFN만 남기지 않고 보존한다. 아래95% 결과는 기존3 repetitions의 값이며 이번 새 forwards/학습 비용에 포함하지 않는다.

| Graph | Plain public PFN % | PCFI+public % | FISF+public % | Harmonic LP % | Restart LP % |
|---|---:|---:|---:|---:|---:|
| Cora | 17.023 | 54.796 | 54.347 | 38.246 | 59.056 |
| CiteSeer | 14.614 | 34.078 | 34.189 | 34.228 | 38.410 |
| PubMed | 40.629 | 55.889 | 56.091 | 51.248 | 73.690 |

Cora995 FISF56.902%, CiteSeer995 PCFI+public39.074% 등의 rate별 강한 결과도 원 historical table에 남아 있다. PubMed95에서 PG가 native public feature PFNs보다 높은 점추정은 사실이지만 LP와 frozen RP-PCFI는 더 높다. 원 physical rows/context/query가 연결돼도 native/public preprocessing·표현·capacity·pretraining/ensemble 예산은 다르다. 따라서 matched learned-arm 비교와 외부 estimator 비교를 동일한 통제 실험으로 부르지 않는다. 기존 외부225 records는 FP04 때 전수 identity/rescore됐고 이번에는 그 summary SHA와 보존 경계를 확인한 재사용 근거다. 새30,828 records 안에 외부225를 다시 실행한 것처럼 더하지 않는다.

## 6. 실제 예산과 감사의 재현 범위

Train bank는256 graphs/1792 episodes지만 각 seed의128 groups가 실제 만난 unique bases는107/101/101이다. Arms는 동일 schedule을 사용한다. Full-natural groups24/29/22로 예약16을 충족했고, seed별 group-level own-observed/missing query exposures는460/1588,516/1527,423/1620이다. Regime50:50가 observed-query50%를 뜻하지 않는다.

12 fits 합계1536 successful groups,12,288 logical trace slots이다. 실제 transformer forward/backward는 **9600회씩**(fixed384, 다른 세 arms3072회씩)이며 retry는0이다. Fixed의 같은000 8 presentations를 한 번 계산한 것은 같은 mean loss/gradient를 보존하지만 **equal logical exposure를 equal compute로 만드는 것은 아니다.** Adapter3,564,288 parameters를 continuation했고 direct/PG는 policy513 parameters가 추가됐다. Frozen LimiX/backbone을 RL로 다시 학습한 실험은 아니다.

| 기록된 비용 항목 | 실제 기록 |
|---|---:|
| Scientific training wall 합 | 2811.498s |
| Fixed / uniform / direct / PG, 각3 seeds wall 합 | 115.781 / 899.279 / 891.809 / 904.628s |
| Dev global calibration | 2688 forwards,197.033s |
| Trained final+real evaluation wall 합 | 3690.870s |
| 전체 deployment(actual model forwards) | 51,747 forwards |
| Preparation wall 합 | 136.076s, 그 안 PCFI17.698s |
| Recorded frozen-reference per-record seconds 합 | 138.878s |
| 위 wall components 합 | 6974.355s |

Calibration은6 fits×dev56 episodes×8 branches다. Prep의 PCFI를 다시 더하지 않았다. 성공 training+calibration+deployment의 실제 모델 forward 합은64,035회이며 scientific optimizer groups와 다른 분모다. LP/prior CPU reference 시간 일부는 기록되지 않았고 checks/dev smoke·별도 감사·과거 public pretraining 비용도 위 합에 포함하지 않는다. Component 합은 critical-path 전체 elapsed나 반복 측정한 robust latency benchmark가 아니다. Concurrent CPU hash/audit 영향이 있을 수 있다. Training allocated peak 최대245.373MiB, real evaluation peak 약480.496MiB는 해당 PyTorch timer/allocated 범위이며 전체 GPU memory 사용량으로 확대하지 않는다.

Uniform8은 같은 test episode에 8 branches를 계산하므로 작은 accuracy 점추정 증가가 free gain이 아니다. Direct/PG greedy는000 한 번이지만 training에서 fixed보다 훨씬 많은 branch computation을 썼다. 따라서 이 실험은 같은 physical schedule/optimizer groups로 objectives를 비교한 것이며 equal-FLOP/equal-wall comparison은 아니다.

## 7. 제한과 종료 결정

입력 생성은 무조건부 SCM 전체를 그대로 포함하지 않는다. Official prior의 feature/finite/train/eval class-coverage sanity filters와 task-type selection을 상속했다. Query segment labels가 synthetic acceptance에 전혀 쓰이지 않았다는 포괄적 표현은 부정확하지만, 그 labels가 policy/predictor 입력에 전달됐다는 뜻은 아니다. 실제 N/F는 요청값과 다르기도 하고 outer retry가 원 sampler 내부 모든 rejection을 세는 것도 아니다. 이 제한은 새로운 performance filter가 있었다는 증거가 아니다.

Train/dev/real 준비 시점 `rl_util.py` old bytes 미보존 제한은 `PRELOCK_SOURCE_DELTA.json`/`BUILD_EXECUTION.md`에 명시됐다. 다른 numeric data dependencies와 실제 저장 입력 감사는 일치하고 final 및 scientific fits는 current design/full lock에 연결된다. 그러나 그 사실이 이전 utility bytes까지 완전히 재현됐다는 주장을 허용하지 않는다. 원 receipts/banks를 보존하고 근거 없는 old snapshot을 만들지 않은 대응이 적절하다.

현재 감사에서 **원결과를 바꿔 재실행해야 할 추가 구현/통계 repair는 발견하지 못했다.** 원인 규명을 가장해 cost weight·entropy·layer choices·calibration·예산·mask 분모를 final 결과에 맞추는 구제 튜닝도 권고하지 않는다. Reporting에는 (1) nominal CI와 서로 다른 분모, (2) replay000의 정확한 일치, (3) frozen/global/LP/prior 및 real 양의 feature contrasts, (4) preparation provenance/비용 한계를 함께 남기면 된다.

종료되는 주장은 **PCFI100 앞단과 released GraphPFN adapter continuation을 공유하는 3-action 재계산 controller, 이 공식 restricted prior/96 final tasks, 128 updates×3 seeds에서 PG가 fixed/exact expectation보다 낫고 관측 효용을 높인다는 가설**이다. 실제 real 관측 효용까지 부정하지 않으며 모든 missing-robust PFN, 모든 RL/computation policy, 충분한 backbone 학습 또는 비graph 적용의 불가능성으로 일반화하지 않는다. 이후 새 연구가 있다면 이 final bank와 반복 real 패널은 이미 관찰된 개발 근거이며, 별도 가설·사전 비교·새 확인 bank로 분리해야 한다.

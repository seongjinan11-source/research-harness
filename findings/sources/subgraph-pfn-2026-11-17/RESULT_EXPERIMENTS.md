# SPFN01 최종 실험·통계 검토

2026-11-17. **저장 결과 및 사전 통계 재검산 PASS, 과학적 복합 gate FAIL.** 같은 token 예산에서 `tail`은 PPR+feature rank `top`보다 synthetic tail recall을 높였고, 모든 anchor를 쓰는 `full` 대비 overall BAcc 비열등성 guard를 통과했다. 그러나 같은 그룹 압축을 쓰는 `proportional`에 대한 희소 라벨 allocation 우위는 확인되지 않았다. 세 실제 개발 그래프의 overall BAcc에서는 balanced PPR이 앞섰다. 현재 후보는 사전 조건에 따라 종료하되, 그룹 압축의 synthetic 결과는 보존한다.

검토자는 v1의 실험·통계 심사자였고 모델/데이터/훈련 코드에는 관여하지 않았다. Locked 결과를 읽고 production modules를 호출하지 않는 CPU NumPy로 저장 확률의 metric 및 CI를 재구성했다. 원자료는 별도 CPU 로드로 확인했다. 새 모델 forward·학습·GPU·bank/소스/branch 작업은 없었고 이 문서만 작성했다.

## 저장 확률·분모·학습 예산

- `results/lock.json`의 source 7개 및 native graph 3개의 SHA256, manifest의 checkpoint 12개와 prediction NPZ 111개의 SHA256이 모두 일치했다. Fit receipts마다 320 updates, finite loss/gradient log, seed별 공통 initialization digest 및 permutation을 확인했다. 이는 raw checkpoint bytes와 기록의 audit이며 별도 optimizer replay는 아니다. Production audit에서 checkpoint 내부의 lock 연계·로드도 검사했다.
- 96 synthetic + 15 real episodes의 NPZ에 각각 4 learned arms ×3 fixed seeds와 4 no-fit rules가 있어 **1,776 probability arrays**를 전수 확인했다. Query row 수는 seed/arm 중복 전 synthetic 3,144와 real 2,560이다. Ordered C/Q의 중복·교집합, context label counts, 공통 y와 probability dimensions를 검사했다. Row sums 최대 절대 오차 2.26e−7, accuracy/BAcc/tail/head recall/NLL/Brier 평균의 원 summary 대비 최대 차이 **8.88e−16**이다.
- Synthetic split seed들은 train320/dev12/final96으로 모두 분리됐다. 320 train bank를 seed별로 한 번 순회하는 **12 fits ×320 = 3,840 successful updates**, arm마다 head 3,705 parameters다. Fixed seeds 및 동일 optimizer work units를 맞췄으나, arm별 pair/diffusion FLOPs까지 같지 않다. Target graph에서는 optimizer updates 0이다. Ridge는 사전 고정 λ의 context-only 해석적 fit이지 target gradient 학습이나 validation 튜닝이 아니다.
- 초기 Windows index dtype 오류와 후속 test의 torch/NumPy assertion 오류는 모두 prelock에서 수정됐다. 첫 오류는 optimizer 전, 둘째는 discarded dev update 1회 뒤였으며 최종 checks가 별도 discarded dev update 1회/arm을 기록한다. Scientific budget과 섞지 않는다. Numerical correctness의 PASS는 아래 효과의 PASS를 뜻하지 않는다.

앵커: lock SHA256 `18e8aba6244630f5257fe2f92e44250276661d19e08d41ebb8c98efdf3796ea1`, artifact manifest `07afee7451a733b29e7a051926ba1676a7471bac51a85bfeb2ca8a5ecef5c9d2`, summary `921ced6618dec806576dea7fe224ddee92a696aef080bb7a5165a7a2b0be9c55`.

## 합성 가설·대조군·gate

Primary는 **관측 context 1–4개 클래스**의 class recall 평균이다. Tail은 true class population minority를 뜻하지 않는다. 각 graph에서 truth class당 정확히 8 query가 있어 overall accuracy와 BAcc는 여기서는 일치한다. 실제로 사용한 양은 먼저 episode별 metric, 다음 고정 seed11/12/13의 metric 평균이다. 확률 ensemble은 아니다. Final96 base graphs는 shape×relation×feature regime의 8 strata에서 각각 12 graphs다. Paired contrast는 이들 graph IDs를 strata 안에서 5,000회 복원 추출했고, locked seed17987100/101/102 및 nominal two-sided 97.5% endpoints를 별도 구현으로 재현했다(차이/CI 최대 차이 4.34e−19). Seed나 queries를 독립 표본으로 세지 않았고, 학습 seed 모집단이나 3개 비교 전체의 동시 95% coverage를 주장하지 않는다.

| 사전 판정 | 차이 또는 비율 | 명목 97.5% CI | 판정 |
|---|---:|---:|---|
| Tail − proportional, tail recall | +0.6583%p | [−0.3400,+1.6565]%p | **FAIL** |
| Tail − top, tail recall | +2.6114%p | [+1.1826,+4.0656]%p | PASS |
| Tail − full, overall BAcc | +0.0231%p | [−0.6160,+0.6775]%p | PASS, lower > −2%p |
| Holdout48, mean(full C) / mean(tail B) | 59.1875 / 9.6458 = **6.136×** | count assertion | PASS |
| Tail vs 최고 no-fit rule tail recall | 74.5081% vs ridge 61.6319% | descriptive floor | PASS |

`tail−proportional`은 양의 점추정이나 CI에 0이 포함되어 사전 우월성 기준을 충족하지 못했다. 동등성이나 열등성이 확인된 것은 아니다. `tail−top`은 같은 graph/features/labels와 per-class quota·decoder를 둔 비교에서 **선택 대신 그룹 집계**가 유용했다는 근거다. `top`은 전체 관측 anchors를 읽고 finite PPR과 feature cosine rank를 결합하므로 PPR-only 약대조군은 아니다. 다만 두 구현은 mass/count/centroid representation과 학습된 head가 다르다. 이 결과는 개별적인 PCA, PPR moment 또는 count weighting의 독립 인과효과나 allocation 우위까지 확립하지 않는다. `tail−full`은 **전체 96 graphs overall** guard이므로 holdout48 alone이나 tail recall에 별도 비열등성을 선언하지 않는다. Full을 이론적 상한으로 해석하지도 않는다.

| Synthetic arm | Overall BAcc % | Tail recall % | Head recall % | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| Tail | 74.682 | 74.508 | 75.109 | 0.6354 | 0.3338 |
| Proportional | 74.728 | 73.850 | 76.367 | 0.6367 | 0.3339 |
| Top | 71.591 | 71.897 | 71.181 | 0.7153 | 0.3716 |
| Full | 74.659 | 75.123 | 74.067 | 0.6473 | 0.3399 |
| Balanced PPR | 34.607 | 28.798 | 44.336 | 1.5778 | 0.7524 |
| Centroid | 69.705 | 59.071 | 86.523 | 1.1379 | 0.6050 |
| Ridge | 71.901 | 61.632 | 88.411 | 1.0648 | 0.5632 |
| Uniform | 25.781 | 24.306 | 28.646 | 1.3830 | 0.7422 |

Synthetic head recall에서 보이는 centroid/ridge의 높은 값과 tail의 full 대비 −0.6149%p도 보존해야 한다. 전 조건/지표에서 tail이 지배한다고 주장할 수 없다. Uniform은 모든 확률이 같아 argmax tie가 특정 class에 고정되므로 overall은 1/K, tail/head는 그 class의 role에 따라 달라진다. Class-balanced query CE와 균형 query에서의 NLL/Brier는 자연 유병률·native class prevalence에서 보정된 posterior라는 근거는 아니다.

## 실제 개발 자료: 15 paired draws와 강한 PPR

V2의 실제 결과는 **Cora2708, CiteSeer3327, full PubMed19717**이다. V1의 induced PubMed2000 가정이나 과거 missing-feature benchmark와 범위가 다르다. Native files의 official train/test가 분리되고 train에는 class당 20개 label이 있음을 별도 재확인했다. 동일 RNG로 fixed draw 15개의 `C,Q,y_C,y_Q` 순서까지 native source와 정확히 재현했다. Context only는 official train에서 2 vs 16개, class당 32 queries는 test에서 뽑았다. Ceil(K/2)개 observed-tail classes이므로 각 draw의 `(C,B,Q)`는 Cora `(56,14,224)`, CiteSeer `(54,12,192)`, PubMed `(20,6,96)`이다. 모든 내부 arms가 label-free RP24→standardization으로 동일한 complete projected X를 받았다. Synthetic K≤5에서 real Cora7/CiteSeer6, N/F 및 feature/graph distribution도 변하지만 특정 실패 원인을 식별한 비교는 아니다.

Learned 수치는 graph별로 고정 training seeds 3개의 metric와 paired draws 5개를 평균했다. Rules는 5 draws다. **3 reused development graphs**이며 15 또는 45 independent real graphs가 아니다. 균형 query에서의 기술적 수치이며 native public checkpoint benchmark나 population CI가 아니다.

| Graph | Overall BAcc %: tail / proportional / top / full / PPR | Tail recall %: tail / proportional / top / full / PPR |
|---|---|---|
| Cora | 41.369 / 42.411 / 38.393 / 39.762 / **62.946** | 43.438 / 48.906 / 45.521 / 48.385 / **59.844** |
| CiteSeer | 24.896 / 23.056 / 25.417 / 23.507 / **42.604** | 31.250 / 32.083 / 28.819 / **34.583** / 31.042 |
| PubMed | 48.125 / 47.639 / 45.764 / 46.042 / **60.000** | 51.042 / **52.708** / 49.479 / 52.396 / 52.188 |

Balanced PPR의 overall BAcc는 tail보다 각각 **+21.577/+17.708/+11.875%p** 앞섰고 Brier도 세 graph 모두 낮다. Tail recall에서는 Cora/PubMed에서 PPR이 tail보다 높지만 CiteSeer에서는 tail이 +0.208%p 앞선다. PPR NLL은 Cora 1.0991로 tail 1.5906보다 낮으나, CiteSeer 4.5177와 PubMed 1.4847은 tail 1.8246/1.0190보다 높다. 저장 PPR에 truth class 확률이 정확히 0인 queries는 Cora 1/1120, CiteSeer 111/960, PubMed 5/480이다. All-class zero-mass fallback은 uniform이지만 다른 class에 mass가 있을 때 truth class zero는 남을 수 있고, scoring floor 1e−12가 NLL에 크게 반영된다. 원 결과를 보고 PPR만 다시 smooth하거나 지표를 교체하지 않는다. Synthetic rule floor PASS를 이 real 결과의 강기준선 우위로 바꿔 쓸 수도 없다.

`tail`의 real tail recall은 세 graph 모두 proportional보다 낮다. `top` 비교도 Cora −2.083%p, CiteSeer +2.431%p, PubMed +1.563%p로 일관된 양의 방향은 아니다. PubMed overall에서 learned arms 중 tail 점추정이 가장 높은 사실은 남기되, 그 자체로 real transfer 우월성을 주장하지 않는다. Centroid/ridge/uniform도 원 결과에 모두 포함되며, 이 cycle에는 NodePFN/GraphPFN 등 released checkpoint 비교가 없으므로 이들에 대한 우위를 추정할 수 없다.

## 비용: 줄어든 배열과 늘어난 연산을 구분

Locked CPU runner는 one thread, draw0/seed11 각 arm의 고정 warmup 후 3회의 `prepare + score`를 기록했다. Native raw load/RP를 반복 timer에 넣지 않고 이미 projected X에서 시작했다. 원 projection timer는 세 graph 순서로 2.64/8.61/7.01 ms (graph당 한 번, 모든 draws에 재사용). 아래는 **stand-alone post-projection** median [min,max], 단위 ms다.

| Graph | Tail | Proportional | Top | Full |
|---|---:|---:|---:|---:|
| Cora | 41.64 [41.48,43.10] | 84.34 [83.41,85.60] | 117.15 [116.17,146.45] | 23.07 [23.02,23.20] |
| CiteSeer | 46.76 [46.18,51.34] | 87.08 [84.76,87.50] | 96.81 [94.37,99.00] | 20.86 [20.81,21.59] |
| PubMed | 112.49 [110.15,114.06] | 318.01 [303.79,353.34] | 52.61 [49.20,55.12] | 32.48 [31.33,32.55] |

Tail은 세 graph 모두 `full`보다 scoring은 빠르지만 post-RP end-to-end는 느리다. 예를 들어 PubMed에서 score median은 0.862 vs 1.247 ms, diffusion median은 97.424 vs 15.848 ms다. Tail은 PubMed에서 full과 top 모두보다 느리고, Cora/CiteSeer에서는 top의 rank/retrieval이 커 tail이 top보다 빠르다. Tail의 moment 확산 columns는 real draw0에서 446/444/150, full·top singleton mass columns는 56/54/20이다. PubMed pair tensor는 tail 36,864 vs full 122,880 bytes로 줄지만, largest seed block은 11,514,728 vs 315,472 bytes로 커진다. 다른 두 graph에서도 같은 방향이다. 이 값은 개별 배열 크기이며 peak process RSS 또는 모든 temporary tensors의 합은 아니다. PPR 등 4 rule 시간이 합쳐져 기록됐으므로 이 자료로 PPR의 stand-alone latency를 추정하지 않는다.

Train-bank/cache preparation 5.892s, 12 fits 기록 합 7.850s, scientific run start부터 production audit 직전까지 30.433s다. 마지막 수치에 앞의 구성 시간이 포함되므로 더하지 않는다. 이는 해당 CPU 작업 범위의 관찰이고 시작 전 검사·실행기 startup·최종 audit 전체 wall이나 독립 반복 latency benchmark가 아니다. Real profile도 class-balanced 일부 Q(최대224)의 한 fixed draw/graph이며 all-node scoring, ogbn-arxiv, N-independent search, global memory 절감 주장으로 확대할 수 없다.

## 판정과 후속 범위

재채점·pairing·CI·provenance에서 결과를 수정하거나 현재 fit을 다시 돌릴 오류는 발견하지 못했다. 결과가 뒷받침하는 범위는 **complete features·모든 클래스가 관측된 고정 prior에서 동일 token budget의 raw retrieval보다 그룹 압축이 나았고, full 대비 전체 합성 BAcc guard를 만족했다**는 것이다. 희소 라벨에 토큰을 우선 배정하는 추가 효과, 실제 그래프에서 PPR을 넘는 overall 성능, preprocessing을 포함한 실행비용 우위는 확인되지 않았다. 이 결과만으로 새로운 방법론, Bayes calibration 또는 feature missingness 대응을 주장할 수 없다. 같은 final bank나 이 세 graph에서 quota·온도·head를 사후 골라 현재 gate를 구제하지 않는다. 다음 후보가 있다면 지금의 압축/선택 차이와 실제 PPR 격차를 새 가설에 반영하고, 독립 계획·새 평가 자료를 사용해야 한다.

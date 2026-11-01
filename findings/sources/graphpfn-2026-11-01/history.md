# Missing-feature GraphPFN: 실험 요약과 현재 판단

> **최신 학습 — Evidence-Amortized Joint PFN (23절):** [고정 BP 위의 shared evidence 학습](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/REPORT.md)을 완료했다. 18개 모델·3,456 updates에서 density-high Coupled KL은 frozen-grid0.07808→EAJ0.03608, independent heads0.04407, exact joint0.01762였다. Density-high/signal-high6개 prior×condition에서 independent/analytic/own-marginal product보다 모두 낮은 KL을 얻었다. 그러나 exact marginals의 곱보다 여전히 나쁘고, strong constant+updated-role 대조에 대한 범용 우위도 없다. Posterior-only는약1.9–2.9배 빠르나 같은 BP를 포함한 시간 감소는약6–12%다. 학습 증분은 확인, exact-grid 대체·확장은 보류한다.

> **최신 검증 — Joint Role–Parameter Grid BP (22절):** [동일 full-grid BP의 joint/frozen/marginal-product 비교](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/REPORT.md)를 완료했다. Density-high Coupled KL은 기존 MRP0.11282, parameter-grid+frozen q0.07808, coherent joint BP0.01762였다. 같은 role·parameter marginals를 곱한 대조0.01905 대비 추가 joint 이득도 확인됐다. Exact decoder뿐 아니라 BP에서도 evidence update의 Conditional Go가 유지된다. 다만 known-family 계산 기준선이고, prior에 따른 실패와 matched의 강한 constant+updated-role 대조가 남는다. 새 학습은0회다.

> **최신 검증 — Known-family Parameter Generalization (21절):** [학습 없는 통계·분포 추정기 비교](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/REPORT.md)를 완료했다. Density-low Exclusive에서 현재 point KL0.00725→관측 밀도 추정0.00129, broad distribution0.00208이었다. 분포 주변화는 특정 조건에서 posterior mean보다 좋지만 가장 강한 통계 대조군을 전반적으로 넘지 못했다. 같은 parameter-integrated exact conditional에서 role posterior만 coherent하게 바꾸면 density-high Coupled0.07936→0.01215로 감소해 frozen role q의 OOD 병목도 확인했다. 새 distributional head의 Go 근거는 아직 부족하며, 새 학습은0회다.

> **최신 학습 — Learned Mask-aware Potential (20절):** [참 SBM potential을 공유 신경망으로 대체한 MVP](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/REPORT.md)를 완료했다. 기존 posterior는 고정하고36개 모델을 같은192-update budget으로 학습했다. 참 density/signal을 입력하지 않아도 matched Exclusive KL0.05259→0.01127, oracle potential+동일 q는0.01007이었다. 세 prior에서 Positive·새 Large Plain·learned signed보다 KL이 낮았다. 하지만 추가 likelihood loss는 predictive-only보다 Independent/Exclusive에서 작게 악화됐고, Coupled는 fitted constant도 못 넘었다. Density 외삽의 실패도 있어, 학습 가능한 decoder 치환은 Go지만 새 likelihood 학습 원리의 우월성과 범용성은 아직 주장하지 않는다.

> **최신 표현력 검증 — Contrastive Relational Decoder (19절):** [새 학습 없는 native likelihood decoder 검증](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/REPORT.md)을 완료했다. 새 graph768개에서 같은 frozen learned posterior를 유지한 Exclusive Bayes KL이 positive D0.04820→contrastive BP0.00867로 줄었다. Exact joint posterior를 주면0.00248이며 같은 decoder의 factorized0.00783보다 낮다. 기존 positive-only 함수족의 부분 KL 하한도 넘었다. 다만 알려진 SBM 확률을 직접 쓴 oracle potential이므로 학습된 새 PFN의 성과로 주장하지 않는다. Decoder 원리는 Go, 실제 likelihood potential을 학습하는 메소드는 아직 미검증이다.

> **최신 학습 — Role-Conditional Distillation (18절):** [동일 JR 아키텍처의 exact-teacher ablation](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/REPORT.md)을 완료했다. 새63개 모델과 초기화·과제·update가 같은 기존18개 기준선을 비교했다. Conditional 감독은 component KL을 줄였지만 최종 predictive KL의 독립적인 C−B 증분은 미확인이고, posterior 감독을 추가한 D는 predictive KL이 커졌다. 현재 decoder에 loss만 추가하는 방식의 Stage2 확대는 보류한다. Role posterior 정보가 예측에 유용하다는 이전 결과를 철회하는 것은 아니다.

> **최신 분석 — Prior-to-Posterior Shift (17절):** [Exact Bayes 정보량·예측 가치 분석](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/posterior_shift/REPORT.md)을 완료했다. Exclusive·총 라벨 4개에서 graph를 넘는 label 추가 정보가0.12447 nats이고 exact posterior 집계의 기대 NLL 이득이0.02278이다. 같은 exact conditional predictor에서 learned posterior도 약53%를 회수한다. 현재 accuracy 증분이 작다고 정보가 없거나 learned posterior가 전혀 작동하지 않는 것은 아니다. Conditional Go를 유지하되 posterior와 conditional decoder의 두 병목을 구분한다. 새 학습0회, 기존 결과 불변이다.

> **최신 추가 — JR-PFN (16절):** [정확한 Bayes 기준과 학습 pilot](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/REPORT.md)을 완료했다. 상관된 역할 prior를 보존할 때 Exclusive에서 큰 Plain 대비 +3.20%p, factorized 대비 +1.76%p였으나 Coupled에서는 개선이 없었다. 같은 decoder에서 학습 posterior가 고정 known prior에 더한 정확도는 Exclusive +0.14%p로 불확실했다. 알려진 상관구조의 보존 효과와 새로운 posterior 학습 원리의 기여를 구분한다. 66개 소형 모델의 개발용 합성 실험이며, unseen-schema·실데이터의 새 메소드로 확정하지 않는다.

> 정리일: 2026-10-31 · 목표: 관측 feature가 매우 적은 새로운 그래프에서, target별 추가 학습 없이 분류하는 GraphPFN.
>
> 아래는 이 연구 방향의 로컬 실험 요약이다. 실데이터의 frozen-model 비교와 새 모델의 합성 실험을 구분했다. 루트 저장소의 별도 시계열 연구는 포함하지 않았다.

> 최신 판정: LP-conditioned residual과 현재 Evidence Head는 종료한다. [Plain의 exact-degree role reversal](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/REPORT.md)에서 기존 소형 Plain이 이미 context 통계에 따라 전역 관계 utility를 추적함을 확인했다. 새 학습은0회이며, 낮은 context(k1)의 전역 label-alignment 통계 퇴화가 구체적인 한계 후보로 남았다. 공식 pretrained GraphPFN이나 새로운 semantic schema의 일반화 증거는 아니다. Gate A/B는 9절, intervention은 10절, residual은 11절, 분리 진단은 12절, kill test는 13절, frozen 확률 진단은 14절, 역할 교환은 15절에 있다.

## 1. 지금 무엇이 베스트인가?

**현재 검증된 최선은 데이터셋·라벨 예산·평가 지표에 따라 다르다. 모든 그래프에서 이기는 새 방법은 아직 없다.**

| 사용 목적 | 현재 판단 | 근거와 주의점 |
|---|---|---|
| 실제 missing-feature PFN의 기본 출발점 | **기존 GraphPFN + 원본 좌표 PCFI/FISF**를 유지 | 강한 비교 기준이다. 동질 그래프 전체 조건 평균은 PCFI가 우세했고, 극심한 node-wise 결측 일부에서는 FISF가 우세했다. |
| ACM의 극심한 결측 | **Metapath별 LP 확률 평균**이 가장 강한 기준선 | 라벨 20개/클래스, 99.5% 결측에서 전체 test 정확도 85.55%. feature 없이도 강하므로 반드시 비교해야 한다. |
| IMDB의 극심한 결측 | **원본 FISF + typed paths + GraphPFN** | 44.55%, LP 35.41%. 실제 관측 피처를 모두 제거한 FISF+PFN도 44.51%여서, 우위를 실제 관측값의 순이득으로 설명할 근거는 약하다. |
| Feature가 원래 없는 Freebase | **RDB-PFN DFS5와 구조 입력 GraphPFN을 모두 유지** | 전체 test 정확도는 56.60% 대 56.13%로 근소하다. Balanced accuracy와 macro-F1은 GraphPFN이 높고, 256-query 부분집합에서는 정확도 순위도 바뀐다. |
| 추가 학습 없는 입력 개선 후보 | **PCFI/FISF 입력 view 조합**이 후속 확인할 만한 후보 | PubMed의 secondary 평가에서 FISF 대비 평균 +1.50%p. Cora/CiteSeer의 전체 조건에서는 일관된 증분 이득이 없었다. |
| 새로 만든 결합 추론 모델 | **양방향 피드백은 보류** | 같은 구성의 일방향 모델과 평균 정확도가 거의 같다. 현재 이를 메인 기여로 확정하거나 확대 학습할 근거는 부족하다. |
| Relation Evidence PFN 소형 후보 | **현재 evidence head는 확대하지 않음** | Conflict에서 Separate가 Mean보다 +0.74%p 높지만 Plain보다 −2.88%p다. NLL/Brier는 더 좋으며 정확도 우위와 구분한다. 공식 GraphPFN·실데이터 비교가 아니다. |

위의 데이터셋별 우승 방법은 **사후 결과 요약**이다. Test 정답 없이 새로운 그래프에서 이 방법을 자동 선택하는 규칙을 확보했다는 뜻은 아니다.

**권장 출발점:** 기존 frozen GraphPFN의 PCFI/FISF 파이프라인을 유지하고, feature 없는 LP와 구조 입력 모델을 함께 비교한다. 새 연구의 핵심은 이 강한 기준선들이 놓치는 조건을 확인한 뒤 정한다.

## 2. 가장 중요한 실데이터 비교

### 2.1 전체 public test 기준 — 결과 해석의 주 기준

라벨 20개/클래스, context·mask seed 11/12/13의 평균 정확도(%). ACM·IMDB의 missing 조건은 target feature의 **99.5% whole-node 결측**이다. 모든 방법은 target 그래프에서 가중치를 업데이트하지 않았다.

| 방법 | ACM full | ACM 99.5% 결측 | IMDB full | IMDB 99.5% 결측 | Freebase: 원래 feature 없음 |
|---|---:|---:|---:|---:|---:|
| GraphPFN: full / missing zero + paths | 86.12 | 72.49 | 46.20 | 40.41 | 56.13 |
| PCFI + paths + GraphPFN, 원본 좌표 보간 | — | 75.45 | — | 42.61 | — |
| FISF + paths + GraphPFN, 원본 좌표 보간 | — | 74.36 | — | **44.55** | — |
| 실제 관측값 없는 FISF + paths + GraphPFN | — | 74.14 | — | 44.51 | — |
| Feature 없는 metapath LP 평균 | 85.55 | **85.55** | 35.41 | 35.41 | 50.90 |
| RDB-PFN: 공식 기본 DFS 깊이 2 | 74.87 | 35.18 | 42.17 | 27.64 | 40.50 |
| RDB-PFN: mean/count DFS 깊이 5 | 81.53 | 66.55 | 42.57 | 34.88 | **56.60** |

해석:

- **ACM:** 이 설정에서는 구조와 context 라벨만으로 이미 강한 예측이 가능하다. Missing PFN의 full 대비 하락만으로 새 모델의 필요성을 주장하기 어렵다.
- **IMDB:** FISF+PFN의 44.55%는 실제 관측값을 제거해도 44.51%로 거의 유지됐다. 현재 LP 대비 우위 대부분을 실제 관측값의 순이득으로 설명하기 어렵다. 관측값이 query별 예측에 영향을 주지 않는다는 뜻은 아니다. 구조·합성 입력 경로와 숫자값 경로를 구분해야 한다.
- **Freebase:** 결측 강건성의 검증 대상이 아니라 feature 없는 구조 추론 대조군이다.
- RDB-PFN의 DFS5는 기본 DFS2와 구분한 확장 설정이다. 공식 기본 설정의 결과로 표현하면 안 된다.

Freebase의 지표별 비교는 다음과 같다.

| 방법 | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| 구조 입력 GraphPFN | 56.13 | **57.30** | **55.27** |
| RDB-PFN DFS5 | **56.60** | 52.70 | 52.86 |

Freebase는 public train에 클래스당 라벨 20개만 있어서, k=20의 세 seed에서 context는 동일하다. 독립적인 세 번의 라벨 추출로 해석하면 안 된다.

### 2.2 RT와 정확히 같은 query에서의 비교

그래프별 **고정 test query 256개**, 라벨 20개/클래스, 세 context/mask seed의 평균 정확도(%). 기존 방법도 같은 query에서 다시 채점했다. 이 표의 값은 위 전체 test 표와 섞어 쓰지 않는다.

| 방법 | ACM 99.5% 결측 | IMDB 99.5% 결측 | Freebase |
|---|---:|---:|---:|
| GraphPFN | 72.92 | 41.02 | **57.03** |
| PCFI + GraphPFN | 75.65 | 42.32 | — |
| FISF + GraphPFN | 73.57 | **43.23** | — |
| Metapath LP 평균 | **85.16** | 33.98 | 49.22 |
| RDB-PFN DFS5 | 65.76 | 35.16 | 55.86 |
| RT-PluRel | 42.71 | 26.69 | 41.54 |
| RT-J | 29.56 | 32.68 | 14.06 |

동일 query에서 full 입력의 정확도:

| 방법 | ACM full | IMDB full |
|---|---:|---:|
| GraphPFN | 85.55 | 43.49 |
| RDB-PFN DFS5 | 80.34 | 41.02 |
| RT-PluRel | 66.28 | 36.59 |
| RT-J | 72.01 | 40.62 |

**RT 결과의 범위:** 공식 공개 가중치를 사용했지만, native graph를 관계형 cell로 바꾸는 Python 어댑터와 멀티클래스 OVR 확장을 사용했다. Windows에서 Rust sampler를 사용할 수 없어 별도 sampler를 구현했고, attention은 동등한 dense 연산으로 교체했다. 전체 forward 수치 검사는 통과했다. 이 결과를 원논문의 원래 benchmark 성능이나 모든 heterogeneous graph에서의 성능으로 일반화하지 않는다.

특히 Freebase의 RT-J 정확도 14.06%에 대응하는 balanced accuracy는 33.33%다. 결측 RT 결과의 NLL도 대체로 3-class 균등 예측의 log(3) ≈ 1.099에 가깝다. 낮은 accuracy만 보고 강한 반대 방향의 예측을 학습했다고 해석하면 안 된다.

### 2.3 RT sampler의 관측 정보 누락 진단

- IMDB, seed 11, 99.5% 결측에서 그래프 전체에는 관측 feature 행이 19개 있었다.
- 최초 256-query 어댑터에서는 약 94.92%의 샘플 안에 관측 feature 행이 하나도 없었다.
- 별도 32-query/seed-11 대조군에서는 남은 4,096-cell 예산을 native BFS로 채웠다. ACM·IMDB 모두 무관측 샘플 비율이 0%가 됐다.
- 그 대조군에서도 결측 예측 확률은 대체로 균등 예측에 가까웠다. RT-J의 결측 정확도는 ACM 25.00%, IMDB 34.38%였다.

**판단:** 최초 점수에는 sampling coverage의 한계가 있다. 작은 예산 채우기 대조군에서는 coverage를 개선했지만 큰 회복을 입증하지 못했다. Sampler만으로 해결된다고도, 모델 자체가 일반적으로 실패한다고도 결론 내리지 않는다.

출처: [Relational baseline 결과](C:/Users/seongjin/Desktop/code/gts_pfn/relational_baselines/RESULTS.html), [동일 query 요약 CSV](C:/Users/seongjin/Desktop/code/gts_pfn/relational_baselines/rt_expanded_results_summary.csv), [전체 test RDB 요약](C:/Users/seongjin/Desktop/code/gts_pfn/relational_baselines/native_rdb_summary.csv).

## 3. 동질 그래프: PCFI/FISF와 입력 확장

### 3.1 전파 기반 보간의 효과는 확인됐다

Cora/CiteSeer에서 raw feature를 먼저 가린 다음 보간·변환·PCA를 적용하고, 공개 frozen GraphPFN을 사용했다. 90%/99.5%, node-wise/entry-wise 결측, 각 세 mask를 비교했다. 이 단계는 해당 public split의 context를 사용하므로, 앞의 heterogeneous 라벨 예산 표와 직접 비교하지 않는다.

99.5% whole-node 결측의 기본 비교:

| 데이터 | FP + GraphPFN | PCFI + GraphPFN | FISF + GraphPFN |
|---|---:|---:|---:|
| Cora | 73.15 | 73.08 | **73.62** |
| CiteSeer | 62.09 | 62.80 | **63.90** |

하지만 90% 결측의 네 데이터/마스크 조합에서는 모두 PCFI가 FISF보다 높았다. Entry-wise 99.5% 결측의 정확도도 PCFI가 높았다. **FISF의 우세는 극심한 whole-node 결측에서의 제한된 신호다.**

### 3.2 Support 정보·다중 view를 추가해도 전반적인 증분은 작았다

Cora/CiteSeer의 8개 데이터/마스크/결측률 조건을 같은 비중으로 평균했다.

| Frozen GraphPFN 입력/출력 구성 | 평균 정확도(%) |
|---|---:|
| PCFI 기본 입력 | 69.65 |
| FISF 기본 입력 | 68.43 |
| PCFI + 관측/support 정보 | 69.64 |
| FISF + 관측/support 정보 | 68.80 |
| Short/medium/PCFI 다중 전파 view | 69.61 |
| PCFI/FISF 입력 view 조합 | 69.73 |
| PCFI/FISF 예측 확률 50:50 평균 | 70.16 |

입력 view 조합은 한 번의 10-member GraphPFN을 사용한다. 확률 평균은 두 predictor를 사용하므로 20 forwards가 필요하다. 같은 추론 비용의 비교는 아니다.

사전에 정한 전체 조건의 후속 진입 기준을 만족한 입력 확장 후보는 없었다.

### 3.3 PubMed secondary 평가에는 긍정적인 신호가 있다

Cora/CiteSeer를 본 뒤 whole-node 결측으로 가설을 좁히고, PubMed 평가 전에 입력 조합과 mask를 고정했다. PubMed별 학습이나 파라미터 선택은 하지 않았다.

| Missing nodes | PCFI | FISF | PCFI/FISF 입력 view | 예측 확률 평균 |
|---|---:|---:|---:|---:|
| 90% | 70.60 | 72.23 | **72.83** | 72.37 |
| 99.5% | 69.50 | 70.77 | **73.17** | 71.57 |
| 두 결측률 평균 | 70.05 | 71.50 | **73.00** | 71.97 |

입력 view 조합은 평균적으로 PCFI 대비 +2.95%p, FISF 대비 +1.50%p, 예측 평균 대비 +1.03%p였다. **후속 확인 가치가 있는 결과지만, 한 그래프의 secondary 평가다.** 전체 조건에서 보편적인 우세나 메소드 novelty를 입증한 것은 아니다.

출처: [기본 보간 비교](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/REPORT.md), [입력 확장](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/context_views/REPORT.md), [PubMed](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/context_views/PUBMED_REPORT.md).

## 4. DBLP: native feature block 결측과 HetGFD 방향

Author keyword, paper title-word, term GloVe를 서로 다른 native 입력으로 유지했다. Feature 차원을 타입 간에 바로 평균하지 않고, 보이는 타입별 신호를 target으로 전달했다.

| 라벨/클래스 | 입력 조건 | Cross-type paths + GraphPFN | Adapted HetGFD paths + GraphPFN |
|---|---|---:|---:|
| 5 | Full | 89.96 | 89.92 |
| 5 | Author feature 제거 | 90.18 | 90.19 |
| 5 | Author·paper 제거, term만 남음 | 82.29 | 82.58 |
| 20 | Full | 93.55 | 93.55 |
| 20 | Author feature 제거 | 93.65 | 93.66 |
| 20 | Author·paper 제거, term만 남음 | 93.66 | 93.61 |

Feature 없는 APVPA label propagation은 라벨 5개에서 **91.12%**, 20개에서 **92.28%**였다. 따라서 낮은 라벨 예산의 term-only 하락만으로 새 모델의 headroom을 주장하기는 어렵다.

**판단:** Cross-type 정보는 유용하다. 현재 구성에서 adapted HetGFD의 추가 정확도 이득은 작다. 여기의 adapted HetGFD는 공개 논문의 HGT+HetGFD 결과를 그대로 재현한 설정이 아니다.

출처: [DBLP 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/hetero_missing/REPORT.md).

## 5. 경로 선택과 단순 혼합

### 5.1 DBLP에서 잘된 context 기반 경로 선택이 일반화되지는 않았다

DBLP에서는 context leave-one-out NLL로 APVPA를 선택해 강한 LP를 회복했다. 같은 규칙을 ACM·IMDB·Freebase로 옮겼을 때는 일관된 이득이 없었다.

ACM, 라벨 20개/클래스:

- Metapath LP 평균: **85.55%**.
- 경로 중 context-LOO NLL 선택: **76.98%**.
- 평균/union까지 후보에 넣은 NLL 선택: **75.78%**.
- 같은 후보에서 context 정확도 선택: **85.55%**.

정확도 기반 선택도 IMDB·Freebase에서는 보편적인 해결책이 아니었다. Context NLL이 낮다는 것이 test accuracy도 높다는 뜻은 아니었다. 이 선택기 진단은 개발 결과를 본 뒤 추가된 분석이다.

### 5.2 50:50 확률 혼합은 두 방법의 강점을 자동으로 회수하지 못했다

99.5% node-wise 결측, 라벨 20개/클래스, 전체 test:

| 방법 | ACM | IMDB |
|---|---:|---:|
| PCFI + GraphPFN | 75.45 | 42.61 |
| FISF + GraphPFN | 74.36 | **44.55** |
| Metapath LP 평균 | **85.55** | 35.41 |
| PCFI/LP 50:50 | 83.65 | 40.80 |
| FISF/LP 50:50 | 82.20 | 42.54 |

IMDB에서 FISF-PFN만 맞히는 test 노드는 23.75%, LP만 맞히는 노드는 14.62%였다. 보완성이 일부 존재하지만, 어느 예측이 맞는지 알아낼 수 있다는 뜻은 아니다. Test 정답을 이용한 oracle 선택 성능을 실제 방법의 성능으로 취급하지 않았다.

출처: [경로 선택 정리](C:/Users/seongjin/Desktop/code/gts_pfn/schema_transfer/DECISION.md), [혼합·오류 중첩](C:/Users/seongjin/Desktop/code/gts_pfn/schema_transfer/FUSION_REPORT.md).

## 6. 새로 학습한 두 방향

### 6.1 출처 중첩·공유 보간 오차를 반영한 attention

보간값의 출처 fingerprint로 PSD 오차 행렬을 만들고, 중복된 오차를 집계하는 attention에 penalty를 주는 작은 synthetic 실험을 수행했다. 로그 보정은 ablation으로 유지했다.

- 공개 GraphPFN에서 일부 adapter·입력 encoder·출력 head를 synthetic 과제로 추가 학습했다. LimiX transformer는 고정했다.
- 학습 seed 1개, arm별 64 updates × 2 episodes의 작은 실험이다.
- Vanilla 추가 학습 정확도: **57.01%**.
- 공유 오차 보정: **57.04%**.
- 차이: 약 **+0.03%p**, paired graph bootstrap 구간은 0을 포함했다.
- 출처 대응을 파괴하면 오차 proxy는 나빠졌지만 분류 성능 변화는 작았다.

**판단:** 출처 구조에 민감한 오차 모델은 만들었지만, 분류에 유용한 증분 이득은 입증하지 못했다. 초기 checkpoint 대비 추가 학습의 이득을 이 모듈의 효과로 주장할 수 없다.

출처: [공유 오차 실험](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/source_risk/REPORT.md).

### 6.2 Feature↔label 결합 추론 프로토타입

관측 feature latent H, class state F, 관계별 precision을 사용해 아래 coupling을 포함한 이차 목적함수를 유한 횟수 갱신했다.

`E_cross = (g / 2) * ||F - H W||²`

Context 라벨은 고정하고, synthetic query cross-entropy로 학습했다. 정확한 Bayesian marginalization이나 검증된 posterior uncertainty를 구현한 것은 아니다.

**학습 조건:** 합성 과제 1,536개, 384 updates × batch 4, 초기화 seed 17/29, 최종 checkpoint만 평가. 실제 그래프는 학습에 사용하지 않았다.

| 모델 | 파라미터 수 | 같은 생성기 형태의 새 그래프 | 관계 수·feature 차원 변경 |
|---|---:|---:|---:|
| Feature 먼저 추론 → label 추론 | 10,469 | **72.72** | **75.05** |
| 결합 모델의 F→H 피드백 제거 | 10,469 | 72.44 | 74.49 |
| Feature↔label 결합 추론 | 10,469 | 72.66 | 75.04 |
| 소형 관계형 attention 대조군 | 10,420 | 62.77 | 66.59 |

정확도는 full / node 90% / node 99.5% / cell 99.5%의 네 조건과 세 과제 family를 평균한 값이다. 관계 수·차원 변경도 같은 합성 생성기 안의 변화이며, 실제 unseen schema 검증은 아니다. 소형 attention 대조군은 공식 RT가 아니다.

| 결합 모델 − 대조군 | 새 그래프 차이(%p) | 그래프 bootstrap 95% 구간 | 형태 변경 차이(%p) | 그래프 bootstrap 95% 구간 |
|---|---:|---|---:|---|
| 일방향 모델 | −0.06 | [−0.50, +0.39] | −0.01 | [−0.52, +0.55] |
| 피드백 제거 모델 | +0.22 | [−0.18, +0.62] | +0.55 | [−0.04, +1.22] |

핵심 해석:

- **양방향 피드백의 추가 이득은 아직 입증하지 못했다.**
- 일방향 모델도 context 라벨로 관계별 전달 precision을 정한다. 피드백 ablation은 모든 라벨 정보를 제거한 실험이 아니라, 반복 갱신 중 F→H 경로의 효과를 검사한 것이다.
- 소형 attention과의 큰 차이는 일방향 모델에도 존재한다. Analytic context ridge map의 유무와 짧은 학습의 수렴 차이가 있어, 이를 결합 추론의 기여로 해석하면 안 된다.
- 구조 중심 합성 과제에서는 label에 유효한 관계 선택 정확도가 100%, mixed 과제에서는 66.67%였다. 이 생성기에서의 진단이며 실제 스키마 전이의 증거는 아니다.
- 관측 feature가 전혀 없으면 H가 0을 유지하고, 피드백을 끊어도 출력이 같았다. Context 라벨이 전혀 없으면 균등 예측을 했다.

출처: [구현·수식](C:/Users/seongjin/Desktop/code/gts_pfn/joint_missing_pfn/METHOD.txt), [결과](C:/Users/seongjin/Desktop/code/gts_pfn/joint_missing_pfn/RESULTS.html), [paired 비교](C:/Users/seongjin/Desktop/code/gts_pfn/joint_missing_pfn/results/paired_comparisons.csv), [검증](C:/Users/seongjin/Desktop/code/gts_pfn/joint_missing_pfn/results/audit.json).

## 7. 지금까지 확인한 것과 남은 것

### 확인한 것

1. 결측이 많을 때 그래프 전파는 강한 baseline이다. FISF가 PCFI를 항상 이기지는 않는다.
2. 구조만으로 충분히 강한 그래프와, feature를 살리는 것이 중요한 그래프가 함께 있다.
3. Native relation과 cross-type 정보의 유용성, 새 heterogeneous PFN의 필요성은 구분해야 한다.
4. 기존 relational 모델도 현재 어댑터·분류 설정에서는 severe missing에 자동으로 강하지 않았다.
5. 공유 오차 attention과 양방향 피드백은 현재의 제한된 실험에서 메소드 고유의 이득을 입증하지 못했다.
6. PubMed의 입력 view 조합은 남겨둘 만한 긍정적 결과다.

### 아직 입증하지 못한 것

- 여러 실제 그래프에서 PCFI/FISF+PFN과 강한 LP를 안정적으로 넘어서는 하나의 방법.
- 결측에서 유리한 prior와 추론 연산의 독립적인 기여.
- 엄격한 real unseen-schema 전이와 결측 메커니즘 변화에 대한 일반화.
- 새 프로토타입의 실데이터 성능, 대규모 사전학습 성능, 논문 수준의 novelty.

### 다음 우선순위에 대한 판단

1. **HeteroGraphPFN과 missing-feature 문제를 분리한다.** Relation/path utility를 새로운 context에서 판단하는 방향은 독립적으로 검토한다. 기존 relational PFN과 구별되는 기여와 학습 필요성은 아직 확인해야 한다.
2. **Predictor gating 대신 LP prior와 학습된 보정의 역할을 분리한다.** Frozen Full/neutral 비율은 강한 baseline을 보존하지 못했다. 소형 합성 pilot의 LP conditioning은 형태 외삽에서 offset-only 대비 +0.35%p였지만 matched에서는 이득이 없었다. 실데이터 전체 우위를 주장하기 전에 plain과 offset-only 대비 증분을 별도 확인해야 한다. LP 입력이나 residual head 자체가 double-counting 제거를 보장하지는 않는다.
3. **PCFI/FISF view 조합은 engineering 후보로 유지한다.** PubMed의 긍정적 결과를 범용 메소드로 확정하지 않는다. 후속 확인에는 별도 holdout과 동일 추론 budget이 필요하다.
4. **양방향 확대와 selector 학습은 진행하지 않는다.** LP-conditioned correction의 소형 학습은 별도 실행했으나, 대규모 GraphPFN 사전학습이나 실제 새 schema 전이를 검증한 것은 아니다.

## 8. 비교·재현 시 지켜야 할 구분

- **전체 test와 256-query 부분집합:** 값과 순위가 다를 수 있다. 특히 Freebase에서 실제로 순위가 바뀌었다.
- **Raw 보간과 SVD 후 보간:** 서로 다른 파이프라인이다. 이 문서의 PCFI/FISF 주요 heterogeneous 표는 원본 좌표 보간 결과다.
- **정의되지 않은 helper feature와 가려진 native feature:** ACM/IMDB의 미리 계산된 helper feature는 원본 target feature의 집계값일 수 있다. 결측 전 파생 정보를 유지하지 않았다.
- **공식 모델과 적응 구현:** DBLP HetGFD, 그래프용 RT 어댑터, 소형 synthetic attention 대조군을 각각 구분한다.
- **Zero-shot 평가와 synthetic 사전학습:** 실제 그래프에 대한 gradient update는 없었다. 새 모델 두 실험에는 합성 과제 학습이 있었다.
- **개발 결과와 확인 평가:** 대부분의 실제 그래프는 이전 실험에서 이미 본 그래프다. 새 context seed가 unseen-dataset 검증을 뜻하지 않는다. PubMed도 가설을 좁힌 뒤의 secondary 평가다.
- **통계적 반복:** 같은 그래프의 mask/context seed를 독립적인 그래프 표본처럼 취급하지 않는다. 새 결합 모델의 두 초기화 seed와 그래프 bootstrap도 광범위한 학습-seed 불확실성을 대표하지 않는다.
- **검증 범위:** 저장 확률의 지표 재계산, context/test 분리, mask·입력·checkpoint hash, 비관측 라벨 변경 검사 등을 단계별 audit에 남겼다. 검증 통과가 모델의 일반화나 novelty를 보장하지는 않는다.

관련 저장 위치:

| 단계 | 보고서/기록 |
|---|---|
| 동질 그래프 보간 | [fisf_graphpfn](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/REPORT.md) |
| 입력 view·support 정보 | [context_views](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/context_views/REPORT.md) |
| PubMed secondary | [PubMed 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/context_views/PUBMED_REPORT.md) |
| 공유 보간 오차 | [source_risk](C:/Users/seongjin/Desktop/code/gts_pfn/fisf_graphpfn/source_risk/REPORT.md) |
| DBLP native block | [hetero_missing](C:/Users/seongjin/Desktop/code/gts_pfn/hetero_missing/REPORT.md) |
| ACM/IMDB/Freebase·선택·혼합 | [schema_transfer 판단](C:/Users/seongjin/Desktop/code/gts_pfn/schema_transfer/DECISION.md) |
| RDB-PFN·RT baseline | [relational_baselines audit](C:/Users/seongjin/Desktop/code/gts_pfn/relational_baselines/audit.json) |
| 새 결합 추론 | [joint_missing_pfn audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_missing_pfn/results/audit.json) |

## 9. 추가: predictor reliability Gate A/B 진단

새 아키텍처 학습에 앞서 기존 ACM·IMDB 전체 test, 99.5% 결측, k=5/20, 세 seed에서 query별 보완성과 학습 없는 선택을 분석했다. 모든 규칙은 새 결과 계산 전에 고정했다. 원래 피처 좌표의 재가림 복원 12회와 context를 실제 제외한 frozen PFN 3-fold 추론 72회를 완료했다. LP도 같은 context fold를 제거했다. 두 그래프는 개발용이다.

**동일 query 안의 보완성은 있지만, 이번 선택 규칙은 두 baseline의 강점을 함께 보존하지 못했다.** k=20, FISF branch 기준:

| Graph | FISF+PFN | LP | Context global 선택 | Query local 선택 | 정답 기반 oracle |
|---|---:|---:|---:|---:|---:|
| ACM | 74.36 | 85.55 | 85.55 | 85.56 | 90.43 |
| IMDB | 44.55 | 35.41 | 41.51 | 41.92 | 59.16 |

Global은 context crossfit Brier가 작은 predictor를 고르고, Local은 그 오차 차이를 union LP로 query에 전달했다. Oracle은 둘 중 하나라도 맞는 경우의 상한이며 실행 가능한 선택기가 아니다.

- 복원 오차 proxy의 동일 task 내부 AUROC는 FISF k=20에서 ACM 0.484 / IMDB 0.511로 약했다. Confidence 차이는 0.702 / 0.572였지만, 실제 선택 개선이나 새로운 그래프에 대한 전이를 검증한 것은 아니다.
- **확률 손실과 accuracy의 목적 불일치**도 있다. IMDB seed=11에서 LP는 context와 test 모두 Brier가 더 좋았지만 accuracy는 35.54%로 FISF+PFN의 44.65%보다 낮았다. 신뢰도 추정 실패를 소수 context의 잡음만으로 설명할 수 없다.
- FISF+PFN도 구조·context 라벨·합성 주입을 사용한다. LP보다 높다는 사실만으로 실제 관측 피처가 원인이라고 단정할 수 없다. 동일 branch의 피처 값 제거·셔플 대조가 필요하다.
- **Gate C 학습과 양방향 모델 확대는 보류한다.** 이번 관측량이 모든 가능한 reliability 신호를 배제하는 것은 아니지만, 현재 proxy만으로 새 selector 학습에 들어갈 근거는 부족하다.

완료된 진단의 72회 외에 전처리 재현 검사 등으로 폐기한 PFN forward 14회가 있어 실제 실행은 86회다. 원래 좌표 복원도 중단 1회가 추가로 있다. 최종 24개 task의 입력 해시, context 제외, 선택 결과 재계산 검사를 통과했다.

출처: [Gate A/B 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/evidence_reliability/REPORT.md), [고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/evidence_reliability/protocol.json), [검증](C:/Users/seongjin/Desktop/code/gts_pfn/evidence_reliability/results/audit.json), [실행 예산 기록](C:/Users/seongjin/Desktop/code/gts_pfn/evidence_reliability/execution_notes.json).

## 10. 추가: 관측 피처 제거·셔플과 독립 injection 통제

같은 k=20 context·query·graph·path·seed에서 99.5% 결측의 남은 실제 관측값을 개입했다. Full은 완전 관측 데이터가 아니라 기존 99.5% 결측 baseline이다. 기존 Full 12개, 새 frozen 추론 48개, 총 60개 예측과 42개 paired contrast를 완료·검증했다. 모델 학습은 없었다.

End-to-end는 관측값을 제거하거나 관측 행들 사이에서 셔플한 뒤 원래 PCFI/FISF와 전처리를 전부 다시 실행한다. 각 수치는 세 seed 평균 정확도(%).

| Graph | Imputer | Full | No-observed | Shuffled | Full−No (%p) | Full−Shuffle (%p) |
|---|---|---:|---:|---:|---:|---:|
| ACM | PCFI | 75.45 | 74.53 | 74.31 | +0.92 | +1.14 |
| ACM | FISF | 74.36 | 74.14 | 73.38 | +0.22 | +0.98 |
| IMDB | PCFI | 42.61 | 41.55 | 42.67 | +1.06 | −0.06 |
| IMDB | FISF | 44.55 | 44.51 | 44.30 | +0.04 | +0.25 |

Controlled FISF는 실제 값·라벨과 독립적으로 정한 injection 채널·anchor·값을 모든 조건에 고정하고, 원래 관측 위치로 계산한 거리 가중치도 고정했다. Full에서 고른 injection이나 Full-fitted 변환을 공유하지 않았다. Controlled Full은 원래 FISF Full과 다른 모델이다.

Test 성능을 평가하기 전에 zero-observed 보조 조건도 고정했다. 관측 위치의 clamp를 유지하되 실제 숫자값만 전부 0으로 바꾼다. 이 조건은 clamp availability와 실제 값의 경로를 구분하기 위한 기전 대조이며 배포 baseline이 아니다.

| Graph | Controlled Full | No-observed | Zero-observed | Shuffled | Full−Zero (%p) |
|---|---:|---:|---:|---:|---:|
| ACM | 75.31 | 74.08 | 74.29 | 72.30 | +1.03 |
| IMDB | 43.78 | 43.47 | 43.75 | 43.07 | +0.03 |

**현재 판정:** IMDB의 LP 대비 약 9.13%p 우위는 실제 관측값을 제거해도 거의 남았다. 구조 입력만 남긴 PCFI No-observed는 기존 구조 입력 GraphPFN 예측과 정확히 같았다. 평균 차이는 산술적으로 구조 입력 PFN−LP +6.13%p, 피처 없는 FISF 경로−구조 입력 PFN +2.96%p, 실제 관측 경로의 순차이 +0.04%p로 정리할 수 있다. 합성 주입 하나의 인과적 효과를 분해한 것은 아니다.

다만 **실제 피처가 영향을 주지 않는다는 뜻은 아니다.** IMDB에서 제거 전후 End-to-end 예측은 30.47% 바뀌었다. Controlled Full−Zero에서도 26.89%가 바뀌고, Full만 정답인 9.95%와 Zero만 정답인 9.92%가 거의 상쇄됐다. Query별 유용성과 방해 효과의 예측 가능성은 아직 미검증이다.

ACM의 controlled shuffle 하락 평균 3.01%p도 seed별 +4.29/+5.53/−0.78%p로 일관되지는 않았다. Full−Zero 평균 +1.03%p는 한 seed의 +6.62%p에 의존한다. 같은 그래프의 반복이므로 일반화 증거로 과대해석하지 않는다.

**Gate C는 계속 보류한다.** 현재는 missing-feature recovery를 메인 가설로 밀기보다 구조·합성 입력이 frozen PFN에 주는 역할을 먼저 분리할 근거가 더 강하다. PCFI 제거에서는 PFN 입력 폭도 64→ACM 5/IMDB 2로 바뀌므로 전체 파이프라인 효과이며, controlled FISF는 모든 arm에서 64열을 유지했다. Controlled contrast도 원래 adaptive FISF의 natural direct effect를 측정한 것은 아니다.

출처: [결과와 판정](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/REPORT.md), [기본 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/protocol.json), [zero 대조의 사전 정의](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/zero_addendum.json), [수식·개입 정의](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/METHOD.txt), [검증](C:/Users/seongjin/Desktop/code/gts_pfn/feature_intervention/results/audit.json).

## 11. LP prior + posterior correction: frozen MVP와 소형 학습을 구분

### 11.1 Frozen counterfactual ratio

`p_final ∝ p_LP × p_Full / p_neutral`을 λ=1, probability floor 1e-12로 먼저 고정했다. ACM/IMDB, k=20, 99.5% 결측, 세 seed, 같은 전체 test를 썼다. 원래 PCFI/FISF Full−No-observed와 controlled FISF Full−No-observed/Zero를 구분했다. 모든 pair에 shuffled numerator 음성 대조도 사전 고정했다. 총 48개 확률 보정이며 새 공식 PFN forward나 모델 학습은 0회다.

| Graph / original FISF | LP | Full PFN | 50:50 | Frozen ratio |
|---|---:|---:|---:|---:|
| ACM | 85.55 | 74.36 | 82.20 | 82.35 |
| IMDB | 35.41 | 44.55 | 42.54 | 34.45 |

Controlled Full−Zero ratio도 ACM 82.22%, IMDB 34.50%였다. PCFI의 IMDB ratio 38.40%는 LP를 넘지만 Full 42.61%와 50:50 40.80%에는 못 미쳤다. λ나 부호를 바꾸어 재선택하지 않았다.

FISF에서 ACM은 LP 오답을 3.50% 고치고 정답을 6.70% 망가뜨렸고, IMDB는 11.13% / 12.10%였다. Shuffled residual이 실제 Full residual보다 높은 경우도 있었다. 확률 floor에 걸린 LP/neutral cell은 없었으므로 이번 실패를 epsilon clipping으로 설명할 수는 없다. Neutral intervention은 피처를 marginalize한 posterior가 아니며, 강한 PFN 구조 경로까지 빼고 약한 LP를 대체 prior로 넣을 위험이 있다.

### 11.2 LP-conditioned 구조 residual의 별도 소형 합성 실험

Frozen ratio 실패를 학습형 모델의 반증으로 취급하지 않고 다음 세 arm을 비교했다. 공식 GraphPFN이 아니라 이전 일방향 quadratic prototype(10,469 parameters)을 공통 warm start로 사용했다. 각 초기화 seed 17/29에서 세 arm이 같은 초기 tensor로 시작해, 새 합성 과제 1,536개를 384 updates × batch 4로 학습했다. 실제 그래프의 weight update는 없었다.

- **plain:** 기존 context one-hot 기반 일방향 추론.
- **LP offset:** 내부는 plain과 같고 출력 logit에 고정 log LP를 더함.
- **LP-conditioned:** 내부 ridge target·label propagation clamp를 `onehot(Y_C) − LP_context_LOO`로 바꾸고, query log LP에 학습된 보정량을 더함. 실제 피처가 없어도 구조 보정을 허용한다.

LP는 native relation별 확률 평균이고 context LP는 자기 seed의 전역 기여를 제거한 LOO다. 기존 양방향 feedback을 재개하거나 predictor selector를 학습한 것은 아니다. 같은 prior에서 새 graph seed를 쓰며, 동일 모델·학습량 대조만 비교한다. 이전 합성 실험의 72.72%와 숫자를 직접 비교하면 안 된다.

| Synthetic setting, 4조건 평균 | LP | plain | LP offset | LP-conditioned |
|---|---:|---:|---:|---:|
| Matched | 66.08 | 74.61 | 74.29 | 74.26 |
| Heldout shape | 63.03 | 70.42 | 70.97 | 71.32 |

각 setting은 새 그래프 36개 × full/node90%/node99.5%/cell99.5%, 학습 seed 2개다. Heldout shape는 같은 생성기에서 relation 수 2→3과 feature 폭 6–12→16–24로 바꾼 조건이다. 새 semantic schema나 실제 graph 전이를 검증한 것이 아니다.

**판정:** 내부 conditioning의 보편적 이득은 미확인이다. Matched에서는 offset-only 대비 −0.03%p, heldout shape에서는 +0.35%p였다. 후자의 graph bootstrap interval은 약 [+0.003,+0.696]%p로 하한이 0에 매우 가깝고, 두 학습 seed·한 생성기의 pilot이다. NLL과 Brier는 두 setting 모두 offset-only가 conditioned보다 좋았다.

99.5% whole-node 조건에서도 matched는 plain 71.20 / offset 70.37 / conditioned 70.42%, shape는 66.46 / 66.87 / 67.29%였다. 미세한 형태 외삽 신호를 남겨두되, 큰 모델 학습이나 실데이터 성공으로 확대해석하지 않는다.

### 11.3 연구 주장에 필요한 구분

LP가 G와 context 라벨의 결정적 함수이면 LP를 추가 conditioning한다고 통계적으로 새로운 정보가 생기는 것은 아니다. 이득은 inductive bias·표현·계산에서 보여줘야 한다. CE residual head도 미설명 증거만 표현하거나 중복 evidence를 자동 제거하도록 식별되지 않는다.

또한 frozen 식의 `base + expert − anti-expert`는 [DExperts Eq. 2–3](https://aclanthology.org/2021.acl-long.522.pdf)와 같은 대수적 형태다. 그래프 context residual 전파는 [Correct and Smooth](https://www.cs.cornell.edu/~arb/papers/simple-LP-ICLR-2021.pdf)와도 연결된다. 식이나 residual 명칭 자체가 아니라 episodic 보정 연산과 plain/offset 대비 새 과제 전이 증분이 기여로 입증돼야 한다.

출처: [Frozen 결과](C:/Users/seongjin/Desktop/code/gts_pfn/counterfactual_residual/REPORT.md), [이론·선행연구·설계](C:/Users/seongjin/Desktop/code/gts_pfn/counterfactual_residual/METHOD.md), [frozen 검증](C:/Users/seongjin/Desktop/code/gts_pfn/counterfactual_residual/results/audit.json), [합성 학습 결과](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/REPORT.md), [합성 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/protocol.json), [합성 검증](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/results/audit.json).

## 12. LP conditioning 분리 진단: 새 학습 없이 A–C 완료

기존 6개 소형 checkpoint를 고정하고 새 master graph 36개에 relation 2/3 × feature 좁음/넓음 × 결측 4조건을 적용했다. 576 episodes, 주 평가 확률 파일 4,032개다. 학습 업데이트와 공식 GraphPFN forward는 0회다. 작은 모델의 주 평가 task forward 3,456회에 batch/singleton 확인 12회가 추가되어 실제 3,468회다.

Master는 기존 R3/wide 생성기를 재현한다. R2는 생성기가 지정한 두 label/feature 관계를 보존하고 나머지 nuisance 관계·helper 노드를 제거한다. Feature는 동일 raw matrix의 6–12열 prefix와 16–24열 전체를 비교한다. 공유 feature·mask·context·query·유지한 edge를 고정했다. 따라서 새로운 semantic schema 평가나 실제 nuisance 관계 식별 알고리즘의 성공이 아니다.

| 새 paired 조건, 네 결측 평균 | LP | Plain | Offset | Conditioned | 내부 효과 (%p) |
|---|---:|---:|---:|---:|---:|
| R2 / narrow | 66.80 | 74.21 | 73.91 | 74.27 | +0.36 |
| R2 / wide | 66.80 | 73.64 | 73.71 | 74.14 | +0.43 |
| R3 / narrow | 66.56 | 73.88 | 73.95 | 74.42 | +0.47 |
| R3 / wide | 66.56 | 73.62 | 73.67 | 74.20 | +0.52 |

**판정:** 작은 내부 효과는 남지만 conditioned−plain의 네 평균 구간은 모두 0을 포함했다. 내부 효과가 relation 추가로 변한 양은 +0.10%p [−0.21,+0.42], feature 확장으로 변한 양은 +0.06%p [−0.11,+0.23]다. Shape 변화의 특별한 이득을 분명히 찾지 못했다. 새 네 cell 모두 평균 NLL/Brier는 offset-only가 더 좋았다. Full에서도 내부 효과가 +0.54~+0.72%p라 심한 결측에만 국한된 현상은 아니다.

LP 정답 여부에 따른 query 진단에서 LP 오답 집단 정확도는 plain 43.61 / offset 33.95 / conditioned 36.94%였다. Conditioned가 offset의 손해 일부를 회복해도 plain의 교정 능력을 보존하지는 못했다. LP 정답 집단은 89.31 / 94.05 / 93.25%였다. Test 정답을 쓰는 사후 분석이며 selector나 모델 채택에는 사용하지 않았다. 반복 query decision을 합친 기술 통계이므로 graph 평균과 가중 방식도 다르다.

다음 연구는 출력 LP offset보다 relation별 query·class context support를 모델 내부에 전달하는 가설로 좁힌다. 기존 plain에도 shared relation MLP가 있으므로 shared encoder 자체가 새 기여는 아니다. 현재 target–helper native relation에서 한 hop의 target label support는 0이므로 target→helper→target typed path kernel이 필요하다. 관계별 histogram은 출처를 보존하지만 원래 그래프 전체를 손실 없이 표현하지는 않는다. 같은 예산의 relation-aware plain·관계 평균 evidence·관계별 evidence를 비교하는 것이 다음 대조이며, 아직 이 새 모델을 학습한 것은 아니다.

출처: [전체 진단 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/diagnostics/REPORT.md), [사전 고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/diagnostics/protocol.json), [검증](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/diagnostics/results/audit.json), [추가 확인을 포함한 실행 횟수](C:/Users/seongjin/Desktop/code/gts_pfn/lp_conditioned_pilot/diagnostics/results/execution_notes.json).

## 13. Relation Evidence PFN: 작은 Conflict kill test 완료

LP-conditioned 학습을 종료하고, 기존 일방향 staged backbone에서 Plain / Mean-Evidence / Separate-Evidence를 새로 학습했다. 공식 pretrained GraphPFN은 아니다. 세 초기화 seed 17/29/43, 같은 새 합성 과제 1,536개, 모델당 384 updates × batch 4, 같은 optimizer, 최종 checkpoint만 사용했다. Mean/Separate는 같은 11,270개 파라미터·초기 가중치를 갖고, Plain은 같은 초기 backbone의 10,469개 파라미터다. Evidence head의 용량 증가는 약 7.7%다.

K_r = B_r D_helper,r^(-1) B_r^T에서 자기 seed의 전체 2-hop self-return을 제거하고, class histogram·절대 evidence·support mass·context LOO soft quality 등을 shared class/relation head에 제공한다. 고정 전처리 descriptor를 인코딩하기 전에 평균하는 Mean과, 인코딩 후 평균하는 Separate를 비교했다. Raw E 하나만 평균하는 것보다 풍부한 descriptor 평균 대조이며, 모델 전체에는 공통 relation-aware backbone이 남는다. 출력 log LP offset, finished-predictor 선택기, context CE는 없다.

세 scenario는 같은 master label·feature·context 순서·관측 난수·보존한 관계 edge를 공유한다. Consistent는 두 관계가 label partition을 따르고, Noisy는 한 관계가 무작위 partition, Conflicting은 한 관계에서 노드 partition 35–55%를 다른 클래스로 바꾼다. 단순한 전체 클래스 이름 치환은 target→helper→target kernel을 바꾸지 않으므로 쓰지 않았다. Utility 역할의 relation ID는 task마다 무작위이며 role reversal과 context shuffle도 평가했다.

주 비교는 새 master 24개, k=1/3/5/10/20/class, 99.5% whole-node missing이다. 아래 accuracy(%)는 k와 세 학습 seed 평균이다.

| Scenario | Uniform histogram | Context weighted | Plain | Mean | Separate |
|---|---:|---:|---:|---:|---:|
| Consistent | 93.53 | 93.24 | 96.53 | 95.76 | 95.82 |
| Noisy | 78.78 | 84.88 | 88.49 | 84.15 | 85.08 |
| Conflicting | 79.90 | 85.05 | 88.96 | 85.34 | 86.09 |

**판정은 현재 설계의 No Go다.** Conflicting의 Separate−Mean은 +0.74%p, graph-bootstrap 95% 구간 [+0.49,+1.03]이며 학습 seed별 +0.51/+0.87/+0.85%p로 반복됐다. 그러나 Separate−Plain은 −2.88%p [−3.66,−2.09]이고 세 seed 모두 음수다. Noisy에서도 Mean 대비 +0.93%p이나 Plain 대비 −3.41%p다. 관계를 분리한 evidence head의 이득은 있지만 기존 구조 추론을 넘어선 것은 아니다. Conflict에서 특히 더 큰 이득이라는 패턴도 확인하지 못했다.

Conflict의 k3/k5에서 Separate−Mean은 +1.39/+1.62%p지만 Plain 대비 −5.78/−5.49%p다. k1은 Mean 대비 −0.11%p, k20은 +0.07%p다. 일부 k만 골라 최종 성공으로 취급하지 않는다. 지원되는 query에서 두 relation의 evidence argmax 불일치율은 Conflict k별 53–59%로 실제 충돌이 있었다.

확률 품질은 다르다. Conflict에서 Plain / Mean / Separate의 NLL은 0.5320 / 0.3751 / 0.3585, Brier는 0.2905 / 0.2151 / 0.2044다. Proper-score 개선은 기록하되 accuracy를 주 목표로 한 Go 조건과 구분한다. 재보정이나 temperature tuning을 추가하지 않았다.

관계 ID 순열의 학습 후 확률 최대 차이는 2.38e−7, 클래스 순열 등변성 차이는 1.19e−7로 통과했다. Context shuffle에서는 Separate 정확도가 scenario별 33.58/35.04/34.92%로 하락했다. 모든 evidence·utility를 다시 계산했으며 작은 context에서 shuffle이 실제 label을 바꾸지 못한 15/360 episodes도 공개했다. Role reversal에서도 Plain 우위를 뒤집지 못했다.

총 1,368 episodes와 확률 파일 15,048개를 완료했다. 주 비교·role reversal·shuffle 1,080개와 k5/k20의 full/node90% 보조 평가 288개다. 저장 예측의 지표 재계산 오차는 0이며, 초기 가중치 일치·자기 seed 제거·query 정답 입력 불변·학습/평가 graph seed 분리·평가 weight 불변·소스와 checkpoint hash 검사를 통과했다. 학습 전 cancellation 잔차로 인한 k1 LOO 검사 실패 1회는 K 대각을 곱셈 전에 제거해 수정했고, 그 버전으로 학습·보고한 예측은 없다.

이번 결과는 같은 synthetic generator에서 evidence head를 추가한 작은 대조다. Plain도 relation별 context 통계를 사용하므로 관계 출처 보존이라는 전체 가설을 반증한 것은 아니다. 다만 현재 global-utility prior에서 premature aggregation이 Plain의 주 병목이라는 근거도 확보하지 못했다. 새로운 semantic schema, relation 수 전이, 공식 GraphPFN과 실제 그래프는 아직 평가하지 않았다.

출처: [전체 보고서와 k별 결과](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/REPORT.md), [사전 고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/protocol.json), [독립 검증](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/audit.json), [실행 횟수와 실패 기록](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/execution_notes.json), [결과 그림](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/results/conflict_kill_test.png).

## 14. Frozen Evidence Utility: 확률 점수와 class decision의 역전 분석

기존 primary base/node99.5% 예측에서 Plain/Mean/Separate를 짝지었다. 24개 graph × scenario 3 × k5조건 × 학습 seed3 = 1,080 matched triples, 저장 확률 3,240개다. 새 모델 forward·학습·calibration fit·확률 조합·loss 변경은 0회다. 같은 개발용 synthetic graph에 대한 사후 분석이다.

### 14.1 과신 완화가 아니라 큰 과소확신의 변화

| Conflict | Accuracy (%) | Mean confidence (%) | Top-label ECE15 | Normalized entropy | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Plain | 88.96 | 63.99 | 0.2498 | 0.7340 | 0.5320 | 0.2905 |
| Mean | 85.34 | 83.85 | 0.0197 | 0.3852 | 0.3751 | 0.2151 |
| Separate | 86.09 | 84.31 | 0.0182 | 0.3764 | 0.3585 | 0.2044 |

Separate는 확률을 평탄하게 만든 것이 아니라 평균 confidence를 높이고 entropy를 낮췄다. Plain은 accuracy에 비해 confidence가 크게 낮았고 ECE10/15·equal-mass15와 class 수별 분석에서도 과소확신이 두드러졌다. Top-label calibration 개선은 보이지만 이 지표만으로 NLL/Brier 변화를 calibration과 resolution의 인과적 기여로 나눌 수는 없다.

오답일 때의 confidence는 Plain 43.11%, Separate 63.90%로 오히려 높아졌다. 전체 query 중 confidence≥0.9 오답도 0.02→0.94%였다. 따라서 오답의 과신을 줄인 것이 확률 점수 개선의 주 설명은 아니다.

### 14.2 Disagreement와 score 기여를 분리

Conflict에서 Plain/Separate argmax가 다른 query는 8.40%다. Plain만 정답은 전체의 5.36%, Separate만 정답은 2.48%로 차이 −2.88%p가 정확히 accuracy 하락과 일치한다. Disagreement 안의 정답 비율은 Plain 63.83%, Separate 29.57%다.

| Conflict partition | Query mass (%) | ΔNLL 기여 (Sep−Plain) | ΔBrier 기여 |
|---|---:|---:|---:|
| 둘 다 정답 | 83.60 | −0.24487 | −0.13482 |
| Plain만 정답 | 5.36 | +0.03080 | +0.02487 |
| Separate만 정답 | 2.48 | −0.01419 | −0.01038 |
| 둘 다 오답·같은 결정 | 8.00 | +0.05133 | +0.03232 |
| 둘 다 오답·다른 결정 | 0.55 | +0.00351 | +0.00190 |

합은 전체 ΔNLL −0.17342, ΔBrier −0.08611과 정확히 같다. 둘 다 맞힌 집단의 평균 p(true)가 0.6798→0.8819로 높아진 기여가, 나머지 집단의 순손해를 상쇄했다. Plain→Separate NLL 개선의 산술적 90.5%, Brier 개선의 87.6%는 이미 Plain→Mean 차이에 있다. 관계 출처를 보존하는 것만의 기여로 대부분의 확률 점수 개선을 설명할 수 없으며, 이 비율이 공통 head의 인과적 효과를 추정하는 것은 아니다.

### 14.3 p(true)를 높였는데 argmax를 잃는 경우는 일부

Plain 정답→Separate 오답 중 p(true)가 증가한 비율은 12.48%, 전체 query의 0.67%다. C2에서는 0이며 C3/C4에서만 발생했다. Multiclass에서는 오답 경쟁 클래스의 확률이 더 크게 증가하면 가능하지만 accuracy 하락의 주 패턴은 아니다. 별도로 Separate가 틀렸지만 Plain보다 p(true)가 높은 query 전체는 2.48%였다.

Conflict 세 학습 seed 모두 accuracy는 낮아지고 NLL/Brier는 개선됐다. Δaccuracy는 −3.24/−3.04/−2.35%p, ΔNLL은 −0.1732/−0.1781/−0.1689다. 한 번의 학습 불안정성만으로 설명하기는 어렵지만, 세 초기화와 최종 확률만으로 정보 중복과 representation 간섭을 구분할 수는 없다. Backbone도 arm별로 함께 학습됐으므로 head의 직접 효과도 식별되지 않는다.

### 14.4 Conflict 강도: 관찰적 연관만 가능

원래 prior/seed로 복원한 실제 partition 변경 비율은 0.306–0.601이었다. Graph별 k·seed 평균의 Separate−Mean accuracy 증분과 Spearman ρ는 +0.068, 탐색적 bootstrap 구간 [−0.334,+0.424]다. Separate−Plain은 −0.110 [−0.499,+0.323]다. 강할수록 분리 인코딩의 accuracy 이득이 커진다는 근거는 확보하지 못했다.

이는 24개 서로 다른 graph 사이의 연관이며 동일 graph에서 강도만 바꾼 sweep이 아니다. 실제 corruption fraction도 class 수와 partition 구조에 따라 단조로운 conflict 척도가 아니어서 1−NMI와 support가 있는 query의 evidence disagreement를 보조 지표로 함께 보고했다. Class 수별·k별 결과와 작은 표본의 불확실성은 전체 보고서에 있다.

**판정:** 현재 head의 No Go를 유지한다. Probability quality 향상과 class decision 손해가 동시에 반복되는 현상은 확인했지만, 새 정보 추가·원인 A/B 구분·margin loss의 효용·새로운 schema 전이를 입증하지 않는다. 새 loss나 selector를 추가하지 않았다.

출처: [전체 진단 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/REPORT.md), [고정 분석 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/protocol.json), [분석 검사](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/results/audit.json), [독립 재집계 검증](C:/Users/seongjin/Desktop/code/gts_pfn/relation_evidence_pfn/frozen_utility/results/verification.json).

## 15. Plain은 이미 전역 relation utility에 적응하는가?

기존 Plain checkpoint(seed17/29/43)만 고정하고 role reversal을 더 엄격하게 검사했다. 기존 role-control은 시도 degree와 난수를 유지했지만 edge 중복 제거 때문에 실제 node degree가 달라질 수 있었다. 이번에는 bipartite double-edge switch로 relation별 모든 target/helper 개별 degree를 정확히 보존했다. Node ID·type·feature·mask·context/query·label도 고정했다. Weight 학습·loss 변경·Evidence Head 최적화는0회다.

원래 그래프, 역할을 유지한 조건부 재샘플, 역할을 교환한 조건부 재샘플을 구분했다. 재샘플 두 조건에 같은 proposal 난수를 쓰고100×edge수의 Metropolis proposal을 고정했다. Noisy/Conflict 48/48쌍에서 전체 target kernel homophily로 utility 역할 교환을 확인했다. Consistent는 두 partition이 같아 두 조건의 graph와 예측이 완전히 같다. 평형 도달이나 degree 이외 모든 고차 구조의 동일성을 보장하는 것은 아니며, 성능으로 graph를 제외하지 않았다.

### 15.1 End-to-end 정확도와 mechanism 적응을 구분

99.5% whole-node missing, master graph24개, k1/3/5/10/20과 세 학습 seed 평균 accuracy(%).

| Scenario | Same-role adaptive | Reversed adaptive | Reversed: old mechanism 고정 | Reversed: mechanism용 label shuffle | Reversed: uniform relation weights |
|---|---:|---:|---:|---:|---:|
| Consistent | 95.27 | 95.27 | 95.27 | 94.68 | 95.92 |
| Noisy | 88.03 | 87.83 | 40.87 | 75.54 | 78.80 |
| Conflicting | 89.19 | 87.11 | 50.60 | 76.60 | 81.07 |

Noisy의 reversal−same은 −0.20%p, paired graph-bootstrap 구간[−2.81,+2.45]다. Conflict는 −2.09%p [−3.99,−0.18]로 완전한 robustness는 아니다. 역할 교환 후 실제 path 품질도 조금 다르므로 이 차이를 곧바로 ID/semantics 암기의 효과로 해석하지 않는다. 예를 들어 designated-good 관계의 전체 kernel homophily는 Conflict에서0.817→0.802다.

반면 mechanism 적응의 기여는 크다. Conflict에서 adaptive−old mechanism은 +36.51%p [+33.31,+39.99], adaptive−shuffled-mechanism은 +10.51%p [+8.18,+12.85], adaptive−uniform은 +6.04%p [+4.44,+7.54]다. Noisy는 각각 +46.96/+12.30/+9.03%p다. 세 seed에서 같은 방향이다.

Old-mechanism 대조는 Same graph의 wx/wy/coupling/ridge를 Reversed graph에 고정한다. Shuffled 대조는 mechanism 계산에만 context label 순열을 넣고, 실제 label propagation의 정답 context와 clamp는 유지한다. 같은 context 위치·개수에서는 원래8개 mechanism 통계 중 label_alignment만 label 순열로 바뀐다. Uniform은 wx/wy의 총 scale과 reversed coupling/ridge를 유지한 채 relation 간 차이만 없앤다. 모두 진단용 intervention이며 새 방법이나 selector가 아니다.

### 15.2 관계 ID가 아니라 context 통계에 따른 역할 추적

Plain에는 relation-ID embedding이나 ID별 학습 파라미터가 없고,8개 relation 통계에 같은 MLP를 적용한다. 학습 prior도 유용한 relation ID를 task별로 무작위화했다. 따라서 실패하더라도 임의 ID lookup 암기라는 설명은 구조적으로 맞지 않는다.

Label propagation weight wy를 합1로 정규화했을 때 새 good 관계의 평균 share는 Noisy0.913, Conflict0.869였다. 역할 교환 뒤 새 good을 더 선호한 비율은 각각94.44%,90.28%다. Before에서 old good을 선호한 경우에 한정한 추적률은95.68%,91.42%다. 이 값은 동일 graph의 k·초기화 반복을 포함한 기술 통계다.

### 15.3 k1은 구체적인 한계 후보

Conflict k1에서 Same71.10→Reversed64.61%로 −6.49%p였다. 새 good의 평균 wy share는0.541이지만 k10/k20에는0.991/0.995였다. 모든 context가 한 클래스씩만 가진 경우 서로 다른 context의 centered one-hot 내적은 항상 −1/C다. 따라서 context pair mass가 양수이면 label_alignment가 −1/C로 고정된다. 독립 2-hop 계산에서 이 항등식의 오차는5.55e−17였다.

이 때문에 k1에서 mechanism용 context label shuffle은 효과가0이다. 전체 모델이 label을 쓰지 않는다는 뜻은 아니며, pair mass나 feature alignment 등 다른 입력은 여전히 관계별로 달라진다. 전역 label-pair 통계의 한계와 전체 task의 식별 불가능성을 동일시하지 않는다.

Degree를 고정해도 path coverage·utility 강도는 달라질 수 있다. Conflict k1의 designated-good 2-hop accuracy는 Same66.01→Reversed64.76%였다. Plain의 더 큰 하락이 기전 한계인지 확정하려면 이러한 구조 차이와 극소 context 불확실성을 함께 고려해야 한다. 이번에는 새 모델을 학습하지 않았다.

### 15.4 판정과 검증

**현재 global role 교환 문제는 Plain이 이미 상당 부분 해결한다.** 이를 새 evidence module로 재현하는 것만으로는 추가 기여가 약하다. 남은 후보는 적은 context에서 유용한 관계를 식별하는 제한이며, 새로운 semantic operator·국소 utility 변화·heterophilic class 변환은 아직 평가하지 않았다. 같은 학습 prior 안의 역할 변화에 대한 진단이지 새로운 heterogeneous schema 일반화의 증명이 아니다.

주 평가 확률파일6,480개, 구현 확인을 포함한 실제 model task forward는6,510회다. 학습 업데이트는0회다. 원래 bank의 확률과 최대 차이는2.38e−7, argmax 변경은0/145,080이었다. 모든 degree sequence, simple graph, Consistent null, paired context/정답/query, mechanism intervention, source/checkpoint 불변과 저장 확률 지표 재계산을 통과했다. 지표 재계산 오차는0이다.

출처: [전체 보고서와 k별 결과](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/REPORT.md), [고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/protocol.json), [독립 검증](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/results/audit.json), [결과 그림](C:/Users/seongjin/Desktop/code/gts_pfn/plain_role_reversal/results/plain_role_reversal.png).

## 16. Joint-Role PFN: 정확한 기준과 실제 학습 비교

Target 8개·공유 helper 4개·relation 2개인 작은 bipartite SBM을 구현했다. 모든 잠재 node label 2^12개와 role 네 상태를 열거하여 query truth 없이 role posterior와 marginal Bayes prediction을 계산한다. Feature는 완전히 없고 context는 **총 1/2/4개**이므로, 앞 절의 99.5% 결측·클래스당 k 조건과 다르다.

Independent, Exclusive(10/01), Coupled(00/11) prior마다 graph 2,048개로 학습하고 다른 graph 192개를 평가했다. 모델당 192 updates × batch16, 학습 seed 17/29/43으로 총66개 모델을 학습했다. Exclusive/Coupled는 Independent screen 이후 고정한 개발용 후속 실험이며, 각 prior는 알려져 있고 모델도 별도로 학습한다. 미지의 prior나 schema를 context에서 식별한 결과가 아니다.

| Accuracy %, context·학습 seed 평균 | Plain capacity+long | Factorized | JR-PFN |
|---|---:|---:|---:|
| Independent | 51.07 | 52.35 | 52.41 |
| Exclusive | 53.83 | 55.27 | 57.03 |
| Coupled | 58.39 | 58.35 | 58.22 |

JR은 14,503개 파라미터다. Capacity+long Plain은 실제 relation MLP를 넓힌14,655개이며 diffusion 반복도 네 배로 늘렸다. 정확한 FLOP equality 주장은 하지 않으며 실측 latency는 보고서에 있다.

Exclusive의 JR−큰 Plain은 +3.20%p [1.44,4.98], JR−factorized는 +1.76%p [0.28,3.27]다. 같은 JR branch에서 집계만 factorize해도 차이가 확인된다. 반면 decoder를 고정하고 known prior로 집계하면 학습 posterior가 추가한 정확도는 +0.14%p [−0.10,0.41]에 불과하다. CE-only 대조에서도 posterior 보조목적의 예측 이득은 확인되지 않았다.

정확한 oracle의 product-of-marginals predictive KL gap은 Independent0.00046, Exclusive0.00645, Coupled0.00348이다. Mean-first gap은 이보다 크지만 relation별 marginal variance를 잃는 효과도 포함한다. 두 층의 선형 propagation도 역할에는 quadratic이므로, affine-in-roles null과 joint dependence 효과를 구분했다.

Coupled에서는 00 branch가 uniform이어서 $p=q_{00}/C+q_{11}p_{11}$의 argmax가 $p_{11}$과 같다. 따라서 같은 decoder에서 posterior weight는 accuracy를 바꿀 수 없고 confidence만 바꾼다. 이 decoder의 제한을 posterior inference 전체의 한계로 일반화하지 않는다.

**판정:** 제한된 correlated prior에서 역할 상관구조 보존의 positive signal은 있다. 극소 context에서 미지의 관계 의존성을 학습하는 새 원리·일반 MoE 대비 차별성·새 schema 일반화는 아직 입증하지 못했다. Generic MoE, R3/4, unseen schema, 실제 heterograph는 미평가다.

출처: [전체 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/REPORT.md), [수학적 범위와 반례](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/THEORY.md), [모든 prior의 paired 분석](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/results/all_prior_analysis.json), [감사](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/results/audit.json).

## 17. Prior-to-Posterior Shift: 정보 부재와 불완전한 추론의 분리

새 학습 없이 기존 평가 bank와 새로운 paired master512개/prior를 사용했다. Graph 정보 $I(H;G)$와 label 추가 정보 $I(H;Y_C\mid G)$를 구분하고, 각 graph에서 모든 context label pattern과 모든 query/helper 잠재 label을 열거했다. 총57개 archive,26,304 graph/context 사례의 분석이며, 기존 checkpoint·bank·source·결과 확률은 불변이다.

기존 Exclusive·총 context4개에서 graph 정보는0.02635 nats, label 추가 정보는0.12447 [0.11344,0.13623], 전체 정보는0.15082다. Exact conditional branch를 고정하고 집계만 바꾸면 fixed prior→exact posterior 기대 NLL 이득은0.02278 [0.01949,0.02649]이고, graph-only posterior→full posterior의 이득도0.01595 [0.01401,0.01803]이다. 현재 조건을 정보 부재로 설명하기는 어렵다.

| 기존 Exclusive·총 context4개, 같은 exact predictor | Empirical NLL | Bayes predictive KL |
|---|---:|---:|
| Known prior | 0.61592 | 0.02278 |
| Learned posterior | 0.60174 | 0.01070 |
| Product of exact marginals | 0.59730 | 0.00699 |
| Exact posterior | 0.59123 | 0 |

Learned posterior도 점추정 기준 가용 이득의 약53%를 회수한다. 따라서 이전 accuracy +0.14%p만으로 learned가 아무 정보를 활용하지 못했다고 해석하지 않는다. 반면 learned decoder에서는 posterior를 exact로 바꿔도 Bayes KL이0.06131→0.05054에 그쳐 decoder 오차가 크게 남는다. Posterior head만 바꾸면 해결된다는 근거도 아니다.

새 sweep은 context-target edge signal만0/0.36/0.72/0.96으로 바꾸고 query-target native incidence, latent labels, role, context order, edge uniforms를 공유했다. Exclusive·총4개에서 label 추가 정보는0/0.01194/0.12688/0.27420, exact role 업데이트의 기대 NLL 이득은0/0.00184/0.02388/0.07008로 증가했다. Strong signal도 항상 식별 가능한 것은 아니며 posterior max≥0.9의 확률을 별도로 측정했다.

총 라벨1개에서는 binary class 대칭성 때문에 $I(H;Y_C\mid G)=0$이다. 이는 label이 class 방향을 알려주는 효과와 다르며, 이전의 클래스당 k=1 조건과도 다르다. Sweep의 다른 signal과 총6개에서 frozen JR/Plain의 수치는 OOD 진단이고 같은 prior로 재학습한 공정 비교가 아니다. 새 학습은하지 않았다.

**판정:** Posterior 추론의 예측 가치는 확인됐고 learned도 일부 활용한다. 다음 메소드의 목표는 role별 conditional prediction과 role evidence 추정을 함께 개선하는 것이다. 연구의 Conditional Go 근거는 강화됐으나 새 학습 원리의 novelty·일반 MoE 대비 이득·unseen-schema 일반화는 미입증이다.

출처: [보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/posterior_shift/REPORT.md), [고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/posterior_shift/protocol.json), [전체 분석과 CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/posterior_shift/results/analysis.json), [검증](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/posterior_shift/results/audit.json).

## 18. Role-Conditional Distillation: 동일 아키텍처의 학습 원리 검증

JR의 아키텍처와 train prior를 고정하고, B=exact predictive KL, C=B+conditional KL, D=C+0.1 posterior KL을 비교했다. Conditional KL의 weight는1이며 detached exact posterior로 네 역할을 가중한다. 주어진 prior에서 확률0인 역할의 conditional weight는0이다. 기존 A도 이미 exact posterior KL을 보조목적으로 사용했으므로 새 posterior teacher를 처음 추가한 것으로 설명하지 않는다. A0=sampled CE only를 추가해 B와의 차이에서 predictive target의 효과를 분리했다.

A/A0의18개 기존 체크포인트는 bank/source/초기 adapter/update/schedule hash 일치를 확인하고 재사용했다. B/C/D, Factorized D, Known-prior D, Plain distill, Large Plain distill의63개 모델을 새로 학습했다. 새 optimizer update는12,096회다. 모델당192 updates×batch16, seed17/29/43, A~D 파라미터14,503개와 전체 초기 state를 같게 했다. Loss weight나 checkpoint를 평가 정답으로 고르지 않았다.

주 평가는 각 prior별 새로운 graph192개, 훈련과 같은 signal0.72·총 context1/2/4개다. Feature는 전부 없으며 target8/helper4/role2의 소형 SBM이다. 아래는 Bayes predictive KL이며 낮을수록 좋다.

| Prior | B Predictive | C + Conditional | D + Posterior | Large Plain distill |
|---|---:|---:|---:|---:|
| Independent | 0.06192 | 0.06211 | 0.06615 | 0.07140 |
| Exclusive | 0.04964 | 0.04946 | 0.04994 | 0.06555 |
| Coupled | 0.08664 | 0.08606 | 0.08762 | 0.08733 |

C−B의 conditional KL은 세 prior에서 모두 줄었지만 predictive KL 차이의95% CI는 모두0을 포함했다. Exclusive에서 conditional KL0.06698→0.06449(약3.7% 감소)인 반면 predictive KL 차이는−0.000181 [−0.000683,+0.000329]에 그쳤다. D−C의 posterior KL도 줄었으나 predictive KL은 증가했다. 개별 component를 더 정확히 맞추는 것이 최종 predictive distribution의 개선으로 바로 이어지지 않았다.

Exclusive 정확도는 기존 A57.92%, B57.94%, C57.97%, D57.95%, Large Plain distill53.91%였다. JR의 기존 구조적 장점은 남지만, 추가 distillation loss의 고유 성과라고 할 근거는 약하다. 훈련에 없던 총 context3/6개와 signal0.36/0.96도 같은 새 paired graph에서 평가했으며 일부 이득은 있으나 일관된 우위는 아니다. 새 semantic schema 전이는 아니다.

또한 모든 feature가 없는 현재 staged positive-diffusion family에서는 관측 context가 한 class뿐이면 그 class에 대한 query 확률이0.5보다 작아질 수 없다. Exact SBM은 nonedge를 이용해 반대 class를 지지할 수 있다. 이 제약에서 얻은 부분 KL 하한은 Independent0.01924, Exclusive0.01569, Coupled0.02362였다. 모든 남은 오차를 설명하는 최적값은 아니지만 teacher 감독만으로 없앨 수 없는 표현 제한은 존재한다. 저장된81개 모델의556,227개 해당 query 예측에서 제약을 검증했다.

**판정:** 현재 loss-only 제안은 메인 메소드·teacher-free scaling 확장으로 가져가지 않는다. Role posterior 추론의 가치는 유지되지만, 다음 decoder는 역할별 conditional target을 표현할 수 있는지를 명시적으로 검증해야 한다. 이 실험에서는 새 아키텍처를 추가하지 않았다.

출처: [전체 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/REPORT.md), [고정 목적함수와 예산](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/protocol.json), [paired 분석](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/results/analysis.json), [감사](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/results/audit.json), [표현 제약 검증](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/distillation/results/family_bound_check.json).

## 19. Contrastive Relational Decoder: 양의 diffusion 함수족을 벗어나는가?

새 학습 없이 native bipartite SBM의 edge/nonedge likelihood를 signed Ising coupling으로 바꿨다. 관측 mask O와 존재 B를 구분하고, O=0인 pair는 likelihood에서 제외했다. Helper의 잠재 label은 cavity sum-product로 합산한다. Typed path를 독립적인 target-target Bernoulli edge로 취급하지 않는다. BP는64회·damping0.5로 사전 고정했으며 수렴 여부로 연산 횟수를 고르지 않았다.

새 graph는 prior별256개, 총768개다. 같은 graph에서 총 context1/2/4개와 pair 관측률1/0.5/0.2를 비교했다. Feature는 전부 없으며 클래스당 context가 아니다. Frozen positive D와 Large Plain은 기존3 seeds의 최종 checkpoint를 그대로 사용했다.

주 평가(full observation, 세 context 수 평균)의 Bayes predictive KL:

| Decoder / posterior | Independent | Exclusive | Coupled |
|---|---:|---:|---:|
| Frozen positive D / learned q | 0.06224 | 0.04820 | 0.09129 |
| Signed diffusion / exact joint | 0.04693 | 0.03751 | 0.07404 |
| Contrast BP / frozen learned q | 0.01472 | 0.00867 | 0.02819 |
| Contrast BP / known prior | 0.01837 | 0.01265 | 0.03522 |
| Contrast BP / exact factorized | 0.00354 | 0.00783 | 0.00682 |
| Contrast BP / exact joint | 0.00370 | 0.00248 | 0.00615 |

같은 learned q에서 decoder를 바꾼 KL 차이의95% paired graph CI는 세 prior 모두0보다 낮았다. Exclusive는−0.03953 [−0.04345,−0.03564]다. Accuracy도54.94→63.76%로 높아졌다. Exact q로 공통 통제한 decoder 비교에서도 개선됐다. 단일 context에서 positive D는 반대 class를 선택하지 못하지만 contrastive BP는 약50%의 query에서 선택하며 Bayes의 반대 방향 선호와 대부분 일치했다. Exact joint BP의 KL은 새 graph에서 계산한 positive-family 부분 하한 Independent0.01707 / Exclusive0.01491 / Coupled0.02823보다 모두 낮았다.

같은 contrastive decoder의 joint−factorized KL 차이는 Exclusive−0.00535 [−0.00611,−0.00456]로 확인됐다. Independent와 Coupled는 CI가0을 포함했다. Known prior 대비 exact posterior 업데이트는 세 prior 모두 유용했지만, 이것은 exact posterior를 공급한 oracle substitution이며 새 학습 posterior의 성과가 아니다.

관측률0.2에서 Exclusive의 올바른 masked decoder KL은0.000001인 반면 unknown pair를 nonedge로 잘못 처리하면0.06290이다. 두 경우 role posterior는 올바른 mask로 계산해 공통 통제했다. 낮은 관측률의 작은 Bayes KL은 graph가 덜 연결되고 target이 uniform에 가까워지는 효과를 포함하며 정보나 정확도가 증가한다는 뜻이 아니다.

**판정:** Decoder 표현력 변경은 Go다. 다만 analytic decoder는 참 density/signal과 hypothesis family를 알고 있으며 exact joint 비교는 posterior oracle도 이용한다. 알려진 SBM의 sum-product 연산 자체를 새 메소드의 novelty로 주장하지 않는다. 일부 loopy BP 메시지는64회 뒤에도 잔차가 남으므로 exact inference나 수렴 보장을 주장하지 않는다. 다음 학습 대상은 observation mask를 받는 role-conditional likelihood potential이며, 현재 oracle potential과의 간극 및 같은 예산의 Plain/일반 decoder 대조를 해결해야 한다. 이 후속 모델은 아직 학습하지 않았다.

출처: [보고서와 수식](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/REPORT.md), [고정 프로토콜](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/protocol.json), [전체 지표](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/results/aggregate.json), [paired CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/results/paired_contrasts.json), [독립 검증과 audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/contrastive_decoder/results/audit.json).

## 20. Learned Mask-aware Relational Potential: oracle 확률을 제거한 MVP

관측 edge·관측 nonedge·unknown을 구분하는 native pair potential을 학습했다. Shared relation encoder가 관측 통계와 context에서 가설별 density/signal을 출력하고, 양·음 likelihood coupling을 만든 뒤64회 cavity sum-product로 helper label을 합산한다. 참 SBM density/signal, sampled role, query/helper label은 모델 forward 입력에 없다. 다만 binary homophilic SBM-shaped potential family와 네 role 가설은 구조적으로 고정돼 있다. 새 schema 일반화 모델은 아니다.

기존 posterior D_all은 prior/seed별로 고정했다. 기존2048개 train latent/role/context bank를 재사용하되 절반의 edge는 원래대로, 나머지는 다른 density/signal에서 생성했다. MCAR 관측률은 .5/.8/1.0이었다. Predictive Bayes KL과 별도로, 관측 pair25%를 가려 auxiliary encoder 입력에서 제거한 후 sampled true role/latent class로 heldout edge NLL을 감독했다. Unknown pair는 음성 정답에 넣지 않았다. Synthetic latent supervision의 접근 권한은 추론과 구분한다.

4 arms(BP+aux / BP-predictive-only / signed+aux / Large Plain distill) ×3 priors ×3 seeds=36개 모델을192 updates×batch16의 같은 task schedule로 학습했다. Potential 모델7,618개 trainable parameter, Large Plain14,655개다. 새 optimizer update6,912회, 기록된 학습 시간 합계297.7초다. Plain은 기존 interface 때문에 O 자체 대신 B*O를 받는다. 동일 update/data budget이지 동일 FLOPs·입력 정보·auxiliary 감독이 아니다.

새 graph는 prior별192개(총576개), context는 총1/2/4개다. Matched 및 held-out density .12/.38, signal .95, MCAR 관측률 .2, block mask를 포함한7case를 paired로 평가했다. 아래는 matched d=.25/eta=.72/full observation의 Bayes predictive KL이다.

| Decoder + frozen learned q | Independent | Exclusive | Coupled |
|---|---:|---:|---:|
| 기존 Positive D | 0.063955 | 0.052586 | 0.085548 |
| 새 Large Plain distill | 0.071414 | 0.073581 | 0.085079 |
| Learned signed + aux | 0.057612 | 0.049333 | 0.079295 |
| Learned BP, predictive-only | 0.015946 | 0.011013 | 0.028391 |
| Learned BP + aux | 0.016319 | 0.011275 | 0.028654 |
| Oracle potential BP | 0.015134 | 0.010069 | 0.028153 |
| Train-fitted constant potential BP | 0.018514 | 0.012652 | 0.028160 |

Learned BP+aux는 세 prior 모두 Positive·새 Large Plain·learned signed보다 KL이 유의하게 낮았다. Exclusive accuracy는58.45→65.70%였다. 같은 learned q의 joint−factorized KL 차이는−0.00976 [−0.01130,−0.00833]으로 joint 보존의 이득도 남았다. 단일 context에서도 약50% query에서 반대 class를 선택해 기존 positive-only 제약을 벗어났다.

반면 auxiliary−predictive-only의 KL 차이는 Independent+0.000374 [0.000187,0.000587], Exclusive+0.000261 [0.000144,0.000364], Coupled+0.000263 [−0.000028,0.000656]였다. 현재 matched 결과에서 추가 likelihood loss의 독립적 성과는 없다. Fitted constant 대비 Independent/Exclusive는 개선됐지만 Coupled는+0.000494 [0.000042,0.001140]로 나빴다. Constant 대조와 비교하지 않고 adaptive encoder의 새 학습 기여로 해석하면 과장이다.

세 병목은 exact/BP 및 exact/frozen q의 모듈 교체로 분리했다. Exclusive learned potential에서 exact inference+exact q KL0.001982, BP+exact q0.003456, BP+frozen q0.011275였다. 이 차이는 interaction 때문에 additive KL decomposition이 아니다. Oracle teacher는 참 d/eta에 조건부이며 student는 이를 받지 않아, varied-parameter 조건에서는 식별 불확실성도 간극에 포함된다.

Density .12 외삽에서는 Independent/Coupled가 새 Plain보다 나빴고, 높은 density/signal에서도 oracle gap이 커졌다. 새로운 mask에서는 대체로 개선을 유지했지만 이것만으로 unseen semantic schema나 실제 heterograph 일반화를 주장하지 않는다. Frozen q도 원래 고정 확률·full observation에서 학습된 모델이라 OOD posterior shift가 남는다.

**판정:** 학습 가능한 signed likelihood decoder로의 치환은 성공했다. 추가 masked-edge likelihood loss의 새 학습 원리 주장은 현재 결과에서 지지되지 않는다. 범용 HeteroGraphPFN 확대는 보류하고, 학습 목적의 증분·constant 대비 적응 가치·다른 생성 family에서의 일반화를 별도로 해결해야 한다. 이 MVP에서 posterior를 재학습하거나 평가 결과로 loss weight를 튜닝하지 않았다.

출처: [전체 보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/REPORT.md), [고정 protocol](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/protocol.json), [전체 지표](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/results/aggregate.json), [paired CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/results/contrasts.json), [병목 분석](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/results/bottlenecks.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/learned_potential/results/audit.json).

## 21. Known-family Parameter Generalization: uncertainty head가 필요한가?

새 학습 없이 기존 MRP의576개 개발용 graph를 재사용했다. 같은 native masked likelihood, 같은 frozen role q와 BP를 유지하고 constant, 기존 amortized point, observed-density 추정, grid marginal-likelihood point(EB), posterior mean, parameter distribution marginalization을 비교했다. 현재 실제 generator는 관계 둘에 하나의 density/signal을 공유하므로 통계 추정기도 그 결합을 보존했다. True parameter·role·query/helper labels는 추정기 입력에 없다.

Broad prior는 d~Uniform(.01,.49), eta~Uniform(0,.98)에 고정한9x9 Gauss-Legendre quadrature다. MRP의 여섯 train parameter 조합과(.5,.1,.1,.1,.1,.1) 질량을 유지한 training-support prior도 비교했다. 모든 latent label을 합산해 parameter posterior를 계산했으며,8개 deterministic posterior 입자를 각각64회 BP로 예측했다. 점추정에는512회 BP를 주어 pair-message update budget을 맞춘 대조도 포함했다. Evidence scan은 공통이지만 latency와 FLOPs가 완전히 같지는 않다.

대표 Exclusive 조건의 Bayes predictive KL:

| 조건 | 현재 point | 관측 density + 고정 signal | Broad grid EB64 | Broad distribution8 | Oracle BP |
|---|---:|---:|---:|---:|---:|
| Matched | 0.011275 | 0.016164 | 0.029068 | 0.019836 | 0.010069 |
| Density .12 | 0.007245 | 0.001285 | 0.009475 | 0.002078 | 0.000272 |
| Density .38 | 0.085760 | 0.070381 | 0.070731 | 0.077996 | 0.064990 |
| Signal .95 | 0.111742 | 0.122959 | 0.090418 | 0.123913 | 0.075175 |

Density-low에서는 세 prior 모두 저렴한 observed-density 점추정이 현재 모델과 broad distribution보다 좋았다. Distribution−observed-density의95% CI도 세 prior 모두0보다 높았다. Signal-high에서는 grid EB 점추정이 세 prior 모두 현재 모델과 broad distribution보다 낮은 KL을 얻었다. 현재 amortized potential의 OOD 적응력이 고전 추정기를 넘는다는 주장은 지지되지 않는다. 다만 matched에서는 현재 amortized point가 broad 통계/분포 추정기보다 대체로 강하므로, 전체 모델이 단순 통계로 대체됐다는 뜻도 아니다.

Distribution8−posterior-mean512는 density-low와 sparse-mask에서 세 prior 모두 유의하게 음수였다. 저장된 point parameter에 exact latent inference를 적용한 모듈 대조에서도 low-density/sparse의 mean 대비 이득이 남았다. 그러나 low-density에서는 더 단순한 observed-density 방법을 못 넘고 sparse-mask에서는 현재 amortized point가 더 좋았다. 특정 상황에서 uncertainty marginalization의 가치가 있다는 것과 전체 방법의 최강 기준선 우위는 구분한다. Training-support distribution이 matched Coupled에서 current point보다 좋았지만 같은 prior의 posterior mean도 거의 같은 개선을 얻어, 이득 전체를 분포 유지에 귀속할 수 없다.

같은 broad parameter-integrated exact conditional을 두고 role weights만 coherent hierarchical q(h|D)로 바꾼 secondary control도 평가했다. Density-high의 frozen q→hierarchical q KL은 Independent0.09030→0.02649, Exclusive0.08051→0.03687, Coupled0.07936→0.01215였다. 이는 새 posterior network의 성과가 아니라 known-family exact 모듈 교체다. Potential만 바꾸고 role posterior의 OOD 적응은 무시해도 된다는 해석 역시 지지되지 않는다.

Broad prior의 exact distribution은 finite quadrature에 조건부로만 exact이며 continuous parameter posterior를 완전히 계산했다고 주장하지 않는다. 사전 지정 subset에서9x9 vs13x13 KL 최대1.5e−7을 확인했고, 그 결과로 grid를 고르지 않았다. Oracle true-parameter BP도 frozen-q·BP 오차 때문에 절대 성능 상한은 아니다. Primary target은 실제 생성 파라미터에 조건부인 Bayes predictive여서 broad hyperprior의 최적성과도 구분한다.

**판정:** Distributional head를 새로 학습할 Go 근거는 아직 부족하다. 고전적인 density/marginal-likelihood 추정기를 강한 baseline으로 유지하고, parameter prior와 role adaptation을 함께 평가해야 한다. 새 optimizer update·head·schema 확장은0회이며, test case에 따라 추정기를 사후 선택하는 배포 규칙을 만들지 않았다.

출처: [보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/REPORT.md), [고정 protocol](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/protocol.json), [전체 지표](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/results/aggregate.json), [paired CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/results/contrasts.json), [exact point 대조](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/results/exact_point_manifest.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/parameter_generalization/results/audit.json).

## 22. Joint Role–Parameter Grid Posterior: BP에서도 효과가 유지되는가?

새 학습 없이 기존576개 개발용 graph·63개 조건에서 모든 parameter grid×role의 BP output을 계산했다. Broad81개/training-support6개 node, role4개, BP64회·damping0.5를 고정했다. Native observed edges/nonedges의 likelihood에서 context 이외의 target/helper labels를2^12 상태로 모두 주변화하고, log prior H+log prior xi+log evidence를 하나의 joint softmax로 정규화했다. 실제 parameter·role·query/helper label은 추정기 입력에 없다.

같은 conditional parameter posterior와 **같은 BP branch output**에 frozen role q 또는 coherent joint weights를 적용했다. Density-high의 Bayes predictive KL:

| Role prior | Existing MRP | Grid + frozen q | Joint-grid BP | Joint marginals의 곱 | Joint-grid exact |
|---|---:|---:|---:|---:|---:|
| Independent | 0.093935 | 0.089169 | 0.026699 | 0.031882 | 0.026490 |
| Exclusive | 0.085760 | 0.078467 | 0.037090 | 0.044102 | 0.036874 |
| Coupled | 0.112820 | 0.078075 | 0.017615 | 0.019049 | 0.012149 |

Joint−frozen BP KL 차이는 Independent−0.06247 [−0.07256,−0.05301], Exclusive−0.04138 [−0.04859,−0.03491], Coupled−0.06046 [−0.07435,−0.04626]였다. Signal-high에서도 세 prior 모두 joint update가 frozen grid mixture보다 좋았다. 이전 exact-decoder 분석에만 한정된 효과는 아니다.

단, density-high에서 joint−frozen accuracy의 CI는 Independent/Exclusive 모두0을 포함하고 Coupled는 둘 다70.38%로 동일했다. Coupled H=00은 uniform, H=11만 informative라 같은 conditional parameter mixture에서 role weight는 confidence만 바꾸고 argmax는 유지한다. 이 큰 KL/NLL/Brier 개선을 class-decision 개선과 동일시하면 안 된다. MRP67.78→joint70.38%의 정확도 개선도 frozen-grid 단계에서 이미 얻었다.

추가 marginal-product 대조는 q_J(H|D)q_J(xi|D)로 joint의 두 주변분포를 그대로 보존했다. H는 전체 role assignment이므로 이 대조에서도 Exclusive의 내부 role 조합 제약은 유지된다. 이전 relation별 Bernoulli factorization과 다르다. Density-high에서 joint−product BP KL 차이는 Independent−0.00518, Exclusive−0.00701, Coupled−0.00143이며 세 CI가 모두0보다 낮았다. Exact conditional decoder에서도 이 이득이 남았다. 다만 큰 frozen→joint 효과 전체를 이 dependence 증분에 귀속하지 않는다. Frozen hybrid도 q_old(H|D)q(xi|H,D)이므로 정규화된 joint distribution이지만, 하나의 선언된 likelihood/prior에서 나온 posterior는 아니다.

강한 fixed-potential 대조도 확인했다. Train-fitted constant를 유지하고 role posterior만 해당 native likelihood로 갱신하면 matched KL은 Independent0.00344, Exclusive0.00541, Coupled0.00607이었다. Training-prior joint의0.00322/0.00526/0.00512와 차이의 CI는 모두0을 포함했다. 따라서 matched에서는 parameter distribution 자체의 필요성이 아직 약하다. 반면 density-high에서는 broad/training joint 모두 constant+updated-role를 세 prior에서 유의하게 이겼다.

Prior는 여전히 중요하다. Broad joint는 matched Exclusive와 sparse-mask에서 기존 MRP보다 나쁘며 training-support joint가 더 좋은 조건이 있다. Low-density에서는 training-support prior가 broad보다 나쁘다. Case별 prior를 사후 선택하지 않았고 하나의 범용 prior를 확립하지 않았다.

Joint/frozen/product의 BP 계산은 같지만, full broad grid는 point BP 대비81배의 branch-message updates와 exact latent evidence 계산을 요구한다. 입력 source·bank·기존 예측 hash는 불변이며 direct native joint evidence 오차8.3e−16, 이전 exact-grid reference readback 오차1.1e−15를 확인했다. 큰 실제 그래프로 이 계산이 그대로 확장된다는 증거는 아니다.

**판정:** Coherent joint evidence updating의 Conditional Go를 유지한다. BP에서도 큰 OOD 이득이 있고 H–xi 결합의 독립적 증분도 일부 조건에서 확인됐다. 다음 amortization의 학습 대상은 같은 evidence로부터 joint log weights와 posterior predictive를 구성하는 연산이어야 한다. 하지만 현재 결과는 알려진 SBM·prior·exact latent evidence를 사용하는 model-based baseline이며 새 학습 메소드의 성과나 unseen-schema novelty는 아니다. 이 실험에서는 새 head/모델을 학습하지 않았다.

출처: [보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/REPORT.md), [고정 protocol](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/protocol.json), [전체 지표](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/results/aggregate.json), [paired CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/results/contrasts.json), [posterior 진단](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/results/posterior_diagnostics.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/joint_role_parameter/results/audit.json).


## 23. Evidence-Amortized Joint PFN: 고정 BP 위에서 joint evidence를 학습할 수 있는가?

Shared candidate encoder가 알려진 Bernoulli local likelihood와 analytic partition 초기값의 correction을 예측하도록 했다. Joint model10,369개/independent heads10,370개 파라미터, 동일 초기 encoder·train tasks·batch/context schedule·192 updates·3개 seed·3개 role prior로 총18개 모델을 학습했다. 기존2,048 train graphs/prior를 재사용하고, 별도128 validation graphs/prior에서 predictive KL로64/128/192 checkpoint를 선택했다. Query/helper labels 및 실제 role/density/signal은 student 입력에 없고, optimizer objective는 exact joint posterior KL 하나다.

Broad81×role4 grid의 potential과64-step masked BP branch를 고정했다. 신경망을 전혀 학습하지 않은 analytic initializer도 유지해 알려진 likelihood inductive bias와 학습 증분을 분리했다. 신규 teacher는 알려지지 않은 parameter를 broad prior로 주변화하는 coherent grid posterior다. 기존 보고서와 비교하는 아래 Bayes KL은 실제 생성 parameter 조건부 reference이며, 두 reference를 혼동하지 않는다.

| Density-high / prior | Existing MRP | Frozen grid | Independent heads | Analytic | Constant + updated role | EAJ | Exact joint BP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent | 0.093935 | 0.089169 | 0.074685 | 0.073120 | 0.054183 | 0.057180 | 0.026699 |
| Exclusive | 0.085760 | 0.078467 | 0.078027 | 0.087409 | 0.050726 | 0.071467 | 0.037090 |
| Coupled | 0.112820 | 0.078075 | 0.044072 | 0.048186 | 0.062036 | 0.036083 | 0.017615 |

Density-high/signal-high6개 prior×condition에서 EAJ−independent, EAJ−analytic, EAJ−own-marginal-product KL의95% paired graph CI가 모두0보다 낮았다. Joint evidence 학습과 학습된 H–parameter dependence의 predictive 증분은 확인됐다. Frozen-grid 대비는5개 개선·1개 악화로, signal-high Exclusive는0.12463→0.13132로 나빠졌다.

하지만 EAJ는6개 주요 OOD 조건 모두 정확한 teacher marginals의 곱보다 predictive KL이 높다. Joint posterior KL도 해당 optimal product posterior의 최소 KL인 teacher MI보다 여전히 높다. Posterior 근사의 잔여 오차가 크며, joint 표현을 유지한다는 사실만으로 정확한 independent inference를 넘지는 못했다. 강한 constant+updated-role 대조와의 density-high 차이는 Independent+0.00300 [−0.00648,+0.01206], Exclusive+0.02074 [+0.01346,+0.02791], Coupled−0.02595 [−0.04309,−0.01097]였다. Classical control에 대한 범용 우위도 없다.

Signal-high에서는 constant+updated-role가 세 prior 모두 EAJ보다 유의하게 좋았다. 이 classical baseline은 parameter81개를 열거할 필요도 없으므로, neural evidence의 범용적인 품질·비용 우위를 입증한 결과가 아니다.

Density-high에서 EAJ−independent accuracy CI는 세 prior 모두0을 포함했다. 이번에 명확한 추가 효과는 확률적 예측 품질이며 독립적인 class-decision 우위를 입증한 것은 아니다. Posterior-only 비용은 exact latent evidence보다약1.9–2.9배 낮지만, 동일한 full-grid BP를 포함하면 총 시간 감소 추정은약6–12%뿐이다. 같은 품질의 exact-grid 대체나 실제 heterograph scalability를 통과했다고 보지 않는다.

학습 전후 관계·클래스·노드 순열과 unknown pair 무시, zero-observation prior 복귀, source/bank hash, validation seed 분리, checkpoint selection readback을 확인했다. Float32 BP cache와 이전 float64 reference를 같은 weights로 합산한 최대 차이는2.17e−8이었다. 평가를 보고 checkpoint를 바꾸지 않았다.

**판정:** 작고 알려진 생성 family에서 shared evidence amortization의 학습 증분은 Conditional Go다. Exact-grid 대체·full HeteroGraphPFN 확장은 보류한다. Unknown semantic schema, 실제 graph, relation/parameter prior를 모르는 상황, 범용성·논문 novelty는 미입증이다.

출처: [보고서](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/REPORT.md), [protocol](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/protocol.json), [지표](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/results/aggregate.json), [paired CI](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/results/contrasts.json), [strong constant control](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/results/constant_contrasts.json), [audit](C:/Users/seongjin/Desktop/code/gts_pfn/joint_role_pfn/evidence_amortized/results/audit.json).

# SPFN01 결과 심사 — method / prior art

검토일: 2026-11-17. **감사 PASS, 복합 성공 기준 FAIL.** 같은 토큰 예산에서 head 전체를 group moment로 요약하는 arm은 정해진 raw-anchor selector보다 synthetic tail recall이 높았다. 그러나 tail에 우선 슬롯을 주는 규칙의 우월성은 확인되지 않았고, real 세 그래프의 overall balanced accuracy에서는 학습 없는 PPR이 이겼다. 세 결과는 서로 다른 질문에 답한다.

검토 근거: `PLAN_v2.md`, `model.py`, `run.py`, `data.py`, `results/summary.json`, `audit.json`, `checks.json`, `training.json`, `artifact_manifest.json`. 본 reviewer는 사전 method 검토에서 raw-selection 대조군을 보강하도록 권고했다. 이 문서는 외부 blind review나 새 모델 실행이 아니다. 저장된 artifact만 읽고 비교·timing을 확인했다.

## 통과한 비교와 통과하지 못한 가설

Main audit는 96 paired synthetic graph, 15 real draws, 1,776 저장 probability arrays를 재채점했고 target optimizer step은 0이다. 감사와 manifest의 lock은 `18e8aba6244630f5257fe2f92e44250276661d19e08d41ebb8c98efdf3796ea1`로 일치한다. 3,705-parameter shared decoder를 arm별로 학습한 12 fits/3,840 optimizer steps다. Model/runner를 읽은 범위에서 이 결과를 실행 장애나 numerical check 실패로 해석할 근거는 없다.

| 사전 synthetic 비교 | 차이(pp) | 명목 97.5% paired CI(pp) | 판정 |
|---|---:|---:|---|
| tail − top, tail recall | +2.611 | [+1.183, +4.066] | PASS |
| tail − proportional, tail recall | +0.658 | [−0.340, +1.657] | FAIL |
| tail − full, overall balanced accuracy | +0.023 | [−0.616, +0.678] | PASS, margin −2pp |

Tail은 관측 라벨이 1–4개인 클래스의 macro recall이다. Graph별 세 seed 평균을 사용한 96-graph stratum bootstrap이며 seed 모집단에 대한 추론은 아니다. Holdout의 토큰 비율 6.136×와 no-training point-estimate floor도 통과했다. Synthetic tail recall 74.508% 대 best rule ridge 61.632%의 차이는 이 bank에서 학습된 decoder가 쓸모 있다는 보조 근거다. 복합 gate에서 빠진 항목을 이 보조 성과로 대신할 수 없다.

**Tail − top이 보존하는 증거.** 두 arm은 per-class quota와 클래스 전체 관측 라벨 수가 같다. Top은 finite PPR과 세 hop의 평균 feature cosine을 class 안에서 rank-fusion하여 고르며, 실제 singleton mass와 count=1을 받는다. 관측 라벨이 1–4개인 클래스에서는 두 arm 모두 그 클래스의 모든 anchor를 singleton으로 보존한다. 달라지는 것은 head에서 전체 anchor의 group moments를 읽느냐, 두 개의 선택된 raw anchors를 읽느냐이다. 따라서 +2.611pp는 **이 summary+decoder training과 이 raw-selection+decoder training의 비교**다. Scarce-anchor 보존 효과를 따로 입증하지 않으며, gain을 PCA partition, PPR weighting, count weighting 중 하나에 귀속할 수도 없다. Gain이 tail뿐 아니라 head point estimate에도 나타난다(75.109% vs 71.181%).

**Tail − proportional이 남긴 불확실성.** 두 arm은 같은 총 토큰 예산·그룹 moment 계산·decoder family를 쓰되 클래스에 할당할 수 있는 슬롯 수가 다르다. Tail의 희소 클래스 recall 점추정치는 74.508%, proportional은 73.850%다. 반면 head recall은 75.109% 대 76.367%, overall balanced accuracy는 74.682% 대 74.728%다. 이 방향은 슬롯 교환과 양립하지만, tail recall CI가 0을 포함하므로 희소 우선 규칙의 개선도 동등성도 확정하지 못한다. Full guard는 **overall** 비열등성이다. Full의 tail recall 점추정치(75.123%)를 tail이 넘지 않았으며 guard를 클래스별 보존으로 확대할 수 없다.

## Real transfer의 한계와 남는 질문

같은 RP24 feature, official train/test 분리, 2 vs 16 관측 라벨 및 클래스별 32개 query를 사용한 5 paired draws/graph의 기술 통계다. Tail 역할은 관측 기준이며 모집단의 minority와 무관하다. 동일 그래프의 draw가 여러 개라는 이유로 외부 그래프 generalization CI를 붙이지 않는다.

| Graph | Tail arm overall / tail (%) | Proportional overall / tail (%) | PPR overall / tail (%) |
|---|---:|---:|---:|
| Cora | 41.37 / 43.44 | 42.41 / 48.91 | 62.95 / 59.84 |
| CiteSeer | 24.90 / 31.25 | 23.06 / 32.08 | 42.60 / 31.04 |
| PubMed | 48.13 / 51.04 | 47.64 / 52.71 | 60.00 / 52.19 |

PPR은 세 그래프 모두에서 **모든 학습 arm보다 overall balanced accuracy가 높았다**. Tail에 대한 overall 우위는 약 21.58/17.71/11.88pp다. 다만 CiteSeer tail recall은 tail arm이 PPR보다 +0.208pp라는 기술적 차이이며, PPR의 NLL도 CiteSeer/PubMed에서는 tail보다 나쁘다(4.518/1.485 vs 1.825/1.019). 그러므로 PPR이 모든 metric에 우세하다는 설명은 틀리다. 반대로, 세 real graph의 tail recall 점추정치 모두에서 proportional이 tail보다 높았고 synthetic allocation 가설을 real 결과로 되살릴 근거도 없다.

현재 자료는 synthetic prior와 이 real 프로토콜 사이의 상대 순위가 뒤집혔음을 보인다. 왜 뒤집혔는지는 식별하지 못한다. RP24, priors의 그래프와 feature 신호, real 측정의 클래스 수/graph 규모, decoder가 사용하는 신호가 모두 다를 수 있다. 간단한 구조 score가 group state에 수학적으로 남는다는 사실은 확인할 수 있다. 모든 anchor를 partition한 경우, 클래스 $c$의 balanced PPR score는 $s_{qc}=n_c^{-1}\sum_{g\subset c}m_{qg}$다. 이는 PPR baseline이 클래스별 seed를 $1/n_c$로 놓는 것과 같은 finite diffusion이다. 클래스 간 합이 0일 때의 uniform fallback까지 공통으로 정의할 수 있다. 현재 shared attention은 mass를 비선형 입력으로 쓰며 이 합을 그대로 출력하도록 강제하지 않는다. 따라서 이 실패를 “압축 과정에서 PPR 정보가 사라졌다”로 기술하면 부정확하다.

남는 한 가지 과학 질문은 **구조-only PPR 증거를 그대로 보존했을 때, 학습된 feature evidence가 그 위에서 real graph 예측을 개선할 수 있는가**다. 이 질문은 현 모델이 구조를 반드시 무시했다는 전제 없이, 실험이 비교한 결과와 원인 사이의 간격을 줄인다. 현재 3개 reused graphs에서 읽은 결과는 개발 증거이며, 사후 혼합이나 같은 final bank 재튜닝으로 답하지 않는다. 지금 실행하거나 새 성능을 주장할 근거는 없다.

## 토큰 절감과 실제 계산은 다르다

48 larger synthetic graphs에서 mean full/tail decoder token 수는 59.188/9.646으로 **6.136×**다. 반면 tail은 group mass와 $3F$ feature moment를 함께 확산하므로 mean diffusion columns가 **246.646**이며, optimized full은 singleton mass만 보내므로 **59.188**이다. Manifest의 mean largest seed block은 tail 401,333 B, full 9,301 B다. 이는 단일 작업 배열의 크기이지 peak process memory가 아니다. Pair tensor 자체는 tail이 더 작다(21,536 B vs 132,565 B 평균). 어느 축의 압축인지 명시해야 한다.

다음은 draw0에 대해 warm-up 뒤 세 번 재어 기록한 seed11 CPU profile의 median [min, max] ms다. Projected input 이후의 graph 변환, 그룹화, diffusion, pair 구성, scoring을 포함한다. Dataset load와 공통 RP는 profile 밖이며, RP의 별도 one-off 기록은 2.64/8.61/7.01ms다.

| Graph | Tail total | Optimized full total | Tail / full median |
|---|---:|---:|---:|
| Cora | 41.64 [41.48, 43.10] | 23.07 [23.02, 23.20] | 1.80× |
| CiteSeer | 46.76 [46.18, 51.34] | 20.86 [20.81, 21.59] | 2.24× |
| PubMed | 112.49 [110.15, 114.06] | 32.48 [31.33, 32.55] | 3.46× |

Substantial time은 tail의 feature-weighted diffusion이며, decoder token 감소가 이 비용을 상쇄하지 못했다. 이 CPU 프로필로 일반 latency나 O(1) inference를 주장할 수 없다. 구현 비용의 명확한 경계는 있다. Group당 $|g|<1+3F$이면 raw singleton mass columns를 diffuse한 뒤 query에서 $\sum_{i\in g}D_T[q,i]H_i$를 계산해 같은 first moment를 얻을 수 있다. 이는 선형 연산 순서 변경일 뿐이며 지금 구현·측정한 개선은 아니다. 이 재배치의 가능성은 real efficacy 실패와 독립적이다.

## novelty와 폐쇄 범위

[LoGIC](https://arxiv.org/html/2609.05955v1)는 labeled row 예산과 PPR/feature/coverage retrieval을, [Ephris](https://arxiv.org/html/2609.37057v1)는 learned compression과 sparse graph ICL을 이미 다뤘다. [NodePFN](https://arxiv.org/abs/2604.19028)의 synthetic graph prior, [PPRGo](https://arxiv.org/abs/2007.01570)의 propagation, [G-META](https://proceedings.neurips.cc/paper_files/paper/2020/file/412604be30f701b1b1e3124c252065e6-Paper.pdf)의 graph-local support prototypes도 선행 범위다. 여기서 보존할 결과는 **같은 decoder와 토큰 예산에서 finite-diffusion support summary가 특정 raw selector보다 synthetic tail 예측을 개선하고 전체 bAcc가 full guard를 통과했다**는 비교 증거다. 실제 LoGIC/NodePFN/Ephris와의 성능 비교를 실행한 것은 아니다. 이 구현은 전체 graph를 diffuse하고 first moments만 유지하므로 materialized-subgraph 가속, logits/posterior의 무손실 복원, teacher distillation을 입증하지 않는다.

닫힌 범위는 이 작은 decoder, prior, 320 updates/fit, 완전 feature, 모든 클래스에 ≥1 관측 라벨, 사전 명시한 quota와 8-round restart다. 그 범위에서 희소 우선 allocation의 composite 가설은 실패했고 실제 transfer와 속도 개선도 확인되지 않았다. 관측 support 부족의 partial positive와 실패한 가설을 함께 보존한다. Missing feature, zero-anchor, 자연 class prevalence 및 대규모 graph 일반화의 결론은 없다.

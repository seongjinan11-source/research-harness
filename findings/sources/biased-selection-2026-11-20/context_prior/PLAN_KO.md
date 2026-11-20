# CPC 사전 명세 v2: 편향 context에서 community별 클래스 사전확률 — 무엇이 선택 편향이고, 라벨을 어디에 쓸 것인가

- 작성일: 2026-11-20. **사전 명세 v2이고, 아직 구현하거나 실행하지 않았다.** v1은 [PLAN_v1_KO.md](PLAN_v1_KO.md)에 보존했다. 이번 판은 독립 리뷰 3개(§9)를 반영했다.
- 선행 근거는 바꾸지 않는다: Phase 0 `DECISION_PHASE0_KO.md`, `../2026-11-20_em_transport/DECISION_S0_KO.md`, `DECISION_S1_KO.md`.
- W와 gts_pfn은 읽기만 한다. W의 target 그래프 8개는 사용자 승인 없이 쓰지 않는다.
- 계산 경로: frozen 공식 GraphPFN, W 프로토콜. dev context는 Cora/CiteSeer seed 300–309, PubMed/photo seed 200–209(W dev draws)이고, ρ는 10/20, 정책은 uniform과 commconc다.

## 1. 출발점과 v1에서 바뀐 이해

**확정 (S1, oracle)**
- 특징 수송 oracle의 상한은 0.43–0.76이다. O-prior는 0.95–1.56이고 uniform 성능도 올린다.

**v1 리뷰가 밝힌 해석 문제 (확정된 수학적 사실)**
- O-prior의 보정항은 (k, c)마다 상수다. 그래서 같은 community 안의 노드끼리는 pairwise AUC가 변하지 않는다. AUC 이득은 전부 **community 간 쌍**에서 나온다. 클래스별 전역 상수는 AUC를 전혀 바꾸지 않으므로, logit adjustment는 AUC에서 no-op이다.
- 따라서 S1의 O-prior 이득이 "선택 편향을 고친 효과"라고 할 수 없다. pool의 community 라벨 구성을 순위에 넣어 준 효과(정보 주입)일 수 있다. uniform 이득은 이 해석과 맞는다.

**누출 정정 (S1에 소급 적용)**
- PubMed와 photo의 dev `eval_mask`는 val뿐이다. 그래서 S1과 Phase 0 oracle의 pool(`~eval_mask`)에 test 노드 라벨이 들어갔다.
- 이번 cycle은 모든 arm에서 PubMed/photo의 pool을 **¬(val ∪ test)** 로 둔다. S1 수치와 달라지는 부분은 기록한다.
- 한계: PubMed와 photo의 S3 확인 평가 노드(test)는 S1과 Phase 0 oracle에서 이미 집계 수준으로 쓰였다. Cora/CiteSeer 절반 B와 같은 수준의 한계다.

**proxy (리뷰 C, GraphPFN 없이 LR 사용, seed 3개, 가설 수준)**
- commconc에서 각 클래스 pool 질량의 65–92%는, 그 클래스의 context가 하나도 없는 community에 있다. uniform에서는 8–26%다.
- context 라벨을 전파하는 E-LP는 선택 특유 이득이 CiteSeer −0.9, PubMed −0.4, Cora −0.2pt다. 편향을 물려받는다.
- Saerens EM과 준지도 GMM-EM도 PubMed/photo에서 평탄 prior와 큰 차이가 없다.
- E-moment는 community 특징 평균 μ_k를 클래스 원형들의 볼록결합으로 푸는 방법이다.
  - 참 원형을 쓰면 TV가 0.07–0.19이고 정책과 무관하다.
  - context 원형은 commconc에서 상대 오차가 1.1–2.4라 TV가 0.33–0.66으로 나빠진다.
  - 그래도 선택 특유 이득이 양수인 그래프가 가장 많다(PubMed +2.9, photo +6.6pt).
- 식별 논리(가설): context가 없는 community의 p(c|k)는 p(x|y,k) 불변 가정이 없으면 context만으로 식별되지 않는다. S1에서 클래스 조건부 offset이 확인되었으므로 그 불변 가정은 깨진다. 남는 경로는 두 가지다.
  - μ_k에서 출발하되 원형의 편향을 받는 경로(E-moment).
  - 진짜 라벨을 쓰는 경로.

**선행 연구 (기여로 주장하지 않는 것)**
- 편향 라벨 선택의 손실과 보정:
  - SR-GNN(Zhu et al., NeurIPS 2021, 2108.01099)
  - DGNN(2201.07708)
  - ReNode(2110.04099)
- 축 분해(구조 이동 대 label shift): Pair-Align(ICML 2024, 2403.01092), StruRW(2306.03221)
- nuisance z와 결합한 test-time prior EM: TTLSA(NeurIPS 2023, 2211.15646). R2/E-EM의 직접 선행이며, "spurious 상관도 유용한 prior"라는 주장을 담고 있다.
- 결합 이동 아래의 식별: Sparse Joint Shift(2209.08436)
- 집단 평균 posterior로 나누는 보정: Contextual/Batch Calibration(2102.09690, 2309.17249)
- partition 기반 coverage 라벨링: GraphPart(2201.09391)
- frozen graph ICL에서의 context 예산: LoGIC(2609.05955)
- 그 밖에 label-shift EM(Saerens 2002), C&S(Huang et al. 2021)
- 위 목록 중 리뷰 B가 원문을 확인한 것: SR-GNN, DGNN, TTLSA, GraphPart, Pair-Align, SJS. 나머지는 기억에 의존했다.

**이번 cycle에서 방어를 시도할 기여 (모두 아직 가설)**
1. commconc 손실 가운데 "선택 특유의 prior 왜곡"이 얼마인지를 정보량을 맞춘 oracle로 측정하고, 정보 주입 효과와 분리한다.
2. 라벨 없는 식별의 경계를 보인다. context가 없는 community의 질량에서 라벨 없는 추정기가 실패하는 조건을 정한다.
3. 같은 진짜 라벨 예산을 context, prior, 원형 중 어디에 쓸지에 대한 곡선을 GraphPart/LoGIC와 대조해 위치시킨다.

## 2. 지표

- **주 지표: pairwise AUC를 community 내부 쌍과 community 간 쌍으로 분해해 보고한다.** 전체 AUC도 함께 보고한다.
- **bAcc를 공동 주 지표로 둔다.** 출력 보정은 bAcc의 community 내부 판정도 바꾼다. W R2의 음성 결과는 bAcc로 보고되었다.
- **판정은 비율이 아니라 차이(pt)로 한다.**
  - 선택 특유 이득 SG = gain_commconc − gain_uniform. 같은 seed, ρ, 설정끼리 짝지어 계산한다.
  - Rec와 SRec(SG / loss, seed 합의 비)는 보조로 보고한다.
- 통계: seed 군집 bootstrap 10k. percentile과 BCa를 함께 보고하고, seed별 값도 표로 보여 준다.
  - 평가 가능성: 같은 단계 context에서 Δ_GP CI 하한이 2pt를 넘어야 한다. S3에서는 S3 context로 다시 계산한다.
  - 투표: 공용 투표 함수(`../2026-11-20_em_transport/stats.py`)를 쓴다.

## 3. 보정식과 추정기

- **보정식:** s_i = logp_i + β · [log p̂(c|k(i)) − log r(c|k(i))]
  - r ∈ {없음, m̄_k}. m̄_k는 community k에서 context가 아닌 노드들의 PFN posterior 평균이다. "없음"은 상수 π_L과 같고, AUC에서는 π_L과 차이가 없다.
- **축소는 count 기준이다:** p̂ ← (N_k p̃ + λ π_L)/(N_k + λ). N_k는 p̂를 추정하는 데 쓴 라벨 수, 또는 유효 표본 수다.
  - oracle은 S1과 같은 Laplace +1을 쓴다.
  - 단위 테스트: 참 p와 Laplace +1을 넣으면 S1 `o_prior`와 같은 값이 나와야 한다.

**oracle (pool 라벨 사용; pool은 위 정정을 따른다)**

| arm | 정의 |
|---|---|
| O-prior | S1과 같다(정정된 pool). |
| **O-swap** | p̂ = 같은 seed와 ρ의 uniform context에서 얻은 m̄_k^unif, r = m̄_k^cc. pool 라벨은 쓰지 않는다. 같은 수의 무작위 라벨이 줄 수 있는 정보까지만 쓰고, "편향 선택이 PFN의 community prior를 얼마나 틀었나"만 되돌린다. **선택 특유 prior 왜곡의 주 측정치다.** |
| O-info-n | 정책과 무관한 무작위 pool 라벨 n개(n = |ctx|)로 p̂를 만들어 두 정책에 똑같이 넣는다. 정보량을 맞춘 대조다. |
| O-prior-ctxK / -noctxK | community를 둘로 나눈 분해다. 교호작용 항 O-prior − ctxK − noctxK + GP도 보고한다. |
| O-ctx × O-prior (2×2) | 특징 축과 prior 축의 결합이다. 가산성과 Shapley 분해를 보고한다. 이 표가 있을 때만 "분해"라고 쓴다. |

**라벨 없는 추정기 (context 라벨, 그래프, 특징만 사용)**

| arm | 지위 | 정의 |
|---|---|---|
| **E-moment** | **유일한 주 후보** | 비context 전 노드로 μ_k를 구하고, context 원형 μ̂_c의 simplex 최소제곱 결합으로 푼다. 특징은 GraphPFN 전처리 특징 F(PCA64)다. 원형 오차와 참 원형 TV를 사전 진단으로 기록한다. |
| E-LP, E-PoE | 음성 대조 | 선택 편향을 물려받는다는 사전 예측을 확인하는 용도다. PPR은 클래스별로 정규화한다. |
| E-R2 | 음성 대조 | W `cpc`를 이식한다. (ref, m, β)는 W 파일에 기록된 dev 선택값으로 고정한다. EM 노드 집합은 비context 전 노드다. W와의 동일성 테스트는 W 노드 집합 모드에서 따로 한다. |
| E-ctxcount | 기준선 겸 추정기 | context의 community별 클래스 개수에 Laplace를 적용한다. 가장 단순한 대안이다. |
| E-flat (community batch calibration) | 진단 | p̂ = π_L, r = m̄_k. |

- **사전 제외 규칙:** context가 없는 community에서 TV가 평탄 π_L보다 개선되지 않는 추정기는 회복률을 계산하지 않고 탈락시킨다. 그 결과는 진단 표에만 남긴다.

**기준선 (같은 logp에서 시작; forward가 필요한 것은 따로 표시)**
- C&S 전체와 Correct 단계만:
  - 입력은 PFN 확률이다. 잔차는 context 노드의 one-hot에서 PFN 출력을 뺀 것이다.
  - context 노드의 PFN 출력은 context 자신에 대한 예측을 그대로 쓴다. 거의 라벨과 같으면 Correct가 아무 일도 하지 않을 수 있으며, 그 사실을 기록한다.
  - autoscale를 쓴다.
- LP-PoE-node: s = logp + β·log LP_i. community로 집계하지 않는다.
- **stacking:** context에서 5-fold cross-fit한 logistic 회귀. 입력은 [logp, community one-hot, LP 점수]다. 학습 가능한 CPC 판이다.
- **context 재표집 [forward 필요]:** community를 층화해 부분표집하고, community당 상한을 |ctx|/K_ctx로 둔다. 클래스 균형은 유지한다.
- REF-GP [forward 필요]: 새로 구현한다(k = 5, ALPHA = 0). W 일치 테스트를 하고, W gate 임계의 출처를 기록한다.

## 4. 소수 진짜 라벨 (Q3; 같은 라벨을 세 방식에 쓴다)

- **총예산:** M = f · |ctx|, f ∈ {0.1, 0.25, 0.5}.
- 라벨은 **pool \ ctx**에서 뽑는다. 뽑는 방식은 community 크기 비례 층화이고(사실상 무작위와 같다), seed는 단위 seed로 고정한다.
- assert 두 개를 둔다: 라벨 노드는 eval과 겹치지 않는다. 세 방식은 같은 노드를 쓴다.
- arm:
  - L-ctx: 라벨을 context에 더한다. forward가 필요하다.
  - L-prior: 라벨로 p(c|k)를 count 축소 추정한다(β = 1, λ = 1로 고정).
  - **L-moment:** 라벨로 클래스 원형을 다시 추정하고 E-moment를 돈다.
  - L-both: L-ctx에 L-moment를 더한다.
  - L-uncov: 같은 M을 context가 없는 community에만 배분한다(W의 +k 결과와 연결). forward가 필요하다.
- 함께 기록할 것: 새 context의 클래스 비율과 ρ, 그래프별 실제 M.
- 판정 정책은 commconc이고, uniform에서도 같은 비교를 보고한다.
- **사전 예측:** L-ctx ≥ L-prior. L-moment는 예측하지 않는다.

## 5. 단계

**E0 캐시 (forward)**
- 단위 160개마다 저장하는 것: GP logp, context 재표집 logp, REF-GP logp, L-ctx(f 3개), L-uncov(f = 0.25), L-both(f = 0.25).
- 단위당 forward 8회다. 단위당 forward 시간 실측치(PubMed 15초, photo 8초, Cora/CiteSeer 2초)로 계산하면 **약 1.3시간**이다. 실측 시간은 FREEZE에 기록한다.

**E1 dev 오프라인 분석**
- 평가 노드는 S1과 같다(PubMed/photo는 val, Cora/CiteSeer는 절반 A).
- grid는 메소드와 기준선에 같은 규모를 준다(각 4개 설정).
  - E-moment: β ∈ {0.5, 1} × r ∈ {없음, m̄_k}
  - C&S: α ∈ {0.5, 0.8} × 반복 {10, 50}
  - LP-PoE: β ∈ {0.5, 1} × PPR α ∈ {0.1, 0.2}
  - stacking: C ∈ {0.1, 1} × 입력 {LP 포함, LP 제외}
- **dev 선택:** gate 조건을 스스로 만족하는 설정 가운데, 평가 가능 그래프 평균 SG(AUC)가 가장 큰 것을 고른다. 기준선에도 같은 절차를 적용한다. 고른 결과를 `selection.json`으로 해시 고정한다. dev 수치로는 주장하지 않는다(승자의 저주).
- **gate (사전 지정):** 아래를 모두 만족해야 E-moment를 S3에 보낸다.
  - (a) 평가 가능 그래프 3개 중 2개 이상에서 SG(AUC) CI 하한 > 0, SRec 점추정 ≥ 0.15.
  - (b) 같은 그래프에서 uniform 변화 CI 하한 > −0.5pt.
  - (c) null_context(community 순열) 대조에서 SG가 0과 구별되지 않아야 한다. 구조만으로 생기는 이득이 아님을 확인한다.
  - 하나라도 실패하면 라벨 없는 경로를 음성으로 기록하고, S3에서는 Q3만 확인한다.
- **O-swap 진단:** O-swap의 SG가 2개 이상의 평가 가능 그래프에서 CI 하한 > 0이 아니면, "prior 축은 선택 편향 축이 아니라 추가 정보 축"이라고 기록한다. 이 경우 기여 1을 철회한다.

**S3 확인 (dev와 독립인 context; 노드 한계는 §1)**
- context: mkpol seed를 새로 쓴다(310–319). 이전 cycle PLAN v2 §5 S3의 counts와 uniform 짝 규칙을 그대로 쓴다. dev seed(200–209, 300–309)와 겹치지 않음을 assert로 확인하고, dev와 S3 context의 Jaccard를 기록한다.
- 평가: PubMed/photo는 test(pool = ¬(val ∪ test)), Cora/CiteSeer는 절반 B.
- **판정 설정은 하나다.** E1의 `selection.json`이 정한 설정을 쓴다. 주 설정(β = 1, r = 없음) 결과는 보조로 보고한다.
- S3 스크립트와 판정 스크립트는 E1 결과를 보기 전에 작성하고 FREEZE_E1에 포함한다.
- 비용: 단위 160개 × forward 약 8회, 약 1.3시간.
- **총 예상 시간: 약 3시간**(E0와 S3, 구현 제외).

## 6. 사전 판정 (S3, 공용 투표 함수, 평가 가능 그래프만)

| 판정 | 성립 | 기각 |
|---|---|---|
| D1 선택 특유 prior 왜곡 (O-swap) | SG(AUC) CI 하한 > 0 | CI 상한 < 0 |
| P1 라벨 없는 회복 (gate 통과 시) | SG CI 하한 > 0, uniform 변화 CI 하한 > −0.5 | SG CI 상한 < 0 |
| P2 기준선 대비 (C&S, Correct-only, LP-PoE, stacking, 재표집, REF-GP) | commconc 이득 차 CI 하한 > 0 | CI 상한 < 0 |
| Q3a L-prior 대 L-ctx (f = 0.25) | 차이 CI 하한 > 0 | CI 상한 < 0 |
| Q3b L-moment 대 L-ctx (f = 0.25) | 위와 같음 | 위와 같음 |
| Q3c L-both 대 L-ctx (f = 0.25) | 위와 같음 | 위와 같음 |

- AUC와 bAcc 두 지표로 판정한다. 두 지표의 결론이 다르면 "지표 의존"으로 보고한다.
- 평가 가능 그래프가 3개면 사실상 만장일치를 요구한다. "결론 없음"이 가장 가능성 높은 결과다.
- **라벨 없는 경로가 음성일 때의 보고 형식(사전 지정):**
  - context가 없는 community의 질량 비율과 추정기별 TV 표
  - 원형 오차와 TV의 관계
  - O-swap 결과
  - 이것들을 묶어 "context만으로는 식별되지 않는 질량"이라는 경계 결과로 쓴다.
- 사전 예측: D1은 불확실하다. P1 통과 가능성은 낮다. Q3a는 기각 또는 결론 없음일 것이다. Q3b는 예측하지 않는다.

## 7. 구현과 동결

- 파일: `cache.py`(E0), `cpc.py`(보정식과 추정기), `baselines.py`, `labels.py`(Q3 표집과 assert), `e1_analyze.py`, `s3_ctx.py`, `s3_run.py`, `s3_analyze.py`, `test_cpc.py`
- 단위 테스트:
  - `o_prior` 동일성
  - AUC 분해가 전체 AUC와 맞는지(장난감 예)
  - 클래스 상수 보정이 AUC에서 no-op인지
  - R2와 W의 동일성
  - C&S 손계산
  - Q3 assert
  - stats 투표
- FREEZE_E0(E0 코드), FREEZE_E1(E1, S3 코드와 판정 코드, E1 실행 전), `selection.json` 해시(E1 직후)를 남긴다.

## 8. 리뷰 기록 (v1 → v2)

- **리뷰 A (프로토콜, 통계, 비용):**
  - dev pool에 test가 들어 있다. → §1 정정, ¬(val ∪ test)
  - L 라벨이 context와 겹칠 수 있다. → pool \ ctx와 assert
  - community당 m 예산이 그래프마다 0.28–1.03배로 다르다. → 총예산 f·|ctx|
  - logit-adj와 r = π_L이 AUC에서 no-op이다. → 제거하고 E-ctxcount 추가
  - gate가 Rec 기반이다. → SG 기반
  - 선택 규칙이 모호하다. → 단일 판정 설정과 gate 내 선택
  - R2 기본값이 없고 eval 행에서 EM을 돈다. → W 선택값 고정과 노드 집합 정의
  - REF-GP 미구현이고 forward가 필요하다. → E0에 포함
  - grid가 비대칭이다. → 각 4개
  - 비용을 과소추정했다. → 약 3시간
  - S3 코드가 없다. → FREEZE_E1에 포함
  - BCa, 교호항, PPR 정규화, PCA 특징, m̄ 노드 집합을 명시했다.
- **리뷰 B (선행, 신규성):**
  - O-prior AUC 이득은 community 간 순위 정보 주입이다. → AUC 내부/간 분해와 bAcc
  - SRec는 정보 상보성을 분리하지 못한다. → O-swap, O-info, 차이 판정
  - 라벨 없는 식별은 S1이 확정한 p(x|y,k) 변화와 모순된다. → 기여 2를 경계 결과로 재서술
  - SR-GNN, DGNN, ReNode, Pair-Align, StruRW, TTLSA, SJS, Contextual/Batch Calibration, GraphPart, LoGIC를 추가했다.
  - 2×2 분해를 넣었다.
  - stacking, context 재표집, Correct-only 기준선을 넣었다.
  - Q3 사전 예측을 정직하게 고쳤다.
  - SR-GNN 전체를 GCN 기준선으로 쓰자는 제안은 이번 범위에서 제외했다. 사유: 비교 대상이 frozen ICL 후처리다. 필요하면 별도로 한다.
- **리뷰 C (메커니즘, proxy):**
  - 클래스 질량의 65–92%가 context 없는 community에 있다. → E-LP와 E-PoE를 음성 대조로 격하
  - Rec gate는 엉뚱한 이유로 통과할 수 있다. → SG gate와 null 대조
  - E-moment만 원리적인 근거가 있다. → 유일한 주 후보, 원형 진단 등록
  - R2/EM은 탈출하지 못한다. → 대조
  - Q3를 다시 설계했다. → 총예산 곡선, L-moment 추가
  - 축소 공식을 count 기준으로 바꿨다.
  - TV는 진단으로만 쓴다.
  - seed 표기를 고쳤다.
  - 음성 결과의 보고 형식을 사전에 정했다.

## 9. 변경 기록

- 2026-11-20 v1 작성.
- 2026-11-20 v2: §8의 리뷰 반영(구현 전, 실데이터 GraphPFN 결과 없음; 리뷰 C proxy는 LR 기반).
- 2026-11-20 구현 중 명세 보완 (E0 실행 전, 결과 없음):
  - stacking: context 노드에서 PFN 출력은 거의 라벨이다. 그래서 logp를 특징으로 쓰는 cross-fit은 context 밖 forward 없이는 불가하다. 대신 context에서 학습한 LR p(y | community one-hot [+ log LP])를 PoE(s = logp + log p_LR − log π_L)로 결합한다.
  - context 재표집: 클래스 c마다 community당 최대 ⌈n_c/K_c⌉개를 남긴다. K_c는 c의 context가 있는 community 수다. 클래스 개수는 기록한다.
  - L-prior는 새로 뽑은 라벨만으로 count 축소 추정한다(context count는 쓰지 않는다). L-moment의 원형은 새 라벨의 클래스 평균이고, 새 라벨이 없는 클래스는 context 원형을 쓴다.
  - E-moment·E-LP·E-PoE의 p̂에는 바닥 ε = 10⁻³ 혼합(π_L)을 쓴다. 크기 기반 λ 축소는 적용하지 않는다(리뷰 C 7).
  - REF-GP: W의 dev HI pass-1 전체 노드 예측은 저장되어 있지 않다. 그래서 선택 노드 일치는 W `picks` 함수 동일성 테스트로 대신한다(test_cpc.py 통과). W gate 임계 0.8의 출처는 미확인으로 남긴다.
  - O-ctx(2×2용)와 null context(gate c) forward를 E0에 추가한다.
  - 실측 비용: 단위당 Cora/CiteSeer 약 15초, photo 약 60초, PubMed 약 130초. E0는 약 2.4시간, S3는 약 2시간이다.
- 2026-11-20 E1·S3 분석 코드 동결 (E1 결과를 보기 전; E0 실행 중):
  - `s3_run.py`와 `s3_analyze.py` 대신 같은 코드에 단계 플래그를 쓴다: `cache.py --stage s3`, `score.py --stage s3`, `judge.py --stage s3`.
  - E1 gate의 세부 정의:
    - 그래프 단위로 (a), (b), (c)와 TV 탈락 규칙을 모두 만족해야 한다.
    - (c)는 null SG의 CI 하한 > 0이면 실패로 본다.
    - dev 선택은 (a)와 (b)를 만족하는 설정 가운데 평가 가능 그래프 평균 SG가 최대인 설정으로 한다. gate 통과는 그 설정이 (c)까지 만족해야 한다.
  - 기준선 선택: 모든 평가 가능 그래프에서 uniform 변화 CI 하한 > −0.5pt인 설정 중 평균 SG가 최대인 설정이다. 그런 설정이 없으면 전체 설정 중 최대다.
  - 평가 가능성은 AUC 손실로 정하고, bAcc 판정에도 같은 그래프 집합을 쓴다.
  - 이 줄을 붙여 PLAN 해시가 바뀌었다. FREEZE_E0의 PLAN 해시는 이 줄 이전 판을 가리킨다. E0 코드 파일의 해시는 바뀌지 않았다.
- 2026-11-20 E0 중단 (사용자 지시; Cora/CiteSeer 완료, photo 14/40, PubMed 0/40). 정식 E1 gate와 판정은 내리지 않았고, `selection.json`도 만들지 않았다. 부분 기술 통계는 `DECISION_E1_PARTIAL_KO.md`와 `results/interim_2graphs/`에 있다. 라벨 없는 CPC 메소드 경로는 중단한다.

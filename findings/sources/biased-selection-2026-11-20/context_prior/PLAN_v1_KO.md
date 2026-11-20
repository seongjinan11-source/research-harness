# CPC 사전 명세 v1: 편향 context에서 community별 클래스 사전확률의 출력 쪽 보정

- 작성일: 2026-11-20. **사전 명세 v1이다. 아직 구현하거나 실행하지 않았다.** 독립 리뷰 3개를 받은 뒤 v2로 고친다.
- 선행 근거(바꾸지 않음):
  - `../2026-11-20_biased_selection/DECISION_PHASE0_KO.md`
  - `../2026-11-20_em_transport/DECISION_S0_KO.md`
  - `../2026-11-20_em_transport/DECISION_S1_KO.md`
- W와 gts_pfn은 읽기만 한다. W의 target 그래프 8개는 사용자 승인 없이 쓰지 않는다.
- 계산 경로: frozen 공식 GraphPFN, Phase 0 dev context(seed 300–309, ρ 10/20, uniform과 commconc), W 프로토콜, pairwise AUC.

## 1. 출발점

**확정 (S1, dev 그래프 4개, 참 pool 라벨 oracle)**
- 특징 수송 oracle의 회복률은 0.43–0.76이다.
- 클래스 무관 양쪽 중심화(O-both)는 CiteSeer와 PubMed에서 해롭다.
- 출력 쪽 보정 O-prior는 logp + log p_pool(c|k) − log π_L 이다. 회복률은 0.95–1.56이다.
  - 다만 uniform에서도 +0.4 ~ +4.6pt를 올린다.
  - 사후 계산한 (commconc 이득 − uniform 이득)/손실은 0.82–0.92다. 사전에 정의한 지표가 아니다.

**음성 근거**
- W R2(`W/novel/driver.py:72-91`)는 PFN posterior에서 community별 Saerens EM을 돈다. target commconc에서 −0.32 bAcc, REF 대비 −1.34였다. 모델은 HI이고, GraphPFN이 아니다.
- S0: commconc의 PFN posterior는 U 노드에서 거의 우연 수준이다. 클래스 질량은 context community 쪽으로 쏠린다(약 3.5배).
- Phase 0: 라벨 전파(LP)도 commconc에서 크게 잃는다(PubMed 21.4pt, CiteSeer 9.9pt). 따라서 그래프 전파로 만든 p̂(c|k)도 같은 선택 편향을 물려받을 수 있다. **이것이 이 계획의 가장 큰 실패 위험이다.**

**아직 모르는 것 (이번 cycle의 질문)**
- Q1. O-prior 이득 가운데 선택 편향에 특유한 부분은 어느 community에서 나오는가? context가 있는 community인가, 없는 community인가?
- Q2. 라벨 없는(context만 쓰는) 추정기 가운데, plug-in했을 때 회복률이 0.15를 넘는 것이 있는가?
- Q3. 진짜 라벨 m개를 쓸 수 있을 때, 그 라벨을 prior 추정에 쓰는 것이 context에 추가하는 것보다 나은가?
  - 이 질문은 W의 "추가 진짜 라벨" 결과와 이어진다.
  - 더 나쁘다면 prior 경로는 실용 가치가 없다.

**선행 연구 (기여로 주장하지 않는 것)**
- label-shift EM: Saerens et al. 2002. 집단별 확장도 여기에 포함된다.
- 그래프 출력 후처리: Correct & Smooth(Huang et al. 2021), LP, APPNP.
- logit adjustment: Menon et al. 2021.
- 방어할 수 있는 기여 후보는 세 가지다.
  1. frozen 그래프 ICL에서 community 편향 context가 만드는 손실을 출력 쪽 사전확률 축과 특징 축으로 분해한 것.
  2. 어떤 정보원으로 그 사전확률이 식별되는지에 대한 근거.
  3. 라벨 예산을 어디에 쓸지에 대한 근거.
- 메소드 자체가 C&S나 Saerens의 변형이라면 그렇다고 쓴다.

## 2. 기호와 공통 보정식

- logp: 원래 context의 GraphPFN log posterior. 한 단위당 forward 1회로 구해 캐시한다.
- k(i): Louvain community(W 규칙, seed 2027). π_L: context 클래스 빈도.
- 보정식: s_i = logp_i + β · [log p̂(c|k(i)) − log r(c|k(i))].
  - 기본 분모 r = π_L (O-prior와 같다).
  - 대안 분모 r = m̄_k: community k 노드에서 PFN posterior의 평균, 라벨 없음. PFN이 이미 반영한 community 사전확률을 걷어낸다는 뜻이다.
- p̂의 Laplace/축소: p̂ ← (n_k p̂ + λ π_L)/(n_k + λ). n_k는 community 크기, λ ∈ {1, 10}.
- 주 설정: β = 1, r = π_L, λ = 1. 이 설정이 O-prior와 같은 형태다.

## 3. arm

**oracle (pool 라벨 사용, Q1 분해용)**
- O-prior: S1과 같다.
- O-prior-ctxK: context가 있는 community에만 참 p(c|k)를 쓰고, 나머지는 π_L을 쓴다.
- O-prior-noctxK: 그 반대다.
- O-prior-mbar: 분모로 m̄_k를 쓴다.

**라벨 없는 추정기 (Q2; 모두 context 라벨과 그래프만 쓴다)**
- E-LP: context 라벨에서 시작하는 PPR(α = 0.1, 10회 반복). p̂(c|k) = community k 안 노드들의 평균.
- E-PFN-R2: W R2 `cpc`를 그대로 이식한다(ref = local, τ = 5, m = W 기본값, iters = 20, β = W 기본값). 동일성은 W의 dev HI 예측으로 단위 테스트한다.
- E-PoE: 노드별로 정규화한 LP × PFN posterior를 community 안에서 평균한다.
- E-moment: community 특징 평균 μ_k를 context 클래스 원형 μ̂_c(PCA64 특징)의 볼록결합으로 푼다(simplex 최소제곱). 특징 offset이 있으면 편향된다. 진단으로 오차를 함께 기록한다.
- E-flat: p̂ = π_L, r = m̄_k. community별 PFN 사전확률을 전역으로 평탄화한다. 라벨 없음.

**소수 라벨 (Q3; 같은 라벨 예산을 두 방식에 쓴다)**
- 예산: community마다 pool에서 무작위로 m개(m ∈ {1, 3})를 뽑는다. 크기가 m 미만인 community는 전부 쓴다. 뽑는 seed는 단위 seed에 고정한다.
- L-prior: 그 라벨로 p̂(c|k)를 Laplace 추정해 보정식에 넣는다. context는 그대로 둔다.
- L-ctx: 같은 라벨을 context에 더해 GraphPFN forward를 다시 한다.
- L-both: 두 가지를 모두 한다.

**기준선 (모두 같은 logp에서 출발하는 후처리)**
- C&S: context 라벨로 Correct 단계와 Smooth 단계를 돈다(α 0.8/0.8, 50회, 원 논문 기본값).
- LP-PoE-node: s = logp + β · log LP_i 를 community 집계 없이 노드 단위로 적용한다.
- REF-GP: 이전 cycle의 이식본(k = 5, ALPHA = 0).
- logit adjustment: s = logp − log π_L.

## 4. 단계

**E0 캐시 (forward)**
- 단위 160개(그래프 4 × seed 10 × ρ 2 × 정책 2)마다 logp를 저장한다.
- 같은 단위에서 L-ctx와 L-both용 forward를 m마다 1회 한다(총 3회/단위).
- 예상: S1이 단위당 6 forward에 약 1시간이 걸렸으므로 약 30분.

**E1 dev 오프라인 분석 (forward 없음)**
- 평가 노드: PubMed/photo는 val, Cora/CiteSeer는 eval 절반 A(perm seed 2026112099). S1과 같다. 절반 B는 쓰지 않는다.
- 각 추정기마다 다음을 기록한다.
  - p̂(c|k)의 오차: community 크기 가중 TV(참 pool p(c|k) 대비).
  - 회복률, 선택 특유 회복률, uniform 변화.
- **선택 특유 회복률을 사전 정의한다:** SRec = (gain_commconc − gain_uniform) / loss. 같은 seed 군집 bootstrap으로 구간을 낸다.
- grid: β ∈ {0.5, 1}, λ ∈ {1, 10}, r ∈ {π_L, m̄_k}. 기준선도 같은 크기의 grid를 받는다(C&S α ∈ {0.5, 0.8} × 반복 {10, 50}; LP-PoE β ∈ {0.5, 1}).
- **E1 gate (사전 지정):** 라벨 없는 추정기 가운데 주 설정에서 다음 둘을 모두 만족하는 것이 하나라도 있어야 S3로 간다.
  - 평가 가능 그래프 3개 중 2개 이상에서 Rec의 CI 하한 > 0.15.
  - 그 그래프들에서 uniform 변화의 CI 하한 > −0.5pt.
  - 없으면 라벨 없는 경로는 음성으로 기록하고, Q3(소수 라벨)만 S3에서 확인한다.
- dev 선택 규칙: gate를 통과한 추정기 중 평가 가능 그래프 평균 SRec가 가장 큰 설정을 고른다. 주 설정 결과도 함께 보고한다.

**S3 확인 (dev와 독립)**
- 이전 cycle PLAN v2 §5 S3를 그대로 쓴다.
  - mkpol seed 310–319를 새로 쓰고, counts 규칙도 같다.
  - PubMed/photo는 test mask로 평가하고, pool = ¬(val ∪ test)로 둔다. Cora/CiteSeer는 eval 절반 B로 평가한다.
- 판정은 주 설정과 dev 선택 설정 각각으로 보고한다. **판정은 dev에서 미리 고른 한 설정으로만 한다(E1 종료 시 동결).**

## 5. 사전 판정 (S3, 공용 투표 함수, 평가 가능 그래프만)

| 판정 | 성립 | 기각 |
|---|---|---|
| P1 회복 | Rec CI 하한 > 0.15 이고 uniform 변화 CI 하한 > −0.5pt | Rec CI 상한 < 0.15 |
| P2 C&S·LP-PoE·REF-GP·logit-adj 대비 | commconc 이득 차이 CI 하한 > 0 | CI 상한 < 0 |
| P3 community 집계의 필요 | 메소드 − LP-PoE-node CI 하한 > 0 | CI 상한 < 0 |
| Q3 라벨 예산 (m = 3) | L-prior − L-ctx CI 하한 > 0 | CI 상한 < 0 |

- 그래프별 판정을 공용 투표 함수(`../2026-11-20_em_transport/stats.py`)로 모은다.
- 평가 가능 그래프가 3개면 사실상 만장일치를 요구한다. 따라서 "결론 없음"이 가장 가능성 높은 결과다.
- 사전 예측:
  - 라벨 없는 gate 통과 가능성은 **낮다.** LP와 PFN posterior 모두 같은 선택 편향을 받는다.
  - Q3는 결과를 예측하지 않는다.

## 6. 통계

- 이전 cycle과 같다: seed 군집 percentile bootstrap(10k). ρ = 10/20을 한 군집에 넣는다. 평가 가능성은 Δ_GP CI 하한 > 2pt로 판단한다.
- Cora는 dev에서 평가 불가였다. 투표에 넣지 않고 보고만 한다.

## 7. 구현과 동결

- 새 파일: `cache.py`(E0), `cpc.py`(추정기와 보정식), `baselines.py`(C&S, LP-PoE, logit-adj), `e1_analyze.py`, `test_cpc.py`.
- 단위 테스트:
  - `cpc.correct`에 참 p를 넣으면 S1의 `o_prior`와 같은 값이 나온다.
  - R2 이식본은 W `cpc`와 같은 값을 낸다.
  - C&S는 장난감 그래프에서 손계산과 같다.
- 실행 전 FREEZE_E0/E1/S3 해시를 남긴다.

## 8. 리뷰 기록

(리뷰 뒤 작성)

## 9. 변경 기록

- 2026-11-20 v1 작성.

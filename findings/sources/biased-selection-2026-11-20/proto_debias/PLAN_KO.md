# PDB 사전 명세 v2: community 혼합비 준지도 EM으로 원형 편향을 줄인다 (라벨 없음)

- 작성일: 2026-11-20. **사전 명세 v2이고, 아직 구현하거나 실행하지 않았다.** v1은 [PLAN_v1_KO.md](PLAN_v1_KO.md)에 보존했다. 독립 리뷰 3개(§8)를 반영했다.
- 선행 근거는 바꾸지 않는다: `../2026-11-20_context_prior/DECISION_E1_PARTIAL_KO.md`, S1, Phase 0.
- W와 gts_pfn은 읽기만 한다. W의 target 그래프는 사용자 승인 없이 쓰지 않는다.

## 1. 출발점

**참 원형 상한 (oracle 진단, dev, CPC 캐시, pool 라벨 사용)**

| 그래프 | AUC SG | SRec |
|---|---|---|
| CiteSeer | +4.8 [+3.7, +6.0] | 0.92 |
| Cora | +2.3 | 0.95 |

- context 원형(M0)은 SRec 0.21이다.
- 계산 스크립트는 scratch에서 돌렸다. v2 구현 때 `oracle_ceiling.py`로 cycle 안에 다시 만들고 해시를 남긴다(리뷰 A 3).

**리뷰 C proxy (Cora만, seed 3개, ρ 10; 가설 수준, 판정 아님)**
- v1 변형 셋은 모두 M0보다 나빴다. commconc gap 중앙값은 M1 −0.57, M2 −0.68, M3 −1.05였다.
- M1(이원 고정효과)
  - pool 라벨 전체로 풀면 TV가 oracle에 가깝다(0.23–0.27).
  - 그러나 context에는 셀이 19–37개뿐이다. community 안 클래스 대비가 거의 없다.
  - 더 근본적으로, context가 없는 community는 offset â_k를 추정할 수 없다. 그래서 M1은 그 community에서 원리상 이득이 없다(리뷰 C 7).
  - 또 oracle 원형은 클래스별 offset 평균을 포함하는 pool 클래스 평균인데, FE는 그 평균을 지운다. 둘은 서로 다른 양이다(리뷰 B 1).
- M3(community 평행이동)
  - 동질성이 높아 community가 거의 클래스와 겹친다(Cora 가중 순도 0.77).
  - 그래서 a_k + P_c가 교락되고, M3는 클래스 구조를 a_k로 흡수했다.
  - Cora 분산 분해: 클래스 R² 0.064, 가산 FE 0.135, 셀 0.176.
- M2(community 혼합비 GMM)
  - 참 원형에서 시작하면 TV가 oracle 수준(0.18–0.25)이다. 즉 **모형은 좋은 해를 갖고 있다.**
  - 실패는 탐색 실패다: M1 초기화의 basin이 나쁘고, 클래스가 붕괴하고, π_L 쪽 수축이 불균형을 다시 주입한다.
  - 64차원 diag GMM에서 책임도는 사실상 hard(최대 0.91–0.94)다.

**선행 (기여로 주장하지 않음)**
- 집단별 혼합비 혼합모형과 Latent Label Shift: Roberts et al., NeurIPS 2022, 2207.13179
- transductive/준지도 ProtoNet: Ren et al. 2018
- BD-CSPN
- label-shift EM (Saerens 2002), TTLSA, BBSE
- 이원 고정효과와 limited mobility bias: AKM 1999, Andrews et al. 2008
- batch correction: Harmony 2019, MoDaH 2512.09259. 교란 시 과보정이 일어난다는 문헌도 있다.
- AdaBN
- **M2 자체는 LLS/준지도 GMM의 적용이다.** 새로 보태는 것은 세 가지뿐이다.
  - (i) 어떤 초기화와 해 선택 규칙이 community 집중 context에서 원형 편향을 피하는가에 대한 경험적 근거
  - (ii) 그 결과를 frozen PFN 출력 prior 보정에 연결하는 것
  - (iii) 실패할 때 순도·연결성과 짝지은 식별 경계

## 2. 메소드 M2* (단일 주 후보)

- **특징:** 전 노드로 적합한 F(PCA64)의 상위 16개 주성분을 쓴다. 라벨은 쓰지 않는다.
- **모형:** F_i ~ Σ_c w_{k(i),c} N(P_c, diag σ²).
  - σ²에는 바닥을 둔다: 전역 분산 × 10⁻³.
  - context 행은 one-hot으로 고정한다. EM은 전 노드에서 돈다(비context 노드의 라벨은 읽지 않는다).
- **w 수축:** w_k ← (Σ_{i∈k} r_i + κ π̂) / (n_k + κ).
  - π̂는 매 반복 비context 노드의 책임도 평균(전역 혼합비, Saerens식)이다. **π_L은 쓰지 않는다**(리뷰 C 5).
  - κ ∈ {2, 10} 의사개수. 이것이 grid의 전부다.
- **다중 초기화:** 원형 초기값 4개.
  - (i) context 클래스 평균(M0)
  - (ii) 이원 FE 원형(M1)
  - (iii) PFN posterior 가중 전 노드 평균
  - (iv) 수축 원형 (n_c μ̂_c + 10 μ_all)/(n_c + 10)
- **수렴:** float64, 최대 200회, 상대 Δloglik < 10⁻⁶.
- **붕괴 판정:** 어떤 클래스의 전체 질량이 그 클래스의 초기 질량의 0.2배 미만이면 "붕괴"로 보고 후보에서 뺀다.
- **해 선택 (라벨 없음, context만 사용):** context 5-fold.
  - 접힌 fold의 context 라벨을 숨기고 같은 EM을 돈다.
  - 숨긴 노드의 log p(y | x, k)를 평균해, 가장 높은 초기화를 고른다.
  - 모두 붕괴하면 M0로 후퇴한다.
  - 한계: 숨긴 노드도 context가 있는 community에 있다. 그래서 이 기준은 context가 없는 community의 적합을 직접 보지 않는다.
- **출력:** p̂(c|k) = w_k. 보정은 s = logp + log p̂(c|k)(β = 1, r = 없음)이다.
- **y 접근 경계:** 메소드 함수는 `ctx_idx`와 `y_ctx`만 받는다. y 전체는 진단과 oracle 모듈만 받고, 테스트로 확인한다.
- **재현성:** OMP_NUM_THREADS = 1. 같은 단위를 두 번 돌려 결과가 비트 단위로 같은지 테스트한다.
- **기록:** 단위마다 선택된 초기화, 반복 수, loglik, 붕괴 여부, ‖w − w_init‖, 셀 수, 성분 수, 단일 클래스 성분 수, community 순도.

**대조와 기준선**
- M0: 동결된 `E_moment|b1.0_none`. P3의 비교 기준이다.
- SH: 수축 원형 + E-moment.
- RW: (k, c) 셀마다 같은 가중을 준 context 원형 + E-moment.
- M2-plain: context 원형으로 단일 초기화, π_L 수축(리뷰 C proxy 판). 절제용이다.
- M2-oracle: 참 원형으로 초기화. 상한이다.
- 후처리 기준선은 결과를 보기 전에 원 논문 또는 주 설정 기본값으로 고정한다.
  - C&S: α = 0.8, 50회
  - LP-PoE: β = 1, α = 0.1
  - stacking: C = 1, LP 포함
  - E-R2: W 선택값
  - REF-GP: S3에서만, forward 필요

## 3. 단계와 gate

**전제 (리뷰 A 1):**
- 투표 함수는 평가 가능 그래프가 3개 미만이면 "결론 없음"이고, 3개면 만장일치를 요구한다.
- dev에서 Cora는 평가 불가다. 따라서 S3 성립에는 **CiteSeer, photo, PubMed가 모두 성립해야 한다.**
- 그래서 dev gate도 세 그래프 모두를 요구한다.
- dev에서 평가 가능 그래프가 3개 미만이면 S3를 돌리지 않고 "결론 없음"으로 기록한다.

**T (TV 진단; GraphPFN 불필요; pool 라벨로 진단만 하고 메소드는 쓰지 않음)**
- 대상: CiteSeer, photo, PubMed의 dev commconc와 uniform 단위.
  - Cora는 리뷰 C가 이미 봤으므로 보고만 한다.
  - photo/PubMed 특징 F: photo는 CPC 캐시에 있다. PubMed는 runner로 계산한다(1회).
- 지표: context가 없는 community의 pool 크기 가중 TV_unc. 단위별 차이 D = TV_M0 − TV_M2*에 seed 군집 CI를 붙인다.
- **gate T** — 세 조건을 모두 만족해야 한다:
  1. 세 그래프 모두 commconc D의 CI 하한 > 0.
  2. 세 그래프 중 2개 이상에서 M2*의 TV_unc가 SH와 RW 중 더 나은 쪽보다 낮다(점추정).
  3. 세 그래프 모두 uniform D의 CI 하한 > −0.02.
- 보조 보고: Σ분자/Σ분모 gap closure, TV_all, 선택된 초기화 분포, 붕괴율, 라벨 없는 선택 기준과 TV 순위의 일치도.
- **T 실패 시 중단하고 음성 결과로 기록한다.** 식별 진단(셀 수, 순도, 성분 수)과 짝지어 보고한다.

**A (dev AUC) — T를 통과한 경우에만**
- 새 `slim_cache.py`로 photo 잔여 26단위와 PubMed 40단위에 GP forward를 보충한다. commconc는 NULL forward도 한다.
  - 같은 runner와 context 경로를 쓴다.
  - 기존 photo 캐시 1단위의 GP와 비트 단위로 같은지 확인한다.
  - PubMed는 1단위를 먼저 실측한다.
- 새 `score2.py`를 쓴다.
  - interim의 Cora/CiteSeer GP와 `E_moment|b1.0_none` 행을 재현하는지 테스트한다.
  - `judge.arm_stats`/`evaluable`을 import해 재사용하고, FREEZE_E1 해시를 assert로 확인한다.
- **photo의 AUC는 A 코드를 동결하기 전까지 채점하지 않는다.**
- **gate A:** 평가 가능 그래프가 3개 이상이어야 하고, 평가 가능한 모든 그래프에서 아래를 만족해야 한다.
  - (a) SG CI 하한 > 0이고 SRec ≥ 0.15
  - (b) uniform 변화 CI 하한 > −0.5pt
  - (c) null SG CI 하한 ≤ 0
  - (d) M2* − M0의 SG 차이 CI 하한 > 0
- **선택:** κ 2개 중 (a)(b)(d)를 만족하는 것 가운데 평균 SG가 최대인 것을 고르고, 그다음 (c)를 확인한다. 두 설정 결과를 모두 보고한다.

**S3 (확인) — A를 통과한 경우에만**
- CPC `s3_ctx`를 해시로 확인한 뒤 그대로 쓴다.
- forward: 단위마다 GP와 REF.
- 판정은 선택된 단일 설정으로, **gatekeeping 순서 P1 → P3 → P2**로 한다.
  - P1: SG CI 하한 > 0이고 uniform 변화 CI 하한 > −0.5
  - P3: M2* − M0 > 0
  - P2: 고정 기준선 대비 > 0
- 주 지표는 AUC, 보조는 bAcc다. 평가 가능성은 AUC 손실로 정하고 S3 context로 다시 계산한다. null은 S3에서 쓰지 않는다.
- S3 노드 상태:
  - Cora/CiteSeer 절반 B는 깨끗하다.
  - PubMed/photo의 test는 S1과 Phase 0 oracle에서 이미 집계 수준으로 쓰였다. 한계로 명시한다.

## 4. 통계

- seed 군집 percentile bootstrap(10k, seed 10개)을 쓴다. **BCa는 쓰지 않는다**(코드에 없음; 이전 PLAN의 BCa 문구를 정정한다).
- SG는 같은 (seed, ρ)의 uniform 짝으로 계산한다. 짝이 존재하는지 assert로 확인한다.

## 5. 비용 (실측 기반: forward당 CiteSeer 약 4초, photo 약 11초, PubMed 추정 16초)

| 단계 | 내용 | 시간 |
|---|---|---|
| T | EM: 초기화 4개 × (전체 1회 + 5 fold) × 단위 120 | 약 10–20분 (실측 후 기록) |
| A 보충 | 모델 적재 포함 | 약 25–30분 |
| S3 | | 약 45–50분 |

- 예산의 2배를 넘으면 중단하고 보고한다.

## 6. 사전 예측

- **gate T 통과 가능성: 낮음.**
  - 근거: 리뷰 C의 Cora proxy, 리뷰 C(CPC) proxy에서 PubMed/photo GMM이 평탄 prior와 차이가 작았던 점, 해 선택 기준이 context 없는 community를 보지 못하는 한계.
- 통과하더라도 세 그래프 만장일치가 필요하므로 S3 성립 가능성은 더 낮다.

## 7. 실패 시 보고 형식 (사전 지정)

- 그래프별 표:
  - context 없는 community의 질량
  - 셀 수와 연결 성분
  - community 순도
  - 원형 오차(M0, M2*, oracle)
  - TV
  - 선택된 초기화
  - 붕괴율
- 이 표를 "라벨 없는 원형 회복의 식별 경계" 근거로 쓴다. O-swap과 참 원형 상한(진단 문제)과 짝짓는다.

## 8. 리뷰 기록 (v1 → v2)

- **리뷰 A (프로토콜):**
  - S3는 세 그래프 만장일치가 필요하다. → §3 전제와 중단 규칙
  - 동결 코드를 그대로 재사용할 수 없다. → slim_cache와 score2, 재현 테스트, 새 FREEZE
  - T는 pool 라벨에 기대고 Cora/CiteSeer는 가설을 만든 그래프다. → oracle 스크립트를 보존하고, Cora는 T에서 빼며, photo는 A 동결 전 채점 금지
  - 단위별 비율이 불안정하다. → 차이 CI
  - grid와 (d)가 모호하다. → β·r 고정, κ만 grid, gatekeeping
  - M1 이동 규칙이 정의되지 않았다. → M1을 초기화 후보로만 둠
  - y 접근 경계를 코드로 강제한다.
  - EM 재현성 명세
  - 비용 단가 정정, BCa 정정, uniform 짝 assert
- **리뷰 B (선행):**
  - M1의 추정량이 oracle과 다르다. → §1, M1을 주 후보에서 뺌
  - M3는 순수 community에서 식별되지 않는다. → M3 제외
  - AKM/limited mobility bias, Harmony/MoDaH 과보정, LLS·준지도 ProtoNet 선행을 넣었다.
  - 기여를 (i)(ii)(iii)으로 한정했다.
  - 단순 기준선 SH/RW를 넣었다. community 중심화(CC)는 비context μ_k를 중심화하면 E-moment가 퇴화하므로 넣지 않았다.
  - 저차원 특징 → 16 PC.
- **리뷰 C (메커니즘, Cora와 합성):**
  - Cora에서 v1 셋 모두 M0보다 나쁘다. → 사전 예측을 "낮음"으로
  - M1은 표본 부족이고, context 없는 community에서 원리상 이득이 없다.
  - M2는 초기값 basin과 클래스 붕괴 문제다. → 다중 초기화, 라벨 없는 해 선택, 붕괴 판정
  - π_L 수축이 불균형을 다시 주입한다. → π̂
  - M3는 클래스를 흡수한다. → 제외
  - hard 책임도 → 16 PC
  - gap 비율이 불안정하다. → 차이 CI, uniform 비악화 조건

## 9. 변경 기록

- 2026-11-20 v1 작성.
- 2026-11-20 v2: §8 반영(구현 전, 실데이터 GraphPFN 결과 없음; 리뷰 C proxy는 Cora와 합성만).

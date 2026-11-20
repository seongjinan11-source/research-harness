# PDB 사전 명세 v1: community 집중 context에서 클래스 원형의 편향 제거 (라벨 없음)

- 작성일: 2026-11-20. **사전 명세 v1이다. 아직 구현하거나 실행하지 않았다.** 독립 리뷰 3개를 받은 뒤 v2로 고친다.
- 선행 근거(바꾸지 않음):
  - `../2026-11-20_context_prior/DECISION_E1_PARTIAL_KO.md` 와 그 PLAN_KO.md
  - `../2026-11-20_em_transport/DECISION_S1_KO.md`
  - Phase 0
- W와 gts_pfn은 읽기만 한다. W의 target 그래프는 사용자 승인 없이 쓰지 않는다.

## 1. 출발점

**CPC 부분 E1 (CiteSeer는 평가 가능, Cora는 평가 불가; 기술 통계)**
- E-moment는 community 특징 평균 μ_k를 클래스 원형 P_c의 볼록결합으로 풀어 p̂(c|k)를 얻는다. 그 p̂로 PFN 출력을 보정한다.
- context 원형으로 하면 선택 특유 회복률(SRec)이 0.21이고, 유의하지 않다.

**새 확인 (2026-11-20, 같은 캐시, oracle 진단)**
- 원형만 참 pool 클래스 평균으로 바꾸고 나머지(보정식, β = 1)는 같게 둔 경우:

| 그래프 | AUC SG | SRec | bAcc SG |
|---|---|---|---|
| CiteSeer | +4.8 [+3.7, +6.0] | 0.92 | +3.6 [+1.3, +5.9] |
| Cora | +2.3 [+1.6, +3.1] | 0.95 | +4.2 |

- O-prior(0.83)나 O-swap(0.77)과 같은 수준이다.
- 따라서 **이 경로의 병목은 원형 추정 하나다.**
  - commconc의 context 원형 상대 오차는 1.34, uniform은 1.03이다.
  - context 없는 community에서 TV: context 원형 0.38, 참 원형 0.27, 평탄 0.62.
- 그래프 2개 기준이다. photo와 PubMed는 아직 확인하지 않았다.

**원형이 틀리는 이유 (가설)**
- 클래스 c의 context는 소수 community S_c에 몰려 있다.
- 그래서 μ̂_c ≈ μ_c + (S_c의 community offset 평균)이 된다. offset이 원형을 오염시킨다.

## 2. 해법 후보 (모두 context 라벨, 그래프, 특징만 쓴다; pool 라벨은 쓰지 않는다)

공통 기호: F = GraphPFN 전처리 특징(PCA64), k(i) = Louvain community, L = context.

**M1 FE: 이원 고정효과 원형**
- context 노드에서 F_i = a_{k(i)} + b_{y_i} + e_i 를 최소제곱으로 푼다.
- 같은 community 안에서 클래스 간 차이를 보면 offset a_k가 상쇄된다. 원형은 P_c = b_c + s로 둔다.
- 식별: context 셀(클래스, community)로 만든 이분 그래프의 연결 성분마다 공통 이동 하나가 식별되지 않는다.
  - 성분 수는 S1 진단 기준 CiteSeer 1.4, Cora 2.1, PubMed 2.9, photo 3.2다.
  - 성분별 이동은 다음 기준으로 정한다: 성분 j의 클래스들이 그 성분 community들의 비context 노드 평균을 설명하도록, E-moment 잔차를 최소화하는 1차원 탐색을 한다.
  - 전체 공통 이동 s는 pool 전체 평균과 맞춘다.

**M2 FE-GMM: community별 혼합비를 둔 준지도 가우시안 혼합 EM**
- 모형: F_i ~ Σ_c w_{k(i),c} N(P_c, Σ). Σ는 대각 행렬이고 모든 클래스가 공유한다.
- context 행은 one-hot으로 고정한다. EM은 비context 전 노드에서 돈다(라벨은 읽지 않는다).
- 초기값은 M1의 원형이다.
- 출력은 p̂(c|k) = w_k다. 리뷰 C proxy의 GMM-EM(CiteSeer TV 0.27)에 FE 초기화를 더한 판이다.
- w에는 π_L 쪽 Dirichlet 축소를 준다(강도 κ).

**M3 Offset-GMM: community 평행이동까지 둔 판**
- 모형: F_i ~ Σ_c w_{k,c} N(a_k + P_c, Σ). Σ_k n_k a_k = 0으로 둔다.
- 1차 모멘트만으로는 a_k와 w_k를 함께 식별할 수 없다. 그러나 community 안의 분포 모양(혼합 구조)이 식별을 돕는다. **가설이다.**
- M-step:
  - a_k = community k에서 (F_i − Σ_c r_ic P_c)의 평균
  - P_c = 클래스 c 책임도(r_ic)로 가중한 (F_i − a_{k(i)})의 평균
  - w_k = community k의 r 평균에 Dirichlet 축소
- 실패 위험: S1에서 클래스 무관 offset을 PFN 입력에서 빼는 것(O-both)은 해로웠다. 여기서는 offset을 PFN에 넣지 않고 w 추정에만 쓴다. 그래도 offset이 실제로 클래스에 따라 다르면(S1 O-ctx 근거) 모형이 틀린다.

**대조와 기준선**
- M0: context 원형(현 E-moment). 비교 기준이다.
- M-oracle: 참 pool 원형. 상한이다.
- M-PL: PFN이 자신 있게 예측한 비context 노드로 만든 원형. 클래스마다 context 없는 community에서 신뢰도 상위 20%를 쓴다. S0의 posterior 쏠림 때문에 실패할 것으로 예측하며, 대조로 둔다.
- C&S, LP-PoE, stacking, E-R2: CPC E1과 같은 기준선이다.

## 3. 단계

**T (원형과 TV 진단; GraphPFN 불필요; 그래프 4개, dev context)**
- 단위마다 기록한다: 원형 상대 오차, TV(전체, context 있는 community, 없는 community), 성분 수.
- **gate T (사전 지정):**
  - 간극 해소율 = (TV_M0 − TV_M)/(TV_M0 − TV_oracle). context 없는 community, commconc 단위의 중앙값으로 본다.
  - 이 값이 0.5 이상인 그래프가 4개 중 3개 이상인 변형만 A로 보낸다.
  - 통과하는 변형이 없으면 중단하고 음성 결과로 기록한다.
- 이 단계는 dev pool 라벨로 진단한다. 메소드는 그 라벨을 쓰지 않는다.

**A (dev AUC)**
- 캐시 보충: photo 26단위와 PubMed 40단위에 GP와 NULL forward만 추가한다(CPC cache와 같은 코드 경로).
- 판정 규칙은 CPC FREEZE_E1의 gate를 그대로 쓴다.
  - (a) SG CI 하한 > 0이고 SRec ≥ 0.15
  - (b) uniform 변화 CI 하한 > −0.5pt
  - (c) null 대조
  - 평가 가능 그래프 2개 이상에서 만족해야 한다.
- 추가로 **(d) M − M0의 SG 차이 CI 하한 > 0** 을 같은 그래프들에서 요구한다.
- grid: 변형마다 설정 2개다(M2와 M3: κ ∈ {1, 10}; M1: 성분 이동 방식 2가지). dev 선택은 CPC 규칙과 같다.

**S3 (확인)**
- CPC PLAN v2 §5 S3 규칙을 그대로 쓴다: mkpol seed 310–319, PubMed/photo는 test(pool = ¬(val ∪ test)), Cora/CiteSeer는 절반 B.
- 캐시: GP, uniform, REF forward.
- 판정은 선택된 한 설정으로 한다:
  - P1: 선택 특유 회복
  - P2: 기준선 대비
  - P3: M0 대비
- AUC와 bAcc 둘 다 보고한다. 투표는 공용 함수로 한다.

## 4. 비용

- T: 수 분.
- A 캐시 보충: 약 25분.
- S3: 약 45분.
- 구현 시간은 별도다.

## 5. 사전 예측

- M1은 CiteSeer처럼 성분 수가 적은 그래프에서 TV를 줄이지만, PubMed와 photo(성분 약 3개)에서는 약할 것이다.
- M2는 CiteSeer와 Cora에서 통과 가능성이 중간이고, PubMed와 photo에서는 낮다(리뷰 C proxy).
- M3는 예측하지 않는다.
- **전체 gate 통과 가능성: 중간 이하.**

## 6. 기여의 위치 (가설)

- 문제: frozen 그래프 ICL에서 community 집중 context가 PFN의 community별 클래스 사전확률을 왜곡한다. O-swap과 참 원형 상한으로 근거를 댄다.
- 해결 원리: 그 사전확률은 비context 노드의 community 특징 분포에서 식별할 수 있다. 다만 원형이 context 선택 편향에서 자유로워야 한다. 이원 고정효과와 community 혼합비 EM이 그 편향을 제거하는 수단이다.
- 선행으로 명시할 것:
  - 고정효과 회귀(계량경제의 within 추정)
  - 집단별 혼합비 혼합모형 / label-shift EM (Saerens 2002, TTLSA)
  - BBSE
  - 원형 정렬 계열 few-shot (BD-CSPN 등)

## 7. 변경 기록

- 2026-11-20 v1 작성.

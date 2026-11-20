# CCT-EM 사전 명세 v2: 편향 context의 클래스 조건부 community 수송, 라벨 없는 추정

- 작성일: 2026-11-20. **사전 명세 v2이며 아직 구현·실행하지 않았다.** v1은 [PLAN_v1_KO.md](PLAN_v1_KO.md)에 보존했다. v2는 세 관점 독립 리뷰(§10)를 반영한다.
- 선행 근거: `../2026-11-20_biased_selection/DECISION_PHASE0_KO.md`. 그 결과와 규칙은 바꾸지 않는다.
- 기존 작업 W와 gts_pfn은 읽기만 한다. W의 target 8개 그래프는 사용자 승인 없이 쓰지 않는다.

## 1. 출발점

**확정 (Phase 0, 그래프 4개, W 프로토콜, AUC)**
- commconc 손실은 2.4–13.8점이다. 크기 보존 community 무작위화(null)에서는 거의 0이다.
- 손실은 GraphPFN 고유가 아니다(BS-GCN, CMD-GCN, LP도 비슷하거나 더 잃음).
- context-only oracle 수송 O-ctx는 평가 가능한 세 그래프에서 손실의 43–67%를 회복한다.
- 라벨 없는 comm-z, ctx-match, density-ratio 복제는 실패했다.
- 합성 POS(평행이동을 심음)에서도 O-ctx 회복은 0.55–0.75에 그쳤다.

**W의 인접 음성 결과**
- R2 community별 label-shift EM(출력 쪽 prior 보정, `W/novel/driver.py:72-91`): target commconc −0.32 bAcc, REF 대비 −1.34.
- REF = gated pseudo-label 확장(`W/pseudo2/pl2_infer.py`, HI, ALPHA=0): target commconc +1.02 bAcc(`W/novel/REPORT.md:86`). W의 이전 최고지만 이득이 작다.
- 원고 `main_v41_r88_final.tex` 1691행 부근: community별 feature offset을 넣은 selection-aware pretraining prior가 commconc를 개선하지 못했다(리뷰어 인용 −0.04 [−0.30, 0.23]). 수단은 다르지만(사전학습 대 test-time) PFN이 offset을 스스로 흡수하지 못한다는 같은 현상이다.

**선행 연구와 기여의 위치**
- posterior 가중 클래스 조건부 평균 정렬은 이미 있다.
  - 검색으로 확인: BD-CSPN(1911.10713), LaplacianShot(2006.15486), CFA(2206.13951), FR-GNN(2308.09259), GTrans(2210.03561), Matcha(2410.06976).
  - 원문 미확인: JDA, CAN, DSAN, SHOT, StruRW, Pair-Align.
- 따라서 **"클래스 조건부 평균 정렬" 자체를 새 원리로 주장하지 않는다.** 방어할 수 있는 기여 후보는 세 가지다.
  1. 학습 없는 ICL(frozen PFN) context에서 community 편향 선택이라는 문제에 적용한 것.
  2. community 순도와 연결 구조에 따른 식별 조건.
  3. oracle 대비 손실 분해(어느 성분이 회복 가능한가).
- v1의 H1(p(x|y,k)이지 p(y|k)가 아님)과 H3(moment로만 쓰면 확증 편향에 덜 취약)은 근거가 부족하다. 둘 다 **탐색적 가설**로 낮추고, 각각 검사 arm을 둔다(§4).

## 2. 어려움 (리뷰 C의 순환성 분석)

- 수송에 필요한 셀은 context가 있는 (k, c)뿐이다. commconc에서는 클래스당 1–2개다(PubMed 1.7). 이 셀의 μ_{k,c}는 라벨 있는 context만으로도 잘 추정된다.
- 진짜 어려운 항은 **전역 클래스 평균 μ_c**다. 편향 context로 만든 posterior는 클래스 c의 질량을 c의 context가 몰린 community로 끌어간다. 그러면 μ̂_c가 μ̂_{k_c,c} 쪽으로 당겨져 δ가 0으로 줄어든다. EM은 GP 근처의 자기일관적 고정점에 머물 수 있다.
- 대리 점검(리뷰 C, logistic 회귀, 그래프 없음, **가설 수준**): CCT-EM 회복률은 약 0.29로 O-ctx의 1/3이었다. 반복해도 개선되지 않았고, δ̂ 크기는 oracle의 약 60%였다.
- 식별성: 라벨 셀의 클래스–community 이분 그래프가 끊겨 있으면, 가산 모형에서 클래스 차이 μ_c − μ_c′는 라벨만으로 식별되지 않는다. 연결 성분 수를 진단으로 기록한다.

## 3. 메소드 후보

공통 기호
- F: GraphPFN 공식 전처리 feature.
- L: context. U: 라벨을 읽지 않는 비평가 노드(pool \ L).
- k(i): Louvain community(W 규칙). π_L: context 클래스 빈도.
- **M-step은 항상 수송 전 F로 계산한다(누적 표류 방지).**

**CCT-EM (주 설정)**
1. q⁰는 원래 context의 GraphPFN posterior에 logit adjustment q ∝ p / π_L^α (α = 1)를 적용한 것이다.
2. 가중치: L은 1[y_i = c], U는 q_ic.
3. 전역 평균 μ̂_c: 기본은 전 노드 가중 평균. 대안 G2는 context가 없는 community의 U만으로 추정한다(편향된 셀을 뺌).
4. δ_{k,c} = n_{k,c}/(n_{k,c} + τ) · (μ̂_c − μ̂_{k,c}). τ = 5는 약한 축소일 뿐 O-ctx의 min_n과 맞춘 값이 아니다(v1 서술 정정).
5. context i만 x_i ← F_i + δ_{k(i), y_i}. 수송된 context로 E-step을 돌린다. T = 3.
- 주 설정: soft, α = 1, τ = 5, T = 3, μ̂_c는 전 노드.

**CCT-both (S1 결과에 따라 조건부로 채택, §5)**
- 위 수송에 더해, 모든 노드에서 클래스 무관 community offset을 뺀다.
- 그 offset은 구성 보정으로 추정한다: δ_k = μ_k − Σ_c p̂(c|k) μ̂_c. p̂(c|k)는 community별 Saerens EM으로 구한다.

## 4. arm

**oracle (pool 라벨 사용, 상한과 분해용)**
- O-ctx: Phase 0와 같다.
- O-both: 모든 노드에서 pool 라벨 기반 δ_k를 빼고, context는 O-ctx처럼 옮긴다.
- O-ctx+nbr: context와 그 1-hop 이웃을 함께 옮긴다. 이웃 집계 때문에 회복이 잘리는지 본다.
- O-prior: 참 p(y|k)로 R2식 출력 보정을 한다. H1 검사용이며, 같은 그래프·같은 AUC에서 본다.
- O-q: CCT-EM의 q를 pool 참 라벨 one-hot으로 바꾼다(τ = 5 유지). 추정 오차와 축소 효과를 분리한다.

**라벨 없는 메소드**
- CCT-EM(주), CCT-1shot(T = 0, EM 반복의 기여 검사), CCT-G2, CCT-both(조건부).
- CCT-kNN, CCT-LP: posterior를 kNN/LP로 바꾼다. "PFN posterior를 추정기로 쓴다"는 주장을 검사한다.

**반드시 이겨야 할 기준선 (K2)**
- CCT-global: community를 무시하고 posterior 가중 전역 클래스 평균으로 context를 옮긴다(JDA/BD-CSPN식).
- comm-center: 클래스 무관, context만.
- comp-center: 구성 보정 community 중심화(Saerens p̂(c|k), one-shot, 양쪽).
- self-train: 확신 상위 k = 5개/클래스를 pseudo-label로 context에 추가.
- soft-PL: 같은 q로 U를 가중 context로 넣는다. H3 검사용으로, "moment로만"과 "라벨로"의 정보 집합을 맞춘다.

**기존 최고와 환산**
- REF-GP: W의 gate k = 5를 ALPHA = 0으로 이식한 것이 주 설정이고, α = 1 판은 보조로 보고한다. W 원본과의 선택 노드 일치는 HI 예측 파일이 있는 dev에서 확인한다. W의 gate 임계(상위 80%)를 W가 dev eval 라벨로 정했는지도 기록한다.
- +k 진짜 라벨: 미커버 community에 진짜 라벨을 클래스당 k ∈ {1, 2, 5}개 더한다. CCT 이득을 라벨 수로 환산하는 데 쓴다.

**대조와 일반성**
- CCT-perm: 크기 보존 community 순열에서 CCT를 돌린다.
- BS-GCN, BS-GCN+CCT: 일반성 검사(K5).

## 5. 단계

**S0 합성 gate (GraphPFN, 약 10–20분)**
- Phase 0의 POS 생성식(seed 5개)에 순도 조절 계열을 더한다(community 안 주 클래스 비율 0.25, 0.5, 0.75, 0.95).
- arm: GP, O-ctx, O-both, O-ctx+nbr, CCT-EM, CCT-1shot, CCT-global.
- **gate:** POS 기본형에서 CCT-EM 회복률의 seed 평균이 0.2 이하이면 실제 그래프 단계로 가지 않는다. 결과를 음성 결과로 기록하고 사용자에게 보고한다.
- 반복마다 다음을 기록한다: ‖δ̂‖/‖δ^O‖, μ̂_c 오차, k_c에 몰린 q 질량.
- **실패 정의(사전 지정):** ‖δ̂‖/‖δ^O‖가 반복에 따라 0.3 아래로 줄면 "고정점 붕괴"로 본다.

**S1 oracle 분해 (dev context, 기술 통계)**
- Phase 0의 dev context와 eval 노드에서 oracle 5개를 돌린다.
- 사전 규칙: O-both − O-ctx의 seed 군집 CI 하한이 2점을 넘는 평가 가능 그래프가 2개 이상이면, CCT-both를 S2의 주 후보에 추가한다. 넘지 않으면 context-only를 유지한다.

**S2 dev (Phase 0 context)**
- 평가 노드: PubMed/photo는 val, Cora/CiteSeer는 eval의 고정 절반 A(seed 2026112099로 미리 나눔).
- grid: τ ∈ {1, 5, 20} × α ∈ {0, 1} × μ̂_c ∈ {전체, G2}. T는 한 실행에서 1/3/5를 중첩 기록한다.
- 기준선도 같은 크기의 grid를 받는다: self-train/REF의 k ∈ {2, 5, 10}, gate 분위 ∈ {0.5, 0.8}.
- 선택 규칙: 평가 가능한 그래프만 쓴다. dev uniform 변화의 그래프별 평균 ≥ −0.5점인 설정 가운데 commconc **이득(gain) 평균**이 가장 큰 것을 고른다. 각 arm에서 하나씩 고른다.
- **K2/K3 판정은 사전 주 설정과 dev 선택 설정 각각으로 보고한다. 판정은 주 설정으로만 한다. 2차 설정은 판정을 바꿀 수 없다.**

**S3 확인 (dev와 독립인 context와 노드)**
- context: 모든 그래프에서 mkpol seed를 새로 쓴다(310–319).
  - Cora/CiteSeer: 기존 W식 counts 규칙.
  - PubMed/photo: counts는 DEVD `random_uniform` seed 200+j의 counts를 쓰고, mkpol seed는 310+j로 한다.
  - uniform 짝: 같은 counts로 새 generator seed(np.random.default_rng([310+j, r, 7, 31]))를 써서 클래스별 무작위 추출한다.
- 평가 노드:
  - PubMed/photo: test mask(1,000 / 440개). **pool = ¬(val ∪ test)** 로 두고, O-ctx를 포함한 모든 arm에 같은 pool을 쓴다.
  - Cora/CiteSeer: eval 절반 B.
- 겹침 진단: dev와 confirm commconc context의 Jaccard를 기록한다.
- **한계:** 같은 그래프, 같은 community다. Phase 0가 B 절반 라벨의 집계값을 이미 봤다. 이것은 context 일반화 확인이지 그래프 일반화가 아니다. 그래프 일반화는 사용자 승인 뒤 별도로 한다.

## 6. 지표와 통계

- 주 지표: commconc pairwise AUC. 보조: bAcc.
- 단위는 seed다. 같은 seed의 ρ = 10/20은 노드 순서를 공유하므로 **seed 군집 bootstrap**(seed를 복원 추출하고 두 ρ를 함께 넣음, 10k)을 쓴다. percentile과 BCa를 함께 보고한다.
- Rec = mean gain / mean Δ_GP (비 구간도 seed 군집). ρ_O = gain_CCT / gain_O-ctx.
- 평가 가능성 gate: Δ_GP 군집 CI 하한 > 2점. confirm context에서 다시 계산한다.
- **공용 투표 함수**를 둔다.
  - 평가 가능한 그래프만 넣고, 분모는 평가 가능한 그래프 수로 한다.
  - 평가 가능 그래프가 3개 미만이면 "결론 없음"이다.
  - 평가 불가 그래프를 주입하는 단위 테스트를 둔다(Phase 0 J1 버그 재발 방지).
  - 평가 가능 그래프가 3개뿐이면 사실상 만장일치를 요구하므로, "결론 없음"이 가장 가능성 높은 결과임을 사전에 인정한다.

## 7. 사전 판정 (S3에서, 주 설정)

| 판정 | 성립 | 기각 |
|---|---|---|
| **K1 라벨 없는 회복** | Rec CI 하한 > 0.15, uniform 변화 CI 하한 > −0.5점(그래프별 비열등) | Rec CI 상한 < 0.15 |
| **K2 단순 기준선 대비** | commconc에서 CCT − {CCT-global, comm-center, comp-center, self-train, soft-PL}의 짝 차이 CI 하한 > 0 (전부) | 하나라도 CI 상한 < 0 |
| **K3 REF 대비** | CCT − REF-GP CI 하한 > 0 | CI 상한 < 0 |
| K4 oracle 근접 (보고) | ρ_O와 구간, O-q 대비 분해 | — |
| K5 일반성 (보조) | BS-GCN+CCT의 BS-GCN 대비 Rec CI 하한 > 0.15 | CI 상한 < 0.15 |

- K1 기준을 0.2에서 0.15로 낮췄다. 이유는 S0 대리 점검에서 실질 상한이 손실의 약 0.15–0.3으로 보였기 때문이다. **이 변경은 실데이터 결과를 보기 전에 한 것이다.**

**탐색적 (판정 아님)**
- H1: O-prior 회복 대 O-ctx 회복.
- H3: U의 q에 정확도를 통제한 잡음을 넣고(정확도 0.5–0.95), CCT-EM과 soft-PL/REF-GP의 열화 곡선을 비교한다.
- 회복 대 community 순도, 연결 성분 수, posterior 정확도.
- 라벨 환산: CCT 이득이 진짜 라벨 몇 개에 해당하는가.

**해석 규칙**
- S0 gate 실패: 실데이터로 가지 않는다. "posterior 기반 클래스 조건부 수송은 commconc에서 고정점 붕괴"를 음성 결과로 남긴다.
- K1, K2, K3가 모두 성립하면 그래프 일반화 단계를 제안한다(사용자 승인 필요).
- K1 성립·K2 실패(CCT-global이나 comp-center와 같음): community 조건화는 기여가 아니라고 기록한다.
- K1 기각: H2(라벨 없이 추정 가능)가 이 그래프들에서 틀렸다고 기록한다.
- K5 성립: "모델 무관 수송, PFN posterior는 추정기 중 하나"로 서술한다. CCT-kNN/LP가 같으면 PFN posterior의 특별함을 주장하지 않는다.

**사전 예측 (결과를 본 뒤 고치지 않는다)**
- S0 gate: 통과 가능성 중간 이하.
- K1: 결론 없음 쪽. ρ_O는 0.2–0.35.
- K2: comm-center는 이길 것. CCT-global과 comp-center는 불확실.
- K3: 결론 없음. K5: 결론 없음.

## 8. 구현

- 폴더: 이 디렉터리. Phase 0의 `arms.py`, `w_proto.py`, `harness.py`, `baselines.py`는 import만 한다.
- 파일: `cct.py`(메소드, 기준선, oracle), `synth.py`(S0), `run.py`(S1–S3, 재개 가능한 jsonl), `stats.py`(seed 군집 bootstrap, 공용 투표), `analyze.py`.
- 단위 테스트
  - O-q를 "τ = 0 + hard min_n = 5 gate" 모드로 돌리면 `arms.o_ctx`와 1e-6 이내로 같아야 한다.
  - 투표 함수 테스트(평가 불가 그래프 주입).
  - S3 context가 dev와 같은 집합이 아님을 확인하는 테스트.
  - REF-GP 이식: W HI 예측을 입력으로 넣었을 때 선택 노드가 W와 같은지 비교한다.
- FREEZE.json: 단계별로 해시와 forward당 실측 시간을 남긴다(S0 전, S2 전, S3 전).

## 9. 비용 (리뷰 B 정정 반영)

- forward 실측은 S0에서 한다. 추정은 PubMed 약 10초, photo 약 5초, Cora/CiteSeer 약 2초.
- 공유: T 중첩, α별 q⁰ 공유, oracle q 재사용.
- S0 약 0.3시간, S1 약 1.5시간, S2 약 6–8시간(grid는 CCT 12설정, 기준선 6설정), S3 약 3–4시간.
- **합계 약 11–14시간.** S0이 실패하면 0.3시간에서 멈춘다.

## 10. 리뷰 기록 (v1 → v2)

- **리뷰 A (선행·중복)**
  - W와 직접 중복은 없다. 원리 자체는 JDA/BD-CSPN식 클래스 조건부 정렬이다.
  - H1과 H3는 과장이다. 원고 1691행의 음성 결과가 빠졌다. REF의 α가 다르다.
  - 반영: §1의 선행·기여 재서술, CCT-global/soft-PL/O-prior/O-q/CCT-perm/+k 라벨 arm 추가, REF α = 0, H1/H3 탐색적 강등.
- **리뷰 B (통계·누출)**
  - 확인 context가 dev와 겹친다(PubMed Jaccard 0.60, 40개 중 11개가 같은 집합). 평가 노드를 재사용한다. 단위 테스트가 무조건 실패한다. 튜닝이 비대칭이다. ρ 군집 문제가 있다. uniform 기준이 약하다. 비용을 과소추정했다.
  - 반영: S3의 새 mkpol seed와 분리된 노드, test를 pool에서 제외, min_n 모드 테스트, 기준선 grid와 주 설정 판정, seed 군집 bootstrap, 비열등 CI, 공용 투표 함수, 비용 재추정, LA 사전확률 문제는 α grid와 comp-center의 Saerens로 부분 대응.
- **리뷰 C (메커니즘·대안)**
  - 순환성 때문에 δ가 0으로 줄어드는 고정점 붕괴가 우려된다. 대리 점검 회복률은 약 0.29였다.
  - context만 옮기는 선택이 상한을 깎는다. 이웃 집계 가설을 제기했다. 식별성 문제가 있다. 단순 대안이 있다.
  - O-ctx gain은 손실 크기와 r = 0.59–0.89로 상관하고, n_comm_per_class와는 상관이 거의 없다.
  - 반영: S0 gate와 붕괴 정의, O-both/O-ctx+nbr, CCT-both 조건부, G2 추정, 연결 성분 진단, CCT-1shot/kNN/LP, K1 0.15.

## 11. 변경 기록

- 2026-11-20 v1 작성.
- 2026-11-20 v2: §10 반영. K1 기준을 0.2에서 0.15로 낮춤(실데이터 결과 전, 대리 합성 근거).
- 2026-11-20 S0 실행과 판정 (결과를 본 뒤의 기록; 규칙 변경 없음): gate 실패(POS commconc CCT-EM T3 회복률 0.126 ≤ 0.2). 사전 규칙에 따라 S1–S3는 진행하지 않는다. 세부는 `DECISION_S0_KO.md`.
- 2026-11-20 S1 진행 (사용자 명시 지시, S0 gate 실패 뒤의 진단 전용; 메소드 판정 없음). S0에서 정의한 oracle 5개(O-ctx, O-q, O-both, O-ctx+nbr, O-prior)를 Phase 0 dev context에서 돈다. Cora/CiteSeer 평가는 eval 절반 A만 쓰고 B는 남겨 둔다. 동결: `FREEZE_S1.json`.

- 2026-11-20 S1 실행과 판정 (결과를 본 뒤의 기록; 규칙 변경 없음): O-both − O-ctx 하한 > 2pt인 평가 가능 그래프 0개라 O-both를 채택하지 않는다. 실데이터에서 특징 수송 oracle 상한은 0.43–0.76이다. O-prior는 0.95–1.56이나 uniform에서도 이득이 있어 과대평가다. 세부는 `DECISION_S1_KO.md`.

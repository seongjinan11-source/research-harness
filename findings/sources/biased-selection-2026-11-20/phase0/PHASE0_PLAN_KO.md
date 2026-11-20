# 0단계 사전 명세 v2: 편향 선택 손실 가운데 기존 작업이 다루지 않은 항목

- **작성일:** 2026-11-20. 이 문서는 사전 명세 v2이며, 아직 실행하지 않았다.
- **v1과의 관계:** v1은 [PHASE0_PLAN_v1_KO.md](PHASE0_PLAN_v1_KO.md)에 그대로 보존했다. v2는 다음 두 가지를 반영한다.
  - 사용자 지시: 기존 작업에 없는 항목만 다룬다.
  - v1에 대한 세 관점 리뷰(§10).
- **기존 작업 W:** `C:\Users\seongjin\.claude-science\orgs\75e1877e-...\workspaces\a763aaa2-...`. **읽기만 하고 쓰지 않는다.** 이 cycle의 모든 출력은 이 폴더에 둔다.
- **이 단계의 성격:** 메소드가 아니라 진단이다. oracle은 test 라벨이 아니라 **pool 라벨만** 쓴다. GraphPFN은 frozen 상태로 쓴다.

---

## 1. 기존 작업과의 경계

| 이미 있음 (W) | 이 cycle에서 다시 하지 않음 |
|---|---|
| class-wise offset 상한과 쌍별 AUC (`novel/ceiling.py`) | ✓ |
| 추가 라벨 획득 (`acq`: random/uncert/degree/coverage) | ✓ |
| edge 제거 (`graphoff`), permutation, community label-shift EM, gated pseudo-label, martingale, local experts, target fine-tuning, expw, DistPFN | ✓ |

**이 cycle의 새 항목 네 가지**
- **N0.** context 행만으로 하는 feature 표준화가 PFN 고유 손실을 만드는가. 리뷰 (b)가 새로 발견한 메커니즘이다.
- **N1.** 같은 편향 context에서 GNN 기준선은 얼마나 잃는가. 이것으로 PFN 고유인지 판정한다.
- **N2.** feature 공간의 community 평행이동을 context 쪽에서만 수송하면(pool 라벨 oracle) 회복되는가. 라벨 없는 대응물도 함께 본다.
- **N3.** 구조를 포함한 판별적 density-ratio로 재가중하면 회복되는가.

## 2. 데이터와 context

**그래프와 역할**
- 개발용으로만 쓴다. W의 **target 8개는 쓰지 않는다.** 원고 평가 bank이고, 이후 메소드 평가에 남겨 두기 위해서다.
  - W의 dev 그래프 2개: PubMed, photo. 경로는 `R\regime_robustness_20261017\dev_inputs`이고, W와 같이 val 행으로 평가한다.
  - 원고 prior의 source 그래프 2개: Cora, CiteSeer.

**context**
- PubMed와 photo는 W의 dev 규약을 그대로 따른다. 입력 파일의 `draws`(random_uniform, random_highdegree, r ∈ {10, 20})를 쓰고, `mkpol`과 같은 논리로 commconc, commdisp, lowdegree를 만든다. community는 W의 `pseudo/comm/{g}.pt`(Louvain, seed 2027)를 쓴다.
  - **이 dev 규약을 원 정의 그대로 재현했는지 확인한다.** 방법은 W에 저장된 HI 예측의 context 개수와 일치하는지 단위 테스트로 보는 것이다.
- Cora와 CiteSeer는 W 프로토콜을 이식해서 만든다.
  - 라벨과 무관한 60/40 pool/test 분할을 쓴다.
  - 예산은 40C이고, step 불균형 ρ ∈ {10, 20}, head는 무작위로 정한다.
  - 같은 `mkpol` 논리를 쓰고, Louvain seed는 2027이다.
- **seed 수:** dev 그래프는 W와 같이 10개다. Cora와 CiteSeer는 pilot(§7)으로 정한다.
- **정책:** uniform, commconc(주), highdegree, lowdegree, commdisp(보조).

## 3. 모델과 방법

| 이름 | 정의 |
|---|---|
| **GP** | 공식 GraphPFN을 frozen 상태로 쓴다(harness). 10 members, PCA64, raw 출력 |
| **GP-allnorm** (N0) | GP와 같고, LimiX `NormalizationEncoder`의 표준화만 **모든 행**에서 한다(런타임 속성 변경). transductive 설정에서 합법적인 변경이다 |
| **HI** | W에 저장된 frozen ImbNodePFN 후보(`hetero__imposed` 7301)의 k0 예측. **새 추론은 하지 않는다.** 새 처방을 HI에 적용하는 것은 N0 결과를 본 뒤 별도로 결정한다 |
| **BS-GCN** (N1 주 비교) | 원고의 BS 기준선과 같은 계열이다. GCN에 balanced softmax를 쓰고, 200 epoch 고정, validation 없음, init 3개의 확률을 평균한다. 입력은 PCA64 feature다 |
| **CMD-GCN** | sigmoid hidden 층을 둔 GCN에 BS loss와 λ·CMD(K=5, [0,1] 범위)를 쓴다. CMD는 context와 비context 노드 1,024개(seed마다 고정) 사이에서 계산한다. λ ∈ {0.1, 1}이고, 같은 구조에 λ=0을 둔 대조도 함께 돌린다 |
| **LP** | 클래스 질량 정규화. tie 비율을 보고한다 |

**N2 수송** (GP에 적용, feature tensor를 바꾼 뒤 forward)
- **O-ctx (pool 라벨 oracle, 주):**
  - 클래스 c의 context 노드 i만 x_i ← x_i + (μ^pool_c − μ^pool_{k(i),c})로 바꾼다.
  - μ^pool_c는 pool 노드(context 포함, test 제외) 중 클래스 c의 평균이다.
  - μ^pool_{k,c}는 그중 community k에 속한 노드의 평균이다.
  - test와 나머지 노드의 feature는 바꾸지 않는다. 따라서 test 라벨이 새어 들어가지 않는다.
- **comm-z (라벨 없음):** 모든 노드에 대해 community별 z-score를 계산한다.
- **ctx-match (라벨 없음):** context의 class 무관 평균과 분산을 전체 노드의 평균과 분산에 맞춘다(CORAL 대각 버전).
- O-ctx는 클래스와 community의 칸에 pool 노드가 5개 미만이면 그 칸을 바꾸지 않는다.

**N3 재가중** (GP)
- 판별 분류기로 context와 비context 노드를 구분한다. 입력은 [PCA64, ÂPCA64, Â²PCA64]이고, class_weight='balanced'인 로지스틱 회귀를 5-fold cross-fit으로 학습한다.
- 가중치 w_i ∝ p(U|x)/p(C|x)를 **클래스 안에서 평균 1로 정규화**하고 [0.2, 5]로 clip한다.
- 복제 수는 m_i = stochastic round(w_i / min w)로 정하고, 총합은 context의 5배를 넘지 않게 한다.
- 복제는 in-edge 전용으로 만든다. 노드별 k_i를 지원하도록 기존 `p2_extended`를 일반화한다.
- 가중치 분포, 즉 ESS와 w > 1인 비율을 보고한다.
- **이 arm에 대한 해석 제약:** 복제가 LimiX 표준화 통계도 바꾼다. 그래서 N3의 결과는 N0와 섞인다. N0 결과와 같이 읽는다.

## 4. 지표와 추정량

**지표**
- **주 지표:** macro 쌍별 AUC. 클래스 쌍 (a, b)마다 log p_b − log p_a로 계산한다. 새로 구현하고, 클래스별 offset에 불변인지 단위 테스트로 확인한다.
- **보조 지표:** raw bAcc.
- **진단 지표:**
  - LimiX 표준화 뒤 test 값이 ±100 clip에 걸린 비율.
  - context 표준편차와 전체 표준편차의 비.
  - SR. 원고 정의인 `calibrate_v3.gap`/`ratio`와, 편향을 보정한 d²_u 버전을 둘 다 계산한다.

**손실과 회복**
- 손실은 Δ_k(정책) = m_k(uniform) − m_k(정책)이다. 같은 (ρ, context seed)끼리 짝을 짓는다.
- **회복의 주 추정량은 선형이다.** 처방 T에 대해 다음 두 값을 따로 보고한다.
  - 정책 쪽 이득: m_T(정책) − m(정책)
  - uniform 쪽 변화: m_T(uniform) − m(uniform)
- **회복률:** Rec_abs = mean[m_T(정책) − m(정책)] / mean[Δ]. 평균끼리의 비이며, 짝 bootstrap(BCa, 10k)으로 구간을 낸다.
- **평가 가능성 gate:** Δ의 CI 하한이 2 AUC점 이하이면 그 그래프는 "평가 불가"다.
- **PFN 고유 격차:** G = Δ_GP − Δ_BS-GCN. 비교 대상은 BS-GCN 하나로 사전 지정한다. min을 취하지 않는다. 보조로 상대 손실 Δ/(m_uniform − chance)도 보고한다.

**불확실성**
- 그래프마다 (ρ, seed) 단위 짝 차이의 t 구간을 쓴다.
- 그래프를 합친 추정은 random-effects로 보조 보고한다.

## 5. 사전 판정

판정은 commconc에서 주 지표로 한다. 각 판정은 성립 / 기각 / 결론 없음 중 하나다. 평가 가능하고 성립 또는 기각인 그래프만 투표에 넣는다. 그래프 4개 중 3개 이상이면 전체 판정이다.

| 판정 | 성립 | 기각 |
|---|---|---|
| **J0 표준화 메커니즘** (N0) | GP-allnorm의 commconc 이득이 uniform 변화보다 크고, 그 짝 차이의 CI 하한이 1점 초과. 동시에 uniform 변화의 CI 하한이 −1점 초과 | 짝 차이의 CI 상한이 1점 미만 |
| **J1 PFN 고유** (N1) | G의 CI 하한 > 1점 | G의 CI 상한 < 1점(GCN이 PFN만큼 또는 그 이상 잃음) |
| **J2 수송 회복** (N2, O-ctx) | Rec_abs의 CI 하한 > 0.2이고, uniform 변화가 −1점보다 크다 | Rec_abs의 CI 상한 < 0.2 |
| J2' 라벨 없는 수송 | comm-z 또는 ctx-match에 대해 J2와 같은 기준 | 같음 |
| **J3 재가중** (N3) | Rec_abs의 CI 하한 > 0.2 | CI 상한 < 0.2 |

**해석 규칙**
- J1이 기각되면 결과를 "PFN 고유 아님"으로 기록한다. 이것만으로 "정보 부재"라고 하지 않는다. GNN은 같은 feature와 같은 context를 쓰기 때문이다.
- J0이 성립하면 N0 자체를 1순위 메소드 후보로 둔다. 비용이 0인 수정이기 때문이다. 이어서 다음을 확인한다.
  - HI에서도 같은 효과가 있는지. NodePFN/HI의 표준화 방식은 Phase 0에서 코드로 확인한다.
  - W의 target bank에서 효과를 확인할지. 이것은 사용자 승인 뒤 별도로 한다.
- J2가 성립하고 J2'가 기각되면, 수송의 원리는 맞지만 라벨 없는 추정이 병목이라고 해석한다. 이 경우 메타리뷰 계열 A의 EM과 수송 메소드 cycle로 간다.
- J2가 기각되면 평행이동 계열로는 회복할 수 없다고 기록한다. 이것은 D7의 범위를 좁히는 결과다.

**사전 예측** (결과를 본 뒤 고치지 않는다)
- J0 성립 가능성: 중간.
- J1: 결론 없음 또는 기각 쪽.
- J2: CiteSeer에서 성립 가능.
- J3: 기각.

## 6. 대조

- **null 대조:** 크기를 보존한 채 community 라벨을 무작위로 바꾸고 commconc를 다시 만든다. SR ≈ 1, Δ ≈ 0이어야 한다.
- **합성 대조:** CSBM에 community 평행이동 offset을 심은 그래프에서는 O-ctx Rec ≈ 1이어야 하고, offset이 없는 class-pure community 그래프에서는 Rec ≈ 0이어야 한다. **결과를 보기 전에 GP로 확인한다.**
- **ceiling 표시:** uniform AUC가 0.98을 넘는 칸은 표시한다.
- **CONC fallback 비율:** 기록한다. 원 정의는 community 순서대로 채우는 방식이라 fallback 개념이 없다. 대신 클래스별로 사용된 community 수를 기록한다.

## 7. 절차

1. **로더와 규약**
   - `w_proto.py`: W dev 입력 로딩, mkpol 이식, Cora/CiteSeer W 규약 이식.
   - 단위 테스트: W에 저장된 HI k0 json의 key 집합과 일치하는지, 클래스 개수가 맞는지.
2. **N0 probe (가장 싸고 결정적):** Cora와 PubMed에서 uniform과 commconc, seed 3개로 GP와 GP-allnorm을 비교한다. 진단 지표도 같이 본다. **개발용 근거이며 판정에는 쓰지 않는다.**
3. **나머지 구현**
   - BS-GCN, CMD-GCN, LP.
   - N2와 N3.
   - 쌍별 AUC 단위 테스트.
   - 합성 대조.
4. **pilot:** Cora, seed 5개로 SD를 잰다. Cora와 CiteSeer의 seed 수를 clip(⌈(2·SD/0.01)²⌉, 10, 30)으로 정한다.
5. **동결:** `FREEZE.json`에 sha256을 기록한 뒤 전체를 실행한다.
6. **결과 리뷰:** 세 관점으로 받는다.

**비용**
- GP forward는 Cora 약 4초, photo 약 10초, PubMed 약 30초로 예상한다. 실측으로 대체한다.
- 정책 5개 × ρ 2 × seed 10 × arm(GP, allnorm, O-ctx, comm-z, ctx-match, N3)이다.
- PubMed가 지배적이다. 수 시간 규모로 예상한다.

## 8. 한계

- 결과는 공식 GraphPFN에 대한 것이다. HI는 저장된 k0 예측과만 비교한다.
- dev와 source 그래프 4개만 쓴다. target bank는 이후 메소드 평가에 남긴다.
- O-ctx는 평행이동 가설 하나만 검사한다.
- BS-GCN은 hyperparameter를 탐색하지 않는다.

## 9. 리뷰

- v1에 대한 세 관점 리뷰는 §10에 반영했다.
- v2는 구현 뒤 코드와 수학 점검(관점 b)을 한 번 더 받는다. 동결하기 전에 한다.

## 10. 변경 기록

- **2026-11-20 v1:** 초안 작성.
- **2026-11-20:** 기존 작업과 겹치는 것을 발견해 실행 전에 멈췄다. v1 §9를 참고한다.
- **2026-11-20 v2:** 사용자 지시("새 항목만 진행")와 리뷰 반영.
  - **범위:** 획득(Q4)과 class-wise offset 상한은 W에 있으므로 뺐다. N0–N3만 남겼다.
  - **리뷰 b (P0.1):** LimiX가 context 행만으로 표준화한다(`normalize_on_train_only=True`, `encoders.py:48-64, 556-565`). 이 사실에서 N0을 새로 넣었다. 전역 affine 변환은 효과가 없다는 점도 확인했다.
  - **리뷰 a·b·c (oracle 누설):** test 라벨 oracle을 폐기했다. pool 라벨로 context 쪽만 수송하는 O-ctx로 바꿨다.
  - **리뷰 b·c (Rec):** 선형 이득을 주 추정량으로 바꿨다. 평균끼리의 비, 평가 가능성 gate, uniform 변화의 별도 보고를 넣었다.
  - **리뷰 c:** 세 값 판정과 동등성 구간을 넣었다. 분기의 이름을 바로잡았다("정보 부재"를 "PFN 고유 아님"으로). null 대조와 합성 대조를 넣었다. ceiling은 표시한다.
  - **리뷰 a:** 원고 선택 규칙과 Louvain(seed 2027)을 그대로 쓴다. SR은 원고 정의와 보정 버전을 함께 보고한다. CRUMB(NeurIPS 2026)를 선행으로 기록했다.
    - SR-GNN PPR-S sampler 재현은 이번 범위에서 뺐다. 편향 규칙은 원고 규칙을 쓴다. CMD-GCN 결과는 그 조건에서만 해석한다.
  - **리뷰 b (KMM):** KMM이라는 이름을 버리고 "판별적 density-ratio 재가중"으로 바꿨다. 클래스 안에서 정규화하고 [0.2, 5]로 clip한다. 구조를 포함한 feature를 쓰고 cross-fit한다.
  - **리뷰 b (CMD):** sigmoid hidden 층을 쓰고, λ=0인 같은 구조를 대조로 둔다.
  - **평가 bank 보호:** W target 8개를 쓰지 않는 것으로 바꿨다. dev 2개와 source 2개만 쓴다.
- **2026-11-20 N0 probe (개발용 근거, `results/n0_probe_*.json`)**
  - 조건: Cora, PubMed, photo × seed 3개 × uniform/commconc, ρ=10.
  - **commconc 손실은 크다.** 쌍별 AUC 기준이다.

    | 그래프 | uniform | commconc |
    |---|---|---|
    | Cora | 0.97–0.98 | 0.93–0.96 |
    | PubMed | 0.94 | 0.59–0.86 |
    | photo | 0.99 | 0.74–0.81 |

  - **context 행 표준화는 원인이 아니다.**
    - test 값이 clip에 걸린 비율: 0.
    - context 표준편차와 전체 표준편차의 비: 0.96–1.2. W 규약의 context는 클래스당 평균 40개이고, commconc도 community 순서로 채우기 때문이다.
    - 모든 행으로 표준화(GP-allnorm)했을 때의 AUC 변화: −1.6~+1.6점이고, 방향이 일정하지 않다.
  - 따라서 J0는 본 실행 전에 **사실상 기각 방향**이다. N0은 본 실행에서 보조 arm으로만 둔다.
- **2026-11-20 N1/N2 probe** (개발용 근거, `results/n12_probe_Cora_photo_PubMed.json`)
  - 설정: seed 3개, ρ=10, uniform/commconc, 쌍별 AUC 기준. 아래 값은 그래프별 평균이다.
  - **N1 (PFN 고유인가):** BS-GCN도 commconc에서 거의 같은 만큼 잃는다. G는 약 1–3점이다.

    | 그래프 | GP 손실 | BS-GCN 손실 | G |
    |---|---|---|---|
    | PubMed | 21.5 | 18.8 | 2.7 |
    | photo | 20.6 | 17.3 | 3.3 |
    | Cora | 3.1 | 2.2 | 0.9 |

    - LP는 photo에서 9.6만 잃는다.
  - **N2 (O-ctx, pool 라벨 oracle, context 쪽만 수송):** 손실이 크게 회복되고, uniform 쪽 손해는 없다(−0.4~+0.1점).

    | 그래프 | 손실 변화 | 회복률 |
    |---|---|---|
    | PubMed | 21.5 → 5.6 | 약 74% |
    | photo | 20.6 → 11.8 | 약 43% |
    | Cora | 3.1 → 1.5 | 약 50% |

  - **라벨 없는 대응물은 실패했다.**
    - comm-z는 오히려 해롭다. uniform까지 떨어뜨린다(PubMed −10점).
    - ctx-match는 효과가 거의 0이다.
  - **잠정 해석 (확정 아님)**
    - community별 클래스 조건부 offset을 알면 손실의 상당 부분을 되살릴 수 있다. 다만 O-ctx는 같은 community·클래스에 속한 pool 노드의 라벨을 쓰는 oracle이다. 정보 가치의 상한으로 읽어야 한다.
    - 병목은 **라벨 없이 클래스 조건부 offset을 추정하는 일**이다. 이 결과는 메타리뷰 계열 A(PFN posterior를 쓰는 EM 수송, FREE-1/PFN-1/PFN-2)의 동기를 준다.
  - **한계:** seed 3개이고, 동결 전 개발용 근거다. CiteSeer, N3, null/합성 대조는 아직 하지 않았다.
  - photo는 raw feature가 없어 W의 common100을 썼다.
- **2026-11-20 합성 대조** (`synth_controls.py`, `results/synth_controls.json`, 동결 전)
  - **POS** (평행이동 offset을 심은 그래프, community마다 클래스가 섞여 있음)
    - commconc 손실: AUC 28–41점.
    - O-ctx 이득: 17–21점. 회복률은 약 0.55–0.75다.
    - 사전 기대는 약 1이었는데 부분 회복에 그쳤다. 그래도 파이프라인이 심어 둔 평행이동을 검출한다는 것은 확인했다.
    - 부작용: uniform에서 O-ctx가 0–4점 손해를 냈다.
  - **NEG** (class-pure community, offset 없음)
    - AUC가 1.0으로 포화되어 정보가 없는 대조가 됐다. O-ctx가 이득을 만들어 내지 않는다는 것만 확인했다.
    - 더 어려운 음성 대조는 만들지 않았다.
- **2026-11-20 본 실행 설정 (동결 전 결정)**
  - 정책: uniform, commconc(주), commconc_null.
    - commconc_null은 community 라벨을 크기를 보존한 채 무작위로 바꾼 null 대조다. GP와 BS-GCN만 돌린다.
  - 범위: ρ ∈ {10, 20}, seed 10개(그래프마다 처음 10개). 그래프는 Cora, CiteSeer, photo, PubMed.
  - **계획 이탈:** §7-4의 pilot 기반 seed 수 결정은 비용 때문에 하지 않았다. 모든 그래프에 seed 10개를 고정했다.
  - **계획 이탈:** highdegree, lowdegree, commdisp 정책은 시간 때문에 본 실행에서 뺐다. 그래서 J3의 HDEG 보조 예측은 검사하지 않는다.
  - 판정 투표: 평가 가능하고 성립/기각인 그래프 가운데 max(3, ⌈0.75·그래프 수⌉)개 이상이면 전체 판정으로 본다. PubMed는 raw Planetoid feature를 썼다. 라벨과 노드 순서가 W와 같음을 확인했다.
- **2026-11-20 본 실행 완료와 판정** (`DECISION_PHASE0_KO.md`, `results/main/report.json`; 결과를 본 뒤의 기록이며 규칙은 바꾸지 않음)
  - 1,920줄 누락 없음. 동결 해시 7개 모두 일치.
  - Cora는 평가 불가(Δ_GP +2.4, CI 하한 1.7).
  - 판정: J0 기각, J1 결론 없음(계획 규칙 기준. 동결 코드가 J1에 gate를 안 걸어 report.json에는 rejected로 찍힘), J2 성립(Rec 0.43–0.67), J2' comm-z/ctx-match 기각, J3 기각.

# 메소드 제안 9건 메타리뷰: 편향된 라벨 선택 아래의 graph PFN

- 작성일: 2026-11-20.
- 배경 문서: [METHOD_BRIEF_KO.md](METHOD_BRIEF_KO.md).
- 제안 agent 9명은 모두 독립적으로 작업했고, 읽기만 했다(파일 수정 없음). 실험은 아무것도 하지 않았다.
- 아래 성공 확률은 각 agent의 자기 평가이고, 선행 연구 표기는 agent가 붙인 것이다.

## 1. 제안 요약

| agent | 제안 | 핵심 원리 | 자기 평가 성공률 | 가장 큰 위험 |
|---|---|---|---|---|
| RL-1 | RL로 test-time context 편집(pseudo-label 추가, 복제, 제거) | 비라벨 노드에 남은 증거를 정책이 골라 넣음 | 15–25% | pseudo-label의 천장이 진짜 라벨 10개(+3.8)보다 낮음. RL이 불필요할 수 있음 |
| RL-2 | 후회 기반 적대적 선택자 + continued pretraining | "GNN 기준선은 맞히는데 PFN은 놓치는" 몫만 학습(PAIRED/RTFM 이식) | ~20% | 실제 그래프에서 기준선도 무너지면 0%에 가까움 |
| RL-3 | 편향을 보정하는 라벨 획득 정책(RL) | 같은 k개로 더 많이 회복할 노드 선택 | 이득 자체 60–70%, RL 덕 20–25% | community 커버리지 휴리스틱이 이미 거의 최적일 가능성 |
| PFN-1 | CTRC: 라벨 context를 query community로 클래스 조건부 수송, community별 context | retrieval 실패를 우회하는 offset transport | ~30% | community centering 하나와 같은 결과 |
| PFN-2 | CTC: 같은 클래스 라벨을 클래스 조건부 offset만큼 이동해 빈 community로 복제 | 결정 보정은 랭킹 불변이므로 context support를 옮겨야 함 | 30–40% | pseudo-label self-training으로 환원 |
| PFN-3 | TSE-FT: target 그래프 위 surrogate 과제로 편향 episode를 만들어 LoRA 적응 | 합성 prior 대신 실제 공분산에서 외삽 bias 학습 | 25–35% | surrogate 클래스 구조가 실제와 다름 |
| FREE-1 | 식별성 진단 + community 평행이동 EM | 선택은 MAR이지만 positivity를 위반. 평행이동 성분만 식별 가능 | 25–35% | class-pure community(Cora)에서 식별 불가 |
| FREE-2 | SR-GNN의 CMD 정렬을 frozen GraphPFN 입력 adapter로 이식 | context 표현과 전체 표현의 moment를 일치시킴 | ~30% | 평균 방향 사영과 같은 결과. 클래스 신호 소거 |
| FREE-3 | SDCQ: 재가중 가능 shift와 support 결손으로 분해, 결손에만 보완 라벨 요청 | importance weighting은 support가 겹칠 때만 작동 | 진단 ~60%, 요청 ~35% | 결론이 "라벨을 더 받아라"뿐 |

## 2. 수렴한 지점

### 2.1 모든 제안이 같은 미검증 전제에 기대고 있다

모든 제안이 기대는 전제는 **"community 집중 손실 중 비라벨 그래프로부터 회복 가능한 몫이 있는가"**이다. 9건 중 7건이 이것을 먼저 재는 0단계 oracle 또는 ceiling 실험을 독립적으로 제안했다.

| 측정 | 제안한 agent | 답하는 질문 |
|---|---|---|
| 같은 편향 context에서 GCN·LP·SR-GNN의 손실, G = Δ_PFN − min_b Δ_b | RL-2 | 손실이 PFN 고유인가, 정보 부재인가 |
| oracle offset 제거, oracle-moment 수송(test 라벨로 offset 계산) | FREE-1, PFN-1, PFN-2 | 평행이동 성분이 손실을 얼마나 설명하는가 |
| oracle greedy 획득 대 community 커버리지 휴리스틱 | RL-3, FREE-3 | 라벨 획득의 headroom이 있고, 휴리스틱으로 충분한가 |
| oracle beam context 편집 | RL-1 | pseudo-label 편집으로 도달할 수 있는 천장 |
| KMM 재가중(복제) | PFN-2, FREE-3 | 실패가 예측된다. 예측이 맞으면 "support 결손" 진단을 지지한다 |
| community–클래스 순도/NMI | FREE-1, FREE-2, PFN-2, PFN-3 | 효과의 조절 변수. class-pure community면 복원 불가 |

**해석:** 이 측정들은 학습 없이 1–2일이면 끝난다. 결과가 어느 쪽으로 나와도 쓸모가 있다.
- 음성이면 원고 D7("빠진 증거는 되살릴 수 없다")을 정량적으로 강화한다.
- 양성이면 메소드의 필요성과 상한을 정해 준다.

### 2.2 처방 후보는 넷으로 묶인다

| 계열 | 해당 제안 | 비용 | 판단 |
|---|---|---|---|
| A. 라벨 없는 community feature 평행이동 제거/수송 | FREE-1, PFN-1, PFN-2, FREE-2 | forward 1–3회, 초–분 단위 | 4건이 독립적으로 수렴했다. 가장 싸다. **반드시 이겨야 할 기준선은 단순 community centering**(AdaBN식)과 self-training이다 |
| B. 보완 라벨 획득 | RL-3, FREE-3 | 라벨 예산이 필요 | 문제 설정이 바뀐다. 커버리지 휴리스틱이 거의 최적일 가능성이 높다 |
| C. 적응·pretraining | PFN-3, RL-2 | 수 시간–1.5일 | 실패한 selection-aware prior의 원인을 가르는 진단 가치가 있다. 원인 후보는 "공분산 불일치"와 "정보 부재"다 |
| D. RL context 편집 | RL-1 | 학습 ~18시간 | 노블티와 성공률 모두 낮다 |

### 2.3 RL에 대해

- RL 그룹 3명 모두 같은 결론을 냈다: RL이 필요하지 않을 가능성이 높다.
- RL이 정당화되는 조건은 셋이다.
  - 휴리스틱과 oracle 사이 간격이 클 때(RL-3).
  - pseudo-label 사이에 연쇄 효과가 있을 때(RL-1).
  - 노드 단위 적대 정책이 파라미터 탐색(TPE)을 이길 때(RL-2).
- 이 셋은 0단계 결과를 보고 나서 판단할 조건부 선택지다.

## 3. 새로 드러난 선행 연구 (원문 확인 수준은 대부분 [초록]/[검색])

- **SR-GNN** (Zhu 외, NeurIPS 2021, arXiv 2108.01099)
  - 6건이 지적했다.
  - GNN에서 "biased training node selection"을 직접 다룬다. 보정 수단은 CMD 정규화와 KMM 재가중이고, 손실의 약 30–40%를 회복한다.
  - **결론: 문제 자체는 이미 다뤄졌다.** 남는 새로움은 두 가지다.
    - PFN/ICL 설정. 학습 없이 context만 바꿀 수 있다.
    - 회복 가능한 성분과 결손 성분의 분해.
- DGNN (2201.07708). 라벨 선택 편향 아래의 GNN, 재가중.
- GPA (NeurIPS 2020), POLAR (2606.25197), Tab-AICL (2603.27385), GraphPart (TMLR 2023), SCARCE (ICLR 2024). 획득 계열.
- RTFM (2512.03307). 기준선 대비 격차를 보상으로 쓰는 적대적 PFN prior.
- Drift-Resilient TabPFN (2411.10634), DistPFN (2605.04363), Real-TabPFN (2507.03971), LoCalPFN, TabDPT, MixturePFN, TuneTables, CACTUs.

**정정:** brief §2의 "McDowell 외에서 oversampling이 해롭다"는 문장은 출처가 불분명하다.
- 이 문장은 내 앞선 리뷰 agent(a)가 [본문 요약]으로 보고한 것이다.
- PFN-2는 초록에서 그런 문장을 찾지 못했다.
- 인용하기 전에 원문을 확인해야 한다.

## 4. 판단과 추천

**사용자 기준에 비춘 평가**
1. **문제·모티브:** 문제 자체는 SR-GNN이 GNN에서 이미 다뤘다. "PFN에서 이 문제를 아무도 다루지 않았다"는 아직 가설이다.
2. **비자명성:** 아직 판단할 수 없다. 단순 처방이 이 문제에서 어디까지 통하는지 아무도 재지 않았기 때문이다. 단순 처방은 community centering, self-training, KMM, 진짜 라벨 추가다.
3. **연결의 검증:** 0단계가 바로 "진단 → 필요성" 단계다.

**추천: 0단계 "회복 가능성 진단"을 먼저 하나의 cycle로 사전 명세하고 실행한다.** 학습은 하지 않는다.
- **데이터:** Cora, CiteSeer, WikiCS, Amazon-ratings, Tolokers. 동질성과 community 순도가 서로 다른 그래프를 고른 것이다.
- **선택 규칙:** 무작위, Louvain 집중, Louvain 분산, 고차수. 클래스 개수는 고정한다. 원고 정의를 찾으면 그것을 따른다.
- **측정:** 위 표의 여섯 항목. 주 지표는 쌍별 AUC이고, 보조 지표는 bAcc다.
- **판정 분기:**
  - oracle 수송이 손실의 30% 이상을 회복하고, 그 효과가 community 순도와 연결되면 → 계열 A 메소드 cycle로 간다. 비교 대상은 community centering, self-training, CMD다.
  - GNN 기준선의 손실이 PFN과 같고 oracle 회복이 작으면 → D7을 정량적 경계 결과로 확정한다(원고 보강). 획득(B)을 선택지로 남긴다.
  - GNN은 회복하는데 PFN만 무너지면(G가 큼) → PFN 고유 문제다. 계열 C(PFN-3, RL-2)가 정당화된다.

## 5. 남은 확인 사항

- 원고의 community 집중·분산 정의와 shift ratio 계산식. 조사 agent가 아직 진행 중이다.
- SR-GNN 본문 확인(CMD와 KMM 세부, 회복 수치).
- McDowell 외의 oversampling 주장 원문 확인.

# 메소드 제안 요청 brief: 편향된 라벨 선택 아래의 graph PFN

작성일: 2026-11-20. 이 문서는 메소드 제안 agent 9명에게 공통으로 주는 배경이다. 아직 실행한 것은 없다.

## 1. 문제

Graph PFN(GraphPFN, NodePFN 계열)은 새 그래프에서 라벨이 붙은 노드 몇 개(context)를 받아, 학습 없이 나머지 노드를 분류한다.

실제 Web 그래프에서는 **어떤 노드에 라벨이 붙는지가 무작위가 아니다.**
- 라벨이 특정 community에 몰릴 수 있다(community 집중).
- 라벨이 고차수 노드나 저차수 노드에 치우칠 수 있다.

그 결과 context가 그 클래스를 대표하지 못한다. 이것은 클래스별 라벨 **개수**의 불균형과 다른 문제다.
- 개수 불균형은 pretraining의 context sampling을 바꾸거나(ImbNodePFN), 결정 단계를 보정해서 상당 부분 다룰 수 있다.
- 선택 편향은 같은 클래스 안에서 context와 query의 분포가 달라지는 **within-class covariate shift**다. 그래서 증거 자체가 빠진다.

## 2. 지금까지의 근거

출처는 사용자의 ImbNodePFN 원고 r88이다: `C:\Users\seongjin\Desktop\code\gts_pfn\imbnodepfn_upload\manuscript\main_v41_r88_final.tex`. 결론은 1536행 근처, 부록 Biased Label Selection은 1664–1691행, 원고 결정 주석 D7·D17은 126–136행과 203–212행이다.

**편향 선택의 손실 크기** (클래스 개수는 고정)
- 편향 선택은 그래프 평균 bAcc를 3.6–11.1점 낮췄다. community 집중이 모든 variant에서 가장 컸다.
- 공개 GraphPFN을 frozen으로 쓰면 WikiCS와 Tolokers에서 고차수 −10.8, 저차수 −15.4, community 집중 −14.7, community 분산 −12.8점이다.
- ImbNodePFN에서 community 집중은 쌍별 AUC를 −8.3점 바꿨다. oracle class-prior 보정(test 라벨을 씀) 후에도 −8.9점이 남는다. **결정 보정으로는 고칠 수 없는 랭킹 손실**이다.

**이미 시도한 것**
- **edge ablation(간선을 self-loop로 대체):** 고차수 손실은 6.0점에서 0.5점으로 줄었다. community 집중 손실은 7.9점으로 그대로다.
- **진짜 라벨 10개 추가:** community 집중에서 3.8점, 고차수에서 2.9점을 회복했다.
- **선택을 고려한 prior 두 개:** pretraining episode의 1/4 또는 1/2에 community 집중 선택을 넣었다. 둘 다 실패했다.
  - 첫째 prior: 합성 그래프에 community 고유 feature가 없었다. 그래서 집중 context도 클래스를 잘 대표했다(shift ratio 0.92, 실제 그래프는 1.4–4.9). dev에서 +0.50점이었고, 무작위 선택에서는 −0.94점이었다.
  - 둘째 prior: community별 feature offset을 넣어 대표성 편향을 재현했다(shift ratio 1.56). 그래도 두 조건 모두에서 낮아졌다(−0.40에서 −1.37점).
- **사용자의 판단(원고 D7):** "pretraining은 치우친 개수를 할인하도록 가르칠 수 있지만, 치우친 선택이 빠뜨린 증거는 되살리지 못한다." 그래서 이 문제를 기여가 아니라 범위의 한계로 두었다.

**관련 사실**
- GraphPFN prior는 context를 무작위 노드로 뽑는다(코드 확인). 그래서 prior에는 label shift와 selection shift가 둘 다 없다.
- frozen GraphPFN은 균형 공개 분할 Cora에서 72.9%다. 같은 조건의 GCN은 약 81%로 알려져 있다.
- TabPFN 계열에서 불균형 context를 oversampling하면 해롭다(McDowell 외, arXiv 2605.21742).

## 3. 쓸 수 있는 자원 (읽기 전용)

- **GPU:** RTX 3090 24GB 1장.
- **공식 GraphPFN:** LimiX-16M과 adapters 1.3. 로더는 `C:\Users\seongjin\Desktop\code\mgpfn\src\mgpfn\upstream.py`(`load_light`, `import_prior`, `prior_config`)이고, prior 과제 샘플러는 `mgpfn/prior_adapter.py::sample_official_task`다.
  - 모델은 **미분 가능**하다. 가중치를 fine-tuning할 수 있고, 입력에 대한 gradient도 계산할 수 있다.
  - context는 노드의 boolean `train_mask`로 넣는다. 그래서 같은 노드를 두 번 넣을 수 없다.
  - in-edge 전용 복제 노드를 추가하면 그래프 메시지를 바꾸지 않고 ICL context만 바꿀 수 있다(구현과 단위 테스트 완료).
- **공개 NodePFN checkpoint:** `C:\Users\seongjin\Desktop\code\gts_pfn\ctxedit\nodepfn_src\models_ckpts`.
- **ImbNodePFN 최종 checkpoint는 로컬에 없다.**
- **검증된 하네스:** `C:\Users\seongjin\Desktop\code\ssp-share\pfn_imbalance\2026-11-20_graphpfn_imbalance_diag\`
  - `harness.py`: frozen GraphPFN의 log-prob, feature 고정, 빈 그래프 F0.
  - `splits.py`: 중첩 분할.
  - `baselines.py`: GCN CE / 재가중 / balanced softmax, MLP, LP.
  - `run.py::p2_extended`: in-edge 전용 복제.
  - `analyze.py`.
  - Cora 호출 1회 약 3–5초, PubMed 수십 초.
- **데이터:**
  - Planetoid Cora, CiteSeer, PubMed: `gts_pfn/ctxedit/data/planetoid_pyg`.
  - WikiCS, Tolokers, Amazon-ratings: `gts_pfn/ctxedit/data/final_four`.
- **평가 프로토콜:** ImbNodePFN 프로토콜을 정확히 따를 필요는 없다. GraphENS식 중첩 분할을 써도 된다. 선택 규칙은 새로 정의해도 된다. 예: Louvain community 안에서의 집중과 분산, 차수 상위와 하위.

## 4. 판단 기준

사용자가 쓰는 연구 판단 기준(`C:\Users\seongjin\Desktop\code\ssp-share\RESEARCH_GOAL.md`의 세 질문)을 이 문제에 맞춰 옮긴 것이다.
1. **문제·모티브:** 관련 분야가 이 문제를 어디까지 다뤘는가? 다룬 범위는 원문으로 확인한다. "아무도 다루지 않았다"는 아직 검증할 가설로 둔다.
2. **해결의 비자명성:** 단순한 처방이 언제 충분하고 언제 부족한가? 단순한 처방의 예는 진짜 라벨 추가, edge 제거, 결정 보정, pseudo-label self-training이다. 복잡성 자체를 비자명성으로 보지 않는다.
3. **연결의 검증:** 문제 진단 → 해결이 필요한 이유 → 해결 원리 → 순위·정확도 회복을 통제 실험으로 연결할 수 있어야 한다.

**새 standalone 모델을 발명하는 것이 목표가 아니다.** 문제를 규명하는 것과 실질적인 해결을 잇는 것이 목표다.

## 5. 각 제안이 담아야 할 것

1. 핵심 아이디어 한 문단과 해결 원리. 왜 이 방법이 빠진 증거를 복원하거나 우회하는가.
2. 구체적인 알고리즘. 입력, 출력, 수식, 하이퍼파라미터, test 시점 비용을 쓴다. pretraining 수정이 필요하면 그 비용도 쓴다.
3. 가장 가까운 선행 3개와 차이. **원문을 확인**하고 [본문]/[초록]/[검색]/[기억]으로 표기한다.
4. 반드시 이겨야 할 단순 기준선과, 이 방법이 실패할 조건.
5. 이 workspace에서 2–3일 안에 돌릴 수 있는 최소 반증 실험. 데이터, 조건, 주 지표, 성립과 기각 기준을 쓴다.
6. 솔직한 평가: 성공 가능성, 노블티의 크기, 가장 큰 위험.

**작업 규칙**
- 파일을 만들거나 고치지 않는다. 결과는 최종 응답으로만 보고한다.
- 동의를 끌어내려 하지 말고 반례를 스스로 찾는다.

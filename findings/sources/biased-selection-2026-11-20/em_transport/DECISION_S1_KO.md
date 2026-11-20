# S1 oracle 분해 판정 (2026-11-20, 진단 전용)

- 실행: `s1.py` → `results/s1/records.jsonl`(160줄 = 그래프 4개 × seed 10개 × ρ 2개 × 정책 2개), 분석: `s1_analyze.py` → `results/s1/report.json`.
- 동결: FREEZE_S1.json의 PLAN_KO.md, cct.py, s1.py, s1_analyze.py, stats.py 해시가 실행 뒤에도 모두 일치했다.
- 지위: S0 gate가 실패한 뒤 사용자의 명시 지시로 실행한 진단이다. 메소드 판정은 없다. oracle은 모두 pool 라벨을 쓴다.
- 평가: PubMed와 photo는 W val 행으로 평가했다. Cora와 CiteSeer는 eval 절반 A로 평가했다(perm seed 2026112099). 절반 B는 점수를 매기지 않았다.

## 사전 규칙의 결과

- 규칙: "평가 가능 그래프 중 O-both − O-ctx의 CI 하한이 2pt를 넘는 그래프가 2개 이상이면 O-both를 채택한다."
- **결과: 0개. O-both 채택 안 함.**
  - photo는 +3.1 [+1.8, +4.3]이다. 하한이 2pt에 못 미친다.
  - CiteSeer(−3.4)와 PubMed(−2.5)는 유의하게 음수다.

## 회복률 (commconc; 손실 = GP uniform − GP commconc, AUC pt; 괄호는 seed-cluster 95% CI)

| 그래프 | 평가 가능 | 손실 | O-ctx | O-q | O-both | O-ctx+nbr | O-prior |
|---|---|---|---|---|---|---|---|
| CiteSeer | 예 | 5.2 [3.9, 6.6] | 0.51 | 0.61 | **−0.14** | 0.42 | 1.44 |
| PubMed | 예 | 13.5 [8.4, 19.4] | 0.67 | 0.61 | 0.49 | **0.76** | 1.17 |
| photo | 예 | 13.8 [11.7, 15.8] | 0.43 | 0.47 | **0.65** | 0.48 | 0.95 |
| Cora | 아니오 | 2.5 [1.8, 3.3] | 0.61 | 0.58 | 0.42 | 0.61 | 1.56 |

**uniform 변화 (pt):**
- O-both는 CiteSeer −2.8, PubMed −1.2다.
- O-ctx+nbr는 CiteSeer −0.6이다.
- O-prior는 CiteSeer +3.2, PubMed +4.6, Cora +1.7, photo +0.4다.

**O-ctx 대비 차이 (pt):**

| 그래프 | O-both − O-ctx | O-ctx+nbr − O-ctx |
|---|---|---|
| CiteSeer | −3.4 | −0.5 |
| PubMed | −2.5 | +1.2 [+0.6, +1.7] |
| photo | +3.1 | +0.8 [+0.5, +1.0] |
| Cora | −0.5 | 0.0 |

## 해석

1. **S0의 합성 결과("이웃과 query까지 옮기면 거의 완전 회복")는 실데이터로 옮겨지지 않았다.**
   - O-both가 도움이 된 곳은 photo뿐이다. CiteSeer, PubMed, Cora에서는 O-ctx보다 나쁘고 uniform도 해친다.
   - O-ctx+nbr의 이득은 PubMed +1.2pt, photo +0.8pt다. 작지만 유의하다.
   - 합성 생성식은 community offset이 클래스와 독립이고 순수한 가산 형태라고 가정한다. 실데이터에서는 그 가정이 맞지 않는 것으로 보인다(가설).
2. **클래스 무관 offset δ_k를 모든 노드에서 빼는 것은 실데이터에서 해롭다.** 이는 Phase 0에서 라벨 없는 comm-z가 해로웠던 결과와 일치한다. 이번에는 참 라벨로 구성 보정을 했는데도 해롭다. 따라서 "라벨 없는 클래스 무관 양쪽 수송" 후보(DECISION_S0 다음 후보 2)는 근거를 잃는다.
3. **특징 수송 oracle의 상한은 실데이터에서 0.4–0.76이다.** 참 라벨로도 손실의 약 절반에서 3/4만 되돌린다.
4. **O-prior(출력 쪽 참 p(y|community) 보정)는 회복률 0.95–1.56이다.** 단 이 값은 과대평가다.
   - O-prior는 uniform에서도 +0.4 ~ +4.6pt를 올린다.
   - 즉 선택 편향 보정만 하는 것이 아니다. pool 라벨에서 얻은 community–라벨 정보 자체를 주입한다.
   - 사후 보조 계산(사전 정의가 아님): (commconc 이득 − uniform 이득) / 손실을 구해 보면 CiteSeer 0.83, PubMed 0.82, photo 0.92, Cora 0.88이다.
   - 이렇게 보정해도 특징 수송 oracle보다 높다. 출력 쪽 community 사전확률 축이 특징 쪽 축보다 상한이 높다는 신호다. 단 이것은 가설 수준이다.
   - 이것은 W의 R2(라벨 없는 community 라벨 이동 EM)가 실패한 것과 별개의 문제다. R2는 p(y|k)를 라벨 없이 추정하지 못해서 실패했다. 상한이 낮아서가 아니다.

## 결론

- **확정 (dev 그래프 4개, oracle):**
  - 실데이터에서 특징 수송의 상한은 O-ctx 수준(0.43–0.67)이다. 이웃과 query까지 옮겨도 크게 오르지 않는다.
  - 클래스 무관 양쪽 중심화는 해롭다.
- **가설:** 회복할 여지가 가장 큰 곳은 출력 쪽 community별 클래스 사전확률이다. 그러나 이를 라벨 없이 추정하는 것은 R2가 이미 실패한 문제다. 소수의 진짜 라벨이나 context 안의 정보(community별 context 클래스 구성)로 추정하는 변형만 남는다.

## 정정 (2026-11-20, CPC v1 리뷰 A·B)

- PubMed/photo의 dev `eval_mask`는 val뿐이다. 그래서 위 oracle의 pool(`~eval_mask`)에 **test 노드 라벨이 포함되었다.** 평가 노드(val) 라벨 누출은 아니다. 다만 S3 확인 평가 노드(test)가 oracle 집계에 이미 쓰였다. 후속 cycle은 pool = ¬(val ∪ test)로 둔다.
- O-prior의 보정항은 (k, c)마다 상수다. 따라서 community 내부 pairwise AUC는 변하지 않고, 이득은 community 간 쌍에서만 나온다. 위 해석 4의 "출력 쪽 축의 상한이 높다"는 "community 라벨 구성 정보 주입의 효과"와 구별되지 않는다. 구별은 `../2026-11-20_context_prior/PLAN_KO.md`의 O-swap/O-info로 한다.

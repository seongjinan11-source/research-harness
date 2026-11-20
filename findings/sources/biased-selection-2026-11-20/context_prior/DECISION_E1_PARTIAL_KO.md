# CPC E0 중단과 부분 E1 기록 (2026-11-20)

## 상태

- **E0 캐시를 사용자 지시로 중단했다.**
  - 완료: Cora 40/40, CiteSeer 40/40, photo 14/40, PubMed 0/40 단위.
  - 캐시 파일(`results/e0/`)은 보존한다.
- **정식 E1 판정과 gate는 내리지 않았다.**
  - 평가 가능한 그래프가 CiteSeer 하나뿐이다.
  - 따라서 사전 규칙상 투표는 "결론 없음"이다.
  - 아래 내용은 모두 **기술 통계**이며 판정이 아니다.
- 동결 상태: 채점은 FREEZE_E1의 `score.py`와 `judge.py`로 했다. 출력은 `results/interim_2graphs/`에 두었다. `selection.json`은 만들지 않았다(scratch에서 생성된 것은 삭제했다).
- 중단 사유: 사용자 판단이다. CiteSeer에서 라벨 없는 주 후보(E-moment)가 gate (a)를 이미 통과하지 못했다. 그래서 전체 gate를 넘으려면 photo와 PubMed가 둘 다 통과해야 했고, 그 가능성은 낮다고 판단했다.

## 부분 결과 (AUC pt; SG = commconc 이득 − uniform 이득; 괄호는 seed 군집 95% CI)

- CiteSeer: 평가 가능, 손실 5.2pt
- Cora: 평가 불가, 손실 2.5pt

| arm | CiteSeer SG | CiteSeer SRec | Cora SG | Cora SRec |
|---|---|---|---|---|
| O-swap (pool 라벨 없음, uniform-context PFN의 community prior로 교체) | +4.0 [+2.8, +5.2] | 0.77 | +2.2 [+1.7, +2.8] | 0.89 |
| O-prior | +4.3 [+3.3, +5.4] | 0.83 | +2.1 | 0.86 |
| O-info (정책과 무관한 무작위 라벨 \|ctx\|개) | +3.7 | 0.72 | +2.0 | 0.82 |
| O-ctx (특징 수송) | +2.9 | 0.55 | +1.4 | 0.55 |
| E-moment (주 설정 β = 1, r = 없음) | +1.1 [−0.3, +2.3] | 0.21 | +0.5 [−0.5, +1.6] | 0.20 |
| E-moment (4개 설정 범위) | +0.9 ~ +1.1, 모두 CI가 0을 포함 | 0.17–0.21 | | |
| E-LP / E-PoE | −3.4 / −2.5 | | +0.2 / +0.6 | |
| E-ctxcount (λ = 1) | −4.6 | | −4.6 | |
| E-R2 (W 선택 설정) | +0.3 | 0.05 | +0.7 | 0.29 |
| C&S (가장 나은 설정) / Correct-only | −5.1 / −3.4 | | −1.9 / −1.1 | |
| LP-PoE / stacking (가장 나은 설정) | −3.5 / −1.0 | | 0.0 / −0.2 | |
| 재표집 / REF-GP | −1.0 / +0.1 | | +0.4 / +0.1 | |

- E-moment null 대조 SG는 약 0이다(CiteSeer −0.0 ~ +0.2).
- TV 탈락 규칙에는 걸리지 않았다. context가 없는 community에서 E-moment TV는 0.38(평탄 0.62)이다.
- 원형 상대 오차는 commconc 1.34, uniform 1.03이다. 참 원형을 넣으면 TV는 0.27이다.

**2×2 분해 (CiteSeer commconc 이득, AUC):**
- 특징 2.7, prior 7.5, 둘 다 8.0, 교호작용 −2.1
- Shapley: 특징 1.6, prior 6.4

**Q3 (예산 25%, CiteSeer 라벨 60개, commconc 이득):**

| arm | CiteSeer | Cora |
|---|---|---|
| L-ctx | **+7.2** | +3.6 |
| L-prior | +4.2 | +2.5 |
| L-moment | +4.5 | +2.8 |
| L-both | +6.2 | +3.7 |
| L-uncov | +5.1 | +2.8 |

- 원형을 새 라벨로 추정하면 TV가 0.38에서 0.26으로 줄어든다.

## 해석 (그래프 1–2개; 가설 수준)

1. **편향 context는 PFN의 community별 클래스 사전확률을 틀어 놓는다.**
   - O-swap은 pool 라벨 없이 손실의 77–89%를 되돌린다(uniform에서는 정의상 0).
   - 따라서 S1 O-prior의 이득을 "정보 주입"만으로는 설명할 수 없다(리뷰 B의 우려에 대한 부분 답).
   - 단 O-swap은 uniform context에서 나온 PFN 출력이 필요한 oracle이다. 실제로는 쓸 수 없다.
2. **라벨 없는 추정으로는 그 왜곡을 실용적으로 되돌리지 못했다.**
   - E-moment가 되돌린 것은 약 1pt로 유의하지 않다.
   - 병목은 commconc context로 만든 클래스 원형의 오차로 보인다.
   - context 라벨을 전파하는 방법(LP, C&S, ctxcount)은 오히려 해롭다. 편향을 키운다.
3. **진짜 라벨이 있으면 context에 넣는 것이 prior나 원형 추정에 쓰는 것보다 낫다.** 사전 예측과 일치한다.

## 결론

- 이 cycle의 라벨 없는 CPC 메소드 경로는 **중단한다.** 사유는 실용 성능이 부족하다는 것이다. 정식 gate 판정은 미완이다.
- 보존할 근거:
  - "선택 특유 prior 왜곡"의 oracle 측정(O-swap). 그래프 1–2개 결과다.
  - 라벨 없는 추정기와 기존 후처리의 음성 결과.
  - Q3 라벨 예산 비교.
- photo와 PubMed는 측정하지 않았으므로 일반화를 주장하지 않는다.

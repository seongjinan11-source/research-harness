# 고정 후보의 독립 확인: 통과

2026-11-04. 이전 12개 개발 시나리오에서 고른 **기존 d3_spread**를 변경하지
않고 새 IID 시나리오 40개에서 확인했다. 데이터 생성·코드·분석·성공 조건을
평가 전 고정했다. 기존 d2_scalar 주 비교 실패는 그대로 보존한다.

## 완료한 실행과 검사

- corrected physical D-optimal tomography, d1/d3 x scalar/spread의 4개 arm.
- 40 시나리오 x4 arms =160 runs, 각100개 unique feasible 평가, 총16,000 charged rows.
- 저장된16,000개 목적함수 값을 별도 replay, HV/AUC 독립 재계산 모두 통과.
- 이전 one()/predict()/SSPd/propose 실행 코드를 그대로 사용했음을 구조·hash로 확인.
- Source·scenario·분석 hash 유지, fallback 0, L 경계 fit 0.
- 원본 fitter는 optimizer status를 노출하지 않으므로 수렴 성공률은 확인하지 않았다.

## 사전에 고정한 주 비교

**d3_spread 대 d1_spread**: size 모델의 추가 효용을 같은 count 분산 정책에서 비교.

| 항목 | 결과 | 사전 기준 |
|---|---:|---:|
| 평균 AUC의 비율 개선 | +1.30919% | >=1% |
| paired bootstrap 95% 구간 | [+1.04467%, +1.57140%] | 하한 >0 |
| 시나리오 승리 | 38/40 | >=28/40 |
| 최악 paired 변화 | -0.80832% | >=-2% |
| 전체 arm wall time 비율 | 1.92707배 | <=3배 |

**다섯 기준 모두 통과했다.** Bootstrap은 40개 IID 시나리오를 네 arm 전체에
공통으로 10,000번 재표집해 ratio-of-means를 재계산했다. Seed와 구간 방식은
평가 전 고정했으며 결과 이후 degree·분포·endpoint를 바꾸지 않았다.

## 보조 결과와 해석

| 방법 | 평균 AUC |
|---|---:|
| d1 scalar | 0.655725 |
| d1 spread | 0.670938 |
| d3 scalar | 0.665127 |
| d3 spread | 0.679722 |

총 조합 d3_spread 대 d1_scalar는 +3.65966%, 40/40 승리였다. 보조 비교의
95% 구간은 [+3.38523%, +3.92978%]지만 다중 비교 보정 없는 탐색적 구간이다.
분산 정책만의 d1_spread 대 d1_scalar는 +2.32009%, d3_scalar 대 d1_scalar는
+1.43388%였다. Degree-by-policy의 additive AUC interaction 구간은0을 포함하므로
특별한 synergy를 입증했다고 주장하지 않는다.

추가 독립 검사에서 주 비교의 두 arm은 **3,200개 paired proposal 모두 실제
count가 같았다**. 같은 count에서는 size-only curve가 상수이므로, 이 +1.31%를
추가 count coverage 효과로 해석할 수 없다. 고차 size 모델의 joint fitting으로
item coefficients 및 후속 archive가 달라지는 composition/trajectory 효과다.
정확히 어떤 추정 오차가 줄었는지는 아직 분리하지 않았다. 이는 nonlinear size
shape가 composition 추정에 미치는 영향을 조사할 구체적 후속 단서다.

## 결론의 범위와 결정

고정한 기존 고차 size 모델이 count 분산 위에 더하는 작은 이득이, 새 시나리오에서도
유지된다는 근거를 확보했다. 지정한 synthetic physical 시나리오·초기 archive·
100평가 예산에서의 실용적 개선 확인이다. 다른 물리 과제의 일반화, 새 알고리즘의
novelty, 동일 시간 예산에서의 우위를 입증한 것은 아니다. Wall ratio는 비용
수용성 검사이며 더 빠른 방법보다 같은 시간에 좋다는 주장은 하지 않는다.

이 bank는 다시 tuning/degree 선택에 쓰지 않는다. 후보는 확인된 실용 대조로
보존한다. 기존 budgeted ROI +16.4%의 물리적 검증, 다른 문제로의 전이와
논문의 독립적인 새 기여는 남아 있다. 이번 known-count D-optimal 확인으로
그 과제를 완료했다고 주장하지 않는다.

세 결과 리뷰와 원시 X/Y, source manifest, 재현 스크립트는 같은 디렉터리에 보존한다.

결과 리뷰 완료: 수학 reviewer의 독립 CSV/bootstrap 재계산이 CI 오차2.18e-14
이내로 일치했다. 실험 reviewer는 분석 작성자 역할을 공개하고 별도 source/raw
metadata/CSV 집계를 감사했다. Scientific reviewer도 좁은 확인 범위와 다섯 기준
통과에 동의했다. 기존 개발 결과나 ROI의 미검증 주장을 새 성공으로 바꾸지 않았다.

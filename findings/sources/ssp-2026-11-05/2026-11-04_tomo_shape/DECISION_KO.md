# Corrected physical size-shape 실행 판정

2026-11-04. 새 개발 시나리오 12개 x 기존 SSPd 차수 1/2/3 x 정책 scalar/spread의
72 runs, 7,200 charged evaluations를 완료했다. 원본 SSPd와 제안 함수를 변경하지
않았다. 모든 saved objective와 HV/AUC의 독립 replay 통과, fallback 0. 4 CPU
workers, 온라인 wall time 12.21초. 원본 fitter는 optimizer status를 노출하지
않으므로 수렴 성공률은 알 수 없다. 유한한 hyperparameters/NLL과 경계는 기록했다.

| 기존 모델 / 정책 | 평균 AUC |
|---|---:|
| d1 scalar | 0.655353 |
| d1 spread | 0.671483 |
| d2 scalar | 0.661337 |
| d2 spread | 0.674112 |
| d3 scalar | 0.664245 |
| d3 spread | 0.678788 |

## 사전 주 비교: 실패

d2_scalar 대 d1_spread는 -1.51098%, 승리 0/12, 최악 -2.34878%.
비용 1.65474배로 비용 조건만 충족했다. '2차 size 모델만으로 강한 count분산
기준선을 넘는다'는 이번 개발 가설은 지지되지 않는다. Degree3 결과로 주 비교를
대체하거나 성공으로 재분류하지 않는다.

## 보존할 양성 보조 결과

- 같은 scalar 정책에서 d2는 d1 대비 +0.91314%(11/12 승리), d3는 +1.35689%(12/12).
- 같은 spread 정책에서 d2는 d1 대비 +0.39153%(9/12), d3는 +1.08789%(11/12).
- Spread는 scalar 대비 d1/d2/d3에서 각각 +2.46131/+1.93170/+2.18937%, 모두 12/12 승리.
- d3_spread 대 d1_scalar의 합산 차이는 +3.57597%. 이는 탐색적으로 보고하는
  기존 모델·정책 조합의 개발 결과이며, 사전 주 비교나 새 방법의 기여가 아니다.

고차 shape의 작은 이득과 count coverage의 추가 이득이 함께 관찰된다. 같은
count에서 size curve는 상수지만, 공동 refit한 item coefficients가 달라지므로
고차 모델의 이득을 size-only 보정에 전부 귀속할 수 없다. 예측 RMSE 개선과
실제 online 효용은 구분한다. PSD logdet의 submodularity만으로 최적 cardinality
profile의 concavity를 가정하지 않는다.

## 메타 결정

기존 고차 모델+coverage는 실용 대조·추후 확인 후보로 보존한다. 이번 bank에서
degree/noise/acquisition을 추가 조정하지 않는다. 기존 X8 negative 결과를 지우지
않으며 corrected physical의 새 개발 결과와 분리한다. 새 oracle나 모델의
novelty를 확보한 것으로 주장하지 않는다.

앞선 utility cycle과 합쳐 실제 144 runs / 14,400 charged objective rows를
실행했고 같은 수의 offline audit replay를 별도로 수행했다. 원고의 알려진
physical-layout 결함과 budgeted ROI +16.4% 검증 미완료는 남아 있다. 이번
known-count D-optimal 실험으로 다른 criterion/constraint의 ROI 결과를 대체하지 않는다.

세 관점의 리뷰: RESULT_REVIEW_EMPIRICAL.md, RESULT_REVIEW_NOVELTY.md,
REPORT_THEORY.md. 모든 원시 X/Y, 계획·수치 검사·코드·source manifest를 보존한다.

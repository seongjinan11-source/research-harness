# 실행 결과와 메타 판정

2026-11-04. 72개 개발 run, 7,200개 charged 평가 완료. 모든 raw objective와
trajectory/AUC를 독립 재계산했다. fallback 0, 720 fits의 optimizer flag 실패 0.
4 CPU workers, online wall time 66.71초. 새 unknown objective 평가를 사용한
추가 학습/탐색은 없다. 원본 코드·결과와 이전 음성 pilot은 보존했다.

## 결과: corrected physical 조건, 6개 새 개발 시나리오

| 정책 | 평균 AUC | 같은 runner의 SSP mean 대비 |
|---|---:|---:|
| SSP mean | 0.653647 | — |
| sampled count | 0.665677 | +1.8405% |
| deterministic count spread | 0.668829 | +2.3227% |
| mean HVI | 0.658311 | +0.7136% |
| candidate-pool EHVI | 0.655745 | +0.3210% |
| 추가 fixed-count swap EHVI | 0.655716 | +0.3165% |

추가 swap은 strongest simple comparator인 spread보다 -1.9606%, 6/6 시나리오
패배다. pool EHVI보다 wall time 20.45배로 성능·비용 기준을 모두 실패했다.
480개 제안 중 339개에서 acquisition 자체는 높아졌으므로 계산한 surrogate
효용을 더 최적화하는 것과 실제 성능 개선을 구분해야 한다. 원인은 확정하지 않는다.

## 판정

**이번 local-search acquisition의 확대와 elaborate EHVI-oracle/bounds 개발을
중단한다.** 작은 exact SSP fixture 6개도 mean-prefix만으로 EHVI 최대값에
도달했고 mean-HVI/EHVI action이 같았다. 이 조건에서 새 복잡한 oracle을
만들어야 한다는 실험 근거가 없다. EHVI 일반을 기각하거나 새로운 수학적
가능성의 부재를 증명한 것은 아니다.

단순 count spread의 개선은 보존한다. 새 알고리즘의 novelty나 독립 확인으로
주장하지 않는다. 모든 arm은 같은 profiled fitter와 augmented pool을 사용한
새 runner 대조이며 원고의 historical d1/d1samp와 정확히 같은 실행은 아니다.

## 실질적으로 발견한 benchmark 결함

원본 Radon 배열의 detector/angle 순서 및 360/180도 불일치를 발견했다.
역사적 행렬·raw 600개 값은 재현되지만 물리적 angle selection 해석은 수정해야
한다. corrected physical을 주 조건으로, historical을 진단 조건으로 분리했다.
이 결함은 X7 budgeted ROI의 cache에도 전파되어 기존 +16.4%의 물리적 검증이
남아 있다. 이번 known-count D-optimal 결과로 그 수치를 확인했다고 주장하지 않는다.
영향 범위는 paper/notes/tomography_operator_audit.md에 기록했다.

## 후속 모델 진단과 범위

사후에는 새 objective를 호출하지 않고 저장된 SSP-mean physical archive 6개의
세 시점에서 다음 8개 실제 관측만 사용했다(모델당 144개). 동일 archived 입력으로
기존 SSPd1/d2/d3를 refit했을 때 prediction RMSE는 .05659/.02196/.01501이었다.
Profiled SSP와 numerical d1은 거의 같아 이 차이를 optimizer 차이로 설명할
근거가 없다. d2/d3는 이미 X8에 존재하는 기준선이며 새로운 방법이 아니다.
선택된 관측은 낮은 K와 높은 K에만 있어서 중간 count 일반화나 online 이득을
보장하지 않는다. 다음 실행은 acquisition 변형 대신 이 모델 오차 가설을
새로운 별도 cycle에서 직접 검증해야 한다.

## 결과 리뷰

세 관점 모두 구현 오류 때문에 결과를 폐기할 근거를 찾지 못했다.
RESULT_REVIEW_EMPIRICAL.md, RESULT_REVIEW_NOVELTY.md, REPORT_THEORY.md 참조.
원시 결과와 수치·적분·swap 검사 및 source locks는 같은 디렉터리에 있다.

# Missing-feature GraphPFN: Gate A/B 진단

ACM·IMDB는 이미 확인한 개발 그래프다. 아래 결과는 새로운 모델의 검증이나 새로운 그래프에 대한 zero-shot 일반화 증거가 아니다. 모든 규칙은 protocol.json에 먼저 고정했고, selector 학습과 predictor 가중치 갱신은 하지 않았다.

**식별상의 주의:** 아래 FP는 PCFI/FISF+GraphPFN branch의 약칭이다. 이 branch도 graph/path/structure 및 context labels를 사용하므로 feature-only evidence가 아니다. FISF에는 합성 피처 주입도 있다. IMDB에서 이 branch가 LP보다 강하다는 사실만으로 실제 관측 피처의 predictive utility가 원인이라고 결론 내릴 수 없다. 이를 확인하려면 동일 branch에서 관측 피처 값만 제거하거나 섞는 matched intervention이 추가로 필요하다.

## 현재 판정

**Gate A에서는 동일 query 집합 안의 보완성을 확인했지만, Gate B의 사전 고정 규칙은 두 그래프의 강점을 모두 보존하는 성공 기준을 충족하지 못했다. Gate C 학습은 보류한다.**

k=20 FISF branch 기준, ACM은 LP 85.55% → Global 85.55% / Local 85.56%다. IMDB는 PFN 44.55% → Global 41.51% / Local 41.92%로 오히려 낮아졌다. Oracle은 각각 90.43%, 59.16%지만, 그 여지를 관측 가능한 신호로 회수한 것은 아니다.

사전 방향으로 측정한 within-task AUROC는 confidence 차이가 ACM 0.702 / IMDB 0.572, 재가림 복원 오차 proxy가 0.484 / 0.511이다. Local context loss는 0.600 / 0.525다. 복원 proxy는 현재 형태에서 약했고, confidence에는 일부 구분 신호가 있다. 이 결과를 근거로 신호를 고르거나 부호를 바꿔 얻는 후속 결과는 개발용으로 취급해야 한다.

**신뢰도와 평가 목적의 불일치도 관측했다.** IMDB k=20, FISF, seed=11에서 context Brier는 LP 0.6446 < PFN 0.6550으로 LP를 선호한다. 실제 test Brier도 LP 0.6624 < PFN 0.6672이지만, accuracy는 LP 35.54% < PFN 44.65%다. 따라서 이번 실패를 context 오차 추정의 불안정성만으로 설명할 수 없다. 확률 손실의 신뢰도와 분류 정확도의 효용은 구분해야 한다. 이 사례는 결과를 본 뒤의 진단이며, 새로운 규칙의 검증은 아니다.

다음 작은 확인 실험으로는 동일 PFN branch의 관측 피처 제거·셔플 대조를 먼저 고정하는 것이 적절하다. 그래프/path/합성 주입 설정과 추론량을 맞춰 실제 피처 값의 추가 효용을 확인하고, 그 뒤 어떤 평가 위험을 추정할 selector가 필요한지 정한다. 양방향 모델 확대와 synthetic selector 학습은 이번 진단만으로 정당화되지 않는다.

## Gate A: 같은 query에서 실제 보완성이 있는가?

99.5% whole-node feature missing, 전체 public test, 동일 context와 결측 마스크. 각 수치는 세 seed의 평균이다. Oracle은 둘 중 하나라도 맞으면 맞는 것으로 계산한 정답 기반 상한이며, 실행 가능한 선택기가 아니다.

| Graph | k/class | PFN input | PFN | LP | FP만 정답 | LP만 정답 | Oracle |
|---|---:|---|---:|---:|---:|---:|---:|
| ACM | 5 | pcfi | 58.52 | 80.79 | 8.46 | 30.73 | 89.25 |
| ACM | 5 | fisf | 64.45 | 80.79 | 8.03 | 24.37 | 88.82 |
| ACM | 20 | pcfi | 75.45 | 85.55 | 5.05 | 15.16 | 90.61 |
| ACM | 20 | fisf | 74.36 | 85.55 | 4.88 | 16.07 | 90.43 |
| IMDB | 5 | pcfi | 32.54 | 32.04 | 14.12 | 13.62 | 46.16 |
| IMDB | 5 | fisf | 32.66 | 32.04 | 14.11 | 13.48 | 46.15 |
| IMDB | 20 | pcfi | 42.61 | 35.41 | 22.24 | 15.05 | 57.66 |
| IMDB | 20 | fisf | 44.55 | 35.41 | 23.75 | 14.62 | 59.16 |

## Gate B: 미리 정한 학습 없는 선택 규칙

Global은 context 3-fold crossfit Brier가 작은 predictor를 전체 query에 사용한다. Local은 같은 context 오차 차이를 union 그래프의 고정 LP 연산으로 전달해 query마다 선택한다. 차이가 0이면 50:50, context 전파 질량이 0이면 global을 사용한다. PFN·LP 모두 같은 context fold를 제외했다. 이 규칙은 단순한 진단 baseline이며 메소드 novelty를 주장하지 않는다.

| Graph | k/class | PFN input | PFN | LP | 50:50 | Global | Local | Local≠Global (%) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ACM | 5 | pcfi | 58.52 | 80.79 | 81.04 | 80.79 | 80.84 | 5.62 |
| ACM | 5 | fisf | 64.45 | 80.79 | 81.11 | 80.79 | 80.94 | 4.31 |
| ACM | 20 | pcfi | 75.45 | 85.55 | 83.65 | 85.55 | 85.60 | 24.47 |
| ACM | 20 | fisf | 74.36 | 85.55 | 82.20 | 85.55 | 85.56 | 25.68 |
| IMDB | 5 | pcfi | 32.54 | 32.04 | 33.79 | 32.13 | 32.13 | 5.43 |
| IMDB | 5 | fisf | 32.66 | 32.04 | 33.76 | 32.12 | 32.24 | 10.26 |
| IMDB | 20 | pcfi | 42.61 | 35.41 | 40.80 | 40.72 | 40.55 | 13.06 |
| IMDB | 20 | fisf | 44.55 | 35.41 | 42.54 | 41.51 | 41.92 | 13.63 |

## 오류의 예측 가능성: 동일 task 내부 AUROC

FP만 맞는 query를 양성, LP만 맞는 query를 음성으로 삼았다. 둘 다 맞거나 둘 다 틀린 query는 AUROC에서 제외한다. 방향은 사전 고정했으므로 0.5 미만도 그대로 보고한다. 신호별로 seed 안에서 AUROC를 계산한 뒤 평균한다. 데이터셋 차이를 맞히는 효과와 구분하기 위한 분석이며, AUROC가 높아도 전체 test 성능 개선을 보장하지 않는다. 신호를 결과에 따라 조합하거나 부호 반전하지 않았다.

| Graph | k/class | PFN input | confidence | entropy | views | stability | distance | 2-step mass | recovery | local loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACM | 5 | pcfi | 0.717 | 0.525 | 0.367 | 0.558 | 0.467 | 0.441 | 0.438 | 0.530 |
| ACM | 5 | fisf | 0.701 | 0.529 | 0.362 | 0.562 | 0.443 | 0.432 | 0.483 | 0.513 |
| ACM | 20 | pcfi | 0.701 | 0.588 | 0.369 | 0.489 | 0.468 | 0.496 | 0.454 | 0.613 |
| ACM | 20 | fisf | 0.702 | 0.602 | 0.400 | 0.471 | 0.468 | 0.507 | 0.484 | 0.600 |
| IMDB | 5 | pcfi | 0.555 | 0.554 | 0.448 | 0.442 | 0.484 | 0.500 | 0.530 | 0.503 |
| IMDB | 5 | fisf | 0.551 | 0.550 | 0.450 | 0.440 | 0.482 | 0.499 | 0.533 | 0.502 |
| IMDB | 20 | pcfi | 0.565 | 0.560 | 0.463 | 0.467 | 0.474 | 0.486 | 0.507 | 0.507 |
| IMDB | 20 | fisf | 0.572 | 0.570 | 0.450 | 0.458 | 0.472 | 0.480 | 0.511 | 0.525 |

## Feature recoverability는 predictive utility와 다르다

각 seed에서 이미 관측된 피처 행의 20%(올림)를 추가로 가리고, 기존 PCFI/FISF를 원래 전체 좌표에서 재실행했다. 숨겨진 원래 결측 피처는 입력에 포함하지 않았다. 정규화 오차는 행별 SSE / max(정답 행 제곱합, 1e-12)다. Mean은 남은 관측 행의 평균값으로 복원한 baseline이다. 복원 오차가 작다고 분류에 필요한 피처가 잘 보존됐다는 뜻은 아니다.

| Graph | Imputer | 관측 행 수/seed | 추가 가림 행 수/seed | 정규화 오차 | Mean baseline |
|---|---|---|---|---:|---:|
| ACM | pcfi | 19,20,15 | 4,4,3 | 1.296 | 0.669 |
| ACM | fisf | 19,20,15 | 4,4,3 | 0.779 | 0.669 |
| IMDB | pcfi | 19,22,16 | 4,5,4 | 1.008 | 1.065 |
| IMDB | fisf | 19,22,16 | 4,5,4 | 1.108 | 1.065 |

## 같은 task 내부에서 신호에 따라 정확도 차이가 달라지는가?

![Reliability quartiles](C:/Users/seongjin/Desktop/code/gts_pfn/evidence_reliability/results/reliability_quartiles.png)

k=20, 각 task 안에서 observable score의 사분위 구간을 먼저 나누고 그 구간의 PFN−LP 정확도를 계산했다. 선은 세 seed의 평균이며 신뢰구간이 아니다. 점수값이 같으면 같은 구간으로 유지하므로 일부 구간은 비어 있을 수 있다. 양의 기울기 자체가 선택 규칙의 성능 개선이나 새로운 그래프에 대한 전이를 의미하지 않는다.

## 범위와 해석 제한

- PFN context 신뢰도는 새로 실행한 3-fold crossfit이다. 자신의 정답이 입력된 저장 context 예측을 validation으로 사용하지 않았다. LP의 exact leave-one-out 결과는 별도 context.csv에 있다.
- Fold마다 quantile/PCA 전처리를 reduced-context membership으로 다시 계산했다. 원래 raw imputation은 라벨을 사용하지 않으므로 고정해서 재사용했다. 모든 predictor 가중치는 frozen이다.
- 관측 행은 seed별 소수뿐이다. 복원 오차를 union LP로 query에 전달한 값은 개발용 proxy이고, 원래 결측 노드의 복원 가능성에 대한 보장은 없다. 관측과 결측 사이의 분포 차이, 재가림으로 줄어든 source 수, 전파로 인한 평활화가 영향을 준다.
- PCFI/FISF 간 JS와 LP native-view 간 JS는 서로 다른 view 수와 생성 과정을 비교하는 휴리스틱이다. 공통 보정 척도로 확립되지 않았다.
- 세 seed는 같은 그래프와 중첩 test를 공유한다. 독립 그래프 표준오차나 일반화 신뢰구간을 계산하지 않는다. k=1~3, 관측 피처가 전혀 없는 그래프, 새로운 schema/실제 그래프, MNAR 결측은 평가하지 않았다.
- Accuracy가 기존 비교의 주 지표다. 클래스 불균형과 확률 보정을 함께 점검할 수 있도록 balanced accuracy, macro-F1, NLL, Brier를 metrics.csv에 함께 저장했다. Context의 balanced sampling과 test 분포 차이는 신뢰도 추정에 영향을 줄 수 있다.
- 추가 비용: 원래 좌표 imputation 12회, frozen PFN crossfit forward 72회(각 10 ensemble members), CPU LP 진단. 최종 query prediction은 기존 full-context bank를 사용한다. 추가 비용을 허용한 진단이며 Gate D의 동일 예산 비교가 아니다.
- 실행 중 전처리 재현 검사가 CPU thread 수 차이를 검출해 원래의 8개로 맞춘 뒤 다시 추론했다. 따라서 실제 PFN forward는 폐기한 디버깅 14회를 포함해 86회이며, 원래 좌표 복원도 완료 12회 외에 중단 1회가 있다. 계산 예산의 계획과 실제 사용 차이는 execution_notes.json에 기록했다.
- Gate C selector 학습, 양방향 모델 추가 학습은 실행하지 않았다. 높은 AUROC나 oracle gap만으로 다음 단계 통과를 선언하지 않는다.

재현: recovery.py → crossfit.py → signals.py → analyze.py → audit.py → plot.py. 자세한 per-seed 결과는 results/*.csv, 고정 규칙은 protocol.json, 구현/입력 해시는 각 단계 metadata에 있다.

# 비선형 size 주입: 실제 composition 회복 확인, count 외삽 실패도 발견

2026-11-05. 사용자 승인 후 세 관점의 계획·코드 리뷰, 개발 전용 수치 검증, source/analysis freeze, 새32개 독립 target 평가와 저장 결과 감사를 완료했다. 연구 목표는 `RESEARCH_GOAL.md`의 문제–해결 연결이다. 특정 새 kernel의 기여를 요구하거나 주장하지 않는다.

## 판정

**사전에 정한 주 비교와 모든 gate가 통과했다.** 같은 h·관측 subset·잡음에서 nonlinear count 효과만 바꾸면 실제 composition 추정과 같은-count 선택이 크게 손상될 수 있고, 올바른 size 함수족을 joint fitting하면 회복할 수 있다는 근거가 강화됐다.

동시에 **고차 곡선의 관측 범위 밖 외삽은 count 선택을 크게 망칠 수 있다.** 구성 회복과 count 회복은 분리해야 한다. 기존 d3를 무조건적으로 더 좋은 범용 정책이라고 주장하지 않는다.

## 실행과 검증

- n16, archive44, 선형 zero-sum composition,4개 size shape, amplitude0/1/4, wide/narrow × uniform/aligned archive4종.
- 새 IID target32개, 논리 dataset1,536개, fixed7/EB6 arms로 저장 record19,968개. a0와 oracle fit의 재사용은 계획대로 수행했다.
- Fixed model API fit7,040회, EB fit5,888회(두 starts 유지). 4CPU,37.46초. Online/비싼 objective 호출0회.
- 모든 source·원본 SSPd·analysis hash 유지. 기존 확인40 bank 및 이전 GP bank는 사용하지 않았다.
- Raw record 전체의133,632 scalar 지표를 독립 계산과 대조, 최대 차이2.84e-14. 같은-prior primal/dual 검증은 각 joint fit에서도 통과했다.
- 사후 size 보정의 item coefficient는 원래 fit과 정확히 동일했다. Saturated-count 대조의 임의 size 주입에 대한 contrast 반응은 최대8.68e-15였다.
- Primary selected EB 실패0/128. Primary 밖에는 tied3의 selected failure가 unique fit2개 있었고, finite 결과를 삭제·재시작하지 않았다(캐시 공유까지 논리 행으로 세면5개 flags). 원본 fitter 상태를 이번 cycle에서는 실제로 기록했다.

`analyze.py`의 최종 console print에서 NumPy bool 직렬화 오류가 있었으나 결과 JSON/CSV 저장과 source 재검사는 그 전에 완료됐다. Frozen source·결과를 바꾸지 않고 독립 reviewer 및 `inspect_results.py`로 검증했다. 상세는 `ANALYSIS_CONSOLE_NOTICE.md`. 재실험이나 지표 변경은 없었다.

## 주 비교: wide archive, cubic, amplitude4, full EB

각 target에서 두 wide archive를 평균한 뒤32 target을 paired bootstrap했다. 같은-count normalized regret는0이 최선이다. Count2..12에서 실제 최선·최악 composition 차이로 정규화한다. 따라서 아래는 AUC 개선율이 아니다.

| Arm | Same-count regret | Composition coefficient MSE | Pairwise ordering accuracy |
|---|---:|---:|---:|
| item | 0.457854 | 7.141715 | 0.512928 |
| postfit cubic | 0.457854 | 7.141715 | 0.512928 |
| SSPd1 | 0.500896 | 3.336550 | 0.495384 |
| tied-scale joint cubic | 0.003264 | 0.010963 | 0.969366 |
| SSPd3 | **0.002959** | **0.010460** | **0.969965** |
| oracle size subtraction | 0.002374 | 0.009287 | 0.971408 |

위 oracle arm은 진단 전용이며 실제 사용 가능한 방법과 구별한다.

| Gate | 결과 |
|---|---|
| SSPd1−SSPd3 regret 차이 | +0.4979367,95% CI[0.4725735,0.5235350],32/32 개선 |
| MSE 차이 | +3.3260897,95% CI[2.9762324,3.7049177],32/32 개선 |
| Primary selected optimizer failure | 0/128 |
| Fixed broad1/broad3 size-response² | 10.2248595 / 3.1275e-12 |
| 동치·postfit·saturated 불변성 검사 | 통과 |

SSPd1의 a0→a4 regret 증가는+0.4985224,95% CI[0.4731459,0.5241152]였다. 같은 자료에 비선형 size 효과를 넣자 실제 구성 선택이 손상됐고, joint cubic은 거의 oracle composition 수준으로 회복했다. 이는 기존 oracle perturbation 진단보다 강한 실제 fitted-model 증거다.

## 해석의 핵심

1. **비선형 size 처리의 실질적 가치가 있다.** Size-only 함수가 고정 count에서 상수라는 이유로 composition에 영향이 없다고 볼 수 없다. Fitting 과정이 중요하다.
2. **단순 joint cubic도 거의 해결한다.** Tied3도 primary를 거의 회복한다. 이번 실험만으로 learned EB size scale의 필수성이나 복잡한 추가 방법의 우월성을 주장하지 않는다. 표준 joint 모델이 충분하면 그것이 적절한 간단 해법이다.
3. **이번 primary는 함수족 misspecification을 직접 다룬다.** Cubic은 d3의 span 안, d1의 span 밖이고 높은 signal이다. 원래 선형 논문의 prior-scale mismatch만을 분리 증명한 것은 아니다. Fixed-hyperparameter 대조는 label response를 분리하지만 full EB 결과에는 출력 SD와 실제 prior scale의 변화도 포함된다.
4. **새 estimator 발명이 아니라 문제–해결 연결의 강화다.** 기존 appendix의 고차 basis나 matrix 식을 새 정리로 바꾸지 않는다. 이전 tomography+1.31%의 실용 근거는 유지되지만 그 효과의 원인을 이번 synthetic 결과만으로 확정하지 않는다.

## 부정적 결과와 적용 한계

### 고차 항이 불필요하면 약간 손해

Size 효과가 없는 wide 조건: d1 regret0.002374, d3 0.002709. 선형 size a4의 wide 조건에서도 d1 0.002383, d3 0.002706이다. 기존 EB가 불필요한 차수를 항상 완전히 끈다고 주장하지 않는다.

### 모델 밖 shape에서는 composition 회복이 불완전

Wide+aligned+step+a4에서 d1 regret0.169149, d3 **0.232292**로 악화했다. 전체 wide step 평균도 d1 0.273598 대 d3 0.277384다. Sigmoid wide에서는 d1 0.181252→d3 0.076472로 줄지만 oracle0.002374와 큰 차이가 남는다. Cubic 하나가 모든 nonlinear shape를 해결하지 않는다.

Narrow는 관측 count5..8의 네 점뿐이다. Cubic은 이 네 count의 임의 값을 보간할 수 있다. Narrow step에서 구성 회복이 좋아도 관측하지 않은 count의 size curve가 맞다는 근거가 아니다.

### 같은-count 구성 회복과 count 선택은 다르다

Narrow sigmoid+a4에서:

| Archive | SSPd1 same-count regret | SSPd3 same-count regret | SSPd1 free-count utility regret | SSPd3 free-count utility regret |
|---|---:|---:|---:|---:|
| uniform | 0.044948 | **0.001650** | 0.029724 | **0.826132** |
| aligned | 0.047783 | **0.011482** | 0.028760 | **0.827834** |

SSPd3는 두 regime의32개 target **모두에서 관측 범위 밖 K2**를 선택했다. Size curve가 정확한 oracle arm은 K10/11을 선택한다. D3의 utility regret 중 count-envelope 성분은0.825954이며 composition 성분은 uniform0.000178,aligned0.001879에 그친다. 이 실패는 same-count composition 회복만으로 자유 count 선택의 성공을 보장할 수 없음을 보여준다. Count diagnostic은 저장된 사전 secondary 결과를 분해한 사후 설명이며 새 fit·objective 호출은 없다.

## 다음 결정

현재 SSPd3를 nonlinear size 주축으로 유지하되, 이를 무조건적인 free-count 기본 정책으로 확대하지 않는다. 다음 메소드 질문을 구체화한다.

- Step/sigmoid처럼 cubic 밖의 size shape를 안정적으로 joint fitting할 수 있는가?
- 관측 count 범위 밖의 예측을 과신하지 않으면서 어떤 count를 추가 관측해야 하는가?
- 유연한 size nuisance가 composition을 흡수해 정보를 잃는 대가와, 외삽·정규화 오류를 줄이는 이득을 어떻게 관리하는가?

이는 임의의 새 커널 필요성을 주장하는 질문이 아니다. 유연한 count 모델, count별 강건 추정, 단순 count spread 및 정보 획득을 같은 조건에서 대조할 명확한 실패 사례를 얻었다. 후속 후보는 별도 사전 계획과 새 개발/확인 bank에서 평가한다. 이번 bank를 degree·policy 재튜닝에 사용해 확인 성공으로 재보고하지 않는다.

## 리뷰 상태

Theory reviewer는 전체19,968 raw행의 top-K regret/MSE와32-target paired bootstrap을 별도 재계산해 확인했다. Empirical reviewer도 전행 coverage·CSV·hash·지표·bootstrap을 독립 확인했다. Scientific reviewer는 primary의 actual recovery와 단순 joint cubic의 충분성, misspecified-shape·외삽 실패를 함께 검토했다. 개별 결과 리뷰는 이 디렉터리의 `RESULT_REVIEW_*.md`에 보존한다.

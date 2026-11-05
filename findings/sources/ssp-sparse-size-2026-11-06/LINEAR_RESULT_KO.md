# Corrected tomography 선형 메커니즘: 실용 개선과 기전 실패를 함께 기록

완료일: 2026-11-06. 사전 명세·세 독립 plan/code 리뷰·별도 개발 자료·새 확인 자료·원자료 감사·동일 세 관점의 결과 리뷰를 완료했다. 원본 SSPd3(E)를 메인 메소드로 유지한다. **SSPd3의 실제 same-count 선택 개선은 확인됐지만, 제안한 양의 누출 방향과 association 조절 효과는 확인되지 않았다.** BOCS breadth 실험으로 자동 진행하지 않았다.

## 무엇을 새로 확인했나

기존 corrected tomography 확인 실험은 SSPd1 대 SSPd3 등 size-aware 변형끼리의 비교였다. 과거 d0/d1 산출물은 남아 있지만 historical physical operator를 사용했으므로 이번 질문의 근거로 재사용하지 않았다. 이 cycle은 corrected operator에서 **centered isotropic d0 대 변경 없는 SSPd1·SSPd3**를 처음으로 직접 비교하는 공백을 메웠다. 과거 원고의 uncentered/proper-intercept baseline 또는 budgeted ROI 표를 corrected operator로 재현한 것은 아니다. 역사적 산출물은 변경하지 않았다. 상세 출처는 [EVIDENCE_AUDIT_EMPIRICAL.md](EVIDENCE_AUDIT_EMPIRICAL.md).

새 IID tomography scenario 24개, item 250개, archive 관측 400개를 사용했다. U/W/S는 같은 count schedule과 결합 난수로 만든 uniform/약한/강한 item–count association archive다. BO archive는 isotropic d0 UCB adapter로 수집한 기술 보고용 자료다. 평가 count는 K=4/12/24/40, 각 count의 독립 후보 768개다. Scenario 안에서 네 count를 같은 가중치로 평균한 후 24개 독립 scenario를 paired bootstrap 20,000회로 집계했다.

| Arm | 모델 | 역할 |
|---|---|---|
| A | 원본 SSPd의 d=0 centered isotropic prior | 주 비교 기준 |
| B | 원본 SSPd1, native EB | 1차 count 분리 대조 |
| C | 원본 SSPd3, native EB | 현재 메인 메소드 |
| R | 독립 uniform reference-count 평균으로 잔차를 만든 뒤 A의 고정 hyperparameter로 composition 재추정 | 추가 관측을 쓰는 기술 진단 |

R은 사후 count-only 함수를 더하는 arm이 아니다. Count 평균을 training response에서 뺀 뒤 composition을 다시 추정하므로 같은-K 순위가 달라질 수 있다. 단순한 사후 count-only 덧셈의 순위 불변성은 unit check로만 검증했다. Native A/B/C는 원본 fitting을 그대로 호출했으며 gate를 위해 고정 scale로 바꾸지 않았다.

## 사전 판정과 실제 수치

아래 구간은 scenario paired-bootstrap **양측 95% CI**다. 실제 gate는 사전 명세한 단측 95% 하한을 사용한다. Regret 수치는 normalized objective의 절대 단위이며 AUC 퍼센트가 아니다.

| 사전 질문 | Strong(S) 확인 결과 | 판정 |
|---|---|---|
| G1_requested: A−C composition 계수 차이가 d와 양의 정렬인가? | cosine −0.263741, CI [−0.297600, −0.229922], 24/24 음수 | 실패 |
| G1_error: A의 실제 방향 오차가 양수이고 C가 절대 오차를 줄이는가? | A −0.00752811 → C −0.00004189; 절대 방향오차 감소 +0.00675393, CI [+0.00553829, +0.00799449] | 양의 부호 조건 실패; 절대 오차 감소는 관찰 |
| G2: A−C same-K regret 차이가 양수이고 평균 최소효과 0.001 이상인가? | +0.00638740, CI [+0.00255181, +0.01003663], 18/24 개선 | 통과 |
| G3: regret 개선이 U보다 S에서 더 큰가? | S−U = −0.00459538, CI [−0.00925639, −0.00002658] | 실패 |

G2의 단측 95% 하한은 +0.00318388이다. 반면 G1의 부호를 뒤집거나 절댓값만으로 대체해 전체 메커니즘 성공으로 선언하지 않는다. 부분 gate·보조 지표의 개별 해석은 nominal 수준이며 별도의 다중검정 보정된 발견이 아니다.

| Archive | A d0 regret | B d1 regret | C d3 regret | A−C 개선, 95% CI |
|---|---:|---:|---:|---|
| U | 0.033101 | 0.030815 | 0.022119 | +0.010983 [+0.007663, +0.014332] |
| W | 0.029484 | 0.029830 | 0.024670 | +0.004813 [+0.001527, +0.007912] |
| S | 0.028936 | 0.026001 | 0.022548 | +0.006387 [+0.002552, +0.010037] |
| BO, 기술 보고 | 0.031035 | 0.030919 | 0.023011 | +0.008024 [+0.004936, +0.011165] |

U에서도 개선이 있었고 평균 개선은 S보다 컸다. 유한한 uniform archive는 정확한 직교/null control이 아니다. 이번 결과를 association 강도에 대한 보편적 역단조성으로 일반화하지 않는다.

![선택 regret, 정렬, 실제 방향오차, paired 개선](linear_mechanism.png)

그림의 arm별 오차막대와 paired 차이의 오차막대는 다른 추정량이다. PDF/SVG도 함께 저장했다. 그림을 직접 열어 축·범례·문구의 잘림 및 겹침이 없음을 확인했다.

## 어떤 메커니즘이 확인됐고 무엇이 남았나

실제 방향 오차는 held-out subset의 **예측−실제 목적함수**를 training-defined d 방향의 centered/unit-SD score에 투영한 값이다. 음수는 그 score가 큰 subset을 상대적으로 과소예측한다는 뜻이다. 모든 item의 진짜 효과가 알려졌거나, 전체 목적함수 평균을 과소평가한다는 뜻은 아니다. 비선형 tomography에는 사전 정의된 유일한 K-independent true linear coefficient가 없다.

S의 A 방향 오차는 네 K 모두 평균 음수이고 scenario 24개 모두 음수였다. K 평균에서 상쇄되어 우연히 반대 부호가 된 결과가 아니다. C는 측정한 절대 방향 오차를 U/W/S 각각 24/24 scenario에서 줄였다. 따라서 “왜곡이나 회복이 없다”는 결론 역시 부정확하다. **예상한 과대평가 기전은 실패했지만, 해당 투영의 실제 과소예측 감소와 선택 개선이 함께 관찰됐다.** C 평균이 0에 가까운 것은 관찰 결과이며 등가성 검정 통과라는 뜻은 아니다.

고정된 공통 hyperparameter에서 d0와 d1의 차이가 rank-one 방향 d에 놓인다는 식은 임의의 response에 대한 Gaussian posterior 항등식이다. 이번 native 비교는 각 모델이 τ·σ를 따로 추정하고 C는 세 count basis를 사용한다. 그 차이는 고정 scale의 1차 항등식과 동일한 estimand가 아니다. 부호 반전은 그 항등식의 반례가 아니며, 항등식 자체가 native fits의 진짜 예측 오차 부호를 보장하지도 않는다. 이 구별은 확인 자료를 만들기 전에 [APPLICABILITY_THEORY.md](APPLICABILITY_THEORY.md)와 [PLAN.md](PLAN.md)에 기록했다.

또한 S에서 A−B(d1)의 regret 개선은 +0.00293482, 95% CI [−0.00059661, +0.00650825]이고 단측 하한도 −0.00001038이다. **1차 count 분리만으로 실용 개선이 확인됐다고 주장할 수 없다.** A−C의 이득에는 고차 count basis와 EB 재추정이 함께 들어간다. S의 τ 중앙값은 A 약 2.221, B 0.765, C 0.194였지만 이 관찰만으로 τ가 원인이라고 단정하지 않는다. 이번 자료는 어떤 변화가 부호와 regret 개선을 매개했는지 유일하게 식별하지 않는다.

BO archive에서는 예상한 양의 패턴이 관찰됐다. A−C alignment +0.645007, A 실제 방향오차 +0.02593065 → C −0.00055631이다. 그러나 사전에 기술 보고로 고정한 archive이므로 S gate의 대체재가 아니다. A 정책이 수집한 자료에서의 사후 surrogate 비교이며, C로 online 정책을 바꿨을 때 incumbent trajectory가 개선된다는 확인 실험도 아니다.

## Reference 진단의 한계

Reference는 각 K=2..40에서 추가 uniform subset 32개로 얻은 Monte Carlo 평균이다. Count별 평균 sampling SE는 약 0.00171~0.00429, 전체 평균은 0.002914다. Full32와 독립 half16 reference를 사용했을 때 추천 일치는 578/768=75.26%였다. 이는 목적함수 관측 noise 추정치가 아니라 reference 평균의 유한 표본 오차다.

R의 S regret은 0.024496으로 C의 0.022548보다 좋지 않았다. R은 정확한 population count curve를 아는 oracle도, native budget에 맞춘 경쟁 arm도, regret 상한도 아니다. Count-only 함수를 사후에 더해서 추천이 바뀐 것이 아니라, reference 차이가 composition 재추정을 통해 반영된 것이다. 이 결과로 SSPd3의 count 추정이 모든 실전 상황에서 충분하다고 결론내리지 않는다.

## BOCS 제안에서 고쳐야 할 사실

공식 원문과 pinned source commit `fef0d4e34e376e8bb0dae9955d70c2155530b9eb`을 독립 확인했다. 자세한 링크와 코드는 [BOCS_SOURCE_AUDIT.md](BOCS_SOURCE_AUDIT.md)에 있다.

1. **Native horseshoe는 현재 isotropic Gaussian 정리의 직접 적용 대상이 아니다.** Conditional covariance는 coefficient별 local scale이 다른 diagonal이고, scale을 적분한 prior의 2차 모멘트는 유한하지 않다. “독립 shrinkage이므로 같은 정리가 바로 적용된다”는 추론은 성립하지 않는다. 고정 scale 조건부 일반화와 native scale adaptation의 실증을 구분해야 한다. [BOCS 원문 §3.1](https://proceedings.mlr.press/v80/baptista18a/baptista18a.pdf)
2. **Ising의 알려진 외부 λK는 공식 구현에서 이미 분리된다.** Surrogate label은 KL이고 acquisition/report에 penalty를 따로 더한다. KL 자체의 unknown size trend가 남는지는 별도 질문이다. λK를 억지로 label에 넣어 native BOCS를 불리하게 만들면 안 된다. [공식 BOCS.m](https://github.com/baptistar/BOCS/blob/fef0d4e34e376e8bb0dae9955d70c2155530b9eb/algorithms/BOCS.m#L68-L81)
3. Contamination은 추가 λK만 분리하고 기본 처리비 K는 model label 안에 남는다. 모든 알려진 비용을 정확히 뺀 단순 residual-learning 대조가 필요하다. 이를 unknown size prior 교정과 구별해야 한다. [공식 contamination wrapper](https://github.com/baptistar/BOCS/blob/fef0d4e34e376e8bb0dae9955d70c2155530b9eb/test_problems/ContStudy/contamination_prob.m#L18-L22)
4. 공식 MATLAB/Python은 binary 대각 중복열 등 basis 처리가 다르고, native 추론은 MAP가 아닌 Gibbs draw를 사용한다. Intercept·known-cost 처리·basis·sampler·acquisition을 고정한 별도 재현 protocol이 필요하다. K와 K(K−1)/2는 원래 2차 monomial 함수공간에도 있으므로 저차 count 항 분리는 표현력보다 prior 구조의 변경이다. [공식 저장소](https://github.com/baptistar/BOCS)

따라서 현재 바로 2~3주 breadth 실험을 시작하는 근거는 확보되지 않았다. 이것은 표준 benchmark 검증의 가치가 없다는 결론이 아니다. **Native BOCS가 같은 기전으로 실패한다는 전제와 단순 비용 분리 대조부터 바로잡아야 한다.** 표준 benchmark 두 개의 개선만으로 ICML 수용이나 보편적 실사용 관행의 결함이 자동 입증되는 것도 아니다.

## 논문에 남길 주장과 다음 범위

현재 지지되는 문장은 다음과 같다.

> Fresh corrected tomography의 명세한 archive와 finite same-K catalog에서, 원본 SSPd3는 matched centered isotropic prior보다 선택 regret과 측정한 절대 방향오차를 줄였다. 다만 예상한 양의 1차 누출 방향 및 association에 따른 개선 증가는 확인되지 않았다.

이는 실제 task의 새 naive 비교 근거다. 이전 d3 대 d1의 확인된 online 이득을 유지하되, 이번 고정 archive 결과와 endpoint를 합쳐 하나의 online 효과처럼 쓰지 않는다. 실패한 외부 kernel 범용화, EB 해로움 가설, 식별 rank로 성능을 예측한 주장도 그대로 실패 기록으로 남긴다.

제목은 **Lost and Leaked** 또는 **Separating Size and Composition Priors**가 적절하다. 이번 cycle은 uncertainty calibration을 평가하지 않았으므로 Calibration이라는 제목을 쓰지 않는다. 전체 기여는 새 독립 kernel 유무가 아니라, size/composition 문제와 실제 추정·선택 회복의 연결로 판단한다.

다음 연구 질문은 부호를 바꿔 같은 gate를 재실행하는 것이 아니다. 새 자료를 쓰는 adaptive archive 확인 또는 native horseshoe의 조건부 prior 개입 중 어느 질문이 논문 주장에 필요한지 먼저 정해야 한다. 특히 BOCS는 위의 native-fidelity·known-cost 대조를 반영한 별도 계획이 필요하다. 이번 cycle은 G1 실패를 보존하고 종료하며 추가 online·BOCS fit은 실행하지 않았다. 원인 매개가 미식별인 상태에서 새로운 원인 설명을 확정하지 않는다.

## 검증과 비용

- 확인 본 실험: physical label 142,080회, optimizer-bearing fit 1,368개, 4 CPU worker, wall time 388.99초. 별도 개발 2 scenario는 11,840 label·114 fit이며 확인 자료와 분리했다.
- 확인 후 physical audit 1,248회, 확인+감사 합계 143,328회로 사전 145,000 ceiling 이내다. 새 확인 자료를 보고 hyperparameter·gate·arm·seed를 변경하지 않았다.
- 최종 native A/B/C 288 fit은 모두 converged. 전체 중 BO-prefix 선택 fit 8개는 finite line-search 미수렴 상태 그대로 보존했으며 기술 BO archive에 영향을 준다. 전체 최적화가 모두 수렴했다고 보고하지 않는다. d0의 원본 두 호출은 같은 시작점이라는 구현 사실도 유지했다.
- [AUDIT.json](AUDIT.json): sampled physical replay 최대오차 0, fixed-hyper replay 9.02e−17, raw metric replay 0, 384 final model 재구성, 8,640 saved-score BO 선택 replay. BO posterior 재구성은 batch0/44와 각 16 pool point 표본 범위다. 감사의 metric replay는 같은 등록 함수이며 독립 리뷰의 별도 산술 검산과 구분했다.
- 세 reviewer가 확인한 독립 raw/CSV/bootstrap 오차는 부동소수점 범위이며, 별도 27개 raw-item covariance posterior 재구성의 예측 오차는 최대 1.19e−13이다. Norm-invalid 제외 사례는 0개다. 모든 archive는 보고한 linear composition full rank다.
- 이전 artifact 12,317개와 frozen source 25개의 hash를 보존했다. 루트에 Git 저장소가 없으므로 branch/commit을 만들었다고 보고하지 않는다. Manifest와 completion hash로 이 workspace의 산출물을 고정한다.

독립 최종 리뷰: [이론](RESULT_REVIEW_THEORY.md), [실험·통계](RESULT_REVIEW_EMPIRICAL.md), [기여·선행](RESULT_REVIEW_NOVELTY.md). 재현 입력과 gate는 [MANIFEST.json](MANIFEST.json), [SUMMARY.json](SUMMARY.json), [per_scenario.csv](per_scenario.csv)에 있다.

# 다른 분야의 해결 원리: 구조적 신호와 regularization 오염

2026-11-06. 사용자는 다른 분야라도 같은 원리로 문제를 해결한 연구를 찾으라고 명시했다. 아래는 1차 원문·metadata를 확인한 문헌과 SSP로의 analyst 대응이다. 새 fitting·objective 호출이나 기존 실험 변경은 없으며, 이번 문헌을 canonical XLSX에 추가했다고 보고하지 않는다.

## 가장 가까운 해결 갈래

| 갈래·원문 | 기존 분야의 문제와 해결 | SSP로 가져올 원리 및 경계 |
|---|---|---|
| Regularization-induced confounding — Hahn·Carvalho·He·Puelz, [원문](https://arxiv.org/pdf/1602.02176) | Observational treatment-effect 추정에서 control 계수를 과도하게 수축하면 treatment 계수가 편향된다. Simultaneous regression으로 prior를 재설계한다. | Nuisance의 regularization이 관심 효과에 전달되는 메커니즘 자체가 매우 가깝다. Count/composition과 treatment/control은 다른 estimand이며 causal 가정을 BO에 그대로 이전하지 않는다. |
| Bayesian causal forests — Hahn·Murray·Carvalho, [원문](https://arxiv.org/pdf/1706.09523) | Regularization과 confounding을 구별하고 propensity estimate를 response model에 포함하며 prognostic/treatment-effect 구조를 분리한다. | 관측이 어떻게 수집됐는지를 반영한 prior/parameterization의 근거. BO archive association을 고려할 동기지만 propensity 하나로 subset ranking이 보호된다는 정리는 아님. |
| Spatial+ — Dupont·Wood·Augustin, [원문](https://arxiv.org/pdf/2009.09420) | Spatial effect의 smoothing이 correlated covariate 효과를 편향시킨다. Covariate에서 spatial dependence를 제거한 residual과 spatial term을 함께 모델링한다. | Φ−E[Φ|K]와 count 모델을 함께 쓰는 설계에 대응시킬 수 있다. 이는 analyst 전이이며 원문이 subset BO를 다룬 것은 아님. 완전 collinearity면 남는 정보가 없고 target/효율이 바뀔 수 있다. |
| Functional horseshoe — Shin·Bhattacharya·Johnson, [원문 §3](https://arxiv.org/pdf/1606.05021); smooth extension — Wiemann·Kneib, [원문](https://arxiv.org/pdf/2101.05630) | 개별 spline 계수를 0으로 보내기보다 함수가 지정된 parametric subspace에서 벗어난 정도를 shrink한다. 후자는 smoothness도 함께 제어한다. | Count basis를 의미 있는 함수 구조로 취급하는 직접 인접 선행. 전체 함수가 pure count에 가깝도록 수축하면 composition도 함께 줄 수 있으므로 native HS contrast 보존 모델과 같다고 부르지 않는다. |
| Nonlinear functional shrinkage — Duda·Wheeler, [2024 preprint](https://arxiv.org/pdf/2407.17113) | 생물학적 dose–response에서 Hill 같은 nonlinear 형태를 기본 구조로 사용하면서 데이터가 요구하면 그 형태를 벗어나게 한다. Taylor/Jacobian으로 nonlinear function family를 locally 근사한다. | 포화·diminishing-return count 곡선을 단단하게 고정하지 않고 구조와 유연성을 함께 주는 후보 원리. 원문의 local approximation은 모든 nonlinear count curve에 대한 exact invariance 보장이 아니다. |
| Pooled HS / dense+sparse — Clark 등 [원문 pp.7–8](https://arxiv.org/pdf/2110.03411); Chernozhukov·Hansen·Liao [LAVA 원문](https://arxiv.org/pdf/1502.03155) | Pooled HS는 Gaussian 공통 mean 주변의 sparse deviation, LAVA는 dense+sparse 계수 총합을 서로 다른 penalty로 추정한다. | Shared count+HS residual의 직접 구조 선행과 강한 단순 대조다. LAVA의 임의 dense 성분과 두 count 방향은 다르고, 잠재 분해 비고유성을 실제 예측·결정과 구별해야 한다. |
| Double/debiased ML — Chernozhukov 등, [발표 논문](https://doi.org/10.1111/ectj.12097), [읽은 preprint v4](https://arxiv.org/pdf/1608.00060v4) | Nuisance ML의 regularization/overfitting이 관심 parameter에 들어오는 효과를 Neyman-orthogonal score와 cross-fitting으로 줄인다. | Nuisance를 잘 예측하는 것과 target을 잘 추정하는 것을 분리하는 설계 원리다. 일반적으로 local/first-order robustness와 조건부 점근보장이며, finite-sample same-K posterior 불변성·BO regret 보장과 같지 않다. |

표는 RIC/BCF를 하나의 원리, functional/NLFS를 구조적 shrinkage의 두 확장으로 묶어 읽을 수 있다. 논문 수를 맞추기 위해 동일한 해결책으로 합치지 않는다.

## 중요한 수학적 대응

일반적인 부분선형 모델을

    y = g(K) + φ(x)'β + ε

로 쓰면 nuisance g의 오류가 β로 전달되는 정도는 φ와 K의 관측상 관계에 의존한다. 공간통계의 location과 correlated covariate 관계가 이 구조와 유사하다. Fixed count basis G에 대한 exact residualization은 y와 Φ를 함께 Q_G로 변환하는 기존 FWL/flat-nuisance 처리다. Estimated smooth conditional means를 쓰는 Spatial+/DML 방식은 nuisance 추정 오차·cross-fitting·모형 가정까지 추가로 다뤄야 한다. 기존 projection 실험의 음성 결과 때문에 이 전이만으로 regret 개선을 전제하지 않는다.

Functional shrinkage에서는 원문의 대표 penalty가 β'Φ'(I−Q0)Φβ, 즉 의미 있는 함수 subspace에서 벗어난 부분이다. Count 방향을 개별 monomial의 sparse support가 아니라 알려진 낮은 차원 함수 구조로 취급한다는 관점에 직접 도움이 된다. 다만 count subspace 보호와 composition sparsity의 동시 보존은 별도 설계 문제다. β+Hγ의 prior contrast 보존 항등식과 이 함수 penalty는 서로 다른 모델이다.

## 우선순위

1. **문제 메커니즘:** RIC와 Spatial+를 먼저 읽는다. 수축/평활화 오류가 다른 효과에 전달되는 과정과 보정의 estimand를 명확히 한다.
2. **해결 모델:** Pooled HS를 최소 대조로 인정하고, functional horseshoe를 count 구조를 다루는 더 일반적인 관점으로 검토한다.
3. **Nonlinear count:** NLFS와 smooth functional shrinkage를 읽되, 현재 증거 없이 full nonlinear 모델 구현으로 확대하지 않는다. 작은 알려진 quadratic count 개입으로 필요성을 먼저 확인한다.
4. **보장:** DML의 orthogonal-score 원리를 참고하되 ranking functional·적응 archive·작은 N의 가정을 새로 검토한다. 예측/추정 보장을 결정 regret로 자동 대체하지 않는다.

이 문헌들은 다른 분야에서 실제 해결을 시도한 연구다. 따라서 일반적인 dense+sparse 분해나 nuisance 보호를 새로 발명했다고 주장할 수 없다. 반면 읽은 범위에서 이들이 sparse polynomial BO의 count-induced same-K 결정 손실과 교정을 입증했다고 확인하지도 못했다. 이 구별을 남기고, prior-art 부재·ICML novelty를 확정하지 않는다.

출판 연도·판본은 metadata와 실제 읽은 preprint를 구분했다. Pooled-HS의 구체적 hierarchy는 읽은 2021 preprint, NLFS는 2024 preprint에 근거한다. 새 canonical 논문 정리 파일 편집과 성능 실험은 이번 단계에서 하지 않았다.

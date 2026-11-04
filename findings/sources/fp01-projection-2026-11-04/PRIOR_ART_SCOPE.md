# FP01 prior-art and claim scope — before results

PLAN_v2와 기존 방법/실패 기록을 기준으로 작성했다. 최종 예측·부분 성능을 열람하지 않았으며 설정·모듈을 바꾸지 않았다. 결과의 성공 여부를 미리 판정하는 문서가 아니다.

**현재 C의 기여 후보는 Gaussian centroid와 BP의 결합 자체가 아니라, 관측이 분리된 조건에서 projection 노출 학습과 centroid family 확장의 추가 예측 가치를 분리해 검증하는 계산·실험이다.** 기존 연구와 같은 원리를 사용하는 것은 실험을 무가치하게 만들지 않지만, 이 cycle이 구현한 수치 근사와 얻은 효과의 범위로 주장을 제한해야 한다.

## 1. 기존 연구와 겹치는 부분

| Primary source | 이미 있는 내용 | FP01과 구분할 점 |
|---|---|---|
| Duranthon & Zdeborová, *Optimal Inference in Contextual Stochastic Block Models*, TMLR 2024 | Appendix D의 식 (88)–(90)과 Algorithm 3은 다중·불균형 class, Gaussian latent class-centroid vector, 일반 symmetric affinity, 준지도 AMP–BP를 명시한다. 본문의 parameter-estimation 절은 EM/Bethe evidence도 다룬다. [공식 원문](https://arxiv.org/html/2306.07948v2) | Class-specific Gaussian centroids, class clamping, graph–feature joint inference가 새 원리는 아니다. C는 해당 알고리즘의 재현이 아니라, input-dependent hyperparameters를 둔 41-state centroid-grid와 유한 class-gated BP다. 원문의 high-dimensional asymptotic 설정과 현재 masked/RP/clipped 입력을 동일시하거나 optimality를 이전할 수 없다. |
| Zhou, Li & Zhang, JSBM/BPGCN, PRR 2020 | Item/feature의 latent group를 공동 BP하고 finite BP를 backpropagation해 관계 parameter를 학습한다. [원문 §II–V](https://arxiv.org/html/1911.00197v2) | 기존 attribute likelihood는 Bernoulli group 관계다. 현재 Gaussian centroid law는 다르지만, “잠재 feature 상태 + BP + end-to-end 학습”만으로는 독립적인 새 원리를 주장할 수 없다. |
| Opipari et al., DNBP | Neural probabilistic factor를 differentiable BP와 결합해 학습한다. [원문](https://arxiv.org/html/2303.04616v1) | C는 continuous sample-based pose inference가 아닌 finite categorical graph prediction이다. 다만 differentiable factor learning이라는 큰 설계는 기존이다. |
| Müller et al., PFN | Synthetic task의 query NLL로 posterior-predictive approximation을 학습한다. Architecture는 Transformer로 한정되지 않는다. [원문 §3](https://arxiv.org/html/2112.10510v7) | 합성 query CE 및 새 graph에서 target gradient update가 없는 적용은 PFN의 기존 방향이다. 유한 모델/최적화가 정확한 Bayesian predictive를 얻었다는 보장은 아니다. |

CSBM 원문에서 **동일한 disjoint-mask/RP/zero-update 평가 전체**를 수행했다는 사실은 이번 확인으로 입증되지 않았다. 역으로 그런 선행연구가 전혀 없다고 주장할 수도 없다. 현재 문헌 검토는 집중적인 closest-prior 확인이며 exhaustive priority search가 아니다.

## 2. C를 정확히 부르는 법

- C는 **visible-statistic-conditioned finite-grid centroid predictor**다. Encoder는 class별 mean을 직접 출력하지 않고 column별 공통 location/tau/sigma를 출력한다. 각 class의 centroid state 차이는 class-gated BP의 belief에서 생긴다.
- C_global도 observed column mean을 center로 쓴다. “관측에 무관한 고정 prior”가 아니라 **3개 전역 학습 계수를 가진 관측조건부 모델**이다. C−C_global이 adaptive shared encoder의 추가 가치에 대응한다.
- Normal likelihood를 쓴다는 사실은 전체 posterior가 Gaussian이라는 뜻이 아니다. Class labels가 고정되면 centroid별 Gaussian conjugate 적분을 할 수 있지만, class가 미지이면 centroid로 가는 message는 `(1−rho_c)+rho_c Normal(x;mu,sigma²)` 형태이고 일반적으로 Gaussian message family에 닫히지 않는다. 이는 현재 식에서 직접 따르는 해석이다. 따라서 C를 exact Gaussian BP나 published AMP–BP의 다른 구현으로 부르면 부정확하다.
- 41 GH state의 exact sum, 그 finite-grid model에 대한 loopy BP 오차, 연속 Gaussian reference에 대한 grid 오차는 서로 다르다. 24-step/damping .5 연산을 미분한다는 것이 수렴한 posterior나 fixed-point gradient를 뜻하지 않는다.
- q(h)는 graph/context로만 계산해 모든 feature arm·개입에 고정한다. **Feature-conditioned joint role posterior를 새로 학습하거나 개선한 방법이 아니다.** No-observed가 공통인 것도 이 protocol의 일부다.

## 3. 결과가 나왔을 때 허용되는 결론

| 관측할 비교/효과 | 지지할 수 있는 주장 | 그 결과만으로 지지하지 않는 주장 |
|---|---|---|
| 같은 projected test input에서 B>A | 이 budget에서 projection-exposed synthetic training이 기존 preferred-class operator에 도움이 됐다. | 실제 실패의 유일 원인이 mean-shape mismatch였다. RP는 covariance/rank/SNR/clipping도 바꾼다. |
| 같은 projected interface에서 C_global>B | 더 넓은 centroid score와 그 finite inference 구성이 기존 preferred-class 구성보다 유리했다. | Gaussian centroid/BP의 최초 제안, 정확한 projected-law recovery, 표현력 변화만의 순수 효과. Factorization과 parameterization도 달라진다. |
| C>B 및 C>C_global | Shared adaptive hyperparameter encoder의 추가 예측 가치가 해당 설정에서 확인됐다. 사전 co-primary accuracy·CI·1%p·NLL·absolute-utility gate를 함께 판단한다. | 모든 graph/feature law에서 필요한 adaptation, foundation-scale 범용성, 정확한 centroid/posterior recovery. |
| C의 observed−none>0, probability/minority metrics도 개선 | 실제 관측 피처의 추가 가치가 해당 query·graph·mask 조건에서 나타났다. | 모든 PCFI 대비 개선이 feature에서 왔다. Structure-only 개선과는 구분해야 한다. |
| Full−shuffle>0 | 올바른 feature/node alignment에 민감하다. | Observed−none>0의 대체 증거. |
| 합성 primary 성공 | 새 seed의 같은 shape/scalar family에서 사전 기준을 충족했다. | Unseen family/schema OOD 성공 또는 independent real holdout 성공. |
| 기존 5 real graph에서 일관된 이득 | 재사용한 development graphs에서 유망한 transfer 신호다. | Graph 모집단의 유의성, 독립 real 일반화, 모든 imputation/PFN recipe 대비 우월성. |

같은 q/p0 때문에 `(C−none)−(B−none)=C−B`다. 이를 두 개의 독립 성공으로 세지 않는다. Full/edge는 사전 분리 panel이며 dataset별 winner를 선택해 하나의 성과로 합치지 않는다. Source 0/1 exchangeability 경계는 관측에 없는 class 정보를 무시한 실패로 읽지 않는다.

## 4. 결과와 무관하게 유지할 사실

K≥3의 one-preferred/equal-other mean family는 일반적인 RP에 닫히지 않는다. 이 수학적 사실은 C가 성능을 높이지 못해도 남는다. 반대로 C가 이기더라도 그것만으로 그 mismatch가 기존 real 실패의 유일 원인임이 입증되지는 않는다. C는 RP-induced covariance나 class-sum constraint를 완전히 모델링하지 않는다.

이전 binary feature-law pilot의 adaptive 우위 미확인은 그대로 보존한다. 그것은 multiclass projection closure를 시험한 결과가 아니며 FP01의 선험적 기각 근거도 아니다. 기존 Factor의 PFN 대비 이득도 새 C의 공으로 이전하지 않는다. B만 성공하면 interface-training evidence, C_global만 성공하면 전역 계수 centroid inference의 성과로 각각 보고할 수 있다.

현 단계에 맞는 기여 문장 예시:

> “우리는 graph-only role mixture를 고정한 disjoint-evidence 예측에서, projection 노출 학습과 class-specific finite-grid centroid inference가 관측 피처의 추가 가치를 높이는지 분리해 검증한다. 이 구성은 기존 CSBM/learned-BP 원리에 기반하며 정확한 연속 Bayesian inference나 최초의 centroid–BP 결합을 주장하지 않는다.”

제출용 방법 기여로 확대하려면 성공한 비교 효과가 먼저 있어야 한다. 결과가 음성이면 같은 실험은 관측 family의 수학적 repair가 feature utility 회복에 충분하지 않았다는 제한된 반례를 제공하며, Gaussian/centroid inference 전체의 불가능성 결론은 제공하지 않는다.

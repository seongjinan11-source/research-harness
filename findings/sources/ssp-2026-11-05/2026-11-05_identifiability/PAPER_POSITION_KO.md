# 논문 강화 판단과 다음 메소드 후보

2026-11-05. 현재 목표는 overlooked size/composition 문제와 충분한 해결의 연결이다. 새 일반 kernel이나 일반 통계 정리의 발명을 요구하지 않는다. 이번 문서는 논문 수정안과 미검증 메소드 후보를 구분한다.

## 현재 확보한 연결

1. 기존 SSP의 variance-share 및 retention/leakage 분석은 특정 kernel/prior가 size 신호와 composition을 어떻게 처리하는지 설명한다. 기존 appendix에 이미 있는 내용을 새 기여로 다시 세지 않는다.
2. 이번 fixed-archive 명제는 count 효과를 자유롭게 허용하면 데이터 자체가 어떤 composition 차이를 식별하지 못하는지 규정한다. M은 pooled within-count scatter이고 full-K contrasts 식별에는 rank(M)=n−1이 필요충분하다. N−m>=n−1은 필요조건이다. 차이가 남는 null 방향에서는 동일 관측법칙을 가진 두 환경을 구성할 수 있고, 어떤 추천 rule도 둘 모두에서 normalized simple regret1/2 미만을 보장하지 못한다.
3. 이 명제는 finite count prior가 주는 정보와 archive가 arbitrary count nuisance 아래 실제 제공하는 정보를 분리한다. 유연한 joint GP의 효용을 부정하지 않으며, cut의 이점이 실패한 기존 증거와 일치한다. 적절한 joint count fitting은 SSP의 유효한 해결 방식이다.
4. 기존 online 확인은 선택 회복의 실제 근거다. 이번 Gaussian simulation은 수학 구현 검증이며 online 효과나 실제 실패 예측의 독립 실증으로 더하지 않는다.

정당한 논문 구조는 **어떤 size 편향이 생기는가 → 어떤 경우 composition 결정까지 오염되는가 → 교정에 필요한 가정과 관측은 무엇인가 → joint correction 및 관측으로 실제 선택이 회복되는가**다. 이번 명제는 세 번째 연결과 한계를 보강한다. 단독 flagship 이론 신규성으로 주장하지 않는다.

## 실제로 수정할 다음 축: probe에서 무엇을 관측할지

현재 고정 빈도 spread probe는 count coverage를 확보한다. 그러나 그 count에서 composition을 greedy하게 고르면 결정에 중요한 item contrast가 계속 함께 움직일 수 있다. 이 상태에서 count flexibility만 늘려도 robust composition 정보가 충분해진다는 보장은 없다. 다음 후보는 유연한 joint fit을 유지하면서 **같은 probe count 안에서 결정에 중요한 composition 차이를 더 잘 구분하는 query**를 선택하는 것이다. 아직 구현·성능 확인하지 않았다.

구체적으로 B좌표의 관측정보 M과 A=M+lambda I를 추적한다. lambda>0는 numerical/regularization scale로 사전 고정하며, 이를 likelihood identification과 혼동하지 않는다. 현재 추천 x_star와 경쟁 subset x 사이의 contrast c=B'(x_star−x)를 고려한다. 기존 관측 t_K개가 있는 count K에서 새 query의 z=B'x_query−bar_u_K, a=t_K/(t_K+1)이면

    Delta V_c = a * (c' A^-1 z)^2 / (1 + a z' A^-1 z).

이는 V_c=c'A^-1c의 정확한 rank-one 감소량이다. Sherman–Morrison/optimal-design의 표준 식이며 새 수학이라고 주장하지 않는다. 추천을 바꿀 수 있는 경쟁 방향들의 불확실성 감소를 query 후보 평가에 쓸 수 있다. 어떤 competitor 집합과 가중치를 쓸지, count 선택과 어떻게 합칠지는 새 계획/리뷰로 고정해야 한다. 기존 probe보다 좋아 보이게 결과를 보고 정하지 않는다.

첫 신규 count에서 t_K=0이면 이 robust composition score는0이다. 그 count의 query는 count 선택을 개선하는 가치로 정당화될 수 있다. 이 구분은 새 count query를 금지하는 규칙이 아니다. 실제 count GP의 credible variance와 arbitrary-nuisance robust information도 같은 양이 아니다.

## 가장 먼저 반증할 비교

새 개발/확인 cycle에서 같은 joint model, 초기 archive, 호출 예산과 사전에 고정한 count schedule을 공유하고, probe의 within-count composition만 비교하면 위 가설을 분리할 수 있다. 기존 mean composition, 균일 random composition, 표준 experimental-design 선택, 위 decision-directed 선택이 필요한 단순 대조군이다. 단순 설계가 충분하면 그것을 채택한다. Adaptive count 정책의 end-to-end 확인은 이 기전 비교와 분리해야 한다.

성공하려면 정보 행렬 수치 개선뿐 아니라 **실제 같은-count 추천 손실과 online 추천 손실**이 줄어야 한다. 낮은 regret의 안전한 후보를 골라야 하는 exploitation 비용도 포함한다. 현재 fixed-frequency probe가 joint mean보다 나쁜 shape/context와 narrow-wave 손해를 그대로 보존해야 한다.

## 신규성의 정확한 위치

Fiez et al.의 transductive linear model 및 Rosa–Harman nuisance-resistant design으로 현재 고정 archive 모델을 표현할 수 있다. 기존 이론과의 대응은 PLAN_REVIEW_NOVELTY에 있다. 따라서 위 제안도 표준 design baseline과 비교해야 하며 첫 nuisance-aware acquisition이라고 주장할 수 없다.

기여 가능성은 subset BO에서 size/composition 실패를 구체적으로 정량화하고, count flexibility와 관측 구조가 함께 필요한 조건을 밝히며, 그 조건에 맞춘 충분한 해결이 실제 선택을 회복한다는 전체 연결에 있다. 이 모티브를 남들이 어느 범위까지 이미 다뤘는지는 계속 primary literature로 검증해야 한다. 논문 채택이나 최초성을 현재 보장할 수 없다.

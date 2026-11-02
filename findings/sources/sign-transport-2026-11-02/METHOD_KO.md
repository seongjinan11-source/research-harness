# PCFI의 부호 전달 한계와 기존 GraphPFN의 분류 능력 분리

추가 학습 없이 기존 공개 GraphPFN Mask Plain/PCFI Plain checkpoint 여섯 개를 평가한다. 두 모델은 이전 같은 official graph/SCM bank와 seed별 schedule에서 각각 192회 학습됐다. 여기서 Plain은 공개 GraphPFN의 직접 결측 입력 대조다. 과거 작은 typed-relation Plain과 다르다. 새 sign-gauged family는 그 학습 prior와 다르므로, 실패를 architecture의 불가능성으로 해석하지 않는다.

연결된 bipartite graph에서 저주파 Gaussian signal `a`를 만들고, feature law가 동일한 값을 전달하거나 `b_i`에 따라 부호를 바꾸도록 한다. `b`는 adjacency만으로 BFS해 구하는 bipartition이다. Label도 `a` 또는 `b*a`에서 만들며 두 축을 독립적으로 교차한다. 따라서 label heterophily와 feature heterophily를 같은 것으로 취급하지 않는다. 각 label 조건 내 graph·정답·context·query·mask를 그대로 두고 feature law만 바꾼다.

결측은 정확히 0/1/8개 source node를 context 밖에서 관측하는 방식이다. 192 nodes에서 1개는 99.48%, 8개는 95.83% node missing이며 Bernoulli 99.5%와 다르다. Full-feature는 별도 reference로 context에도 feature가 있다. Primary는 자기 feature가 없는 query다. 피처 제거(0 sources), context 1/4개 per class, full reference도 함께 보고한다.

원 PCFI와 같은 100회 전파·1회 채널 보정을 쓴다. 알려진 feature-law reference에서는 g=1일 때 관측값에 `b`를 곱해 PCFI를 수행한 뒤 `b`를 다시 곱한다. 이는 원 피처를 숫자로 공개하는 oracle은 아니지만, 참 feature-law g를 아는 진단이다. 새 그래프에서 g를 추론한 방법이나 학습 성과로 주장하지 않는다. 양의 LP와 signed LP도 둘 다 고정된 비학습 대조로 평가한다.

Primary는 label/feature 모두 sign-gauged인 조건, context4/class·sources8에서 `known-signed PCFI→같은 PFN`과 원 PCFI→PFN의 missing-query accuracy 차이다. 기존 세 training seed를 graph 안에서 평균하고 독립 base graph 32개를 paired bootstrap 5,000회 한다. 차이 CI 하한이 0을 넘는지 확인한다. Mask Plain, 양의 LP, signed LP와 피처 제거 대조까지 넘어야 새 predictor 설계의 근거로 삼는다. 이 진단만으로 새 학습을 자동 실행하지 않는다.

복원 MSE 감소와 classification accuracy 개선은 별개다. NLL/Brier도 별도 보고한다. 개발 graph 두 개에서 수치/poison/paired/no-source/node permutation/원 checkpoint replay를 확인한 뒤 source·config·checkpoint 해시를 잠그고 최종 bank를 생성한다. 기존 bank와 checkpoint는 수정하지 않는다. 이진 class·bipartite sign law에 한정하며 임의 heterophily·관계 schema·실제 graph 전이와 novelty를 주장하지 않는다.

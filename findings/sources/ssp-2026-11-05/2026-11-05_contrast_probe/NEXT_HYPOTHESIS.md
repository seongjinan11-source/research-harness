# 실패에서 남은 다음 가설 — 미구현

2026-11-05. 이번 확인 bank는 잠긴 음성 근거다. 아래는 새 method cycle의 후보이며 성능 확인이나 실행 예약을 뜻하지 않는다.

## 수정할 대상

첫 신규 count에서 arbitrary count nuisance를 제거한 composition 정보 증가량은0이다. 기존 farthest-count probe를 그대로 쓰고 myopic within-count score만 바꾸면 narrow support에서는 그 규칙을 쓸 기회가 없다. 동시에 wide에서는 한 개의 가장 애매한 swap을 목표로 한 새 규칙이 단순 A-opt보다 우월하지 않았다. 첫 문제는 query budget 배치, 두 번째는 decision target의 범위다.

## 가장 작은 다른 가설

유연한 joint fitting은 유지한다. 네 probe count를 두 쌍으로 묶어 K_a,K_a,K_b,K_b로 관측하고, 각 count의 두 query는 서로 다른 subset을 사용한다. K_a/K_b는 관측 count support에 대해 사전 고정한 coverage rule로 선택한다. 두 responses 차이는 g(K)가 소거되어

    y(x1)−y(x2)=(x1−x2)'p+(epsilon1−epsilon2)

를 준다. 이것은 고전 paired/blocked design 원리다. 식의 신규성을 주장하지 않는다. 두 observations 모두 full joint GP에 넣으며 cut을 강제하지 않는다. 기존 count에서 같은 subset을 재관측하는 설계도 아니다.

추가 query 예산을 주지 않으면4개 신규 count coverage를2개로 줄이는 비용이 생긴다. Narrow/wave 등의 count 일반화가 나빠질 수 있으므로 composition만 개선하고 끝내면 안 된다.

가장 싼 새 반증 실험은 **count schedule(old spread vs paired spread)×within-count rule(greedy vs A-opt)**의2×2다. 초기 데이터/모델/16query예산을 공유하고, schedule 주효과·selection 주효과·상호작용을 구별한다. 이번 후보를 새 baseline bank에서 임의로 다시 튜닝하지 않는다. A-opt의 이번 서술적 우위를 독립 확인해야 하며 기존 joint mean/probe의 end-to-end 대비도 필요하다.

Primary는 실제 noisy-incumbent latent utility regret를 포함해야 한다. Count support와 composition 학습을 함께 배분했을 때 전체 결정이 회복되는지가 목표다. 정밀 count schedule, pair 간격, 통계 gate, 개발/확인 seeds와 예산은 새 cycle에서 리뷰 후 고정한다. 이 문서만으로 구현·실행 완료를 주장하지 않는다.

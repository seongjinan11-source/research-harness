# 입력 범위에 관한 사후 기술 점검

2026-11-01. 초기 seed 학습 도중 관측 입력을 점검했다. 이 점검으로 protocol, 학습 코드, curriculum, checkpoint 선택을 바꾸지 않았다. 결과를 구제하는 추가 학습도 하지 않는다.

고정 전처리는 관측값만으로 feature별 평균과 표준편차를 계산한다. Whole-node observation이 한 행뿐이면 각 feature의 그 관측값이 평균이므로 표준화 후 0이다. PCFI에도 양의 값이나 음의 값이 남지 않아 모든 node feature가 0이 된다. 관측 행이 0개인 경우에도 모두 0이다.

Test의 99.5% uniform-node episode 32개 중 20개, clustered-node episode 32개 중 32개가 이 조건에 해당했다. Train에서는 각각 37/64, 64/64였다. 다른 결측률 50/90% whole-node episode에서는 전체 입력이 0인 경우가 없었다. Cell 99.5% test에서는 1/32였다. 전체 수치는 results/input_audit.json에 있다.

현재 GraphPFN forward에는 feature observation mask를 별도 채널로 주지 않는다. 따라서 위의 all-zero episode에서는 관측 source의 identity도 직접 제공하지 않는다. 이 부분의 예측은 구조와 context labels에 의존한다. FISF는 동일한 관측에서 synthetic injection을 추가하는 별도 전처리이므로 입력 표현이 다르다.

이것은 고정된 전처리의 정보 손실에 관한 설명이다. 관측 피처가 본래 무용했다거나, normalization을 바꾸면 성능이 개선된다는 결과가 아니다. 또한 sampling 전략만 바꿔서 지워진 정보를 복구할 수 있다고 주장하지 않는다.

이번 pilot은 reference-regret curriculum의 작은 비교로만 판정한다. Missing-feature 정보의 보존·활용, 실제 그래프 전이, 새로운 generator/library 학습으로 기여를 확대하지 않는다.

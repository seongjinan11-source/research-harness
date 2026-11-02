# Grid6 exact teacher pilot

기존 disjoint 과제의 생성 prior와 PFN 구조는 유지하고, sampled-label CE를 exact Grid6 posterior predictive distillation으로 바꾼다. 공개 Mask GraphPFN이 주 비교이고 Factor PFN은 보조 구조 대조다. 기존 CE checkpoint는 새96 graph bank에서 재평가한다.

Grid6는 synthetic graph를 생성하는 새로운 신경망이 아니다. 기존 생성 family의 정확한 조건부 예측을 계산하는 teacher다. Student 입력에는 sampled query/helper labels, 실제 role/density/signal/feature orientation을 주지 않는다. 배포 때 teacher나 target-graph gradient update를 사용하지 않는다.

새 학습은2 arms×3 seeds, 각2048 exposures/256 successful updates다. 초기화·task order·optimizer·learning-rate schedule·trainable weights는 원래 CE 학습과 동일하게 맞춘다. Teacher 생성 비용은 추가 학습 비용으로 따로 보고한다. 평가 bank는 새 seed지만 동일한 known family이며, 현실 전이나 새로운 메소드의 검증이 아니다.

사전 기준은 `protocol.json`. `pilot.py prepare`, `pilot.py preflight`, `pilot.py train --seed 11` (12/13도 동일), `pilot.py evaluate --seed 11`, `report.py` 순서로 실행한다. Source와 bank는 학습 전에 잠그고, 결과에 맞춰 loss나 gate를 바꾸지 않는다. 과거 source/checkpoint는 수정하지 않는다.

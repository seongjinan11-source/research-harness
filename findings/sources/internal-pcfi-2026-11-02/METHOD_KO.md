# 실제 PCFI를 내부 연산으로 쓰는 GraphPFN

이전 joint-imputation pilot의 공유 MLP numeric refiner를 재사용하지 않는다. 실제 PCFI의 거리 pseudo-confidence,100회 관측값 고정 전파와 마지막1회 채널 상관 보정을 사용한다. Confidence는 calibrated probability가 아니다.

각 cell의 c_if=.9^distance(i,observed_f)를 계산한다. 이웃 전파는 P_ij,f=A_ij*c_jf / sum_k A_ik*c_kf이며, learned arm은 분자에 exp(a_theta)를 곱한 뒤 같은 row normalization을 한다. a_theta=log2*tanh(sharedMLP(structure10,context4))다. 초기 마지막 layer가0이라 learned arm들의 초기 연산은 internal-fixed와 같다. 조절량은25-step block 안에서 고정한다. 관측값은 매 step 원래 값으로 고정하고 임의의 numeric innovation을 추가하지 않는다.

External은100회 전파+상관 보정을 마친 뒤 공개 PFN을 실행한다. Internal 세 모델은 PFN0/3/6/9층 앞에서25회씩 전파한다. 채널 상관 보정은 네 번째 block에서만1회 수행하고 마지막 +/-10 clip과 observed clamp도 같다. PFN은12층 한 번만 실행하며 각 갱신은 feature token에 E(Unew)-E(Uold)로 반영한다. TokenControl/mask embedding은 네 모델 모두 동일하다. Fixed internal과 external의 초기 예측은 일반적으로 다르며, 단순 삽입이 기존 파이프라인과 동치라고 주장하지 않는다.

Context router는 현재 PFN feature+label token의 context-class prototypes로 class-name-invariant support affinity와 entropy 통계를 만든다. Query 정답은 쓰지 않는다. Label-blind는 이 block 전체를0으로 하고 PFN hidden state나 labels를 전혀 읽지 않는다. Main classifier는 양쪽 모두 labels를 쓴다. 따라서 이 대조는 context+PFN feedback 경로 전체의 효과이며, hidden representation을 동일하게 둔 labels-only 효과가 아니다. Router 자체는513개 parameter로 동일하지만 blinded 입력의 일부 weight는 비활성이다.

공개 GraphPFN1.3/LimiX의 같은 초기화와 frozen-transformer policy를 사용하고, 기존 graph adapters/feature encoder/label decoder와 작은 공통 adapters를 학습한다. 동일 official-prior의 새 graph64개×4 masks=256 train tasks, seed11/12/13마다 같은 permutation으로256 successful CE updates다. 총12개 새 모델3072 updates다. 가변 실제 node/feature 수와 classes2~4의 small family이며 새 semantic schema나 실제 graph 전이는 아니다.

별도 dev4 graphs에서 SciPy PCFI 일치, zero-source, 관측값 clamp, hidden poison, node/class-name permutation, strict label blindness, finite-gradient 실제 optimizer step과 저장/복원을 검증한다. 네 smoke step을 폐기하고 구현을 잠근 뒤 final64 graphs를 생성한다. Odd feature padding은 관측0으로 두되 physical observation/primary membership에서 제외한다. 관측값이0개인 graph도 유지한다.

Primary는 context최대4/class, node uniform/clustered90/99.5% 네 조건에서 자기 피처 없는 query의 accuracy다. Query 평균→조건4개와 fixed3 seeds 평균→base graph64개5000 paired bootstrap을 한다. Context가 external/internal-fixed/label-blind 모두의 CI 하한>0으로 이겨야 한다. Full-feature2%p 비열등 guard도 고정한다. NLL/Brier, k1/cell, 전체 query, 조건별 결과와 end-to-end 비용은 별도로 보고한다. 기준 미통과 시 새 head/loss/temperature/학습량으로 구제하지 않는다.

실행: pcfi_tasks.py prepare → pcfi_checks.py → pcfi_tasks.py finalize → pcfi_trial.py --seed11/12/13 → pcfi_analysis.py audit/summary/cost. Lock은 모델/설계/입력 bank와 실제 공개 checkpoint hash를 포함한다. 이는 PCFI를 내장하는 하나의 구현 검증이며 novelty의 증명이 아니다.

# Frozen sampling: random-fill controls

2026-11-02. `context_feature_belief/DESIGN.md`의 첫 실행 단계다. 새 neural feature law나 공유 PFN 학습과 구별한다.

이전 `feature_marginal_pfn`에서 mean·PCFI 입력보다 좋아진 sampling이, 실제 관측값에 맞춘 completion을 필요로 하는지 확인한다. 기존32 graph×6 conditions와 같은 frozen Mask Plain3 seeds, 4 probability views를 쓴다. 이미 검토한 bank이므로 개발용 기전 비교다.

## Arm 정의

- `conditional_joint`: 원래 Gaussian conditional sampler를 같은 process에서 다시 평가한다.
- `iid_variance_matched`: Missing mean0, 독립 Gaussian noise. 원래 mixture의 analytical conditional variance를 cell별로 맞춘다.
- `graph_prior_variance_matched`: Missing mean0, uniform rho prior와 unconditional graph covariance를 쓰고 위 variance에 맞춘다. Observed-conditioned mean·rho posterior·conditional covariance를 사용하지 않는다.
- `iid_unit`: Missing mean0·variance1의 독립 noise. Missing proposal에는 관측값 정보가 없다.
- `wrong_sign_mean`: 원래 conditional sampler의 rho·covariance·noise를 유지하고 missing 위치 mean만 부호를 뒤집는다.

모든 arm은 **실제로 관측된 값은 그대로 PFN에 전달**한다. 따라서 이 대조가 검사하는 것은 대치 경로에 관측값을 반영하는 효과다. 관측 feature 전체의 효용을 제거한 실험이 아니다.

분산 맞춤 null은 `Var(X_U|D) = sum_r w_r[diag(Sigma_r)+(mu_r-mu_mix)^2]`를 사용한다. 그 scale에는 원래 관측값에 따른 second-moment 정보가 남는다. 이를 완전히 unconditional한 baseline이라고 부르지 않는다. Unit-noise 대조를 별도로 둔 이유다. Gaussian-law의 variance를 맞추며, 네 실제 draw의 empirical variance를 강제로 맞추지는 않는다.

`wrong_sign_mean`은 정렬 효과 진단이며 배포 메소드로 선택하지 않는다. Missing sample은 공통 ±10 범위로 clip하고 비율을 기록한다.

## 판정

`protocol.json`에 결과 전 설정했다. Conditional이 두 variance-matched null을 Uniform/Clustered 각각에서 paired NLL로 모두 이겨야 주 gate 통과다. Rates90/99.5와 고정 checkpoint3 seeds를 graph별 평균한 뒤 bootstrap5000회, 다중 비교 보정 없음. k1·full은 보조다. Accuracy/Brier도 보고한다.

실패 시 닫는 범위는 현재 frozen Gaussian conditional의 null 대비 우위 주장이다. Context-conditioned law를 학습하는 PFN 일반을 실패로 기록하지 않는다. 성공 시에도 새 neural model, actual feature law 복원, calibration 또는 real-graph/schema 전이 성과가 되는 것은 아니다.

## 재현·비용

기존 Windows GraphPFN Python으로 순서대로 실행한다.

```text
python experiment.py prepare
python experiment.py preflight
python experiment.py run --seed 11
python experiment.py run --seed 12
python experiment.py run --seed 13
python runtime/finalize.py
```

Prepare가 completion bank를 한 번 고정한다. Seed workers는 서로 다른 결과 폴더에 쓰므로 병렬 실행할 수 있다. Every scored arm has4 views; full-observed에서는 배열이 정확히 같아 같은 forward 결과를 공유한다. Full 캐싱은 수학적 input identity를 이용한 중복 제거이며 방법 간의 latency 우위로 사용하지 않는다.

기존 실험 코드는 변경하지 않는다. Source·completion/raw bank·checkpoint hash를 고정하고 저장 확률을 독립 재채점한다. Original conditional 결과와의 probability/NLL replay 차이는 별도로 보고한다. Primary pair는 새 process에서 replay한 conditional과 같은 process의 null 비교다.

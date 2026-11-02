# GraphPFN 연구 기록

실패·중단, 보존한 효과, 교훈, 미실행을 구분한 누적 기록이다. 숫자는 보존한 원문 보고서에서 옮겼고, 원인 미상인 결과에 사후 인과 설명을 붙이지 않는다. 새 논문과의 연결은 revisit.md의 재검토 단서에서 확인한다.

총 36개 프로젝트 · 152개 기록 (실패·중단 46개, 보존 34개, 교훈 50개, 미실행 22개)

## Unknown-law adaptive feature transport around frozen GraphPFN

실험 조건: Known bipartite Gaussian-signal family;64 new graphs/2048episodes;all4 label-feature-law cells;sources0/1/8/full;k1/4 per class;6 frozen checkpoints;0 new models/updates.

### ADTR-F01 [실패·중단] 관측 feature와 context로 추론한 q(g)를 두 PCFI completion의 가중치로 쓰면 frozen PFN이 고정 평균·feature-only 적응보다 정확해진다.

mask:accuracy52.357424%/NLL0.705444/Brier0.509802; identity:accuracy51.565242%/NLL0.715365/Brier0.518345; parity:accuracy52.205386%/NLL0.719126/Brier0.520209; uniform_two:accuracy52.354767%/NLL0.702633/Brier0.508327; feature_adapt:accuracy52.461961%/NLL0.704598/Brier0.509727; joint_adapt:accuracy52.555985%/NLL0.705381/Brier0.510321; known_g_reference:accuracy52.292214%/NLL0.715630/Brier0.517637; positive_lp:accuracy52.796435%/NLL0.733438/Brier0.523781; signed_lp:accuracy53.113760%/NLL0.765261/Brier0.545701; adaptive_lp:accuracy53.046855%/NLL0.694709/Brier0.500752; gaussian_joint:accuracy56.154827%/NLL0.672120/Brier0.479871; gaussian_label_only:accuracy53.687863%/NLL0.681201/Brier0.488462; known_t_lp_reference:accuracy57.135862%/NLL0.688915/Brier0.493230; uniform:accuracy49.302684%/NLL0.693147/Brier0.500000; uniform_two accuracy%p +0.201218 [95% CI -0.127618, +0.537932]; feature_adapt accuracy%p +0.094024 [95% CI -0.066909, +0.271023]; identity accuracy%p +0.990743 [95% CI +0.151828, +1.837583]; gaussian_joint accuracy%p -3.598842 [95% CI -5.635105, -1.536745]. Primary=False,strong_floor=False.

- 판정: 참 법칙을 주지 않는 적응 PCFI→frozen PFN은 사전 primary를 통과하지 못했다. 피처만 쓰는 적응과50:50 평균을 정확도에서 확실히 넘지 못했고, NLL/Brier는 두 대조보다 악화했다. 같은 입력의 직접 Gaussian/probit 추론은 유의한 피처 활용 이득을 보였다. 이번 입력 가설 선택기를 새 PFN 메소드로 확대하지 않는다.
- 원인·병목 [원인 미상]: 정확도 primary 두 CI가0을 포함하고 NLL/Brier는 두-completion 평균과 feature-only보다 악화했다. Known-family Gaussian/probit는 직접 예측에서 더 좋았다. 입력 후보·prior mismatch·decoder·weight 추정 중 고유 원인은 분리하지 않았다.
- 닫은 범위: 고정 Gaussian/probit24가설에서 q(g)를 계산하고, 기존192-update PCFI Plain의 identity/parity completion2개 예측만 가중 평균하는 frozen 진단 및 이를 근거로 한 자동 확대.
- 닫지 않은 것: 새 prior matched 학습, 내부 관측-라벨 공동추론, 임의 heterophily·heterogeneous schema와 실제 graph 전이를 기각하지 않는다. 이번에 하지 않은 neural training을 학습 실패로 세지 않는다. 직접 통계 추론의 피처 활용 이득은 보존한다.
- 재검토 조건: 추정한 법칙을 입력 가중치로만 전달하지 않고 classifier의 조건부 추론에 활용하는 구체적 연산이 생기며, 고정 직접 Gaussian/signed 추론이 하는 기능과 학습의 추가 가치를 분리할 수 있을 때.
- 기준선: 같은 정보·2branches의50:50와 strict feature-only; Mask/identity/parity PCFI; graph-only adaptive LP; 직접 Gaussian/probit; 동일-query zero-feature; privileged known-law는 reference로만 보고; 비용도 비교.
- 최소 재검증: 현재 bank를 조정하지 않는다. Label/feature law 교차와 source-context 분리를 유지한 새 graph bank에서 missing-query accuracy·동일-query 피처 제거·NLL/Brier·실제 비용을 사전 고정해 확인한다.
- 원문: [findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md](../findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md); [findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md](../findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md); [findings/sources/adaptive-transport-2026-11-02/protocol.json](../findings/sources/adaptive-transport-2026-11-02/protocol.json); [findings/sources/adaptive-transport-2026-11-02/results/statistics.json](../findings/sources/adaptive-transport-2026-11-02/results/statistics.json); [findings/sources/adaptive-transport-2026-11-02/results/audit.json](../findings/sources/adaptive-transport-2026-11-02/results/audit.json); [findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json); [findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json); [findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json](../findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json); [findings/sources/adaptive-transport-2026-11-02/results/cost.json](../findings/sources/adaptive-transport-2026-11-02/results/cost.json); [findings/sources/adaptive-transport-2026-11-02/results/preflight.json](../findings/sources/adaptive-transport-2026-11-02/results/preflight.json)

### ADTR-P01 [보존] 같은 관측 feature·분리된 context labels를 작은 Gaussian/probit working model에서 직접 결합하면, 이 family에서 피처 없는 예측보다 missing-query classification이 개선된다.

Gaussian accuracy56.154827%,without-features53.687863%; paired accuracy%p +2.466964 [95% CI +0.884420, +4.166715]; NLL -0.009081 [95% CI -0.013619, -0.004556]; Brier -0.008591 [95% CI -0.012854, -0.004317]. Frozen joint-adaptive PFN accuracy52.555985%; PFN-Gaussian accuracy%p -3.598842 [95% CI -5.635105, -1.536745].

- 확대 금지: 고정 known-family 통계 추론의 보존 결과다. 새 neural PFN 학습·novelty·참 Bayes posterior·calibration·실제 graph/schema·unknown Gaussian covariance 일반화의 성과가 아니다. 이 결과만으로 adaptive completion의 실패 원인을 특정하지 않는다. 보조 feature-intervention CI이며 primary 성공으로 바꾸지 않는다.
- 원문: [findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md](../findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md); [findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md](../findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md); [findings/sources/adaptive-transport-2026-11-02/protocol.json](../findings/sources/adaptive-transport-2026-11-02/protocol.json); [findings/sources/adaptive-transport-2026-11-02/results/statistics.json](../findings/sources/adaptive-transport-2026-11-02/results/statistics.json); [findings/sources/adaptive-transport-2026-11-02/results/audit.json](../findings/sources/adaptive-transport-2026-11-02/results/audit.json); [findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json); [findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json); [findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json](../findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json); [findings/sources/adaptive-transport-2026-11-02/results/cost.json](../findings/sources/adaptive-transport-2026-11-02/results/cost.json); [findings/sources/adaptive-transport-2026-11-02/results/preflight.json](../findings/sources/adaptive-transport-2026-11-02/results/preflight.json)

### ADTR-L01 [교훈] 법칙 식별 가능성, 입력 후보의 가중치, 조건부 예측 연산과 비용을 구분한다.

0/1source q(g)는수치오차내0.5다. 8source/k4 feature-only g accuracy60.15625%,joint59.375%; label-only t60.9375%,joint64.0625%. Primary joint PCFI52.555985%,50:50 52.354767%,feature-only52.461961%,direct Gaussian56.154827%. Gaussian은 동일-query feature-removal accuracy이득을 보였지만 adaptive PFN은 확인하지 못했다.

- 원문: [findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md](../findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md); [findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md](../findings/sources/adaptive-transport-2026-11-02/METHOD_KO.md); [findings/sources/adaptive-transport-2026-11-02/protocol.json](../findings/sources/adaptive-transport-2026-11-02/protocol.json); [findings/sources/adaptive-transport-2026-11-02/results/statistics.json](../findings/sources/adaptive-transport-2026-11-02/results/statistics.json); [findings/sources/adaptive-transport-2026-11-02/results/audit.json](../findings/sources/adaptive-transport-2026-11-02/results/audit.json); [findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/classical_audit.json); [findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json](../findings/sources/adaptive-transport-2026-11-02/results/latent_accuracy_audit.json); [findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json](../findings/sources/adaptive-transport-2026-11-02/runtime/latent_accuracy_addendum.json); [findings/sources/adaptive-transport-2026-11-02/results/cost.json](../findings/sources/adaptive-transport-2026-11-02/results/cost.json); [findings/sources/adaptive-transport-2026-11-02/results/preflight.json](../findings/sources/adaptive-transport-2026-11-02/results/preflight.json)

### ADTR-P02 [보존] 기존 primary 조건의 평균에서 직접 Gaussian/probit가 비특권적 대조11개보다 accuracy·NLL·Brier에서 우세했다.

Gaussian accuracy56.154827%,NLL0.672120,Brier0.479871. Mask PFN:accuracy%p +3.797403 [95% CI +1.555741, +6.019568]; 일반 PCFI→PFN:accuracy%p +4.589585 [95% CI +2.569802, +6.587119]; 고정 parity PCFI→PFN:accuracy%p +3.949441 [95% CI +1.813129, +6.118670]; 50:50 두 복원 평균:accuracy%p +3.800061 [95% CI +1.681170, +5.912160]; Feature-only 적응 PCFI→PFN:accuracy%p +3.692866 [95% CI +1.652477, +5.729643]; Context 적응 PCFI→PFN:accuracy%p +3.598842 [95% CI +1.536745, +5.635105]; 양의 LP:accuracy%p +3.358392 [95% CI +1.028041, +5.675842]; Signed LP:accuracy%p +3.041067 [95% CI +0.602935, +5.374370]; Context 적응 LP:accuracy%p +3.107972 [95% CI +0.741358, +5.489381]; 같은 Gaussian/probit·피처 제거:accuracy%p +2.466964 [95% CI +0.884420, +4.166715]; Uniform:accuracy%p +6.852143 [95% CI +4.297486, +9.340412]. 11개 개별95CI에서 accuracy하한>0,NLL/Brier상한<0.

- 확대 금지: 기존 bank를 본 뒤 계산한 보조 비교이며 multiplicity 미보정 pointwise CI다. 새 독립 검증·사전 Gaussian primary 통과·모든 세부 cell·Grid6·SSGP/GGP 재현 대비 승리·같은 prior matched PFN 우위·메소드 novelty가 아니다. 32개 cell 중22개에서 비특권적 대조의 점추정이 더 높았다.
- 원문: [findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md](../findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md); [findings/sources/gaussian-probit-review-2026-11-02/PRIMARY_COMPARISONS.json](../findings/sources/gaussian-probit-review-2026-11-02/PRIMARY_COMPARISONS.json); [findings/sources/gaussian-probit-review-2026-11-02/SOURCE_VERIFICATION.json](../findings/sources/gaussian-probit-review-2026-11-02/SOURCE_VERIFICATION.json); [findings/sources/gaussian-probit-review-2026-11-02/papers/SSGP.yaml](../findings/sources/gaussian-probit-review-2026-11-02/papers/SSGP.yaml); [findings/sources/gaussian-probit-review-2026-11-02/papers/GGP.yaml](../findings/sources/gaussian-probit-review-2026-11-02/papers/GGP.yaml); [findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md](../findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md); [findings/sources/adaptive-transport-2026-11-02/protocol.json](../findings/sources/adaptive-transport-2026-11-02/protocol.json); [findings/sources/adaptive-transport-2026-11-02/results/statistics.json](../findings/sources/adaptive-transport-2026-11-02/results/statistics.json); [findings/sources/adaptive-transport-2026-11-02/results/cost.json](../findings/sources/adaptive-transport-2026-11-02/results/cost.json)

### ADTR-L02 [교훈] Gaussian 대조의 관측된 승리와 graph-GP 선행 원리·PFN 메소드의 기여를 구분한다.

SSGP는 graph regularizer/probit/EP/evidence 선택, GGP는 P K(X,X) P^T/robust-max/variational ELBO를 사용한다. 현재 구현은 관측행 singular score와24 sign/amplitude 가설의 고정 Gaussian/probit 추론이며 두 논문의 재현이 아니다. 참 label-law LP57.135862%와 Gaussian56.154827%의 accuracy차이는 -0.981035 [95% CI -3.304919, +1.346377]%p였다.

- 원문: [findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md](../findings/sources/gaussian-probit-review-2026-11-02/REVIEW_KO.md); [findings/sources/gaussian-probit-review-2026-11-02/PRIMARY_COMPARISONS.json](../findings/sources/gaussian-probit-review-2026-11-02/PRIMARY_COMPARISONS.json); [findings/sources/gaussian-probit-review-2026-11-02/SOURCE_VERIFICATION.json](../findings/sources/gaussian-probit-review-2026-11-02/SOURCE_VERIFICATION.json); [findings/sources/gaussian-probit-review-2026-11-02/papers/SSGP.yaml](../findings/sources/gaussian-probit-review-2026-11-02/papers/SSGP.yaml); [findings/sources/gaussian-probit-review-2026-11-02/papers/GGP.yaml](../findings/sources/gaussian-probit-review-2026-11-02/papers/GGP.yaml); [findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md](../findings/sources/adaptive-transport-2026-11-02/REPORT_KO.md); [findings/sources/adaptive-transport-2026-11-02/protocol.json](../findings/sources/adaptive-transport-2026-11-02/protocol.json); [findings/sources/adaptive-transport-2026-11-02/results/statistics.json](../findings/sources/adaptive-transport-2026-11-02/results/statistics.json); [findings/sources/adaptive-transport-2026-11-02/results/cost.json](../findings/sources/adaptive-transport-2026-11-02/results/cost.json)

## Common-context witness 고정 곱셈

실험 조건: 실제 개발 graph, feature-free LP, 새 학습0회, 고정 same-context product.

### BIND-F01 [실패·중단] Class pooling 전 context identity를 보존해 곱하면 실제 graph에서 좋아진다.

k20 pooling-before-product: ACM76.390% vs uniform LP85.555%, IMDB28.311% vs35.413%. IMDB k20 test row 약96.1%는 공통 seed support가0이었다.

- 판정: 고정 common-witness product 중단
- 원인·병목 [진단이 지지]: 동일 witness를 강제하는 곱셈이 현재 sparse-support graph에서 대부분의 evidence를 제거했다. 표현력 collision toy가 실제 오류 원인이라는 증거는 없었다.
- 닫은 범위: 현재 모든 relation이 같은 context witness를 가져야 하는 고정 곱셈
- 닫지 않은 것: Task별 variable binding·논리 연산을 추론하는 일반 executor는 별개다.
- 재검토 조건: 공유 witness가 필요한 경우와 별개 witness를 허용할 경우를 관측 context로 구분하는 연산이 나올 때.
- 기준선: Uniform LP, class-pooling 후 product, 같은 graph의 강한 PFN
- 최소 재검증: 먼저 실제 common-support 빈도와 오류의 관계를 확인하고, 고정 곱셈이 아닌 task-adaptive binding의 independent gain을 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/binding.md](../findings/sources/graphpfn-2026-11-01/binding.md)

## Context-conditioned feature belief GraphPFN 설계 검토

실험 조건: 문헌 검토와 정확한 두 상태 산술 예시. 새 neural density·PFN·공유 suffix·coherence 학습은 미실행. 기존 FMARG 실험 판정을 덮어쓰지 않는다.

### CFB-U01 [미실행] Context labels에 조건화한 joint feature law와 shared observed encoder/particle suffix를 학습한 GraphPFN

최종 목표와 loss/계산 공유의 설계만 정리했다. 새 neural model 학습, random-fill null control, label-blind 대비, CE-only 대비 coherence의 독립 이득, latency·memory 및 실제 graph/schema 전이는 아직 실행하지 않았다.

- 원문: [findings/sources/context-feature-belief-2026-11-02/DESIGN.md](../findings/sources/context-feature-belief-2026-11-02/DESIGN.md)

### CFB-L01 [교훈] 관측 refinement의 martingale coherence와 효율적 marginalization 자체는 선행 원리다.

MCSSL(2026)의 §3–4가 관측 refinement의 평균 일관성과 two-sample estimator를 다룬다. MaM(ICML2024)은 marginalization self-consistency와 arbitrary marginal 직접 근사를 다룬다. GraphPFN에 해당 loss를 더했다는 사실만으로 새 원리를 주장하지 않는다.

- 원문: [findings/sources/context-feature-belief-2026-11-02/DESIGN.md](../findings/sources/context-feature-belief-2026-11-02/DESIGN.md)

### CFB-L02 [교훈] 관측 전 예측도 Monte Carlo 평균이면 coarse–refined 두 replica 전체를 독립화해야 하는 경우가 있다.

정확한 scalar 예시에서 올바른 tower squared residual은0, 한 refinement에 직접 맞춘 기대 penalty는0.1024다. 두 독립 refinement를 써도 분산0.01인 coarse estimate를 공용하면 기대 penalty0.01이 남는다. Coarse/refined replica 전체를 독립화하면 약8.7e-19다. 이는 수학 예시이고 GraphPFN 성능 수치가 아니다.

- 원문: [findings/sources/context-feature-belief-2026-11-02/DESIGN.md](../findings/sources/context-feature-belief-2026-11-02/DESIGN.md); [findings/sources/context-feature-belief-2026-11-02/math_sanity.py](../findings/sources/context-feature-belief-2026-11-02/math_sanity.py); [findings/sources/context-feature-belief-2026-11-02/math_sanity.json](../findings/sources/context-feature-belief-2026-11-02/math_sanity.json)

### CFB-L03 [교훈] Query CE와 coherence만으로 feature 가설의 의미가 식별되지는 않는다.

Decoder가 가설을 무시해도 관측 입력의 predictive distribution을 직접 표현할 수 있다. 또한 feature 확률[.1,.2,.3,.4]와[.14,.1,.36,.4]는 label probabilities[.1,.4,.6,.9]에서 모두 평균.63을 준다. 이 coarse/full-reveal 예시에서는 Bayes CE와 coherence가 같아도 feature law가 다르다. 임의의 틀린 q와 모든 refinement에서 성립한다는 주장은 아니다.

- 원문: [findings/sources/context-feature-belief-2026-11-02/DESIGN.md](../findings/sources/context-feature-belief-2026-11-02/DESIGN.md); [findings/sources/context-feature-belief-2026-11-02/math_sanity.json](../findings/sources/context-feature-belief-2026-11-02/math_sanity.json)

## Frozen counterfactual PFN residual

실험 조건: λ=1 고정, LP × PFN-full/PFN-neutral; 실제 개발 graph, 새 학습 없음.

### RATIO-F01 [실패·중단] Full−Neutral log probability가 LP에 더할 새로운 feature evidence다.

FISF ratio ACM82.35%, IMDB34.45%; LP85.55/35.41%, Full PFN74.36/44.55%. Controlled ratio ACM82.22/IMDB34.50%도 개선하지 못했다.

- 판정: Frozen 비율 보정 중단
- 원인·병목 [진단이 지지]: Neutral이 정확한 feature marginalization이 아니므로 비율을 likelihood ratio로 해석할 근거가 없다. IMDB LP log margin은0.134~0.151로 작고, residual argmax 변경률34.68%였다. 원인별 기여는 분리하지 않았다.
- 닫은 범위: 현재 frozen full/neutral 확률 비율의 λ=1 보정
- 닫지 않은 것: LP-conditioned 학습과 일관된 생성모형의 likelihood ratio 일반은 별개다.
- 재검토 조건: 통제 입력의 의미와 feature marginalization을 정의하고 확률 비율이 유효한 증거가 되는 조건을 제시하는 방법이 나올 때.
- 기준선: LP/PFN alone, 50:50, controlled neutral
- 최소 재검증: 먼저 feature intervention으로 순효용을 확인하고, 동일 구조·주입 아래 ratio가 정답 방향 보정인지 새 graph에서 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/ratio.md](../findings/sources/graphpfn-2026-11-01/ratio.md); [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

## Expected observation value training priors on public GraphPFN

실험 조건: Consistent64-state finite latent mixture;96nodes/2features;64train/64test graphs;source-context-query disjoint;primary4sources/context8TOTAL;30new checkpoints/6912updates;3fixed seeds;not old192node Gaussian family.

### DVP-F01 [실패·중단] J: entropy 감소로 학습 mask prior를 구성하면 같은 Mask PFN이 random·geometry·2배 random 학습보다 새 random-mask query를 정확히 분류한다.

entropy_mask accuracy55.794271%/NLL0.690654/Brier0.495232; entropy_pcfi accuracy57.313368%/NLL0.680086; vsrandom_mask: accuracy%p +0.499132 [95CI -0.347222,+1.345486]; vsgeometry_mask: accuracy%p +0.412326 [95CI -0.564236,+1.388889]; vsrandom_cost_mask: accuracy%p -0.911458 [95CI -2.083876,+0.239258]; Predeclared Mask primary=False.

- 판정: 사전 Mask accuracy conjunction을 통과하지 못했다. 이번 mask mixture를 메인 메소드로 확대하지 않는다.
- 원인·병목 [원인 미상]: Mask primary의 세 강한 대조를 모두 넘어야 하는 기준이 충족되지 않았다. J/V의 mask 차이·기대 가치 증가가 작다는 관찰은 있지만, score 추정·학생 fitting·budget·task family의 고유 원인은 분리하지 않았다.
- 닫은 범위: 64-state finite mixture,96nodes/2features,context2/8total,sources1/4/8,12candidate masks,256draw scores,50%random+50%top3 mixture,unchanged public GraphPFN192updates/seed의 설계와 이를 근거로 한 자동 확대.
- 닫지 않은 것: 모든 observation-value prior,더 큰 학습,다른 생성 family,새 architecture,실제 missing/heterogeneous graph를 기각하지 않는다. 기존 Gaussian working-model 결과와 이번 consistent finite-family reference는 별개다.
- 재검토 조건: 관측 설계의 가치가 학생의 transferable decision 연산으로 전달되지 않는 구체적 병목을 새 원리나 독립 증거로 특정하며, 단순 mixture/candidate/update 튜닝을 넘어서는 설계를 제시할 때.
- 기준선: 같은 family/정보/CE/초기화의random,geometry,extra-random-training;동일 prior에서 실제 학습한 PCFI-PFN;known-family reference;동일-query feature removal/shuffle;전체 cold bank/scoring 비용.
- 최소 재검증: 기존 final bank를 튜닝하지 않는다. 실제 realization 독립 설계와 teacher consistency를 유지하고 새 family/graph bank에서 accuracy primary·전체 seed·비용을 사전 고정한다.
- 원문: [findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md](../findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md); [findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md](../findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md); [findings/sources/decision-value-prior-2026-11-02/protocol.json](../findings/sources/decision-value-prior-2026-11-02/protocol.json); [findings/sources/decision-value-prior-2026-11-02/results/statistics.json](../findings/sources/decision-value-prior-2026-11-02/results/statistics.json); [findings/sources/decision-value-prior-2026-11-02/results/audit.json](../findings/sources/decision-value-prior-2026-11-02/results/audit.json); [findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json](../findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json); [findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json](../findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json)

### DVP-F02 [실패·중단] V: Bayes 분류 오류 감소로 학습 mask prior를 구성하면 같은 Mask PFN이 random·geometry·2배 random 학습보다 새 random-mask query를 정확히 분류한다.

decision_mask accuracy55.664062%/NLL0.690199/Brier0.495521; decision_pcfi accuracy56.944444%/NLL0.680549; vsrandom_mask: accuracy%p +0.368924 [95CI -0.781250,+1.605903]; vsgeometry_mask: accuracy%p +0.282118 [95CI -0.889757,+1.540799]; vsrandom_cost_mask: accuracy%p -1.041667 [95CI -2.343750,+0.260417]; Predeclared Mask primary=False.

- 판정: 사전 Mask accuracy conjunction을 통과하지 못했다. 이번 mask mixture를 메인 메소드로 확대하지 않는다.
- 원인·병목 [원인 미상]: Mask primary의 세 강한 대조를 모두 넘어야 하는 기준이 충족되지 않았다. J/V의 mask 차이·기대 가치 증가가 작다는 관찰은 있지만, score 추정·학생 fitting·budget·task family의 고유 원인은 분리하지 않았다.
- 닫은 범위: 64-state finite mixture,96nodes/2features,context2/8total,sources1/4/8,12candidate masks,256draw scores,50%random+50%top3 mixture,unchanged public GraphPFN192updates/seed의 설계와 이를 근거로 한 자동 확대.
- 닫지 않은 것: 모든 observation-value prior,더 큰 학습,다른 생성 family,새 architecture,실제 missing/heterogeneous graph를 기각하지 않는다. 기존 Gaussian working-model 결과와 이번 consistent finite-family reference는 별개다.
- 재검토 조건: 관측 설계의 가치가 학생의 transferable decision 연산으로 전달되지 않는 구체적 병목을 새 원리나 독립 증거로 특정하며, 단순 mixture/candidate/update 튜닝을 넘어서는 설계를 제시할 때.
- 기준선: 같은 family/정보/CE/초기화의random,geometry,extra-random-training;동일 prior에서 실제 학습한 PCFI-PFN;known-family reference;동일-query feature removal/shuffle;전체 cold bank/scoring 비용.
- 최소 재검증: 기존 final bank를 튜닝하지 않는다. 실제 realization 독립 설계와 teacher consistency를 유지하고 새 family/graph bank에서 accuracy primary·전체 seed·비용을 사전 고정한다.
- 원문: [findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md](../findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md); [findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md](../findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md); [findings/sources/decision-value-prior-2026-11-02/protocol.json](../findings/sources/decision-value-prior-2026-11-02/protocol.json); [findings/sources/decision-value-prior-2026-11-02/results/statistics.json](../findings/sources/decision-value-prior-2026-11-02/results/statistics.json); [findings/sources/decision-value-prior-2026-11-02/results/audit.json](../findings/sources/decision-value-prior-2026-11-02/results/audit.json); [findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json](../findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json); [findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json](../findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json)

### DVP-P03 [보존] Teacher와 일치하는 finite family에서 분리된 피처·라벨이 query 예측에 쓸모 있는 과제를 실제로 구성하고 exact enumeration reference로 검사했다.

Exact-family accuracy74.348958%,no-feature70.312500%;same-query gain%p +4.036458 [95CI +1.888021,+6.380208]. Posterior64-state enumeration exact;J/V expectations Monte Carlo256,independent audit4096draws.

- 확대 금지: 학습된 PFN이나 J/V prior의 성공,unknown-family/GP exact inference,과거 bank의 무정보 원인 확정,실제 heterograph/generalization의 성과가 아니다.
- 원문: [findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md](../findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md); [findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md](../findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md); [findings/sources/decision-value-prior-2026-11-02/protocol.json](../findings/sources/decision-value-prior-2026-11-02/protocol.json); [findings/sources/decision-value-prior-2026-11-02/results/statistics.json](../findings/sources/decision-value-prior-2026-11-02/results/statistics.json); [findings/sources/decision-value-prior-2026-11-02/results/audit.json](../findings/sources/decision-value-prior-2026-11-02/results/audit.json); [findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json](../findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json); [findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json](../findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json)

### DVP-L01 [교훈] J와 V는 다른 목표지만 이번 family의 관측 설계를 충분히 분리하지 못했다. 선택 점수 증가와 학생의 새 mask 분류 개선을 구분해야 한다.

J/V selected-mask equality73.44~87.50% across6cells;50%shared uniform component;independent4096draw audit on first16train graphs stored. All30 models share4012259 trainable parameters;ordinary192updates,random_cost384;actual6912updates.

- 원문: [findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md](../findings/sources/decision-value-prior-2026-11-02/REPORT_KO.md); [findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md](../findings/sources/decision-value-prior-2026-11-02/METHOD_KO.md); [findings/sources/decision-value-prior-2026-11-02/protocol.json](../findings/sources/decision-value-prior-2026-11-02/protocol.json); [findings/sources/decision-value-prior-2026-11-02/results/statistics.json](../findings/sources/decision-value-prior-2026-11-02/results/statistics.json); [findings/sources/decision-value-prior-2026-11-02/results/audit.json](../findings/sources/decision-value-prior-2026-11-02/results/audit.json); [findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json](../findings/sources/decision-value-prior-2026-11-02/results/bank_audit.json); [findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json](../findings/sources/decision-value-prior-2026-11-02/runtime/scope_errata.json)

## Conditional diffusion and ESS-controlled SMC in frozen FactorPFN

실험 조건: Known binary12node/2feature typed SBM;3frozen CEFactors;3new8476parameter denoisers/3072updates;96newgraphs/576disjointepisodes;primary48matchedgraphs;50/75%missing,not99.5%.

### DFI-F01 [실패·중단] 작은 conditional diffusion proposal과 동일 potential의 MH/SMC 보정이 비슷한 추론 비용의 classical SMC보다 정확한 query inference를 제공한다.

factor_bp:accuracy71.403770%/FactorKL0.085031/Grid6KL0.091114; factor_exact:accuracy71.644896%/FactorKL0.000000/Grid6KL0.011446; smc_gibbs1:accuracy69.765487%/FactorKL0.008698/Grid6KL0.020647; smc_gibbs_cost:accuracy68.853340%/FactorKL0.007171/Grid6KL0.017499; diffusion_only:accuracy65.673225%/FactorKL0.047351/Grid6KL0.060687; diffusion_smc_fixed:accuracy65.151014%/FactorKL0.079318/Grid6KL0.090732; diffusion_smc_ess:accuracy64.952601%/FactorKL0.071562/Grid6KL0.085439;fixed-costSMC accuracy%p-3.702326[95CI-6.346588,-1.234258],FactorKL+0.072148[+0.060790,+0.084608].Inferencesuccess=False,classificationsuccess=False.

- 판정: 현재 learned proposal+correction을 메인 방법으로 확대하지 않는다. 비용 대조 SMC보다 accuracy와 target-fit이 악화했고,더 저렴한1sweep SMC도 평균적으로 좋았다.
- 원인·병목 [원인 미상]: Exact frozen-model KL·accuracy 사전 gate를 통과하지 못했다. Small-state kernel validation은 통과했지만 finite-particle/short-schedule/denoiser fit/mixing/representation의 고유 원인은 분리하지 않았다.
- 닫은 범위: 기존 binary12node+2column Gaussian-factor family,frozen CEFactor3개,512task×64exactjoint draws,odd/even GNNwidth24/1024updates,64particles/8stages,reset-diffusion reverse proposal+MH와고정temperature의 구현 및 자동 확대.
- 닫지 않은 것: Diffusion 전체,DPM-SNC/ReDiSC/RDSMC 원본 방법,다른 proposal/학습량,실제 graph/unseen schema,RNP objective,새 feature-law의 실패로 세지 않는다. 기존 Factor의 positive results는 보존한다.
- 재검토 조건: 같은 잠재 state와 local factor에서 proposal이 어느 예측적으로 중요한 joint mode를 누락하는지 독립 증거를 제시하고,classical Gibbs/blocked sampler와 비용을 맞춰 이를 해결하는 연산이 있을 때.
- 기준선: Ordinary1sweep/개발 latency로 정한2sweep Gibbs-SMC;original Factor BP,constant BP,exact learned-Factor,Grid6exact/BP;동일새 bank의 frozen Mask/PCFI;teacher/training/inference 비용.
- 최소 재검증: 이번 final bank를 튜닝하지 않는다. 새 독립 graph/명시된 구조 family에서 finite sampler exact-target error와accuracy·NLL/Brier·role coverage·비용을 사전 분리하고 fullsupport/양방향proposal/MH 검사를 유지한다.
- 원문: [findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md](../findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md); [findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md](../findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md); [findings/sources/diffusion-factor-2026-11-02/protocol.json](../findings/sources/diffusion-factor-2026-11-02/protocol.json); [findings/sources/diffusion-factor-2026-11-02/results/statistics.json](../findings/sources/diffusion-factor-2026-11-02/results/statistics.json); [findings/sources/diffusion-factor-2026-11-02/results/audit.json](../findings/sources/diffusion-factor-2026-11-02/results/audit.json); [findings/sources/diffusion-factor-2026-11-02/results/benchmark.json](../findings/sources/diffusion-factor-2026-11-02/results/benchmark.json); [findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json](../findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json)

### DFI-F02 [실패·중단] Rényi-2/ESS 기반 adaptive likelihood tempering이 같은 learned diffusion+SMC fixed schedule보다 예측 품질을 개선한다.

Fixed FactorKL0.079318,ESS0.071562;difference-0.007756[95CI-0.019864,+0.006172].Accuracy65.151014%→64.952601%.ESSgate=False.

- 판정: 현재.8K threshold/8stage budget의ESS controller 증분은 미입증이며 추가 tuning하지 않는다.
- 원인·병목 [원인 미상]: KL 증분 CI가0을 포함하고accuracy도 개선되지 않았다. Weight entropy/step schedule만으로 latent coverage와 predictive fit을 보장할 수 없으며 고유 원인은 미분리다.
- 닫은 범위: 같은64particles/8stages/학습된reverse kernel을 유지하고32-step bisection으로incremental ESS≥.8K의β를 택한뒤 마지막stageβ=1을 강제하는 controller.
- 닫지 않은 것: Rényi Neural Processes의 variational objective는 실행하지 않았다. 모든 Rényi divergence/entropy 활용,다른SMC bridge/step budget·posterior inference를 기각하지 않는다.
- 재검토 조건: Weight entropy와 label-predictive joint mode error의 관계를 분리하고,같은 particle/latency budget에서 ESS 외의 필요한 정보를 이용하는 제어 연산이 생길 때.
- 기준선: 동일proposal/potential의fixed schedule,ordinary Gibbs-SMC,exact frozen-model reference와accuracy를유지한다. Adaptive extra overhead를 포함한다.
- 최소 재검증: 새graph에서fixed-ESS controller 차이를posterior fit/분류/시간으로나누어검증한다. 최종ESS상승을성공으로대체하거나RNP논문방법을실행했다고부르지않는다.
- 원문: [findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md](../findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md); [findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md](../findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md); [findings/sources/diffusion-factor-2026-11-02/protocol.json](../findings/sources/diffusion-factor-2026-11-02/protocol.json); [findings/sources/diffusion-factor-2026-11-02/results/statistics.json](../findings/sources/diffusion-factor-2026-11-02/results/statistics.json); [findings/sources/diffusion-factor-2026-11-02/results/audit.json](../findings/sources/diffusion-factor-2026-11-02/results/audit.json); [findings/sources/diffusion-factor-2026-11-02/results/benchmark.json](../findings/sources/diffusion-factor-2026-11-02/results/benchmark.json); [findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json](../findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json)

### DFI-P01 [보존] Frozen learned Factor의local potential을 정확히 합산하면 model-based Grid6 예측에 가까워진다. 분류 accuracy headroom은 별도로 작게 관측됐다.

FactorBP Grid6KL0.091114→exactFactor0.011446;accuracy71.403770%→71.644896%;delta%p+0.241127[95CI-0.847388,+1.320030].Same frozen8099parameters,65536-state offline enumeration.

- 확대 금지: 새 학습이나 diffusion의성과가아니다. 정확한conditional-model inference이지 참SBM posterior복원은아니며,accuracy 동등성/비열등성·일반적인headroom부재·largegraph효율성을 증명하지 않는다. Constant-exact와 비교하지 않았으므로학습potential의독립적인일반우위로확대하지 않는다.
- 원문: [findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md](../findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md); [findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md](../findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md); [findings/sources/diffusion-factor-2026-11-02/protocol.json](../findings/sources/diffusion-factor-2026-11-02/protocol.json); [findings/sources/diffusion-factor-2026-11-02/results/statistics.json](../findings/sources/diffusion-factor-2026-11-02/results/statistics.json); [findings/sources/diffusion-factor-2026-11-02/results/audit.json](../findings/sources/diffusion-factor-2026-11-02/results/audit.json); [findings/sources/diffusion-factor-2026-11-02/results/benchmark.json](../findings/sources/diffusion-factor-2026-11-02/results/benchmark.json); [findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json](../findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json)

### DFI-L01 [교훈] 높은final ESS·MH acceptance·고유state 수가 정확한joint posterior나높은classification을보장하지않는다. 추론primary와분류gate를분리해야한다.

전체576episodes에서corrected diffusion의평균finalESS 약60.6~60.9/64,acceptance 약.62~.65였지만ordinarySMC보다targetKL과accuracy가나빴다. Pure diffusion은uniformweights라ESS64가정의상나온다.Production8640samplerreplays최대오차0,독립4spin/4096particleauditmaxnodeerror<.01.

- 원문: [findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md](../findings/sources/diffusion-factor-2026-11-02/REPORT_KO.md); [findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md](../findings/sources/diffusion-factor-2026-11-02/METHOD_KO.md); [findings/sources/diffusion-factor-2026-11-02/protocol.json](../findings/sources/diffusion-factor-2026-11-02/protocol.json); [findings/sources/diffusion-factor-2026-11-02/results/statistics.json](../findings/sources/diffusion-factor-2026-11-02/results/statistics.json); [findings/sources/diffusion-factor-2026-11-02/results/audit.json](../findings/sources/diffusion-factor-2026-11-02/results/audit.json); [findings/sources/diffusion-factor-2026-11-02/results/benchmark.json](../findings/sources/diffusion-factor-2026-11-02/results/benchmark.json); [findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json](../findings/sources/diffusion-factor-2026-11-02/results/convergence_audit.json)

## 직접 주변화하는 graph-conditioned tree circuit

실험 조건: 새12-node binary typed SBM, target8/helper4·relation2·Gaussian feature2. 학습된3 arms×3 fixed seeds, 각2048 task exposures/256 updates. Primary matched48 base graphs, 전체 OOD 포함192 graphs.

### DMARG-P01 [보존] 유효한 conditional circuit을 학습해 두 학습된 직접 예측기보다 좋은 예측을 얻었다.

Primary circuit61.3839%/NLL0.605128/Brier0.430740, public Mask PFN53.4846%/0.687080/0.493828, direct graph51.1512%/0.693134/0.499986. Circuit−Mask accuracy +7.899306%p CI[+3.889009,+11.790761], NLL−0.081952 CI[−0.101972,−0.062402]. Circuit−direct accuracy +10.232722%p CI[+5.688365,+14.589172], NLL−0.088005 CI[−0.106528,−0.069288].

- 확대 금지: Grid6나 feature-only 확률모형보다 우월함, 일반 GSPN/MaM 재현, 원래 cyclic SBM의 exact posterior, 새로운 semantic schema 또는 실제 heterogeneous graph 성과로 확대하지 않는다. 파라미터 수는 같지 않다.
- 원문: [findings/sources/direct-marginal-2026-11-02/REPORT_KO.md](../findings/sources/direct-marginal-2026-11-02/REPORT_KO.md); [findings/sources/direct-marginal-2026-11-02/results/paired.json](../findings/sources/direct-marginal-2026-11-02/results/paired.json); [findings/sources/direct-marginal-2026-11-02/results/decision.json](../findings/sources/direct-marginal-2026-11-02/results/decision.json)

### DMARG-P02 [보존] 이 소형 family의 circuit은 실제 관측 feature와 node 대응을 활용한다.

Context2/clustered50 보조 조건, 전체192 base graphs. No-observed−standard accuracy−15.451389%p CI[−19.386574,−11.805556], NLL+0.172725 CI[+0.147664,+0.197846]. Shuffled−standard accuracy−13.686343%p CI[−17.534722,−9.953704], NLL+0.352257 CI[+0.280898,+0.432729].

- 확대 금지: 원래 IMDB에서 관측값의 평균 효용을 회수했다는 뜻이 아니다. Feature가 class와 연결되도록 만든 Gaussian family의 보조 intervention이며 primary 조건과 다르다. Graph-specific 증분이나 calibrated true posterior를 단독 입증하지 않는다.
- 원문: [findings/sources/direct-marginal-2026-11-02/results/interventions.json](../findings/sources/direct-marginal-2026-11-02/results/interventions.json); [findings/sources/direct-marginal-2026-11-02/protocol.json](../findings/sources/direct-marginal-2026-11-02/protocol.json)

### DMARG-F01 [실패·중단] 현재 conditional tree circuit이 알려진 family의 강한 classical inference와 경쟁하는 메인 메소드다.

Grid6-BP69.5602%/NLL0.540755, Grid6-exact69.3680%/0.541211. Circuit−Grid6-BP accuracy−8.176257%p CI[−13.111376,−3.230252], NLL+0.064373 CI[+0.024247,+0.105302]. Grid6-exact predictive KL0.064716으로 목표.01 초과. 같은GPU forward median circuit12.369ms, Grid6-BP13.443ms, Grid6-exact10.427ms. 2배 latency 기준도 미충족.

- 판정: Neural-baseline gate는 성공했지만 classical 품질·비용 gate는 실패. 현재 구성의 메인 메소드 확대는 보류.
- 원인·병목 [원인 미상]: 유효한 주변화에도 예측 fit의 격차가 남았다. Tree family의 표현력, G만 보는 potential encoder의 근사, 학습 예산의 기여를 분리하지 않았다. 정확한 고유 원인은 미상이다. 작은 known-family baseline의 우위를 모델 학습 전체의 불필요함으로 확대하지 않는다.
- 닫은 범위: 현재56,385-parameter/32-component conditional tree circuit, 고정12-node·2-feature·2-relation family,256-update query-CE 학습을 그대로 메인 방법으로 확대.
- 닫지 않은 것: 다른 유효 circuit 구조, family/schema 일반화, exact variable elimination의 다른 근사·공유 연산, MaM 자체를 실행하거나 실패시킨 결과가 아니다. DMARG-P01/P02의 실제 개선과 feature 이용은 보존한다.
- 재검토 조건: 강한 known-family inference의 예측 품질을 유지하거나, 작은 고정 grid가 다루지 못하는 family에서 실제 관측값과 구조의 결합 이득을 회수하는 연산이 나올 때.
- 기준선: Gaussian-feature orientation을 적분하는 Grid6 exact/BP, known-feature-only, constant tree, 실제 public Mask PFN 및 같은 예산 direct graph. 더 작은 network 대비 승리만으로 재개하지 않는다.
- 최소 재검증: 같은 관측 정보·학습 task·전체 비용으로 fixed Gaussian 통계를 넘는 graph-specific 이득부터 확인한다. 유효한 circuit의 범위와 원 데이터 posterior를 분리하고 fresh bank·heldout family를 사용한다.
- 원문: [findings/sources/direct-marginal-2026-11-02/results/paired.json](../findings/sources/direct-marginal-2026-11-02/results/paired.json); [findings/sources/direct-marginal-2026-11-02/results/benchmark.json](../findings/sources/direct-marginal-2026-11-02/results/benchmark.json); [findings/sources/direct-marginal-2026-11-02/results/decision.json](../findings/sources/direct-marginal-2026-11-02/results/decision.json)

### DMARG-L01 [교훈] 직접 예측기 대비 큰 개선을 graph-specific uncertainty 추론의 기여로 전부 귀속하지 않는다.

설계 고정 뒤 추가한 known-family feature-only 대조는 accuracy58.1329%/NLL0.605147. Circuit−feature-only accuracy+3.251075%p CI[−0.142816,+6.531240], NLL−0.000019 CI[−0.013783,+0.013696]. NLL 추가 개선은 확인되지 않았다. 이 대조는 Gaussian magnitude.8/sd.6를 알고 있으나 graph를 사용하지 않는다. Constant-tree NLL0.685252 대비 circuit NLL−0.080124 CI[−0.125201,−0.035681] 개선은 별도로 보존한다.

- 원문: [findings/sources/direct-marginal-2026-11-02/results/feature_controls.json](../findings/sources/direct-marginal-2026-11-02/results/feature_controls.json); [findings/sources/direct-marginal-2026-11-02/runtime/feature_controls.py](../findings/sources/direct-marginal-2026-11-02/runtime/feature_controls.py)

### DMARG-L02 [교훈] 관측 feature가 label context 바깥에만 있는 조건은 아직 회수하지 못한 관계 추론 정보를 드러낸다.

Visible context/mask로 나눈 primary 사후 분할에서 circuit52.6899%/NLL0.699547, Grid6-BP64.8682%/0.624596. Context와 noncontext 양쪽에 관측 feature가 있는 분할은 circuit74.8846%/0.453902였다. 분할은 query truth로 정하지 않았고 추가 유의성·독립 확인 주장을 하지 않는다.

- 원문: [findings/sources/direct-marginal-2026-11-02/results/descriptive_breakdowns.json](../findings/sources/direct-marginal-2026-11-02/results/descriptive_breakdowns.json); [findings/sources/direct-marginal-2026-11-02/runtime/breakdowns.py](../findings/sources/direct-marginal-2026-11-02/runtime/breakdowns.py)

### DMARG-L03 [교훈] Valid conditional circuit의 계산 유효성과 데이터 fit을 별도 검증한다.

Circuit의 evidence·marginal 열거 오차, label 합 규칙, Gaussian 좌표 적분, conditional ratio 오차가 약10^-15 수준이었다. Parameters는 G와 알려진 family만 보고 evidence 변경 시 고정됨을 검사했다. 총9개 학습/2304 성공 updates/18432 task exposures/44928 예측을 감사했다. Test4608 episodes 중 실제 feature 관측0개는1353개다.

- 원문: [findings/sources/direct-marginal-2026-11-02/results/preflight.json](../findings/sources/direct-marginal-2026-11-02/results/preflight.json); [findings/sources/direct-marginal-2026-11-02/results/audit.json](../findings/sources/direct-marginal-2026-11-02/results/audit.json); [findings/sources/direct-marginal-2026-11-02/results/final_integrity.json](../findings/sources/direct-marginal-2026-11-02/results/final_integrity.json)

### DMARG-U01 [미실행] NEXT-U03의 가변 feature/schema·대형 graph circuit 및 독립 MaM 구현

실행한 것은 고정2-feature·2-relation binary family의 conditional tree mixture다. 연속·가변 차원 MaM, 새 semantic relation vocabulary, 대형 cyclic graph의 정확한 주변화, full sparse-cell PFN과 실제 graph 전이는 실행하지 않았다.

- 원문: [findings/sources/direct-marginal-2026-11-02/protocol.json](../findings/sources/direct-marginal-2026-11-02/protocol.json); [findings/sources/direct-marginal-2026-11-02/model.py](../findings/sources/direct-marginal-2026-11-02/model.py)

### DMARG-P03 [보존] Feature 노드와 label context가 겹치지 않는 조건에서도 known-family 추론은 관측 feature의 추가 예측 정보를 이용한다.

기존 matched48 graphs의 사후 disjoint subset280 episodes를 선택하고 같은 graph/context/query의 no-feature setting1과 짝지었다. Graph별 episode·고정 training seed 평균 기준 Grid6-exact accuracy60.4874→65.1069%, 차이+4.6195%p CI[+1.9881,+7.7814], NLL0.655644→0.625848, 차이−0.029796 CI[−0.048503,−0.011942]. Grid6-BP accuracy59.4674→65.3938%, 차이+5.9264%p. 같은 조건의 circuit accuracy53.0929→52.4214%, 차이−0.6714%p CI[−3.0503,+1.7070]; public Mask PFN51.5483→51.7390%, 차이+0.1907%p CI[−0.8857,+1.1744].

- 확대 금지: 기존 bank의 탐색적 재분석이며 새 primary·독립 검증·다중 비교 보정·학습된 새 메소드의 성과가 아니다. 이전 episode 평균52.69/64.87%와 이번 graph 평균을 섞지 않는다. Query 자체도 feature가 없는 추가 분할에서는 Grid6-exact accuracy+2.5173%p CI[−0.0241,+5.2686], NLL−0.014387 CI[−0.029204,+0.000323]로0을 포함한다. 이 더 엄격한 조건의 exact 이득을 확정하지 않는다. Graph에서 feature/label 정보를 연결한다는 발상 자체의 novelty, 실제 graph·새 schema 일반화 또는 circuit 실패의 고유 원인을 입증하지 않는다.
- 원문: [findings/sources/disjoint-feature-2026-11-02/disjoint_feature_reanalysis_20261102.py](../findings/sources/disjoint-feature-2026-11-02/disjoint_feature_reanalysis_20261102.py); [findings/sources/disjoint-feature-2026-11-02/disjoint_feature_reanalysis_20261102.json](../findings/sources/disjoint-feature-2026-11-02/disjoint_feature_reanalysis_20261102.json); [findings/sources/direct-marginal-2026-11-02/protocol.json](../findings/sources/direct-marginal-2026-11-02/protocol.json); [findings/sources/direct-marginal-2026-11-02/results/descriptive_breakdowns.json](../findings/sources/direct-marginal-2026-11-02/results/descriptive_breakdowns.json)

## Disjoint feature–label factor PFN pilot

실험 조건: 새96 binary typed-SBM graphs 중 matched48 primary. 관측feature node와 label context 교집합0. 3 arms×3 seeds, 동일2048 task exposures/256 updates. Context는 총1/2/4개,12개 node 중3/6개 관측.

### DFACT-P01 [보존] 관측 feature와 label context가 겹치지 않는 새 graph bank에서 학습된 structured predictor가 두 공개 GraphPFN 입력 대조보다 높은 정확도를 얻었다.

Factor PFN65.4004%/NLL0.628722, Mask PFN57.0974%/0.676368, 실제 PCFI 입력으로 학습한 PFN55.2304%/0.677401. Factor−Mask accuracy+8.3030%p CI[+2.7750,+14.0753], Factor−PCFI+10.1700%p CI[+4.4253,+16.3513]. NLL 차이는 각각−0.047645 CI[−0.141633,+0.050681], −0.048678 CI[−0.142740,+0.049630]로0을 포함한다.

- 확대 금지: 확률 품질의 neural 대비 유의한 우위, strong classical 대비 우위, variable schema·실제 graph 전이, 논문급 novelty로 확대하지 않는다. Matching은 task/step budget이며 파라미터 수·학습 시간은 다르다. Source raw-values/graph intervention과 learned potential의 독립적 가치도 별도로 본다. Low-density의 Factor63.7621%/NLL0.632631 vs Constant60.4249%/0.696110은 보조 point estimate이며 해당 paired CI를 보고하지 않았으므로 일반적인 OOD adaptation 성공으로 승격하지 않는다.
- 원문: [findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md](../findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md); [findings/sources/disjoint-factor-2026-11-02/protocol.json](../findings/sources/disjoint-factor-2026-11-02/protocol.json); [findings/sources/disjoint-factor-2026-11-02/results/paired.json](../findings/sources/disjoint-factor-2026-11-02/results/paired.json); [findings/sources/disjoint-factor-2026-11-02/results/summary.json](../findings/sources/disjoint-factor-2026-11-02/results/summary.json)

### DFACT-P02 [보존] 학습된 predictor는 disjoint 조건의 feature–node 대응과 graph 구조 변화에 민감하다.

사전 context2/observed6 보조 intervention, matched48 graphs: feature shuffle−11.2269%p CI[−20.3704,−2.0833], graph permutation−21.7593%p CI[−34.3779,−8.7963]. No-observed−standard는−6.1343%p CI[−12.9630,+0.9259]였으며 NLL 차이는−0.046387 CI[−0.276343,+0.134338]였다.

- 확대 금지: Shuffle 민감도만으로 feature 추가의 유익한 순효과를 확정하거나, 원래 feature-removal gate를 통과했다고 쓰지 않는다. CI가0을 포함하는 제거 결과도 feature 효용0의 증명은 아니다. Loopy feedback 차단을 새 방법으로 골라 gate를 바꾸지 않는다.
- 원문: [findings/sources/disjoint-factor-2026-11-02/results/interventions.json](../findings/sources/disjoint-factor-2026-11-02/results/interventions.json); [findings/sources/disjoint-factor-2026-11-02/results/decision.json](../findings/sources/disjoint-factor-2026-11-02/results/decision.json)

### DFACT-P03 [보존] Query 자체도 feature가 없는 사전 secondary 분할에서 학습된 factor predictor의 neural 대조 대비 개선이 나타났다.

Factor64.6030%/NLL0.600055, Mask56.9842%/0.679388, PCFI54.4846%/0.684093, Constant63.3009%/0.599397, Grid6-BP65.7650%/0.585340. Factor−Mask accuracy+7.6188%p CI[+1.7946,+13.6193], NLL−0.079333 CI[−0.137580,−0.023154]; Factor−PCFI accuracy+10.1184%p CI[+3.5331,+16.8865], NLL−0.084038 CI[−0.142503,−0.027776].

- 확대 금지: 같은 bank의 사전 secondary이며 primary gate를 대체하지 않는다. Constant 대비 accuracy/NLL CI는0을 포함한다. Grid6-BP 대비 NLL 차이+0.014715 CI[−0.004854,+0.037785]도0을 포함하며 동등성을 증명한 것은 아니다. 전체 context·query 조건의 확률 우위나99.5% missing 전이로 확대하지 않는다.
- 원문: [findings/sources/disjoint-factor-2026-11-02/results/query_without_features.json](../findings/sources/disjoint-factor-2026-11-02/results/query_without_features.json); [findings/sources/disjoint-factor-2026-11-02/results/secondary_partitions.json](../findings/sources/disjoint-factor-2026-11-02/results/secondary_partitions.json); [findings/sources/disjoint-factor-2026-11-02/runtime/secondary_partitions.py](../findings/sources/disjoint-factor-2026-11-02/runtime/secondary_partitions.py)

### DFACT-F01 [실패·중단] 현재 learned node/feature factor PFN이 강한 model-based inference와 경쟁하는 독립적인 메인 방법이다.

Neural accuracy gate=True지만 feature-removal gate=False, mechanism gate=False. Constant factor66.0053%/NLL0.621022 대비 learned65.4004%/0.628722: accuracy−0.6049%p CI[−3.4860,+1.9721], NLL+0.007700 CI[−0.056868,+0.098511]. Grid6-BP65.9557%/NLL0.555143 대비 learned NLL+0.073580 CI[+0.019579,+0.150308]. Grid6-exact predictive KL0.056253>.01. Single-graph forward median6.837ms vs Grid6-BP11.874ms로 약1.74배이며 사전2배 비용 기준 미충족.

- 판정: 정확도 개선과 입력 민감도는 보존하지만 현재 known-family 구성의 메인 메소드 확대는 보류한다.
- 원인·병목 [원인 미상]: 유의한 neural accuracy 개선만으로 learned potential의 필요성이나 classical 대비 품질·비용 우위를 확보하지 못했다. Potential fit, binary model-family 제약, finite BP/Bethe 근사, 학습량의 기여를 분리하지 않았다. 고유 원인은 미상이다.
- 닫은 범위: 현재8099-parameter·24-step BP·binary node/column latent·12-node/2-feature/2-relation known-family·256-update 구성을 범용 missing GraphPFN 메인 방법으로 승격하는 것.
- 닫지 않은 것: DFACT-P01/P02/P03의 실제 성과를 지우지 않는다. 다른 feature law, 추론 구조, variable schema, 큰 graph에서 학습한 potential의 효용을 기각하지 않는다. 미실행 후속을 실패로 세지 않는다.
- 재검토 조건: 작은 알려진 Gaussian family의 고정 factor/parameter grid로 충분하지 않은 과제에서, 실제 관측값과 context를 결합하는 학습 원리가 독립적 품질·비용 이득을 제공할 근거가 생길 때.
- 기준선: 동일 입력·예산의 실제 Mask/PCFI/FISF-trained PFN, fixed-factor BP, known-family Grid6 exact/BP, feature-only, LP/class prior. 실제 graph/schema 확장에서는 원래 Ephris/Wander와 해당 model-based 추론도 검토한다.
- 최소 재검증: 실패 bank에서 새 temperature/head/loss로 구제하지 않는다. 어떤 likelihood/function-family 또는 inference operation을 바꾸는지 먼저 명시하고 새 독립 bank에서 accuracy와 probabilistic quality·cost를 분리해 검증한다. Context total과 per-class, primary와 secondary를 구분한다.
- 원문: [findings/sources/disjoint-factor-2026-11-02/results/decision.json](../findings/sources/disjoint-factor-2026-11-02/results/decision.json); [findings/sources/disjoint-factor-2026-11-02/results/paired.json](../findings/sources/disjoint-factor-2026-11-02/results/paired.json); [findings/sources/disjoint-factor-2026-11-02/results/benchmark.json](../findings/sources/disjoint-factor-2026-11-02/results/benchmark.json); [findings/sources/disjoint-factor-2026-11-02/results/final_integrity.json](../findings/sources/disjoint-factor-2026-11-02/results/final_integrity.json)

### DFACT-L01 [교훈] Latent node class와 feature centroid를 함께 추론하는 원리는 기존 CSBM 연구와 겹친다.

CSBMInference는 latent community와 Gaussian feature centroid에 대한 AMP–BP, Bethe free entropy와 parameter estimation을 다룬다. 이번 binary column state·mask 처리·learned potential만으로 novelty를 주장하지 않는다.

- 원문: [papers/CSBMInference.yaml](../papers/CSBMInference.yaml); [findings/sources/disjoint-factor-2026-11-02/METHOD.md](../findings/sources/disjoint-factor-2026-11-02/METHOD.md)

### DFACT-U01 [미실행] 새로운 feature law·semantic schema·real graph·99.5% missing으로의 확장

이번 실행은12 nodes·binary classes·2 Gaussian features·2 relations다. Context1/2/4는 총 label 수이며 per-class가 아니다. 3/6 observed nodes는75/50% node missing이며99.5% 조건이 아니다. Variable feature count를 표현할 수 있는 코드여도 평가는2 features만 수행했다.

- 원문: [findings/sources/disjoint-factor-2026-11-02/protocol.json](../findings/sources/disjoint-factor-2026-11-02/protocol.json); [findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md](../findings/sources/disjoint-factor-2026-11-02/REPORT_KO.md)

## 실제 관측 feature 기여의 개입 분석

실험 조건: ACM/IMDB k20/class, 99.5% missing의 Full/No-observed/Shuffle 및 독립 FISF injection 통제.

### FEAT-F01 [실패·중단] IMDB의 PFN 우위는 남은 실제 관측 feature의 평균 추가 효용이다.

IMDB FISF Full44.55%→No-observed44.51%; controlled Full−Zero0.03%p. 제거 전후 argmax는 평균30.47% 바뀌었으므로 feature가 예측에 무영향이라는 뜻은 아니다.

- 판정: 해당 feature-reliability 동기 약화
- 원인·병목 [진단이 지지]: 평균상 실제 값 효과가 거의 상쇄됐다. 구조 입력 PFN과 synthetic-input 경로의 차이를 feature-only evidence로 오인할 수 없다. Node별 양·음의 효과가 왜 상쇄되는지는 미상이다.
- 닫은 범위: 현재 IMDB/99.5% 결과의 PFN−LP 격차를 실제 feature 복원 이득으로 설명
- 닫지 않은 것: 다른 결측률·graph와 query별 feature utility는 기각하지 않는다.
- 재검토 조건: 실제 관측 feature의 양·음 utility를 누수 없이 분리하거나, 유효 feature가 남는 조건을 특정한 방법이 나올 때.
- 기준선: Structure-only PFN, no-observed end-to-end, independent-injection controlled Full/Zero
- 최소 재검증: Mask·context·query·graph를 맞춘 제거/셔플 개입에서 실제 값 경로와 synthetic injection 변화를 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/feature_intervention.md](../findings/sources/graphpfn-2026-11-01/feature_intervention.md)

## Feature↔Label 양방향 추론

실험 조건: 10K 규모 quadratic prototype, 합성1536 tasks, 2 seeds, 384 updates; 공식 GraphPFN 아님.

### COUP-F01 [실패·중단] Label→feature 반복 피드백이 일방향 추론보다 유용하다.

Matched 일방향72.72%, 양방향72.66%; 차이−0.06%p CI[−0.50,+0.39]. Shape shift 차이−0.01%p CI[−0.52,+0.55].

- 판정: 피드백 증분 미확인
- 원인·병목 [진단이 지지]: 피드백의 추가 정보를 확보하지 못했다. 관측 feature가 전혀 없으면 latent H가0에 머물고 피드백 제거와 출력이 같았다. 이 구조적 퇴화가 모든 조건의 미개선을 설명하지는 않는다.
- 닫은 범위: 현재 quadratic coupling을 더 반복·확대하는 방향
- 닫지 않은 것: 비선형 label-conditioned feature inference와 다른 유효 prior는 평가하지 않았다.
- 재검토 조건: 라벨 피드백이 실제로 새로운 feature 상태를 만들고 그 정보가 일방향 모델에 없음을 보이는 추론 연산이 나올 때.
- 기준선: 동일 예산 일방향 모델, 피드백 제거, 강한 context ridge baseline
- 최소 재검증: Zero-observation 퇴화와 nonzero 조건을 나눠 같은 초기화·task·예산으로 실제 예측 증분을 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

## Feature-law adaptive observed-cell Factor

실험 조건: 동일 frozen graph checkpoint 위에서 Original/static/adaptive cell만 바꾼3seeds CE. 새 prior로 Mask/PCFI도 실제 학습. Final192 graphs 중 off-diagonal parameter tuple96개, 자기 피처 없는 query accuracy primary.

### FLAW-F01 [실패·중단] 관측 cell의 부호 제약을 제거하고 새 graph의 column 통계에 조건부인 factor를 학습하면 static과 기존 Factor보다 자기 피처가 없는 query 분류가 좋아진다.

original: accuracy64.628114%/NLL0.643004/Brier0.445450/referenceKL0.022515; static: accuracy64.412478%/NLL0.643406/Brier0.446408/referenceKL0.021705; adaptive: accuracy64.467455%/NLL0.650797/Brier0.449291/referenceKL0.027830; fixed_exact: accuracy63.406498%/NLL0.631744/Brier0.440029/referenceKL0.004696; fixed_bp: accuracy64.081101%/NLL0.646232/Brier0.446208/referenceKL0.023510; mask_pfn: accuracy53.488343%/NLL0.693767/Brier0.499510/referenceKL0.068593; pcfi_pfn: accuracy52.969715%/NLL0.693928/Brier0.499014/referenceKL0.068545. Adaptive−static accuracy%p +0.054977 CI[-0.976294,+1.037881], adaptive−original -0.160659 CI[-1.202481,+0.821535]. 두 primary CI 모두0 포함. Gaussian guard: adaptive−original +0.543981 CI[-0.842978,+2.276235], adaptive−static +0.866127 CI[-0.434124,+2.338011]. Adaptation gate=False, Gaussian guard=True, competitive gate=False. Single-graph cost adaptive6.759ms, fixed exact8.861ms, fixed BP10.077ms.

- 판정: 사용자가 정한 primary accuracy 기준을 충족하지 못했으므로, 현재 cell 함수 교체만으로 독립 메인 메소드가 된다는 가설은 닫는다. 추가 head/loss/temperature/updates로 이 bank에서 구제하지 않는다.
- 원인·병목 [원인 미상]: 부호 제약 제거와 관측 column 통계 conditioning만으로 이 예산에서 분류 이득을 확인하지 못했다. 관측 정보량·통계 요약·함수족·비선형 최적화·finite BP/Bethe·학습량의 기여를 분리하지 않아 고유 원인은 미상이다. CI에0을 포함한 것을 효과0 또는 동등성의 증명으로 쓰지 않는다.
- 닫은 범위: 12-node binary native graph/2relations/2features, context총1/2/4개와 observed nodes3/6개 완전 분리, Gaussian mean/Student-t3 mean/variance law를 학습에 모두 넣고 diagonal train/off-diagonal heldout parameter tuples를 비교한 frozen-graph cell-only256-update CE pilot.
- 닫지 않은 것: 기존 disjoint feature/label 연결 연산·signed likelihood·자기 role-update와 이전 Factor/teacher의 국소 개선은 보존한다. 모든 학습된 feature factor, unseen feature laws, 가변 feature/schema, 더 큰 실제 graph,99.5% missing의 효과를 기각한 결과는 아니다. 이번에도 law 이름은 이미 보았다.
- 재검토 조건: 관측 통계로 pointwise rho만 조절하는 것과 다른, 현재 함수족이 놓치는 feature–relation 의존성을 학습·전이하는 구체적인 연산이 등장하고 강한 명시적 법칙 주변화보다 유리할 근거가 생길 때.
- 기준선: Original sign-constrained Factor, parameter-matched Flexible-static/adaptive, 동일 frozen graph의 full normalized law/parameter/orientation fixed-exact 및 same24-step fixed-BP, 실제 새 prior로 학습한 Mask/PCFI public PFN. Accuracy/NLL/Brier/reference KL과 end-to-end cost를 분리한다.
- 최소 재검증: 이 final bank를 추가 tuning에 쓰지 않는다. 새 연산의 구체적인 차이를 먼저 고정하고 독립 base graphs에서 자기 피처 없는 query accuracy를 primary로 비교하며 Gaussian 보존과 고정 추론 품질–비용을 함께 확인한다.
- 원문: [findings/sources/feature-law-2026-11-02/REPORT_KO.md](../findings/sources/feature-law-2026-11-02/REPORT_KO.md); [findings/sources/feature-law-2026-11-02/METHOD.md](../findings/sources/feature-law-2026-11-02/METHOD.md); [findings/sources/feature-law-2026-11-02/protocol.json](../findings/sources/feature-law-2026-11-02/protocol.json); [findings/sources/feature-law-2026-11-02/results/statistics.json](../findings/sources/feature-law-2026-11-02/results/statistics.json); [findings/sources/feature-law-2026-11-02/results/cost.json](../findings/sources/feature-law-2026-11-02/results/cost.json); [findings/sources/feature-law-2026-11-02/results/audit.json](../findings/sources/feature-law-2026-11-02/results/audit.json); [findings/sources/feature-law-2026-11-02/results/preflight.json](../findings/sources/feature-law-2026-11-02/results/preflight.json); [findings/sources/feature-law-2026-11-02/results/runtime_amendment.json](../findings/sources/feature-law-2026-11-02/results/runtime_amendment.json)

### FLAW-P01 [보존] 세 feature law를 포함한 새 bank에서도 adaptive Factor는 같은 prior로 실제 학습한 공개 Mask/PCFI PFN보다 자기 피처 없는 query accuracy가 높았다.

Adaptive−Mask accuracy%p +10.979112 CI[+7.342324,+14.631011], NLL -0.042970 CI[-0.081804,-0.001247]. Adaptive−PCFI accuracy%p +11.497740 CI[+7.718815,+15.256125], NLL -0.043131 CI[-0.082211,-0.001461].

- 확대 금지: 두 공개 PFN 비교는 보조·multiplicity 미보정이다. Adaptive는 primary에서 static과 original을 이기지 못했으므로 adaptive cell의 독립적인 기여나 논문 메소드 성공이 아니다. 공개 PFN과 Factor는 architecture/사전학습/graph factor 경로가 달라 정확한 이득 원인을 분리하지 않았다. 이번에는 feature 제거 개입을 하지 않았으므로 이 차이만으로 variance feature를 활용했다고 단정하지 않는다. 고정 law 주변화보다 정확도가 낮은 것이라고 주장하지도 않는다. Adaptive−fixed exact accuracy CI도0을 포함한다.
- 원문: [findings/sources/feature-law-2026-11-02/REPORT_KO.md](../findings/sources/feature-law-2026-11-02/REPORT_KO.md); [findings/sources/feature-law-2026-11-02/results/statistics.json](../findings/sources/feature-law-2026-11-02/results/statistics.json); [findings/sources/feature-law-2026-11-02/results/audit.json](../findings/sources/feature-law-2026-11-02/results/audit.json)

### FLAW-L01 [교훈] 점별 cell 함수의 표현력 제약 제거와, 새 graph에서 feature law를 추론해 분류를 개선하는 것은 다른 검증이다.

분산 차이의 참 half log-likelihood ratio는 x²의 함수라 큰 양수와 큰 음수가 같은 evidence를 준다. Original x*positiveNN은 이를 그대로 표현할 수 없지만 새 static/adaptive 함수는 짝수 evidence도 허용한다. 그 구조 변경 이후에도 primary adaptive−static과 adaptive−original accuracy 우위는 확인되지 않았다.

- 원문: [findings/sources/feature-law-2026-11-02/METHOD.md](../findings/sources/feature-law-2026-11-02/METHOD.md); [findings/sources/feature-law-2026-11-02/REPORT_KO.md](../findings/sources/feature-law-2026-11-02/REPORT_KO.md); [findings/sources/feature-law-2026-11-02/results/preflight.json](../findings/sources/feature-law-2026-11-02/results/preflight.json); [findings/sources/feature-law-2026-11-02/results/statistics.json](../findings/sources/feature-law-2026-11-02/results/statistics.json)

## Frozen GraphPFN joint feature-completion marginalization

실험 조건: 2026-11-01 시작·11-02 완료. Fresh32 official-prior small graphs, 6 conditions, frozen Mask Plain3 seeds, 6 input arms×4 forwards, 추가 optimizer update0. Bayesian GP surrogate의 exact conditional이며 공식 SCM posterior 아님.

### FMARG-F01 [실패·중단] 단일 대치 대신 graph-correlated feature completion의 예측을 평균하면 두 주 결측 조건에서 point/diagonal/zero/PCFI 대조를 모두 넘는다.

uniform NLL zero0.92888, PCFI0.93539, mean0.94890, joint0.91589, diagonal0.91596; clustered NLL zero0.93084, PCFI0.93279, mean0.94698, joint0.91352, diagonal0.91743。Joint−Mean: Uniform-0.03301 CI[-0.06064,-0.01167], Clustered-0.03346 CI[-0.06395,-0.00811]。Joint−Diagonal: Uniform-0.00007 CI[-0.00394,+0.00354], Clustered-0.00391 CI[-0.00697,-0.00087]。

- 판정: 현재 frozen surrogate completion operator의 사전 primary gate 미통과·확대 중단
- 원인·병목 [원인 미상]: 명시한 Gaussian-process surrogate와 4-sample frozen PFN 연산에서 필요한 동시 우위가 없었다. Feature-law misspecification, zero-fill 학습 PFN의 입력 활용 한계, 유한 Monte Carlo, 실제 feature 정보 부족의 원인은 분리하지 못했다.
- 닫은 범위: 현재 graph Gaussian surrogate, 고정 zero+mask 학습 PFN, 4 completion·동일 생성 family fresh32 graphs에서 이 연산을 확대해 메인 메소드로 삼는 것
- 닫지 않은 것: Context labels에 조건화한 feature law, joint predictive training, cross-feature dependency, 정확한 SCM conditional, 실제 graph·unseen schema는 평가하지 않았다. GP likelihood는 공식 SCM posterior가 아니다.
- 재검토 조건: 관측된 feature evidence와 graph의 joint missing-feature law를 검증 가능하게 연결하고, point fill 및 같은 예산의 covariance 제거·구조 random-feature 대조를 안정적으로 넘는 추론 연산이 나올 때.
- 기준선: 동일 PFN의 zero/PCFI/mean/native-NaN·independent sampling·context prior. 후속 positive 주장은 별도의 PCFI/FISF-trained 대조, 비조건부 구조 random features, GCNmf/GSPN/PEMix와의 차별화 필요.
- 최소 재검증: Source·data·예산을 먼저 고정한 새 graph에서 feature conditional의 품질과 predictive marginalization의 증분을 분리한다. Frozen replacement와 distributional-input 재학습을 같은 성과로 세지 않는다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

### FMARG-P02 [보존] 여러 completion의 확률 평균은 같은 PFN의 단일 mean 대치와 PCFI 입력보다 낮은 NLL을 얻었다.

Joint−Mean NLL: Uniform-0.03301 CI[-0.06064,-0.01167], Clustered-0.03346 CI[-0.06395,-0.00811]。Joint−PCFI: Uniform-0.01950 CI[-0.03171,-0.00815], Clustered-0.01926 CI[-0.03135,-0.00633]。

- 확대 금지: 사전 primary gate 통과나 새 학습법·정확한 posterior 추론·관측 feature 실제 정보 활용·실제 graph 전이·PCFI-trained 모델 초과로 확대하지 않는다. 현재 PCFI는 같은 zero-input-trained checkpoint의 입력 대조다. 일반 ensemble과의 forward 수는 맞췄지만 구조 random-feature 설명은 분리하지 않았다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

### FMARG-P03 [보존] Clustered missing에서는 rho 내 node covariance 보존이 diagonal sampling보다 작은 추가 NLL 이득을 보였다.

Joint−Diagonal NLL: Clustered-0.00391 CI[-0.00697,-0.00087], Uniform-0.00007 CI[-0.00394,+0.00354], k1+0.00369 CI[-0.00590,+0.01254]。

- 확대 금지: 전 결측 조건의 covariance 우위, k1 해결, feature law 정확성 또는 실제 observed-value 정보 활용이라고 주장하지 않는다. Diagonal arm도 graph 공유 rho에 의한 의존성을 유지한다. 다중 비교 보정 없는 fixed-RNG 소형 pilot이다.
- 원문: [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md); [findings/sources/feature-marginal-2026-11-02/paired.json](../findings/sources/feature-marginal-2026-11-02/paired.json)

### FMARG-L01 [교훈] 분포형 input을 적용한 frozen 연산의 성과와 Bayesian posterior·새 학습법을 구분한다.

Completion law는 observed-only RMS 좌표의 Gaussian surrogate이며 context labels에는 조건화하지 않는다. PFN은 zero+mask로 학습했던 checkpoint다. PCFI/native-NaN도 같은 checkpoint의 입력 대조이며 각 입력으로 별도 재학습한 모델 성과가 아니다. Point와 sampling 모두 동일4회 PFN forward의 확률을 평균했다.

- 원문: [findings/sources/feature-marginal-2026-11-02/README.md](../findings/sources/feature-marginal-2026-11-02/README.md); [findings/sources/feature-marginal-2026-11-02/protocol.json](../findings/sources/feature-marginal-2026-11-02/protocol.json)

### FMARG-L02 [교훈] 노드 간 covariance의 진단은 per-cell variance와 일반 ensemble 효과를 통제한다.

Joint와 diagonal에 같은 rho draws·조건부 평균·대각 분산·standard normal draws를 주고 rho 내 off-diagonal covariance만 바꿨다. 두 arm 모두 rho는 graph 공유다. 관측값 clamp·conditional moment node permutation·12000-draw covariance sanity를 확인했다. 이것이 새 정보나 method superiority를 입증하지는 않는다.

- 원문: [findings/sources/feature-marginal-2026-11-02/preflight.json](../findings/sources/feature-marginal-2026-11-02/preflight.json); [findings/sources/feature-marginal-2026-11-02/experiment.py](../findings/sources/feature-marginal-2026-11-02/experiment.py)

### FMARG-U01 [미실행] Joint feature law의 학습과 probabilistic graph 방법 대비 독립적 기여

새 neural law·auxiliary loss를 학습하지 않았다. Context-conditioned density, 별도 PCFI/FISF 학습 checkpoint, 비조건부 구조 random input, PEMix/GSPN 직접 재현, 실제 heterograph 전이는 이번 실행에서 비교하지 않았다. PEMix를 하네스에 추가한 것은 재검토 단서이지 기존 실패 해결 판정이 아니다.

- 원문: [findings/sources/feature-marginal-2026-11-02/PEMix_RECHECK.md](../findings/sources/feature-marginal-2026-11-02/PEMix_RECHECK.md); [findings/sources/feature-marginal-2026-11-02/REPORT.md](../findings/sources/feature-marginal-2026-11-02/REPORT.md)

## Label-context와 observed-feature memory 분리

실험 조건: 새64 train/64 test base graphs, public GraphPFN1.3, 4 arms×3 fixed seeds, arm/seed당512 성공 updates. 공식 및 controlled-crossed 과제의 같은 stream으로 학습.

### FMEM-F01 [실패·중단] Label context와 관측 feature memory를 분리하면 기존 context-only 및 all-node feature attention보다 accuracy가 높다.

Primary accuracy context52.3336%, all_nodes51.9552%, observed52.2888%, PCFI53.9246%, LP53.3203%. Observed−context −0.044759%p CI[−1.501516,+1.460826], −all_nodes +0.3337%p CI[−0.9481,+1.6135], −PCFI −1.6357%p CI[−3.4202,+0.0633]. Observed NLL0.692156은 context0.687776 대비 +0.004380 CI[+0.000906,+0.007896], PCFI0.683446 대비 +0.008710 CI[+0.003985,+0.013666] 높았다.

- 판정: 현재 feature-group routing의 사전 accuracy gate 미통과. NLL은 기존 context 및 PCFI보다 악화.
- 원인·병목 [원인 미상]: 관측 전용 memory에서 추가 예측 가치를 회수하지 못했다. Frozen sample-attention projection, native feature-group 표현, sparse 입력 정보량,512-update prior fit 가운데 고유 원인을 분리하지 않았다. Context-only 제한이 이전 실패의 원인이라는 가설도 확인되지 않았다.
- 닫은 범위: 현재 공개 frozen-transformer GraphPFN의 native 두 좌표 group에 적용한 C-only/all-node/observed-key routing을 메인 방법으로 확대.
- 닫지 않은 것: 완전한 sparse cell-token 모델, attention projection까지 새로 학습하는 구조, 다른 prior·규모·실제 graph는 평가하지 않았다. MCP-P02의 +3.43%p는 이 실험으로 재현하거나 기각한 것이 아니다.
- 재검토 조건: Label이 없는 노드의 실제 관측값을 읽는 연산이 strong all-node memory와 PCFI/FISF-trained PFN보다 같은 예산에서 유용하다는 근거가 생길 때.
- 기준선: 동일 초기화·성공 update·prior의 C-only, all-node memory, PCFI/FISF-trained PFN, LP, class prior. Observed-group과 실제 sparse cell 연산을 구분.
- 최소 재검증: 새 graph bank에서 accuracy primary를 먼저 고정하고 no-observed/shuffled-observed와 전체 비용을 함께 비교한다. 단순 KV mask 변경이나 추가 학습만의 효과로 재개하지 않는다.
- 원문: [findings/sources/feature-memory-2026-11-02/REPORT_KO.md](../findings/sources/feature-memory-2026-11-02/REPORT_KO.md); [findings/sources/feature-memory-2026-11-02/results/paired.json](../findings/sources/feature-memory-2026-11-02/results/paired.json); [findings/sources/feature-memory-2026-11-02/results/interventions.json](../findings/sources/feature-memory-2026-11-02/results/interventions.json)

### FMEM-L01 [교훈] 더 넓은 key/value 접근의 구현 가능성과 실제 feature utility를 구분한다.

12개 모델, 총6144 성공 updates, 저장 예측16896개를 감사했다. Frozen parameters와 source/bank hashes를 보존했다. Observed-group routing은 dense attention mask를 바꾼 구현으로 sparse kernel의 메모리·속도 이점은 주장하지 않는다. Clustered90/off-diagonal에서 관측값 shuffle은 observed arm NLL을 −0.007934 CI[−0.014027,−0.002201] 낮춰 유익한 feature alignment 회수를 지지하지 않았다.

- 원문: [findings/sources/feature-memory-2026-11-02/results/preflight.json](../findings/sources/feature-memory-2026-11-02/results/preflight.json); [findings/sources/feature-memory-2026-11-02/results/audit.json](../findings/sources/feature-memory-2026-11-02/results/audit.json); [findings/sources/feature-memory-2026-11-02/results/final_integrity.json](../findings/sources/feature-memory-2026-11-02/results/final_integrity.json); [findings/sources/feature-memory-2026-11-02/results/interventions.json](../findings/sources/feature-memory-2026-11-02/results/interventions.json)

### FMEM-U01 [미실행] NEXT-U02/REOPEN-U01 중 full sparse-cell 아키텍처와 end-to-end feature-attention 학습

이번 실행은 native feature-pair grouping을 유지한 공개 PFN routing kill test다. 원래 attention projection은 고정했고 별도의 sparse token encoder나 Ephris/Wander 전체를 재현하지 않았다.

- 원문: [findings/sources/feature-memory-2026-11-02/protocol.json](../findings/sources/feature-memory-2026-11-02/protocol.json); [findings/sources/feature-memory-2026-11-02/experiment.py](../findings/sources/feature-memory-2026-11-02/experiment.py)

## Exact Grid6 teacher for disjoint-feature PFN

실험 조건: 기존2048 train tasks/초기화/task order/256 updates를 유지해 Mask·Factor 각3 seeds를 exact predictive KL로 새 학습. 기존 CE9개를 fresh96 graph bank에서 재평가. Primary는 matched disjoint48 graphs의 공개 Mask teacher KL이며 Factor는 secondary다.

### G6T-F01 [실패·중단] 기존 생성 prior의 exact Grid6 predictive를 가르치면 현재 공개 Mask GraphPFN이 teacher 추론을 더 잘 압축한다.

Mask CE accuracy55.3475%/NLL0.678973/Grid6 KL0.104788, distill54.6255%/0.679253/0.104134. Distill−CE primary KL−0.000654 CI[−0.002034,+0.000583], accuracy−0.7220%p CI[−3.9174,+2.5422], NLL+0.000280 CI[−0.002496,+0.002959]. Primary teacher-effect gate=False. Mask forward median59.735ms vs Grid6-BP12.425ms, matched KL>.01로 competitive compression gate=False.

- 판정: 현재 공개 Mask 구성에 exact teacher 감독만 교체하는 갈래의 메인 방법 확대는 보류한다. Factor secondary의 별도 성과는 보존한다.
- 원인·병목 [원인 미상]: 조밀한 exact 감독만으로 현재 학생의 predictive gap을 줄이는 확실한 이득을 확인하지 못했다. Input 경로·표현력·최적화·frozen backbone·학습량의 기여를 분리하지 않아 고유 원인은 미상이다.
- 닫은 범위: 12-node binary/2-relation/2-Gaussian-feature,50·75% node missing, 원래 frozen-backbone/adapters와256-update 예산에서, sampled-label CE만 exact unknown-parameter Grid6 predictive KL로 교체한 공개 Mask 학생의 확장.
- 닫지 않은 것: G6T-P01/P02의 Factor 효과와 이전 disjoint-factor 정확도 개선은 보존한다. Distillation 전체, fully trained 다른 구조, 다른 feature law·real graph·unseen schema·99.5% missing·큰 graph의 teacher 학습은 기각하지 않는다. 기존 feature-free JR distillation과 다른 이번 범위를 구분한다.
- 재검토 조건: 학생이 teacher의 관계·관측 정보 추론을 표현하고 회수할 수 있도록 바꾸는 구체적 학습 또는 추론 연산이, 강한 기준선 대비 효율적 예측을 제공할 근거가 생길 때.
- 기준선: 동일 architecture/initialization/tasks/updates의 sampled CE, 실제 PCFI 입력으로 학습한 PFN, Factor CE/distill, fixed-mode factor BP, exact Grid6/Grid6-BP. Teacher 생성과 학생 학습 비용을 포함해 보고한다.
- 최소 재검증: 실패 bank에서 temperature·새 head·loss 혼합·학습량을 조정해 구제하지 않는다. 무엇을 바꾸는지 먼저 고정하고 새 graph/schema에서 primary KL·accuracy·NLL/Brier와 비용을 분리한다. Student architecture 변경과 teacher 변경을 동시에 하지 않는다.
- 원문: [findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md](../findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md); [findings/sources/grid6-teacher-2026-11-02/protocol.json](../findings/sources/grid6-teacher-2026-11-02/protocol.json); [findings/sources/grid6-teacher-2026-11-02/results/paired.json](../findings/sources/grid6-teacher-2026-11-02/results/paired.json); [findings/sources/grid6-teacher-2026-11-02/results/decision.json](../findings/sources/grid6-teacher-2026-11-02/results/decision.json); [findings/sources/grid6-teacher-2026-11-02/results/benchmark.json](../findings/sources/grid6-teacher-2026-11-02/results/benchmark.json); [findings/sources/grid6-teacher-2026-11-02/results/final_integrity.json](../findings/sources/grid6-teacher-2026-11-02/results/final_integrity.json)

### G6T-P01 [보존] Factor PFN에서는 exact Grid6 predictive distillation이 같은 CE 학생보다 teacher 분포에 대한 KL을 낮췄다.

동일 새 bank, Factor CE accuracy69.8468%/NLL0.575272/KL0.067257, distill69.5877%/0.568642/0.049731. Teacher KL 차이−0.017526 CI[−0.026031,−0.010538], 약26.1% 감소. Fixed-mode factor BP 대비 teacher KL−0.057650 CI[−0.089157,−0.031682]. CE 대비 accuracy−0.2590%p CI[−1.2112,+0.7551], NLL−0.006630 CI[−0.023930,+0.007280].

- 확대 금지: Factor는 사전 secondary라 공개 Mask primary 실패를 대체하지 않는다. Accuracy·실제 sampled-label NLL의 확실한 개선이나 calibration 개선으로 확대하지 않는다. Query 자신도 feature가 없는 사전 secondary에서는 CE64.7164%→distill63.5556%, accuracy−1.1609%p CI[−2.4190,−0.0482]로 오히려 하락했다. 이 보조 비교는 multiplicity 미보정이며 NLL 차이+0.004205 CI[−0.002231,+0.010960]는0을 포함한다. Grid6-BP 대비 전체 accuracy+1.2594%p CI[−0.9537,+3.5799], NLL+0.016771 CI[−0.016408,+0.053202]는 우위나 동등성의 증명이 아니다. Factor median8.764ms vs Grid6-BP12.425ms의 약1.42배와 teacher KL0.049731은 사전 .01/2배 경쟁 기준에 못 미친다. 실제 graph·unseen schema 전이나 novelty를 주장하지 않는다.
- 원문: [findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md](../findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md); [findings/sources/grid6-teacher-2026-11-02/results/summary.json](../findings/sources/grid6-teacher-2026-11-02/results/summary.json); [findings/sources/grid6-teacher-2026-11-02/results/paired.json](../findings/sources/grid6-teacher-2026-11-02/results/paired.json); [findings/sources/grid6-teacher-2026-11-02/results/benchmark.json](../findings/sources/grid6-teacher-2026-11-02/results/benchmark.json); [findings/sources/grid6-teacher-2026-11-02/results/query_without_features.json](../findings/sources/grid6-teacher-2026-11-02/results/query_without_features.json)

### G6T-P02 [보존] 새 disjoint bank의 사전 보조 개입에서 기존 CE와 distilled Factor 모두 실제 관측feature 활용 효과를 보였다.

Matched disjoint context2/observed6에서 distilled Factor의 feature 제거 accuracy−11.9213%p CI[−19.0972,−4.8611], shuffle−20.7176%p CI[−28.3594,−12.7315], graph permutation−29.0509%p CI[−39.5833,−17.9398]. 기존 CE Factor도 같은 bank에서 제거−12.0370%p CI[−19.6759,−4.2824].

- 확대 금지: Feature 활용 자체가 distillation으로 새로 생겼다는 뜻이 아니다. CE 모델에서도 나타났다. 보조 개입은 primary Mask 성공을 대체하지 않는다. 제거에 대한 NLL/Brier 차이 CI는0을 포함한다. 이전 bank DFACT의 feature-removal 미통과 결과를 삭제하지 않으며 모든결측률·실제 graph로 확대하지 않는다.
- 원문: [findings/sources/grid6-teacher-2026-11-02/results/interventions.json](../findings/sources/grid6-teacher-2026-11-02/results/interventions.json); [findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md](../findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md); [findings/sources/grid6-teacher-2026-11-02/protocol.json](../findings/sources/grid6-teacher-2026-11-02/protocol.json)

### G6T-L01 [교훈] 생성 prior 재사용과 exact predictive teacher 사용은 다른 변경이며, 이번 효과는 후자에 한정한다.

기존 master 생성기가 이미 Grid6와 같은 six-support density/signal·joint role·Gaussian orientation prior로 graph와 feature를 생성했다. 이번에는 생성기·train bank를 유지하고 단일 sampled query label 대신 관측에 조건부인 exact Grid6 query marginal을 감독으로 사용했다.

- 원문: [findings/sources/grid6-teacher-2026-11-02/METHOD.md](../findings/sources/grid6-teacher-2026-11-02/METHOD.md); [findings/sources/grid6-teacher-2026-11-02/protocol.json](../findings/sources/grid6-teacher-2026-11-02/protocol.json); [findings/sources/grid6-teacher-2026-11-02/results/preflight.json](../findings/sources/grid6-teacher-2026-11-02/results/preflight.json)

### G6T-L02 [교훈] Factor가 확보한 evidence 연결 연산과 그 연산을 학습하는 독립적인 필요성을 분리한다.

Context label→graph factor→관측feature node→shared feature-column latent→다른 node class의 경로가 실제 구현됐다. 새 bank의 feature 제거 하락은 CE Factor에도 있었다. Distillation의 추가 확인 효과는 teacher predictive KL 감소이며 실제 분류·분포 전이의 일반적 우위는 미확인이다.

- 원문: [findings/sources/factor-position-2026-11-02/user_review.md](../findings/sources/factor-position-2026-11-02/user_review.md); [findings/sources/disjoint-factor-2026-11-02/METHOD.md](../findings/sources/disjoint-factor-2026-11-02/METHOD.md); [findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md](../findings/sources/grid6-teacher-2026-11-02/REVIEW_KO.md)

## Native heterogeneous 입력·기존 relational FM 대조

실험 조건: DBLP native block missing 및 ACM/IMDB/Freebase 개발용 비교. Adapted implementation과 공식 모델 성과를 구분.

### HET-F01 [실패·중단] Adapted HetGFD가 cross-type path 입력보다 큰 증분을 준다.

DBLP k=5 full: cross-type 89.96%, adapted HetGFD 89.92%; drop_A 90.18/90.19%. k=20 full은 둘 다93.55%.

- 판정: 새 메소드 headroom 부족
- 원인·병목 [진단이 지지]: 관계 경로와 남은 native source가 이미 강한 신호를 전달했다. 격차가 작다는 관찰은 확인했지만 redundancy만이 원인이라는 인과 분해는 하지 않았다.
- 닫은 범위: 현재 DBLP adapted diffusion의 작은 차이를 범용 missing hetero-PFN 기여로 확대
- 닫지 않은 것: 원래 HetGFD/HGT 재현이나 다른 schema에서의 결론이 아니다. 다중 block 소실 조건은 별도 여지가 있다.
- 재검토 조건: 타입별 feature 의미와 관측 pattern을 보존하며 기존 path redundancy를 넘어서는 장치가 나올 때.
- 기준선: Native cross-type paths + frozen GraphPFN, 강한 attribute-free metapath LP
- 최소 재검증: 파생 helper feature 누수를 제거하고, 실제 native block graph에서 missing/full 대비와 최강 구조 기준선을 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/native_blocks.md](../findings/sources/graphpfn-2026-11-01/native_blocks.md)

### HET-P01 [보존] Typed path 정보를 column으로 전달해도 기존 PFN이 이미 활용한다.

Gate1-P의 B5g는 k=3의 세 decision graph에서 사전 val-oracle−1%p 기준을 충족했다. 새 learned path adapter를 그 격차만으로 정당화하지 않았다.

- 확대 금지: Val oracle은 배포 모델이 아니며, 새로운 학습된 heterogeneous PFN의 성과도 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/hetero_gate1p.md](../findings/sources/graphpfn-2026-11-01/hetero_gate1p.md); [findings/sources/graphpfn-2026-11-01/hetero_gate0.md](../findings/sources/graphpfn-2026-11-01/hetero_gate0.md)

### HET-L01 [교훈] 기존 relational FM도 현재 node-classification adapter에서는 자동으로 missing에 강하지 않았다.

RDB-PFN의 ACM k20 mean_count_depth5 full81.53%/missing66.55%, official_depth2 full74.87%/missing35.18%. RT budget 표는 64-query·1 seed 대조로 제한된다.

- 원문: [findings/sources/graphpfn-2026-11-01/rdb.csv](../findings/sources/graphpfn-2026-11-01/rdb.csv); [findings/sources/graphpfn-2026-11-01/rt.csv](../findings/sources/graphpfn-2026-11-01/rt.csv); [findings/sources/graphpfn-2026-11-01/rt_coverage.csv](../findings/sources/graphpfn-2026-11-01/rt_coverage.csv); [findings/sources/graphpfn-2026-11-01/history.md](../findings/sources/graphpfn-2026-11-01/history.md)

### HET-L02 [교훈] 미리 집계된 helper feature를 독립 native 관측값으로 취급하지 않는다.

ACM/IMDB helper feature는 target feature에서 파생된 집계일 수 있음을 확인했다. DBLP는 서로 다른 native source block으로 별도 처리했다.

- 원문: [findings/sources/graphpfn-2026-11-01/native_blocks.md](../findings/sources/graphpfn-2026-11-01/native_blocks.md)

## PCFI/FISF 입력 확장

실험 조건: Cora/CiteSeer 개발 그래프와 PubMed secondary, frozen GraphPFN, context/mask 일치.

### VIEW-F01 [실패·중단] Support metadata·다중 전파 view·PCFI/FISF 입력 결합이 범용 증분을 준다.

8조건 평균 PCFI 69.65%, PCFI/FISF views 69.73%, multiscale 69.61%. 사전 +1%p 양쪽 parent 초과 기준을 만족한 후보는 없었다.

- 판정: 전체 조건 진입 기준 미통과
- 원인·병목 [원인 미상]: 출력에 보인 증분이 작고 조건별 부호가 달랐다. 표현 정보 중복, projection 손실, frozen backbone의 활용 한계 중 무엇이 주원인인지는 분리하지 못했다.
- 닫은 범위: 현재 frozen 저차원 support/view 추가를 범용 메소드로 확대
- 닫지 않은 것: Native cell-wise provenance encoder를 학습하는 방법과 입력 자체를 바꾸는 학습은 평가하지 않았다.
- 재검토 조건: 관측 provenance를 정보 손실 없이 표현·학습하고 equal-width 값 대조를 넘는 장치가 나올 때.
- 기준선: PCFI/FISF parents, equal-width value controls, 같은 비용의 single predictor
- 최소 재검증: 같은 차원·추론 비용으로 새 실제 graph에서 parent 둘과 metadata 없는 값 대조를 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md); [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md)

### VIEW-P01 [보존] PubMed whole-node secondary에서 input view의 작은 긍정 신호

평균 PCFI 70.05%, FISF 71.50%, input views 73.00%, prediction average 71.97%.

- 확대 금지: 한 graph의 secondary이며 전체 조건 우위·novelty·독립 다중 domain 검증으로 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/pubmed.md](../findings/sources/graphpfn-2026-11-01/pubmed.md); [findings/sources/graphpfn-2026-11-01/views.md](../findings/sources/graphpfn-2026-11-01/views.md)

## Fixed internal PCFI independent-graph confirmation

실험 조건: Reused six checkpoints;0 updates; same official generator family;128 new base graphs; missing-query accuracy primary.

### IPCFIC-F01 [실패·중단] 개발 bank에서 선택한 내부 고정 PCFI의 외부 PCFI 대비 정확도 이득이 추가 학습 없이 새 graph bank에서 재현된다.

external:accuracy 49.821540%/NLL 0.856885/Brier 0.552034; internal_fixed:accuracy 50.041098%/NLL 0.856498/Brier 0.552432; fixed-external accuracy%p +0.219558, 95% CI [-0.331926, +0.796665], NLL -0.000388, 95% CI [-0.003777, +0.002808], Brier +0.000398, 95% CI [-0.001679, +0.002518]. Primary=False, full2pp guard=True, continuation=False.

- 판정: 새 bank에서 고정 내부 PCFI의 정확도 우위를 사전 기준대로 확인하지 못했다. FP/FISF 내부화 후속 학습은 시작하지 않는다. 이 결과가 모든 내부 PCFI 방식의 무효나 동등성을 의미하지는 않는다.
- 원인·병목 [원인 미상]: 사전에 고정한 새 bank accuracy 확인 기준을 통과하지 못했다. 고유 원인은 분리하지 않아 미상이며, CI가0을 포함한 경우 효과 없음/동등성을 증명한 것으로 쓰지 않는다.
- 닫은 범위: 같은 official graph/SCM family와 training seeds11/12/13에서, 현재100-step literal PCFI의 fixed25-step interleaving을 원 checkpoint 그대로 사용한 독립128graph accuracy confirmation 및 이를 근거로 한 FP/FISF 자동 확장.
- 닫지 않은 것: IPCFI-F01은 context router의 원 판정으로 유지한다. 개발 bank의 fixed 양의 점추정은 삭제하지 않는다. 모든 내부화, native missing interface, 다른 학습/관측 연산, 실제 graph 또는 unseen schema 일반을 기각하지 않는다.
- 재검토 조건: 같은 배치를 반복하거나 bank를 tuning하는 대신, 현재 checkpoint/입력/연산의 한계를 바꾸는 구체적인 근거가 생기고 강한 외부 PCFI 대조와 새 training seeds 및 독립 graph 검증을 사전 설계할 수 있을 때.
- 기준선: 동일 public PFN 초기화/학습 budget의 외부 actual PCFI, 내부 fixed, context/strict label-blind 개발 대조. Primary missing-query accuracy와 NLL/Brier/full guard 및 CPU→CPU 비용을 분리한다.
- 최소 재검증: 현재 두 bank를 재선택에 쓰지 않고, 변경할 원리·single primary와 충분한 training-seed/graph 단위를 먼저 잠근다. 원 CI0을 equivalence로 쓰지 않는다.
- 원문: [findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md](../findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md); [findings/sources/internal-pcfi-confirmation-2026-11-02/METHOD_KO.md](../findings/sources/internal-pcfi-confirmation-2026-11-02/METHOD_KO.md); [findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/design.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/design.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/statistics.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/statistics.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/audit.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/audit.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/cost.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/cost.json)

### IPCFIC-L01 [교훈] Context router 실패, 내부 고정의 개발 신호, 새 graph 확인, 새 training-seed 일반화를 구분한다.

원64graph bank에서 fixed−external accuracy +0.65743%p CI[-0.11984,+1.37358]였고 세 seed·네 primary cell의 점추정은 모두 양수였다. 그 후보를 선택한 뒤 새128graph bank의 정확도 기준을 잠갔다. 새 결과: external:accuracy 49.821540%/NLL 0.856885/Brier 0.552034; internal_fixed:accuracy 50.041098%/NLL 0.856498/Brier 0.552432; fixed-external accuracy%p +0.219558, 95% CI [-0.331926, +0.796665], NLL -0.000388, 95% CI [-0.003777, +0.002808], Brier +0.000398, 95% CI [-0.001679, +0.002518].

- 원문: [findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md](../findings/sources/internal-pcfi-confirmation-2026-11-02/REPORT_KO.md); [findings/sources/internal-pcfi-confirmation-2026-11-02/METHOD_KO.md](../findings/sources/internal-pcfi-confirmation-2026-11-02/METHOD_KO.md); [findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/protocol.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/design.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/design.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/statistics.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/statistics.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/audit.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/audit.json); [findings/sources/internal-pcfi-confirmation-2026-11-02/results/cost.json](../findings/sources/internal-pcfi-confirmation-2026-11-02/results/cost.json)

## Literal PCFI inside public GraphPFN

실험 조건: Same public initialization/frozen policy.4 arms×3seeds×256 updates, actual PCFI100steps/1correlation/one12-layer PFN. Primary fresh64 graphs의 자기 피처 없는 query accuracy.

### IPCFI-F01 [실패·중단] 실제 PCFI 전파를 공개 PFN 내부에 배치하고 context/PFN 상태로 전파 가중치를 조절하면 외부 PCFI·내부 고정 PCFI·label-blind router보다 분류가 좋아진다.

external:accuracy49.243267%/NLL0.867128/Brier0.559929; internal_fixed:accuracy49.900698%/NLL0.863062/Brier0.558362; context:accuracy49.413514%/NLL0.863255/Brier0.558625; label_blind:accuracy49.690554%/NLL0.863001/Brier0.558301. context-external accuracy%p +0.170247 CI[-0.557668,+0.866045], NLL -0.003872 CI[-0.010121,+0.001957]; context-internal_fixed accuracy%p -0.487184 CI[-0.861821,-0.130108], NLL +0.000193 CI[-0.001404,+0.001615]; context-label_blind accuracy%p -0.277040 CI[-0.704964,+0.174764], NLL +0.000254 CI[-0.001276,+0.001713]. Primary=False, full2%p guard=True, expand=False.

- 판정: 현재100-step literal PCFI/25-step interleaving/513-parameter router/256-update 구현의 메인 방법 확대를 보류한다. 이 final bank에서 head/loss/temperature/전파 횟수/학습량을 바꿔 구제하지 않는다.
- 원인·병목 [원인 미상]: 사전 비교를 모두 만족하는 분류 이득을 확인하지 못했다. 전파가 이미 충분한지, context 요약·bounded router·frozen backbone/input interface·prior·최적화·학습량 중 무엇 때문인지는 분리하지 않아 고유 원인은 미상이다. 상수 신호를 row-stochastic 전파로 바꿀 수 없다는 수학적 제약을 전체 실패의 단일 원인으로 쓰지 않는다.
- 닫은 범위: Official graph/SCM small family의 nodes154~192, physical features8~16, classes2~4. 새64 train/4dev/64final graphs, uniform/clustered node90/99.5 missing, native pair/RMS/mask input,4 arms×3seeds×256 CE updates, 자기 피처 없는 query accuracy primary.
- 닫지 않은 것: JIMP-F01의 generic refiner와 별개 실행이다. 외부 PCFI의 실제 그래프 성과, 모든 내부 PCFI 방식, 다른 전파/표현/학습 원리, native NaN interface, 실제 heterograph·unseen schema 일반을 기각하지 않는다.
- 재검토 조건: 현재 동일한 numeric propagation을 다시 배치하거나 bounded weight만 조절하는 것을 넘어, 새 그래프의 context가 전달해야 할 정보를 바꾸는 구체적인 연산이 강한 외부 PCFI/PFN보다 유리할 근거가 생길 때.
- 기준선: 동일 actual PCFI/100steps/1correlation/12PFN layers의 외부 PCFI, 내부 고정, context와 strict label-blind router, 같은 train tasks/CE/initialization/updates. Context-frequency/uniform prior와 NLL/Brier/CPU→CPU end-to-end 비용도 보고한다.
- 최소 재검증: 본 final bank는 tuning하지 않는다. 변경할 정보·연산과 사전 primary를 먼저 정하고 독립 graph/schema에서 node-level 대신 base-graph paired 분석으로 한 번 검증한다. Labels-only 효과와 PFN-feedback 전체 효과를 구분한다.
- 원문: [findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md](../findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md); [findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md](../findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md); [findings/sources/internal-pcfi-2026-11-02/protocol.json](../findings/sources/internal-pcfi-2026-11-02/protocol.json); [findings/sources/internal-pcfi-2026-11-02/results/statistics.json](../findings/sources/internal-pcfi-2026-11-02/results/statistics.json); [findings/sources/internal-pcfi-2026-11-02/results/cost.json](../findings/sources/internal-pcfi-2026-11-02/results/cost.json); [findings/sources/internal-pcfi-2026-11-02/results/audit.json](../findings/sources/internal-pcfi-2026-11-02/results/audit.json); [findings/sources/internal-pcfi-2026-11-02/results/preflight.json](../findings/sources/internal-pcfi-2026-11-02/results/preflight.json); [findings/sources/internal-pcfi-2026-11-02/results/development_actual_pcfi_parity.json](../findings/sources/internal-pcfi-2026-11-02/results/development_actual_pcfi_parity.json)

### IPCFI-L01 [교훈] 실제 PCFI 내부화와 이전 MLP refiner, strict router label-blindness와 labels-only 효과를 분리한다.

새 구현은 PCFI의 공간 전파와 채널 상관 보정 연산을 유지했고 초기 context/label-blind 예측은 내부 고정과 같았다. 작은 그래프의 SciPy 차이는 최대4.43e-7, 실제 dev FP32 kernel/cache 차이는8.54e-5였다. Label-blind router에 PFN states와 labels를 오염시켜도 routing은 같았다.

- 원문: [findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md](../findings/sources/internal-pcfi-2026-11-02/METHOD_KO.md); [findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md](../findings/sources/internal-pcfi-2026-11-02/REPORT_KO.md); [findings/sources/internal-pcfi-2026-11-02/results/preflight.json](../findings/sources/internal-pcfi-2026-11-02/results/preflight.json); [findings/sources/internal-pcfi-2026-11-02/results/development_numeric_repair.json](../findings/sources/internal-pcfi-2026-11-02/results/development_numeric_repair.json)

## Public GraphPFN 내부 대치 상태 갱신

실험 조건: 공개 GraphPFN 1.3 + LimiX 초기화. 합성64 train/32 test graphs, raw features7~16, 전체 graph166~192 test nodes, context최대4/class 및 보조1/class, query32. 4 arms×3 seeds×192 successful updates. 기존 graph adapters/feature encoder/label decoder 추가 학습, 원래 LimiX transformer weights 고정. 공통 observed-only RMS/mask interface, 공식 전처리 재현 아님.

### JIMP-F01 [실패·중단] PFN 내부의 반복 대치가 동일한 대치기를 PFN 앞에서 실행하는 것과 강한 동일 예산 Plain을 넘는다.

Uniform NLL: Mask Plain0.88681, PCFI Plain0.89527, Frontend0.89276, Interleaved0.89745. Clustered: 0.88318/0.90248/0.89339/0.89508. Interleaved−Frontend는 Uniform+0.00469 CI[−0.01612,+0.02143], Clustered+0.00169 CI[−0.01132,+0.01571]. 사전에 정한 두 조건·세 대조군 동시 우위 gate 미통과.

- 판정: 현재 구현 확대 보류·내부 대치 증분 미입증
- 원인·병목 [원인 미상]: 실제 공개 PFN과 대치 상태를 연결해 gradient를 학습했지만, 위치를 바꾼 독립 예측 이득은 확인하지 못했다. 정보 중복·현재 numeric 함수족·prior·학습 예산 중 원인은 분리하지 못했다.
- 닫은 범위: 네 번의 공유 bounded numeric 갱신을 layers0/3/6/9에 넣는 현재 연산·small official-prior family·192 updates를 더 키워 메인 메소드로 확장
- 닫지 않은 것: 다른 대치 함수족, pretrained feature decoder, latent belief representation, native NaN-input 대조와 실제 그래프 전이는 별개다. 이 실험은 이전 10K quadratic prototype과 다른 공개 GraphPFN 구현이다.
- 재검토 조건: 대치 상태를 PFN 추론 중 갱신할 때 사전 대치에 없던 예측 정보를 보존·전달하고, 동일 예산 Plain/PCFI 대조를 넘는 연산이 나올 때.
- 기준선: 동일 synthetic missing prior로 추가 학습한 PCFI Plain·Mask Plain·Frontend; native NaN GraphPFN의 같은 예산 학습도 향후 필요
- 최소 재검증: 공통 전처리·초기화·학습 과제·성공 update 수를 맞춘 새 bank에서 반복 위치 효과를 비교한다. 같은 checkpoint의 피드백 제거와 재학습 대조를 구분하고 NLL/Brier/accuracy를 따로 보고한다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/paired.json](../findings/sources/joint-imputation-2026-11-01/paired.json)

### JIMP-F02 [실패·중단] PFN의 query 분류 목적에 맞춘 사전 대치기로 PCFI를 교체하면 주 결측 조건에서 독립 개선을 얻는다.

Frontend−PCFI Plain NLL: Uniform−0.00252 CI[−0.01509,+0.00930], Clustered−0.00909 CI[−0.04275,+0.01993]. Frontend−Mask Plain은 +0.00594/ +0.01021이며 두 CI 모두0 포함. 이 대조 CI는 학습·평가 완료 후 보조 분석으로 계산했고 주 gate는 바꾸지 않았다.

- 판정: 현재 task-trained frontend의 주 조건 우위 미입증
- 원인·병목 [원인 미상]: Query CE로 학습된 네 단계 대치가 고정 PCFI나 일반 mask/token adapter보다 유용하다는 증거가 부족했다. 정확한 원인은 미상이며 복원 목적이 아닌 분류 목적을 썼다는 사실만으로 개선되지 않았다.
- 닫은 범위: 현재 11.6K 추가 파라미터의 shared frontend와 예산을 확장해 우수한 범용 대치기로 주장
- 닫지 않은 것: 모든 task-aware imputation과 PFN용 전처리가 실패했다는 뜻은 아니다. 피처 복원 MSE를 최소화한 모델과 비교한 결과도 아니다.
- 재검토 조건: PFN에 필요한 정보와 단순 복원값을 구분하는 대치·입력 연산이 강한 동일 학습 대조보다 안정적인 예측 이득을 보일 때.
- 기준선: PCFI Plain, Mask Plain, native NaN input, 동일 예산·초기화·query CE
- 최소 재검증: 대치 연산의 개선과 backbone 추가 학습을 분리하고, 먼저 새 그래프 bank에서 prior-only 확률 대조까지 함께 비교한다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json](../findings/sources/joint-imputation-2026-11-01/frontend_comparisons.json)

### JIMP-P01 [보존] Context가 클래스당1개인 보조 조건에서 Interleaved는 두 Plain보다 낮은 NLL을 얻었다.

k1/node99.5 NLL은 Mask Plain1.00109, PCFI Plain1.16219, Frontend0.99158, Interleaved0.93901. Interleaved−Mask Plain−0.06208 CI[−0.13935,−0.00259], −PCFI−0.22317 CI[−0.39541,−0.09165]. −Frontend−0.05257 CI[−0.12008,+0.00099].

- 확대 금지: 사전 대치보다 유의한 우위, 주 gate 통과, 독립 확인 또는 극소 context 문제 해결로 확대하지 않는다. 같은 k1에서 단순 uniform/context prior의 NLL0.91181이 더 낮다. 이 NLL 비교를 accuracy의 유의 개선으로 바꾸지 않는다.
- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/paired.json](../findings/sources/joint-imputation-2026-11-01/paired.json)

### JIMP-L01 [교훈] Joint imputation의 실행 가능성과 반복 피드백의 예측 가치를 구분한다.

실제 공개 GraphPFN의 feature-pair token 구조를 유지했다. 관측값 clamp, 원래 mask 유지, refiner/encoder gradient, 노드 순열, 동일 Frontend/Interleaved 초기화·파라미터·schedule, frozen weights 보존과 성공 update192회를 검사했다. 이 검사는 구현 연결성을 보여주며 메소드 우수성의 증거는 아니다.

- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/preflight.json](../findings/sources/joint-imputation-2026-11-01/preflight.json)

### JIMP-L02 [교훈] FP16 실행 보완을 새 loss·아키텍처 개선과 구분하고 성공 optimizer update 수를 맞춘다.

최초 Interleaved seed11은 nonfinite gradient를 clipping이 거부해 중단됐다. 같은 episode에서 GradScaler scale을 낮춰 finite gradient를 얻은 뒤에만 update를 세도록 보완했다. 모델·prior·loss·LR schedule은 유지했다. 완료한 이전 대조군을 재학습하거나, 실패한 partial weights에서 추가 학습하지 않았다. 수치 재시도는 추가 계산 비용으로 별도 기록했다.

- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/numerical_repair.json](../findings/sources/joint-imputation-2026-11-01/numerical_repair.json)

### JIMP-U01 [미실행] 실제 그래프·새 semantic schema·native NaN baseline·다른 결측 추론 함수족

이 실행은 작은 동일 생성 family의 새 graph와 일부 mask/context shift를 평가했다. Native NaN-input GraphPFN의 동일 예산 추가 학습, 원래 feature decoder 활용, 복원 분포 학습, 실제 heterograph 전이는 실행하지 않았다. True missing feature를 복원했다고 검증한 것도 아니다.

- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md)

### JIMP-L03 [교훈] 피드백 제거 intervention은 FP16 수치 순서를 맞춰 해석하고, 별도 재학습 대조와 구분한다.

No-feedback 대조의 h+new−old를 h+(new−old)로 묶어 cancellation을 보완하고, 동일 process의 standard와 다시 짝지었다. Full-observed96건의 차이는 정확히0, 별도 process replay의 최대 확률 차이는0.00246981이었다. 보정된 Uniform의 standard−no-feedback NLL은+0.00306 CI[+0.00063,+0.00581]로 feedback이 더 나빴다. Clustered와 k1의 CI는0 포함. 학습·주 예측·checkpoint를 바꾸지 않았다.

- 원문: [findings/sources/joint-imputation-2026-11-01/REPORT.md](../findings/sources/joint-imputation-2026-11-01/REPORT.md); [findings/sources/joint-imputation-2026-11-01/paired_corrected.json](../findings/sources/joint-imputation-2026-11-01/paired_corrected.json); [findings/sources/joint-imputation-2026-11-01/feedback_audit.json](../findings/sources/joint-imputation-2026-11-01/feedback_audit.json)

## 초기 JR-PFN과 decoder 표현력

실험 조건: Feature-free binary SBM, target8+helper4, 총 context1/2/4, Exact Bayes 소형 열거.

### JR-P01 [보존] Exclusive의 joint-role 제약 보존은 강한 Plain/factorized를 넘었다.

Exclusive JR−Large Plain+3.20%p, JR−Factorized+1.76%p. Learned posterior−known prior는+0.14%p에 그쳤다.

- 확대 금지: 모든 prior에서 우월하거나 context에서 미지 joint dependence를 새로 식별했다는 성과가 아니다.
- 원문: [findings/sources/graphpfn-2026-11-01/jr.md](../findings/sources/graphpfn-2026-11-01/jr.md)

### JR-P02 [보존] Context posterior에는 실제 predictive value가 있었다.

Exclusive 총 context4에서 역할 정보0.12447 nats, exact role update 기대 NLL 이득0.02278. Decoder/predictive 병목도 함께 있었다.

- 확대 금지: 오라클 posterior의 이득이 곧 학습 모델의 성과나 효율을 보장하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/posterior_shift.md](../findings/sources/graphpfn-2026-11-01/posterior_shift.md)

### JR-L01 [교훈] Positive-only diffusion의 구조적 불가능성은 loss로 고칠 수 없었다.

Context label이 모두 A일 때 당시 decoder는 모든 query에서 p(A)≥0.5. Exact teacher는 nonedge로 반대 class를 지지할 수 있었다. 이후 contrastive/BP decoder가 이 제한을 넘었다.

- 원문: [findings/sources/graphpfn-2026-11-01/distillation.md](../findings/sources/graphpfn-2026-11-01/distillation.md); [findings/sources/graphpfn-2026-11-01/contrastive.md](../findings/sources/graphpfn-2026-11-01/contrastive.md)

### JR-L02 [교훈] 이미 해결한 중간 병목을 현재 미해결 문제로 다시 제안하지 않는다.

양의 diffusion 표현력 문제는 contrastive decoder로 전환했다. 이후 EAJ/R1의 경쟁력 실패는 별개의 문제이며 기존 LLBP 항목에 보존한다.

- 원문: [findings/sources/graphpfn-2026-11-01/contrastive.md](../findings/sources/graphpfn-2026-11-01/contrastive.md); [findings/sources/graphpfn-2026-11-01/llbp.md](../findings/sources/graphpfn-2026-11-01/llbp.md)

## 실패 기록과 기존 논문을 연결한 후속 후보 선별

실험 조건: 기존 32 papers와 103 findings를 읽고 원문 및 현재 호출 경로를 점검한 문헌 검토. 새 학습·평가 결과가 아니다.

### REOPEN-L01 [교훈] 현재 LimiX의 기본 sample attention에서 label-context와 feature-memory의 접근 범위가 묶여 있다.

common.py는 context 노드를 앞에 놓고 모든 base layer에 eval_pos=len(context)를 전달한다. Vendor layer.py의 call_sequence_attention은 context와 query 양쪽의 key/value에 x[:, :eval_pos]를 사용한다. 따라서 label이 없는 노드의 관측값을 해당 base attention의 직접 key/value로 읽지는 않는다. Residual의 query 자기 feature와 graph adapter 경로는 존재하므로 '그 관측값을 전혀 사용하지 못한다'는 뜻이 아니다.

- 원문: [findings/sources/literature-reopening-2026-11-02/audit.json](../findings/sources/literature-reopening-2026-11-02/audit.json); [findings/sources/literature-reopening-2026-11-02/common.py](../findings/sources/literature-reopening-2026-11-02/common.py); [findings/sources/literature-reopening-2026-11-02/limix_layer.py](../findings/sources/literature-reopening-2026-11-02/limix_layer.py)

### REOPEN-U01 [미실행] Label memory와 observed-feature memory를 분리하는 graph PFN

TabImpute의 query 상호작용과 Ephris의 전체 노드 feature Gather를 단서로, label 읽기는 C, feature f의 실측값 읽기는 O_f={j:M_jf=1}로 두는 후보. NEXT-U02의 sparse observation 안을 구체화한 것으로, 별개의 발명·학습된 성공으로 세지 않는다.

- 원문: [findings/sources/literature-reopening-2026-11-02/audit.json](../findings/sources/literature-reopening-2026-11-02/audit.json); [papers/TabImpute.yaml](../papers/TabImpute.yaml); [papers/Ephris.yaml](../papers/Ephris.yaml); [findings/post-random-search.yaml](../findings/post-random-search.yaml); [findings/joint-imputation-pfn.yaml](../findings/joint-imputation-pfn.yaml); [findings/random-fill-controls.yaml](../findings/random-fill-controls.yaml)

### REOPEN-U02 [미실행] Missing 영역을 통과하는 경로 자체를 읽는 observation-aware walk PFN

Wander는 긴 walk와 feature-free 예측의 기존 실증을 갖는다. 이를 관측 cell이나 context에 이르는 경로 읽기로 확장하는 후보이며, 원래 Wander부터 강한 baseline으로 쓴다. Wander의 label 방문 walk filter 자체를 새 제안으로 삼지 않는다.

- 원문: [papers/Wander.yaml](../papers/Wander.yaml); [findings/observation-budget.yaml](../findings/observation-budget.yaml); [findings/feature-intervention.yaml](../findings/feature-intervention.yaml)

### REOPEN-U03 [미실행] PFN 전체의 random-fill 평균 대신 확률적인 첫 feature 변환을 학습하는 후보

PEMix는 구조 조건부 mixture의 불확실성을 첫 GNN layer에서 해석적으로 처리한다. PFN의 입력 비선형 변환에 비슷한 생각을 적용할 여지는 있지만 CFB-U01/NEXT-U03의 미해결 사항을 해결한 것은 아니며 우선순위는 낮다.

- 원문: [papers/PEMix.yaml](../papers/PEMix.yaml); [papers/GSPN.yaml](../papers/GSPN.yaml); [findings/context-feature-belief.yaml](../findings/context-feature-belief.yaml); [findings/random-fill-controls.yaml](../findings/random-fill-controls.yaml)

### REOPEN-L02 [교훈] 이번 문헌 대조는 기존 gate를 바꾸지 않고 후보와 진단 대조를 구분한다.

MCP-P02의 off-diagonal/clustered accuracy +3.43%p는 보존한 미독립 재현 신호이며 원래 NLL gate는 실패 그대로다. Feature memory, walk, 기대 token의 우열은 아직 측정하지 않았다. 기존 논문은 이 3.43%p의 원인이나 LLBP-F11의 Grid6 우위를 뒤집지 않았다.

- 원문: [findings/mechanism-crossed-prior.yaml](../findings/mechanism-crossed-prior.yaml); [findings/llbp-eaj.yaml](../findings/llbp-eaj.yaml); [findings/sources/literature-reopening-2026-11-02/audit.json](../findings/sources/literature-reopening-2026-11-02/audit.json)

## Learned-Likelihood BP / EAJ

실험 조건: known-family binary SBM, feature-free 관계 그래프, target 8개 + helper 4개, 두 관계와 네 role 가설, 총 context 1/2/4 (클래스당 아님). 작은 latent state는 exact enumeration, conditional BP는 고정 반복.

### LLBP-F01 [실패·중단] Role-conditional 감독을 더하면 최종 예측이 개선된다.

C_conditional − B_predictive에서 conditional KL은 세 prior 모두 줄었지만, predictive KL 증분의 paired 95% CI는 세 prior 모두 0을 포함했다 (Independent +0.000188, Exclusive −0.000181, Coupled −0.000582).

- 판정: 추가 이득 미확인
- 원인·병목 [진단이 지지]: Component conditional fit 개선이 최종 predictive 개선으로 이어지지 않았다. Positive diffusion의 표현력 제한도 뒤에 확인했지만 전체 미개선의 유일 원인으로 분해하지 않았다.
- 닫은 범위: 기존 staged decoder에 conditional loss를 더해 메인 메소드로 확장하는 갈래
- 닫지 않은 것: conditional distillation 일반이 불가능하다는 결론은 아니다. 추가 감독은 의도한 conditional 오차를 실제로 줄였다.
- 재검토 조건: 표현 가능한 decoder에서 conditional 감독의 독립 predictive 이득을 보이는 방법이 나올 때.
- 기준선: Predictive-only JR, Plain predictive distillation
- 최소 재검증: Decoder와 teacher를 고정하고 auxiliary의 독립 증분을 같은 task·예산으로 비교한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F02 [실패·중단] Posterior까지 정확하게 학습하면 최종 예측도 좋아진다.

D_all − C_conditional에서 posterior KL은 줄었지만 predictive KL은 세 prior 모두 늘었고 CI도 모두 0을 제외했다 (예: Independent +0.004039, CI [+0.002707, +0.005418]).

- 판정: 유의 악화
- 원인·병목 [원인 미상]: Posterior fit 감소와 predictive KL 악화가 함께 관측됐다. 목적의 불일치나 decoder 오차 보상 중 원인을 하나로 확정하지 못했다.
- 닫은 범위: 현재 architecture·loss weight·예산에서 posterior loss를 추가하는 것
- 닫지 않은 것: posterior 감독 일반의 기각이 아니라, component 오차 감소가 최종 품질을 보장하지 않는다는 결과다.
- 재검토 조건: Posterior 정확성과 deployment prediction을 동시에 보존하는 감독·추론 원리가 나올 때.
- 기준선: Predictive-only, conditional-only, exact posterior substitution
- 최소 재검증: 같은 decoder에서 posterior KL과 deployment predictive KL을 따로 평가한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F03 [실패·중단] Masked-edge likelihood auxiliary가 learned BP의 핵심 학습 기여다.

BP + auxiliary − BP predictive-only가 Independent +0.000374 [+0.000187, +0.000587], Exclusive +0.000261 [+0.000144, +0.000364]로 유의하게 나빴고, Coupled는 CI가 0을 포함했다.

- 판정: 유의 악화 (새 loss 기여 주장 중단)
- 원인·병목 [원인 미상]: 추가 likelihood auxiliary는 predictive-only보다 악화했다. Local likelihood fit이 label risk와 정렬되지 않을 가능성은 가설이며 정확한 원인은 미상이다.
- 닫은 범위: masked-edge likelihood loss를 새 학습 원리·핵심 성능 증분으로 주장하는 것
- 닫지 않은 것: auxiliary 없이도 potential을 학습하고 BP로 좋은 예측을 만들 수 있었다. 실패한 것은 학습 가능성이 아니라 추가 loss의 우수성 주장이다.
- 재검토 조건: Local factor 감독이 query risk에 도움이 되는 조건을 증명·검증한 방법이 나올 때.
- 기준선: Learned BP predictive-only, fitted constant
- 최소 재검증: 같은 potential·BP·role update에서 loss만 바꿔 independent gain을 본다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F04 [실패·중단] Learned potential + 기존 frozen role q만으로 넓은 조건에서 adaptive OOD 추론이 된다.

Matched에서는 좋아졌지만, Coupled에서는 auxiliary 포함 learned BP가 constant보다 나빴다. Density 0.12 외삽에서는 일부 Plain/signed 대조보다도 나빴고, density/signal 이동에서 oracle potential과의 차이가 커졌다.

- 판정: 범용성 미확보
- 원인·병목 [진단이 지지]: Potential과 frozen role posterior의 OOD 불일치 및 conditional 근사 오차가 후보였다. 뒤의 coherent-update 대조가 일부 병목을 지지하지만 모든 shift의 원인을 확정하지 않는다.
- 닫은 범위: 관측 통계 encoder로 potential을 학습해 고정 role head와 결합하는 것만으로 범용 추론기를 주장하는 것
- 닫지 않은 것: 원인은 확정하지 못했다: potential point estimate, frozen posterior의 분포 이동, conditional BP 근사, 이들의 상호작용.
- 재검토 조건: 새 graph 통계로 일관된 role/parameter evidence를 추론하고 단순 EB/grid를 넘는 방법이 나올 때.
- 기준선: Constant+update, density estimator, Grid6/81
- 최소 재검증: 같은 conditional branches에서 posterior·potential 교차 대조와 새 density/signal bank를 평가한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F05 [실패·중단] Parameter 후보를 혼합하되 기존 role q를 유지해도 충분하다.

Coupled/high density에서 true-parameter Bayes KL이 기존 MRP 0.112820, full-grid parameter mixture + frozen role q 0.078075, 같은 BP branches + coherent joint weights 0.017615였다. 개선의 대부분은 role marginal 갱신에서 왔다 (dependence만 분리한 same-marginal 차이는 −0.001434 [−0.001946, −0.000872]).

- 판정: 기대 기각 + 해석 기각
- 원인·병목 [직접 확인]: 같은 BP branches에서 coherent joint weights가 frozen role q보다 좋았다. 큰 개선을 같은 주변분포의 dependence 효과 전체로 돌릴 수는 없다.
- 닫은 범위: frozen-hybrid로 충분하다는 기대, 그리고 전체 개선을 H–parameter dependence 보존만의 효과로 보는 해석
- 닫지 않은 것: frozen hybrid도 정규화된 분포이며 일부 조건부 dependence를 보존한다. 단순 독립 head의 곱이라고 부르지 않는다.
- 재검토 조건: 일관된 evidence 갱신을 미지 family에서도 정확하고 저렴하게 구현하는 장치가 나올 때.
- 기준선: Joint grid, same-marginal product, frozen-grid
- 최소 재검증: Conditional branches와 marginals를 맞춰 role update와 dependence 증분을 분리한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F06 [실패·중단] EAJ(Evidence-Amortized Joint PFN)가 exact-grid를 비슷한 품질·훨씬 낮은 전체 비용으로 대체한다.

Independent heads, analytic initializer, own-product보다는 좋았지만 teacher exact marginals의 곱보다 여섯 조건 모두 나빴다. Signal-high에서는 Constant+update가 세 prior 모두 EAJ보다 좋았다 (EAJ−Constant CI 모두 0 초과). Posterior-only는 exact evidence보다 약 1.9–2.9배 빨랐지만 공통 81×4 BP 때문에 end-to-end 감소는 약 6–12%였다.

- 판정: 경쟁력 미달 (대체·확장 중단)
- 원인·병목 [진단이 지지]: 학습 evidence의 정확도와 공통 BP 비용이 병목이었다. Posterior-only 속도 증분이 end-to-end에 크게 전달되지 않았다.
- 닫은 범위: EAJ에 새 surrogate loss를 더하거나 모델·학습량을 늘려 경쟁력을 추격하는 갈래
- 닫지 않은 것: joint 표현 자체의 장점(약한 head 대비)은 있었다.
- 재검토 조건: 강한 classical update의 품질을 유지하며 전체 branch/evidence 비용을 줄이는 연산이 나올 때.
- 기준선: Constant+role-update, exact-product, Joint Grid6/81
- 최소 재검증: Posterior 추정만의 시간과 shared BP를 포함한 end-to-end 품질·비용을 분리한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F07 [실패·중단] EAJ를 직접 predictive 학습하면 실용성과 posterior 해석을 함께 회복한다.

B는 A보다 unknown-parameter exact-grid predictive KL을 여섯 조건 모두 개선했다 (상대 약 15.7–31.9%, CI 0 제외). 그러나 posterior KL은 여섯 조건 모두 악화했고, signal-high 세 prior 모두 Constant+role-update보다 NLL·Brier가 유의하게 나빴으며, accuracy 개선은 미확인이었다. Convex oracle 기준 A→oracle gap 중 B가 회수한 비율은 약 16–32%였다.

- 판정: 부분 성공·메소드 구제 실패 (predictive 개선을 posterior 복원 개선으로 보는 해석 기각)
- 원인·병목 [진단이 지지]: 실제 decoder에 맞춘 predictive weights와 정확한 Bayesian posterior 복원은 다른 목적이었다. Direct predictive 개선은 있었지만 classical 기준선을 넘지 못했다.
- 닫은 범위: EAJ에 새 surrogate loss를 추가하는 갈래
- 닫지 않은 것: 직접 predictive 학습이 작동하지 않았다는 결론은 틀리다. B의 출력은 decoder에 맞춘 aggregation weights로 남긴다.
- 재검토 조건: Decoder 오차를 다루면서 posterior 해석과 predictive quality를 함께 보장하는 방법이 나올 때.
- 기준선: Direct predictive EAJ, Constant+update, exact-grid
- 최소 재검증: 두 KL과 실제 NLL/Brier를 함께 보고 calibrated posterior라는 주장을 별도 검증한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F08 [실패·중단] 같은 potential에서 role weights를 다시 계산하는 자기-update가 OOD 문제를 전반적으로 해결한다.

Exclusive high density에서는 자기 exact update 0.04029가 Constant+update 0.05073보다 좋았다 (세 prior 모두 CI 0 제외). 하지만 low density에서는 자기 exact update 0.01159가 frozen q 0.00915, Constant+update 0.00917보다 나빴다. Bethe를 exact로 바꿔도 해결되지 않았다.

- 판정: 보편적 OOD 해결 주장 기각
- 원인·병목 [진단이 지지]: Low density에서는 exact role update도 실패해 Bethe partition 근사만으로 설명할 수 없었다. Potential·conditional·parameter uncertainty 상호작용은 미해결이다.
- 닫은 범위: potential과 weights를 맞추면 분포 이동도 전반적으로 해결된다는 주장
- 닫지 않은 것: 자기-update는 특정 조건에서 유효한 추론 개선이다.
- 재검토 조건: Sparse/OOD potential uncertainty와 conditional inference를 함께 해결하는 연산이 나올 때.
- 기준선: Constant+update, R1, Grid6/81
- 최소 재검증: Low density를 포함한 새 bank에서 exact/Bethe·conditional 교차를 고정해 본다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F09 [실패·중단] 자기-update 경로까지 학습하는 T1이 R1보다 우수하다.

T1−R1 KL −0.000192, 95% CI [−0.000576, +0.000164], accuracy 약 −0.093%p. Low-density 세 prior에서는 R1보다 좋았지만 별도 16조건에서 KL 점추정이 악화했고, low density에서도 R0에는 못 미쳤다.

- 판정: 추가 이득 미확인
- 원인·병목 [원인 미상]: Self-update를 미분해 학습하는 경로는 작동했지만 R1 대비 이득이 없었다. Optimization·model bias 중 고유 원인은 미상이다.
- 닫은 범위: T1의 우수성을 확보하기 위해 loss·예산을 계속 늘리는 것
- 닫지 않은 것: head 없는 학습 경로와 gradient 흐름은 확인됐다. 더 큰 예산에서 절대 개선되지 않는다는 증명은 아니다.
- 재검토 조건: 동일 계산 budget에서 inference-aware 학습의 독립 이득을 보이는 원리가 나올 때.
- 기준선: Frozen R1, R0, Constant+update, Grid6
- 최소 재검증: 추론 시 head 제거 효과와 새 학습 효과를 분리하고 initial state·schedule·예산을 맞춘다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F10 [실패·중단] Low-density 실패는 partition 근사 또는 role weights만의 문제다.

Constant/learned weights × constant/learned conditional 네 조합 교차에서 learned conditional 교체는 세 prior 모두 손해였다. Learned weights 교체는 Independent·Exclusive에서 손해, Coupled에서 도움이었고, 비가법적 상호작용이 있었다. Exact role update에서도 실패했다.

- 판정: 단일 병목 설명 기각
- 원인·병목 [진단이 지지]: 교차 대조에서 learned conditional 손해와 비가법 interaction을 확인했다. Role weights만 고치면 된다는 단일 원인 설명은 지지되지 않는다.
- 닫은 범위: role posterior만 고치면 된다는 설명, exact partition이면 해결된다는 설명
- 닫지 않은 것: 기각하지 못한 가능성: potential 추정 편향, parameter uncertainty 부족, conditional BP 오차, 과확신, 이들의 상호작용.
- 재검토 조건: Potential·partition·conditional 오차를 일관되게 제어하는 추론 연산이 나올 때.
- 기준선: Constant/learned weights×conditional 교차, exact update
- 최소 재검증: 네 교차 조합과 exact-role/BP conditional을 함께 비교해 단일 병목 주장을 검증한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F11 [실패·중단] R1은 강한 model-based 기준선보다 좋은 메인 메소드다.

Bayes predictive KL / NLL이 R1 0.020488 / 0.615840, Joint Grid81 0.014249 / 0.610493, Joint Grid6 0.011939 / 0.608083이었다. Grid6이 KL·NLL·Brier 모두 우수했고 paired CI가 0을 제외했다. R1−Constant는 −0.001744 [−0.002072, −0.001425]로 유의하게 좋았다.

- 판정: 경쟁력 미달
- 원인·병목 [직접 확인]: 현재 known-family에서 작은 training-support joint grid가 품질·비용 기준을 더 잘 충족했다. Learned 모델 전체가 불필요하다는 보편 원인은 아니다.
- 닫은 범위: R1을 강한 기준선보다 우수한 메인 메소드로 승격하는 것
- 닫지 않은 것: R1−R0, R1−Constant의 유의한 개선은 그대로 남는다. Grid6과 Grid81은 격자 수뿐 아니라 prior 자체가 다르므로 '작은 grid가 일반적으로 낫다'로 해석하지 않는다.
- 재검토 조건: 작은 명시적 grid가 처리하지 못하는 실제 family/schema에서 우위를 보이는 범용 추론법이 나올 때.
- 기준선: 사전 prior 고정 Grid6/81, Constant+update, R1
- 최소 재검증: 알려진 family·prior 접근 조건과 end-to-end 비용을 맞춘 새 bank에서 비교한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-F12 [실패·중단] R1은 품질이 낮아도 큰 속도 이점으로 정당화된다.

RTX 3090에서 단일 graph latency R1 5.17 ms vs Grid6 4.66 ms, Batch16 6.41 ms vs 6.81 ms, Batch16 추가 작업 메모리 1.44 MiB vs 64.04 MiB. Latency 반복 측정의 분산·CI는 없다.

- 판정: 큰 속도 우위 미확보
- 원인·병목 [직접 확인]: 병렬 grid와의 실제 latency가 비슷했고 posterior/branch 수 감소가 큰 전체 속도 이점으로 이어지지 않았다. 작업 메모리 절감은 별도로 보존한다.
- 닫은 범위: R1의 존재 이유를 추론 가속으로 바꾸는 것
- 닫지 않은 것: 추가 작업 메모리 절감은 사실이다. 다만 더 큰 graph 처리나 배포상 이익은 검증하지 않았고, 이를 사후 메인 주장으로 바꾸지 않기로 했다.
- 재검토 조건: 실제 처리 규모나 memory constraint에서 quality를 유지하는 비용 우위를 입증하는 연산이 나올 때.
- 기준선: 병렬 Grid6/81, R1
- 최소 재검증: 동일 hardware·batch·quality 범위에서 latency 반복 변동과 total/working memory 및 최대 처리 graph를 측정한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P1 [보존] Learned BP checkpoint: 원래 positive/signed decoder보다 좋은 matched 확률 예측

Matched Bayes predictive KL이 Frozen positive D 0.063955 / 0.052586 / 0.085548에서 Learned BP predictive-only 0.015946 / 0.011013 / 0.028391로 줄었다 (Independent / Exclusive / Coupled). 단순 signed diffusion만으로는 이 개선을 설명할 수 없었다.

- 확대 금지: 모든 생성 family에서 learned likelihood가 필요하다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P2 [보존] R1 자기-update: 같은 potential의 활용 방식만 바꿔 개선했고, 새 graph bank에서 재현됐다.

R1−R0 KL −0.012995 (약 38.8% 감소), accuracy +0.376%p.

- 확대 금지: 학습 단계도 head가 필요 없었다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P3 [보존] R1−Constant paired 결과

최신 비교에서 KL 개선 CI가 0을 제외했다 (−0.001744 [−0.002072, −0.001425]).

- 확대 금지: 강한 모든 baseline보다 우수하다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P4 [보존] T1: head 없는 학습 경로와 gradient 흐름 확인

Role-update 경로까지 gradient가 흐르는 것을 검증했다.

- 확대 금지: R1보다 우수한 학습법이다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P5 [보존] EAJ A→B: 직접 predictive 학습의 효과

여섯 조건 모두 grid predictive KL 개선 CI가 0을 제외했다.

- 확대 금지: posterior 복원도 개선됐다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P6 [보존] Convex oracle: 보고된 certificate와 큰 미회수 gap

Graph-shared convex oracle 12,096개의 최적성 certificate를 확인했다고 보고됐다. 정확한 bound는 메시지 손상으로 미상이다.

- 확대 금지: 현재 모델이 그 gap을 쉽게 회수한다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P7 [보존] Nonedge/mask 대조: 관측 상태에 맞는 factor를 쓰는 것이 중요하다.

Observed nonedge를 버리거나 unknown pair를 nonedge로 처리하면 크게 악화했다. 예: Independent에서 edge-only BP 0.082736 vs full edge/nonedge BP 0.016319, 관측률 0.2에서 올바른 mask 0.000226 vs unknown→nonedge 0.044486.

- 확대 금지: 모든 현실 데이터에서 해당 관측 구분이 주어진다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P8 [보존] 작업 메모리: 지정 Batch16 측정에서 큰 절감

추가 작업 메모리 1.44 MiB vs 64.04 MiB (총 GPU 메모리 차이가 아님).

- 확대 금지: 실제 처리 가능한 graph 규모가 더 크다.
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-P9 [보존] Joint Grid6/81과 Constant+update는 이후 비교 자산으로 보존한다.

알려진 family와 선언된 prior를 쓰는 model-based 기준선이 learned 방법이 넘어야 할 기준이 됐다.

- 확대 금지: 어떤 하나의 grid prior가 모든 조건을 지배한다 (Grid81은 matched Exclusive·sparse observation에서, Grid6은 low density에서 상대적으로 나쁠 수 있었다).
- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L0 [교훈] Teacher를 정확하게 주는 것과 decoder가 그 target을 표현할 수 있는 것은 다르다.

양의 relation weight와 normalization을 쓰는 label diffusion은 context가 전부 클래스 a이면 모든 query에서 p(a) ≥ 1/2를 유지한다. Nonedge로 반대 클래스를 지지하는 exact SBM teacher를 표현할 수 없었고, 이후 decoder 자체를 바꾸는 방향으로 이동했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L1 [교훈] Component fitting이 좋아져도 전체 predictor가 좋아진다는 보장은 없다.

초기 conditional/posterior loss, EAJ predictive 학습의 posterior 악화, low-density 교차 분석에서 반복해서 분리됐다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L2 [교훈] 더 직접적인 loss가 메소드 경쟁력을 해결해주지 않는다. Loss 변경의 효과와 메소드 경쟁력은 다른 질문이다.

EAJ에서는 직접 predictive 학습이 지표를 개선했지만 Constant+update를 넘지 못했고, T1은 R1 대비 추가 이득이 없었다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L3 [교훈] 일관된 추론이라고 OOD가 해결되지 않는다. 같은 potential 기반 추론은 잘못 맞는 potential을 자동으로 교정하지 않는다.

Self-update는 효과가 있었지만 low density를 해결하지 못했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L4 [교훈] Neural amortization이나 branch 수 감소가 곧 wall-clock 우위는 아니다.

EAJ에서는 공통 BP 계산이 비용을 지배했고, R1은 병렬 Grid6과 latency가 비슷했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L5 [교훈] 기준선 하나를 이겼다고 메인 메소드가 되지 않는다. 약한 비교의 유의성과 강한 비교의 경쟁력은 다른 기준이다.

R1−R0, R1−Constant는 성공이었지만 Grid6과의 비교에서 승격 근거가 사라졌다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L6 [교훈] 강한 기준선(이미 존재하는 명시적 grid 등)과의 품질·비용 비교를 재학습·확장 실험보다 먼저 한다.

Joint-grid가 이미 강했는데도 R1 개선 후 T1 재학습을 먼저 했다. R1 대 병렬 Grid6/81 비교가 T1보다 앞에 있어야 했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L7 [교훈] '추론 변경 효과가 확인된 후보'를 '메인 방법'으로 부르지 않는다. 반대로 강한 기준선에 밀렸다고 국소 개선의 실재까지 부정하지 않는다.

R1을 성급하게 '메인 방법으로 유지'라고 표현했다가 Grid6 비교 후 정정했다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L8 [교훈] Posterior KL, predictive KL, accuracy, 비용, 일반화는 각각 다른 검증이 필요하다. 한 단계의 성공을 다음 단계의 성공처럼 이어 붙이지 않는다.

여러 목표의 성공을 하나의 강한 주장으로 연결할 위험이 있었다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L9 [교훈] Oracle까지 gap이 크다는 것은 다음 투자의 자동 근거가 아니다.

Gap을 관측 정보와 현재 모델로 회수할 수 있는지, 회수해도 강한 baseline을 넘는지는 별개였다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-L10 [교훈] 원래 목표가 무너진 뒤 남은 장점 하나(메모리 등)에 맞춰 목표를 사후 변경하지 않는다. 실패 조건에 맞춘 gate·temperature 추가도 기본 연장으로 채택하지 않는다.

메모리 이점은 사실이지만 검증 없이 메인 주장으로 바꾸지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U1 [미실행] 관계의 관측과정 자체를 학습하는 BP

제안만 있었고 완료된 실험이 없다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U2 [미실행] Relation missingness가 label/role에 의존하는 설정

실험하지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U3 [미실행] Unseen semantic schema, 실제 heterograph, 큰 graph scaling

실험하지 않았다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

### LLBP-U4 [미실행] Temperature/shrinkage/density gate, 더 큰 모델, 추가 surrogate loss, 더 긴 T1 학습

추가 실행을 제안하지 않았거나 중단한 것이지, 실제로 학습해서 실패를 증명한 것은 아니다.

- 원문: [findings/sources/llbp-eaj-retrospective-2026-11-01.md](../findings/sources/llbp-eaj-retrospective-2026-11-01.md)

## LP-conditioned structural residual

실험 조건: 소형 일방향 prototype에 LP offset/내부 conditioning 추가, 2 seeds 및 새 paired factorial; 새 schema 아님.

### LPC-F01 [실패·중단] LP internal conditioning이 output offset과 Plain을 일관되게 넘는다.

Matched conditioned−Plain−0.35%p, heldout_shape+0.90%p; offset 대비−0.03/+0.35%p. 새 factorial 내부 효과+0.36~0.52%p이나 relation 추가효과+0.10 CI[−0.21,+0.42], feature 효과+0.06 CI[−0.11,+0.23]. NLL/Brier는 offset-only가 더 좋았다.

- 판정: LP residual tuning·대규모 확장 중단
- 원인·병목 [원인 미상]: 보인 이득 대부분과 output offset 효과를 구분해야 했고, schema adaptation 특유의 이득이라는 증거가 약했다. 조기 relation aggregation이 원인이라는 설명은 가설이지 확정 원인이 아니다.
- 닫은 범위: 현재 LP offset/residual conditioning 확대
- 닫지 않은 것: Relation 정보를 합치기 전 보존하는 구조와 별도 semantic schema 전이는 별개다.
- 재검토 조건: 압축된 LP 확률보다 풍부한 context 정보를 활용하며 output offset만의 효과를 넘어서는 연산이 나올 때.
- 기준선: Plain, offset-only, 같은 파라미터·초기화·학습 budget
- 최소 재검증: Relation 수·feature 폭·missing을 분리한 factorial과 새로운 semantic schema에서 내부 conditioning의 독립 증분을 확인한다.
- 원문: [findings/sources/graphpfn-2026-11-01/lpc.md](../findings/sources/graphpfn-2026-11-01/lpc.md); [findings/sources/graphpfn-2026-11-01/lpc_factorial.md](../findings/sources/graphpfn-2026-11-01/lpc_factorial.md)

## Mechanism-crossed GraphPFN controlled-prior pilot

실험 조건: 새64 train/32 test base graphs,4 arms×3 seeds×256 updates; 공개 Mask Plain continuation, 마지막 checkpoint 평가.

### MCP-F01 [실패·중단] 동일 주변분포의 feature-style/label-rule crossing이 강한 입력 기준선을 넘어 공식-prior 결측 과제로 전이된다.

Gate A=False, Gate B=False; controlled:offdiagonal:uniform vs coupled_mask: -0.00119 [-0.00603, +0.00402]; controlled:offdiagonal:clustered vs coupled_mask: -0.00288 [-0.00817, +0.00207]; controlled:offdiagonal:uniform vs coupled_pcfi: +0.00396 [-0.00917, +0.01896]; controlled:offdiagonal:clustered vs coupled_pcfi: -0.01740 [-0.03587, -0.00337]; official:any:uniform vs official_mask: -0.00210 [-0.00970, +0.00530]; official:any:clustered vs official_mask: -0.00369 [-0.00928, +0.00194]

- 판정: 두 gate 동시 통과 실패; 현재 controlled prior를 메인 방법으로 확대하지 않음
- 원인·병목 [원인 미상]: 정해진 초기화·256 성공 updates·함수족에서 요구한 predictive 우위를 확보하지 못했다. Prior의 한계, 학습량, preprocessing, context 식별 난도 중 원인을 분리한 실험은 아니다.
- 닫은 범위: 이 finite controlled family와 공개 Mask Plain continuation 설정에서 단순 mechanism crossing만으로 범용 missing GraphPFN 성과를 주장하는 것
- 닫지 않은 것: 다른 그래프 prior·관측 정보가 충분한 task family·실제 graph·동적 learned generator 일반을 기각하지 않는다. 기존 공식 prior의 induced coupling 자체가 실패 원인이라고 입증하지 않았다.
- 재검토 조건: 어떤 관측 정보와 예측 rule의 조합이 전이에 필요한지를 구체적으로 바꾸는 generator/inference 원리가 나오고 동일 비용의 강한 대조를 넘을 근거가 생길 때.
- 기준선: 동일 구조·초기화·update의 Official/Coupled Mask, 입력으로 학습한 Coupled PCFI, context/uniform prior, LP, 4-view random-fill; 실제 graph에서는 PCFI/FISF-trained PFN·Ephris도 필요.
- 최소 재검증: 새 독립 bank에서 full/node/cell 조건과 actual-feature 제거·셔플을 고정해 검증한다. 같은 bank의 loss/epochs 튜닝으로 구제하지 않는다. Controlled/official family 변경과 coupling만의 효과를 분리한다.
- 원문: [findings/sources/mechanism-crossed-2026-11-02/REPORT.md](../findings/sources/mechanism-crossed-2026-11-02/REPORT.md); [findings/sources/mechanism-crossed-2026-11-02/paired.json](../findings/sources/mechanism-crossed-2026-11-02/paired.json); [findings/sources/mechanism-crossed-2026-11-02/protocol.json](../findings/sources/mechanism-crossed-2026-11-02/protocol.json)

### MCP-L01 [교훈] Mechanism-crossing의 실제 학습을 기존 finite-bank curriculum과 분리해 기록한다.

NEXT-U01의 첫 pilot을 실행했다. 12 checkpoints, 3072 successful updates, 12288 probability files 재채점. Concrete feature/mask/query 및 rule-instance marginals를 맞췄고 model 구조·초기화·schedule을 유지했다. One graph/update이므로 paired loss는 없다.

- 원문: [findings/sources/mechanism-crossed-2026-11-02/audit.json](../findings/sources/mechanism-crossed-2026-11-02/audit.json); [findings/sources/mechanism-crossed-2026-11-02/exposure.json](../findings/sources/mechanism-crossed-2026-11-02/exposure.json); [findings/sources/mechanism-crossed-2026-11-02/preflight.json](../findings/sources/mechanism-crossed-2026-11-02/preflight.json)

### MCP-U01 [미실행] Real dense-feature graph, heterogeneous schema 및 본격적인 separable generative prior 검증

현재 실행은 graph/function seed가 새로운 작은 binary synthetic family다. 원래 GraphPFN 실패의 causal 설명, 실제 데이터 전이, FISF/Ephris 비교, 모든 induced laws의 분리성 보장은 수행하지 않았다. Known-rule sign-only 정보량 진단은 배포 predictor의 성과가 아니다.

- 원문: [findings/sources/mechanism-crossed-2026-11-02/REPORT.md](../findings/sources/mechanism-crossed-2026-11-02/REPORT.md); [findings/sources/mechanism-crossed-2026-11-02/oracle_information.json](../findings/sources/mechanism-crossed-2026-11-02/oracle_information.json)

### MCP-P02 [보존] Constructed off-diagonal 조건의 Coupled Mask 대비 정확도 신호

controlled:offdiagonal:clustered: accuracy difference +3.434%p, CI [+0.960, +6.152]; controlled:offdiagonal:node: accuracy difference +2.702%p, CI [+0.431, +5.143]

- 확대 금지: 사후 선별해 보존한 보조 metric 결과이며 다중 비교 보정과 독립 재검증은 없다. LP·PCFI·Official Mask 대비 같은 정확도 우위, NLL 우위 또는 실제 graph 전이로 확대하지 않고 원래 두 gate의 No-Go를 유지한다.
- 원문: [findings/sources/mechanism-crossed-2026-11-02/secondary_accuracy.json](../findings/sources/mechanism-crossed-2026-11-02/secondary_accuracy.json); [findings/sources/mechanism-crossed-2026-11-02/paired.json](../findings/sources/mechanism-crossed-2026-11-02/paired.json)

## 초기 Missingness-aware GraphPFN

실험 조건: 공식 prior의 소형 binary tasks, missingness mechanism mixture, 단일 학습 seed. NaN 학습과 official imputation 경로를 구분.

### MISS-F01 [실패·중단] Missingness prior 추가가 공식 입력 경로와 미학습 MNAR에서 유리하다.

NaN-input Ours−B4는 label_dep에서 +0.75/+1.43/+0.78%p였지만 degree_dep/self_mask는 약 ±0.1%p. Official imputation의 released GraphPFN이 NaN-trained Ours보다 label_dep에서 0.54/0.63/0.12%p 높았다.

- 판정: 범용 우위 미확보·해당 라인 중단
- 원인·병목 [진단이 지지]: Raw-NaN 기준선이 공식 imputation보다 약했고, official fill만으로도 mask가 잘 드러났다(AUROC 0.97). Context-query mask 관계 불일치에서 mixed-prior 이득이 유지되지 않았다. 이것만으로 모든 원인을 확정하지는 않는다.
- 닫은 범위: NaN 학습의 작은 이득을 official-input robustness나 unseen-MNAR 일반화로 승격
- 닫지 않은 것: Imputed-input에 맞춘 prior의 효과는 미평가다. 더 넓은 MNAR 식별 문제도 기각하지 않는다.
- 재검토 조건: 관측 메커니즘의 context-query shift를 다루면서 official preprocessing 대비 추가 가치를 보이는 학습 원리가 나올 때.
- 기준선: 공개 GraphPFN + official imputation, 동일 예산 B4
- 최소 재검증: 동일 입력 경로·초기화·예산의 새 bank에서 context-query mismatch와 미학습 mask를 paired 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md); [findings/sources/graphpfn-2026-11-01/missing_pilot.md](../findings/sources/graphpfn-2026-11-01/missing_pilot.md); [findings/sources/graphpfn-2026-11-01/missing_context.md](../findings/sources/graphpfn-2026-11-01/missing_context.md)

### MISS-U01 [미실행] Official imputed-input에 맞춘 Ours-Imp 최종 비교

B4imp는 255/500 update에서 중단, 저장 checkpoint는 update 200. Ours-Imp와 seed 1 및 imp evaluation은 실행하지 않았다.

- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md)

### MISS-L01 [교훈] 공식 전처리와 신호 대비 headroom을 먼저 확인한다.

No-missing accuracy 약72.9%, majority 약62%, MCAR 80% 손실 약3.5%p. 약한 입력 경로와의 비교만으로 새 prior의 가치를 주장하지 않는다.

- 원문: [findings/sources/graphpfn-2026-11-01/missing_closure.md](../findings/sources/graphpfn-2026-11-01/missing_closure.md)

## Observation-aware subgraph policy

실험 조건: Frozen 공개 GraphPFN, 새64/32 graphs, 2 policies×3 seeds, node budget64 학습·32/64/96평가, query8/graph.

### OBS-F01 [실패·중단] Mask-aware 학습 정책이 강한 경로 규칙을 넘는다.

Budget64 Observation/Paths NLL matched1.07193/1.05353, cluster1.08551/1.07685. 차이+0.01839 CI[−0.06529,+0.10213], +0.00866 CI[−0.11473,+0.11820].

- 판정: 주 성공 기준 미통과·확장 보류
- 원인·병목 [원인 미상]: Mask를 보는 것은 유용했지만 비슷한 효과를 고정 shortest-path closure도 얻었다. Joint subset 표현력·stochastic train/deterministic inference·학습 budget 중 정확한 원인은 분리하지 않았다.
- 닫은 범위: 현재 mask/geometry shared-node MLP를 더 학습해 메인 방법으로 확대
- 닫지 않은 것: 모든 learned retrieval을 기각하지 않는다. Budget32 cluster 차이−0.19284 CI[−0.39771,+0.00284]는 보조 개발 신호이며 승리가 아니다.
- 재검토 조건: 동일 resident budget에서 mask-aware 공동 부분그래프 구성이 강한 경로 규칙보다 나은 utility를 보이는 장치가 나올 때.
- 기준선: Evidence-paths, PPR, equal-capacity structure policy, random, context prior
- 최소 재검증: Fresh graph에서 사전 budget·mask gate를 고정하고 preprocessing/search를 포함한 비용과 label 예측을 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-F02 [실패·중단] 부분그래프 선택 개선으로 경쟁력 있는 확률 예측을 확보했다.

사후 Laplace context-prior NLL0.85241/Brier0.55736. Budget64 Observation의 matched NLL은+0.21952 CI[+0.12555,+0.31655], cluster+0.23310 CI[+0.14044,+0.32727] 더 나빴다.

- 판정: 단순 확률 기준선에도 경쟁력 부족
- 원인·병목 [원인 미상]: Frozen PFN끼리의 상대 개선이 단순 class prior 대비 utility를 보장하지 않았다. Calibration·적은 context·subgraph mismatch·전처리 중 원인은 미상이다.
- 닫은 범위: PPR/구조 정책 대비 개선만으로 좋은 missing-feature predictor라고 주장
- 닫지 않은 것: 확률 점수 비교를 accuracy의 같은 결론으로 자동 확대하지 않는다. Prior 대조는 사후 sanity check다.
- 재검토 조건: 극소 context·불완전 입력에서 단순 prior보다 좋은 확률 정보를 실제로 회수하는 추론 원리가 나올 때.
- 기준선: Uniform class probability, Laplace context prior, full graph, strongest fixed retrieval
- 최소 재검증: 학습 전에 단순 확률·majority 대조를 고정하고 NLL/Brier/accuracy를 독립 graph에서 함께 평가한다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-P01 [보존] 관측 상태를 선택 정책에 주는 독립 NLL 효과

Budget64 Observation−Structure NLL matched−0.16128 CI[−0.27147,−0.05871], cluster−0.24695 CI[−0.38936,−0.11246]. PPR 대비도 두 조건 NLL CI가0 미만이었다.

- 확대 금지: Feature 값의 인과적 효용, strong rule 초과, 확실한 accuracy 개선이나 실제 heterograph 전이로 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

### OBS-L01 [교훈] 학습 정책의 선택 단계 속도를 end-to-end 우위로 승격하지 않는다.

Matched budget64 selection1.76ms vs Paths13.73ms지만 PCFI+forward 포함 약146.82/155.90ms. Global statistics와 정책학습 비용은 이 합계 밖이다.

- 원문: [findings/sources/graphpfn-2026-11-01/observation.md](../findings/sources/graphpfn-2026-11-01/observation.md)

## Context path 선택과 고정 LP/PFN 혼합

실험 조건: 기존 실제 개발 graph, k5/20/class, 같은 context와 public test. 새 dataset holdout 아님.

### XFER-F01 [실패·중단] DBLP에서 통했던 context-LOO path 선택을 다른 graph로 옮기면 강하다.

Expanded LOO는 ACM k20 75.78% vs uniform LP85.55%, IMDB k5 27.53% vs uniform32.04%. Context accuracy로 목적을 바꿔도 IMDB k20 33.55%였다.

- 판정: 고정 선택 규칙의 전이 실패
- 원인·병목 [진단이 지지]: 소수 context의 ranking이 query utility와 맞지 않았다. 후보 pool 확대와 accuracy 목적 변경만으로는 해결되지 않았다. 추정 분산·risk mismatch의 상대 기여는 미상이다.
- 닫은 범위: 현재 LOO scalar score로 하나의 metapath를 고르는 규칙
- 닫지 않은 것: Context가 유용한 관계를 알려주는 다른 설정이나 relation 의미 추론 일반은 기각하지 않는다.
- 재검토 조건: 극소 context의 추정 불확실성과 실제 query risk를 다루며 task 내 utility를 식별하는 방법이 나올 때.
- 기준선: Uniform/union LP, individual paths, fixed GraphPFN path inputs
- 최소 재검증: 독립 graph에서 동일 context·후보 pool·선택 예산으로 사전 규칙을 고정해 검증한다.
- 원문: [findings/sources/graphpfn-2026-11-01/transfer.md](../findings/sources/graphpfn-2026-11-01/transfer.md); [findings/sources/graphpfn-2026-11-01/expanded.md](../findings/sources/graphpfn-2026-11-01/expanded.md); [findings/sources/graphpfn-2026-11-01/accuracy_selector.md](../findings/sources/graphpfn-2026-11-01/accuracy_selector.md)

### XFER-F02 [실패·중단] LP와 PFN 확률의 고정 혼합이 각자의 강점을 보존한다.

k20 FISF/LP 50:50: ACM82.20% vs LP85.55%, IMDB42.54% vs FISF44.55%. 같은 query에서 상보성은 있지만 혼합은 최강 단일 예측기를 못 넘었다.

- 판정: 고정 혼합 메인 방향 중단
- 원인·병목 [원인 미상]: 서로 맞히는 query가 다르다는 사실이 고정 확률 평균의 utility를 보장하지 않았다. 중복 증거가 원인일 수 있으나 직접 분리한 결과는 아니다.
- 닫은 범위: 현재 50:50/equal-thirds 사후 확률 혼합
- 닫지 않은 것: 모든 ensemble이 불가능한 것도, 각 branch가 feature-only/structure-only인 것도 아니다.
- 재검토 조건: 중복 evidence와 decision risk를 실제 관측량으로 구분하며 최강 단일 predictor를 보존하는 원리가 나올 때.
- 기준선: LP alone, PCFI/FISF+PFN alone, compute-matched fixed mixtures
- 최소 재검증: 같은 query·context에서 within-task 오류 보완성과 독립 개발/확인 graph의 평균·worst-case utility를 함께 본다.
- 원문: [findings/sources/graphpfn-2026-11-01/fusion.md](../findings/sources/graphpfn-2026-11-01/fusion.md)

## Plain relation adaptation과 CRR 제안 철회

실험 조건: 기존 Plain, exact-degree bipartite 재배선, 역할 교환·mechanism freeze·label shuffle 진단. CRR 추가 학습은 하지 않음.

### ROLE-L01 [교훈] Plain이 전역 relation utility 적응을 못한다는 전제로 새 router를 정당화하지 않는다.

공유 relation MLP에 relation-ID embedding이 없고, context 통계를 따라 역할 교환에 적응했다. Conflict end-to-end는 약2.09%p 하락해 완전 robustness는 아니다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

### ROLE-L02 [교훈] k=1의 centered label-alignment 통계에는 식별 한계가 있다.

클래스당 context 하나면 서로 다른 class centered one-hot 내적은 모두 −1/C다. 이 통계만 복잡하게 인코딩해도 새 정보가 생기지 않는다. 재배선 경로 품질 차이도 함께 고려해야 한다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

### ROLE-U01 [미실행] 새 CRR router·query-conditioned relation inference의 독립 학습

CRR 제안은 Plain과 기능이 겹쳐 철회했다. 기존 모델 진단을 새 router를 실제 학습해 실패한 결과로 세지 않는다.

- 원문: [findings/sources/graphpfn-2026-11-01/role_reversal.md](../findings/sources/graphpfn-2026-11-01/role_reversal.md)

## Random-fill controls 이후의 문헌 기반 후보 선별

실험 조건: 문헌 원문과 기존 실패의 범위를 대조한 연구 제안. 새로운 GPU 학습·성능 결과가 아니다.

### NEXT-L01 [교훈] Missing PFN·flow posterior·separable prior·tractable marginalization 자체를 새 발명이라고 제안하지 않는다.

StructuralMissingPFN은 incomplete context의 direct prediction과 auxiliary flow posterior를 이미 제시한다. PFNSeparability는 feature SCM과 label SCM을 독립적으로 뽑는 원리도 명시한다. GSPN은 모델 내 missing-variable marginalization을 제공한다. 새로운 적용이나 구현만으로 novelty가 확보되지 않는다.

- 원문: [papers/PFNSeparability.yaml](../papers/PFNSeparability.yaml); [papers/StructuralMissingPFN.yaml](../papers/StructuralMissingPFN.yaml); [papers/GSPN.yaml](../papers/GSPN.yaml)

### NEXT-U01 [미실행] 그래프·feature 생성 방식과 label rule의 coupling을 바꾸는 mechanism-crossed graph prior

현재 1순위 연구 후보지만 독립 SCM 원리 자체는 선행연구다. 단순 pairing 순서가 아닌 생성 joint distribution을 바꾸고 같은 Mask Plain 구조·학습량으로 비교해야 한다. REG-F01은 finite bank의 재가중 실패여서 이 후보를 실행해 실패한 결과가 아니다. RFILL-F01이 이 coupling을 실패 원인으로 입증한 것도 아니다.

- 원문: [papers/PFNSeparability.yaml](../papers/PFNSeparability.yaml); [findings/reference-regret.yaml](../findings/reference-regret.yaml); [findings/random-fill-controls.yaml](../findings/random-fill-controls.yaml)

### NEXT-U02 [미실행] Observed-cell 집합과 원래 graph를 직접 처리하는 sparse observation GraphPFN

숫자 completion을 생성하지 않고 (node,feature,value) 관측 token과 context-label token을 원래 graph에 연결한다. GRAPE의 관측 bipartite 표현과 Ephris의 sparse ICL과 겹치므로 단순 tokenization 조합은 novelty가 약하다. 성능·연산 예산을 동시에 확인할 보조 후보로 남긴다.

- 원문: [papers/Ephris.yaml](../papers/Ephris.yaml); [findings/source-error.yaml](../findings/source-error.yaml); [findings/binding-product.yaml](../findings/binding-product.yaml)

### NEXT-U03 [미실행] Sampling 대신 tractable circuit으로 missing variables를 소거하는 prior-fitted graph predictor

확률 circuit의 query conditional을 synthetic tasks에서 학습하는 고위험 후보다. GSPN의 exact local marginal과 conditional SPN은 기존 방법이므로 PFN을 붙였다는 사실만으로 새 메소드가 되지 않는다.

- 원문: [papers/GSPN.yaml](../papers/GSPN.yaml)

## Frozen feature sampling random-fill controls

실험 조건: 기존 개발32 graphs×6 conditions, 3 frozen checkpoints, 5 arms×4 views. Parameter updates0. 같은 process conditional replay와 null 비교.

### RFILL-F01 [실패·중단] 현재 frozen conditional Gaussian completion이 분산을 맞춘 random-fill/graph-prior null보다 안정적으로 좋은 예측을 제공한다.

uniform: Conditional0.91589, iid-matched0.91367, graph-prior-matched0.91325, iid-unit0.91305, wrong-sign0.91511; clustered: Conditional0.91352, iid-matched0.91728, graph-prior-matched0.91166, iid-unit0.91666, wrong-sign0.91460; Conditional−iid matched +0.00222 [-0.00277, +0.00718] / -0.00376 [-0.00973, +0.00184]; Conditional−graph prior matched +0.00265 [-0.00076, +0.00626] / +0.00186 [-0.00105, +0.00512]

- 판정: 사전 random-control gate 미통과; 현재 frozen conditional의 독립 가치 미입증
- 원인·병목 [원인 미상]: 동일 관측 입력·checkpoint·분산/forward-budget 대조에서 요구한 conditional law 우위가 없었다. 관측값에 조건화한 복원, 구조 random feature, 확률 smoothing의 원인별 기여를 완전히 식별한 결과는 아니다.
- 닫은 범위: 현재 zero-input-trained PFN과 Gaussian surrogate를 실제 feature posterior의 메인 메소드로 확대하는 것
- 닫지 않은 것: 실제 관측 피처의 직접 효용·context-conditioned feature-law 학습·shared suffix·coherence 일반은 평가하지 않았다. 분산 맞춤 null에는 second-moment 정보가 남고 unit null에는 실제 값 clamp가 남는다.
- 재검토 조건: Context와 관측 피처가 joint feature law에 제공하는 추가 정보를 proper feature score와 실제 query improvement로 입증하고, 같은 예산의 random/구조 입력보다 좋은 연산이 나올 때.
- 기준선: Variance-matched iid/graph prior, iid unit, zero/PCFI/FISF 입력 및 같은 예산·별도 학습의 강한 PFN. 실제 관측 입력 직접 경로와 proposal 경로를 구분.
- 최소 재검증: 새 학습 가설을 source/data/budget 고정 후 별도 bank에서 비교한다. Label-blind law 대조와 true-label-context conditional-law 증분을 분리하며 query labels를 forward에 사용하지 않는다.
- 원문: [findings/sources/random-fill-controls-2026-11-02/REPORT.md](../findings/sources/random-fill-controls-2026-11-02/REPORT.md); [findings/sources/random-fill-controls-2026-11-02/paired.json](../findings/sources/random-fill-controls-2026-11-02/paired.json)

### RFILL-L01 [교훈] 분산 맞춤 null을 완전히 무정보 random baseline이라고 부르지 않는다.

Iid/graph-prior variance-matched null에는 원래 conditional variance 정보가 남는다. Unit-noise에는 missing proposal의 관측값 정보가 없지만 실제 관측 위치의 값은 여전히 PFN에 제공한다. Null 우위/동등성을 전체 observed feature의 무용함으로 확대하지 않는다.

- 원문: [findings/sources/random-fill-controls-2026-11-02/README.md](../findings/sources/random-fill-controls-2026-11-02/README.md); [findings/sources/random-fill-controls-2026-11-02/protocol.json](../findings/sources/random-fill-controls-2026-11-02/protocol.json)

### RFILL-U01 [미실행] 이 대조 이후의 context-conditioned neural feature law와 shared PFN 학습

이번 실행의 optimizer update는0회다. 새 모델을 실제 학습한 결과가 아니며 CFB-U01의 미실행 항목을 완료나 실패로 바꾸지 않는다.

- 원문: [findings/sources/random-fill-controls-2026-11-02/REPORT.md](../findings/sources/random-fill-controls-2026-11-02/REPORT.md)

## Reference-regret prior curriculum

실험 조건: 공개 GraphPFN continued training, 새64/32 graphs, 3 schedule seeds, 15 runs. Finite bank sampling만 변경.

### REG-F01 [실패·중단] Reference-regret priority가 hard-loss sampling 및 같은 비용의 Random보다 유리하다.

Matched NLL Regret0.76377, Loss0.76467, RandomBudget0.76266. Regret−Loss−0.00090 CI[−0.00543,+0.00404], Regret−RandomBudget+0.00111 CI[−0.00145,+0.00387].

- 판정: 메인 방법 확장 보류
- 원인·병목 [원인 미상]: Reference의 독립 utility와 채점 비용을 정당화하지 못했다. Stale priority·짧은 학습·input loss 중 원인별 영향은 분리하지 않았다. Loss와 regret의 초기 Spearman0.4264는 다른 task를 강조함을 보일 뿐 효용 증명이 아니다.
- 닫은 범위: 현재 cached finite-bank reference-regret sampler 확대
- 닫지 않은 것: 학습된 생성기·library·무한 stream prior learning을 구현하거나 실패시킨 것은 아니다.
- 재검토 조건: Reference의 추가 감독 비용을 회수하면서 실제로 회수 가능한 task difficulty를 식별하는 학습·생성 원리가 나올 때.
- 기준선: Random, Manual, Loss priority, cold-score-cost-matched RandomBudget, frozen FISF
- 최소 재검증: 새 graph bank와 고정 비용으로 Loss/RandomBudget을 함께 넘는지 확인하고 task weighting과 generator 학습을 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

### REG-P01 [보존] 같은 update 수의 Random 대비 작은 Matched NLL 개선은 있었다.

Regret−Random−0.00185 CI[−0.00353,−0.00034]. Accuracy58.78→58.79%로 정확도 개선은 미확인.

- 확대 금지: Loss priority·같은 비용 대조·OOD를 넘었다고 확대하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md)

### REG-L01 [교훈] Singleton observed feature centering은 입력 정보를 없앨 수 있다.

Test99.5% whole-node에서 PCFI input 전체가0인 episode는 uniform20/32, clustered32/32. 관측 행0~1개에서 centering 손실을 직접 확인했다.

- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md); [findings/sources/graphpfn-2026-11-01/regret_scope.md](../findings/sources/graphpfn-2026-11-01/regret_scope.md)

### REG-U01 [미실행] 학습된 prior generator·새 mechanism library

실행한 것은 같은 official generator의 finite task bank를 재가중하는 pilot이다.

- 원문: [findings/sources/graphpfn-2026-11-01/regret.md](../findings/sources/graphpfn-2026-11-01/regret.md)

## Relation-wise Evidence Head

실험 조건: 소형 shared relation backbone, 3 seeds, consistent/noisy/conflict, k1/3/5/10/20/class, 99.5% missing.

### EVID-F01 [실패·중단] Relation 출처를 보존하는 head가 강한 Plain의 class decision을 개선한다.

Conflict Separate86.09%, Mean85.34%, Plain88.96%. Separate−Plain−2.88%p CI[−3.66,−2.09], Separate−Mean+0.74%p CI[+0.49,+1.03].

- 판정: 현재 Evidence Head No-Go
- 원인·병목 [원인 미상]: Plain 정답을 잃는 mass5.36%가 구제2.48%보다 컸다. 정보 중복과 representation 간섭 중 원인은 식별하지 못했다. Confidence를 낮춰서 NLL이 좋아졌다는 설명은 후속 분석에서 기각됐다.
- 닫은 범위: 현재 histogram/evidence head의 확대와 calibration/margin loss로 구제
- 닫지 않은 것: Relation source 정보가 불필요하다는 결론이나 다른 내부 추론 연산의 기각은 아니다.
- 재검토 조건: Plain의 강한 결정 능력을 유지하면서 출처별 정보의 독립 가치를 회수하는 장치가 나올 때.
- 기준선: Plain, Mean/Separate 동일 용량 control, fixed context-weighted evidence
- 최소 재검증: 동일 task에서 harmed/rescued partition과 NLL/Brier/accuracy를 함께 비교하고 conflict 강도별 증분을 확인한다.
- 원문: [findings/sources/graphpfn-2026-11-01/evidence.md](../findings/sources/graphpfn-2026-11-01/evidence.md); [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

### EVID-L01 [교훈] 확률 점수 개선을 relation 출처의 새 정보나 accuracy 개선으로 치환하지 않는다.

Conflict NLL0.5320→0.3585, Brier0.2905→0.2044; NLL 개선의 산술90.5%는 Mean에서도 나타났다. Confidence63.99→84.31%였다.

- 원문: [findings/sources/graphpfn-2026-11-01/evidence_utility.md](../findings/sources/graphpfn-2026-11-01/evidence_utility.md)

## Compiler–Executor GraphPFN pilot

실험 조건: Fully observed Boolean graph, 문법222개(학습186/미학습36), train/test512/64 graphs, 3 seeds, context총4/8/16.

### PROG-F01 [실패·중단] 고정 executor 위 compiler가 조합 일반화와 품질·비용에서 강한 대조를 넘는다.

미학습 조합: compiler61.59%/NLL0.68349, Plain56.15%/0.68425, relational features+TabPFN85.18%/0.33352, ProgramGrid87.73%/0.23073. Grid latency도 더 짧았다.

- 판정: 현재 compiler 확장 보류
- 원인·병목 [원인 미상]: 관측 정보는 Grid/TabPFN이 회수했지만 compiler는 context 증가에도 약61.6%에 머물렀다. 표현·search guidance·학습 최적화 중 정확한 원인은 식별하지 못했다.
- 닫은 범위: 현재 작은 compiler를 missing·대형 heterograph로 확대
- 닫지 않은 것: 프로그램 합성·library learning 일반은 기각하지 않는다. 후보 문법 자체는 학습 중 모두 제공했고 missing은 실험하지 않았다.
- 재검토 조건: Context에서 계산 구성을 더 효율적으로 식별하고 bounded enumeration/강한 PFN feature 기준선을 넘어서는 장치가 나올 때.
- 기준선: Cached ProgramGrid grammar prior, relational execution features+TabPFN, matched Plain
- 최소 재검증: 같은 grammar와 program prior, 후보 구조 접근, fresh graph와 heldout composition에서 품질과 end-to-end 비용을 맞춰 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/program.md](../findings/sources/graphpfn-2026-11-01/program.md)

### PROG-U01 [미실행] Primitive library를 실제로 학습하거나 missing hidden fact를 공동 주변화하는 확장

실행된 pilot은 고정 grammar/executor의 fully observed 비교다. BDD/weighted model counting, learned library, 무한 grammar 탐색 성과는 없다.

- 원문: [findings/sources/graphpfn-2026-11-01/program.md](../findings/sources/graphpfn-2026-11-01/program.md)

## Predictor reliability Gate A/B

실험 조건: ACM/IMDB 99.5% missing, k5/20/class, 세 context/mask draw의 개발용 진단.

### SEL-F01 [실패·중단] Context Brier·local reliability로 PFN/LP의 정답을 선택할 수 있다.

IMDB k20 FISF: PFN44.55%, LP35.41%, Global41.51%, Local41.92%, Oracle59.16%. Local은 최강 predictor보다2.63%p 낮았다.

- 판정: Gate B 실패, Gate C 보류
- 원인·병목 [진단이 지지]: Brier와 accuracy 목적이 불일치하는 사례를 직접 관측했다. IMDB seed11에서 LP의 test Brier0.6624가 PFN0.6672보다 좋지만 accuracy는35.54% vs44.65%였다. Context noise만으로 설명할 수 없다.
- 닫은 범위: 현재 crossfit Brier·local propagation 기반 predictor 선택 규칙
- 닫지 않은 것: Oracle headroom을 실제로 회수하는 다른 risk estimator는 기각하지 않는다. 학습 selector는 이 실험에서 미실행이다.
- 재검토 조건: 극소 context에서 정확도 효용과 uncertainty를 함께 식별하는 추정법이 나올 때.
- 기준선: 최강 단일 PFN/LP, global/local selector, fixed mixture
- 최소 재검증: Query 정답 없는 입력만으로 실제 선택 utility를 독립 graph에서 평가하고 proper score와 accuracy를 분리한다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

### SEL-P01 [보존] Within-query complementarity는 실제로 있다.

ACM k20 FISF oracle90.43%, IMDB59.16%. Oracle은 query 정답으로 만든 상한이다.

- 확대 금지: 큰 oracle gap만으로 selector 학습의 성공 가능성을 입증하지 않는다.
- 원문: [findings/sources/graphpfn-2026-11-01/reliability.md](../findings/sources/graphpfn-2026-11-01/reliability.md)

## Frozen GraphPFN sign-transport diagnostic

실험 조건: Known bipartite sign-gauged signal family; label/feature-law2x2;0/1/8/full sources;context1/4 per class;6 frozen checkpoints;0 updates.

### SIGN-P01 [보존] 관계를 따라 feature 부호가 바뀌는 과제에서, 알려진 법칙으로 PCFI 복원을 바꾸면 같은 frozen 공개 GraphPFN의 classification이 개선된다.

mask:accuracy52.405028%/NLL0.709999/Brier0.512094; pcfi:accuracy50.190226%/NLL0.715966/Brier0.520255; known_signed:accuracy56.026584%/NLL0.707501/Brier0.506394; positive_lp:accuracy46.683098%/NLL0.796353/Brier0.572964; signed_lp:accuracy56.260834%/NLL0.687817/Brier0.492625; uniform:accuracy50.265421%/NLL0.693147/Brier0.500000; known_signed-pcfi accuracy%p +5.836358 [95% CI +2.291311, +9.440175]; known_signed-mask accuracy%p +3.621556 [95% CI +0.027501, +7.207857]; known_signed-positive_lp accuracy%p +9.343486 [95% CI +5.338770, +13.482302]; known_signed-signed_lp accuracy%p -0.234249 [95% CI -5.629258, +4.934528]. Primary headroom=True, strong direct/LP/feature-use floor=False.

- 확대 금지: Known-law g bit를 받은 reconstruction 진단이다. 새 graph에서 부호 법칙을 추론한 학습 결과, arbitrary heterophily·real graph·schema 전이 또는 메소드 novelty가 아니다. 강한 baseline floor=False. Signed LP 대비 우위와 동일-query zero-source 대비 feature-use 이득은 확인하지 못했다. Frozen out-of-prior 비교이며 새 prior에서 matched 학습한 구조 우열이 아니다. 새로운 학습은 실행하지 않았다.
- 원문: [findings/sources/sign-transport-2026-11-02/REPORT_KO.md](../findings/sources/sign-transport-2026-11-02/REPORT_KO.md); [findings/sources/sign-transport-2026-11-02/METHOD_KO.md](../findings/sources/sign-transport-2026-11-02/METHOD_KO.md); [findings/sources/sign-transport-2026-11-02/protocol.json](../findings/sources/sign-transport-2026-11-02/protocol.json); [findings/sources/sign-transport-2026-11-02/results/statistics.json](../findings/sources/sign-transport-2026-11-02/results/statistics.json); [findings/sources/sign-transport-2026-11-02/results/common_query_interventions.json](../findings/sources/sign-transport-2026-11-02/results/common_query_interventions.json); [findings/sources/sign-transport-2026-11-02/results/audit.json](../findings/sources/sign-transport-2026-11-02/results/audit.json); [findings/sources/sign-transport-2026-11-02/results/cost.json](../findings/sources/sign-transport-2026-11-02/results/cost.json); [findings/sources/sign-transport-2026-11-02/results/preflight.json](../findings/sources/sign-transport-2026-11-02/results/preflight.json)

### SIGN-L01 [교훈] Feature 복원의 부호 제약, label heterophily, frozen PFN 분류, known-law 정보 우위를 구분한다.

단일 channel path의+1 source/0 missing start에서 PCFI는 음수를 만들지 못했고 parity 변환은 반대 부호를 만들었다. 최종 factorial은 graph/labels/context/query/mask를 feature law 간 고정했다. Primary 결과 mask:accuracy52.405028%/NLL0.709999/Brier0.512094; pcfi:accuracy50.190226%/NLL0.715966/Brier0.520255; known_signed:accuracy56.026584%/NLL0.707501/Brier0.506394; positive_lp:accuracy46.683098%/NLL0.796353/Brier0.572964; signed_lp:accuracy56.260834%/NLL0.687817/Brier0.492625; uniform:accuracy50.265421%/NLL0.693147/Brier0.500000; known_signed-pcfi accuracy%p +5.836358 [95% CI +2.291311, +9.440175]; known_signed-mask accuracy%p +3.621556 [95% CI +0.027501, +7.207857]; known_signed-positive_lp accuracy%p +9.343486 [95% CI +5.338770, +13.482302]; known_signed-signed_lp accuracy%p -0.234249 [95% CI -5.629258, +4.934528].

- 원문: [findings/sources/sign-transport-2026-11-02/REPORT_KO.md](../findings/sources/sign-transport-2026-11-02/REPORT_KO.md); [findings/sources/sign-transport-2026-11-02/METHOD_KO.md](../findings/sources/sign-transport-2026-11-02/METHOD_KO.md); [findings/sources/sign-transport-2026-11-02/protocol.json](../findings/sources/sign-transport-2026-11-02/protocol.json); [findings/sources/sign-transport-2026-11-02/results/statistics.json](../findings/sources/sign-transport-2026-11-02/results/statistics.json); [findings/sources/sign-transport-2026-11-02/results/common_query_interventions.json](../findings/sources/sign-transport-2026-11-02/results/common_query_interventions.json); [findings/sources/sign-transport-2026-11-02/results/audit.json](../findings/sources/sign-transport-2026-11-02/results/audit.json); [findings/sources/sign-transport-2026-11-02/results/cost.json](../findings/sources/sign-transport-2026-11-02/results/cost.json); [findings/sources/sign-transport-2026-11-02/results/preflight.json](../findings/sources/sign-transport-2026-11-02/results/preflight.json)

## 출처 중첩·공유 보간 오차 attention

실험 조건: 1 seed, 64 updates×2 episodes, 16 heldout synthetic graphs; stochastic feature 위치를 맞춘 paired 결과.

### SRC-F01 [실패·중단] Source covariance를 보정하면 분류가 개선된다.

Vanilla accuracy 57.01%, covariance 57.04%. Covariance−vanilla +0.032%p, graph CI [−0.092,+0.157]. Source correspondence 파괴 시 working error score는 0.1395 증가했지만 분류 변화는 작았다.

- 판정: 독립적인 task-level 이득 미확인
- 원인·병목 [원인 미상]: 오차 proxy가 출처 구조를 감지하는 것과 label decision에 유용한 보정은 달랐다. Working covariance가 실제 posterior error인지도 보장되지 않는다. 분류 실패의 고유 원인은 미상이다.
- 닫은 범위: 현재 source-overlap penalty/log 보정을 메인 방법으로 확대
- 닫지 않은 것: 출처 의존성이 label 예측을 실제로 왜곡하는 다른 prior와 정확한 uncertainty 추론은 기각하지 않는다.
- 재검토 조건: 공유 imputation error와 label risk를 연결하는 검증 가능한 오차 모델·추론법이 나올 때.
- 기준선: Vanilla continuation, matched/log/diagonal, covariance-shuffled controls
- 최소 재검증: 분류 성능을 바꾸는 source dependency task에서 correspondence control과 새 graph·seed를 함께 비교한다.
- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

### SRC-L01 [교훈] 추가 사전학습의 효과를 오차 보정 모듈 효과로 치환하지 않는다.

Initial 대비 accuracy는 올랐지만 matched continuation 대비 증분은 약0.03%p였다. Balanced accuracy와 macro-F1은 initial 대비 하락했다.

- 원문: [findings/sources/graphpfn-2026-11-01/source_error.md](../findings/sources/graphpfn-2026-11-01/source_error.md)

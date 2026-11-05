# FP05 method/prior-art review

2026-11-05. 검토: PLAN_v1, observation_budget_pilot/REPORT, internal_pcfi_pilot/REPORT_KO, internal_pcfi_confirmation/REPORT_KO, FP04 DECISION_KO/RESULT_METHOD. 현재 주기의 다른 review는 읽지 않았다. 구현·GPU·학습·branch 변경은 하지 않았다. 이 파일만 추가했다.

**판정: 범위를 바로잡고 작은 명세 충돌을 닫은 뒤 bounded pilot 진행 가능.** 강한 공개 PFN에서 내부 연산을 선택하고 exact expected-gradient와 비교하는 질문은 실행할 가치가 있다. 그러나 현재 구조는 **RL controller + CE로 학습한 graph adapters**이며 LimiX Transformer를 RL로 fine-tune하는 실험이 아니다. 새 RL 알고리즘이나 유용한 feature inference의 성과를 미리 가정할 근거도 없다.

점수(1–5): 실행 가능한 메소드 4, 현재 명세 완결성 3, RL 고유 이득을 식별하는 설계 3, novelty의 사전 확실성 1. 아래 수정들은 새 architecture·추가 trained arm 없이 대부분 처리할 수 있다.

## 닫힌 실험에서 이어받아야 할 사실

- Observation-budget은 frozen GraphPFN에 넣을 subgraph를 REINFORCE로 골랐다. Mask-aware 선택의 상대 이득은 있었지만 강한 evidence-paths를 이기지 못했고, missing 조건 NLL은 context-frequency prior보다 나빴다. 새 방법은 resident-node 선택이 아닌 내부 compute 선택이라는 차이가 있다. “RL을 처음 쓴다”는 주장은 틀리다.
- Internal-PCFI는 실제 공개 PFN과 약 4M adapters/encoder/decoder를 CE로 학습한 12 fits×256 updates였다. Context router는 fixed internal보다 primary accuracy가 약 .487%p 낮았고, fixed placement의 후속 독립 bank 차이도 +.220%p CI[−.332,+.797]로 확인되지 않았다. 새 계획은 literal PCFI 내부 배치의 반복이 아니지만, “이제 처음으로 pretrained PFN을 실제 학습한다”는 서술도 틀리다.
- FP04는 19,748-parameter rank4/4-round encoder를 모두 학습했지만 네 gate가 모두 실패했다. Query가 항상 feature-free였고 full도 nonquery-visible이었다. FP05가 original 12 Transformer blocks와 observed-query training support를 보존하는 것은 실제 변경이다. 이 변경이 실패 원인을 해결한다고 확정할 수는 없다.

## 주요 이슈 — 최대 5개

### 1. RL backbone tuning과 controller tuning의 경계를 명시해야 한다

Policy input hidden states를 detach하고 graph adapters에 CE만 전달하면 policy-gradient가 LimiX backbone으로 흐르지 않는다. Backbone/feature embedding/y-decoder를 freeze하는 현재 선택은 명확하고 저비용이지만, 사용자가 기대한 “강한 PFN 자체를 RL reward로 바꾼다”와는 범위가 다르다. 반복되는 graph adapters도 reward-gradient가 아닌 sampled/policy-weighted CE로 바뀐다.

허용되는 결론은 “공개 PFN의 CE adapter adaptation과 learned computation controller의 결합이 고정 연산 및 exact expected-gradient보다 유리한가”다. 이름·표·parameter audit에서 controller policy-gradient / adapter CE / frozen Transformer를 나눠야 한다. 진짜 backbone RL을 추가하는 큰 프로젝트로 지금 범위를 넓힐 필요는 없다. 대신 현재 질문을 정확하게 진행하고, 그 결과를 RL-backbone 검증으로 재분류하지 않는다.

새 trained arm 없이 각 학습 checkpoint의 `000` replay를 보존하면 동일 adapted parameters에서 추가 routing의 증분을 볼 수 있다. 이는 별도 policy-only 훈련의 인과 효과를 완전히 분리하지는 않지만, 다른 arm의 서로 다른 adapters를 비교한 차이를 policy 효과라고 부르는 오류는 줄인다. 필요한 추가 inference 비용은 사전에 예산에 포함한다.

### 2. Trace별 feature RNG에 대한 두 문장은 충돌한다 — exact control의 blocking issue

V1 fixed arm은 update 안에서 서로 다른 8 feature RNG를 쓰고, 다른 곳에서는 trace 간 동일 feature-position RNG를 요구한다. Exact enumeration은 하나의 `(task,mask,context,ξ)`에 대해

`p_phi(a | task,ξ) = product_t pi_phi(a_t | detached_state_t,prefix)`

가 같은 prefix tree를 이루고 합이 1이어야 한다. 서로 다른 ξ로 만든 8 trace의 확률은 같은 tree의 leaf distribution이 아니다. Group reward 차이에도 action 외의 변화가 섞인다.

**최소 수정:** update당 하나의 ξ를 모든 arm·모든 trace에 공유하고, ξ는 update 사이에서만 바꾼다. Fixed의 8 duplicate presentations는 동일 CE의 평균이라 단일 deterministic gradient와 같다. 실제 forward를 재사용한다면 8 independent tasks나 8 실제 forwards로 기록하면 안 된다. Prefix probabilities·old log-probability를 어느 parameters/ξ에서 얻었는지도 고정해야 한다. Root는 이 공통-RNG 수정안을 v2에 반영하겠다고 회신했다.

### 3. 한 번의 on-policy step에서 PPO clipping은 효력이 없고, group normalization은 exact objective와 다르다

Sampling 직후 parameters를 바꾸지 않고 한 번만 gradient를 구하면 `ratio=exp(logpi−logpi_old)=1`이다. 따라서 clip .2가 trust-region 역할을 한다는 설명은 성립하지 않는다. Group-normalized score-function gradient로 학습 자체는 가능하다. Root가 arm을 **group_pg: GRPO-inspired group-normalized REINFORCE**로 바꾸겠다는 수정은 적절하다.

Sample group의 mean/std가 자기 reward를 포함하므로 이것을 exact expected reward gradient의 불편 추정량이라고 부르면 안 된다. 특히 reward 표준편차로 나누면 episode별 action contrast 크기에 따른 가중치가 바뀐다. Direct와 group_pg의 결과 차이는 분산뿐 아니라 이 재가중의 영향도 포함한다. 모두 collapse하면 advantage는 0이며, cost만 reward 차이를 만들 수도 있다. Advantage-zero fraction, entropy, trace uniqueness, CE spread와 cost는 함께 남긴다.

8개뿐인 trace에 exact direct arm을 두는 것은 강점이다. Group_pg가 direct보다 낫더라도 PPO/GRPO의 일반적 우위나 정확한 목적함수 최적화의 우위를 뜻하지 않는다. V1의 entropy 항도 exact trajectory entropy인지 방문-state entropy인지 불명확했다. Root는 양 learned-policy arm의 entropy coefficient를 0으로 고정하겠다고 회신했으므로, 그 변경을 protocol에 명시하면 이 혼동은 닫힌다.

### 4. Primary+utility gate가 모두 통과해도 RL이 feature inference를 개선했다고 단정할 수 없다

예를 들어 fixed observed/none=50/40%, learned observed/none=55/45%면 learned가 fixed를 이기고 feature utility도 양수지만, feature utility의 증분은 0이다. 기존 PCFI/PFN의 feature 이용은 그대로 두고 graph-only 연산만 개선했을 수 있다. FP04에서 none/shuffle이 실제 성능을 거의 유지한 사실 때문에 이 구분은 중요하다.

기존 모든 arm에 저장할 none/shuffle 예측을 이용해 아래 두 difference-in-differences의 paired graph CI를 **사전 secondary mechanism 판정**으로 추가하는 것이 최소 수정이다.

`(Acc_group_pg,obs − Acc_group_pg,none) − (Acc_fixed,obs − Acc_fixed,none)`

및 none을 shuffle로 바꾼 비교다. 같은 eligible disjoint graph/rate cases와 원래 physical missing-query subset을 유지한다. Imputed row가 dense가 됐다는 이유로 own-missing query를 다시 정의하면 안 된다. None은 mask도 바꾸므로 availability와 값의 효과가 함께 들어가며, mask를 보존하는 shuffle 비교가 값의 node association에 더 직접적이다. Root는 이 DiD를 v2에 고정하겠다고 회신했다.

원 primary를 성공시키기 위해 DiD를 새 주 지표로 바꾸지는 않는다. 대신 DiD 근거가 없으면 성공하더라도 “adaptive graph computation improvement with retained feature utility” 범위로 보고한다. LP/context-frequency와 frozen PCFI+PFN도 반드시 함께 두고, 약한 fixed CE arm만 이긴 것으로 missing inference 전체의 경쟁력을 주장하지 않는다.

### 5. 128 updates×12 fits는 pilot ceiling으로는 타당하지만 학습 충분성을 보증하지 않는다

네 arm×세 seeds는 CE continuation, random trace training, exact direct, sampled policy-gradient를 분리하는 최소한의 비교다. 넓은 loss/rank/temperature sweep은 아니므로 **12 fits 유지에 조건부 찬성**한다. 그러나 update당 8 trace는 같은 episode에서 나온 computation alternatives다. 128 updates는 1,024 independent graph tasks가 아니라 최대 128 task groups/fit이며, 256 train base graphs 전체가 학습된다는 보장도 없다. “12,288 presentations”와 unique base/episode coverage를 구분해 기록해야 한다.

이미 pretrained 모델을 조정하는 bounded pilot이므로 128을 성공 보장이 없는 고정 cap으로 정하는 것은 가능하다. 이전 observation policy가 같은 128을 썼다는 사실은 충분성 근거가 아니다. 최종 source lock 전에 작은 resource/numeric smoke로 실제 AdamW update·action의 작동·예상 비용을 확인하고, plan에 허용한 compute 사유의 감축만 명시적 revision으로 처리한다. 실패 후 같은 final bank를 보고 step 수를 늘려 구제하지 않는다.

Official prior의 크기·자식 수 제약으로 structural draw를 retry하면 실제 prior는 조건부로 바뀐다. Retry 이유/비율·실현 K·class coverage를 기록해야 한다. Real random projection의 위치(RMS/PCFI 전후), RNG, 물리적 mask 전달도 구현 전에 고정한다. Historical native/PCA references와 새 16-dimensional PCFI는 같은 front end가 아니다. Full natural-query guard의 point/CI 판정과 rare undefined own-missing subset의 집계 규칙도 사전에 닫아야 한다.

## 선행 연구와 허용할 기여

Policy-GNN은 RL로 node별 aggregation 횟수를 고르고, RMS는 downstream reward로 heterogeneous meta-path를 선택한다. Adaptive graph computation 자체는 새 원리가 아니다. 이번의 차별 질문은 strong pretrained PFN의 내부 residual computation과 missing observation 조건에서, exact direct/SFT 통제를 넘는 증분이 있는가다. Homogeneous 3-bit trace 실험으로 heterograph meta-path transfer를 주장할 수 없다. [Policy-GNN](https://arxiv.org/abs/2006.15097), [RMS](https://arxiv.org/abs/2112.12845).

DeepSeekMath의 GRPO는 group-relative advantage를 사용하는 PPO 계열의 선행이다. 현재 한-step controller는 그 이름을 그대로 재현했다기보다 제한된 group-normalized policy gradient다. Graph-GRPO는 graph flow model의 graph generation을 RL로 학습하며 GraphPFN node classification과 목적이 다르다. 공통 약어가 선행 차별성이나 성능 근거가 되지 않는다. [DeepSeekMath](https://arxiv.org/html/2402.03300v3), [Graph-GRPO](https://proceedings.mlr.press/v306/zhu26b.html).

## 최소 확인 — 최대 3개

1. **Original000와 optimizer의 실제 일치:** public wrapper parity, live adapter CE gradients/실제 parameter 변화, detached policy state·frozen Transformer hash, serialized state의 query-truth 부재를 확인한다. 000이 같은 입력/RNG의 원 모델과 다르면 수정 후 비교한다.
2. **공통 ξ의 8-leaf 수학:** 확률 합1, exact direct policy gradient의 finite difference/score-function 합 일치, group_pg gradient sign·detached reward, same-task grouping을 확인한다. FP16 roundoff와 trace 간 실제 CE 차이를 분리한다.
3. **잠긴 집계·비용:** primary physical missing-query mask, utility eligibility, graph 안 seed/rate 평균 후 CI, DiD, full guard, empty handling을 toy saved probabilities로 독립 재현한다. Common initialization/schedule/feature ξ, unique task groups, 실제 forward/backward와 원 PCFI/projection 비용을 기록한다.

**종료:** 위 공통-RNG·명칭/gradient 경계·feature DiD를 반영하면, 새 arm을 늘리지 않고 현재 내부 compute-selection pilot을 진행할 수 있다. Root가 예고한 v2는 아직 이 문서에서 구현·실행 검증한 대상이 아니다. 결과가 좋더라도 controller+CE adaptation의 증분으로 보고하며, 결과가 나쁘면 이 3-bit operator/official restricted prior/128-step 예산을 닫는다.

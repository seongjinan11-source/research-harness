# FP05 수학·추론 사전 검토

검토일: 2026-11-05. 대상은 `PLAN_v1.md`와 pinned upstream commit `3b9b115490249cc777227c846babfb55f35bd8c4`의 실제 `light.py`, `layers.py`, `limix.py` 및 vendored Transformer 일부다. 경로는 `C:/Users/seongjin/Desktop/code/mgpfn/src/mgpfn/upstream.py`의 SRC를 따라 확인했다. 다른 현 주기 리뷰는 읽지 않았고 구현·모델 실행·GPU·branch 변경은 하지 않았다.

**판정: 아래 확률/RNG 계약과 gradient 경계를 고정하면 구현 가능하다.** Root가 채택한 common ε, 독립 action RNG, `group_pg` 명칭, 두 policy arm의 entropy0 수정은 적절하다. 이 파일은 V1의 문제와 전달받은 수정 방향을 기록하며 아직 V2 구현 통과 판정은 아니다.

## 핵심 수정 사항

1. **Eight traces가 하나의 확률분포가 되려면 upstream randomness를 공유해야 한다.** V1의 “8개의 서로 다른 feature RNG seeds”를 direct의 각 trace에 하나씩 배정하면 같은 prefix의 state가 trace마다 달라진다. 그때 여덟 경로 확률은 일반적으로 합1이 아니며 exact expectation이 아니다. Task/update group마다 한 ε를 고정해 모든 branch와 arm에 공유한다. ε는 Light의8 random node features뿐 아니라 vendored `add_embeddings()`의 random orthogonal feature positions도 포함한다. Eight independent ε마다 exact enumeration을 원하면64 trajectories가 필요하므로 현재 예산과 다르다. Fixed000을8번 같은 ε로 반복하는 경우 중복임을 기록한다. Policy action RNG는 별도 NumPy generator로 유지하여 forward RNG 복원이8번 같은 action을 강제하지 않게 한다. Root가 이 수정을 채택했다.

2. **One-step on-policy 업데이트에서 PPO clip은 작동하지 않는다.** Old/current policy가 같고 같은 저장 state를 쓰면 ratio=1이므로 첫 gradient에서 clip.2는 활성화되지 않는다. 이는 결함이 아니라 계획한 one-update의 수학이다. `group_pg`/group-normalized REINFORCE로 부르고 clipped-PPO 안정성 효과를 주장하지 않는 root 수정에 동의한다. 유한 group의 표본 mean/std로 정규화한 advantage는 raw expected reward의 불편 추정량과 같지 않다. Exact direct와 group_pg의 차이를 variance 하나로만 해석해서는 안 된다.

3. **θ/φ의 stop-gradient와 update 순서를 명시한다.** Policy 입력 hidden만 detach한다. Frozen LimiX 파라미터는 `requires_grad=False`로 두되 Transformer forward 전체를 `no_grad()`로 감싸면 앞 adapter의 CE gradient가 끊긴다. 여덟 trajectory는 동일한 pre-update θ/φ에서 계산하며 모두 backward한 뒤 optimizer를 한 번만 step한다. Trajectory 사이에 θ를 바꾸고 old states/logp를 재평가해서는 안 된다. Gradient accumulation은 가능하다. θ gradient clip1을 모든 arm에서 동일하게, φ clip1을 두 policy arm에서 별도로 적용하면 policy gradient 크기로 adapter clipping이 달라지는 혼선을 피할 수 있다.

4. **Upstream 연산·출력 계약을 유지한다.** Wrapper의 정상 연산은 base Transformer→residual conv→residual MLP다. Action1은 그 결과에 같은 `conv(graph,x)`→같은 `mlp(graph,x)`를 추가로 한 번 적용한다. Transformer를 한 번 더 돌리거나 label token에만 adapter를 적용하면 제안한8-trace 모델이 아니다. Graph adapter는 각 token을 받는 [1,N,T,192] tensor에서 동작한다. Policy는 그 중 `x[0,:,-1,:]` label-token state만 읽으면 된다. Light의 원래 context-first graph/feature permutation과 전체 preprocessing을 유지하는 wrapper hook 접근이 적절하다.

5. **Loss·평가 subset과 전처리 순서를 고정한다.** Light는 context 위치에 dummy zero 출력을 붙이고, LimiXWrapper는 이미 log-softmax를 반환한다. 실제 noncontext query에만 CE/NLL을 적용하고, supplied K에 해당하는 출력을 선택한 뒤 같은 방식으로 재정규화한다. Query truth의 unique/max로 K를 정하지 않는다. Random projection16과 observed RMS/PCFI 순서는 명시해야 한다: project→PCFI와 native PCFI→project는 같은 연산이 아니다. Seed-only random matrix에는 데이터 fitting이 필요 없으며 hidden 값·label을 사용하지 않아야 한다. All arms 및 frozen matched000 reference에 동일한 경계를 적용한다.

## 고정 가능한 목적식

Task/mask/context를 D, 공통 모델 RNG를 ε, 추가 adapter 파라미터를 θ, policy 파라미터를 φ라 하자. a=(a1,a2,a3)∈{0,1}³이고 state는 해당 prefix를 실제 실행한 뒤 정상 wrapper 직후에 얻는다. 필요한 경우 이전 action bits와 valid-prefix mask를 정적 입력에 명시한다. 요약 통계만으로 서로 다른 prefix가 항상 구별된다고 보장할 수는 없다.

```text
s_t(a_<t) = summary(stopgrad(h_theta,t(a_<t;D,epsilon)), allowed_metadata)
p_phi(a)  = product_t pi_phi(a_t | s_t(a_<t))
ell_a     = mean query CE of the declared training query set
c_a       = a1+a2+a3
```

Common ε와 동일 prefix state를 사용할 때 ∑a pφ(a)=1이다. Enumerated eight traces의 prefix별 policy logits가 일치하는지 먼저 검사한다. Policy probabilities/logp, CE, reward, advantage는 FP32 이상으로 계산한다. 비용 c는0..3의 추가 adapter pair 수이며 전체 GPU latency나 전체 Transformer depth가 아니다.

Direct의 두 gradient는 다음처럼 구분할 수 있다. sg는 stop-gradient다. β=0은 root가 채택한 수정이다.

```text
L_theta = sum_a sg(p_phi(a)) * ell_a
L_phi   = sum_a p_phi(a) * sg(ell_a + .01*c_a)
```

Policy hidden detach 때문에 θ update는 policy-state dependence를 제외한 **conditional/semi-gradient**다. 전체 coupled expected objective의 모든 θ 경로를 미분한다고 주장하지 않는다. 두 loss를 더해 backward할 수 있지만 scalar CE를 이중으로 더한 값을 실제 expected CE라고 보고해서는 안 된다. Fixed는 ell000, uniform은 sampled8 CE 평균을 θ에 사용한다. Group_pg의 θ update도 sampled8 CE 평균이며, 현재 p에 대한 이 conditional CE gradient의 표본 추정이다.

Group_pg에서는 같은 D/ε/current policy에서 독립적으로 복원 추출한8 trajectories를 사용한다.

```text
R_g = sg(-ell_g - .01*c_g)
A_g = sg((R_g-mean(R))/(population_sd(R)+1e-6))
L_phi_group = -(1/8) * sum_g A_g * log p_phi(a_g | saved_detached_states_g)
```

Old-logp ratio로 표현해도 step 전 값은1이고 같은 first gradient다. Duplicate traces를 그대로 유지하고 unique trace 수, σ≈0/advantage0 빈도, action probabilities와 비용을 기록한다. 표본 평균 baseline의 자기 포함과 표본 σ 정규화 때문에 이 estimator를 direct의 정확 gradient와 숫자로 같아야 한다고 검사하면 잘못이다. **Raw unnormalized reward의 exact score-function expectation**과 direct φ gradient의 일치는 별도 toy oracle로 검사한다.

Entropy를 다시 도입하려면 두 arm 모두 같은 trajectory entropy `−sum_a p(a)log p(a)` 또는 정확히 정의한 conditional entropy estimator를 사용해야 한다. 단순히 eight leaf에 있는 step entropy를 무가중 평균하는 것은 다른 objective다. 이번 pilot에서 양쪽 β=0으로 고정하는 것이 가장 간결하다.

## 최소 독립 검사·대조군

- **확률·gradient toy:** Three-stage binary tree의 독립 float64 계산으로 leaf probability 합1, prefix 일관성, direct φ finite difference와 exact raw-reward score expectation을 비교한다. θ finite difference는 p를 기준 θ에서 고정하고 CE만 변화시킨다. p를 다시 계산하는 finite difference는 의도적으로 detach한 θ 경로까지 포함하므로 올바른 oracle가 아니다. Group_pg는 known reward에서 좋은 action logp를 올리는 부호, reward detach, φ nonzero gradient, all-equal reward의 zero advantage를 검사한다.
- **실제 upstream parity:** 같은 public checkpoint, 같은 PCFI input, 동일 random node features/feature positions에서 original Light와 disabled-hook/forced000의 출력 및 adapter gradient가 일치해야 한다. Model `.eval()`은 dropout 모드만 고정하며 autograd는 유지한다. Conv/MLP parameter와 residual norm을 포함한 정확한 trainable-name whitelist를 저장하고 frozen LimiX digest 불변을 확인한다. Backward/checkpoint 재계산이 action을 새로 뽑거나 rollout metadata를 덧붙이지 않도록 trace를 고정한다.
- **누출·대칭·reference:** Internal context는 original Light의 stable context-first 순서이고 y_train 역시 그 순서와 맞춰야 한다. Policy same/different-class cosine은 context equality만 사용하고 support<2/zero norm에서 validity flag와 유한 fallback을 갖는다. Query-summary mask는 external query IDs를 내부 permutation으로 변환한다. No query labels/hidden original values가 policy state나 PCFI/projection에 들어가지 않도록 전체 pipeline poison 검사를 한다. Policy 요약 연산의 node/context-class-group permutation 불변성과 전체 pretrained model의 class/feature-order sensitivity를 구분한다. Fixed random row noise도 node permutation에 따라 같이 permute해야 하며, 같은 seed로 새 noise를 그리는 것만으로는 node parity가 되지 않는다. Upstream label embedding과 grouped/random feature embedding이 있으므로 전체 모델의 엄격한 class/feature equivariance를 새로 보장한다고 주장하지 않는다.

Frozen matched-PCFI000은 같은 input/RNG에서 adapter CE adaptation 이전 성능을 보여 주는 필수 reference다. Native-NaN/public ensemble/LP는 추가로 유용하지만 preprocessing이 다른 점을 분리한다. Uniform inference의8-forward probability mixture는 greedy policy의1-forward prediction보다 계산량이 크므로 실제 per-prediction forward 수와 추가 adapter 총수도 보고한다. Local sequential argmax는 global MAP trace나 최저 query-loss trace와 같지 않으며, 평가 도중 oracle trace로 바꾸지 않는다.

## 해석 범위

이 설계는 pretrained12-block PFN의 세 위치에서 추가 graph computation을 학습하는 유한8-action controller 시험이다. Source0에서도 graph/context 계산 선택은 가능하지만 관측 feature utility의 증거가 되지는 않는다. θ가 각 arm의 trajectory 분포에 따라 다르게 CE-adapt되므로 policy-only와 adapter-only의 인과적 기여가 자동 분해되지 않는다. Root가 채택한 “controller의 RL + adapter의 CE” 표현이 정확하다. Direct보다 group_pg가 좋아져도 새 RL 알고리즘이나 backbone policy-gradient 학습을 입증하지 않으며, 실패해도 실제 Transformer 전체나 모든 adaptive computation을 반증하지 않는다. 위 계약을 V2와 독립 검사에 고정하면 추가 수학적 veto 없이 bounded pilot을 구현하는 데 동의한다.

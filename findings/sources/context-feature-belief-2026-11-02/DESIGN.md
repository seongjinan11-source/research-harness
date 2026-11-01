# Context-conditioned feature belief를 사용하는 GraphPFN — 다음 설계

2026-11-02. **연구 방향으로는 진행 후보, 새 학습·아키텍처는 미실행**이다. Gaussian frozen pilot의 수치와 아래 수학 예시를 새로운 모델의 학습 성과로 합치지 않는다.

## 현재 근거와 목표

`feature_marginal_pfn`에서 completion 확률 평균은 같은 fixed PFN의 mean·PCFI 입력보다 NLL이 낮았다. 노드 covariance의 추가 효과는 Clustered에서만 확인됐다. 실제 feature evidence의 활용인지 구조 random-feature 효과인지는 분리하지 않았다. 기존 No-Go와 보존 결과 FMARG-F01/P02/P03을 유지한다.

목표는 입력 `D=(G,X_obs,M,Y_C)`에서 새 graph의 context가 **joint feature law**를 갱신하고, 그 불확실성을 예측에 전달하는 것이다. 추론 시 target graph의 weight update와 사전 PCFI는 사용하지 않는 최종 형태를 지향한다. 가변 feature 폭은 shared cell/feature encoder로 처리하며, 피처 수가 같아졌다는 이유만으로 semantic schema 전이를 입증했다고 하지 않는다.

## 선행연구를 확인해 바꾼 주장

[MartingaleSSL](https://arxiv.org/abs/2605.11846)은 사용자가 제안한 관측 refinement의 평균 일관성과 직접 겹친다. 이 identity나 해당 loss를 GraphPFN에 적용하는 사실을 새 원리라고 부르지 않는다. [MaM](https://proceedings.mlr.press/v235/liu24az.html)은 arbitrary marginal의 직접 근사를 다룬다. 두 논문을 이 설계의 성공 근거로 취급하지 않는다.

우리가 남겨야 할 주장은 **context-conditioned graph feature law의 타당성, 이를 이용한 예측의 증분, 실제 계산 공유의 품질–비용 이득**이다. 세 요소의 결합도 아직 novelty 확정은 아니다.

## 모델 연산의 제안

\[
H=E_\theta(G,X_{\rm obs},M,Y_C),\qquad
z^{(s)}\sim q_\phi(z\mid H),\qquad
X_U^{(s)}=T_\phi(H,z^{(s)},\epsilon^{(s)}).
\]

U는 missing cell 집합이다. 관측값은 복사해서 고정한다. 하나의 global draw가 graph 전체의 missing cells를 함께 결정해야 여러 query 사이에서 양립하는 feature world가 된다. 각 cell의 독립 noise만 주는 것은 이 공동 가설과 구별한다. Query 정답은 q와 encoder 입력에 넣지 않는다.

구체적인 첫 density 함수족 후보는 mixture of low-rank Gaussians다.

\[
q_\phi(X_U\mid D)=\sum_{k=1}^K\pi_k(H)\,
\mathcal N\left(X_U;\mu_k(H),\operatorname{diag}v_k(H)+L_k(H)L_k(H)^\top\right).
\]

이는 알려진 분포 함수족이며 새로운 발명이 아니다. K개 mode와 low-rank global noise가 multimodality와 node/feature dependency를 표현하게 하는 구현 출발점이다. 단일 Gaussian head보다 좋아진다는 가정도 하지 않는다.

첫 density 구현은 연속 피처의 제한된 synthetic family에서 검증할 후보이며, categorical·count feature까지 적절한 likelihood를 제공한다고 주장하지 않는다. Raw feature 의미가 다른 실제 graph로의 전이는 별도다.

분류 연산은

\[
\widehat P(D)=\frac1S\sum_{s=1}^S F_\psi(H,X_U^{(s)},G,M).
\]

여기서 F는 normalized class probabilities를 출력한다. 일반적인 MNAR까지 주장하려면 complete-input branch에도 M을 보존해야 한다. 현재 MCAR/graph-structured MCAR에서는 이 가정을 더 단순하게 고정할 수 있다.

## 학습에서 먼저 고정할 것

Synthetic complete task에서 context와 missing mask를 먼저 만든다. q는 그 관측 입력만 받고 숨겨 둔 synthetic feature를 **학습 target**으로 사용한다.

\[
\mathcal L_{\rm feature}=\mathbb E[-\log q_\phi(X_U^{\rm true}\mid D)].
\]

이는 conditional law의 의미를 학습하기 위한 proper density supervision이다. Query CE만으로 feature posterior가 식별되는 것은 아니다. 실제 target graph에서는 이 feature target도 weight update도 사용하지 않는다. 정규화 통계는 관측값에서만 계산한다.

예측 목적의 기본형은

\[
\mathcal L_{\rm pred}=\mathbb E\left[-\log\left(\frac1S\sum_s F_\psi(H,X_U^{(s)},G,M)_{y_q}\right)\right].
\]

평균 CE와 평균 확률의 CE는 다르다. 유한 S의 이 목적도 exact expected prediction의 NLL과 같지 않으므로 S를 고정해 비교하고 Monte Carlo 오차를 보고한다. Feature law가 잘 맞는지와 label prediction이 좋은지는 각각 평가한다.

처음부터 feature/predictive/coherence weight를 여러 개 조정하지 않는다. 먼저 conditional density를 proper supervision으로 학습하고, 고정한 법칙을 사용하는 prediction 연산의 증분을 비교하는 방안을 우선한다. Shared encoder까지 움직이면 q head만 freeze해도 분포는 변하므로, encoder 포함 freeze 또는 별도 fixed teacher 등 실제 고정 범위를 명시해야 한다.

## 관측 refinement의 학습 조건

같은 coarse D에서 추가 관측 집합 A를 먼저 고르고, D'=D+X_A라 한다. A의 선택은 D에서 정해져야 한다. 관측 선택 자체가 추가 정보를 주면 그 protocol도 조건에 포함한다. 이상적인 조건은

\[
P(D)=\mathbb E_{X_A\mid D}[P(D+X_A)].
\]

한 개의 실제 refinement에 두 예측을 직접 맞추고 두 경로를 함께 학습하면 다음 분산 항까지 벌한다.

\[
\mathbb E\|u-V\|^2=\|u-\mathbb EV\|^2+\operatorname{tr}\operatorname{Cov}(V).
\]

Refined predictor가 고정 teacher라면 추가 분산은 coarse output에 대한 상수다. 그러나 양쪽을 학습할 때는 유용한 관측 후 변화도 줄일 수 있다. S개 평균의 squared loss에도 분산/S 항이 남는다. 같은 complete graph에 두 mask를 씌웠다는 사실만으로 두 refined feature 값이 D 조건부 독립 sample이 되지는 않는다.

알려진 two-sample estimator는 아래와 같다. 이는 MartingaleSSL의 연산을 재사용하는 것이며 우리 novelty가 아니다.

\[
\widehat{\mathcal L}_{\rm coh}=(u-V_a)^\top(u-V_b),\qquad V_a\perp V_b\mid D.
\]

같은 A에 대해 조건부 독립 completion이 필요하다. q에서 샘플하면 q에 대한 일관성을 측정하며, 참 conditional에 대한 일관성을 보장하지 않는다. 이 signed estimator를 0으로 clamp하면 unbiasedness가 사라진다. 적절한 conditional law를 확인한 뒤 CE-only와 CE+coherence를 같은 계산 예산에서 비교한다.

**우리 모델에서는 coarse prediction도 random하다.** U를 finite-particle estimate로 만들고 두 항에서 재사용하면, U와 두 refinement가 서로 독립인 경우에도

\[
\mathbb E[(U-V_a)^\top(U-V_b)\mid D]
=\|\mathbb EU-\mathbb EV\|^2+\operatorname{tr}\operatorname{Cov}(U\mid D)
\]

가 된다. 이를 피하려면 두 replica에서 parent와 child의 Monte Carlo 계산을 함께 독립화한다.

\[
\widehat{\mathcal L}_{\rm coh}
=\left(\widehat P_a(D)-\widehat P_a(D+X_A^{(a)})\right)^\top
 \left(\widehat P_b(D)-\widehat P_b(D+X_A^{(b)})\right).
\]

두 괄호 전체가 D 조건부 독립이어야 한다. 각 괄호 안에서는 marginal law를 유지하는 coupling을 쓸 수 있다. Coarse particles를 독립 두 묶음으로 나누는 구현도 가능하지만 finite-sample variance와 훈련비를 실제 비교해야 한다. Encoder는 deterministic observed path이면 공유할 수 있다. Training dropout/공유 random state가 독립성에 미치는 범위도 명시한다. 이는 estimator를 이 모델에 적용하기 위한 조건이지 새 통계 원리의 발명은 아니다.

**추가 식별 문제:** Decoder가 `F(H,z)=P_0(H)`로 z를 무시해도 관측 입력의 예측 분포를 직접 표현할 수 있다. 또한 coherence는 predictive moment만 제한하므로 서로 다른 feature law가 같은 예측 평균을 줄 수 있다. 네 상태에서 참 확률 `[.1,.2,.3,.4]`와 다른 확률 `[.14,.1,.36,.4]`는 conditional label probabilities `[.1,.4,.6,.9]`에 대해 모두 평균 `.63`을 준다. 이 coarse/full-reveal 예시에서는 Bayes CE와 coherence를 맞혀도 feature law는 식별되지 않는다. 임의의 틀린 q와 모든 refinement에서 항상 성립한다는 주장은 아니다. Constant predictor collapse를 막는 것만으로 feature belief 학습이 입증되지 않는 이유이며, conditional feature score와 가설 사용 개입이 필요하다.

적절한 mask prior에서 모든 partial view의 query log loss를 이상적으로 최소화하면 population Bayes predictor도 tower identity를 만족한다. 따라서 coherence 목적의 기여는 유한 데이터·용량·최적화에서의 추가 이득으로 검증해야 한다. Coherence가 낮다는 사실만으로 calibration이 보장되지는 않는다.

## 계산 공유의 실질적 설계

관측 입력의 encoder는 한 번 계산하고 branch는 가설 주입 이후의 작은 suffix에서만 나눈다. 비용 목표는 `C_encoder + C_law + S*(C_decode + C_suffix)`다. 단순 S번 전체 PFN보다 빠른지는 실제 측정해야 한다.

현재 공개 PFN의 complete-input prefix는 가설마다 달라져 그대로 cache할 수 없다. 공유하려면 **observed-only prefix와 hypothesis-conditioned suffix**로 계산 그래프를 바꾸고 그 구조로 학습해야 한다. 예컨대 기존12층의 앞8층을 공유하고 뒤4층만 분기하는 안은 층 계산 수의 예시이지 speedup 측정이 아니다. 기존12층 full-completion PFN과 정확히 같은 함수라고 주장하지 않는다.

## 다음 실행 순서와 판정

1. **남은 전제 확인:** 기존 frozen bank에서 같은 분산·관측 clamp·4회 호출을 유지한 non-conditioned random-fill/구조 random-feature 대조를 추가한다. 이것은 새 학습법 성과가 아니라 기존 sampling 이득의 범위를 좁히는 비교다. 기존 bank는 개발용으로 명시한다.
2. **학습 모델:** 유효 conditional law를 학습하는 작은 shared-encoder 모델을 구현한다. Context-conditioned q와 label-blind q는 같은 구조·파라미터·task·budget을 사용한다. 두 모델의 classifier에는 동일한 Y_C를 준다. Context-shuffle sensitivity만으로 context의 유익함을 주장하지 않고 true-context의 적절한 feature score와 query 이득을 확인한다.
3. **Coherence 증분:** q와 branch 구조가 고정된 대조를 확보한 후 CE-only와 CE+coherence를 비교한다. 필요 추가 forward를 CE-only 대조에도 주어 데이터·계산 효과를 통제한다. 단순 paired invariance도 구별한다.
4. **효율 검증:** Shared suffix와 전체 PFN S회 실행을 동일 S·입력·학습 가능 용량 조건으로 비교하고, 같은 latency/memory 예산의 강한 Plain·PCFI/FISF+PFN·random-fill ensemble도 확인한다. 가격 차이가 있으면 품질–비용 곡선으로 보고한다.

새 bank에서 NLL/Brier/accuracy, conditional feature score, true-law 기준 coherence, 신규 관측 후 적절한 변화, 단일·다중 가설, end-to-end 비용을 각각 평가한다. 같은 모델이 만든 q로 측정한 consistency만 낮아졌다는 결과를 gate 통과로 삼지 않는다. Real graph와 unseen feature/schema 전이는 추가 검증이 남는다.

Class ID가 exchangeable한 synthetic prior에서는 context label을 일관되게 이름만 바꿔도 feature law는 같아야 하고, class prediction은 그 이름 변경을 따라야 한다. Label shuffle과 이 class renaming은 다르다. Node/feature permutation도 별도로 검사하며, 특정 class ID와 feature 방향의 고정 대응을 외운 효과를 context inference로 세지 않는다.

현재 문서는 **설계안**이다. 새 모델 학습 결과와 GPU 실행은 없다. `math_sanity.json`은 아래 구현상의 함정을 확인한 정확한 두 상태 산술 예시다.

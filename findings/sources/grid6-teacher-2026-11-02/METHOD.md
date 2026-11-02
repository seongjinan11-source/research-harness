# GraphPFN에 exact Grid6 predictive를 가르치는 비교

## 무엇을 유지하고 무엇을 바꿨나

이번 실험은 생성 prior나 student architecture를 새로 제안하지 않는다. 기존 `disjoint_factor_pilot`의2048 training tasks, 각 seed의 초기화·task order·optimizer·learning-rate schedule·256 successful updates를 유지하고 query 감독만 바꾼다. 기존 CE checkpoint에서 fine-tuning하지 않는다.

- 주 학생: 실제 공개 GraphPFN1.3/LimiX의 Mask 입력 모델. 원래 transformer16M weights는 고정하고 adapter 등4,012,259 parameters를 학습한다.
- 보조 학생: 기존8,099-parameter Factor PFN. Node-class와 feature-column latent 사이의 관측 cell factors, finite24-step BP, 같은 potential에서 계산하는 Bethe role update를 그대로 유지한다.
- 대조: 동일 train bank로 이전에 학습한 Mask/PCFI/Factor CE 모델9개를 새 test bank에서 평가한다. Fixed-mode factor BP와 exact Grid6/Grid6-BP도 유지한다.

파라미터 수는 architecture 간 다르고, 같은 architecture의 CE/distill 사이에서는 정확히 같다. 이 실험은 distilled student끼리 임의로 좋은 것을 골라 primary를 바꾸지 않는다.

## 생성기와 teacher는 다른 역할이다

기존 생성기는 latent node labels `Y`, role hypothesis `h`, density/signal `xi=(d,eta)`, feature orientation `z`를 샘플링해 graph와 연속 feature를 만든다.

\[
(Y,h,\xi,z)\sim p,\qquad (G,X)\sim p(G,X\mid Y,h,\xi,z).
\]

그 뒤 original mask와 context를 적용하여

\[
D=(G,X_{\mathrm{obs}},M,Y_C,\mathrm{family})
\]

만 공개한다. 관측feature node와 context node가 완전히 분리된 episode도 포함한다. Mask/context index 선택은 sampled class label에 의존하지 않는다.

Teacher는 이 공개된 D를 사용해

\[
t_q(c)=p_{\mathrm{Grid6}}(Y_q=c\mid D)
=\sum_{h,\xi,z,Y_{\setminus(C\cup\{q\})}}
p(Y_q=c,h,\xi,z,Y_{\setminus(C\cup\{q\})}\mid D)
\]

를 계산한다. True task parameter나 sampled query/helper label을 teacher 입력으로 넣지 않는다. 소형12-node 설정에서는 모든4096 node-label assignments를 열거한다. Role4개, parameter6개, feature orientation4개를 주어진 prior로 주변화한다. Context와 충돌하는 label assignment는 제외한다.

여섯 `(d,eta)`는 `(.25,.72),(.18,.45),(.18,.72),(.25,.45),(.32,.45),(.32,.72)`이고 질량은 `(.5,.1,.1,.1,.1,.1)`다. Gaussian feature law의 계수 `.8`, noise sd `.6`과 이 prior를 teacher는 명시적으로 안다. Student는 실제 hidden parameter를 받지 않고 synthetic tasks에서 학습한다. 따라서 동일한 정보라는 말은 공개 evidence의 동등성을 뜻하며, 알려진 생성법에 대한 inductive knowledge까지 동일하다는 뜻은 아니다.

Grid6-BP는 별도 reference다. 가설 weight는 exact enumeration, 각 가설의 conditional query prediction은64-step damped BP인 hybrid inference다. **학습 teacher에는 그 BP 근사가 아니라 exact Grid6 query marginal을 사용한다.**

## 학습 연산

기존 감독은 샘플링된 query label에 대한 CE였다.

\[
\mathcal L_{\mathrm{CE}}=-\frac1{|Q|}\sum_{q\in Q}\log p_\theta(y_q\mid D).
\]

새 감독은

\[
\mathcal L_{\mathrm{distill}}=
\frac1{|Q|}\sum_{q\in Q}\sum_c
t_q(c)\left[\log t_q(c)-\log p_\theta(c\mid D)\right].
\]

Temperature는1이고 teacher는 detach한다. Sampled-label CE 혼합, auxiliary loss, 새 posterior head를 추가하지 않는다. Teacher의 context/helper 행은 학습 loss에서 제외한다. Factor student의 기존 self-role-weight 계산을 포함해 전체 finite inference computation으로 gradient를 전파한다.

올바른 family 안에서 고정된 D에 조건부로 보면

\[
\mathbb E_{Y_q\mid D}[-\log p_\theta(Y_q\mid D)]
=H(t_q)+D_{\mathrm{KL}}(t_q\Vert p_\theta).
\]

따라서 이상적인 기대 목표 자체를 바꾸기보다, 단일 label draw의 조건부 잡음을 제거한 조밀한 감독을 준다. 이것이 제한된 task/update budget에서 개선을 보장하지는 않는다. Student가 표현하거나 최적화할 수 없는 분포를 teacher가 해결해 주는 것은 아니다.

Query별 marginal을 가르치는 것이며 joint query-label 분포 전체를 distill한 것은 아니다. 각 marginal에서 label을 독립 샘플링해 새 graph의 공동 정답으로 사용하지 않는다.

## 판정과 한계

Primary는 새 matched48 base graphs, disjoint episode의 Mask-distill−Mask-CE Grid6 predictive KL이다. Context1/2/4와 observed nodes3/6, 세 fixed training seeds를 graph 내 평균한 뒤 paired bootstrap5000회로 평가한다. Accuracy와 sampled-label NLL/Brier는 독립적으로 보고한다. Factor, query-no-feature, overlap와 OOD는 secondary다.

Teacher effect는 primary KL 차이의95%CI 상한이0 미만인지로 판단한다. Competitive compression은 matched teacher KL<=.01, 같은GPU에서 Grid6-BP보다 단일 graph median2배 이상 빠른지를 추가로 요구한다. Offline teacher 준비 비용과 학생 학습 비용도 별도 공개한다.

같은 known family의 fresh graph validation이며 새로운 semantic schema, arbitrary feature law, 실제 heterograph,99.5% missing, 큰 graph의 검증이 아니다. Density/signal OOD의 Grid6 KL은 misspecified Grid6 reference에 대한 일치도이지 참 OOD Bayes KL이 아니다. Distillation은 기존 원리이고 과거 feature-free JR 실험에서도 사용했으므로 이번 비교만으로 새로운 메소드나 novelty를 주장하지 않는다.

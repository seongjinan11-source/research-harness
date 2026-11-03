# Confidence Sheaf PFN 연구 초안

2026년 11월 3일. Intro, Related Work, Method, Experiments를 정리한 짧은 심사용 초안이다. 현재 구현과 완료한 실험을 기술하며, 논문 기여 후보와 아직 입증하지 못한 주장을 구분한다. 작업 제목은 **Confidence-weighted Coordinate Transport for Missing-Feature GraphPFN**이다.

## 1 Introduction

새 그래프에서 node feature가 대부분 결측이면, 적은 관측값과 context label을 그래프 관계를 통해 연결해야 한다. 특히 target graph에서 모델을 다시 학습하지 않는 설정에서는, 어떤 값을 얼마나 신뢰하고 어떻게 이웃으로 전달할지를 사전학습된 추론기가 판단해야 한다. 우리는 **하나의 모델로 homophilic 및 heterophilic graph의 심한 feature missing을 처리하는 gradient-free node prediction**을 목표로 한다. 이번 범위는 homogeneous graph이며 heterogeneous schema는 포함하지 않는다.

Feature propagation은 관측값을 경계로 고정하고 전파해 강한 결측 기준선을 제공한다. PCFI는 관측 source까지의 거리를 confidence로 사용한다. 한편 graph의 연결이 모든 feature의 같은 방향 유사성을 뜻하지는 않는다. 여기서 label heterophily와 feature의 비유사성은 다른 개념이며, heterophilic label graph라는 이유만으로 모든 numeric feature의 부호를 바꿔야 하는 것은 아니다. [FP](https://proceedings.mlr.press/v198/rossi22a.html), [PCFI](https://arxiv.org/abs/2305.16618).

우리의 가설은 **관측값의 신뢰도와 값을 전달하는 좌표계를 함께 다루는 학습된 결측 추정기**가 고정 전파보다 유연한 inductive bias를 제공할 수 있다는 것이다. PCFI의 confidence와 관측값 고정을 유지하면서, graph와 관측값 및 context 통계에서 node별 frame을 계산한다. 이 frame으로 변환한 메시지를 확산하고 다시 숫자 피처를 읽어 같은 GraphPFN에 제공한다. 추정기는 synthetic task의 query classification loss로 학습하며 실제 graph에서는 모든 가중치를 고정한다.

기여 후보는 이 계산을 **극심한 결측에서 새 그래프에 적용하는 PFN용 수치 추론 연산**으로 정식화하고, 단순한 전파와 signed scalar 대조에 비해 유용한 조건을 밝히는 것이다. Confidence, orthogonal transport, classification loss, synthetic pretraining 각각의 발명을 주장하지 않는다. 현재 pilot은 구성 요소 분석까지 완료했으며 PCFI 대비 일관된 분류 우위와 unseen real graph 전이는 아직 확보하지 못했다.

## 2 Related Work

| 연구 | 이미 제공하는 원리 | 우리와 구분해 검증할 부분 |
|---|---|---|
| [FP](https://proceedings.mlr.press/v198/rossi22a.html) | Dirichlet energy에 기반한 결측 feature diffusion과 downstream graph prediction | 고정 scalar 전파 대신 task-conditioned transport를 학습하는 증분 |
| [PCFI](https://arxiv.org/abs/2305.16618) | Source 거리의 channel-wise confidence와 공간 및 채널 전파 | Confidence 자체가 아니라 값을 전달하는 frame의 학습과 cross-graph 적용 |
| [FISF](https://proceedings.mlr.press/v267/um25a.html) | 전파 결과의 저분산 문제를 synthetic feature 주입으로 완화 | 현재 방법은 numeric source를 변환하며 synthetic feature injection을 추가하지 않음. FISF+PFN은 강한 최종 비교군으로 필요 |
| [Neural Sheaf Diffusion](https://arxiv.org/html/2202.04579) | 학습한 restriction map과 sheaf diffusion으로 heterophily 및 oversmoothing을 다룸 | 관측 경계를 고정한 numeric imputation과 synthetic-to-real 고정 가중치 전이. 일반 sheaf나 orthogonal map 자체는 기존 원리 |
| [GRAPE](https://papers.neurips.cc/paper_files/paper/2020/file/dc36f18a9a0a776671d4879cae69b551-Paper.pdf) | 관측 sample과 feature의 bipartite 표현에서 imputation과 label prediction을 GNN 과제로 학습 | Task loss로 불완전 입력을 처리한다는 주장만으로는 부족. 여기서는 주어진 원 graph에서의 confidence transport와 고정 PFN 전이를 검증해야 함 |
| [GraphPFN](https://proceedings.mlr.press/v306/eremeev26a.html) | Synthetic graph prior 학습과 graph-aware ICL node prediction | 기존 PFN 앞의 결측 추정 연산을 학습하는 독립 가치. Synthetic pretraining과 zero-shot 자체는 기존 원리 |
| [Wander](https://arxiv.org/abs/2610.06694) | 부분 관측 graph의 completion 관점과 random-walk 기반 graph 및 task 간 전이 | 부분 관측을 다룬다는 포괄적 주장도 이미 존재. 우리의 제한된 numeric 연산이 실제 결측 평가에서 주는 증분을 비교해야 함 |

따라서 핵심 차별점 후보는 **관측 경계와 confidence를 유지한 task-conditioned transport를 PFN 예측 목적에 맞춰 학습하고, target graph 재학습 없이 적용하는 구체적 설계**다. 이 설정 차이가 충분한 메소드 기여인지는 심사에서 검토할 사항이다. 제한된 검색에서 직접 일치하는 논문을 찾지 못했다는 사실을 최초성의 증거로 쓰지 않는다. Signed 또는 adaptive graph filter와의 추가 비교도 필요하다.

## 3 Method

### 3.1 입력과 confidence

입력은 \(D_{obs}=(G,X_{obs},M,Y_C)\)이고 \(Q\)의 정답은 추론 입력에 들어가지 않는다. \(M_{if}=1\)은 관측 cell이다. 각 column의 관측 source까지 거리에 기반한 PCFI pseudo-confidence \(c_{if}=0.9^{d(i,O_f)}\)를 사용하고 self-loop를 포함한다. Source가 없거나 도달 불가능한 경우는 기존 PCFI 구현 convention을 유지한다. 이 confidence를 보정된 확률이나 학습된 uncertainty라고 해석하지 않는다.

\[
P_{ij,f}=\frac{A_{ij}c_{jf}}{\sum_k A_{ik}c_{kf}}.
\]

### 3.2 입력에서 frame 구성

737-parameter 공유 MLP가 node와 column별 각도를 출력한다.

\[
\theta_{if}=\pi\tanh\Phi_\phi(d_{if},\bar d_{if},a_i),\quad
Q_{if}=R(\theta_{if}),\quad T_{ij,f}=Q_{if}^{\top}Q_{jf}.
\]

Descriptor는 관측값과 절댓값, 관측 이웃 평균, mask, confidence, degree, context 위치와 context-only LP의 entropy 및 class support, context edge agreement 및 affinity다. 숨은 feature, query label, 참 homophily나 feature-law ID는 읽지 않는다. Frame은 한 과제의 100회 전파 동안 고정한다. 현재 descriptor를 사용했다는 것과 올바른 label semantics를 실제로 활용했다는 것은 구분한다.

### 3.3 관측 경계가 있는 공간 전파

관측 cell은 \((x_{if},0)\), 결측 cell은 \((0,0)\)으로 시작한다.

\[
z_{if}^{t+1}=
\begin{cases}
(x_{if},0),&M_{if}=1,\\
\sum_jP_{ij,f}T_{ij,f}z_{jf}^{t},&M_{if}=0.
\end{cases}
\]

100회 후 첫 좌표를 읽어 기존 PCFI 채널 상관 보정을 한 번 적용하고 관측값을 다시 고정한다. 이렇게 얻은 \(\tilde X_\phi\)와 원래 graph 및 context를 같은 고정 PFN에 제공한다. 추가 warmup diffusion, hidden-token diffusion, 숫자 completion의 다중 샘플링은 사용하지 않는다. \(T=I\)이면 기존 PCFI 공간 연산으로 환원된다. Orthogonality는 개별 벡터 norm을 보존하지만 이웃 평균의 상쇄를 막아주지는 않는다.

### 3.4 학습과 추론

\[
\mathcal L(\phi)=\mathbb E_{T\sim p_{syn}}
\left[-\frac1{|Q|}\sum_{q\in Q}
\log F_\psi(y_q\mid G,X_{obs},M,\tilde X_\phi,Y_C)\right].
\]

이번 frontend 학습 동안 \(\psi\)는 고정한다. 별도 reconstruction loss나 teacher, posterior head는 없다. 실제 graph에서는 관측 입력으로 frame과 confidence를 계산하지만 \(\phi,\psi\)의 optimizer update는 0회다. 실험용 PFN은 기존 synthetic-trained PCFI-PFN checkpoint를 재사용한 wrapper이며, 공개 GraphPFN의 전체 학습과 공식 full-pipeline 성과를 재현한 것은 아니다. Joint PFN 학습은 이 메소드의 필수 구성으로 두지 않는다.

### 3.5 Flat frame의 정확한 재표현

현재 Node-frame의 transport는 경로 독립적이다. 같은 관측 경계와 P를 쓰는 100회 scalar 공간 전파를 D라 하면, 채널 보정 전 출력은 다음과 같다.

\[
\hat x^{sp}_{if}=
\cos\theta_{if}\,D_f(x_f\cos\theta_f)_i+
\sin\theta_{if}\,D_f(x_f\sin\theta_f)_i
=\sum_jH_{ij,f}\cos(\theta_{jf}-\theta_{if})x_{jf}.
\]

H는 source의 scalar 전파 계수다. 이는 두 채널의 보통 전파와 회전 readout으로 구현할 수 있다는 뜻이며, 새 정리나 일반 sheaf의 표현력 증명으로 주장하지 않는다. Global frame rotation은 출력에서 상쇄되고, 별도의 cycle-dependent transport 효과는 없다. 이 재표현을 알면서도 해당 inductive bias가 필요한지를 심사받고자 한다.

## 4 Experiments

### 4.1 완료한 비교와 평가 범위

기존 512개 synthetic train episode와 동일 초기 Phi 및 과제 순서를 재사용했다. Label 통계를 제거한 Node, 공간 가중치를 균등하게 한 Node, 직접 signed scalar를 각각 3 seed×1,024 update로 학습했다. 새 frontend는 9개이고, 기존 Node와 I 기준선 및 geometry-only label shuffle을 함께 평가했다. 학습 frontend는 각각 737개 parameter다.

새 개발 bank는 같은 label-SBM×feature-filter family의 32개 base graph다. Label homo/hetero와 feature smooth/rough 네 cell에 8개씩 배정했다. Context는 1/4개/class, primary 결측은 uniform/clustered whole-node 90/99.5%다. Primary는 자기 피처가 없는 query accuracy이며 NLL과 Brier도 보고한다. CI는 3개 고정 학습 seed와 조건을 graph 안에서 평균한 뒤 base graph 단위 paired bootstrap 5,000회로 계산했고 다중비교 미보정이다.

| 비교 | Accuracy 차이 %p | 95% CI |
|---|---|---|
| Node − PCFI | +0.080 | [−0.137, +0.303] |
| Node − label 통계 제거 후 재학습 | +0.269 | [+0.042, +0.506] |
| 공간 confidence × Node interaction | +0.371 | [+0.055, +0.697] |
| Node − 직접 signed scalar | +0.253 | [−0.021, +0.531] |
| Node − geometry-only label shuffle | −0.035 | [−0.122, +0.048] |

Node의 절대 정확도는 44.852%, PCFI는 44.772%다. NLL/Brier는 각각 Node 1.30707/0.64125, PCFI 1.30606/0.64059다. 현재 PCFI 대비 전반적 우위는 입증되지 않았다.

Interaction은 (Node−I)의 confidence 유무 차이다. 양의 값에는 균등 전파에서 Node가 I보다 낮아진 효과가 포함되므로, 이를 PCFI를 넘는 성능 증거로 해석하지 않는다. Label 통계 제거 대조는 낮았지만 geometry shuffle은 k=4에서도 뚜렷한 손실이 없었다. k=1의 shuffle은 class 이름 순열 대칭이 될 수 있다. Signed scalar는 음수 gain을 실제로 학습하지 않았고 +1 clipping 비율이 약 67%였으므로 2D와 충분히 최적화된 signed 함수족의 우열은 미결정이다.

### 4.2 실제 graph의 개발 결과

| 방법 | Cora % | CiteSeer % | Texas % | Wisconsin % |
|---|---|---|---|---|
| PCFI | 50.188 | 30.852 | 28.005 | 21.557 |
| Node | 49.800 | 31.275 | 28.178 | 22.114 |
| Native-PCFI | 52.030 | 33.867 | 33.260 | 24.445 |

네 실제 graph는 이전부터 사용한 개발 입력이고, Native-PCFI는 원본 고차원에서 먼저 전파하는 더 강한 전체 pipeline 기준이다. 동일표현의 geometry-only 비교와 구분한다. 고정 negative LP도 Texas 63.537%, Wisconsin 42.725%로 여전히 강한 기준이다. 이 결과를 독립 holdout이나 heterophily 전반의 우위로 부르지 않는다.

Source0와 full 관측 조건에서는 같은 PFN 입력으로 환원된다. 별도 동치 구현의 CPU frontend 비용은 Cora 130.89→75.11ms였지만, PFN을 포함한 전체 속도나 새 정확도 개선은 아니다. GPU replay의 선택된 예측에서 최대 argmax 변경은 0.3%였고 위 CI는 이 수치 변동 전체를 포함하지 않는다.

### 4.3 아직 필요한 실험

1. Smooth한 parameterization으로 signed scalar의 clipping 문제를 분리하고, 같은 예산의 adaptive graph filter와 비교한다.
2. Geometry에 올바른 label 묶음이 필요한 조건과 그렇지 않은 조건을 구분한다. Ablation 차이만으로 context semantics의 사용을 주장하지 않는다.
3. 동일 입력 표현의 PCFI/FISF+PFN 및 강한 전체 pipeline을 비교하고, 구조를 고정한 뒤 새로운 실제 graph에서 평가한다.

## 심사에서 우선 묻는 질문

- 개별 부품의 최초성보다 **이 결합 전체에 독립적인 계산 또는 학습 기여가 있는가?**
- 성능을 충분히 개선한다고 가정해도 단순한 PCFI와 flat sheaf 및 PFN의 조합 이상으로 인정할 근거가 있는가?
- 두 채널의 보통 전파 또는 cosine source kernel이라는 정확한 재표현이 novelty를 얼마나 제한하는가?
- Context 의미 사용의 미확인과 scalar clipping은 단순한 실험 보완인지, 핵심 메소드 가설의 약점인지?
- 가장 가까운 선행연구와 직접 겹치는 부분, 남는 기여 후보, 최소한 필요한 설계 또는 증거를 구분해 달라. 현재 수치의 약함과 메소드적 독창성은 별도로 평가해 달라.

로컬 근거: [메소드와 프로토콜](../METHOD_KO.md), [최신 실험 보고서](../REPORT_KO.md), [정확한 재표현 검토](../../pcfi_orthogonal_pilot/reviewer_development/REVIEW_KO.md). 문헌 링크는 각 Related Work 항목의 원문을 가리킨다.
